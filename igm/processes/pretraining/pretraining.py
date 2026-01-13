#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import random

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings
from igm.processes.iceflow.emulate.utils.artifacts import save_emulator_artifact


 
def initialize(cfg, state):
    cfg_pretraining = cfg.processes.pretraining
    cfg_iceflow = cfg.processes.iceflow
    Nz = cfg_iceflow.numerics.Nz
    tfrecord_root = Path('/home/srosier/work/synthetic_glaciers/tfrecords/set1/')
    out_dir = Path(cfg_pretraining.experiment_name)
    meta = load_metadata(tfrecord_root)

    shapes = meta["example_shapes_by_nz"][str(Nz)]
    H, W, Cx = shapes["x"]

    shard_files = list_shards(tfrecord_root, Nz)

    # shuffle shard files to remove any possible ordering bias
    rng = random.Random(getattr(cfg_pretraining, "split_seed", 0))
    rng.shuffle(shard_files)

    train_ds, val_ds = make_datasets(
        shard_files=shard_files,
        H=H, W=W, Nz=Nz,
        compression="GZIP",
        batch_size=cfg_pretraining.batch_size,
        val_fraction=0.1,
    )

    # # Input normalization (stats persisted in the saved model)
    # normalizer = tf.keras.layers.Normalization(axis=-1, name="x_norm")
    # # Adapt on a bounded sample to keep “first run” practical; adjust as needed.
    # norm_adapt = train_ds.unbatch().map(lambda x, y: x).take(5000).batch(256)
    # normalizer.adapt(norm_adapt)

    # Initialize mapping
    mapping_args = InterfaceMappings["network"].get_mapping_args(cfg, state)
    mapping = Mappings["network"](**mapping_args)
    state.iceflow.mapping = mapping

    opt = tf.keras.optimizers.Adam(learning_rate=cfg_pretraining.learning_rate)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")

    def make_loss_fn(mapping, cfg):
        lt = cfg.processes.pretraining.loss_type.lower()
        delta = float(getattr(cfg.processes.pretraining, "huber_delta", 50.0))
        huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)

        @tf.function
        def loss_fn(x, y):
            U, V = mapping.get_UV(x)
            Ut, Vt = y[..., 0], y[..., 1]
            if lt == "huber":
                return tf.reduce_mean(huber(Ut, U) + huber(Vt, V))
            if lt == "mse":
                return tf.reduce_mean(tf.square(U - Ut) + tf.square(V - Vt))
            raise ValueError(f"loss_type must be 'mse' or 'huber', got {lt!r}")

        return loss_fn

    loss_fn = make_loss_fn(mapping, cfg)

    @tf.function
    def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = loss_fn(x_batch, y_batch)
        grads = tape.gradient(loss, state.iceflow_model.trainable_variables)
        opt.apply_gradients(zip(grads, state.iceflow_model.trainable_variables))
        train_loss.update_state(loss)

    @tf.function
    def val_step(x_batch: tf.Tensor, y_batch: tf.Tensor):
        loss = loss_fn(x_batch, y_batch)
        val_loss.update_state(loss)

    # Checkpointing
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=opt, model=state.iceflow_model)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, str(out_dir / "checkpoints"), max_to_keep=3)

    val_vis_ds = val_ds.unbatch().shuffle(4096, reshuffle_each_iteration=True)\
                    .batch(cfg_pretraining.batch_size, drop_remainder=True)
    val_vis_it = iter(val_vis_ds.repeat())

    fig_dir = out_dir / "figures"
    train_hist = []
    val_hist = []
    for epoch in range(cfg_pretraining.epochs):
        train_loss.reset_state()
        val_loss.reset_state()

        # Train
        for x_b, y_b in train_ds:
            train_step(x_b, y_b)
            ckpt.step.assign_add(1)
    
        # Validate
        itv = iter(val_ds)
        for _ in range(50):
            x_b, y_b = next(itv)
            val_step(x_b, y_b)

        print(f"[epoch {epoch+1}/{cfg_pretraining.epochs}] train_loss={train_loss.result().numpy():.6e} "
              f"val_loss={val_loss.result().numpy():.6e}")
        
        tr = float(train_loss.result().numpy())
        va = float(val_loss.result().numpy())
        train_hist.append(tr)
        val_hist.append(va)

        # Save loss curve
        save_loss_plot(train_hist, val_hist, fig_dir / "loss_curve.png")

        # Save a random surface speed comparison from validation
        x_vis, y_vis = next(val_vis_it)
        save_speed_compare(mapping, x_vis, y_vis, Nz, fig_dir / f"speed_compare_epoch{epoch+1:04d}.png")


        ckpt_mgr.save()

    # after training is done
    artifact_dir = Path(cfg.processes.pretraining.out_dir) / cfg.processes.pretraining.experiment_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # inputs must match cfg.processes.iceflow.unified.inputs exactly
    inputs = list(cfg.processes.iceflow.unified.inputs)

    save_emulator_artifact(
        artifact_dir=artifact_dir,
        cfg=cfg,
        model=state.iceflow_model,
        inputs=inputs,
    )
    print(f"[export] saved emulator artifact to {artifact_dir}")

    k = min(5, len(val_hist))
    avg_last5 = float(np.mean(val_hist[-k:]))
    state.score = avg_last5
        
def update(cfg, state):
    pass

def finalize(cfg, state):
    pass
 
def load_metadata(tfrecord_root: Path) -> dict:
    meta_path = tfrecord_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json at {meta_path}")
    return json.loads(meta_path.read_text())

def list_shards(tfrecord_root: Path, nz: int) -> List[str]:
    shard_dir = tfrecord_root / f"nz{nz}"
    if not shard_dir.exists():
        raise FileNotFoundError(f"Missing Nz folder: {shard_dir}")
    files = sorted(str(p) for p in shard_dir.glob("data_nz*_shard*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No TFRecord shards found in {shard_dir}")
    return files

def parse_example(serialized: tf.Tensor, H: int, W: int, Nz: int) -> Tuple[tf.Tensor, tf.Tensor]:
    feat = {
        "geom_id": tf.io.FixedLenFeature([], tf.int64),
        "seed": tf.io.FixedLenFeature([], tf.int64),
        "t": tf.io.FixedLenFeature([], tf.float32),
        "nz": tf.io.FixedLenFeature([], tf.int64),
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(serialized, feat)

    x = tf.io.parse_tensor(ex["x"], out_type=tf.float32)
    y = tf.io.parse_tensor(ex["y"], out_type=tf.float32)

    # Enforce shapes from metadata (post-crop)
    x = tf.ensure_shape(x, [H, W, 2])           # thk, usurf
    y = tf.ensure_shape(y, [Nz, H, W, 2])       # (U,V) on Nz levels
    return x, y
    
def make_datasets(
    shard_files: List[str],
    H: int,
    W: int,
    Nz: int,
    compression: str,
    batch_size: int,
    val_fraction: float = 0.1,
    shuffle_buffer: int = 2048,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # Deterministic split by shard files
    n_val = max(1, int(len(shard_files) * val_fraction))
    val_files = shard_files[-n_val:]
    train_files = shard_files[:-n_val] if len(shard_files) > n_val else shard_files

    def ds_from(files: List[str], training: bool) -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(files, compression_type=compression)
        ds = ds.map(lambda s: parse_example(s, H, W, Nz), num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return ds_from(train_files, True), ds_from(val_files, False)

def save_loss_plot(train_hist, val_hist, fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(train_hist) + 1), train_hist, label="train")
    plt.plot(np.arange(1, len(val_hist) + 1), val_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")  # usually helpful for MSE-like losses
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def save_speed_compare(mapping, x_b, y_b, Nz: int, fig_path: Path) -> None:
    """
    Saves a 3-panel image: true surface speed, predicted surface speed, difference.
    Assumes 'surface' is the last vertical level (Nz-1). If your convention is opposite, change k=0.
    """
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # pick one random sample from the batch
    b = int(np.random.randint(0, x_b.shape[0]))
    k = Nz - 1  # surface level assumption

    # Forward (BCs applied to predictions via mapping)
    U_pred, V_pred = mapping.get_UV(x_b)
    U_pred = U_pred.numpy()
    V_pred = V_pred.numpy()

    # Targets
    U_true = y_b[..., 0].numpy()
    V_true = y_b[..., 1].numpy()

    sp_true = np.sqrt(U_true[b, k] ** 2 + V_true[b, k] ** 2)
    sp_pred = np.sqrt(U_pred[b, k] ** 2 + V_pred[b, k] ** 2)
    sp_diff = sp_pred - sp_true

    # shared scaling for true/pred (robust upper bound)
    vmax = np.nanpercentile(np.concatenate([sp_true.ravel(), sp_pred.ravel()]), 99)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else None

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axs[0].imshow(sp_true, origin="lower", vmin=0, vmax=vmax)
    axs[0].set_title("true surface speed")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(sp_pred, origin="lower", vmin=0, vmax=vmax)
    axs[1].set_title("pred surface speed")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # difference plot: symmetric bounds
    dmax = np.nanpercentile(np.abs(sp_diff.ravel()), 99)
    dmax = float(dmax) if np.isfinite(dmax) and dmax > 0 else None
    im2 = axs[2].imshow(sp_diff, origin="lower", vmin=-dmax, vmax=dmax)
    axs[2].set_title("pred - true")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
