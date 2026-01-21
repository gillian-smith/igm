#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import random

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings
from igm.processes.iceflow.emulate.utils.artifacts import save_emulator_artifact
from igm.processes.iceflow.unified.utils import get_cost_fn



def initialize(cfg, state):
    cfg_pretraining = cfg.processes.pretraining
    cfg_iceflow      = cfg.processes.iceflow
    cfg_physics      = cfg.processes.iceflow.physics
    Nz = cfg_iceflow.numerics.Nz

    tfrecord_root = Path(cfg_pretraining.data_dir)

    # Put everything under the configured output root
    out_dir = Path(cfg_pretraining.out_dir) / cfg_pretraining.experiment_name
    # make out_dir if not exist
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(getattr(cfg_pretraining, "resume", False))

    meta = load_metadata(tfrecord_root)
    shapes = meta["example_shapes_by_nz"][str(Nz)]
    H, W, Cx = shapes["x"]

    # ================== safety checks ==================
    inputs = tuple(cfg_pretraining.inputs)

    if Cx != len(inputs):
        raise ValueError(
            f"TFRecord x has C={Cx} channels (from metadata), but cfg.processes.pretraining.inputs "
            f"has {len(inputs)} entries: {inputs}. These must match in count and order."
        )

    if Cx == 2 and inputs != ("thk", "usurf"):
        raise ValueError(
            f"parse_example() assumes x channels are ('thk','usurf') in that order, but cfg inputs are {inputs}. "
            "Either set pretraining.inputs=['thk','usurf'] or update parse_example()/TFRecords accordingly."
        )

    if int(getattr(cfg_physics, "dim_arrhenius", 1)) == 3 and Cx <= 2:
        raise ValueError(
            "cfg.processes.iceflow.physics.dim_arrhenius == 3 but TFRecord inputs appear to have only 2 channels. "
            "If 3D Arrhenius is enabled, you likely need additional Arrhenius-related channels in TFRecords "
            "(or set dim_arrhenius=1 for this pretraining run)."
        )

    if not hasattr(state, "iceflow") or not hasattr(state.iceflow, "vertical_discr") or state.iceflow.vertical_discr is None:
        raise RuntimeError(
            "state.iceflow.vertical_discr is missing, but the physics cost requires it. "
            "Ensure the iceflow vertical discretization is initialized before pretraining."
        )
    
    # ================== directories / resume checks ==================
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"

    if resume:
        if not out_dir.exists():
            raise FileNotFoundError(
                f"resume=True but experiment directory does not exist: {out_dir}"
            )
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"resume=True but checkpoints directory missing: {ckpt_dir}"
            )
        # history.yaml existence checked later by load_history_yaml()
    else:
        # Prevent silently overwriting an existing run
        if ckpt_dir.exists() and any(ckpt_dir.glob("ckpt-*")):
            raise FileExistsError(
                f"Experiment already has checkpoints at {ckpt_dir} but resume=False. "
                "Set cfg.processes.pretraining.resume=true or use a new experiment_name."
            )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)


    # ================== datasets ==================
    shard_files = list_shards(tfrecord_root, Nz)

    rng = random.Random(getattr(cfg_pretraining, "split_seed", 0))
    rng.shuffle(shard_files)

    train_ds, val_ds = make_datasets(
        shard_files=shard_files,
        H=H, W=W, Nz=Nz,
        compression="GZIP",
        batch_size=cfg_pretraining.batch_size,
        val_fraction=0.1,
    )

    # ================== mapping/model/opt ==================
    mapping_args = InterfaceMappings["network"].get_mapping_args(cfg, state)
    mapping = Mappings["network"](**mapping_args)
    state.iceflow.mapping = mapping

    opt = tf.keras.optimizers.Adam(learning_rate=cfg_pretraining.learning_rate)

    # ================== loss pieces ==================
    physics_cost_fn = get_cost_fn(cfg, state)

    lt = cfg_pretraining.loss_type.lower()
    if lt not in ("mse", "huber"):
        raise ValueError(f"loss_type must be 'mse' or 'huber', got {lt!r}")

    if lt == "huber":
        delta = float(getattr(cfg_pretraining, "huber_delta", 50.0))
        huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)

    def compute_losses(x_batch: tf.Tensor, y_batch: tf.Tensor):
        U, V = mapping.get_UV(x_batch)
        Ut, Vt = y_batch[..., 0], y_batch[..., 1]

        if lt == "huber":
            data_loss = tf.reduce_mean(huber(Ut, U) + huber(Vt, V))
        else:  # mse
            data_loss = tf.reduce_mean(tf.square(U - Ut) + tf.square(V - Vt))

        # Ensure scalar
        phys_loss = physics_cost_fn(U, V, x_batch)
        return data_loss, phys_loss

    def safe_global_norm(grads):
        gs = [g for g in grads if g is not None]
        return tf.linalg.global_norm(gs) if gs else tf.constant(0.0, tf.float32)

    EMA          = tf.constant(0.99, tf.float32)      # smoothing of lambda updates
    UPDATE_EVERY = tf.constant(100, tf.int64)         # per-term grads only every N steps
    LAM_MIN      = tf.constant(1e-3, tf.float32)     # clip range 
    LAM_MAX      = tf.constant(1e3, tf.float32)
    EPS          = tf.constant(1e-12, tf.float32)
    WARMUP_STEPS = tf.constant(1000, tf.int64)  # number of initial steps to keep lambda_phys frozen at 0

    step = tf.Variable(0, trainable=False, dtype=tf.int64, name="step")
    lambda_phys = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="lambda_phys")

    # ================== metrics ==================
    train_total = tf.keras.metrics.Mean(name="train_total")
    train_data  = tf.keras.metrics.Mean(name="train_data")
    train_phys  = tf.keras.metrics.Mean(name="train_phys")
    train_lam   = tf.keras.metrics.Mean(name="lambda_phys")
    val_total   = tf.keras.metrics.Mean(name="val_total")
    val_data  = tf.keras.metrics.Mean(name="val_data")
    val_phys  = tf.keras.metrics.Mean(name="val_phys")
    
    @tf.function
    def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor):
        vars_ = state.iceflow_model.trainable_variables
        step.assign_add(1)

        in_warmup = step <= WARMUP_STEPS
        do_update = tf.equal(step % UPDATE_EVERY, 0)

        def update_branch():
            # Only run the expensive per-term gradients when NOT in warmup.
            with tf.GradientTape(persistent=True) as tape:
                data_loss, phys_loss = compute_losses(x_batch, y_batch)

            g_data = tape.gradient(data_loss, vars_)
            g_phys = tape.gradient(phys_loss, vars_)
            del tape

            norm_data = safe_global_norm(g_data)
            norm_phys = safe_global_norm(g_phys)

            lam_hat = norm_data / (norm_phys + EPS)

            # Optional but strongly recommended: limit multiplicative change per update
            MAX_UP   = tf.constant(2.0, tf.float32)
            MAX_DOWN = tf.constant(2.0, tf.float32)
            lam_hat = tf.clip_by_value(lam_hat, lambda_phys / MAX_DOWN, lambda_phys * MAX_UP)

            lam_new = EMA * lambda_phys + (1.0 - EMA) * tf.stop_gradient(lam_hat)
            lam_new = tf.clip_by_value(lam_new, LAM_MIN, LAM_MAX)
            lambda_phys.assign(lam_new)

            # Apply combined grads using updated lambda
            grads = []
            for gd, gp in zip(g_data, g_phys):
                if gd is None and gp is None:
                    grads.append(None)
                elif gd is None:
                    grads.append(lam_new * gp)
                elif gp is None:
                    grads.append(gd)
                else:
                    grads.append(gd + lam_new * gp)

            opt.apply_gradients([(g, v) for g, v in zip(grads, vars_) if g is not None])

            total_loss = data_loss + lam_new * phys_loss
            return data_loss, phys_loss, total_loss, lam_new

        def normal_branch():
            # Cheap branch: one tape for total loss only.
            with tf.GradientTape() as tape:
                data_loss, phys_loss = compute_losses(x_batch, y_batch)
                # Warmup uses lam=0 without mutating lambda_phys.
                lam = tf.where(in_warmup, tf.constant(0.0, tf.float32), lambda_phys)
                total_loss = data_loss + lam * phys_loss

            grads = tape.gradient(total_loss, vars_)
            opt.apply_gradients([(g, v) for g, v in zip(grads, vars_) if g is not None])
            return data_loss, phys_loss, total_loss, lam

        # If warmup: NEVER do update_branch (keeps lambda frozen and avoids pointless per-term grads)
        # Else: do update_branch every UPDATE_EVERY steps, normal_branch otherwise
        data_loss, phys_loss, total_loss, lam = tf.cond(
            in_warmup,
            normal_branch,
            lambda: tf.cond(do_update, update_branch, normal_branch),
        )

        train_data.update_state(data_loss)
        train_phys.update_state(phys_loss)
        train_total.update_state(total_loss)
        train_lam.update_state(lam)


    @tf.function
    def val_step(x_batch: tf.Tensor, y_batch: tf.Tensor):
        data_loss, phys_loss = compute_losses(x_batch, y_batch)
        total_loss = data_loss + lambda_phys * phys_loss
        val_data.update_state(data_loss)
        val_phys.update_state(phys_loss)
        val_total.update_state(total_loss)

    # ================== checkpointing ==================
    ckpt = tf.train.Checkpoint(
        step=step,
        optimizer=opt,
        model=state.iceflow_model,
        lambda_phys=lambda_phys,
    )
    ckpt_mgr = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=3)
    # Restore if requested
    if resume:
        latest = ckpt_mgr.latest_checkpoint
        if not latest:
            raise FileNotFoundError(
                f"resume=True but no checkpoints found in {ckpt_dir}"
            )
        ckpt.restore(latest).expect_partial()
        print(f"[ckpt] restored {latest} (step={int(step.numpy())}, lambda={float(lambda_phys.numpy()):.3e})")

        # Load history.yaml (strict)
        start_epoch, train_total_hist, val_total_hist, train_data_hist, val_data_hist, train_phys_hist, val_phys_hist, lambda_hist = \
            load_history_yaml(out_dir)
    else:
        start_epoch = 0
        train_total_hist, val_total_hist = [], []
        train_data_hist,  val_data_hist  = [], []
        train_phys_hist,  val_phys_hist  = [], []
        lambda_hist = []

    if start_epoch > int(cfg_pretraining.epochs):
        raise ValueError(
            f"history.yaml says epoch={start_epoch} but cfg_pretraining.epochs={cfg_pretraining.epochs}."
        )

    # ================== visuals ==================
    val_vis_ds = (
        val_ds.unbatch()
        .shuffle(4096, reshuffle_each_iteration=True)
        .batch(cfg_pretraining.batch_size, drop_remainder=True)
    )
    val_vis_it = iter(val_vis_ds.repeat())

    # Ensure fig dir exists in resume mode too
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ================== training loop ==================
    for epoch in range(start_epoch, cfg_pretraining.epochs):
        train_total.reset_state()
        train_data.reset_state()
        train_phys.reset_state()
        train_lam.reset_state()

        val_total.reset_state()
        val_data.reset_state()
        val_phys.reset_state()

        for x_b, y_b in train_ds:
            train_step(x_b, y_b)

        itv = iter(val_ds)
        for _ in range(50):
            x_b, y_b = next(itv)
            val_step(x_b, y_b)

        # append histories
        train_total_hist.append(float(train_total.result().numpy()))
        train_data_hist.append(float(train_data.result().numpy()))
        train_phys_hist.append(float(train_phys.result().numpy()))

        val_total_hist.append(float(val_total.result().numpy()))
        val_data_hist.append(float(val_data.result().numpy()))
        val_phys_hist.append(float(val_phys.result().numpy()))

        lambda_hist.append(float(train_lam.result().numpy()))

        print(
            f"[epoch {epoch+1}/{cfg_pretraining.epochs}] "
            f"train_total={train_total_hist[-1]:.6e} "
            f"train_data={train_data_hist[-1]:.6e} "
            f"train_phys={train_phys_hist[-1]:.6e} "
            f"lambda_phys={float(train_lam.result().numpy()):.3e} "
            f"val_total={val_total_hist[-1]:.6e}"
        )

        # plots + comparisons
        save_loss_plot(
            train_total_hist, val_total_hist,
            train_data_hist,  val_data_hist,
            train_phys_hist,  val_phys_hist,
            lambda_hist,                      # <-- NEW
            fig_dir / "loss_curve.png",
        )

        x_vis, y_vis = next(val_vis_it)
        save_speed_compare(mapping, x_vis, y_vis, Nz, fig_dir / f"speed_compare_epoch{epoch+1:04d}.png")

        ckpt_mgr.save()
        save_history_yaml(
            out_dir=out_dir,
            epoch=epoch + 1,
            train_total_hist=train_total_hist,
            val_total_hist=val_total_hist,
            train_data_hist=train_data_hist,
            val_data_hist=val_data_hist,
            train_phys_hist=train_phys_hist,
            val_phys_hist=val_phys_hist,
            lambda_hist=lambda_hist,   # <-- NEW
        )


    # ================== export ==================
    save_emulator_artifact(
        artifact_dir=out_dir,
        cfg=cfg,
        model=state.iceflow_model,
        inputs=list(inputs),
    )
    print(f"[export] saved emulator artifact to {out_dir}")

    k = min(5, len(val_total_hist))
    state.score = float(np.mean(val_total_hist[-k:]))

        
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

def save_loss_plot(
    train_total_hist,
    val_total_hist,
    train_data_hist,
    val_data_hist,
    train_phys_hist,
    val_phys_hist,
    lambda_hist,          # <-- NEW
    fig_path: Path,
) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(train_total_hist)
    epochs = np.arange(1, n + 1)

    train_total = np.asarray(train_total_hist, dtype=float)
    val_total   = np.asarray(val_total_hist, dtype=float)
    train_data  = np.asarray(train_data_hist, dtype=float)
    val_data    = np.asarray(val_data_hist, dtype=float)
    train_phys  = np.asarray(train_phys_hist, dtype=float)
    val_phys    = np.asarray(val_phys_hist, dtype=float)
    lam         = np.asarray(lambda_hist, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7, 8), sharex=True)
    ax0, ax1 = axes[0]
    ax2, ax3 = axes[1]

    # ---- subplot 1: total  ----
    ax0.plot(epochs, train_total, label="train_total")
    ax0.plot(epochs, val_total,   label="val_total")
    ax0.set_ylabel("loss (total)")
    min_total = np.min([train_total.min(), val_total.min()])
    if min_total > 0:
        ax0.set_yscale("log")
        ax0.grid(True, which="both", alpha=0.3)
    else:
        ax0.grid(True, which="major", alpha=0.3)
    ax0.legend(ncol=2, fontsize=8)

    # ---- subplot 2: physics ----
    ax1.plot(epochs, train_phys, label="train_phys", linestyle=":")
    ax1.plot(epochs, val_phys,   label="val_phys",   linestyle=":")
    ax1.set_ylabel("physics loss")
    min_phys = np.min([train_phys.min(), val_phys.min()])
    if min_phys > 0:
        ax1.set_yscale("log")
        ax1.grid(True, which="both", alpha=0.3)
    else:
        ax1.grid(True, which="major", alpha=0.3)
    ax1.legend(ncol=2, fontsize=8)

    # ---- subplot 3: data ----
    ax2.plot(epochs, train_data, label="train_data", linestyle="--")
    ax2.plot(epochs, val_data,   label="val_data",   linestyle="--")
    ax2.set_ylabel("data loss")
    ax2.set_yscale("log")
    ax2.grid(True, which="major", alpha=0.3)
    ax2.legend(ncol=2, fontsize=8)

    # ---- subplot 4: lambda_phys ----
    ax3.plot(epochs, lam, label="lambda_phys")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("lambda_phys")
    if np.all(lam > 0):
        ax3.set_yscale("log")  # lambda is usually positive and spans orders of magnitude
        ax3.grid(True, which="both", alpha=0.3)
    else:
        ax3.grid(True, which="major", alpha=0.3)
    ax3.legend(fontsize=8)

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

def _history_path(out_dir: Path) -> Path:
    return out_dir / "history.yaml"

def load_history_yaml(out_dir: Path):

    path = _history_path(out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"resume=True but missing history file: {path}. "
            "Expected history.yaml alongside checkpoints."
        )

    data = yaml.safe_load(path.read_text()) or {}
    epoch = int(data.get("epoch", 0))

    def _lst(key):
        v = data.get(key, [])
        if v is None:
            v = []
        return list(v)

    return (
        epoch,
        _lst("train_total"),
        _lst("val_total"),
        _lst("train_data"),
        _lst("val_data"),
        _lst("train_phys"),
        _lst("val_phys"),
        _lst("lambda_phys"),   # <-- NEW
    )

def save_history_yaml(
    out_dir: Path,
    epoch: int,
    train_total_hist,
    val_total_hist,
    train_data_hist,
    val_data_hist,
    train_phys_hist,
    val_phys_hist,
    lambda_hist,   # <-- NEW
) -> None:

    payload = {
        "epoch": int(epoch),
        "train_total": [float(x) for x in train_total_hist],
        "val_total":   [float(x) for x in val_total_hist],
        "train_data":  [float(x) for x in train_data_hist],
        "val_data":    [float(x) for x in val_data_hist],
        "train_phys":  [float(x) for x in train_phys_hist],
        "val_phys":    [float(x) for x in val_phys_hist],
        "lambda_phys": [float(x) for x in lambda_hist],  # <-- NEW
    }

    path = _history_path(out_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False))
    tmp.replace(path)
