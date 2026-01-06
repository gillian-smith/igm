#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings


 
def initialize(cfg, state):
    cfg_pretraining = cfg.processes.pretraining
    cfg_iceflow = cfg.processes.iceflow
    Nz = cfg_iceflow.numerics.Nz
    tfrecord_root = Path('/home/srosier/work/synthetic_glaciers/tfrecords/set1/')
    out_dir = Path('/home/srosier/work/synthetic_glaciers/trained_models/')
    meta = load_metadata(tfrecord_root)

    shapes = meta["example_shapes_by_nz"][str(Nz)]
    H, W, Cx = shapes["x"]

    shard_files = list_shards(tfrecord_root, Nz)
    train_ds, val_ds = make_datasets(
        shard_files=shard_files,
        H=H, W=W, Nz=Nz,
        compression="GZIP",
        batch_size=cfg_pretraining.batch_size,
        val_fraction=0.1,
    )

    # Input normalization (stats persisted in the saved model)
    normalizer = tf.keras.layers.Normalization(axis=-1, name="x_norm")
    # Adapt on a bounded sample to keep “first run” practical; adjust as needed.
    norm_adapt = train_ds.unbatch().map(lambda x, y: x).take(5000).batch(256)
    normalizer.adapt(norm_adapt)

    # Initialize mapping
    mapping_args = InterfaceMappings["network"].get_mapping_args(cfg, state)
    mapping = Mappings["network"](**mapping_args)
    state.iceflow.mapping = mapping

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")

    @tf.function
    def loss_fn(x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        # Forward with BCs (via mapping)
        U_pred, V_pred = mapping.get_UV(x_batch)

        # Targets
        U_true = y_batch[..., 0]  # (B,Nz,H,W)
        V_true = y_batch[..., 1]  # (B,Nz,H,W)
        # Mask ice-free where thk <= 0 (prevents learning garbage off-ice)
        thk = x_batch[..., 0]  # (B,H,W)
        m2d = tf.cast(thk > 0.0, tf.float32)
        m = m2d[:, None, :, :]  # (B,1,H,W) -> broadcast to Nz
        denom = tf.reduce_sum(m) * tf.cast(Nz, tf.float32) + 1e-12

        du2 = tf.square(U_pred - U_true) * m
        dv2 = tf.square(V_pred - V_true) * m
        return (tf.reduce_sum(du2) + tf.reduce_sum(dv2)) / denom
    
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