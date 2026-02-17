#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
import tensorflow as tf

def load_metadata(tfrecord_root: Path) -> dict:
    meta_path = tfrecord_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json at {meta_path}")
    return json.loads(meta_path.read_text())

def list_shards(tfrecord_root: Path, nz: int, split: str) -> List[str]:
    shard_dir = tfrecord_root / split / f"nz{nz}"
    if not shard_dir.exists():
        raise FileNotFoundError(f"Missing Nz folder for split '{split}': {shard_dir}")

    # be robust to filename prefix changes
    files = sorted(str(p) for p in shard_dir.glob("*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No TFRecord shards found in {shard_dir}")
    return files

def parse_example(serialized: tf.Tensor, H: int, W: int, Nz: int) -> Tuple[tf.Tensor, tf.Tensor]:
    feat = {
        "seed": tf.io.FixedLenFeature([], tf.int64),
        "t": tf.io.FixedLenFeature([], tf.float32),
        "nz": tf.io.FixedLenFeature([], tf.int64),
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(serialized, feat)

    x = tf.io.parse_tensor(ex["x"], out_type=tf.float32)
    y = tf.io.parse_tensor(ex["y"], out_type=tf.float32)

    x = tf.ensure_shape(x, [H, W, 3])      # thk, usurf, slidingco
    y = tf.ensure_shape(y, [Nz, H, W, 2])  # (U,V)
    return x, y

def make_datasets(
    train_files: List[str],
    val_files: List[str],
    H: int,
    W: int,
    Nz: int,
    compression: str,
    batch_size: int,
    shuffle_buffer: int = 2048,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    def ds_from(files: List[str], training: bool) -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(files, compression_type=compression)
        ds = ds.map(lambda s: parse_example(s, H, W, Nz), num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return ds_from(train_files, True), ds_from(val_files, False)
