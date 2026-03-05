#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import yaml


def _history_path(out_dir: Path) -> Path:
    return out_dir / "history.yaml"


def load_history_yaml(out_dir: Path) -> Tuple[int, List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    path = _history_path(out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"resume=True but missing history file: {path}. "
            "Expected history.yaml alongside checkpoints."
        )

    data = yaml.safe_load(path.read_text()) or {}
    epoch = int(data.get("epoch", 0))

    def _lst(key: str) -> List[float]:
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
        _lst("lambda_phys"),
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
    lambda_hist,
) -> None:
    payload = {
        "epoch": int(epoch),
        "train_total": [float(x) for x in train_total_hist],
        "val_total":   [float(x) for x in val_total_hist],
        "train_data":  [float(x) for x in train_data_hist],
        "val_data":    [float(x) for x in val_data_hist],
        "train_phys":  [float(x) for x in train_phys_hist],
        "val_phys":    [float(x) for x in val_phys_hist],
        "lambda_phys": [float(x) for x in lambda_hist],
    }

    path = _history_path(out_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False))
    tmp.replace(path)
