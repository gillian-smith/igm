#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
artifacts.py

Core save/load + weight IO. Validation lives in schema modules.

Policy:
- Saving: schema v3 ONLY.
- Loading: supports schema v2 and v3.
- Normalization:
  * stats are stored in manifest.yaml
  * on load, attach FixedChannelStandardization built from manifest stats
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
import yaml

from rich.console import Console, Group
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel

from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.utils.math.precision import normalize_precision
from igm.processes.iceflow.emulate.utils.normalizations import FixedChannelStandardization

from .artifacts_schema_v2 import EmulatorManifestV2, parse_manifest_v2, validate_cfg_or_raise_v2
from .artifacts_schema_v3 import (
    EmulatorManifestV3,
    build_manifest_v3,
    parse_manifest_v3,
    validate_and_reconcile_cfg_v3,
)

# -----------------------------------------------------------------------------
# Rich printing (minimal + consistent)
# -----------------------------------------------------------------------------

_emulator_theme = Theme(
    {
        "label": "bold #e5e7eb",
        "value": "#06b6d4",
        "path": "#a78bfa",
        "ok": "bold #22c55e",
        "warn": "bold #f59e0b",
        "muted": "italic #64748b",
    }
)
_console = Console(theme=_emulator_theme)


def _print_loaded_banner(
    artifact_dir: Path,
    manifest: Union[EmulatorManifestV2, EmulatorManifestV3],
    dtype: tf.DType,
    notes: List[str],
) -> None:
    info = Table(show_header=False, border_style="green", expand=False)
    info.add_column("Label", style="label")
    info.add_column("Value", style="value")

    info.add_row("Architecture", str(manifest.architecture.name))
    info.add_row("I/O", f"{manifest.nb_inputs} → {manifest.nb_outputs}   (Nz={manifest.Nz}, dtype={dtype.name})")
    info.add_row("Artifact", f"[path]{artifact_dir}[/path]")

    body = [info]

    if notes:
        warn = Table(show_header=False, border_style="yellow", expand=False)
        warn.add_column("", width=2)
        warn.add_column("Warnings", style="warn")
        for m in notes:
            warn.add_row("⚠", m)
        body.append(warn)
        title = "[warn]⚠ Emulator loaded (warnings)[/warn]"
    else:
        title = "[ok]✅ Emulator loaded successfully[/ok]"

    _console.print()
    _console.print(
        Panel(
            Group(*body),
            title=title,
            subtitle="[muted]Ready for inference[/muted]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    _console.print()


# -----------------------------------------------------------------------------
# Save (schema v3 only)
# -----------------------------------------------------------------------------

def save_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
    model: tf.keras.Model,
    inputs: List[str],
) -> Path:
    """
    Save a schema v3 artifact:
      - export/weights.weights.h5  (weights only)
      - manifest.yaml             (schema v3 with full required fields)
    """
    artifact_dir = Path(artifact_dir)
    export_dir = artifact_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    cfg_numerics = cfg.processes.iceflow.numerics
    desired_dtype = normalize_precision(cfg_numerics.precision)

    nb_inputs = int(len(inputs))

    # Force-build model weights (subclassed model safety)
    dummy = tf.zeros((1, 8, 8, nb_inputs), dtype=desired_dtype)
    y = model(dummy, training=False)
    nb_outputs = int(y.shape[-1])

    # Save weights
    weights_path = export_dir / "weights.weights.h5"
    model.save_weights(str(weights_path))

    # Write schema v3 manifest
    manifest = build_manifest_v3(
        cfg=cfg,
        model=model,
        inputs=inputs,
        nb_outputs=nb_outputs,
    )
    (artifact_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))

    return artifact_dir


# -----------------------------------------------------------------------------
# Load (schema v2 + v3)
# -----------------------------------------------------------------------------

def load_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
) -> Tuple[tf.keras.Model, Union[EmulatorManifestV2, EmulatorManifestV3]]:
    """
    Load schema v2 or v3 artifacts.

    Flow:
      1) read + parse manifest
      2) validate (v2 for old format, v3 is the current standard)
      3) rebuild model, load weights
      4) attach FixedChannelStandardization from manifest stats
      5) print rich summary (including warnings)
    """
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.yaml at {manifest_path}")

    raw = yaml.safe_load(manifest_path.read_text())
    schema = int(raw.get("schema_version", -1))

    cfg_numerics = cfg.processes.iceflow.numerics
    desired_dtype = normalize_precision(cfg_numerics.precision)

    notes: List[str] = []

    if schema == 2:
        manifest = parse_manifest_v2(raw)
        # v2: simple strict errors like before
        validate_cfg_or_raise_v2(cfg, manifest)
    elif schema == 3:
        manifest = parse_manifest_v3(raw)
        # v3: rich error/warn logic + cfg reconciliation
        notes = validate_and_reconcile_cfg_v3(cfg, manifest, artifact_dir)
    else:
        raise ValueError(f"Unsupported schema_version={schema!r}. Supported: 2, 3")

    arch_name = str(manifest.architecture.name)
    if arch_name not in Architectures:
        raise ValueError(f"Unknown architecture {arch_name!r}. Available: {list(Architectures.keys())}")

    # 1) Rebuild model (cfg may have been reconciled for schema v3)
    model = Architectures[arch_name](cfg, manifest.nb_inputs, manifest.nb_outputs)
    model.input_normalizer = None

    # Ensure variables exist
    dummy = tf.zeros((1, 8, 8, manifest.nb_inputs), dtype=desired_dtype)
    _ = model(dummy, training=False)

    # 2) Load weights
    weights_path = artifact_dir / "export" / "weights.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file at {weights_path}")
    model.load_weights(str(weights_path))

    # 3) Attach FixedChannelStandardization from manifest stats
    p = manifest.normalization.params
    eps = float(p.get("epsilon", p.get("variance_epsilon", 1e-7)))
    mean_1d = np.asarray(p["mean_1d"], dtype=np.float64).reshape(-1)
    var_1d = np.asarray(p["var_1d"], dtype=np.float64).reshape(-1)

    if mean_1d.shape[0] != manifest.nb_inputs or var_1d.shape[0] != manifest.nb_inputs:
        raise ValueError(
            f"Normalization stats length mismatch: mean={mean_1d.shape}, var={var_1d.shape}, "
            f"nb_inputs={manifest.nb_inputs}"
        )

    model.input_normalizer = FixedChannelStandardization(
        mean_1d=mean_1d,
        var_1d=var_1d,
        epsilon=eps,
        dtype=desired_dtype,
        name="input_norm",
    )
    _ = model.input_normalizer(tf.zeros((1, 2, 2, manifest.nb_inputs), dtype=desired_dtype))

    # dtype sanity
    y = model(tf.zeros((1, 2, 2, manifest.nb_inputs), dtype=desired_dtype), training=False)
    if tf.as_dtype(y.dtype) != desired_dtype:
        raise RuntimeError(
            f"Model forward dtype is {tf.as_dtype(y.dtype).name}, expected {desired_dtype.name}."
        )

    _print_loaded_banner(artifact_dir, manifest, desired_dtype, notes)
    return model, manifest