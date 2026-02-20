#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
artifacts.py (schema v2 ONLY)

Normalization policy (ONLY supported approach):
- The network uses tf.keras.layers.Normalization(axis=-1) in the forward pass.
- The layer is adapted exactly once during training (NOT here).
- We persist the adapted per-channel mean/variance explicitly in manifest.yaml.
- On load, we rebuild the model, load network weights, then rebuild a Normalization
  layer and assign mean/variance from the manifest (decoupled from Keras weight paths).

No backwards compatibility:
- schema_version must be 2
- normalization.method must be "keras_normalization"
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import yaml
import warnings

from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel

from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.utils.math.precision import normalize_precision
from igm.processes.iceflow.emulate.utils.normalizations import FixedChannelStandardization

# -----------------------------------------------------------------------------
# Manifest dataclasses (minimal + explicit)
# -----------------------------------------------------------------------------

@dataclass
class ArchitectureSpec:
    name: str
    params: Dict[str, Any]


@dataclass
class NormalizationSpec:
    method: str
    params: Dict[str, Any]


@dataclass
class EmulatorManifest:
    schema_version: int
    Nz: int
    inputs: List[str]
    nb_inputs: int
    nb_outputs: int
    output_scale: float
    architecture: ArchitectureSpec
    normalization: NormalizationSpec

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------

_emulator_theme = Theme({
    "label": "bold #e5e7eb",
    "value": "#06b6d4",
    "path": "#a78bfa",
    "ok": "bold #22c55e",
    "muted": "italic #64748b",
})

def _print_emulator_loaded_banner(artifact_dir: Path, manifest: EmulatorManifest, dtype: tf.DType) -> None:
    console = Console(theme=_emulator_theme)

    # 2–3 lines of info, compact + readable
    table = Table(show_header=False, border_style="green", expand=False)
    table.add_column("Label", style="label")
    table.add_column("Value", style="value")

    table.add_row("Architecture", str(manifest.architecture.name))
    table.add_row("I/O", f"{manifest.nb_inputs} → {manifest.nb_outputs}   (Nz={manifest.Nz}, dtype={dtype.name})")
    table.add_row("Artifact", f"[path]{artifact_dir}[/path]")

    console.print()
    console.print(
        Panel(
            table,
            title="[ok]✅ Emulator loaded successfully[/ok]",
            subtitle="[muted]Weights + fixed input standardization attached[/muted]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

# -----------------------------------------------------------------------------
# Simple cfg <-> manifest checks (keep strict but minimal)
# -----------------------------------------------------------------------------

def _require_equal(label: str, run: Any, artifact: Any) -> None:
    if run != artifact:
        raise ValueError(f"{label} mismatch: run={run!r} vs artifact={artifact!r}")


def validate_cfg_against_manifest(cfg, manifest: EmulatorManifest) -> None:
    cfg_unified = cfg.processes.iceflow.unified
    cfg_numerics = cfg.processes.iceflow.numerics

    _require_equal("schema_version", 2, int(manifest.schema_version))
    _require_equal("Nz", int(cfg_numerics.Nz), int(manifest.Nz))
    _require_equal("inputs", list(cfg_unified.inputs), list(manifest.inputs))
    _require_equal("architecture.name", str(cfg_unified.network.architecture), str(manifest.architecture.name))
    _require_equal("output_scale", float(cfg_unified.network.output_scale), float(manifest.output_scale))

    # Enforce normalization contract (policy), not cfg mirroring
    _require_equal("normalization.method", "keras_normalization", str(manifest.normalization.method))

# -----------------------------------------------------------------------------
# Normalization spec extraction + rebuild (schema v2 only)
# -----------------------------------------------------------------------------

def extract_normalization_spec(model: tf.keras.Model) -> NormalizationSpec:
    """
    Extract adapted stats from model.input_normalizer (must be tf.keras.layers.Normalization).
    This function does NOT adapt; it only reads the already-adapted layer.
    """
    norm = getattr(model, "input_normalizer", None)
    if norm is None:
        raise ValueError("Model has no input_normalizer attached; cannot extract normalization stats.")
    if not isinstance(norm, tf.keras.layers.Normalization):
        raise TypeError(
            "Expected model.input_normalizer to be tf.keras.layers.Normalization. "
            f"Got: {type(norm)}"
        )

    # Ensure stats exist (adapt called) and are finite
    if not hasattr(norm, "mean") or not hasattr(norm, "variance"):
        raise RuntimeError("Normalization layer is missing mean/variance attributes (unexpected Keras version?).")

    mean = np.asarray(norm.mean.numpy(), dtype=np.float64).reshape(-1)
    var = np.asarray(norm.variance.numpy(), dtype=np.float64).reshape(-1)

    if mean.size == 0 or var.size == 0:
        raise RuntimeError(
            "Normalization layer mean/variance appear empty. "
            "Was norm.adapt(...) called and did the layer build correctly?"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise RuntimeError("Normalization layer mean/variance contain NaN/Inf.")
    if np.any(var < 0):
        raise RuntimeError("Normalization layer variance contains negative values (unexpected).")

    # Keras Normalization uses variance_epsilon (default ~1e-7)
    eps = float(getattr(norm, "variance_epsilon", 1e-7))

    return NormalizationSpec(
        method="keras_normalization",
        params={
            "axis": -1,
            "variance_epsilon": eps,
            "mean_1d": mean.tolist(),
            "var_1d": var.tolist(),
            "stats_source": "trained_once_via_norm.adapt",
        },
    )


# -----------------------------------------------------------------------------
# Minimal architecture spec (for traceability; loader uses cfg to rebuild model)
# -----------------------------------------------------------------------------

def extract_architecture_spec(cfg) -> ArchitectureSpec:
    """
    Store architecture name + a small set of hyperparams from cfg for traceability.
    Loader rebuilds the model from cfg and only checks architecture name for consistency.
    """
    arch_name = str(cfg.processes.iceflow.unified.network.architecture)

    # If you prefer even more minimal, you can set params={} and keep only name.
    net_cfg = cfg.processes.iceflow.emulator.network
    params = {
        "nb_layers": int(getattr(net_cfg, "nb_layers")),
        "nb_out_filter": int(getattr(net_cfg, "nb_out_filter")),
        "conv_ker_size": int(getattr(net_cfg, "conv_ker_size")),
        "activation": str(getattr(net_cfg, "activation")),
        "weight_initialization": str(getattr(net_cfg, "weight_initialization")),
        "batch_norm": bool(getattr(net_cfg, "batch_norm", False)),
        "residual": bool(getattr(net_cfg, "residual", False)),
        "separable": bool(getattr(net_cfg, "separable", False)),
        "dropout_rate": float(getattr(net_cfg, "dropout_rate", 0.0)),
        "l2_reg": float(getattr(net_cfg, "l2_reg", 0.0)),
        "cnn3d_for_vertical": bool(getattr(net_cfg, "cnn3d_for_vertical", False)),
        "trained_precision": getattr(cfg.processes.iceflow.numerics, "precision"),
    }
    return ArchitectureSpec(name=arch_name, params=params)

# -----------------------------------------------------------------------------
# Save / Load (schema v2 only)
# -----------------------------------------------------------------------------

def save_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
    model: tf.keras.Model,
    inputs: List[str],
) -> Path:
    """
    Saves:
      - export/weights.weights.h5 (network weights ONLY)

    Assumptions:
    - model weights are built (we force-build with a dummy call)
    - normalization stats are stored in manifest.yaml (written during training), not in weights
    """
    artifact_dir = Path(artifact_dir)
    export_dir = artifact_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    cfg_numerics = cfg.processes.iceflow.numerics

    nb_inputs = int(len(inputs))

    # Ensure model variables exist before saving weights (subclassed Model)
    desired_dtype = normalize_precision(cfg_numerics.precision)
    dummy = tf.zeros((1, 8, 8, nb_inputs), dtype=desired_dtype)
    _ = model(dummy, training=False)

    weights_path = export_dir / "weights.weights.h5"
    model.save_weights(str(weights_path))

    return artifact_dir




def load_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
) -> Tuple[tf.keras.Model, EmulatorManifest]:
    """
    Strict schema v2 loader:
      - reads manifest.yaml
      - validates cfg invariants
      - rebuilds model from cfg via Architectures[arch_name]
      - loads weights (network weights only)
      - rebuilds tf.keras.layers.Normalization and assigns stats from manifest
      - attaches it as model.input_normalizer
    """
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.yaml at {manifest_path}")

    raw = yaml.safe_load(manifest_path.read_text())

    if int(raw.get("schema_version", -1)) != 2:
        raise ValueError(f"Only schema_version=2 is supported. Found: {raw.get('schema_version')!r}")

    arch = raw["architecture"]
    norm = raw["normalization"]

    manifest = EmulatorManifest(
        schema_version=int(raw["schema_version"]),
        Nz=int(raw["Nz"]),
        inputs=list(raw["inputs"]),
        nb_inputs=int(raw["nb_inputs"]),
        nb_outputs=int(raw["nb_outputs"]),
        output_scale=float(raw["output_scale"]),
        architecture=ArchitectureSpec(name=str(arch["name"]), params=dict(arch.get("params", {}))),
        normalization=NormalizationSpec(method=str(norm["method"]), params=dict(norm.get("params", {}))),
    )

    validate_cfg_against_manifest(cfg, manifest)

    arch_name = str(manifest.architecture.name)
    if arch_name not in Architectures:
        raise ValueError(f"Unknown architecture {arch_name!r}. Available: {list(Architectures.keys())}")

    desired_dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    # Warn if artifact was trained with a different precision than requested.
    trained_p = manifest.architecture.params.get("trained_precision", None)
    if trained_p and trained_p != "unknown":
        try:
            trained_dt = normalize_precision(str(trained_p))
        except Exception:
            warnings.warn(
                f"Artifact reports trained_precision={trained_p!r}, but it could not be parsed as a TF dtype. "
                f"Proceeding with cfg precision={desired_dtype.name}."
            )
        else:
            if trained_dt != desired_dtype:
                direction = "up-cast" if desired_dtype.size >= trained_dt.size else "down-cast"
                warnings.warn(
                    f"Precision mismatch: artifact trained in {trained_dt.name}, "
                    f"but cfg requests {desired_dtype.name}. Weights will be {direction} on load."
                )
    # 1) Rebuild model WITHOUT normalizer first to keep weight-loading independent of normalizer vars/paths.
    model = Architectures[arch_name](cfg, manifest.nb_inputs, manifest.nb_outputs)
    model.input_normalizer = None

    # Ensure model vars exist
    dummy = tf.zeros((1, 8, 8, manifest.nb_inputs), dtype=desired_dtype)
    _ = model(dummy, training=False)

    weights_path = artifact_dir / "export" / "weights.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file at {weights_path}")

    # 2) Load network weights
    model.load_weights(str(weights_path))

    # 3) Attach FixedChannelStandardization using manifest stats (forward-pass normalizer)
    p = manifest.normalization.params
    eps = float(p.get("variance_epsilon", 1e-7))

    mean_1d = np.asarray(p["mean_1d"], dtype=np.float64).reshape(-1)
    var_1d  = np.asarray(p["var_1d"],  dtype=np.float64).reshape(-1)

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

    # Build fixed normalizer once
    _ = model.input_normalizer(tf.zeros((1, 2, 2, manifest.nb_inputs), dtype=desired_dtype))


    y = model(tf.zeros((1, 2, 2, manifest.nb_inputs), dtype=desired_dtype), training=False)
    if tf.as_dtype(y.dtype) != desired_dtype:
        raise RuntimeError(
            f"Model forward dtype is {tf.as_dtype(y.dtype).name}, expected {desired_dtype.name}. "
        )
    
    _print_emulator_loaded_banner(artifact_dir, manifest, desired_dtype)

    return model, manifest
