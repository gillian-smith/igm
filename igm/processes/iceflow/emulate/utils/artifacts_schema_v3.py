#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors

"""
artifacts_schema_v3.py

Schema v3:
- manifest is complete: Nz, basis_vertical, basis_horizontal, inputs are required
- numerics mismatches (Nz/basis_*) => rich error + exception (do NOT override cfg)
- other mismatches => rich warnings + override cfg to manifest
- precision mismatch => rich warning (no cfg change; weights cast via TF policy)
- also owns v3 manifest writing to avoid format drift
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from omegaconf import open_dict

from rich.console import Console, Group
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel

from igm.utils.math.precision import normalize_precision


# -----------------------------------------------------------------------------
# Rich printing
# -----------------------------------------------------------------------------

_theme = Theme(
    {
        "label": "bold #e5e7eb",
        "value": "#06b6d4",
        "path": "#a78bfa",
        "warn": "bold #f59e0b",
        "err": "bold #ef4444",
        "muted": "italic #64748b",
    }
)
_console = Console(theme=_theme)


# -----------------------------------------------------------------------------
# Manifest dataclasses
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
class EmulatorManifestV3:
    schema_version: int  # must be 3

    # Required numerics + semantics
    Nz: int
    basis_vertical: str
    basis_horizontal: str
    inputs: List[str]

    nb_inputs: int
    nb_outputs: int
    output_scale: float

    architecture: ArchitectureSpec
    normalization: NormalizationSpec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# v3 parsing / writing
# -----------------------------------------------------------------------------

def parse_manifest_v3(raw: Dict[str, Any]) -> EmulatorManifestV3:
    if int(raw.get("schema_version", -1)) != 3:
        raise ValueError(f"parse_manifest_v3 expects schema_version=3, got {raw.get('schema_version')!r}")

    # Required keys (fail fast on corrupt v3 manifests)
    for k in ("Nz", "basis_vertical", "basis_horizontal", "inputs", "nb_inputs", "nb_outputs", "output_scale", "architecture", "normalization"):
        if k not in raw:
            raise ValueError(f"Schema v3 manifest missing required field {k!r}")

    arch = raw["architecture"]
    norm = raw["normalization"]

    return EmulatorManifestV3(
        schema_version=3,
        Nz=int(raw["Nz"]),
        basis_vertical=str(raw["basis_vertical"]),
        basis_horizontal=str(raw["basis_horizontal"]),
        inputs=list(raw["inputs"]),
        nb_inputs=int(raw["nb_inputs"]),
        nb_outputs=int(raw["nb_outputs"]),
        output_scale=float(raw["output_scale"]),
        architecture=ArchitectureSpec(name=str(arch["name"]), params=dict(arch.get("params", {}))),
        normalization=NormalizationSpec(method=str(norm["method"]), params=dict(norm.get("params", {}))),
    )


def _extract_normalization_spec(model: tf.keras.Model) -> NormalizationSpec:
    """
    Reads adapted stats from model.input_normalizer (tf.keras.layers.Normalization).
    Does NOT adapt.
    """
    norm = getattr(model, "input_normalizer", None)
    if norm is None or not isinstance(norm, tf.keras.layers.Normalization):
        raise TypeError("Saving schema v3 expects model.input_normalizer to be tf.keras.layers.Normalization")

    mean = np.asarray(norm.mean.numpy(), dtype=np.float64).reshape(-1)
    var = np.asarray(norm.variance.numpy(), dtype=np.float64).reshape(-1)
    eps = float(getattr(norm, "variance_epsilon", 1e-7))

    if mean.size == 0 or var.size == 0:
        raise RuntimeError("Normalization stats empty; did you call norm.adapt(...) during training?")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise RuntimeError("Normalization stats contain NaN/Inf.")

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


def _extract_architecture_spec(cfg) -> ArchitectureSpec:
    """
    Minimal traceability; loader uses cfg to rebuild but v3 validation can reconcile.
    """
    arch_name = str(cfg.processes.iceflow.unified.network.architecture)

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


def build_manifest_v3(cfg, model: tf.keras.Model, inputs: List[str], nb_outputs: int) -> EmulatorManifestV3:
    """
    Single source of truth for how schema v3 is written.
    """
    cfg_unified = cfg.processes.iceflow.unified
    cfg_numerics = cfg.processes.iceflow.numerics

    # If caller passes inputs different from cfg, that’s almost always a bug.
    cfg_inputs = list(cfg_unified.inputs)
    if list(inputs) != cfg_inputs:
        raise ValueError(f"Refusing to save: inputs arg {list(inputs)!r} != cfg.unified.inputs {cfg_inputs!r}")

    arch = _extract_architecture_spec(cfg)
    norm = _extract_normalization_spec(model)

    return EmulatorManifestV3(
        schema_version=3,
        Nz=int(cfg_numerics.Nz),
        basis_vertical=str(cfg_numerics.basis_vertical),
        basis_horizontal=str(cfg_numerics.basis_horizontal),
        inputs=list(inputs),
        nb_inputs=int(len(inputs)),
        nb_outputs=int(nb_outputs),
        output_scale=float(cfg_unified.network.output_scale),
        architecture=arch,
        normalization=norm,
    )


# -----------------------------------------------------------------------------
# v3 validation / reconciliation
# -----------------------------------------------------------------------------

def _raise_numerics_incompatibility(artifact_dir: Path, manifest: EmulatorManifestV3, errors: List[str]) -> None:
    info = Table(show_header=False, border_style="red", expand=False)
    info.add_column("Label", style="label")
    info.add_column("Value", style="err")

    info.add_row("Artifact", f"[path]{artifact_dir}[/path]")
    info.add_row("Nz", str(manifest.Nz))
    info.add_row("basis_vertical", manifest.basis_vertical)
    info.add_row("basis_horizontal", manifest.basis_horizontal)

    err = Table(show_header=False, border_style="red", expand=False)
    err.add_column("", width=2)
    err.add_column("Incompatibilities", style="err")
    for m in errors:
        err.add_row("✖", m)

    _console.print()
    _console.print(
        Panel(
            Group(info, err),
            title="[err]✖ Emulator artifact incompatible with current numerics[/err]",
            subtitle="[muted]Nz/basis must match because discretization is configured before loading[/muted]",
            border_style="red",
            padding=(1, 2),
        )
    )
    _console.print()
    raise ValueError("Emulator artifact incompatible with numerics (see panel above).")


def validate_and_reconcile_cfg_v3(cfg, manifest: EmulatorManifestV3, artifact_dir: Path) -> List[str]:
    """
    Schema v3 behavior:
      - ERROR if Nz/basis_vertical/basis_horizontal mismatch (no overrides)
      - WARN + override cfg for inputs, architecture, output_scale
      - WARN (no override) if trained_precision differs from cfg precision
    Returns: warnings list (to be shown by core loader).
    """
    cfg_unified = cfg.processes.iceflow.unified
    cfg_numerics = cfg.processes.iceflow.numerics
    desired_dtype = normalize_precision(cfg_numerics.precision)

    errors: List[str] = []
    warnings: List[str] = []

    # Strict numerics invariants (must match; do NOT override)
    if int(cfg_numerics.Nz) != int(manifest.Nz):
        errors.append(f"Nz mismatch: cfg={int(cfg_numerics.Nz)} vs artifact={int(manifest.Nz)}")
    if str(cfg_numerics.basis_vertical) != str(manifest.basis_vertical):
        errors.append(
            f"basis_vertical mismatch: cfg={str(cfg_numerics.basis_vertical)!r} vs artifact={str(manifest.basis_vertical)!r}"
        )
    if str(cfg_numerics.basis_horizontal) != str(manifest.basis_horizontal):
        errors.append(
            f"basis_horizontal mismatch: cfg={str(cfg_numerics.basis_horizontal)!r} vs artifact={str(manifest.basis_horizontal)!r}"
        )

    if errors:
        _raise_numerics_incompatibility(artifact_dir, manifest, errors)

    # Non-strict: override cfg to artifact (noisy warnings)
    with open_dict(cfg):
        cfg_inputs = list(cfg_unified.inputs)
        art_inputs = list(manifest.inputs)
        if cfg_inputs != art_inputs:
            warnings.append(f"inputs mismatch: cfg={cfg_inputs!r} → using artifact={art_inputs!r} (overriding cfg)")
            cfg_unified.inputs = art_inputs

        cfg_arch = str(cfg_unified.network.architecture)
        art_arch = str(manifest.architecture.name)
        if cfg_arch != art_arch:
            warnings.append(f"architecture mismatch: cfg={cfg_arch!r} → using artifact={art_arch!r} (overriding cfg)")
            cfg_unified.network.architecture = art_arch

        cfg_os = float(cfg_unified.network.output_scale)
        art_os = float(manifest.output_scale)
        if cfg_os != art_os:
            warnings.append(f"output_scale mismatch: cfg={cfg_os} → using artifact={art_os} (overriding cfg)")
            cfg_unified.network.output_scale = art_os

    # Precision warning (no cfg change)
    trained_p = manifest.architecture.params.get("trained_precision", None)
    if trained_p and str(trained_p) != "unknown":
        try:
            trained_dt = normalize_precision(str(trained_p))
        except Exception:
            warnings.append(
                f"trained_precision={trained_p!r} present but unparsable; cfg requests {desired_dtype.name}. "
                "Weights will load under the requested TF policy."
            )
        else:
            if trained_dt != desired_dtype:
                warnings.append(
                    f"precision differs: artifact trained in {trained_dt.name} but cfg requests {desired_dtype.name}. "
                    "Weights/vars will be cast to the requested precision via the active TF policy."
                )

    return warnings