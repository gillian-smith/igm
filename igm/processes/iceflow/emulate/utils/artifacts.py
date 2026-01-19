from __future__ import annotations

import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.emulate.utils.normalizations import (
    AdaptiveAffineLayer,
    FixedAffineLayer,
    IdentityTransformation,
    StandardizationLayer,
    NormalizationLayer,
)


# ----------------------------
# Manifest structures
# ----------------------------

@dataclass(frozen=True)
class NormalizationSpec:
    method: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class EmulatorManifest:
    schema_version: int
    Nz: int
    inputs: List[str]          # ordered list of input channel names
    nb_inputs: int
    nb_outputs: int
    output_scale: float
    architecture: ArchitectureSpec
    normalization: NormalizationSpec


# ----------------------------
# Helpers: extract specs from cfg
# ----------------------------

def _extract_cnn_params(cfg) -> Dict[str, Any]:
    net_cfg = cfg.processes.iceflow.emulator.network

    params = {
        "nb_layers": int(net_cfg.nb_layers),
        "nb_out_filter": int(net_cfg.nb_out_filter),
        "conv_ker_size": int(net_cfg.conv_ker_size),
        "activation": str(net_cfg.activation),
        "weight_initialization": str(net_cfg.weight_initialization),
        "batch_norm": bool(getattr(net_cfg, "batch_norm", False)),
        "residual": bool(getattr(net_cfg, "residual", False)),
        "separable": bool(getattr(net_cfg, "separable", False)),
        "dropout_rate": float(getattr(net_cfg, "dropout_rate", 0.0)),
        "l2_reg": float(getattr(net_cfg, "l2_reg", 0.0)) if hasattr(net_cfg, "l2_reg") else None,
        "cnn3d_for_vertical": bool(getattr(net_cfg, "cnn3d_for_vertical", False)),
    }
    # Remove None so json stays clean
    return {k: v for k, v in params.items() if v is not None}


def extract_architecture_spec(cfg, architecture_name: str) -> ArchitectureSpec:
    name = str(architecture_name)

    if name.lower() == "cnn":
        return ArchitectureSpec(name=name, params=_extract_cnn_params(cfg))

    raise NotImplementedError(
        f"Artifact saving/loading is strict and currently implemented for architecture={name!r}. "
        f"Add an extractor in extract_architecture_spec() for this architecture."
    )


def extract_normalization_spec(cfg, inputs: List[str], nb_inputs: int) -> NormalizationSpec:
    cfg_unified = cfg.processes.iceflow.unified
    method = str(cfg_unified.normalization.method)

    if method == "none":
        return NormalizationSpec(method="none", params={})

    if method == "automatic":
        # TO DO: confirm no params needed
        return NormalizationSpec(method="automatic", params={})

    if method == "adaptive":
        # Stats live in the weights (mean/variance variables), but we must reconstruct the layer shape.
        dtype = "float64" if str(cfg.processes.iceflow.numerics.precision).lower() == "double" else "float32"
        params = {
            "nb_channels": int(nb_inputs),
            "epsilon": float(getattr(cfg_unified.normalization, "epsilon", 1e-6)),
            "dtype": dtype,
        }
        return NormalizationSpec(method="adaptive", params=params)

    if method == "fixed":
        fixed = cfg_unified.normalization.fixed

        # IMPORTANT: FixedAffineLayer takes dicts. Preserve channel order by inserting keys in inputs order.
        offsets_cfg = dict(fixed.inputs_offsets)
        variances_cfg = dict(fixed.inputs_variances)

        offsets = {name: float(offsets_cfg[name]) for name in inputs}
        variances = {name: float(variances_cfg[name]) for name in inputs}

        dtype = "float64" if str(cfg.processes.iceflow.numerics.precision).lower() == "double" else "float32"
        params = {
            "offsets": offsets,
            "variances": variances,
            "epsilon": float(getattr(cfg_unified.normalization, "epsilon", 1e-6)),
            "dtype": dtype,
        }
        return NormalizationSpec(method="fixed", params=params)

    if method == "standardization":
        params = {
            "mode": str(getattr(cfg_unified.normalization, "mode", "channel")),
            "epsilon": float(getattr(cfg_unified.normalization, "epsilon", 1e-6)),
        }
        return NormalizationSpec(method="standardization", params=params)

    if method == "normalization":
        params = {
            "mode": str(getattr(cfg_unified.normalization, "mode", "channel")),
            "scale_range": tuple(getattr(cfg_unified.normalization, "scale_range", (0, 1))),
            "epsilon": float(getattr(cfg_unified.normalization, "epsilon", 1e-6)),
        }
        return NormalizationSpec(method="normalization", params=params)

    raise ValueError(f"Unknown normalization method: {method!r}")


# ----------------------------
# Helpers: build normalizer from spec
# ----------------------------

def build_normalizer_from_spec(spec: NormalizationSpec) -> tf.keras.layers.Layer:
    m = spec.method
    p = spec.params

    if m in ("none", "automatic"):
        return IdentityTransformation()

    if m == "adaptive":
        return AdaptiveAffineLayer(
            nb_channels=int(p["nb_channels"]),
            epsilon=float(p.get("epsilon", 1e-6)),
            dtype=str(p.get("dtype", "float32")),
        )

    if m == "fixed":
        return FixedAffineLayer(
            offsets=p["offsets"],
            variances=p["variances"],
            epsilon=float(p.get("epsilon", 1e-6)),
            dtype=str(p.get("dtype", "float32")),
        )

    if m == "standardization":
        return StandardizationLayer(mode=str(p.get("mode", "channel")), epsilon=float(p.get("epsilon", 1e-6)))

    if m == "normalization":
        return NormalizationLayer(
            mode=str(p.get("mode", "channel")),
            scale_range=tuple(p.get("scale_range", (0, 1))),
            epsilon=float(p.get("epsilon", 1e-6)),
        )

    raise ValueError(f"Unknown normalization method in spec: {m!r}")


def _force_build_normalizer(normalizer: tf.keras.layers.Layer, nb_inputs: int) -> None:
    """
    Ensures normalizer variables exist BEFORE loading weights.
    Fixed/Adaptive layers create weights in build(); we trigger it with a dummy input.
    """
    dummy = tf.zeros((1, 1, 1, nb_inputs), dtype=normalizer.dtype if hasattr(normalizer, "dtype") else tf.float32)
    _ = normalizer(dummy)


# ----------------------------
# Strict validation
# ----------------------------

def _require_equal(label: str, a: Any, b: Any) -> None:
    if a != b:
        raise ValueError(f"{label} mismatch: run={a!r} vs artifact={b!r}")


def validate_cfg_against_manifest(cfg, manifest: EmulatorManifest) -> None:
    cfg_unified = cfg.processes.iceflow.unified
    cfg_numerics = cfg.processes.iceflow.numerics

    _require_equal("Nz", int(cfg_numerics.Nz), int(manifest.Nz))
    _require_equal("inputs", list(cfg_unified.inputs), list(manifest.inputs))
    _require_equal("architecture.name", str(cfg_unified.network.architecture), str(manifest.architecture.name))
    _require_equal("normalization.method", str(cfg_unified.normalization.method), str(manifest.normalization.method))
    _require_equal("output_scale", float(cfg_unified.network.output_scale), float(manifest.output_scale))

    # Strict architecture hyperparameter validation (CNN only for now)
    arch = str(manifest.architecture.name)
    if arch.lower() == "cnn":
        expected = manifest.architecture.params
        got = _extract_cnn_params(cfg)
        _require_equal("architecture.params", got, expected)
    else:
        raise NotImplementedError(f"Strict validation not implemented for architecture={arch!r}")


# ----------------------------
# Public API: save / load
# ----------------------------

def save_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
    model: tf.keras.Model,
    inputs: List[str],
) -> Path:
    """
    Saves:
      - manifest.json
      - export/weights.weights.h5  (model.save_weights)
    Assumes model.input_normalizer has already been attached.
    """
    artifact_dir = Path(artifact_dir)
    export_dir = artifact_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_unified = cfg.processes.iceflow.unified

    Nz = int(cfg_numerics.Nz)
    nb_inputs = int(len(inputs))
    nb_outputs = int(2 * Nz)

    arch_spec = extract_architecture_spec(cfg, cfg_unified.network.architecture)
    norm_spec = extract_normalization_spec(cfg, inputs=inputs, nb_inputs=nb_inputs)

    manifest = EmulatorManifest(
        schema_version=1,
        Nz=Nz,
        inputs=list(inputs),
        nb_inputs=nb_inputs,
        nb_outputs=nb_outputs,
        output_scale=float(cfg_unified.network.output_scale),
        architecture=arch_spec,
        normalization=norm_spec,
    )

    manifest_path = artifact_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(asdict(manifest), sort_keys=False, default_flow_style=False)
    )
    weights_path = export_dir / "weights.weights.h5"
    model.save_weights(str(weights_path))

    return artifact_dir


def load_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
) -> Tuple[tf.keras.Model, EmulatorManifest]:
    """
    Strict loader:
      - reads manifest.json
      - validates cfg invariants (Nz, inputs, architecture hyperparams, normalization method, output_scale)
      - rebuilds model via Architectures[...] using cfg
      - reconstructs and attaches normalizer from manifest
      - loads weights via model.load_weights()
    """
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.yaml at {manifest_path}")

    raw = yaml.safe_load(manifest_path.read_text())
    manifest = EmulatorManifest(
        schema_version=int(raw["schema_version"]),
        Nz=int(raw["Nz"]),
        inputs=list(raw["inputs"]),
        nb_inputs=int(raw["nb_inputs"]),
        nb_outputs=int(raw["nb_outputs"]),
        output_scale=float(raw["output_scale"]),
        architecture=ArchitectureSpec(**raw["architecture"]),
        normalization=NormalizationSpec(**raw["normalization"]),
    )

    # Strict checks against run cfg
    validate_cfg_against_manifest(cfg, manifest)

    arch_name = str(manifest.architecture.name)
    if arch_name not in Architectures:
        raise ValueError(f"Unknown architecture {arch_name!r}. Available: {list(Architectures.keys())}")

    model = Architectures[arch_name](cfg, manifest.nb_inputs, manifest.nb_outputs)

    # Rebuild and attach normalizer from manifest spec
    normalizer = build_normalizer_from_spec(manifest.normalization)
    _force_build_normalizer(normalizer, nb_inputs=manifest.nb_inputs)
    model.input_normalizer = normalizer

    # Make sure model vars exist 
    model.build((None, None, None, manifest.nb_inputs))

    weights_path = artifact_dir / "export" / "weights.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file at {weights_path}")

    model.load_weights(str(weights_path))
    return model, manifest
