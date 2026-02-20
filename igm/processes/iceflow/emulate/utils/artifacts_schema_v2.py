#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors

"""
artifacts_schema_v2.py

Schema v2:
- strict, simple validation (raise on mismatch)
- checks a smaller set of fields
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class ArchitectureSpec:
    name: str
    params: Dict[str, Any]

@dataclass
class NormalizationSpec:
    method: str
    params: Dict[str, Any]

@dataclass
class EmulatorManifestV2:
    schema_version: int
    Nz: int
    inputs: List[str]
    nb_inputs: int
    nb_outputs: int
    output_scale: float
    architecture: ArchitectureSpec
    normalization: NormalizationSpec


def parse_manifest_v2(raw: Dict[str, Any]) -> EmulatorManifestV2:
    if int(raw.get("schema_version", -1)) != 2:
        raise ValueError(f"parse_manifest_v2 expects schema_version=2, got {raw.get('schema_version')!r}")

    arch = raw["architecture"]
    norm = raw["normalization"]

    return EmulatorManifestV2(
        schema_version=int(raw["schema_version"]),
        Nz=int(raw["Nz"]),
        inputs=list(raw["inputs"]),
        nb_inputs=int(raw["nb_inputs"]),
        nb_outputs=int(raw["nb_outputs"]),
        output_scale=float(raw["output_scale"]),
        architecture=ArchitectureSpec(name=str(arch["name"]), params=dict(arch.get("params", {}))),
        normalization=NormalizationSpec(method=str(norm["method"]), params=dict(norm.get("params", {}))),
    )


def _require_equal(label: str, run: Any, artifact: Any) -> None:
    if run != artifact:
        raise ValueError(f"{label} mismatch: cfg={run!r} vs artifact={artifact!r}")


def validate_cfg_or_raise_v2(cfg, manifest: EmulatorManifestV2) -> None:
    """
    Keep v2 validation simple + strict (as before).
    """
    cfg_unified = cfg.processes.iceflow.unified
    cfg_numerics = cfg.processes.iceflow.numerics

    _require_equal("Nz", int(cfg_numerics.Nz), int(manifest.Nz))
    _require_equal("inputs", list(cfg_unified.inputs), list(manifest.inputs))
    _require_equal("architecture.name", str(cfg_unified.network.architecture), str(manifest.architecture.name))
    _require_equal("output_scale", float(cfg_unified.network.output_scale), float(manifest.output_scale))
    _require_equal("normalization.method", "keras_normalization", str(manifest.normalization.method))