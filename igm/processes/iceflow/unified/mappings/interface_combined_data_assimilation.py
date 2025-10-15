#!/usr/bin/env python3
# Copyright ...
from __future__ import annotations

from typing import Any, Dict, List, Optional

import tensorflow as tf
from omegaconf import DictConfig

from igm.common.core import State
from .interface import InterfaceMapping
from .mapping_network import MappingNetwork
from .interface_data_assimilation import InterfaceDataAssimilation
from .mapping_combined_data_assimilation import CombinedVariableSpec


class InterfaceCombinedDataAssimilation(InterfaceMapping):

    @staticmethod
    def _to_combined_specs(cfg: DictConfig) -> List[CombinedVariableSpec]:
        # Reuse your DA spec parser to keep YAML compatibility, then convert items
        parsed = InterfaceDataAssimilation._parse_specs(cfg)  # returns VariableSpec-like objects
        specs: List[CombinedVariableSpec] = []
        for item in parsed:
            specs.append(
                CombinedVariableSpec(
                    name=getattr(item, "name"),
                    transform=str(getattr(item, "transform", "identity")).lower(),
                    lower_bound=getattr(item, "lower_bound", None),
                    upper_bound=getattr(item, "upper_bound", None),
                    mask=getattr(item, "mask", None),  # may be absent; handled as None
                )
            )
        return specs

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        # Pull network pieces from the existing MappingNetwork
        if not hasattr(state.iceflow, "mapping") or state.iceflow.mapping is None:
            raise ValueError("❌ state.iceflow.mapping is not set. Initialize MappingNetwork first.")
        base_map = state.iceflow.mapping
        if not isinstance(base_map, MappingNetwork):
            raise TypeError("❌ Combined DA expects the current mapping to be a MappingNetwork.")

        bcs = cfg.processes.iceflow.unified.bcs
        specs = InterfaceCombinedDataAssimilation._to_combined_specs(cfg)

        emu_cfg = cfg.processes.iceflow.emulator
        fieldin = getattr(emu_cfg, "fieldin", None)
        field_to_channel = {str(name): i for i, name in enumerate(fieldin)}

        if field_to_channel is not None:
            missing = [s.name for s in specs if s.name not in field_to_channel]
            if missing:
                raise ValueError(
                    "❌ The following DA variables are not in iceflow.emulator.fieldin "
                    f"(or field_to_channel): {missing}. "
                    "Please include them in cfg.iceflow.emulator.fieldin"
                )

        da_cfg = getattr(getattr(cfg, "processes", object()), "data_assimilation", object())
        precision = getattr(da_cfg, "precision", "single")

        return {
            "bcs": bcs,
            "network": base_map.network,
            "Nz": base_map.Nz,
            "output_scale": base_map.output_scale,
            "state": state,
            "variables": specs,
            "field_to_channel": field_to_channel,
            "precision": precision,
        }
