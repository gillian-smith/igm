#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), see LICENSE

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable

from omegaconf import DictConfig

from igm.common.core import State
from .interface import InterfaceMapping
from .mapping_data_assimilation import MappingDataAssimilation, VariableSpec


class InterfaceDataAssimilation(InterfaceMapping):
    """
    Reads Hydra config:
      data_assimilation:
        variables:
          - { name: thk,        transform: identity }
          - { name: slidingco,  transform: log10 }

    and produces the kwargs for `MappingDataAssimilation`.
    """

    @staticmethod
    def _parse_specs(cfg: DictConfig) -> List[VariableSpec]:
        specs = []
        for item in cfg.processes.data_assimilation.variables:
            name = str(item["name"])
            transform = str(item.get("transform", "identity")).lower()
            if transform not in ("identity", "log10"):
                raise ValueError(f"❌ Unsupported transform '{transform}' for '{name}'.")
            specs.append(VariableSpec(name=name, transform=transform))
        return specs



    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        bcs = cfg.processes.iceflow.unified.bcs
        variables = InterfaceDataAssimilation._parse_specs(cfg)
        
        # Get the existing mapping from state (should be set up before data assimilation)
        if not hasattr(state.iceflow, 'mapping') or state.iceflow.mapping is None:
            raise ValueError(
                "❌ No base mapping found in state.iceflow.mapping. "
                "The main iceflow mapping must be initialized before data assimilation mapping."
            )
        
        base_mapping = state.iceflow.mapping
        
        return {
            "bcs": bcs,
            "base_mapping": base_mapping,
            "state": state,  # Still needed for initialization to read field values
            "variables": variables,
        }
