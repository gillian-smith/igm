#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Optional

import tensorflow as tf
from omegaconf import DictConfig

from igm.common.core import State
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from .interface import InterfaceMapping
from ..network import MappingNetwork
from ..combined_data_assimilation import CombinedVariableSpec


class InterfaceCombinedDataAssimilation(InterfaceMapping):
    """
    Standalone interface for the Combined Data Assimilation mapping.
    Expects:
      cfg.processes.data_assimilation.variables: list of dicts with keys:
        - name: str
        - transform: str (default 'identity')
        - lower_bound / upper_bound (float, optional) OR bounds: [low, up]
        - mask: (optional; path or tensor ref — mapping will resolve/ignore as needed)
      cfg.processes.data_assimilation.store_freq: int (default 0)
      cfg.processes.data_assimilation.precision: 'single' | 'double' (default 'single')
      cfg.processes.iceflow.emulator.fieldin: list[str] used to build field_to_channel
    """

    @staticmethod
    def _parse_specs(cfg: DictConfig) -> List[CombinedVariableSpec]:
        da_cfg = getattr(
            getattr(cfg, "processes", object()), "data_assimilation", object()
        )
        variables = getattr(da_cfg, "variables", None)
        if variables is None:
            raise ValueError(
                "❌ cfg.processes.data_assimilation.variables is required."
            )

        specs: List[CombinedVariableSpec] = []
        for i, v in enumerate(variables):
            if "name" not in v:
                raise ValueError(
                    f"❌ Variable #{i} in data_assimilation.variables is missing 'name'."
                )

            # Allow either separate lower/upper or a single 'bounds: [low, up]'
            lower = v.get("lower_bound", None)
            upper = v.get("upper_bound", None)
            if "bounds" in v and (lower is None and upper is None):
                b = v["bounds"]
                if not (isinstance(b, (list, tuple)) and len(b) == 2):
                    raise ValueError(
                        f"❌ 'bounds' for variable '{v['name']}' must be [lower, upper]."
                    )
                lower, upper = b[0], b[1]

            specs.append(
                CombinedVariableSpec(
                    name=str(v["name"]),
                    transform=str(v.get("transform", "identity")).lower(),
                    lower_bound=lower,
                    upper_bound=upper,
                    mask=v.get("mask", None),
                )
            )
        return specs

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        # Ensure we have a MappingNetwork already constructed on state
        if not hasattr(state.iceflow, "mapping") or state.iceflow.mapping is None:
            raise ValueError(
                "❌ state.iceflow.mapping is not set. Initialize MappingNetwork first."
            )
        base_map = state.iceflow.mapping
        if not isinstance(base_map, MappingNetwork):
            raise TypeError(
                "❌ Combined DA expects the current mapping to be a MappingNetwork."
            )

        # Parse DA specs locally (no dependency on InterfaceDataAssimilation)
        specs = InterfaceCombinedDataAssimilation._parse_specs(cfg)

        # BCs for the physical forward model
        bcs = cfg.processes.iceflow.unified.bcs

        # Build field_to_channel from emulator.fieldin (keeps your existing contract)
        emu_cfg = cfg.processes.iceflow.emulator
        fieldin = getattr(emu_cfg, "fieldin", None)
        if not fieldin:
            raise ValueError(
                "❌ cfg.processes.iceflow.emulator.fieldin must list emulator inputs."
            )
        field_to_channel = {str(name): i for i, name in enumerate(fieldin)}

        # Validate that all DA variables appear in the emulator inputs
        missing = [s.name for s in specs if s.name not in field_to_channel]
        if missing:
            raise ValueError(
                "❌ The following DA variables are not in cfg.processes.iceflow.emulator.fieldin: "
                f"{missing}. Please include them."
            )

        # Precision + store_freq live under processes.data_assimilation
        da_cfg = cfg.processes.data_assimilation
        precision = cfg.processes.iceflow.numerics.precision
        store_freq = int(getattr(da_cfg, "store_freq", 0))

        bcs = init_bcs(cfg, state, cfg.processes.iceflow.unified.bcs)

        return {
            "bcs": bcs,
            "network": base_map.network,
            "Nz": base_map.Nz,
            "output_scale": base_map.output_scale,
            "state": state,
            "variables": specs,
            "field_to_channel": field_to_channel,
            "precision": precision,
            "store_freq": store_freq,
        }
