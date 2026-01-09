#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import warnings
from omegaconf import DictConfig
from typing import Any, Dict


import igm
from igm.common import State
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)
from .interface import InterfaceMapping
from igm.processes.iceflow.emulate import Architectures, NormalizationsDict


class InterfaceNetwork(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:

        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics
        cfg_unified = cfg.processes.iceflow.unified

        # ----- Build/load model -----
        if cfg_unified.network.pretrained:
            dir_path = get_pretrained_emulator_path(cfg, state)
            iceflow_model = load_model_from_path(dir_path, cfg_unified.inputs)

            # Prefer self-contained pretrained artifacts:
            # the loaded model should already include input_normalizer.
            if not hasattr(iceflow_model, "input_normalizer"):
                iceflow_model.input_normalizer = None

            if cfg_unified.normalization.method != "none" and iceflow_model.input_normalizer is None:
                raise ValueError("Pretrained model has no input_normalizer attached. ")
        else:
            # ----- Build normalizer from cfg and attach to model -----
            normalizing_method = cfg_unified.normalization.method
            normalizing_class = NormalizationsDict[normalizing_method]

            if normalizing_method == "adaptive":
                nb_channels = len(cfg_unified.inputs) + (cfg_physics.dim_arrhenius == 3) * (cfg_numerics.Nz - 1)
                normalizing_layer = normalizing_class(nb_channels)
            elif normalizing_method == "fixed":
                offsets = cfg_unified.normalization.fixed.inputs_offsets
                variances = cfg_unified.normalization.fixed.inputs_variances
                normalizing_layer = normalizing_class(offsets, variances)
            elif normalizing_method in ("automatic", "none"):
                normalizing_layer = normalizing_class()
            else:
                raise ValueError(f"Unknown normalizing method: {normalizing_method}")
            nb_inputs = len(cfg_unified.inputs) + (cfg_physics.dim_arrhenius == 3) * (cfg_numerics.Nz - 1)
            nb_outputs = 2 * cfg_numerics.Nz

            architecture_name = cfg_unified.network.architecture
            if architecture_name not in Architectures:
                raise ValueError(f"Unknown network architecture: {architecture_name}")

            architecture_class = Architectures[architecture_name]
            iceflow_model = architecture_class(cfg, nb_inputs, nb_outputs)

            iceflow_model.input_normalizer = normalizing_layer

        state.iceflow_model = iceflow_model
        state.iceflow_model.compile(
            jit_compile=False
        )  # not all architectures support jit_compile=True

        bcs = init_bcs(cfg, state, cfg.processes.iceflow.unified.bcs)

        return {
            "bcs": bcs,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
            "precision": cfg_numerics.precision,
        }
