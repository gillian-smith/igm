#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import warnings
from omegaconf import DictConfig
from typing import Any, Dict
import numpy as np

import igm
from igm.common.core import State
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)
from .interface import InterfaceMapping
from igm.processes.iceflow.emulate.utils.networks import cnn_og, unet, StandardizationLayer, ManualNormalizationLayer
from .utils import process_inputs_scales, process_inputs_variances

class InterfaceNetwork(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:

        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics
        cfg_unified = cfg.processes.iceflow.unified

        if cfg_unified.pretrained:
            dir_path = get_pretrained_emulator_path(cfg, state)
            iceflow_model = load_model_from_path(dir_path, cfg_unified.inputs)
        else:
            warnings.warn("No pretrained emulator found. Starting from scratch.")

            nb_inputs = len(cfg_unified.inputs) + (cfg_physics.dim_arrhenius == 3) * (
                cfg_numerics.Nz - 1
            )

            nb_outputs = 2 * cfg_numerics.Nz
            
            if cfg_unified.scaling.method.lower() == "automatic":
                norm = StandardizationLayer()
            elif cfg_unified.scaling.method.lower() == "manual":
                scales = process_inputs_scales(
                    cfg_unified.scaling.manual.inputs_scales, cfg_unified.inputs
                )
                variances = process_inputs_variances(
                    cfg_unified.scaling.manual.inputs_variances, cfg_unified.inputs
                )
                norm = ManualNormalizationLayer(
                        scales=scales,
                        variances=variances
                    )
            else:
                norm = None # a bit reduundant if we make manual normalization 1 everywhere

            architecture_name = cfg_unified.network.architecture

            # Get the function from the networks module
            if hasattr(igm.processes.iceflow.emulate.utils.networks, architecture_name):
                architecture_class = getattr(
                    igm.processes.iceflow.emulate.utils.networks, architecture_name
                )

                iceflow_model = architecture_class(
                    cfg, nb_inputs, nb_outputs, input_normalizer=norm
                )

            else:
                raise ValueError(
                    f"Unknown network architecture: {architecture_name}. "
                    f"Available architectures: cnn, unet"
                )

        state.iceflow_model = iceflow_model
        state.iceflow_model.compile(jit_compile=False) # not all architectures support jit_compile=True
        
        if cfg.processes.iceflow.unified.network.print_summary:
            print(state.iceflow_model.summary())

        return {
            "bcs": cfg_unified.bcs,
            "vertical_discr": state.iceflow.vertical_discr,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
            "precision": cfg_numerics.precision,
        }
