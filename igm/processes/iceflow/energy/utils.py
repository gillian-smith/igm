#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import List

from . import (
    EnergyComponent,
    EnergyComponents,
    EnergyParams,
    get_energy_params_args,
)


def get_energy_components(cfg: DictConfig) -> List[EnergyComponent]:

    cfg_physics = cfg.processes.iceflow.physics

    energy_components = []
    for component in cfg_physics.energy_components:
        if component not in EnergyComponents:
            raise ValueError(f"❌ Unknown energy component: <{component}>.")

        # Get component and params class
        if component == "sliding":
            law = cfg_physics.sliding.law
            component_class = EnergyComponents[component][law]
            params_class = EnergyParams[component][law]
        else:
            component_class = EnergyComponents[component]
            params_class = EnergyParams[component]

        # Get args extractor
        get_params_args = get_energy_params_args[component]

        # Instantiate params and component classes
        params_args = get_params_args(cfg)
        params = params_class(**params_args)
        component_obj = component_class(params)

        # Add component to the list of components
        energy_components.append(component_obj)

    return energy_components
