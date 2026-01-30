#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State
from igm.processes.vert_flow.vert_flow_v1 import compute_vertical_velocity_v1
from igm.processes.vert_flow.vert_flow_v2 import compute_vertical_velocity_v2
from igm.processes.vert_flow.vert_flow_v3 import compute_vertical_velocity_v3
from igm.processes.iceflow.utils.velocities import get_velbase_1, get_velsurf_1


def initialize(cfg: DictConfig, state: State) -> None:
    pass


def update(cfg: DictConfig, state: State) -> None:
    version = cfg.processes.vert_flow.version

    if version == 1:
        compute_vertical_velocity = compute_vertical_velocity_v1
    elif version == 2:
        compute_vertical_velocity = compute_vertical_velocity_v2
    elif version == 3:
        compute_vertical_velocity = compute_vertical_velocity_v3
    else:
        raise ValueError(f"❌ Unknown vert_flow version: <{version}>.")

    state.W = compute_vertical_velocity(cfg, state)
    state.wvelbase = get_velbase_1(state.W, state.iceflow.discr_v.V_b)
    state.wvelsurf = get_velsurf_1(state.W, state.iceflow.discr_v.V_s)


def finalize(cfg: DictConfig, state: State) -> None:
    pass
