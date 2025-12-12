#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def initialize_iceflow_fields(cfg: DictConfig, state: State) -> None:
    """Initialize iceflow fields: arrhenius, slidingco, U, V."""

    cfg_physics = cfg.processes.iceflow.physics
    Nz = cfg.processes.iceflow.numerics.Nz
    Ny = state.thk.shape[0]
    Nx = state.thk.shape[1]
    shape_2d = (Ny, Nx)
    shape_3d = (Nz, Ny, Nx)

    if not hasattr(state, "arrhenius"):
        init_value = cfg_physics.init_arrhenius * cfg_physics.enhancement_factor
        shape = shape_3d if cfg_physics.dim_arrhenius == 3 else shape_2d
        state.arrhenius = tf.ones(shape) * init_value

    if not hasattr(state, "slidingco"):
        state.slidingco = tf.ones(shape_2d) * cfg_physics.init_slidingco

    if not hasattr(state, "U"):
        state.U = tf.zeros(shape_3d)

    if not hasattr(state, "V"):
        state.V = tf.zeros(shape_3d)
