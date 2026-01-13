#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .utils import compute_arrhenius_3d


def compute_arrhenius(cfg: DictConfig, state: State) -> None:

    cfg_physics = cfg.processes.iceflow.physics
    enhancement_factor = cfg_physics.enhancement_factor

    arrhenius = enhancement_factor * compute_arrhenius_3d(cfg, state)

    weights = state.enthalpy.vertical_discr.weights
    arrhenius_avg = tf.reduce_sum(arrhenius * weights, axis=0)

    if cfg_physics.dim_arrhenius == 2:
        state.arrhenius = arrhenius_avg
    else:
        state.arrhenius = arrhenius
        state.arrhenius_avg = arrhenius_avg
