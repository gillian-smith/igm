#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .utils import compute_arrhenius_3d


def compute_arrhenius(cfg: DictConfig, state: State) -> None:
    """
    Compute the Arrhenius factor for ice flow, accounting for temperature and water content.

    Calculates the 3D Arrhenius factor scaled by an enhancement factor, then computes
    a vertically-averaged value. Depending on configuration, stores either only the
    averaged value (2D) or both the full 3D field and its vertical average.

    Updates state.arrhenius (MPa^-n yr^-1) and optionally state.arrhenius_avg (MPa^-n yr^-1).
    """
    cfg_physics = cfg.processes.iceflow.physics
    enhancement_factor = cfg_physics.enhancement_factor

    weights = state.iceflow.vertical_discr.enthalpy.weights
    V_E_to_U_q = state.iceflow.vertical_discr.enthalpy.V_E_to_U_q

    arrhenius = enhancement_factor * compute_arrhenius_3d(cfg, state)

    arrhenius_avg = tf.reduce_sum(arrhenius * weights, axis=0)

    if cfg_physics.dim_arrhenius == 2:
        state.arrhenius = arrhenius_avg
    else:
        state.arrhenius = tf.einsum("ij,jkl->ikl", V_E_to_U_q, arrhenius)
        state.arrhenius_avg = arrhenius_avg
