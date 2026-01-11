#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def compute_phi(cfg: DictConfig, state: State) -> tf.Tensor:

    cfg_friction = cfg.processes.enthalpy.till.friction

    bed_min = cfg_friction.bed_min
    bed_max = cfg_friction.bed_max

    if bed_min is None or bed_max is None:
        Ny = state.thk.shape[0]
        Nx = state.thk.shape[1]
        shape_2d = (Ny, Nx)

        return cfg_friction.phi * tf.ones(shape_2d)

    phi_min = cfg_friction.phi_min
    phi_max = cfg_friction.phi_max

    return compute_phi_tf(state.topg, bed_min, bed_max, phi_min, phi_max)


@tf.function()
def compute_phi_tf(
    bed: tf.Tensor,
    bed_min: tf.Tensor,
    bed_max: tf.Tensor,
    phi_min: tf.Tensor,
    phi_max: tf.Tensor,
) -> tf.Tensor:
    phi = phi_min + (phi_max - phi_min) * (bed - bed_min) / (bed_max - bed_min)

    return tf.where(bed <= bed_min, phi_min, tf.where(bed >= bed_max, phi_max, phi))


def compute_tauc(cfg: DictConfig, state: State) -> tf.Tensor:

    N = state.N
    phi = state.phi
    h_ice = state.thk

    cfg_friction = cfg.processes.enthalpy.till.friction

    tauc_ice_free = cfg_friction.tauc_ice_free
    tauc_min = cfg_friction.tauc_min
    tauc_max = cfg_friction.tauc_max

    return compute_tauc_tf(N, phi, h_ice, tauc_ice_free, tauc_min, tauc_max)


@tf.function()
def compute_tauc_tf(
    N: tf.Tensor,
    phi: tf.Tensor,
    h_ice: tf.Tensor,
    tauc_ice_free: tf.Tensor,
    tauc_min: tf.Tensor,
    tauc_max: tf.Tensor,
) -> tf.Tensor:

    tauc = N * tf.math.tan(phi * tf.constant(np.pi) / 180.0)
    tauc = tf.where(h_ice > 0.0, tauc, tauc_ice_free)

    return tf.clip_by_value(tauc, tauc_min, tauc_max)


def compute_slidingco(cfg: DictConfig, state: State) -> tf.Tensor:

    tauc = state.tauc

    u_ref = cfg.processes.enthalpy.till.friction.u_ref
    m = cfg.processes.iceflow.physics.sliding.weertman.exponent

    return compute_slidingco_tf(tauc, u_ref, m)


@tf.function()
def compute_slidingco_tf(tauc: tf.Tensor, u_ref: tf.Tensor, m: tf.Tensor) -> tf.Tensor:

    return tauc * tf.pow(u_ref, -1.0 / m) * 1e-6
