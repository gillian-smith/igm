#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple
from omegaconf import DictConfig

from igm.common import State

from ...temperature.utils import compute_pmp_tf
from .utils import compute_drainage


def update_drainage(cfg: DictConfig, state: State) -> None:

    spy = 31556926.0

    # apply drainage
    # Drainage along ice column
    cfg_thermal = cfg.processes.enthalpy.thermal
    cfg_drainage = cfg.processes.enthalpy.drainage
    cfg_physics = cfg.processes.iceflow.physics
    water_density = cfg_drainage.water_density
    drain_ice_column = cfg_drainage.drain_ice_column
    omega_target = cfg_drainage.omega_target
    omega_threshold_1 = cfg_drainage.omega_threshold_1
    omega_threshold_2 = cfg_drainage.omega_threshold_2
    omega_threshold_3 = cfg_drainage.omega_threshold_3
    ice_density = cfg_physics.ice_density
    L_ice = cfg_thermal.L_ice

    cfg_physics = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal

    rho_ice = cfg_physics.ice_density
    g = cfg_physics.gravity_cst

    beta = cfg_thermal.beta
    c_ice = cfg_thermal.c_ice
    T_pmp_ref = cfg_thermal.T_pmp_ref
    T_ref = cfg_thermal.T_ref

    depth_ice = state.enthalpy.vertical_discr.depth * state.thk[None, ...]
    dzeta = state.enthalpy.vertical_discr.dzeta
    dz = dzeta * state.thk[None, ...]

    _, E_pmp = compute_pmp_tf(rho_ice, g, depth_ice, beta, c_ice, T_pmp_ref, T_ref)

    if (state.dt > 0) and drain_ice_column:
        E, basal_melt_rate = _apply_drainage(
            E,
            E_pmp,
            state.dt,
            state.dz,
            basal_melt_rate,
            L_ice,
            ice_density,
            water_density,
            spy,
            omega_target,
            omega_threshold_1,
            omega_threshold_2,
            omega_threshold_3,
        )


@tf.function
def _apply_drainage(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    dt: tf.Tensor,
    dz: tf.Tensor,
    basal_melt_rate: tf.Tensor,
    L_ice: float,
    ice_density: float,
    water_density: float,
    spy: float,
    target_water_fraction: float,
    threshold_1: float,
    threshold_2: float,
    threshold_3: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply water drainage through ice column."""
    # Cell-centered layer thickness
    DZ = tf.concat([dz[0:1], dz[:-1] + dz[1:], dz[-1:]], axis=0) / 2.0

    # Water content
    omega = tf.maximum((E - E_pmp) / L_ice, 0.0)

    # Identify cells to drain
    CD = omega > target_water_fraction

    # Drainage fraction
    fraction_drained = (
        compute_drainage(omega, threshold_1, threshold_2, threshold_3) * dt
    )
    fraction_drained = tf.minimum(fraction_drained, omega - target_water_fraction)

    # Drained water thickness
    H_drained = tf.where(CD, fraction_drained * DZ, 0.0)  # [m]

    # Update enthalpy
    E = tf.where(CD, E - fraction_drained * L_ice, E)

    # Total drained water
    H_total_drained = tf.reduce_sum(H_drained, axis=0)  # [m]

    # Add to basal melt rate
    basal_melt_rate += (ice_density / water_density) * H_total_drained / dt

    return E, basal_melt_rate
