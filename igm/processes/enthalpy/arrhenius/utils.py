#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def compute_T_pa(cfg: DictConfig, state: State) -> tf.Tensor:
    """
    Compute the pressure-adjusted temperature field.

    Adjusts the temperature field for the pressure-melting point depression
    using the Clausius-Clapeyron relation.

    Returns:
        Pressure-adjusted temperature (K).
    """
    cfg_physics = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal

    rho_ice = cfg_physics.ice_density
    g = cfg_physics.gravity_cst
    beta = cfg_thermal.beta

    depth_ice = state.enthalpy.vertical_discr.depth * state.thk[None, ...]

    return compute_T_pa_tf(state.T, beta, rho_ice, g, depth_ice)


@tf.function()
def compute_T_pa_tf(
    T: tf.Tensor,
    beta: tf.Tensor,
    rho_ice: tf.Tensor,
    g: tf.Tensor,
    depth_ice: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute pressure-adjusted temperature.

    Args:
        T: Temperature field (K).
        beta: Clausius-Clapeyron constant (K Pa^-1).
        rho_ice: Ice density (kg m^-3).
        g: Gravitational acceleration (m s^-2).
        depth_ice: Depth below ice surface (m).

    Returns:
        Pressure-adjusted temperature (K).
    """
    return T + beta * rho_ice * g * depth_ice


def compute_arrhenius_3d(cfg: DictConfig, state: State) -> tf.Tensor:
    """
    Compute the 3D Arrhenius factor field for ice viscosity.

    Uses a two-regime Arrhenius law (cold/warm ice) with water content enhancement
    to compute the rate factor throughout the ice column.

    Returns:
        3D Arrhenius factor field (MPa^-n yr^-1).
    """
    T_pa = compute_T_pa(cfg, state)

    cfg_arrhenius = cfg.processes.enthalpy.arrhenius

    T_threshold = cfg_arrhenius.T_threshold
    A_cold = cfg_arrhenius.A_cold
    A_warm = cfg_arrhenius.A_warm
    Q_cold = cfg_arrhenius.Q_cold
    Q_warm = cfg_arrhenius.Q_warm
    omega_coef = cfg_arrhenius.omega_coef
    omega_max = cfg_arrhenius.omega_max
    R = cfg_arrhenius.R

    return compute_arrhenius_3d_tf(
        state.omega,
        T_pa,
        T_threshold,
        A_cold,
        A_warm,
        Q_cold,
        Q_warm,
        omega_coef,
        omega_max,
        R,
    )


@tf.function()
def compute_arrhenius_3d_tf(
    omega: tf.Tensor,
    T_pa: tf.Tensor,
    T_threshold: tf.Tensor,
    A_cold: tf.Tensor,
    A_warm: tf.Tensor,
    Q_cold: tf.Tensor,
    Q_warm: tf.Tensor,
    omega_coef: tf.Tensor,
    omega_max: tf.Tensor,
    R: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute the 3D Arrhenius factor.

    Args:
        omega: Water content fraction (-).
        T_pa: Pressure-adjusted temperature (K).
        T_threshold: Temperature threshold between cold and warm regimes (K).
        A_cold: Pre-exponential factor for cold ice (Pa^-n s^-1).
        A_warm: Pre-exponential factor for warm ice (Pa^-n s^-1).
        Q_cold: Activation energy for cold ice (J mol^-1).
        Q_warm: Activation energy for warm ice (J mol^-1).
        omega_coef: Water content enhancement coefficient (-).
        omega_max: Maximum water content for enhancement (-).
        R: Universal gas constant (J mol^-1 K^-1).

    Returns:
        3D Arrhenius factor field (MPa^-n yr^-1).
    """
    A = tf.where(T_pa < T_threshold, A_cold, A_warm)
    Q = tf.where(T_pa < T_threshold, Q_cold, Q_warm)

    omega_factor = 1.0 + omega_coef * tf.minimum(omega, omega_max)

    spy = 31556926.0

    return omega_factor * A * tf.math.exp(-Q / (R * T_pa)) * spy * 1e18
