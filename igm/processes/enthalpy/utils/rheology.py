#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Optional
import tensorflow as tf
from omegaconf import DictConfig


@tf.function
def arrhenius_from_temperature(
    Tpa: tf.Tensor,
    omega: tf.Tensor,
    temp_threshold: float,
    A_cold: float,
    A_warm: float,
    Q_cold: float,
    Q_warm: float,
    gas_constant: float,
    omega_enhancement: float,
    max_omega: float,
    spy: float,
) -> tf.Tensor:
    """
    Compute Arrhenius factor from pressure-adjusted temperature and water content.

    Uses the Budd-Paterson law adapted for (T, omega) following
    Aschwanden et al. (JOG, 2012) and Paterson (1994).

    Args:
        Tpa: Pressure-adjusted temperature [K]
        omega: Water content [dimensionless]
        temp_threshold: Temperature threshold for cold/warm regimes [K]
        A_cold: Pre-exponential factor for cold ice [s⁻¹ Pa⁻³]
        A_warm: Pre-exponential factor for warm ice [s⁻¹ Pa⁻³]
        Q_cold: Activation energy for cold ice [J/mol]
        Q_warm: Activation energy for warm ice [J/mol]
        gas_constant: Universal gas constant [J/(mol·K)]
        omega_enhancement: Water content enhancement factor
        max_omega: Maximum omega for enhancement
        spy: Seconds per year [s/y]

    Returns:
        Arrhenius factor [MPa⁻³ y⁻¹]
    """
    # Temperature-dependent parameters
    A = tf.where(Tpa < temp_threshold, A_cold, A_warm)
    Q = tf.where(Tpa < temp_threshold, Q_cold, Q_warm)

    # Water content enhancement (limited to specified water fraction)
    water_enhancement = 1.0 + omega_enhancement * tf.minimum(omega, max_omega)

    # Unit conversion: Pa⁻³ s⁻¹ to MPa⁻³ y⁻¹
    unit_conversion = 1e18 * spy

    return (
        water_enhancement * A * unit_conversion * tf.math.exp(-Q / (gas_constant * Tpa))
    )


def compute_arrhenius_factor(
    cfg: DictConfig,
    Tpa: tf.Tensor,
    omega: tf.Tensor,
    vert_weight: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Compute arrhenius factor with optional vertical averaging.

    Args:
        cfg: Configuration object
        Tpa: Pressure-adjusted temperature [K]
        omega: Water content [dimensionless]
        vert_weight: Vertical weights for averaging (if dim_arrhenius == 2)

    Returns:
        Arrhenius factor [MPa⁻³ y⁻¹], either 3D or vertically averaged
    """
    cfg_enthalpy = cfg.processes.enthalpy
    cfg_physics = cfg.processes.iceflow.physics

    arrhenius_3d = arrhenius_from_temperature(
        Tpa,
        omega,
        cfg_enthalpy.arrhenius_temp_threshold,
        cfg_enthalpy.arrhenius_A_cold,
        cfg_enthalpy.arrhenius_A_warm,
        cfg_enthalpy.arrhenius_Q_cold,
        cfg_enthalpy.arrhenius_Q_warm,
        cfg_enthalpy.gas_constant,
        cfg_enthalpy.omega_enhancement_factor,
        cfg_enthalpy.max_omega_enhancement,
        cfg_enthalpy.spy,
    )

    enhancement = cfg_physics.enhancement_factor

    if cfg_physics.dim_arrhenius == 2:
        if vert_weight is None:
            raise ValueError("vert_weight required for 2D arrhenius")
        return tf.reduce_sum(arrhenius_3d * enhancement * vert_weight, axis=0)
    else:
        return arrhenius_3d * enhancement
