#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function()
def compute_basal_melt_rate(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    E_s: tf.Tensor,
    h_water_till: tf.Tensor,
    q_basal: tf.Tensor,
    k_ice: tf.Tensor,
    c_ice: tf.Tensor,
    K_ratio: tf.Tensor,
    rho_water: tf.Tensor,
    L_ice: tf.Tensor,
    dz: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute basal melt rate from heat balance.

    Calculates melt rate from the imbalance between basal heat input and
    conductive heat flux into the ice. Returns zero for cold, dry beds.

    Args:
        E: Enthalpy field (J kg^-1).
        E_pmp: Pressure melting point enthalpy (J kg^-1).
        E_s: Surface enthalpy (J kg^-1).
        h_water_till: Till water layer thickness (m).
        q_basal: Total basal heat flux (W m^-2).
        k_ice: Thermal conductivity of ice (W m^-1 K^-1).
        c_ice: Specific heat capacity of ice (J kg^-1 K^-1).
        K_ratio: Diffusivity ratio for temperate ice (-).
        rho_water: Water density (kg m^-3).
        L_ice: Latent heat of fusion for ice (J kg^-1).
        dz: Basal layer thickness (m).

    Returns:
        Basal melt rate (m yr^-1).
    """
    shape_2d = E_s.shape

    COND_1 = (E[0] < E_pmp[0]) & (h_water_till <= 0.0)
    COND_2 = E[1] < E_pmp[1]

    q_ice = tf.where(
        COND_2,
        -(k_ice / c_ice) * (E[1] - E[0]) / dz,
        -K_ratio * (k_ice / c_ice) * (E[1] - E[0]) / dz,
    )

    basal_melt_rate = tf.where(
        COND_1,
        tf.zeros(shape_2d),
        (q_basal - q_ice) / (rho_water * L_ice),
    )

    spy = 31556926.0
    basal_melt_rate = basal_melt_rate * spy

    return basal_melt_rate
