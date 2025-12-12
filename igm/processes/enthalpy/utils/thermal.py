#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Tuple
import tensorflow as tf


@tf.function
def compute_surface_temperature(
    air_temp: tf.Tensor,
    temperature_offset: float,
    melt_temp: float,
) -> tf.Tensor:
    """
    Compute surface temperature from air temperature.

    Args:
        air_temp: Air temperature tensor [K]
        temperature_offset: Offset to convert air temp to ice temp [K]
        melt_temp: Melting temperature [K]

    Returns:
        Surface temperature [K]
    """
    return (
        tf.minimum(
            tf.math.reduce_mean(air_temp + temperature_offset, axis=0),
            0.0,
        )
        + melt_temp
    )


@tf.function
def compute_pressure_melting_point(
    depth: tf.Tensor,
    gravity_cst: float,
    ice_density: float,
    claus_clape: float,
    melt_temp: float,
    ci: float,
    ref_temp: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute pressure melting point temperature and enthalpy.

    Args:
        depth: Ice depth [m]
        gravity_cst: Gravitational constant [m/s²]
        ice_density: Ice density [kg/m³]
        claus_clape: Clausius-Clapeyron constant [K/Pa]
        melt_temp: Melting temperature at surface [K]
        ci: Specific heat capacity of ice [J/(kg·K)]
        ref_temp: Reference temperature [K]

    Returns:
        Tuple of (Tpmp [K], Epmp [J/kg])
    """
    pressure = ice_density * gravity_cst * depth  # [Pa]
    Tpmp = melt_temp - claus_clape * pressure  # [K]
    Epmp = ci * (Tpmp - ref_temp)  # [J/kg]
    return Tpmp, Epmp


@tf.function
def temperature_from_enthalpy(
    E: tf.Tensor,
    Tpmp: tf.Tensor,
    Epmp: tf.Tensor,
    ci: float,
    ref_temp: float,
    Lh: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convert enthalpy to temperature and water content.

    Args:
        E: Enthalpy [J/kg]
        Tpmp: Pressure melting point temperature [K]
        Epmp: Pressure melting point enthalpy [J/kg]
        ci: Specific heat capacity of ice [J/(kg·K)]
        ref_temp: Reference temperature [K]
        Lh: Latent heat of fusion [J/kg]

    Returns:
        Tuple of (T [K], omega [dimensionless water fraction])
    """
    T = tf.where(E >= Epmp, Tpmp, E / ci + ref_temp)
    omega = tf.where(E >= Epmp, (E - Epmp) / Lh, 0.0)
    return T, omega


@tf.function
def surface_enthalpy_from_temperature(
    T: tf.Tensor,
    melt_temp: float,
    ci: float,
    ref_temp: float,
) -> tf.Tensor:
    """
    Compute surface enthalpy from temperature.

    Args:
        T: Temperature [K]
        melt_temp: Melting temperature [K]
        ci: Specific heat capacity of ice [J/(kg·K)]
        ref_temp: Reference temperature [K]

    Returns:
        Enthalpy [J/kg]
    """
    return tf.where(
        T < melt_temp,
        ci * (T - ref_temp),
        ci * (melt_temp - ref_temp),
    )
