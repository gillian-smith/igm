#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Tuple
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common.core import State


def compute_sliding_coefficient(
    cfg: DictConfig,
    state: State,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute basal yield stress and sliding coefficient.

    Args:
        cfg: Configuration object
        state: State object containing thk, tillwat, phi

    Returns:
        Tuple of (tauc [Pa], slidingco [MPa·m⁻¹/ⁿ·y¹/ⁿ])
    """
    cfg_enthalpy = cfg.processes.enthalpy
    cfg_physics = cfg.processes.iceflow.physics

    return compute_slidingco_tf(
        state.thk,
        state.tillwat,
        cfg_physics.ice_density,
        cfg_physics.gravity_cst,
        cfg_enthalpy.till_wat_max,
        state.phi,
        cfg_physics.sliding.weertman.exponent,
        cfg_enthalpy.uthreshold,
        cfg_enthalpy.tauc_min,
        cfg_enthalpy.tauc_max,
        cfg_enthalpy.till_void_ratio,
        cfg_enthalpy.till_compressibility,
        cfg_enthalpy.till_delta,
        cfg_enthalpy.till_reference_pressure,
    )


@tf.function
def compute_slidingco_tf(
    thk: tf.Tensor,
    tillwat: tf.Tensor,
    ice_density: float,
    gravity_cst: float,
    tillwat_max: float,
    phi: tf.Tensor,
    exp_weertman: float,
    uthreshold: float,
    tauc_min: float,
    tauc_max: float,
    void_ratio: float,
    compressibility: float,
    delta: float,
    reference_pressure: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute sliding coefficient using till mechanics.

    Following the approach of Bueler and Brown (2009) and Tulaczyk et al. (2000).

    Args:
        thk: Ice thickness [m]
        tillwat: Till water content [m]
        ice_density: Ice density [kg/m³]
        gravity_cst: Gravitational constant [m/s²]
        tillwat_max: Maximum till water content [m]
        phi: Till friction angle [degrees]
        exp_weertman: Weertman sliding law exponent
        uthreshold: Threshold velocity [m/y]
        tauc_min: Minimum yield stress [Pa]
        tauc_max: Maximum yield stress [Pa]
        void_ratio: Void ratio at reference
        compressibility: Till compressibility coefficient
        delta: Transition parameter
        reference_pressure: Reference effective pressure [Pa]

    Returns:
        Tuple of (tauc [Pa], slidingco [MPa·m⁻¹/ⁿ·y¹/ⁿ])
    """
    # Till water saturation fraction
    s = tillwat / tillwat_max

    # Ice overburden pressure
    P = ice_density * gravity_cst * thk  # [Pa]

    # Effective pressure following Bueler and Brown (2009)
    effpress = tf.minimum(
        P,
        reference_pressure
        * ((delta * P / reference_pressure) ** s)
        * 10 ** (void_ratio * (1 - s) / compressibility),
    )  # [Pa]

    # Basal yield stress
    tauc = effpress * tf.math.tan(phi * np.pi / 180)  # [Pa]

    # Set high value where ice-free
    tauc = tf.where(thk > 0, tauc, 1e6)

    # Clip to min/max bounds
    tauc = tf.clip_by_value(tauc, tauc_min, tauc_max)

    # Sliding coefficient
    slidingco = (tauc * 1e-6) * uthreshold ** (-1.0 / exp_weertman)

    return tauc, slidingco


def compute_phi(cfg: DictConfig, state: State) -> tf.Tensor:
    """
    Compute spatially variable till friction angle.

    Args:
        cfg: Configuration object
        state: State object containing topg

    Returns:
        Till friction angle [degrees]
    """
    cfg_enthalpy = cfg.processes.enthalpy
    bed_min = cfg_enthalpy.till_friction_angle_bed_min

    if bed_min is None:
        # Uniform friction angle
        return cfg_enthalpy.till_friction_angle * tf.ones_like(state.thk)

    # Spatially variable friction angle based on bed elevation
    bed_max = cfg_enthalpy.till_friction_angle_bed_max
    phi_min = cfg_enthalpy.till_friction_angle_phi_min
    phi_max = cfg_enthalpy.till_friction_angle_phi_max

    return tf.where(
        state.topg <= bed_min,
        phi_min,
        tf.where(
            state.topg >= bed_max,
            phi_max,
            phi_min
            + (phi_max - phi_min) * (state.topg - bed_min) / (bed_max - bed_min),
        ),
    )
