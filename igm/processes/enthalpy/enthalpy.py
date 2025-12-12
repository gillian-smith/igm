#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.utils.vertical_discretization import (
    compute_levels,
    compute_dz,
    compute_depth,
)
from .utils.thermal import (
    compute_surface_temperature,
    compute_pressure_melting_point,
    temperature_from_enthalpy,
    surface_enthalpy_from_temperature,
)
from .utils.friction import compute_sliding_coefficient, compute_phi
from .utils.heat_sources import compute_strain_heating, compute_friction_heating
from .utils.rheology import compute_arrhenius_factor
from .utils.solver import solve_enthalpy_equation


def initialize(cfg: DictConfig, state: State) -> None:
    """Initialize enthalpy module state variables."""
    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required for the 'enthalpy' module.")

    cfg_enthalpy = cfg.processes.enthalpy
    Ny, Nx = state.thk.shape
    Nz = cfg.processes.iceflow.numerics.Nz

    # Initialize state variables
    state.basalMeltRate = tf.Variable(tf.zeros_like(state.thk), trainable=False)
    state.T = tf.Variable(
        tf.ones((Nz, Ny, Nx)) * cfg_enthalpy.melt_temp,
        trainable=False,
    )
    state.omega = tf.Variable(tf.zeros_like(state.T), trainable=False)
    state.E = tf.Variable(
        tf.ones_like(state.T)
        * cfg_enthalpy.ci
        * (cfg_enthalpy.melt_temp - cfg_enthalpy.ref_temp),
        trainable=False,
    )
    state.tillwat = tf.Variable(tf.zeros_like(state.thk), trainable=False)

    if not hasattr(state, "bheatflx"):
        state.bheatflx = tf.Variable(
            tf.ones_like(state.thk) * cfg_enthalpy.default_bheatflx,
            trainable=False,
        )

    # Initialize friction angle and sliding coefficient
    state.phi = compute_phi(cfg, state)
    state.tauc, state.slidingco = compute_sliding_coefficient(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    """Update enthalpy and related thermal fields."""
    if hasattr(state, "logger"):
        state.logger.info(f"Update ENTHALPY at time: {state.t.numpy()}")

    cfg_enthalpy = cfg.processes.enthalpy
    cfg_physics = cfg.processes.iceflow.physics
    cfg_numerics = cfg.processes.iceflow.numerics

    # Compute surface temperature
    surftemp = compute_surface_temperature(
        state.air_temp,
        cfg_enthalpy.temperature_offset_air_to_ice,
        cfg_enthalpy.melt_temp,
    )

    # Get vertical discretization
    levels = compute_levels(cfg_numerics.Nz, cfg_numerics.vert_spacing)
    dz = compute_dz(state.thk, levels)
    depth = compute_depth(dz)

    # Compute pressure melting point
    Tpmp, Epmp = compute_pressure_melting_point(
        depth,
        cfg_physics.gravity_cst,
        cfg_physics.ice_density,
        cfg_enthalpy.claus_clape,
        cfg_enthalpy.melt_temp,
        cfg_enthalpy.ci,
        cfg_enthalpy.ref_temp,
    )

    # Convert enthalpy to temperature
    state.T, state.omega = temperature_from_enthalpy(
        state.E,
        Tpmp,
        Epmp,
        cfg_enthalpy.ci,
        cfg_enthalpy.ref_temp,
        cfg_enthalpy.Lh,
    )

    # Compute pressure-adjusted temperature
    state.Tpa = (
        state.T
        + cfg_enthalpy.claus_clape
        * cfg_physics.ice_density
        * cfg_physics.gravity_cst
        * depth
    )
    state.temppabase = state.Tpa[0]
    state.temppasurf = state.Tpa[-1]

    # Compute arrhenius factor
    state.arrhenius = compute_arrhenius_factor(
        cfg, state.Tpa, state.omega, state.vert_weight
    )

    # Compute corrected vertical velocity
    Wc = _compute_corrected_vertical_velocity(state)

    # Compute heat sources
    state.strainheat = compute_strain_heating(
        cfg, state.U, state.V, state.arrhenius, state.dx, dz
    )
    state.frictheat = compute_friction_heating(
        cfg, state.U, state.V, state.slidingco, state.topg, state.dX
    )

    # Compute surface enthalpy
    surfenth = surface_enthalpy_from_temperature(
        surftemp,
        cfg_enthalpy.melt_temp,
        cfg_enthalpy.ci,
        cfg_enthalpy.ref_temp,
    )

    # Solve enthalpy equation
    state.E, state.basalMeltRate = solve_enthalpy_equation(
        cfg,
        state,
        state.E,
        Epmp,
        state.dt,
        dz,
        Wc,
        surfenth,
        state.bheatflx,
        state.strainheat,
        state.frictheat,
        state.tillwat,
    )

    state.basalMeltRate = tf.clip_by_value(state.basalMeltRate, 0.0, 1e10)

    # Update till water content
    state.tillwat = _update_till_water(cfg, state)

    # Update sliding coefficient
    state.phi = compute_phi(cfg, state)
    state.tauc, state.slidingco = compute_sliding_coefficient(cfg, state)

    # Compute vertically averaged hardness
    state.hardav = _compute_hardness(cfg, state)
    if cfg_physics.dim_arrhenius == 3:
        state.arrheniusav = tf.reduce_sum(state.arrhenius * state.vert_weight, axis=0)


def finalize(cfg: DictConfig, state: State) -> None:
    """Finalize enthalpy module (currently no cleanup needed)."""
    pass


def _compute_corrected_vertical_velocity(state: State) -> tf.Tensor:
    """Compute vertical velocity corrected for melting rate."""
    if hasattr(state, "W"):
        return state.W - tf.expand_dims(state.basalMeltRate, axis=0)
    else:
        return tf.zeros_like(state.U) - tf.expand_dims(state.basalMeltRate, axis=0)


def _update_till_water(cfg: DictConfig, state: State) -> tf.Tensor:
    """Update till water content based on basal melt and drainage."""
    cfg_enthalpy = cfg.processes.enthalpy

    tillwat = state.tillwat + state.dt * (state.basalMeltRate - cfg_enthalpy.drain_rate)
    tillwat = tf.clip_by_value(tillwat, 0.0, cfg_enthalpy.till_wat_max)
    return tf.where(state.thk > 0, tillwat, 0.0)


def _compute_hardness(cfg: DictConfig, state: State) -> tf.Tensor:
    """Compute vertically averaged ice hardness."""
    cfg_physics = cfg.processes.iceflow.physics
    unit_conversion = 1e6 * (365.25 * 24 * 3600) ** (1 / 3)

    if cfg_physics.dim_arrhenius == 2:
        return state.arrhenius ** (-1 / 3) * unit_conversion
    else:
        return (
            tf.reduce_sum(state.arrhenius ** (-1 / 3) * state.vert_weight, axis=0)
            * unit_conversion
        )
