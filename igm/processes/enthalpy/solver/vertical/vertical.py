#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .utils.assembly import assemble_system
from .utils.bc import compute_bc
from .utils.diffusivity import compute_diffusivity
from .utils.melt import compute_basal_melt_rate
from .utils.solver import solve_tridiagonal_system
from .utils.velocity import correct_vertical_velocity


def update_vertical(cfg: DictConfig, state: State) -> None:

    cfg_thermal = cfg.processes.enthalpy.thermal
    cfg_drainage = cfg.processes.enthalpy.drainage
    cfg_physics = cfg.processes.iceflow.physics
    cfg_solver = cfg.processes.enthalpy.solver

    rho_ice = cfg_physics.ice_density
    dz_min = cfg_physics.thr_ice_thk

    rho_water = cfg_drainage.water_density

    k_ice = cfg_thermal.k_ice
    c_ice = cfg_thermal.c_ice
    L_ice = cfg_thermal.L_ice
    T_ref = cfg_thermal.T_ref
    T_min = cfg_thermal.T_min
    K_ratio = cfg_thermal.K_ratio
    correct_w_for_melt = cfg_solver.correct_w_for_melt

    dzeta = state.enthalpy.vertical_discr.dzeta
    dz = dzeta * state.thk[None, ...]

    # Correct vertical velocity
    Wc = state.W if hasattr(state, "W") else tf.zeros_like(state.U)
    Wc = correct_vertical_velocity(Wc, state.basal_melt_rate, correct_w_for_melt)

    # Thermal diffusivity
    K = compute_diffusivity(state.E, state.E_pmp, k_ice, rho_ice, c_ice, K_ratio)

    # Source term
    f = state.strain_heat / rho_ice

    # Boundary conditions
    q_basal = state.basal_heat_flux + state.friction_heat
    dEdz_dry = -(c_ice / k_ice) * q_basal
    BCB, VB, VS = compute_bc(
        state.E, state.E_pmp, state.E_s, state.h_water_till, dEdz_dry
    )

    # Assemble system
    spy = 31556926.0
    L, M, U, R = assemble_system(
        state.E, state.dt * spy, tf.maximum(dz, dz_min), Wc / spy, K, f, BCB, VB, VS
    )

    # Solve system
    state.E = solve_tridiagonal_system(L, M, U, R)

    # Enforce bounds
    E_min = c_ice * (T_min - T_ref)
    E_max = state.E_pmp + L_ice
    state.E = tf.clip_by_value(state.E, E_min, E_max)

    # Compute basal heat flux
    state.basal_melt_rate = compute_basal_melt_rate(
        state.E,
        state.E_pmp,
        state.E_s,
        state.h_water_till,
        q_basal,
        k_ice,
        c_ice,
        K_ratio,
        rho_water,
        L_ice,
        tf.maximum(dz[0], dz_min),
    )
