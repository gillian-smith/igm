#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State

from .utils import compute_pmp_tf, compute_T_tf, compute_omega_tf


def compute_temperature(cfg: DictConfig, state: State) -> None:

    cfg_physics = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal

    rho_ice = cfg_physics.ice_density
    g = cfg_physics.gravity_cst

    beta = cfg_thermal.beta
    c_ice = cfg_thermal.c_ice
    L_ice = cfg_thermal.L_ice
    T_pmp_ref = cfg_thermal.T_pmp_ref
    T_ref = cfg_thermal.T_ref

    depth_ice = state.enthalpy.vertical_discr.depth * state.thk[None, ...]

    T_pmp, E_pmp = compute_pmp_tf(rho_ice, g, depth_ice, beta, c_ice, T_pmp_ref, T_ref)

    state.T = compute_T_tf(state.E, E_pmp, T_pmp, T_ref, c_ice)
    state.omega = compute_omega_tf(state.E, E_pmp, L_ice)

    state.T_b = state.T[0]
    state.T_s = state.T[-1]
