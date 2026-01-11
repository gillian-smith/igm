#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State
from .utils import solve_enthalpy_equation
from ..temperature.utils import compute_pmp_tf


def update_enthalpy(cfg: DictConfig, state: State) -> None:

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

    state.E, state.basal_melt_rate = solve_enthalpy_equation(
        cfg,
        state,
        state.E,
        E_pmp,
        state.dt,
        dz,
        state.E_s,
        state.basal_heat_flux,
        state.strain_heat,
        state.friction_heat,
        state.h_water_till,
    )
