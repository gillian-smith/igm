#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State

from .utils import compute_fraction_drained
from ...temperature.utils import compute_pmp_tf


def update_drainage(cfg: DictConfig, state: State) -> None:

    cfg_physics = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal
    cfg_drainage = cfg.processes.enthalpy.drainage

    rho_ice = cfg_physics.ice_density
    g = cfg_physics.gravity_cst

    beta = cfg_thermal.beta
    c_ice = cfg_thermal.c_ice
    T_pmp_ref = cfg_thermal.T_pmp_ref
    T_ref = cfg_thermal.T_ref
    L_ice = cfg_thermal.L_ice

    rho_water = cfg_drainage.water_density
    drain_ice_column = cfg_drainage.drain_ice_column
    omega_target = cfg_drainage.omega_target
    omega_threshold_1 = cfg_drainage.omega_threshold_1
    omega_threshold_2 = cfg_drainage.omega_threshold_2
    omega_threshold_3 = cfg_drainage.omega_threshold_3

    if (state.dt == 0.0) or not drain_ice_column:
        return

    depth_ice = state.enthalpy.vertical_discr.depth * state.thk[None, ...]
    dzeta = state.enthalpy.vertical_discr.dzeta
    dz = dzeta * state.thk[None, ...]

    _, E_pmp = compute_pmp_tf(rho_ice, g, depth_ice, beta, c_ice, T_pmp_ref, T_ref)

    fraction_drained, h_drained = compute_fraction_drained(
        state.E,
        E_pmp,
        L_ice,
        omega_target,
        omega_threshold_1,
        omega_threshold_2,
        omega_threshold_3,
        dz,
        state.dt,
    )

    state.E -= fraction_drained * L_ice

    state.basal_melt_rate += (rho_ice / rho_water) * h_drained / state.dt
