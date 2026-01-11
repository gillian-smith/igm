#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


class Enthalpy:
    pass


def initialize_enthalpy_fields(cfg: DictConfig, state: State) -> None:
    """Initialize enthalpy fields."""

    cfg_thermal = cfg.processes.enthalpy.thermal
    cfg_friction = cfg.processes.enthalpy.till.friction
    cfg_hydro = cfg.processes.enthalpy.till.hydro
    Nz = cfg.processes.enthalpy.numerics.Nz
    Ny = state.thk.shape[0]
    Nx = state.thk.shape[1]
    shape_2d = (Ny, Nx)
    shape_3d = (Nz, Ny, Nx)

    if not hasattr(state, "enthalpy"):
        state.enthalpy = Enthalpy()

    if not hasattr(state, "basal_melt_rate"):
        state.basal_melt_rate = tf.zeros(shape_2d)

    if not hasattr(state, "T"):
        T_pmp_ref = cfg_thermal.T_pmp_ref
        state.T = T_pmp_ref * tf.ones(shape_3d)

    if not hasattr(state, "omega"):
        state.omega = tf.zeros(shape_3d)

    if not hasattr(state, "E"):
        c_ice = cfg_thermal.c_ice
        T_pmp_ref = cfg_thermal.T_pmp_ref
        T_ref = cfg_thermal.T_ref
        state.E = c_ice * (T_pmp_ref - T_ref) * tf.ones(shape_3d)

    if not hasattr(state, "h_water_till"):
        state.h_water_till = tf.zeros(shape_2d)

    if not hasattr(state, "basal_heat_flux"):
        basal_heat_flux_ref = cfg_thermal.basal_heat_flux_ref
        state.basal_heat_flux = basal_heat_flux_ref * tf.ones(shape_2d)

    if not hasattr(state, "N"):
        N_ref = cfg_hydro.N_ref
        state.N = N_ref * tf.ones(shape_2d)

    if not hasattr(state, "phi"):
        phi = cfg_friction.phi
        state.phi = phi * tf.ones(shape_2d)

    if not hasattr(state, "tauc"):
        tauc = cfg_friction.tauc_ice_free
        state.tauc = tauc * tf.ones(shape_2d)
