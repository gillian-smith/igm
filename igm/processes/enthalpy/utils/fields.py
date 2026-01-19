#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def initialize_enthalpy_fields(cfg: DictConfig, state: State) -> None:
    """
    Initialize enthalpy-related fields with default values.

    Sets up 2D and 3D fields required for the enthalpy model if they do not
    already exist in state. Initializes temperature at the pressure melting point,
    zero water content, and default values for till hydrology and friction.

    Initializes state.basal_melt_rate (m yr^-1), state.T (K), state.T_pmp (K),
    state.omega (-), state.E (J kg^-1), state.E_pmp (J kg^-1),
    state.h_water_till (m), state.basal_heat_flux (W m^-2), state.N (Pa),
    state.phi (°), and state.tauc (Pa).
    """
    cfg_thermal = cfg.processes.enthalpy.thermal
    cfg_friction = cfg.processes.enthalpy.till.friction
    cfg_hydro = cfg.processes.enthalpy.till.hydro
    Nz = cfg.processes.enthalpy.numerics.Nz
    Ny = state.thk.shape[0]
    Nx = state.thk.shape[1]
    shape_2d = (Ny, Nx)
    shape_3d = (Nz, Ny, Nx)

    if not hasattr(state, "basal_melt_rate"):
        state.basal_melt_rate = tf.zeros(shape_2d)

    if not hasattr(state, "T"):
        T_pmp_ref = cfg_thermal.T_pmp_ref
        state.T = T_pmp_ref * tf.ones(shape_3d)

    if not hasattr(state, "T_pmp"):
        T_pmp_ref = cfg_thermal.T_pmp_ref
        state.T_pmp = T_pmp_ref * tf.ones(shape_3d)

    if not hasattr(state, "omega"):
        state.omega = tf.zeros(shape_3d)

    if not hasattr(state, "E"):
        c_ice = cfg_thermal.c_ice
        T_pmp_ref = cfg_thermal.T_pmp_ref
        T_ref = cfg_thermal.T_ref
        state.E = c_ice * (T_pmp_ref - T_ref) * tf.ones(shape_3d)

    if not hasattr(state, "E_pmp"):
        state.E_pmp = tf.zeros(shape_3d)

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

    # Fallback: initialize vertical discretization for testing purposes.
    # In principle, the iceflow module should provide this.
    if not hasattr(state, "iceflow"):
        from igm.processes.iceflow.iceflow import Iceflow
        from igm.processes.iceflow.vertical import VerticalDiscrs

        state.iceflow = Iceflow()
        cfg_numerics = cfg.processes.iceflow.numerics

        vertical_basis = cfg_numerics.vert_basis.lower()
        vertical_discr = VerticalDiscrs[vertical_basis](cfg)
        state.iceflow.vertical_discr = vertical_discr

    if not hasattr(state, "arrhenius"):
        cfg_physics = cfg.processes.iceflow.physics
        arrhenius = cfg_physics.init_arrhenius

        if cfg_physics.dim_arrhenius == 2:
            state.arrhenius = arrhenius * tf.ones(shape_2d)
        else:
            vertical_discr_E = state.iceflow.vertical_discr.enthalpy
            V_E_to_U_q = vertical_discr_E.V_E_to_U_q
            state.arrhenius = arrhenius * tf.einsum(
                "ij,jkl->ikl", V_E_to_U_q, tf.ones(shape_3d)
            )
