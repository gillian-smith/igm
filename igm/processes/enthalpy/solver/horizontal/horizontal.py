#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State
from .utils import compute_advection_upwind


def update_horizontal(cfg: DictConfig, state: State) -> None:
    """
    Update enthalpy field for horizontal advection over a time step.

    Applies an upwind advection scheme to transport enthalpy horizontally
    with the ice velocity field.

    Updates state.E (J kg^-1).
    """
    V_U_to_E = state.iceflow.vertical_discr.enthalpy.V_U_to_E
    state.E -= state.dt * compute_advection_upwind(
        state.U, state.V, V_U_to_E, state.E, state.dx
    )
