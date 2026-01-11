#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State

from .arrhenius import compute_arrhenius
from .dissipation import compute_dissipation
from .solver import update_enthalpy
from .surface import compute_surface
from .temperature import compute_temperature
from .till.hydro import compute_hydro, update_hydro
from .till.friction import compute_friction
from .utils import checks, initialize_enthalpy_fields
from .vertical import initialize_vertical_discr


def initialize(cfg: DictConfig, state: State) -> None:
    """Initialize enthalpy module state variables."""

    # Do preliminary checks
    checks(cfg, state)

    # Initialize enthalpy fields
    initialize_enthalpy_fields(cfg, state)

    # Initialize vertical discretization
    initialize_vertical_discr(cfg, state)

    # Compute (T, omega) from E
    compute_temperature(cfg, state)

    # Compute A = A(T. omega)
    compute_arrhenius(cfg, state)

    # Compute N
    compute_hydro(cfg, state)

    # Compute phi, tauc, slidingco
    compute_friction(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    """Update enthalpy and related thermal fields."""
    if hasattr(state, "logger"):
        state.logger.info(f"Update ENTHALPY at time: {state.t.numpy()}")

    # (i) SOURCE & BOUNDARY TERMS

    # Compute T_s, E_s
    compute_surface(cfg, state)

    # Compute strain_heat, frictional_heat
    compute_dissipation(cfg, state)

    # (ii) SOLVE FOR ENTHALPY

    # Update E and basal_melt_rate
    update_enthalpy(cfg, state)

    # (iii) DERIVE QUANTITIES

    # Compute (T, omega) from E
    compute_temperature(cfg, state)

    # Compute A = A(T. omega)
    compute_arrhenius(cfg, state)

    # Update h_w and compute N
    update_hydro(cfg, state)

    # Compute phi, tauc, slidingco
    compute_friction(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    pass
