#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State

from .utils import update_h_water_till, compute_N


def compute_hydro(cfg: DictConfig, state: State) -> None:
    """
    Compute the effective pressure for the till hydrology model.

    Calculates effective pressure based on current till water content
    without updating the water layer thickness.

    Updates state.N (Pa).
    """
    state.N = compute_N(cfg, state)


def update_hydro(cfg: DictConfig, state: State) -> None:
    """
    Update the till hydrology state over a time step.

    Evolves the till water layer thickness based on basal melt and drainage,
    then recomputes the effective pressure.

    Updates state.h_water_till (m) and state.N (Pa).
    """
    state.h_water_till = update_h_water_till(cfg, state)
    state.N = compute_N(cfg, state)
