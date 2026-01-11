#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State

from .utils import update_h_water_till, compute_N


def compute_hydro(cfg: DictConfig, state: State) -> None:
    state.N = compute_N(cfg, state)


def update_hydro(cfg: DictConfig, state: State) -> None:
    state.h_water_till = update_h_water_till(cfg, state)
    state.N = compute_N(cfg, state)
