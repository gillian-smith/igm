#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State
from .utils import compute_advection_upwind


def update_horizontal(cfg: DictConfig, state: State) -> None:

    state.E -= state.dt * compute_advection_upwind(state.U, state.V, state.E, state.dx)
