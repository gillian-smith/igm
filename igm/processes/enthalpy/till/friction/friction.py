#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State

from .utils import compute_phi, compute_tauc, compute_slidingco


def compute_friction(cfg: DictConfig, state: State) -> None:
    state.phi = compute_phi(cfg, state)
    state.tauc = compute_tauc(cfg, state)
    state.slidingco = compute_slidingco(cfg, state)
