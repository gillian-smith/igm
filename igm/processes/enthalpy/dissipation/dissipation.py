#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from .utils import compute_strain_heat, compute_friction_heat


def compute_dissipation(cfg: DictConfig, state: State) -> None:

    state.strain_heat = compute_strain_heat(cfg, state)
    state.friction_heat = compute_friction_heat(cfg, state)
