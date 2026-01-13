#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .utils import compute_zeta, compute_dzeta, compute_depth, compute_weights


class VerticalDiscr:
    depth: tf.Tensor
    weights: tf.Tensor
    dzeta: tf.Tensor


def initialize_vertical_discr(cfg: DictConfig, state: State) -> None:

    cfg_numerics = cfg.processes.enthalpy.numerics
    Nz = cfg_numerics.Nz

    zeta = compute_zeta(Nz)
    dzeta = compute_dzeta(zeta)
    depth = compute_depth(dzeta)
    weights = compute_weights(dzeta)

    dzeta = dzeta[..., None, None]
    depth = depth[..., None, None]
    weights = weights[..., None, None]

    vertical_discr = VerticalDiscr()
    vertical_discr.depth = depth
    vertical_discr.weights = weights
    vertical_discr.dzeta = dzeta

    state.enthalpy.vertical_discr = vertical_discr
