#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict, Tuple

from igm.processes.iceflow.emulate.utils.misc import get_effective_pressure_precentage
from igm.processes.iceflow.energy.utils import stag4h
from igm.processes.iceflow.utils.velocities import get_velbase
from igm.utils.gradient.compute_gradient import compute_gradient
from igm.processes.iceflow.vertical import VerticalDiscr

from ..sliding import SlidingComponent


class BuddParams(tf.experimental.ExtensionType):

    regu: float
    exponent: float
    vert_basis: str


class Budd(SlidingComponent):

    def __init__(self, params: BuddParams):
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict,
        vert_disc: VerticalDiscr,
        staggered_grid: bool,
    ) -> tf.Tensor:
        return cost_budd(U, V, fieldin, vert_disc, staggered_grid, self.params)


def cost_budd(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict,
    vert_disc: VerticalDiscr,
    staggered_grid: bool,
    budd_params: BuddParams,
) -> tf.Tensor:

    thk, usurf, slidingco, dX = (
        fieldin["thk"],
        fieldin["usurf"],
        fieldin["slidingco"],
        fieldin["dX"],
    )
    V_b = vert_disc.V_b

    expo = budd_params.exponent
    regu = budd_params.regu

    return _cost(
        U,
        V,
        thk,
        usurf,
        slidingco,
        dX,
        expo,
        regu,
        V_b,
        staggered_grid,
    )


@tf.function()
def _cost(
    U,
    V,
    thk,
    usurf,
    slidingco,
    dX,
    expo,
    regu,
    V_b,
    staggered_grid,
):
    # Temporary fix for effective pressure - should be within the inputs
    N = get_effective_pressure_precentage(thk, percentage=0.0)
    N = tf.where(N < 1e-3, 1e-3, N)

    # Coefficient and effective exponent
    C = 1.0 * slidingco
    s = 1.0 + 1.0 / expo

    # Bed gradients
    dbdx, dbdy = compute_gradient(usurf - thk, dX, dX, staggered_grid)

    # Optional staggering
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        C = stag4h(C)

    # Retrieve basal velocity
    ux_b = tf.einsum("j,bjkl->bkl", V_b, U)
    uy_b = tf.einsum("j,bjkl->bkl", V_b, V)

    # Compute basal velocity magnitude (with norm M and regularization)
    corr_bed = ux_b * dbdx + uy_b * dbdy

    u_b = tf.sqrt(ux_b * ux_b + uy_b * uy_b + regu * regu + corr_bed * corr_bed)

    return C * N * tf.pow(u_b, s) / s
