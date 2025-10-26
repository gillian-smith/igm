#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict

from igm.processes.iceflow.vertical import VerticalDiscr
from igm.utils.grad.grad import grad_xy
from igm.utils.stag.stag import stag4h
from ..sliding import SlidingComponent


class WeertmanParams(tf.experimental.ExtensionType):
    """Parameters for Weertman sliding law."""

    regu: float
    exponent: float


class Weertman(SlidingComponent):
    """Sliding component implementing Weertman's sliding law."""

    def __init__(self, params: WeertmanParams) -> None:
        """Initialize Weertman sliding component with parameters."""
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        vert_disc: VerticalDiscr,
        staggered_grid: bool,
    ) -> tf.Tensor:
        """Compute Weertman sliding cost."""
        return cost_weertman(U, V, fieldin, vert_disc, staggered_grid, self.params)


def cost_weertman(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    vert_disc: VerticalDiscr,
    staggered_grid: bool,
    weertman_params: WeertmanParams,
) -> tf.Tensor:
    """Compute Weertman sliding cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    C = fieldin["slidingco"]
    dx = fieldin["dX"]

    V_b = vert_disc.V_b

    dtype = U.dtype
    m = tf.cast(weertman_params.exponent, dtype)
    u_regu = tf.cast(weertman_params.regu, dtype)

    return _cost(U, V, h, s, C, dx, m, u_regu, V_b, staggered_grid)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    C: tf.Tensor,
    dx: tf.Tensor,
    m: tf.Tensor,
    u_regu: tf.Tensor,
    V_b: tf.Tensor,
    staggered_grid: bool,
) -> tf.Tensor:
    """
    Compute the Weertman sliding law cost term.

    Calculates the sliding energy dissipation using Weertman's power law:
    C * |u_b|^(1+1/m) / (1+1/m), where u_b is the basal velocity magnitude
    corrected for bed topography.

    Parameters
    ----------
    U : tf.Tensor
        Horizontal velocity along x axis (m/year)
    V : tf.Tensor
        Horizontal velocity along y axis (m/year)
    h : tf.Tensor
        Ice thickness (m)
    s : tf.Tensor
        Upper-surface elevation (m)
    C : tf.Tensor
        Friction coefficient (MPa (m/year)^(-1/m))
    dx : tf.Tensor
        Grid spacing (m)
    m : tf.Tensor
        Weertman exponent (-)
    u_regu : tf.Tensor
        Regularization parameter for velocity magnitude (m/year)
    V_b : tf.Tensor
        Basal extraction vector: dofs -> basal
    staggered_grid : bool
        Staggering of (U, V, C)

    Returns
    -------
    tf.Tensor
        Weertman sliding cost in MPa m/year
    """

    # Staggering
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        C = stag4h(C)

    # Retrieve basal velocity
    ux_b = tf.einsum("j,bjkl->bkl", V_b, U)
    uy_b = tf.einsum("j,bjkl->bkl", V_b, V)

    # Compute bed gradient ∇b
    b = s - h
    dbdx, dbdy = grad_xy(b, dx, dx, staggered_grid, "extrapolate")

    # Compute basal velocity magnitude (with norm M and regularization)
    u_corr_b = ux_b * dbdx + uy_b * dbdy
    u_b = tf.sqrt(ux_b * ux_b + uy_b * uy_b + u_regu * u_regu + u_corr_b * u_corr_b)

    # Effective exponent
    dtype = U.dtype
    s = tf.constant(1.0, dtype=dtype) + tf.constant(1.0, dtype=dtype) / m

    # C * |u_b|^s / s
    return C * tf.pow(u_b, s) / s
