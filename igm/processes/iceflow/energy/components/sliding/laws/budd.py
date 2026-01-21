#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict

from ..sliding import SlidingComponent
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr
from igm.processes.iceflow.emulate.utils.misc import get_effective_pressure_precentage
from igm.utils.grad.grad import grad_xy
from igm.utils.stag.stag import stag4h


class BuddParams(tf.experimental.ExtensionType):
    """Parameters for Budd sliding law."""

    regu: float
    exponent: float


class Budd(SlidingComponent):
    """Sliding component implementing Budd's sliding law."""

    def __init__(self, params: BuddParams) -> None:
        """Initialize Budd sliding component with parameters."""
        self.name = "budd"
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute Budd sliding cost."""
        return cost_budd(U, V, fieldin, discr_h, discr_v, self.params)


def cost_budd(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    budd_params: BuddParams,
) -> tf.Tensor:
    """Compute Budd sliding cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    C = fieldin["slidingco"]
    dx = fieldin["dX"]

    V_b = discr_v.V_b

    dtype = U.dtype
    m = tf.cast(budd_params.exponent, dtype)
    u_regu = tf.cast(budd_params.regu, dtype)

    return _cost(U, V, h, s, C, dx, m, u_regu, V_b)


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
) -> tf.Tensor:
    """
    Compute the Budd sliding law cost term.

    Calculates the sliding energy dissipation using Budd's power law:
    C * N * |u_b|^(1+1/m) / (1+1/m), where u_b is the basal velocity magnitude
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
        Friction coefficient ((m/year)^(-1/m))
    dx : tf.Tensor
        Grid spacing (m)
    m : tf.Tensor
        Budd exponent (-)
    u_regu : tf.Tensor
        Regularization parameter for velocity magnitude (m/year)
    V_b : tf.Tensor
        Basal extraction vector: dofs -> basal

    Returns
    -------
    tf.Tensor
        Budd sliding cost in MPa m/year
    """

    # Staggering
    U = stag4h(U)
    V = stag4h(V)
    C = stag4h(C)

    # Retrieve basal velocity
    ux_b = tf.einsum("j,bjkl->bkl", V_b, U)
    uy_b = tf.einsum("j,bjkl->bkl", V_b, V)

    # Compute bed gradient ∇b
    b = s - h
    dbdx, dbdy = grad_xy(b, dx, dx, staggered_grid=True, mode="extrapolate")

    # Compute basal velocity magnitude (with norm M and regularization)
    u_corr_b = ux_b * dbdx + uy_b * dbdy
    u_b = tf.sqrt(ux_b * ux_b + uy_b * uy_b + u_regu * u_regu + u_corr_b * u_corr_b)

    # Temporary fix for effective pressure - should be within the inputs
    N = get_effective_pressure_precentage(h, percentage=0.0)
    N = tf.where(N < 1e-3, 1e-3, N)

    # Effective exponent
    s = 1.0 + 1.0 / m

    # C * N * |u_b|^s / s
    return C * N * tf.pow(u_b, s) / s
