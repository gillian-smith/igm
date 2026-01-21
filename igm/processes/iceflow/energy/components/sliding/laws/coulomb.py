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


class CoulombParams(tf.experimental.ExtensionType):
    """Parameters for Coulomb sliding law."""

    regu: float
    exponent: float
    mu: float


class Coulomb(SlidingComponent):
    """Sliding component implementing Coulomb's sliding law."""

    def __init__(self, params: CoulombParams):
        """Initialize Coulomb sliding component with parameters."""
        self.name = "coulomb"
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute Coulomb sliding cost."""
        return cost_coulomb(U, V, fieldin, discr_h, discr_v, self.params)


def cost_coulomb(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    coulomb_params: CoulombParams,
) -> tf.Tensor:
    """Compute Coulomb sliding cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    C = fieldin["slidingco"]
    dx = fieldin["dX"]

    V_b = discr_v.V_b

    dtype = U.dtype
    m = tf.cast(coulomb_params.exponent, dtype)
    u_regu = tf.cast(coulomb_params.regu, dtype)
    μ = tf.cast(coulomb_params.mu, dtype)

    return _cost(U, V, h, s, C, dx, m, μ, u_regu, V_b)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    C: tf.Tensor,
    dx: tf.Tensor,
    m: tf.Tensor,
    μ: tf.Tensor,
    u_regu: tf.Tensor,
    V_b: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the Coulomb sliding law cost term.

    Calculates the sliding energy dissipation using a regularized
    Coulomb power law, following following Shapero et al. (2021).

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
        Friction coefficient (Pa (m/year)^(-1/m))
    dx : tf.Tensor
        Grid spacing (m)
    m : tf.Tensor
        Coulomb exponent (-)
    μ: tf.Tensor
        Till coefficient (-)
    u_regu : tf.Tensor
        Regularization parameter for velocity magnitude (m/year)
    V_b : tf.Tensor
        Basal extraction vector: dofs -> basal

    Returns
    -------
    tf.Tensor
        Coulomb sliding cost in MPa m/year
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

    # Compute smooth transition between Weertman and Coulomb following Shapero et al. (2021)
    τ_c = μ * N
    u_c = tf.pow(τ_c / C, m)
    # τ_c * [ (|u_b|^s + |u_c|^s)^(1/s) - u_c]
    return τ_c * (tf.pow(tf.pow(u_b, s) + tf.pow(u_c, s), 1.0 / s) - u_c)
