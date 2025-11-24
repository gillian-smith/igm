#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict

from igm.processes.iceflow.vertical import VerticalDiscr
from igm.processes.iceflow.emulate.utils.misc import get_effective_pressure_precentage
from igm.utils.grad.grad import grad_xy
from igm.utils.stag.stag import stag4h
from ..sliding import SlidingComponent


class CoulombParams(tf.experimental.ExtensionType):
    """Parameters for Coulomb sliding law."""

    regu: float
    exponent: float
    mu: float


class Coulomb(SlidingComponent):
    """Sliding component implementing Coulomb's sliding law."""

    name = "coulomb"
    
    def __init__(self, params: CoulombParams):
        """Initialize Coulomb sliding component with parameters."""
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        vert_disc: VerticalDiscr,
        staggered_grid: bool,
    ) -> tf.Tensor:
        """Compute Coulomb sliding cost."""
        return cost_coulomb(U, V, fieldin, vert_disc, staggered_grid, self.params)


def cost_coulomb(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    vert_disc: VerticalDiscr,
    staggered_grid: bool,
    coulomb_params: CoulombParams,
) -> tf.Tensor:
    """Compute Coulomb sliding cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    C = fieldin["slidingco"]
    dx = fieldin["dX"]

    V_b = vert_disc.V_b

    dtype = U.dtype
    m = tf.cast(coulomb_params.exponent, dtype)
    u_regu = tf.cast(coulomb_params.regu, dtype)
    μ = tf.cast(coulomb_params.mu, dtype)

    return _cost(U, V, h, s, C, dx, m, μ, u_regu, V_b, staggered_grid)


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
    staggered_grid: bool,
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
    staggered_grid : bool
        Staggering of (U, V, C)

    Returns
    -------
    tf.Tensor
        Coulomb sliding cost in MPa m/year
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

    # Temporary fix for effective pressure - should be within the inputs
    dtype = U.dtype
    N = get_effective_pressure_precentage(h, percentage=tf.constant(0.0, dtype=dtype))
    N = tf.where(N < tf.constant(1e-3, dtype=dtype), tf.constant(1e-3, dtype=dtype), N)

    # Effective exponent
    s = tf.constant(1.0, dtype=dtype) + tf.constant(1.0, dtype=dtype) / m

    # Compute smooth transition between Weertman and Coulomb following Shapero et al. (2021)
    τ_c = μ * N
    u_c = tf.pow(τ_c / C, tf.constant(m, dtype=dtype))
    # τ_c * [ (|u_b|^s + |u_c|^s)^(1/s) - u_c]
    return τ_c * (
        tf.pow(tf.pow(u_b, s) + tf.pow(u_c, s), tf.constant(1.0, dtype=dtype) / s) - u_c
    )
