#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict
from omegaconf import DictConfig

from .energy import EnergyComponent
from igm.processes.iceflow.vertical import VerticalDiscr
from igm.utils.grad.grad import grad_xy
from igm.utils.stag.stag import stag4h


class GravityParams(tf.experimental.ExtensionType):
    """Parameters for gravity energy component."""

    rho: float
    g: float
    fnge: bool


class GravityComponent(EnergyComponent):
    """Energy component representing gravitational potential energy."""

    def __init__(self, params: GravityParams) -> None:
        """Initialize gravity component with parameters."""
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        vert_disc: VerticalDiscr,
        staggered_grid: bool,
    ) -> tf.Tensor:
        """Compute gravitational energy cost."""
        return cost_gravity(U, V, fieldin, vert_disc, staggered_grid, self.params)


def get_gravity_params_args(cfg: DictConfig) -> Dict[str, Any]:
    """Extract gravity parameters from configuration."""

    cfg_physics = cfg.processes.iceflow.physics

    return {
        "rho": cfg_physics.ice_density,
        "g": cfg_physics.gravity_cst,
        "fnge": cfg_physics.force_negative_gravitational_energy,
    }


def cost_gravity(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    vert_disc: VerticalDiscr,
    staggered_grid: bool,
    gravity_params: GravityParams,
) -> tf.Tensor:
    """Compute gravitational energy cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    dx = fieldin["dX"]

    V_q = vert_disc.V_q
    w = vert_disc.w

    dtype = U.dtype
    rho = tf.cast(gravity_params.rho, dtype)
    g = tf.cast(gravity_params.g, dtype)
    fnge = gravity_params.fnge

    return _cost(U, V, h, s, dx, rho, g, fnge, V_q, w, staggered_grid)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    dx: tf.Tensor,
    rho: tf.Tensor,
    g: tf.Tensor,
    fnge: bool,
    V_q: tf.Tensor,
    w: tf.Tensor,
    staggered_grid: bool,
) -> tf.Tensor:
    """
    Compute the gravitational energy cost term.

    Calculates the work done by gravity: rho * g * h * ∫(u·∇s)dz, where the
    integral is computed over the ice thickness using vertical quadrature.

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
    dx : tf.Tensor
        Grid spacing (m)
    rho : float
        Ice density (kg m^-3)
    g : float
        Gravity acceleration (m s^-2)
    fnge : bool
        Force negative gravitational energy flag
    V_q : tf.Tensor
        Quadrature matrix: dofs -> quads
    w : tf.Tensor
        Weights for vertical integration
    staggered_grid : bool
        Staggering of (U, V, h)

    Returns
    -------
    tf.Tensor
        Gravitational energy cost in MPa m/year
    """

    # Staggering
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        h = stag4h(h)

    # Retrieve velocity at quadrature points
    u_q = tf.einsum("ij,bjkl->bikl", V_q, U)
    v_q = tf.einsum("ij,bjkl->bikl", V_q, V)

    # Compute upper surface gradient ∇s
    dsdx, dsdy = grad_xy(s, dx, dx, staggered_grid, "extrapolate")
    dsdx_q = dsdx[:, None, :, :]
    dsdy_q = dsdy[:, None, :, :]

    dtype = U.dtype

    # Product (u,v)*∇s
    u_dsdl_q = u_q * dsdx_q + v_q * dsdy_q
    if fnge:
        u_dsdl_q = tf.minimum(u_dsdl_q, tf.constant(0.0, dtype=dtype))

    # rho * g * h * ∫ [(u,v)*∇s] dz in MPa * m/year
    w_q = w[None, :, None, None]
    scale_factor = tf.constant(10.0 ** (-6), dtype=dtype)
    return scale_factor * rho * g * h * tf.reduce_sum(u_dsdl_q * w_q, axis=1)
