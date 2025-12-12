#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict, Tuple
from omegaconf import DictConfig

from .energy import EnergyComponent
from igm.processes.iceflow.vertical import VerticalDiscr


class FloatingParams(tf.experimental.ExtensionType):
    """Parameters for floating ice-shelf calving front."""

    rho: float
    rho_water: float
    g: float
    cf_eswn: Tuple[str, ...]


class FloatingComponent(EnergyComponent):
    """Energy component for floating ice-shelf calving front."""

    name = "floating"

    def __init__(self, params: FloatingParams) -> None:
        """Initialize floating component with parameters."""
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        vert_disc: VerticalDiscr,
        _: bool,
    ) -> tf.Tensor:
        """Compute calving front energy cost."""
        return cost_floating(U, V, fieldin, vert_disc, self.params)


def get_floating_params_args(cfg: DictConfig) -> Dict[str, Any]:
    """Extract floating ice parameters from configuration."""

    cfg_physics = cfg.processes.iceflow.physics

    return {
        "cf_eswn": cfg_physics.cf_eswn,
        "rho": cfg_physics.ice_density,
        "rho_water": cfg_physics.water_density,
        "g": cfg_physics.gravity_cst,
    }


def cost_floating(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    vert_disc: VerticalDiscr,
    floating_params: FloatingParams,
) -> tf.Tensor:
    """Compute calving front energy cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    dx = fieldin["dX"]

    V_q = vert_disc.V_q
    w = vert_disc.w

    dtype = U.dtype
    rho = tf.cast(floating_params.rho, dtype)
    rho_w = tf.cast(floating_params.rho_water, dtype)
    g = tf.cast(floating_params.g, dtype)
    cf_eswn = floating_params.cf_eswn

    return _cost(U, V, h, s, dx, rho, rho_w, g, cf_eswn, V_q, w)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    dx: tf.Tensor,
    rho: tf.Tensor,
    rho_w: tf.Tensor,
    g: tf.Tensor,
    cf_eswn: Tuple[str, ...],
    V_q: tf.Tensor,
    w: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the calving front energy cost term.

    Calculates the work done by gravity: ∫_Γ P · u·n ds, where the integral
    is computed over the ice thickness using vertical quadrature, and where
    P = 0.5 * ρ * g * h² * (1 - ρ/ρ_w * r²) accounts for the stress balance
    at the calving front. This expression is divided by dx to account so
    that it is properly accounted for when summing all the cost terms.

    Parameters
    ----------
    U : tf.Tensor
        Horizontal velocity along x axis (m/year)
    V : tf.Tensor
        Horizontal velocity along y axis (m/year)
    h : tf.Tensor
        Ice thickness (m)
    s : tf.Tensor
        Upper surface elevation (m)
    dx : tf.Tensor
        Grid spacing (m)
    rho : float
        Ice density (kg m^-3)
    rho_w : float
        Water density (kg m^-3)
    g : float
        Gravity acceleration (m s^-2)
    cf_eswn : tuple
        Calving front boundaries: ("E", "S", "W", "N")
    V_q : tf.Tensor
        Quadrature matrix: dofs -> quads
    w : tf.Tensor
        Weights for vertical integration

    Returns
    -------
    tf.Tensor
        Calving front energy cost in MPa m/year
    """

    # Get dtype from input tensors
    dtype = U.dtype

    # Lower surface elevation
    l = s - h

    # Pad to detect calving front (pad with 1.0 if edge NOT in cf_eswn)
    pad_value = lambda edge: 1.0 if edge not in cf_eswn else 0.0

    # Pad h: [[batch], [top, bottom], [left, right]]
    h_ext = tf.pad(h, [[0, 0], [1, 0], [0, 0]], constant_values=pad_value("S"))
    h_ext = tf.pad(h_ext, [[0, 0], [0, 1], [0, 0]], constant_values=pad_value("N"))
    h_ext = tf.pad(h_ext, [[0, 0], [0, 0], [1, 0]], constant_values=pad_value("W"))
    h_ext = tf.pad(h_ext, [[0, 0], [0, 0], [0, 1]], constant_values=pad_value("E"))

    # Pad l
    l_ext = tf.pad(l, [[0, 0], [1, 0], [0, 0]], constant_values=pad_value("S"))
    l_ext = tf.pad(l_ext, [[0, 0], [0, 1], [0, 0]], constant_values=pad_value("N"))
    l_ext = tf.pad(l_ext, [[0, 0], [0, 0], [1, 0]], constant_values=pad_value("W"))
    l_ext = tf.pad(l_ext, [[0, 0], [0, 0], [0, 1]], constant_values=pad_value("E"))

    # Detect calving front: floating ice (l≤0, h>0) next to ocean (h=0, l≤0)
    is_float = (l <= 0.0) & (h > 0.0)
    is_ocean = lambda h, l: (h == 0.0) & (l <= 0.0)
    is_cf = lambda h, l: tf.cast(is_float & is_ocean(h, l), dtype)

    CF_W = is_cf(h_ext[:, 1:-1, :-2], l_ext[:, 1:-1, :-2])
    CF_E = is_cf(h_ext[:, 1:-1, 2:], l_ext[:, 1:-1, 2:])
    CF_S = is_cf(h_ext[:, :-2, 1:-1], l_ext[:, :-2, 1:-1])
    CF_N = is_cf(h_ext[:, 2:, 1:-1], l_ext[:, 2:, 1:-1])

    # Depth-integrated velocity using vertical quadrature
    u_q = tf.einsum("ij,bjkl->bikl", V_q, U)
    v_q = tf.einsum("ij,bjkl->bikl", V_q, V)
    w_q = w[None, :, None, None]
    U_int = tf.reduce_sum(u_q * w_q, axis=1)
    V_int = tf.reduce_sum(v_q * w_q, axis=1)

    # Pre-factor
    r = tf.maximum(-l / tf.maximum(h, 1.0), 0.0)
    P = 0.5 * g * rho * h * h * (1.0 - (rho_w / rho) * r * r)

    # ∫ P · u·n ds
    C_float = P * (U_int * (CF_E - CF_W) + V_int * (CF_N - CF_S)) / dx

    # Convert to Mpa*m/year
    return 1e-6 * C_float
