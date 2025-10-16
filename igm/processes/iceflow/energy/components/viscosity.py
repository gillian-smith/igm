#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict, Tuple

from .energy import EnergyComponent
from igm.processes.iceflow.energy.utils import stag4h
from igm.processes.iceflow.vertical import VerticalDiscr
from igm.utils.gradient.compute_gradient import compute_gradient


class ViscosityParams(tf.experimental.ExtensionType):
    """Parameters for viscous energy component."""

    n: float
    h_min: float
    ε_dot_regu: float
    ε_dot_min: float
    ε_dot_max: float


class ViscosityComponent(EnergyComponent):
    """Energy component representing viscous energy dissipation."""

    def __init__(self, params) -> None:
        """Initialize viscous component with parameters."""
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        vert_disc: VerticalDiscr,
        staggered_grid: bool,
    ) -> tf.Tensor:
        """Compute viscous energy cost."""
        return cost_viscosity(U, V, fieldin, vert_disc, staggered_grid, self.params)


def get_viscosity_params_args(cfg) -> Dict[str, Any]:
    """Extract viscous parameters from configuration."""

    cfg_physics = cfg.processes.iceflow.physics

    return {
        "n": cfg_physics.exp_glen,
        "h_min": cfg_physics.thr_ice_thk,
        "ε_dot_regu": cfg_physics.regu_glen,
        "ε_dot_min": cfg_physics.min_sr,
        "ε_dot_max": cfg_physics.max_sr,
    }


def cost_viscosity(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict,
    vert_disc: VerticalDiscr,
    staggered_grid: bool,
    viscosity_params: ViscosityParams,
) -> tf.Tensor:
    """Compute viscous energy cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    A = fieldin["arrhenius"]
    dx = fieldin["dX"]

    V_q = vert_disc.V_q
    V_q_grad = vert_disc.V_q_grad
    w = vert_disc.w

    n = viscosity_params.n
    h_min = viscosity_params.h_min
    ε_dot_regu = viscosity_params.ε_dot_regu
    ε_dot_min = viscosity_params.ε_dot_min
    ε_dot_max = viscosity_params.ε_dot_max

    return _cost(
        U,
        V,
        h,
        s,
        A,
        dx,
        n,
        h_min,
        ε_dot_regu,
        ε_dot_min,
        ε_dot_max,
        V_q,
        V_q_grad,
        w,
        staggered_grid,
    )


@tf.function()
def compute_horizontal_derivatives(
    U: tf.Tensor, V: tf.Tensor, dx: float, staggered_grid: bool
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute horizontal velocity derivatives using finite differences."""

    if staggered_grid:

        dUdx = (U[..., :, :, 1:] - U[..., :, :, :-1]) / dx
        dVdx = (V[..., :, :, 1:] - V[..., :, :, :-1]) / dx
        dUdy = (U[..., :, 1:, :] - U[..., :, :-1, :]) / dx
        dVdy = (V[..., :, 1:, :] - V[..., :, :-1, :]) / dx

        dUdx = (dUdx[..., :, :-1, :] + dUdx[..., :, 1:, :]) / 2
        dVdx = (dVdx[..., :, :-1, :] + dVdx[..., :, 1:, :]) / 2
        dUdy = (dUdy[..., :, :, :-1] + dUdy[..., :, :, 1:]) / 2
        dVdy = (dVdy[..., :, :, :-1] + dVdy[..., :, :, 1:]) / 2

    else:

        paddings = [[0, 0]] * (len(U.shape) - 2) + [[1, 1], [1, 1]]
        U = tf.pad(U, paddings, mode="SYMMETRIC")
        V = tf.pad(V, paddings, mode="SYMMETRIC")

        dUdx = (U[..., :, 1:-1, 2:] - U[..., :, 1:-1, :-2]) / (2 * dx)
        dVdx = (V[..., :, 1:-1, 2:] - V[..., :, 1:-1, :-2]) / (2 * dx)
        dUdy = (U[..., :, 2:, 1:-1] - U[..., :, :-2, 1:-1]) / (2 * dx)
        dVdy = (V[..., :, 2:, 1:-1] - V[..., :, :-2, 1:-1]) / (2 * dx)

    return dUdx, dVdx, dUdy, dVdy


@tf.function()
def compute_ε_dot2_xy(
    dUdx: tf.Tensor, dVdx: tf.Tensor, dUdy: tf.Tensor, dVdy: tf.Tensor
) -> tf.Tensor:
    """Compute horizontal contribution to squared strain rate."""

    dtype = dUdx.dtype
    half = tf.constant(0.5, dtype=dtype)
    
    Exx = dUdx
    Eyy = dVdy
    Ezz = -dUdx - dVdy
    Exy = half * dVdx + half * dUdy

    return half * (Exx**2 + Exy**2 + Exy**2 + Eyy**2 + Ezz**2)


@tf.function()
def compute_ε_dot2_z(dUdz: tf.Tensor, dVdz: tf.Tensor) -> tf.Tensor:
    """Compute vertical contribution to squared strain rate."""

    dtype = dUdz.dtype
    half = tf.constant(0.5, dtype=dtype)
    
    Exz = half * dUdz
    Eyz = half * dVdz

    return half * (Exz**2 + Eyz**2 + Exz**2 + Eyz**2)


@tf.function()
def compute_ε_dot2(
    dUdx: tf.Tensor,
    dVdx: tf.Tensor,
    dUdy: tf.Tensor,
    dVdy: tf.Tensor,
    dUdz: tf.Tensor,
    dVdz: tf.Tensor,
) -> tf.Tensor:
    """Compute squared strain rate."""

    ε_dot2_xy = compute_ε_dot2_xy(dUdx, dVdx, dUdy, dVdy)
    ε_dot2_z = compute_ε_dot2_z(dUdz, dVdz)

    return ε_dot2_xy + ε_dot2_z


def dampen_ε_dot_z_floating(
    dUdz: tf.Tensor, dVdz: tf.Tensor, C: tf.Tensor, factor: float = 1e-2
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Dampen vertical velocity gradients in floating regions."""

    dtype = dUdz.dtype
    zero = tf.constant(0.0, dtype=dtype)
    factor_const = tf.constant(factor, dtype=dtype)
    
    dUdz = tf.where(C[:, None, :, :] > zero, dUdz, factor_const * dUdz)
    dVdz = tf.where(C[:, None, :, :] > zero, dVdz, factor_const * dVdz)

    return dUdz, dVdz


@tf.function()
def correct_for_change_of_coordinate(
    dUdx: tf.Tensor,
    dVdx: tf.Tensor,
    dUdy: tf.Tensor,
    dVdy: tf.Tensor,
    dUdz: tf.Tensor,
    dVdz: tf.Tensor,
    dldx: tf.Tensor,
    dldy: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Correct derivatives for terrain-following coordinate transformation."""

    # This correct for the change of coordinate z -> z - l

    dUdx = dUdx - dUdz * dldx[:, None, :, :]
    dUdy = dUdy - dUdz * dldy[:, None, :, :]
    dVdx = dVdx - dVdz * dldx[:, None, :, :]
    dVdy = dVdy - dVdz * dldy[:, None, :, :]

    return dUdx, dVdx, dUdy, dVdy


@tf.function()
def _cost(
    U,
    V,
    h,
    s,
    A,
    dx,
    n,
    h_min,
    ε_dot_regu,
    ε_dot_min,
    ε_dot_max,
    V_q,
    V_q_grad,
    w,
    staggered_grid,
):
    """
    Compute the viscous energy dissipation cost term.

    Calculates the energy dissipation due to ice viscosity using Glen's flow law:
    h * ∫(B * ε_dot^(1+1/n) / (1+1/n))dz, where ε_dot is the effective strain rate
    and B is the ice stiffness parameter.

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
    A : tf.Tensor
        Arrhenius factor (Pa^-n year^-1)
    dx : tf.Tensor
        Grid spacing (m)
    n : float
        Glen's flow law exponent (-)
    h_min : float
        Minimum ice thickness threshold (m)
    ε_dot_regu : float
        Regularization parameter for strain rate (year^-1)
    ε_dot_min : float
        Minimum strain rate (year^-1)
    ε_dot_max : float
        Maximum strain rate (year^-1)
    V_q : tf.Tensor
        Quadrature matrix: dofs -> quads
    V_q_grad : tf.Tensor
        Gradient quadrature matrix: dofs -> grad at quads
    w : tf.Tensor
        Weights for vertical integration
    staggered_grid : bool
        Additional staggering of (U, V, h, B)

    Returns
    -------
    tf.Tensor
        Viscous energy dissipation cost in MPa m/year
    """

    # Get dtype from input tensors
    dtype = U.dtype
    
    # Ice stiffness parameter
    B = tf.constant(2.0, dtype=dtype) * tf.pow(A, tf.constant(-1.0, dtype=dtype) / tf.constant(n, dtype=dtype))

    if len(B.shape) == 3:
        B = B[:, None, :, :]

    # Effective exponent
    p = tf.constant(1.0, dtype=dtype) + tf.constant(1.0, dtype=dtype) / tf.constant(n, dtype=dtype)

    dUdx, dVdx, dUdy, dVdy = compute_horizontal_derivatives(
        U, V, dx[0, 0, 0], staggered_grid
    )

    # TODO: dldx, dldy must be the elevaion of layers! not the bedrock, little effects?
    l = s - h
    dldx, dldy = compute_gradient(l, dx, dx, staggered_grid)

    # Optional additional staggering
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        h = stag4h(h)
        B = stag4h(B)

    # Retrieve ice stiffness at quadrature points
    if B.shape[-3] > 1:
        B_q = tf.einsum("ij,bjkl->bikl", V_q, B)
    else:
        B_q = B

    # Retrieve velocity gradients at quadrature points
    dudx_q = tf.einsum("ij,bjkl->bikl", V_q, dUdx)
    dvdx_q = tf.einsum("ij,bjkl->bikl", V_q, dVdx)
    dudy_q = tf.einsum("ij,bjkl->bikl", V_q, dUdy)
    dvdy_q = tf.einsum("ij,bjkl->bikl", V_q, dVdy)

    dudz_q = tf.einsum("ij,bjkl->bikl", V_q_grad, U)
    dvdz_q = tf.einsum("ij,bjkl->bikl", V_q_grad, V)

    dudz_q = dudz_q / tf.expand_dims(tf.maximum(h, h_min), axis=1)
    dvdz_q = dvdz_q / tf.expand_dims(tf.maximum(h, h_min), axis=1)
    # dudz_q, dvdz_q = dampen_ε_dot_z_floating(dudz_q, dvdz_q, C)

    # Correct for terrain-following coordinates
    dudx_q, dvdx_q, dudy_q, dvdy_q = correct_for_change_of_coordinate(
        dudx_q, dvdx_q, dudy_q, dvdy_q, dudz_q, dvdz_q, dldx, dldy
    )

    # Compute strain rate
    ε_dot2_q = compute_ε_dot2(dudx_q, dvdx_q, dudy_q, dvdy_q, dudz_q, dvdz_q)

    dtype = U.dtype

    ε_dot2_min = tf.pow(tf.constant(ε_dot_min, dtype=dtype), 2.0)
    ε_dot2_max = tf.pow(tf.constant(ε_dot_max, dtype=dtype), 2.0)
    ε_dot2_q = tf.clip_by_value(ε_dot2_q, ε_dot2_min, ε_dot2_max)

    # Compute viscous contribution
    ε_dot2_regu = tf.pow(tf.constant(ε_dot_regu, dtype=dtype), tf.constant(2.0, dtype=dtype))
    exponent = (p - tf.constant(2.0, dtype=dtype)) / tf.constant(2.0, dtype=dtype)
    visc_term_q = tf.pow(ε_dot2_q + ε_dot2_regu, exponent) * ε_dot2_q / p

    # h * ∫ [B * ε_dot^(1+1/n) / (1+1/n)] dz
    w_q = w[None, :, None, None]
    return h * tf.reduce_sum(B_q * visc_term_q * w_q, axis=1)
