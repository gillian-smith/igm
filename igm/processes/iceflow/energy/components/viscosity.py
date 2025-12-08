#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict, Tuple

from .energy import EnergyComponent
from igm.processes.iceflow.vertical import VerticalDiscr
from igm.utils.grad.grad import grad_xy
from igm.utils.stag.stag import stag4h


class ViscosityParams(tf.experimental.ExtensionType):
    """Parameters for viscous energy component."""

    n: float
    h_min: float
    eps_dot_regu: float
    eps_dot_min: float
    eps_dot_max: float


class ViscosityComponent(EnergyComponent):
    """Energy component representing viscous energy dissipation."""

    name = "viscosity"

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
        "eps_dot_regu": cfg_physics.regu_glen,
        "eps_dot_min": cfg_physics.min_sr,
        "eps_dot_max": cfg_physics.max_sr,
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
    zeta = vert_disc.zeta

    dtype = U.dtype
    n = tf.cast(viscosity_params.n, dtype)
    h_min = tf.cast(viscosity_params.h_min, dtype)
    eps_dot_regu = tf.cast(viscosity_params.eps_dot_regu, dtype)
    eps_dot_min = tf.cast(viscosity_params.eps_dot_min, dtype)
    eps_dot_max = tf.cast(viscosity_params.eps_dot_max, dtype)

    return _cost(
        U,
        V,
        h,
        s,
        A,
        dx,
        n,
        h_min,
        eps_dot_regu,
        eps_dot_min,
        eps_dot_max,
        V_q,
        V_q_grad,
        w,
        zeta,
        staggered_grid,
    )


@tf.function()
def compute_eps_dot2_xy(
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
def compute_eps_dot2_z(dUdz: tf.Tensor, dVdz: tf.Tensor) -> tf.Tensor:
    """Compute vertical contribution to squared strain rate."""

    dtype = dUdz.dtype
    half = tf.constant(0.5, dtype=dtype)

    Exz = half * dUdz
    Eyz = half * dVdz

    return half * (Exz**2 + Eyz**2 + Exz**2 + Eyz**2)


@tf.function()
def compute_eps_dot2(
    dUdx: tf.Tensor,
    dVdx: tf.Tensor,
    dUdy: tf.Tensor,
    dVdy: tf.Tensor,
    dUdz: tf.Tensor,
    dVdz: tf.Tensor,
) -> tf.Tensor:
    """Compute squared strain rate."""

    eps_dot2_xy = compute_eps_dot2_xy(dUdx, dVdx, dUdy, dVdy)
    eps_dot2_z = compute_eps_dot2_z(dUdz, dVdz)

    return eps_dot2_xy + eps_dot2_z


def dampen_eps_dot_z_floating(
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
    dudx_q: tf.Tensor,
    dudy_q: tf.Tensor,
    dvdx_q: tf.Tensor,
    dvdy_q: tf.Tensor,
    dudz_q: tf.Tensor,
    dvdz_q: tf.Tensor,
    dldx: tf.Tensor,
    dldy: tf.Tensor,
    dsdx: tf.Tensor,
    dsdy: tf.Tensor,
    zeta: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Correct derivatives for terrain-following coordinate transformation."""

    dtype = dudx_q.dtype
    one = tf.constant(1.0, dtype=dtype)

    # Evaluate at quadrature points
    zeta_q = zeta[None, :, None, None]
    dldx_q = dldx[:, None, :, :]
    dldy_q = dldy[:, None, :, :]
    dsdx_q = dsdx[:, None, :, :]
    dsdy_q = dsdy[:, None, :, :]

    # ∇ζ = - [(1 - ζ) * ∇b + ζ * ∇s] / h
    # We omit the division by h because dudz_q and dvdz_q are
    # already physical vertical derivatives (∂u/∂z = ∂u/∂ζ / h)
    dzetadx_q = -((one - zeta_q) * dldx_q + zeta_q * dsdx_q)
    dzetady_q = -((one - zeta_q) * dldy_q + zeta_q * dsdy_q)

    # ∇_z = ∇_ζ + (∂/∂ζ) * ∇ζ
    dudx_q = dudx_q + dudz_q * dzetadx_q
    dudy_q = dudy_q + dudz_q * dzetady_q
    dvdx_q = dvdx_q + dvdz_q * dzetadx_q
    dvdy_q = dvdy_q + dvdz_q * dzetady_q

    return dudx_q, dudy_q, dvdx_q, dvdy_q


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    A: tf.Tensor,
    dx: tf.Tensor,
    n: tf.Tensor,
    h_min: tf.Tensor,
    eps_dot_regu: tf.Tensor,
    eps_dot_min: tf.Tensor,
    eps_dot_max: tf.Tensor,
    V_q: tf.Tensor,
    V_q_grad: tf.Tensor,
    w: tf.Tensor,
    zeta: tf.Tensor,
    staggered_grid: bool,
) -> tf.Tensor:
    """
    Compute the viscous energy dissipation cost term.

    Calculates the energy dissipation due to ice viscosity using Glen's flow law:
    h * ∫(B * eps_dot^(1+1/n) / (1+1/n))dz, where eps_dot is the effective strain rate
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
        Arrhenius factor (MPa^-n year^-1)
    dx : tf.Tensor
        Grid spacing (m)
    n : tf.Tensor
        Glen's flow law exponent (-)
    h_min : tf.Tensor
        Minimum ice thickness threshold (m)
    eps_dot_regu : tf.Tensor
        Regularization parameter for strain rate (year^-1)
    eps_dot_min : tf.Tensor
        Minimum strain rate (year^-1)
    eps_dot_max : tf.Tensor
        Maximum strain rate (year^-1)
    V_q : tf.Tensor
        Quadrature matrix: dofs -> quads
    V_q_grad : tf.Tensor
        Gradient quadrature matrix: dofs -> grad at quads
    w : tf.Tensor
        Weights for vertical integration
    zeta: tf.Tensor
        Position for vertical integration
    staggered_grid : bool
        Staggering of (U, V, h, B)

    Returns
    -------
    tf.Tensor
        Viscous energy dissipation cost in MPa m/year
    """

    # Get dtype from input tensors
    # ! There is a bug here. Even if the model is fully in float64, the cost functions are not and there is a mismatch.
    # For now, we will lose precision this way for convenience but it is not true double precision...

    dtype = U.dtype

    n = tf.cast(n, dtype=dtype)

    # Ice stiffness parameter
    B = tf.constant(2.0, dtype=dtype) * tf.pow(A, tf.constant(-1.0, dtype=dtype) / n)

    if len(B.shape) == 3:
        B = B[:, None, :, :]

    # Effective exponent
    p = tf.constant(1.0, dtype=dtype) + tf.constant(1.0, dtype=dtype) / n

    # Velocity gradients
    dUdx, dUdy = grad_xy(U, dx, dx, staggered_grid)
    dVdx, dVdy = grad_xy(V, dx, dx, staggered_grid)

    # Compute upper and lower surface gradients ∇l, ∇s
    l = s - h
    dldx, dldy = grad_xy(l, dx, dx, staggered_grid, "extrapolate")
    dsdx, dsdy = grad_xy(s, dx, dx, staggered_grid, "extrapolate")

    # Staggering
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
    # dudz_q, dvdz_q = dampen_eps_dot_z_floating(dudz_q, dvdz_q, C)

    # Correct for terrain-following coordinates
    dudx_q, dudy_q, dvdx_q, dvdy_q = correct_for_change_of_coordinate(
        dudx_q,
        dudy_q,
        dvdx_q,
        dvdy_q,
        dudz_q,
        dvdz_q,
        dldx,
        dldy,
        dsdx,
        dsdy,
        zeta,
    )

    # Compute strain rate
    eps_dot2_q = compute_eps_dot2(dudx_q, dvdx_q, dudy_q, dvdy_q, dudz_q, dvdz_q)

    eps_dot2_min = tf.pow(eps_dot_min, 2.0)
    eps_dot2_max = tf.pow(eps_dot_max, 2.0)
    eps_dot2_q = tf.clip_by_value(eps_dot2_q, eps_dot2_min, eps_dot2_max)

    # Compute viscous contribution
    eps_dot2_regu = tf.pow(eps_dot_regu, tf.constant(2.0, dtype=dtype))
    exponent = (p - tf.constant(2.0, dtype=dtype)) / tf.constant(2.0, dtype=dtype)
    visc_term_q = tf.pow(eps_dot2_q + eps_dot2_regu, exponent) * eps_dot2_q / p

    # h * ∫ [B * eps_dot^(1+1/n) / (1+1/n)] dz
    w_q = w[None, :, None, None]
    return h * tf.reduce_sum(B_q * visc_term_q * w_q, axis=1)
