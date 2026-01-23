#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple


@tf.function()
def compute_eps_dot2_xy(
    dudx: tf.Tensor, dvdx: tf.Tensor, dudy: tf.Tensor, dvdy: tf.Tensor
) -> tf.Tensor:
    """Compute horizontal contribution to squared strain rate."""

    eps_xx = dudx
    eps_yy = dvdy
    eps_zz = -(dudx + dvdy)
    eps_xy = 0.5 * dvdx + 0.5 * dudy

    return 0.5 * (eps_xx**2 + eps_xy**2 + eps_xy**2 + eps_yy**2 + eps_zz**2)


@tf.function()
def compute_eps_dot2_z(dudz: tf.Tensor, dvdz: tf.Tensor) -> tf.Tensor:
    """Compute vertical contribution to squared strain rate."""

    eps_xz = 0.5 * dudz
    eps_yz = 0.5 * dvdz

    return 0.5 * (eps_xz**2 + eps_yz**2 + eps_xz**2 + eps_yz**2)


@tf.function()
def compute_eps_dot2(
    dudx: tf.Tensor,
    dvdx: tf.Tensor,
    dudy: tf.Tensor,
    dvdy: tf.Tensor,
    dudz: tf.Tensor,
    dvdz: tf.Tensor,
) -> tf.Tensor:
    """Compute squared strain rate."""

    eps_dot2_xy = compute_eps_dot2_xy(dudx, dvdx, dudy, dvdy)
    eps_dot2_z = compute_eps_dot2_z(dudz, dvdz)

    return eps_dot2_xy + eps_dot2_z


def dampen_eps_dot_z_floating(
    dudz: tf.Tensor, dvdz: tf.Tensor, C: tf.Tensor, factor: float = 1e-2
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Dampen vertical velocity gradients in floating regions."""

    dudz = tf.where(C[:, None, :, :] > 0.0, dudz, factor * dudz)
    dvdz = tf.where(C[:, None, :, :] > 0.0, dvdz, factor * dvdz)

    return dudz, dvdz


@tf.function()
def correct_grad_zeta_to_z(
    dudx: tf.Tensor,
    dudy: tf.Tensor,
    dvdx: tf.Tensor,
    dvdy: tf.Tensor,
    dudz: tf.Tensor,
    dvdz: tf.Tensor,
    dldx: tf.Tensor,
    dldy: tf.Tensor,
    dsdx: tf.Tensor,
    dsdy: tf.Tensor,
    zeta: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Correct derivatives for terrain-following coordinate transformation."""

    # ∇ζ = - [(1 - ζ) * ∇b + ζ * ∇s] / h
    # We omit the division by h because dudz_q and dvdz_q are
    # already physical vertical derivatives (∂u/∂z = ∂u/∂ζ / h)
    dzetadx = -((1.0 - zeta) * dldx + zeta * dsdx)
    dzetady = -((1.0 - zeta) * dldy + zeta * dsdy)

    # ∇_z = ∇_ζ + (∂/∂ζ) * ∇ζ
    dudx = dudx + dudz * dzetadx
    dudy = dudy + dudz * dzetady
    dvdx = dvdx + dvdz * dzetadx
    dvdy = dvdy + dvdz * dzetady

    return dudx, dudy, dvdx, dvdy
