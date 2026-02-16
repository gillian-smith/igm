#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional
import tensorflow as tf

from .utils import masked_integral


def lap_sq(
    field: tf.Tensor,
    dx: tf.Tensor,
    lam: tf.Tensor,
    mask: Optional[tf.Tensor],
    A_domain: tf.Tensor,
    eps: float = 1e-12,
    ref: Optional[tf.Tensor] = None,  # unused here
) -> tf.Tensor:
    """
    0.5 * lam * ( ∫ lap(field)^2 dA ) / A_domain
    """
    dtype = field.dtype
    f = field[None, ..., None]  # [1, Ny, Nx, 1]

    k = tf.constant(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=dtype
    )[:, :, None, None]

    fpad = tf.pad(f, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    lap = tf.nn.conv2d(fpad, k, strides=1, padding="VALID")  # [1, Ny, Nx, 1]

    # dx^2 can be scalar [] or field [Ny,Nx].
    dx2 = tf.cast(dx * dx, dtype)

    dx2_4d = tf.reshape(dx2, tf.concat([[1], tf.shape(dx2), [1]], axis=0))

    lap = lap / dx2_4d

    lap2 = tf.square(lap)[0, :, :, 0]  # [Ny, Nx]

    m = tf.ones_like(lap2, dtype=tf.bool) if mask is None else tf.cast(mask, tf.bool)
    integral = masked_integral(lap2, m, dx)

    denom = tf.cast(A_domain, dtype) + tf.cast(eps, dtype)
    return tf.cast(0.5, dtype) * lam * integral / denom


def penalty_l2(
    field: tf.Tensor,
    dx: tf.Tensor,
    lam: tf.Tensor,
    mask: Optional[tf.Tensor],
    A_domain: tf.Tensor,
    eps: float = 1e-12,
    ref: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    0.5 * lam * ( ∫ (field - ref)^2 dA ) / A_domain   if ref is provided
    0.5 * lam * ( ∫ field^2 dA ) / A_domain           otherwise
    """
    dtype = field.dtype
    diff = field if ref is None else (field - tf.cast(ref, dtype))
    sq = tf.square(diff)

    m = tf.ones_like(sq, dtype=tf.bool) if mask is None else tf.cast(mask, tf.bool)
    integral = masked_integral(sq, m, dx)

    denom = tf.cast(A_domain, dtype) + tf.cast(eps, dtype)
    return tf.cast(0.5, dtype) * lam * integral / denom


PENALTY_REGISTRY = {
    "squared_laplacian": lap_sq,
    "l2": penalty_l2,
}
