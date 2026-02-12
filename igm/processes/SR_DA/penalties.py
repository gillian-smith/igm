#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import tensorflow as tf
from .utils import masked_mean

def penalty_biharmonic(
    field: tf.Tensor,
    dx: tf.Tensor,
    lam: tf.Tensor,
    mask: Optional[tf.Tensor],
    eps: float = 1e-12,
    ref: Optional[tf.Tensor] = None, # not needed for this penalty
) -> tf.Tensor:
    """
      0.5 * lam * mean( lap(field)^2 )
    """
    dtype = field.dtype
    f = field[None, ..., None]  # [1, Ny, Nx, 1]

    k = tf.constant([[0., 1., 0.],
                     [1., -4., 1.],
                     [0., 1., 0.]], dtype=dtype)[:, :, None, None]

    fpad = tf.pad(f, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    lap = tf.nn.conv2d(fpad, k, strides=1, padding="VALID")  # [1, Ny, Nx, 1]

    dx2 = tf.reshape(dx * dx, [1, tf.shape(dx)[0], tf.shape(dx)[1], 1])
    lap = lap / tf.cast(dx2, dtype)

    lap2 = tf.square(lap)[0, :, :, 0]  # [Ny, Nx]

    if mask is None:
        mean_lap2 = tf.reduce_mean(lap2)
    else:
        mean_lap2 = masked_mean(lap2, tf.cast(mask, tf.bool), eps=eps)

    return tf.cast(0.5, dtype) * lam * mean_lap2

def penalty_l2(
    field: tf.Tensor,
    dx: tf.Tensor,
    lam: tf.Tensor,
    mask: Optional[tf.Tensor],
    eps: float = 1e-12,
    ref: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
      0.5 * lam * mean( (field - ref)^2 )   if ref is provided
      0.5 * lam * mean( field^2 )          otherwise
    """
    dtype = field.dtype
    diff = field if ref is None else (field - tf.cast(ref, dtype))
    sq = tf.square(diff)

    if mask is None:
        val = tf.reduce_mean(sq)
    else:
        val = masked_mean(sq, tf.cast(mask, tf.bool), eps=eps)

    return tf.cast(0.5, dtype) * lam * val

PENALTY_REGISTRY = {
    "biharmonic": penalty_biharmonic,
    "l2": penalty_l2,
}