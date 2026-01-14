#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function()
def compute_zeta_linear(Nz: int, dtype: tf.DType = tf.float32) -> tf.Tensor:
    """Compute linearly spaced vertical coordinates from 0 to 1."""
    return tf.cast(tf.range(Nz) / (Nz - 1), dtype)


@tf.function()
def compute_zeta_quadratic(
    Nz: int, slope_init: float, dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Compute quadratically spaced vertical coordinates with specified initial slope."""
    zeta = compute_zeta_linear(Nz, dtype)
    return slope_init * zeta + (1.0 - slope_init) * zeta**2


@tf.function()
def compute_zeta(
    Nz: int, slope_init: float = 1.0, dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Compute vertical coordinate distribution (default quadratic)."""
    return compute_zeta_quadratic(Nz, slope_init, dtype)


@tf.function()
def compute_zeta_mid(zeta: tf.Tensor) -> tf.Tensor:
    """Compute midpoints between consecutive zeta values."""
    Nz = zeta.shape[0]
    if Nz > 1:
        return (zeta[1:] + zeta[:-1]) / 2.0
    else:
        return 0.5 * tf.ones((1), dtype=zeta.dtype)


@tf.function()
def compute_dzeta(zeta: tf.Tensor) -> tf.Tensor:
    """Compute spacings between consecutive zeta values."""
    Nz = zeta.shape[0]
    if Nz > 1:
        return zeta[1:] - zeta[:-1]
    else:
        return 1.0 * tf.ones((1), dtype=zeta.dtype)


@tf.function()
def compute_depth(dzeta: tf.Tensor) -> tf.Tensor:
    """Compute normalized depth below ice surface at each level."""
    zero = tf.zeros((1,), dtype=dzeta.dtype)
    D = tf.concat([dzeta, zero], axis=0)
    return tf.math.cumsum(D, axis=0, reverse=True)


@tf.function()
def compute_weights(dzeta: tf.Tensor) -> tf.Tensor:
    """Compute integration weights for vertical quadrature (midpoint rule)."""
    Nz = dzeta.shape[0] + 1  # dzeta has Nz-1 elements

    if Nz == 1:
        return tf.ones((1,), dtype=dzeta.dtype)

    # Surface weight: half of first spacing
    w_surface = dzeta[0:1] / 2.0

    # Bed weight: half of last spacing
    w_bed = dzeta[-1:] / 2.0

    # Interior weights: average of adjacent spacings
    if Nz > 2:
        w_interior = (dzeta[:-1] + dzeta[1:]) / 2.0
        weights = tf.concat([w_surface, w_interior, w_bed], axis=0)
    else:
        weights = tf.concat([w_surface, w_bed], axis=0)

    return weights
