#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from typing import Callable, Tuple


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


@tf.function
def compute_zetas(
    Nz: int, slope_init: float = 1.0, dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute zeta coordinates, midpoints, and spacings."""
    zeta = compute_zeta(Nz, slope_init, dtype)
    zeta_mid = compute_zeta_mid(zeta)
    dzeta = compute_dzeta(zeta)
    return zeta, zeta_mid, dzeta


def compute_gauss_quad(
    order: int, dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute Gauss-Legendre quadrature points and weights on [0,1]."""
    x_quad, w_quad = np.polynomial.legendre.leggauss(order)

    x_quad = 0.5 * (x_quad + 1.0)
    w_quad = 0.5 * w_quad

    x_quad_tf = tf.constant(x_quad, dtype=dtype)
    w_quad_tf = tf.constant(w_quad, dtype=dtype)

    return x_quad_tf, w_quad_tf


def compute_basis_vector(
    basis: Tuple[Callable[[tf.Tensor], tf.Tensor], ...], x: tf.Tensor
) -> tf.Tensor:
    """Evaluate all basis functions at a single point."""
    V = [fct(x) for fct in basis]
    return V


def compute_basis_matrix(
    basis: Tuple[Callable[[tf.Tensor], tf.Tensor], ...], x: tf.Tensor
) -> tf.Tensor:
    """Evaluate all basis functions at multiple points to form a matrix."""
    M = [fct(x) for fct in basis]
    M = tf.stack(M, axis=1)
    return M


def compute_matrices(
    basis_fct: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    basis_fct_grad: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    basis_fct_int: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    x_quad: tf.Tensor,
    w_quad: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute basis matrices at quadrature points, boundaries, and vertical average."""

    V_q = compute_basis_matrix(basis_fct, x_quad)
    V_q_grad = compute_basis_matrix(basis_fct_grad, x_quad)
    V_q_int = compute_basis_matrix(basis_fct_int, x_quad)

    x_b = tf.constant(0.0, dtype=x_quad.dtype)
    x_s = tf.constant(1.0, dtype=x_quad.dtype)

    V_b = compute_basis_vector(basis_fct, x_b)
    V_s = compute_basis_vector(basis_fct, x_s)

    V_b = tf.stack(V_b)
    V_s = tf.stack(V_s)

    V_bar = tf.reduce_sum(V_q * w_quad[:, None], axis=0)

    return V_q, V_q_grad, V_q_int, V_b, V_s, V_bar
