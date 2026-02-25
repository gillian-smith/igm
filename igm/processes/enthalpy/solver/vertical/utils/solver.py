#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function
def solve_tridiagonal_system(
    L: tf.Tensor,
    M: tf.Tensor,
    U: tf.Tensor,
    R: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to solve tridiagonal system using Thomas Algorithm (TDMA).

    Performs forward elimination and backward substitution to solve the
    tridiagonal linear system arising from the implicit enthalpy discretization.

    Args:
        L: Lower diagonal coefficients (nz-1, ny, nx).
        M: Main diagonal coefficients (nz, ny, nx).
        U: Upper diagonal coefficients (nz-1, ny, nx).
        R: Right-hand side vector (nz, ny, nx).

    Returns:
        Solution vector (nz, ny, nx).
    """
    nz = tf.shape(M)[0]

    w = tf.TensorArray(dtype=tf.float32, size=nz - 1)
    g = tf.TensorArray(dtype=tf.float32, size=nz)
    p = tf.TensorArray(dtype=tf.float32, size=nz)

    # Forward sweep
    w = w.write(0, U[0] / M[0])
    g = g.write(0, R[0] / M[0])

    for i in tf.range(1, nz - 1):
        w = w.write(i, U[i] / (M[i] - L[i - 1] * w.read(i - 1)))

    for i in tf.range(1, nz):
        g = g.write(
            i, (R[i] - L[i - 1] * g.read(i - 1)) / (M[i] - L[i - 1] * w.read(i - 1))
        )

    # Backward substitution
    p = p.write(nz - 1, g.read(nz - 1))

    for i in tf.range(nz - 2, -1, -1):
        p = p.write(i, g.read(i) - w.read(i) * p.read(i + 1))

    return p.stack()
