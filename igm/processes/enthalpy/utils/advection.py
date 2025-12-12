#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function
def compute_upwind_advection(
    U: tf.Tensor,
    V: tf.Tensor,
    E: tf.Tensor,
    dx: float,
) -> tf.Tensor:
    """
    Compute horizontal advection using upwind scheme.

    Computes: U·∂E/∂x + V·∂E/∂y

    Args:
        U: X-velocity [m/s]
        V: Y-velocity [m/s]
        E: Field to advect (e.g., enthalpy) [any units]
        dx: Horizontal grid spacing [m]

    Returns:
        Advection rate [E·s⁻¹]
    """
    # Extend E with symmetric boundary conditions
    Ex = tf.pad(E, [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")  # shape: (nz, ny, nx+2)
    Ey = tf.pad(E, [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")  # shape: (nz, ny+2, nx)

    # Upwind differences in x-direction
    Rx = U * tf.where(
        U > 0,
        (Ex[:, :, 1:-1] - Ex[:, :, :-2]) / dx,  # Forward difference
        (Ex[:, :, 2:] - Ex[:, :, 1:-1]) / dx,  # Backward difference
    )

    # Upwind differences in y-direction
    Ry = V * tf.where(
        V > 0,
        (Ey[:, 1:-1, :] - Ey[:, :-2, :]) / dx,  # Forward difference
        (Ey[:, 2:, :] - Ey[:, 1:-1, :]) / dx,  # Backward difference
    )

    return Rx + Ry
