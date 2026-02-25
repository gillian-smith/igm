#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from igm.utils.grad.grad import pad_x, pad_y


@tf.function
def compute_advection_upwind(
    U: tf.Tensor,
    V: tf.Tensor,
    V_U_to_E: tf.Tensor,
    E: tf.Tensor,
    dx: tf.Tensor,
    mode_pad_xy: str = "symmetric",
) -> tf.Tensor:
    """
    TensorFlow function to compute horizontal advection using upwind scheme.

    Calculates the advective flux divergence using a first-order upwind
    differencing scheme for numerical stability.

    Args:
        U: Horizontal velocity in x-direction (m yr^-1).
        V: Horizontal velocity in y-direction (m yr^-1).
        V_U_to_E: Map velocity DOFs to values at enthalpy nodes (Ndof_E, Ndof_U).
        E: Enthalpy field (J kg^-1).
        dx: Horizontal grid spacing (m).
        mode_pad_xy: Padding mode for horizontal boundaries.

    Returns:
        Advective rate of change of enthalpy (J kg^-1 yr^-1).
    """
    U = tf.einsum("ij,jkl->ikl", V_U_to_E, U)
    V = tf.einsum("ij,jkl->ikl", V_U_to_E, V)

    # Extend E
    Ex = pad_x(E, mode_pad_xy)
    Ey = pad_y(E, mode_pad_xy)

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
