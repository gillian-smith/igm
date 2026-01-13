#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from igm.utils.grad.grad import pad_x, pad_y


@tf.function
def compute_advection_upwind(
    U: tf.Tensor,
    V: tf.Tensor,
    E: tf.Tensor,
    dx: tf.Tensor,
    mode_pad_xy: str = "symmetric",
) -> tf.Tensor:
    """
    Compute horizontal advection using upwind scheme.
    """
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
