#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function()
def stag2x(X: tf.Tensor) -> tf.Tensor:
    """Average along x-axis (last dimension) to destagger grid."""
    return 0.5 * (X[..., 1:] + X[..., :-1])


@tf.function()
def stag2y(X: tf.Tensor) -> tf.Tensor:
    """Average along y-axis (second-to-last dimension) to destagger grid."""
    return 0.5 * (X[..., 1:, :] + X[..., :-1, :])


@tf.function()
def stag2z(X: tf.Tensor) -> tf.Tensor:
    """Average along z-axis (third-to-last dimension) to destagger grid."""
    return 0.5 * (X[..., 1:, :, :] + X[..., :-1, :, :])


@tf.function()
def stag4xy(X: tf.Tensor) -> tf.Tensor:
    """Average over 2x2 cells in xy-plane to destagger grid."""
    return 0.25 * (
        X[..., 1:, 1:] + X[..., 1:, :-1] + X[..., :-1, 1:] + X[..., :-1, :-1]
    )


@tf.function()
def stag2(X: tf.Tensor) -> tf.Tensor:
    """Alias for stag2x."""
    return stag2x(X)


@tf.function()
def stag2v(X: tf.Tensor) -> tf.Tensor:
    """Alias for stag2z."""
    return stag2z(X)


@tf.function()
def stag4h(X: tf.Tensor) -> tf.Tensor:
    """Alias for stag4xy."""
    return stag4xy(X)
