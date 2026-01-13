#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function
def compute_drainage(
    omega: tf.Tensor,
    omega_threshold_1: tf.Tensor,
    omega_threshold_2: tf.Tensor,
    omega_threshold_3: tf.Tensor,
) -> tf.Tensor:
    result = tf.fill(tf.shape(omega), 0.05)
    result = tf.where(omega <= omega_threshold_3, 4.5 * omega - 0.085, result)
    result = tf.where(omega <= omega_threshold_2, 0.5 * omega - 0.005, result)
    result = tf.where(omega <= omega_threshold_1, 0.0, result)
    return result
