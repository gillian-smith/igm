#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function
def compute_T_s_tf(
    T_air: tf.Tensor,
    T_offset: tf.Tensor,
    T_pmp_ref: tf.Tensor,
) -> tf.Tensor:

    return tf.minimum(tf.math.reduce_mean(T_air + T_offset, axis=0), 0.0) + T_pmp_ref
