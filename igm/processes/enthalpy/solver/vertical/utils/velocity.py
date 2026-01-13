#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


def correct_vertical_velocity(
    W: tf.Tensor,
    basal_melt_rate: tf.Tensor,
    correct_w_for_melt: bool = True,
) -> tf.Tensor:

    if correct_w_for_melt:
        W -= tf.expand_dims(basal_melt_rate, axis=0)

    return W
