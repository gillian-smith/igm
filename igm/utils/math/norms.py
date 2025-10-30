#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf


def compute_norm(tensor: tf.Tensor, ord: str = "l2") -> tf.Tensor:

    if ord == "l2":
        norm_value = tf.norm(tensor, ord=2)
    elif ord == "linf":
        norm_value = tf.norm(tensor, ord=np.inf)
    elif ord == "l2_weighted":
        n = tf.cast(tf.size(tensor), tensor.dtype)
        norm_value = tf.norm(tensor, ord=2) / tf.sqrt(n)
    else:
        raise ValueError(
            f"Unknown norm type: {ord}. Use 'l2', 'linf', or 'l2_weighted'"
        )

    return norm_value
