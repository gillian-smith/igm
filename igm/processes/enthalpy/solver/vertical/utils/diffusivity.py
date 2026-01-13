#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function()
def compute_diffusivity(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    k_ice: tf.Tensor,
    rho_ice: tf.Tensor,
    c_ice: tf.Tensor,
    K_ratio: tf.Tensor,
) -> tf.Tensor:

    K_factor = k_ice / (rho_ice * c_ice)

    E_mid = (E[:-1] + E[1:]) / 2.0
    E_pmp_mid = (E_pmp[:-1] + E_pmp[1:]) / 2.0

    return tf.where(E_mid >= E_pmp_mid, K_factor * K_ratio, K_factor)
