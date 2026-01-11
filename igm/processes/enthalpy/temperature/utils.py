#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple


@tf.function
def compute_pmp_tf(
    rho_ice: tf.Tensor,
    g: tf.Tensor,
    depth_ice: tf.Tensor,
    beta: tf.Tensor,
    c_ice: tf.Tensor,
    T_pmp_ref: tf.Tensor,
    T_ref: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    p_ice = rho_ice * g * depth_ice
    Tpmp = T_pmp_ref - beta * p_ice
    Epmp = c_ice * (Tpmp - T_ref)
    return Tpmp, Epmp


@tf.function
def compute_T_tf(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    T_pmp: tf.Tensor,
    T_ref: tf.Tensor,
    c_ice: tf.Tensor,
) -> tf.Tensor:
    return tf.where(E >= E_pmp, T_pmp, E / c_ice + T_ref)


@tf.function
def compute_omega_tf(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    L_ice: tf.Tensor,
) -> tf.Tensor:
    return tf.where(E >= E_pmp, (E - E_pmp) / L_ice, 0.0)


@tf.function
def compute_E_cold_tf(
    T: tf.Tensor,
    T_pmp: tf.Tensor,
    T_ref: tf.Tensor,
    c_ice: tf.Tensor,
) -> tf.Tensor:
    return tf.where(
        T < T_pmp,
        c_ice * (T - T_ref),
        c_ice * (T_pmp - T_ref),
    )
