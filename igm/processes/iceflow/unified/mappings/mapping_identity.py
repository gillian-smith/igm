#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import List, Tuple

from .mapping import Mapping
from ...vertical import VerticalDiscr
from ..bcs import BoundaryCondition
from .normalizer import IdentityNormalizer


class MappingIdentity(Mapping):
    def __init__(
        self,
        bcs: List[BoundaryCondition],
        U_guess: tf.Tensor,
        V_guess: tf.Tensor,
        precision: str = "float32",
    ):

        if U_guess.shape != V_guess.shape:
            raise ValueError("❌ U_guess and V_guess must have the same shape.")

        super().__init__(bcs, precision)
        self.shape = U_guess.shape
        self.type = U_guess.dtype
        self.U = tf.Variable(U_guess, trainable=True)
        self.V = tf.Variable(V_guess, trainable=True)
        self.input_normalizer = IdentityNormalizer()

    def get_UV_impl(self) -> Tuple[tf.Variable, tf.Variable]:
        return self.U, self.V

    def copy_w(self, w: list[tf.Variable]) -> list[tf.Tensor]:
        return [w[0].read_value(), w[1].read_value()]

    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(w_flat)

    def get_w(self) -> list[tf.Variable]:
        return [self.U, self.V]

    def set_w(self, w: list[tf.Tensor]) -> None:
        self.U.assign(w[0])
        self.V.assign(w[1])

    def flatten_w(self, w: list[tf.Variable | tf.Tensor]) -> tf.Tensor:
        u_flat = tf.reshape(w[0], [-1])
        v_flat = tf.reshape(w[1], [-1])
        return tf.concat([u_flat, v_flat], axis=0)

    def unflatten_w(self, w_flat: tf.Tensor) -> list[tf.Tensor]:
        n = tf.size(w_flat) // 2
        u_flat = w_flat[:n]
        v_flat = w_flat[n:]
        U = tf.reshape(u_flat, self.shape)
        V = tf.reshape(v_flat, self.shape)
        return [U, V]

    def check_halt_criterion(
        self, iteration: int, cost: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        halt = tf.constant(False, dtype=tf.bool)
        halt_message = tf.constant("", dtype=tf.string)
        return halt, halt_message
