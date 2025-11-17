#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple, Union

TV = Union[tf.Tensor, tf.Variable]


class BoundaryCondition(ABC):
    def __init__(self, V_b: tf.Tensor):
        self.V_b = V_b

    def __call__(self, U: TV, V: TV) -> Tuple[TV, TV]:
        return self.apply(U, V)

    @abstractmethod
    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        return U, V


class FrozenBed(BoundaryCondition):
    def __init__(self, V_b: tf.Tensor):
        super().__init__(V_b)
        if self.V_b[0] == 0:
            raise ValueError(f"❌ The frozen bed BC requires V_b ≠ 0.")
        self.weights = -self.V_b[1:] / self.V_b[0]

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        U0 = tf.einsum("i,bijk->bjk", self.weights, U[:, 1:, :, :])
        V0 = tf.einsum("i,bijk->bjk", self.weights, V[:, 1:, :, :])

        U0 = tf.expand_dims(U0, axis=1)
        V0 = tf.expand_dims(V0, axis=1)

        U = tf.concat([U0, U[:, 1:, :, :]], axis=1)
        V = tf.concat([V0, V[:, 1:, :, :]], axis=1)

        return U, V


class PeriodicNS(BoundaryCondition):
    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        U = tf.concat([U[:, :, :-1, :], U[:, :, :1, :]], axis=2)
        V = tf.concat([V[:, :, :-1, :], V[:, :, :1, :]], axis=2)
        return U, V


class PeriodicWE(BoundaryCondition):
    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        U = tf.concat([U[:, :, :, :-1], U[:, :, :, :1]], axis=3)
        V = tf.concat([V[:, :, :, :-1], V[:, :, :, :1]], axis=3)
        return U, V
