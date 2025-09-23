#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple, Union

TV = Union[tf.Tensor, tf.Variable]


class BoundaryCondition(ABC):
    def __call__(self, U: TV, V: TV) -> Tuple[TV, TV]:
        return self.apply(U, V)

    @abstractmethod
    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        return U, V


class FrozenBed(BoundaryCondition):
    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        U = tf.concat([tf.zeros_like(U[:, :1, :, :]), U[:, 1:, :, :]], axis=1)
        V = tf.concat([tf.zeros_like(V[:, :1, :, :]), V[:, 1:, :, :]], axis=1)
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
