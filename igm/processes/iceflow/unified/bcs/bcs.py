#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple, Union

TV = Union[tf.Tensor, tf.Variable]


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions on velocity fields."""

    def __init__(self, V_b: tf.Tensor):
        """Initialize boundary condition with basal basis values."""
        self.V_b = V_b

    def __call__(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply boundary condition to velocity components (callable interface)."""
        return self.apply(U, V)

    @abstractmethod
    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply boundary condition to velocity components (must be implemented by subclasses)."""
        pass


class FrozenBed(BoundaryCondition):
    """Frozen bed boundary condition enforcing zero basal velocity."""

    def __init__(self, V_b: tf.Tensor):
        """Initialize weights to enforce zero basal velocity."""
        super().__init__(V_b)
        if self.V_b[0] == 0:
            raise ValueError(f"❌ The frozen bed BC requires V_b ≠ 0.")
        self.weights = -self.V_b[1:] / self.V_b[0]

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply frozen bed condition by enforcing zero basal velocity."""
        U0 = tf.einsum("i,bijk->bjk", self.weights, U[:, 1:, :, :])
        V0 = tf.einsum("i,bijk->bjk", self.weights, V[:, 1:, :, :])

        U0 = tf.expand_dims(U0, axis=1)
        V0 = tf.expand_dims(V0, axis=1)

        U = tf.concat([U0, U[:, 1:, :, :]], axis=1)
        V = tf.concat([V0, V[:, 1:, :, :]], axis=1)

        return U, V


class PeriodicNS(BoundaryCondition):
    """Periodic boundary condition in north-south direction."""

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply periodic boundary condition in north-south direction."""
        U = tf.concat([U[:, :, :-1, :], U[:, :, :1, :]], axis=2)
        V = tf.concat([V[:, :, :-1, :], V[:, :, :1, :]], axis=2)
        return U, V


class PeriodicWE(BoundaryCondition):
    """Periodic boundary condition in west-east direction."""

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply periodic boundary condition in west-east direction."""
        U = tf.concat([U[:, :, :, :-1], U[:, :, :, :1]], axis=3)
        V = tf.concat([V[:, :, :, :-1], V[:, :, :, :1]], axis=3)
        return U, V
