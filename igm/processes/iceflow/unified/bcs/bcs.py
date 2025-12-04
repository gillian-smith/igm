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
        self.V_b = V_b  # currently saves state in full - change on monday

    def __call__(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply boundary condition to velocity components (callable interface)."""
        return self.apply(U, V)

    @abstractmethod
    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply boundary condition to velocity components (must be implemented by subclasses)."""
        pass


# class FrozenBed(BoundaryCondition):
#     # ! Should be able to do this with an ansatz as its dichelet - for periodic BCs its harder so we can leave them here...
#     def __init__(self, V_b: tf.Tensor):
#         super().__init__(V_b)
#         if self.V_b[0] == 0:
#             raise ValueError(f"❌ The frozen bed BC requires V_b ≠ 0.")
#         self.weights = -self.V_b[1:] / self.V_b[0]

#     def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
#         U0 = tf.einsum("i,bijk->bjk", self.weights, U[:, 1:, :, :])
#         V0 = tf.einsum("i,bijk->bjk", self.weights, V[:, 1:, :, :])

#         U0 = tf.expand_dims(U0, axis=1)
#         V0 = tf.expand_dims(V0, axis=1)

#         U = tf.concat([U0, U[:, 1:, :, :]], axis=1)
#         V = tf.concat([V0, V[:, 1:, :, :]], axis=1)

#         return U, V


class FrozenBed(BoundaryCondition):
    """Temporary changing this to avoid an error but will change back or move to an ansatz"""

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        U0 = tf.zeros_like(U[:, 0, ...])
        V0 = tf.zeros_like(V[:, 0, ...])

        U0 = U[:, 1:, ...]
        V0 = U[:, 1:, ...]

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


class PeriodicNSGlobal(BoundaryCondition):
    def __init__(self, state):
        Nx = state.thk.shape[1]
        Ny = state.thk.shape[0]
        Nz = state.Nz
        self.original_shape = (Nz, Ny, Nx)

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        current_shape = U.shape

        # ! Assumes reshaping with correct order (and noneven patching...) - check with Seb as this is probably wrong!
        U = tf.reshape(
            U, [1, *self.original_shape]
        )  # ! maybe make it not reshape if its already batch size of 1
        V = tf.reshape(V, [1, *self.original_shape])

        U = tf.concat([U[:, :, :-1, :], U[:, :, :1, :]], axis=2)
        V = tf.concat([V[:, :, :-1, :], V[:, :, :1, :]], axis=2)

        U = tf.reshape(U, current_shape)
        V = tf.reshape(V, current_shape)

        return U, V


class PeriodicWEGlobal(BoundaryCondition):
    def __init__(self, state):
        Nx = state.thk.shape[1]
        Ny = state.thk.shape[0]
        Nz = state.Nz
        self.original_shape = (Nz, Ny, Nx)

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        current_shape = U.shape

        # ! Assumes reshaping with correct order (and noneven patching...) - check with Seb as this is probably wrong!
        U = tf.reshape(U, [1, *self.original_shape])
        V = tf.reshape(V, [1, *self.original_shape])

        U = tf.concat([U[:, :, :, :-1], U[:, :, :, :1]], axis=3)
        V = tf.concat([V[:, :, :, :-1], V[:, :, :, :1]], axis=3)

        U = tf.reshape(U, current_shape)
        V = tf.reshape(V, current_shape)

        return U, V


class PeriodicWE(BoundaryCondition):
    """Periodic boundary condition in west-east direction."""

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply periodic boundary condition in west-east direction."""
        U = tf.concat([U[:, :, :, :-1], U[:, :, :, :1]], axis=3)
        V = tf.concat([V[:, :, :, :-1], V[:, :, :, :1]], axis=3)
        return U, V
