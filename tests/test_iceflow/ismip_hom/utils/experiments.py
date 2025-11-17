#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

from igm.common.core import State


class Experiment(ABC):
    CONST_FIELDS = {"x", "y", "X", "Y"}

    @abstractmethod
    def init_fields(self, L: float, n: int) -> Dict[str, np.ndarray]:
        pass

    @classmethod
    def init_state(
        cls, state: State, L: float, n: int, dtype: tf.DType = tf.float32
    ) -> None:
        fields_np = cls().init_fields(L, n)

        for field, value in fields_np.items():
            tensor = (
                tf.constant(value, dtype=dtype)
                if field in cls.CONST_FIELDS
                else tf.Variable(value, dtype=dtype)
            )
            setattr(state, field, tensor)


class ExperimentA(Experiment):
    def init_fields(self, L: float, n: int) -> Dict[str, np.ndarray]:
        nx, ny = n, n
        x = np.linspace(0.0, L, nx)
        y = np.linspace(0.0, L, ny)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        α_x = np.deg2rad(0.5)
        ω = 2 * np.pi / L
        z_s = -X * np.tan(α_x)
        h = 1000.0 - 500.0 * np.sin(ω * X) * np.sin(ω * Y)
        z_b = z_s - h
        C = 1.0 * np.ones_like(X)
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": C,
            "arrhenius": A,
        }


class ExperimentB(Experiment):
    def init_fields(self, L: float, n: int) -> Dict[str, np.ndarray]:
        nx = n
        x = np.linspace(0.0, L, nx)
        dx = x[1] - x[0]
        y = np.array([0.0, dx])
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        α_x = np.deg2rad(0.5)
        ω = 2 * np.pi / L
        z_s = -X * np.tan(α_x)
        h = 1000.0 - 500.0 * np.sin(ω * X)
        z_b = z_s - h
        C = 1.0 * np.ones_like(X)
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": C,
            "arrhenius": A,
        }
