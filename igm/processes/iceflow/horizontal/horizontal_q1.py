#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import math
import tensorflow as tf
from typing import Tuple
from omegaconf import DictConfig

from .horizontal import HorizontalDiscr


class Q1Discr(HorizontalDiscr):

    def _compute_discr(self, cfg: DictConfig) -> None:
        self.w_h = tf.constant([0.25, 0.25, 0.25, 0.25], self.dtype)

        gp_a = 0.5 - 0.5 / math.sqrt(3.0)
        gp_b = 0.5 + 0.5 / math.sqrt(3.0)

        self.gp_xi = tf.constant([gp_a, gp_a, gp_b, gp_b], self.dtype)
        self.gp_eta = tf.constant([gp_a, gp_b, gp_a, gp_b], self.dtype)

    @tf.function
    def grad_h(self, X: tf.Tensor, dX: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Extract corner values
        X_sw = X[..., :-1, :-1]
        X_se = X[..., :-1, 1:]
        X_nw = X[..., 1:, :-1]
        X_ne = X[..., 1:, 1:]

        # Edge derivatives
        dx_inv = 1.0 / dX[0, 1:, 1:]
        dXdx_s = (X_se - X_sw) * dx_inv
        dXdx_n = (X_ne - X_nw) * dx_inv
        dXdy_w = (X_nw - X_sw) * dx_inv
        dXdy_e = (X_ne - X_se) * dx_inv

        # Reshape for broadcasting
        ndim = len(X_sw.shape) - 1
        eta = tf.reshape(self.gp_eta, [1, 4] + [1] * ndim)
        xi = tf.reshape(self.gp_xi, [1, 4] + [1] * ndim)

        # Vectorized interpolation
        dXdx = (1.0 - eta) * dXdx_s[:, tf.newaxis, ...] + eta * dXdx_n[
            :, tf.newaxis, ...
        ]
        dXdy = (1.0 - xi) * dXdy_w[:, tf.newaxis, ...] + xi * dXdy_e[:, tf.newaxis, ...]

        return dXdx, dXdy

    @tf.function
    def interp_h(self, X: tf.Tensor) -> tf.Tensor:
        # Extract corner values
        X_sw = X[..., :-1, :-1]
        X_se = X[..., :-1, 1:]
        X_nw = X[..., 1:, :-1]
        X_ne = X[..., 1:, 1:]

        # Reshape for broadcasting
        ndim = len(X_sw.shape) - 1
        xi = tf.reshape(self.gp_xi, [1, 4] + [1] * ndim)
        eta = tf.reshape(self.gp_eta, [1, 4] + [1] * ndim)

        # Bilinear interpolation
        return (
            (1.0 - xi) * (1.0 - eta) * X_sw[:, tf.newaxis, ...]
            + xi * (1.0 - eta) * X_se[:, tf.newaxis, ...]
            + (1.0 - xi) * eta * X_nw[:, tf.newaxis, ...]
            + xi * eta * X_ne[:, tf.newaxis, ...]
        )
