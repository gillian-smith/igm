#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple
from omegaconf import DictConfig

from .horizontal import HorizontalDiscr


class P1Discr(HorizontalDiscr):

    def _compute_discr(self, cfg: DictConfig) -> None:
        self.w_h = tf.constant([0.5, 0.5], self.dtype)
        self._one_third = tf.constant(1.0 / 3.0, self.dtype)

    @tf.function
    def grad_h(self, X: tf.Tensor, dX: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Extract corner values
        X_sw = X[..., :-1, :-1]
        X_se = X[..., :-1, 1:]
        X_nw = X[..., 1:, :-1]
        X_ne = X[..., 1:, 1:]

        dx_inv = 1.0 / dX[0, 1:, 1:]

        # T1 (lower triangle: SW-SE-NE)
        dXdx_T1 = (X_se - X_sw) * dx_inv
        dXdy_T1 = (X_ne - X_se) * dx_inv

        # T2 (upper triangle: SW-NW-NE)
        dXdx_T2 = (X_ne - X_nw) * dx_inv
        dXdy_T2 = (X_nw - X_sw) * dx_inv

        # Stack at axis 1 (after batch)
        dXdx = tf.stack([dXdx_T1, dXdx_T2], axis=1)
        dXdy = tf.stack([dXdy_T1, dXdy_T2], axis=1)

        return dXdx, dXdy

    @tf.function
    def interp_h(self, X: tf.Tensor) -> tf.Tensor:
        # Extract corner values
        X_sw = X[..., :-1, :-1]
        X_se = X[..., :-1, 1:]
        X_nw = X[..., 1:, :-1]
        X_ne = X[..., 1:, 1:]

        # Centroids (average of 3 vertices)
        X_T1 = (X_sw + X_se + X_ne) * self._one_third
        X_T2 = (X_sw + X_nw + X_ne) * self._one_third

        # Stack at axis 1 (after batch)
        return tf.stack([X_T1, X_T2], axis=1)
