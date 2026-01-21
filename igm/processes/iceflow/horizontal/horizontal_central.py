#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple
from omegaconf import DictConfig

from .horizontal import HorizontalDiscr
from igm.utils.stag.stag import stag4h
from igm.utils.grad.grad import grad_xy


class CentralDiscr(HorizontalDiscr):

    def _compute_discr(self, cfg: DictConfig) -> None:
        self.w_h = tf.constant([1.0], self.dtype)

    @tf.function
    def grad_h(self, X: tf.Tensor, dX: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        dUdx, dUdy = grad_xy(X, dX, dX, staggered_grid=True)

        return dUdx[..., tf.newaxis, :, :, :], dUdy[..., tf.newaxis, :, :, :]

    @tf.function
    def interp_h(self, X: tf.Tensor) -> tf.Tensor:
        return stag4h(X)[..., tf.newaxis, :, :, :]
