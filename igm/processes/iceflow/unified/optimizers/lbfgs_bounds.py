#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Optional, Tuple

from .lbfgs import OptimizerLBFGS
from .line_searches import LineSearches, ValueAndGradient

tf.config.optimizer.set_jit(True)


class OptimizerLBFGSBounds(OptimizerLBFGS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "lbfgs_bounded"

        if not hasattr(self.map, "get_box_bounds_flat"):
            raise ValueError(
                "❌ Mapping must provide get_box_bounds_flat() for bounded optimization."
            )

    @tf.function(reduce_retracing=True)
    def _project(self, theta: tf.Tensor, L: tf.Tensor, U: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(theta, L, U)

    @tf.function(reduce_retracing=True)
    def _get_mask(
        self, w: tf.Tensor, g: tf.Tensor, L: tf.Tensor, U: tf.Tensor
    ) -> tf.Tensor:
        eps = tf.cast(self.eps, w.dtype)
        interior = tf.logical_and(w > L + eps, w < U - eps)
        at_lower = tf.logical_and(w <= L + eps, g > 0.0)
        at_upper = tf.logical_and(w >= U - eps, g < 0.0)
        return tf.logical_or(interior, tf.logical_or(at_lower, at_upper))

    def _force_descent(
        self, p_flat: tf.Tensor, grad_theta_flat: tf.Tensor, theta_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        L, U = self.map.get_box_bounds_flat()
        mask = self._get_mask(theta_flat, grad_theta_flat, L, U)
        p_flat = tf.where(mask, p_flat, tf.zeros_like(p_flat))
        dot_gp = self._dot(grad_theta_flat, p_flat)
        return tf.cond(dot_gp >= 0.0, lambda: -grad_theta_flat, lambda: p_flat), mask

    def _apply_step(
        self, theta_flat: tf.Tensor, alpha: tf.Tensor, p_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        L, U = self.map.get_box_bounds_flat()
        theta_trial = theta_flat + alpha * p_flat
        theta_proj = self._project(theta_trial, L, U)
        return theta_proj, theta_trial

    def _constrain_pair(
        self,
        s: tf.Tensor,
        y: tf.Tensor,
        w_prev: tf.Tensor,
        theta_trial: Optional[tf.Tensor],
        mask: Optional[tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor]:
        theta_flat = w_prev + s
        proj_changed = tf.not_equal(tf.abs(theta_flat - theta_trial), 0.0)

        s = tf.where(
            tf.logical_or(proj_changed, tf.logical_not(mask)), tf.zeros_like(s), s
        )
        y = tf.where(
            tf.logical_or(proj_changed, tf.logical_not(mask)), tf.zeros_like(y), y
        )

        return s, y

    @tf.function
    def _line_search(
        self, theta_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor
    ) -> tf.Tensor:
        L, U = self.map.get_box_bounds_flat()

        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            theta_backup = self.map.copy_theta(self.map.get_theta())
            theta_alpha, _ = self._apply_step(theta_flat, alpha, p_flat)

            self.map.set_theta(self.map.unflatten_theta(theta_alpha))
            f, _, grad = self._get_grad(input)
            grad_flat = self.map.flatten_theta(grad)

            mask = self._get_mask(theta_alpha, grad_flat, L, U)
            p_masked = tf.where(mask, p_flat, tf.zeros_like(p_flat))
            df = self._dot(grad_flat, p_masked)
            df = tf.cast(df, grad_flat.dtype)

            self.map.set_theta(theta_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(theta_flat, p_flat, eval_fn)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:

        L, U = self.map.get_box_bounds_flat()
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        theta_proj = self._project(theta_flat, L, U)
        self.map.set_theta(self.map.unflatten_theta(theta_proj))

        return super().minimize_impl(inputs)
