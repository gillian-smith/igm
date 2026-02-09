#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import tensorflow as tf
from typing import Tuple, List, Any

from .lbfgs_bounds import OptimizerLBFGSBounds


class OptimizerLBFGSBoundsDA(OptimizerLBFGSBounds):
    """
    Bounded L-BFGS for data assimilation.
    Here we always optimize total_cost but keep (data, reg) for logging.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robust_curvature", True)
        super().__init__(*args, **kwargs)

        # Diagnostics (TF variables so they can be assigned inside tf.function)
        dtype = getattr(self.map, "precision", tf.float32)
        self.last_total = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_data  = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_reg   = tf.Variable(0.0, trainable=False, dtype=dtype)

    @tf.function(reduce_retracing=True)
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        theta = self.map.get_theta()

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            for t in theta:
                tape.watch(t)

            # Forward pass (mapping may patch/synchronize inputs internally)
            U, V = self.map.get_UV(inputs)

            # IMPORTANT: cost must see the exact inputs used by the mapping/network
            inputs_used = self.map.inputs if hasattr(self.map, "inputs") else inputs

            total, data, reg = self.cost_fn(U, V, inputs_used)

        # Grads
        grad_u = tape.gradient(total, [U, V])
        grad_theta = tape.gradient(total, theta)
        del tape

        # Replace None grads 
        grad_theta = [
            tf.zeros_like(t) if g is None else g
            for g, t in zip(grad_theta, theta)
        ]

        # Store diagnostics
        self.last_total.assign(tf.stop_gradient(total))
        self.last_data.assign(tf.stop_gradient(data))
        self.last_reg.assign(tf.stop_gradient(reg))

        return total, grad_u, grad_theta
