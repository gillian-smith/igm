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
    I've also added a lot of additional safeguards and stricter criteria for updating lbfgs memory to try and help hard problems
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Diagnostics (TF variables so they can be assigned inside tf.function)
        dtype = getattr(self.map, "precision", tf.float32)
        self.last_total = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_data  = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_reg   = tf.Variable(0.0, trainable=False, dtype=dtype)

        # --- running stats of y^T s (accepted pairs only) ---
        self.ys_mean  = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.ys_count = tf.Variable(0,   trainable=False, dtype=tf.int64)

        # how many accepted pairs before we start capping rho
        self.ys_warmup = tf.constant(5, dtype=tf.int64)

        # allow rho up to (rho_spike_factor * typical_rho)
        # typical_rho ≈ 1 / ys_mean
        self.rho_spike_factor = tf.constant(100.0, dtype=dtype)

        # optional counters if you still want acceptance-rate prints in DA
        self.mem_accept = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.mem_total  = tf.Variable(0, trainable=False, dtype=tf.int64)

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

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        # reset stats each run
        self.ys_mean.assign(tf.cast(0.0, self.ys_mean.dtype))
        self.ys_count.assign(0)
        self.mem_accept.assign(0)
        self.mem_total.assign(0)
        return super().minimize(inputs)
    
    @tf.function(reduce_retracing=True)
    def _rho_cap(self) -> tf.Tensor:
        # cap = spike_factor / ys_mean  (since rho = 1 / (y^T s))
        mean = tf.maximum(self.ys_mean, tf.cast(self.eps, self.ys_mean.dtype))
        cap = tf.cast(self.rho_spike_factor, mean.dtype) / mean

        inf = tf.constant(float("inf"), dtype=cap.dtype)
        return tf.cond(self.ys_count >= self.ys_warmup, lambda: cap, lambda: inf)
    
    @tf.function(reduce_retracing=True)
    def _update_memory(
        self,
        s_flat_mem: tf.Tensor,
        y_flat_mem: tf.Tensor,
        idx_memory: tf.Tensor,
        s: tf.Tensor,
        y: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        dot_ys = self._dot(y, s)
        accept = tf.math.is_finite(dot_ys) & (dot_ys > self.eps)

        self.mem_total.assign_add(1)
        self.mem_accept.assign_add(tf.cast(accept, tf.int64))

        # update running mean of y^T s (only when accepted)
        def _update_stats():
            c = self.ys_count
            c_new = c + 1

            # update mean in float64 for numerical stability, store back in dtype
            mean64 = tf.cast(self.ys_mean, tf.float64)
            c64    = tf.cast(c, tf.float64)
            ys64   = tf.cast(dot_ys, tf.float64)

            new_mean64 = (mean64 * c64 + ys64) / (c64 + 1.0)

            self.ys_mean.assign(tf.cast(new_mean64, self.ys_mean.dtype))
            self.ys_count.assign(c_new)
            return tf.constant(0, tf.int32)

        tf.cond(accept, _update_stats, lambda: tf.constant(0, tf.int32))
        tf.print("ys_mean", self.ys_mean, "rho_cap", self._rho_cap(), "accept", accept)

        # store memory (same as base)
        def update():
            def append():
                return (
                    tf.tensor_scatter_nd_update(s_flat_mem, [[idx_memory]], [s]),
                    tf.tensor_scatter_nd_update(y_flat_mem, [[idx_memory]], [y]),
                    idx_memory + 1,
                )

            def shift():
                return (
                    tf.concat([s_flat_mem[1:], [s]], axis=0),
                    tf.concat([y_flat_mem[1:], [y]], axis=0),
                    idx_memory,
                )

            return tf.cond(idx_memory < self.memory, append, shift)

        return tf.cond(accept, update, lambda: (s_flat_mem, y_flat_mem, idx_memory))

    @tf.function(reduce_retracing=True)
    def _compute_direction(
        self,
        grad: tf.Tensor,
        s_list: tf.Tensor,
        y_list: tf.Tensor,
        num_elems: tf.Tensor,
        tau: tf.Tensor,
    ) -> tf.Tensor:
        if tf.equal(num_elems, 0):
            return -grad

        rho_cap = self._rho_cap()  

        q = grad
        alpha_list = tf.TensorArray(dtype=grad.dtype, size=num_elems, dynamic_size=False)

        for i in tf.range(num_elems - 1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]

            rho = 1.0 / (self._dot(y_i, s_i) + self.eps)
            rho = tf.minimum(rho, tf.cast(rho_cap, rho.dtype))

            alpha_i = rho * self._dot(s_i, q)
            alpha_i = tf.cast(alpha_i, q.dtype)
            alpha_list = alpha_list.write(i, alpha_i)
            q = q - alpha_i * y_i

        last_y = y_list[num_elems - 1]
        last_s = s_list[num_elems - 1]
        gamma = self._dot(last_y, last_s) / (self._dot(last_y, last_y) + self.eps)

        gamma = tf.where(tf.math.is_finite(gamma), gamma, tf.constant(1.0, gamma.dtype))
        gamma = tf.clip_by_value(gamma, tf.constant(1e-6, gamma.dtype), tf.constant(1e6, gamma.dtype))
        gamma = tf.cast(gamma, q.dtype)

        r = tau * gamma * q

        for i in tf.range(num_elems):
            s_i = s_list[i]
            y_i = y_list[i]

            rho = 1.0 / (self._dot(y_i, s_i) + self.eps)
            rho = tf.minimum(rho, tf.cast(rho_cap, rho.dtype))  

            beta = rho * self._dot(y_i, r)
            beta = tf.cast(beta, r.dtype)
            alpha_i = alpha_list.read(i)
            r = r + s_i * (alpha_i - beta)

        return -r

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        acc = tf.tensordot(tf.cast(a, tf.float64), tf.cast(b, tf.float64), axes=1)
        return tf.cast(acc, self.precision)