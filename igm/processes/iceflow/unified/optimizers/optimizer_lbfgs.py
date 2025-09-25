#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable

from ..mappings import Mapping
from .optimizer import Optimizer
from .line_search import LineSearches, ValueAndGradient

tf.config.optimizer.set_jit(True)


class OptimizerLBFGS(Optimizer):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        print_cost: bool,
        print_cost_freq: int,
        line_search_method: str,
        iter_max: int = int(1e5),
        alpha_min: float = 0.0,
        memory: int = 10,
    ):
        super().__init__(cost_fn, map, print_cost, print_cost_freq)
        self.name = "lbfgs"

        self.line_search = LineSearches[line_search_method]()
        self.iter_max = tf.Variable(iter_max)
        self.alpha_min = tf.Variable(alpha_min)
        self.memory = memory

    def update_parameters(self, iter_max: int, alpha_min: float) -> None:
        self.iter_max.assign(iter_max)
        self.alpha_min.assign(alpha_min)

    @tf.function(reduce_retracing=True)
    def _two_loop_recursion(
        self, grad: tf.Tensor, s_list: tf.Tensor, y_list: tf.Tensor
    ) -> tf.Tensor:
        q = grad
        alpha_list = tf.TensorArray(dtype=grad.dtype, size=0, dynamic_size=True)
        num_elems = tf.shape(s_list)[0]

        # First loop
        for i in tf.range(num_elems - 1, -1, -1):
            s = s_list[i]
            y = y_list[i]
            rho = 1.0 / tf.tensordot(y, s, axes=1)
            alpha = rho * tf.tensordot(s, q, axes=1)
            alpha_list = alpha_list.write(i, alpha)
            q = q - alpha * y

        def compute_gamma_fn() -> tf.Tensor:
            last_y = y_list[num_elems - 1]
            last_s = s_list[num_elems - 1]
            ys = tf.tensordot(last_y, last_s, axes=1)
            yy = tf.tensordot(last_y, last_y, axes=1)
            return ys / yy

        gamma = tf.cond(
            num_elems > 0, compute_gamma_fn, lambda: tf.constant(1.0, dtype=grad.dtype)
        )

        r = gamma * q

        # Second loop
        for i in tf.range(num_elems):
            s = s_list[i]
            y = y_list[i]
            alpha = alpha_list.read(i)
            rho = 1.0 / tf.tensordot(y, s, axes=1)
            beta = rho * tf.tensordot(y, r, axes=1)
            r = r + s * (alpha - beta)

        return -r

    @tf.function
    def _line_search(
        self, w_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor
    ) -> tf.Tensor:
        def value_and_gradients_function(alpha: tf.Tensor) -> ValueAndGradient:
            # Backup
            w_backup = self.map.copy_w(self.map.get_w())

            # New w
            w_alpha = w_flat + alpha * p_flat
            w_alpha = self.map.unflatten_w(w_alpha)

            # Compute grad
            self.map.set_w(w_alpha)
            f, _, grad = self._get_grad(input)
            grad_flat = self.map.flatten_w(grad)
            df = tf.reduce_sum(grad_flat * p_flat)

            # Reset backup
            self.map.set_w(w_backup)

            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(w_flat, p_flat, value_and_gradients_function)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:

        n_batches = inputs.shape[0]
        if n_batches > 1:
            raise NotImplementedError(
                "❌ Multiple batches is not compatible with the LBFGS optimizer, "
                + "check data preparation settings and ensure everything fits into one batch "
                + f"(n_batches = {n_batches})."
            )
        input = inputs[0, :, :, :, :]

        # Initial state
        w_flat = self.map.flatten_w(self.map.get_w())
        U, V = self.map.get_UV(inputs[0, :, :, :])

        # Evaluate at initial point
        cost, grad_u, grad_w = self._get_grad(input)
        grad_w_flat = self.map.flatten_w(grad_w)

        # Pre-allocate structures related to memory handling
        w_dim = tf.shape(w_flat)[0]
        idx_memory = tf.constant(0, dtype=w_dim.dtype)
        s = tf.zeros([w_dim], dtype=w_flat.dtype)
        y = tf.zeros([w_dim], dtype=w_flat.dtype)
        s_flat = tf.zeros([self.memory, w_dim], dtype=w_flat.dtype)
        y_flat = tf.zeros([self.memory, w_dim], dtype=w_flat.dtype)

        # Pre-allocate costs
        costs = tf.TensorArray(dtype=w_flat.dtype, size=self.iter_max)

        for iter in tf.range(self.iter_max):
            # Save previous solution
            U_prev = tf.identity(U)
            V_prev = tf.identity(V)
            w_flat_prev = w_flat
            grad_w_flat_prev = grad_w_flat

            # Compute direction
            p_flat = tf.cond(
                idx_memory > 0,
                lambda: self._two_loop_recursion(
                    grad_w_flat, s_flat[:idx_memory], y_flat[:idx_memory]
                ),
                lambda: -grad_w_flat,
            )

            # Line search
            alpha = self._line_search(w_flat, p_flat, input)
            alpha = tf.maximum(alpha, self.alpha_min)

            # Apply increment
            w_flat += alpha * p_flat
            self.map.set_w(self.map.unflatten_w(w_flat))

            # Evaluate at new point
            cost, grad_u, grad_w = self._get_grad(input)
            grad_w_flat = self.map.flatten_w(grad_w)

            # Update history
            s = w_flat - w_flat_prev
            y = grad_w_flat - grad_w_flat_prev

            cond_update = tf.tensordot(y, s, 1) > 1e-10

            def memory_append():
                return (
                    tf.tensor_scatter_nd_update(s_flat, [[idx_memory]], [s]),
                    tf.tensor_scatter_nd_update(y_flat, [[idx_memory]], [y]),
                    idx_memory + 1,
                )

            def memory_circ_add():
                return (
                    tf.concat([s_flat[1:], tf.expand_dims(s, 0)], axis=0),
                    tf.concat([y_flat[1:], tf.expand_dims(y, 0)], axis=0),
                    idx_memory,
                )

            s_flat, y_flat, idx_memory = tf.cond(
                cond_update,
                lambda: tf.cond(
                    idx_memory < self.memory,
                    memory_append,
                    memory_circ_add,
                ),
                lambda: (s_flat, y_flat, idx_memory),
            )

            # Post-process
            U, V = self.map.get_UV(input)

            self._progress_update(iter, cost)

            costs = costs.write(iter, cost)

        return costs.stack()
