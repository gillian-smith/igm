#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Optional, Tuple

from .optimizer import Optimizer
from .line_search import LineSearches, ValueAndGradient
from ..mappings import Mapping
from ..halt import Halt, HaltStatus

tf.config.optimizer.set_jit(True)


class OptimizerLBFGS(Optimizer):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        halt: Optional[Halt] = None,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_w: str = "l2_weighted",
        line_search_method: str = "armijo",
        iter_max: int = int(1e5),
        alpha_min: float = 0.0,
        memory: int = 10,
    ):
        super().__init__(
            cost_fn,
            map, 
            halt,
            print_cost,
            print_cost_freq,
            precision,
            ord_grad_u,
            ord_grad_w,
        )
        self.name = "lbfgs"

        self.line_search = LineSearches[line_search_method]()
        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.alpha_min = tf.Variable(alpha_min, dtype=self.precision)
        self.memory = memory
        if self.precision == tf.float32:
            self.eps = tf.constant(1e-12, self.precision)
        else:
            self.eps = tf.constant(1e-20, self.precision)

    def update_parameters(self, iter_max: int, alpha_min: float) -> None:
        self.iter_max.assign(iter_max)
        self.alpha_min.assign(alpha_min)

    @tf.function(reduce_retracing=True)
    def _compute_tau(self, iter: tf.Tensor) -> tf.Tensor:
        iter = tf.cast(iter, self.precision)
        return 1.0 - tf.exp(-iter / 5.0)

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        dtype = self.precision
        return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

    @tf.function(reduce_retracing=True)
    def _compute_direction(
        self,
        grad: tf.Tensor,
        s_list: tf.Tensor,
        y_list: tf.Tensor,
        num_elems: tf.Tensor,
        tau: tf.Tensor,
    ) -> tf.Tensor:
        # Naive gradient descent
        if tf.equal(num_elems, 0):
            return -grad

        q = grad
        alpha_list = tf.TensorArray(
            dtype=grad.dtype, size=num_elems, dynamic_size=False
        )

        # First loop
        for i in tf.range(num_elems - 1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]
            rho = 1.0 / (self._dot(y_i, s_i) + self.eps)
            alpha_i = rho * self._dot(s_i, q)
            alpha_i = tf.cast(alpha_i, q.dtype)
            alpha_list = alpha_list.write(i, alpha_i)
            q = q - alpha_i * y_i

        last_y = y_list[num_elems - 1]
        last_s = s_list[num_elems - 1]
        gamma = self._dot(last_y, last_s) / (self._dot(last_y, last_y) + self.eps)
        gamma = tf.cast(gamma, q.dtype)

        # Tempering
        r = tau * gamma * q

        # Second loop
        for i in tf.range(num_elems):
            s_i = s_list[i]
            y_i = y_list[i]
            rho = 1.0 / (self._dot(y_i, s_i) + self.eps)
            beta = rho * self._dot(y_i, r)
            beta = tf.cast(beta, r.dtype)
            alpha_i = alpha_list.read(i)
            r = r + s_i * (alpha_i - beta)

        return -r

    def _force_descent(
        self, p_flat: tf.Tensor, grad_w_flat: tf.Tensor, _: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        dot_gp = self._dot(grad_w_flat, p_flat)
        return tf.cond(dot_gp >= 0.0, lambda: -grad_w_flat, lambda: p_flat), None

    def _apply_step(
        self, w_flat: tf.Tensor, alpha: tf.Tensor, p_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return w_flat + alpha * p_flat, None

    def _constrain_pair(
        self,
        s: tf.Tensor,
        y: tf.Tensor,
        w_old: tf.Tensor,
        w_trial: Optional[tf.Tensor],
        mask: Optional[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return s, y

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

        return tf.cond(
            dot_ys > self.eps, update, lambda: (s_flat_mem, y_flat_mem, idx_memory)
        )

    @tf.function
    def _line_search(
        self, w_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor
    ) -> tf.Tensor:
        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            w_backup = self.map.copy_w(self.map.get_w())
            w_alpha, _ = self._apply_step(w_flat, alpha, p_flat)

            self.map.set_w(self.map.unflatten_w(w_alpha))
            f, _, grad = self._get_grad(input)
            grad_flat = self.map.flatten_w(grad)
            df = self._dot(grad_flat, p_flat)
            df = tf.cast(df, grad_flat.dtype)

            self.map.set_w(w_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(w_flat, p_flat, eval_fn)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        first_batch = self.sampler(inputs)  # [M, B, H, W, C]
        n_batches = first_batch.shape[0]
        if n_batches != 1:
            raise NotImplementedError("❌ L-BFGS requires a single batch.")
        
        if getattr(self.sampler, "dynamic_augmentation", False):
            static_batches = None
            dynamic_augmentation = True
        else:
            # Sampler does not change data between calls → build once and reuse.
            static_batches = first_batch
            dynamic_augmentation = False

        input = first_batch[0, :, :, :, :]  # Define before loop for AutoGraph

        # State variables
        w_flat = self.map.flatten_w(self.map.get_w())
        cost, grad_u, grad_w = self._get_grad(input)
        grad_w_flat = self.map.flatten_w(grad_w)
        U, V = self.map.get_UV(input)
        self._init_step_state(U, V, w_flat)

        # Memory variables
        w_dim = tf.shape(w_flat)[0]
        idx_memory = tf.constant(0, dtype=tf.int32)
        s_flat_mem = tf.zeros([self.memory, w_dim], dtype=w_flat.dtype)
        y_flat_mem = tf.zeros([self.memory, w_dim], dtype=w_flat.dtype)

        # Accessory variables
        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            # Sample fresh augmented batch for this iteration
            if dynamic_augmentation:
                next_batch = self.sampler(inputs)  # [M, B, H, W, C]
            else:
                next_batch = static_batches       # [M, B, H, W, C]

            input = next_batch[0, :, :, :, :]

            w_prev = w_flat
            grad_w_prev = grad_w_flat

            # Tempering
            tau = self._compute_tau(iter)

            # Direction
            p_flat = self._compute_direction(
                grad_w_flat,
                s_flat_mem[:idx_memory],
                y_flat_mem[:idx_memory],
                idx_memory,
                tau,
            )

            # Force descent
            p_flat, mask = self._force_descent(p_flat, grad_w_flat, w_flat)

            # Line search
            alpha = self._line_search(w_flat, p_flat, input)
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            # Apply step
            w_flat, w_trial = self._apply_step(w_flat, alpha, p_flat)

            # New weights, cost, and grads
            self.map.set_w(self.map.unflatten_w(w_flat))
            cost, grad_u, grad_w = self._get_grad(input)
            grad_w_flat = self.map.flatten_w(grad_w)

            # Curvature pair
            s, y = w_flat - w_prev, grad_w_flat - grad_w_prev
            s, y = self._constrain_pair(s, y, w_prev, w_trial, mask)

            # Update memory
            s_flat_mem, y_flat_mem, idx_memory = self._update_memory(
                s_flat_mem, y_flat_mem, idx_memory, s, y
            )

            # TODO: check if this is necessary
            self.map.on_step_end(iter)

            costs = costs.write(iter, cost)

            U, V = self.map.get_UV(input)
            grad_u_norm, grad_w_norm = self._get_grad_norm(grad_u, grad_w)

            self._update_step_state(iter, U, V, w_flat, cost, grad_u_norm, grad_w_norm)
            halt_status = self._check_stopping()
            self._update_display()

            iter_last = iter

            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break

        self._finalize_display(halt_status)
        return costs.stack()[: iter_last + 1]
