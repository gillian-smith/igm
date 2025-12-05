#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# GNU GPL v3

import tensorflow as tf
from typing import Callable, Optional

from .optimizer import Optimizer
from .line_searches import LineSearches, ValueAndGradient
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

        if memory < 1:
            raise ValueError("L-BFGS memory must be at least 1")

        self.dtype = tf.float32 if precision == "float32" else tf.float64
        self.line_search = LineSearches[line_search_method]()
        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.alpha_min = tf.Variable(alpha_min, dtype=self.dtype)
        self.memory = memory
        self.eps = tf.constant(
            1e-12 if precision == "float32" else 1e-20, dtype=self.dtype
        )

    def update_parameters(self, iter_max: int, alpha_min: float) -> None:
        self.iter_max.assign(iter_max)
        self.alpha_min.assign(alpha_min)

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(tf.cast(a, self.dtype), tf.cast(b, self.dtype), axes=1)

    @tf.function(reduce_retracing=True)
    def _compute_direction(
        self,
        grad: tf.Tensor,
        s_mem: tf.Tensor,
        y_mem: tf.Tensor,
        n: tf.Tensor,
        tau: tf.Tensor,
    ) -> tf.Tensor:
        if tf.equal(n, 0):
            return -grad

        q = grad
        alphas = tf.TensorArray(dtype=grad.dtype, size=n, dynamic_size=False)

        for i in tf.range(n - 1, -1, -1):
            rho_hp = tf.cast(1.0, self.dtype) / (
                self._dot(y_mem[i], s_mem[i]) + self.eps
            )
            alpha_hp = rho_hp * self._dot(s_mem[i], q)
            alpha = tf.cast(alpha_hp, q.dtype)
            alphas = alphas.write(i, alpha)
            q = q - alpha * y_mem[i]

        gamma_hp = self._dot(y_mem[n - 1], s_mem[n - 1]) / (
            self._dot(y_mem[n - 1], y_mem[n - 1]) + self.eps
        )
        gamma = tf.cast(gamma_hp, q.dtype)
        r = tau * gamma * q

        for i in tf.range(n):
            rho_hp = tf.cast(1.0, self.dtype) / (
                self._dot(y_mem[i], s_mem[i]) + self.eps
            )
            beta_hp = rho_hp * self._dot(y_mem[i], r)
            beta = tf.cast(beta_hp, r.dtype)
            alpha = alphas.read(i)
            r = r + s_mem[i] * (alpha - beta)

        return -r

    def _prepare_search_direction(
        self, p: tf.Tensor, g: tf.Tensor, w: tf.Tensor
    ) -> tf.Tensor:
        gp = self._dot(g, p)
        return tf.cond(gp >= 0.0, lambda: -g, lambda: p)

    def _perform_step(self, w: tf.Tensor, alpha: tf.Tensor, p: tf.Tensor) -> tf.Tensor:
        return w + alpha * p

    def _mask_curvature_pair(
        self, s: tf.Tensor, y: tf.Tensor, w_old: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        return s, y

    @tf.function
    def _line_search_step(
        self, w: tf.Tensor, p: tf.Tensor, input: tf.Tensor
    ) -> tf.Tensor:
        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            w_backup = self.map.copy_w(self.map.get_w())
            w_new = self._perform_step(w, alpha, p)

            self.map.set_w(self.map.unflatten_w(w_new))
            f, _, grad = self._get_grad(input)
            grad_flat = self.map.flatten_w(grad)
            df_hp = self._dot(grad_flat, p)
            df = tf.cast(df_hp, grad_flat.dtype)

            self.map.set_w(w_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(w, p, eval_fn)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        if inputs.shape[0] > 1:
            raise NotImplementedError("L-BFGS requires single batch")

        input = inputs[0]
        w = self.map.flatten_w(self.map.get_w())

        cost, grad_u, grad_w = self._get_grad(input)
        g = self.map.flatten_w(grad_w)

        w_dim = tf.shape(w)[0]
        s_mem = tf.zeros([self.memory, w_dim], dtype=w.dtype)
        y_mem = tf.zeros([self.memory, w_dim], dtype=w.dtype)
        n_mem = tf.constant(0, dtype=tf.int32)

        U_map, V_map = self.map.get_UV(inputs[0, :, :, :])
        costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))
        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        last_iter = tf.constant(-1, dtype=tf.int32)

        self._init_step_state(U_map, V_map, w)

        for it in tf.range(self.iter_max, dtype=tf.int32):
            w_old, g_old = w, g

            tau = tf.constant(1.0, w.dtype) - tf.exp(
                -tf.cast(it, w.dtype) / tf.constant(5.0, w.dtype)
            )
            p = self._compute_direction(g, s_mem[:n_mem], y_mem[:n_mem], n_mem, tau)
            p = self._prepare_search_direction(p, g, w)

            alpha = self._line_search_step(w, p, input)
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            w = self._perform_step(w, alpha, p)
            self.map.set_w(self.map.unflatten_w(w))

            cost, grad_u, grad_w = self._get_grad(input)
            g = self.map.flatten_w(grad_w)

            s, y = w - w_old, g - g_old
            s, y = self._mask_curvature_pair(s, y, w_old)

            ys_hp = self._dot(y, s)
            if ys_hp > self.eps:
                s_mem = tf.cond(
                    n_mem < self.memory,
                    lambda: tf.tensor_scatter_nd_update(s_mem, [[n_mem]], [s]),
                    lambda: tf.concat([s_mem[1:], [s]], axis=0),
                )
                y_mem = tf.cond(
                    n_mem < self.memory,
                    lambda: tf.tensor_scatter_nd_update(y_mem, [[n_mem]], [y]),
                    lambda: tf.concat([y_mem[1:], [y]], axis=0),
                )
                n_mem = tf.minimum(n_mem + 1, self.memory)

            costs = costs.write(it, cost)
            grad_u_norm, grad_w_norm = self._get_grad_norm(grad_u, grad_w)

            self.map.on_step_end(it)
            U_map, V_map = self.map.get_UV(input)
            self._update_step_state(it, U_map, V_map, w, cost, grad_u_norm, grad_w_norm)
            halt_status = self._check_stopping()
            self._update_display()

            last_iter = it

            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break

        self._finalize_display(halt_status)
        return costs.stack()[: last_iter + 1]


class OptimizerLBFGSBounded(OptimizerLBFGS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "lbfgs_bounded"

        if not hasattr(self.map, "get_box_bounds_flat"):
            raise ValueError(
                "Mapping must provide get_box_bounds_flat() for bounded optimization"
            )

    @tf.function(reduce_retracing=True)
    def _project(self, w: tf.Tensor, L: tf.Tensor, U: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(w, L, U)

    @tf.function(reduce_retracing=True)
    def _free_mask(
        self, w: tf.Tensor, g: tf.Tensor, L: tf.Tensor, U: tf.Tensor
    ) -> tf.Tensor:
        eps = tf.cast(self.eps, w.dtype)
        interior = tf.logical_and(w > L + eps, w < U - eps)
        at_lower = tf.logical_and(w <= L + eps, g > 0.0)
        at_upper = tf.logical_and(w >= U - eps, g < 0.0)
        return tf.logical_or(interior, tf.logical_or(at_lower, at_upper))

    def _prepare_search_direction(
        self, p: tf.Tensor, g: tf.Tensor, w: tf.Tensor
    ) -> tf.Tensor:
        L, U = self.map.get_box_bounds_flat()
        mask = self._free_mask(w, g, L, U)
        p = tf.where(mask, p, tf.zeros_like(p))
        gp = self._dot(g, p)
        return tf.cond(gp >= 0.0, lambda: -g, lambda: p)

    def _perform_step(self, w: tf.Tensor, alpha: tf.Tensor, p: tf.Tensor) -> tf.Tensor:
        L, U = self.map.get_box_bounds_flat()
        return self._project(w + alpha * p, L, U)

    def _mask_curvature_pair(
        self, s: tf.Tensor, y: tf.Tensor, w_old: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        L, U = self.map.get_box_bounds_flat()
        g_old = y + s
        mask = self._free_mask(w_old, g_old, L, U)
        return tf.where(mask, s, tf.zeros_like(s)), tf.where(mask, y, tf.zeros_like(y))

    @tf.function
    def _line_search_step(
        self, w: tf.Tensor, p: tf.Tensor, input: tf.Tensor
    ) -> tf.Tensor:
        L, U = self.map.get_box_bounds_flat()

        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            w_backup = self.map.copy_w(self.map.get_w())
            w_new = self._project(w + alpha * p, L, U)

            self.map.set_w(self.map.unflatten_w(w_new))
            f, _, grad = self._get_grad(input)
            grad_flat = self.map.flatten_w(grad)

            mask = self._free_mask(w_new, grad_flat, L, U)
            p_eff = tf.where(mask, p, tf.zeros_like(p))
            df_hp = self._dot(grad_flat, p_eff)
            df = tf.cast(df_hp, grad_flat.dtype)

            self.map.set_w(w_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(w, p, eval_fn)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        L, U = self.map.get_box_bounds_flat()
        w_init = self.map.flatten_w(self.map.get_w())
        w_proj = self._project(w_init, L, U)
        self.map.set_w(self.map.unflatten_w(w_proj))

        return super().minimize_impl(inputs)
