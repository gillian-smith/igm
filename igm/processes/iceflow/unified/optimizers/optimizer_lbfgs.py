#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# GNU GPL v3

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
        # Options
        use_trust_region: bool = True,   # when True: cap by TR and fallback if alpha < 0
        trust_radius_c: float = 1e-2,    # Δ0 = c * ||w0|| / sqrt(d)
        trust_radius_growth: float = 1.5,
        cautious_pairs: bool = True,     # angle test + Powell damping
        store_pairs_float64: bool = False,  # store s,y in float64 (optional - potentially memory hungry!)
    ):
        super().__init__(cost_fn, map, print_cost, print_cost_freq)
        self.name = "lbfgs"

        self.line_search = LineSearches[line_search_method]()
        self.iter_max = tf.Variable(iter_max)
        self.alpha_min = tf.Variable(alpha_min)
        self.memory = memory

        self.use_trust_region = bool(use_trust_region)
        self.trust_radius_c = float(trust_radius_c)
        self.trust_radius_growth = float(trust_radius_growth)
        self.cautious_pairs = bool(cautious_pairs)
        self.store_pairs_float64 = bool(store_pairs_float64)

    def update_parameters(self, iter_max: int, alpha_min: float) -> None:
        self.iter_max.assign(iter_max)
        self.alpha_min.assign(alpha_min)

    # ---------- float64 reduction helpers ----------
    @tf.function(reduce_retracing=True)
    def _dot64(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(tf.cast(a, tf.float64), tf.cast(b, tf.float64), axes=1)

    @tf.function(reduce_retracing=True)
    def _norm64(self, a: tf.Tensor) -> tf.Tensor:
        return tf.norm(tf.cast(a, tf.float64))

    # ---------- L-BFGS two-loop (with H0 tempering via `tau`, 64-bit reductions) ----------
    @tf.function(reduce_retracing=True)
    def _two_loop_recursion(
        self, grad: tf.Tensor, s_list: tf.Tensor, y_list: tf.Tensor, tau: tf.Tensor
    ) -> tf.Tensor:
        q = grad
        alpha_list = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)  # store scalars as f64
        num_elems = tf.shape(s_list)[0]

        # First loop
        for i in tf.range(num_elems - 1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]
            rho64 = 1.0 / (self._dot64(y_i, s_i) + tf.constant(1e-300, tf.float64))
            alpha64 = rho64 * self._dot64(s_i, q)
            alpha_list = alpha_list.write(i, alpha64)
            # q = q - alpha * y
            q = q - tf.cast(alpha64, q.dtype) * tf.cast(y_i, q.dtype)

        def compute_gamma64_fn() -> tf.Tensor:
            last_y = y_list[num_elems - 1]
            last_s = s_list[num_elems - 1]
            ys64 = self._dot64(last_y, last_s)
            yy64 = self._dot64(last_y, last_y)
            return ys64 / (yy64 + tf.constant(1e-300, tf.float64))

        gamma64 = tf.cond(
            num_elems > 0, compute_gamma64_fn, lambda: tf.constant(1.0, tf.float64)
        )

        # H0 tempering: r = (tau * gamma) * q   (tau in (0,1], grows toward 1)
        r = tf.cast(tau, q.dtype) * tf.cast(gamma64, q.dtype) * q

        # Second loop
        for i in tf.range(num_elems):
            s_i = s_list[i]
            y_i = y_list[i]
            alpha64 = alpha_list.read(i)
            rho64 = 1.0 / (self._dot64(y_i, s_i) + tf.constant(1e-300, tf.float64))
            beta64 = rho64 * self._dot64(y_i, r)
            r = r + tf.cast(s_i, r.dtype) * (tf.cast(alpha64 - beta64, r.dtype))

        return -r

    # ---------- Trust-region helpers ----------
    @staticmethod
    def _suggest_trust_radius_init(w_flat: tf.Tensor, c: float) -> tf.Tensor:
        d = tf.cast(tf.shape(w_flat)[0], w_flat.dtype)
        rms = tf.norm(w_flat) / (tf.sqrt(d) + tf.constant(1e-12, w_flat.dtype))
        safe = tf.maximum(rms, tf.constant(1e-8, w_flat.dtype))
        return tf.cast(c, w_flat.dtype) * safe

    @tf.function(reduce_retracing=True)
    def _cap_by_trust_region(
        self,
        alpha: tf.Tensor,
        p_flat: tf.Tensor,
        it: tf.Tensor,
        delta_init: tf.Tensor,
        growth: tf.Tensor,
    ) -> tf.Tensor:
        # compute alpha_max in float64 for robustness, then cast back
        delta_t64 = tf.cast(delta_init, tf.float64) * tf.pow(tf.cast(growth, tf.float64), tf.cast(it, tf.float64))
        pnorm64 = self._norm64(p_flat)
        alpha_max64 = delta_t64 / (pnorm64 + tf.constant(1e-300, tf.float64))
        alpha_max = tf.cast(alpha_max64, alpha.dtype)
        return tf.minimum(alpha, alpha_max)

    @tf.function(reduce_retracing=True)
    def _fallback_when_bad_alpha(
        self,
        p_flat: tf.Tensor,
        grad_w_flat: tf.Tensor,
        it: tf.Tensor,
        delta0: tf.Tensor,
        growth: tf.Tensor,
    ):
        # Ensure descent (if g^T p >= 0, switch to steepest descent)
        gTp64 = self._dot64(grad_w_flat, p_flat)
        p_safe = tf.cond(gTp64 >= 0.0, lambda: -grad_w_flat, lambda: p_flat)

        # Cauchy-style fallback, limited by trust region (compute in f64)
        delta_t64 = tf.cast(delta0, tf.float64) * tf.pow(tf.cast(growth, tf.float64), tf.cast(it, tf.float64))
        pnorm64 = self._norm64(p_safe)
        alpha_tr64 = delta_t64 / (pnorm64 + tf.constant(1e-300, tf.float64))
        alpha_fb = tf.cast(tf.minimum(tf.constant(1.0, tf.float64), alpha_tr64), p_flat.dtype)
        return alpha_fb, p_safe

    # ---------- Line search ----------
    @tf.function
    def _line_search(
        self, w_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor
    ) -> tf.Tensor:
        def value_and_gradients_function(alpha: tf.Tensor) -> ValueAndGradient:
            w_backup = self.map.copy_w(self.map.get_w())

            w_alpha = w_flat + alpha * p_flat
            w_alpha = self.map.unflatten_w(w_alpha)

            self.map.set_w(w_alpha)
            f, grad = self._get_grad(input)
            grad_flat = self.map.flatten_w(grad)
            df = tf.cast(self._dot64(grad_flat, p_flat), grad_flat.dtype)

            self.map.set_w(w_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(w_flat, p_flat, value_and_gradients_function)

    # ---------- Memory update (basic, with 64-bit gate) ----------
    @tf.function(reduce_retracing=True)
    def _update_memory_basic(
        self,
        s_flat_mem: tf.Tensor,
        y_flat_mem: tf.Tensor,
        idx_memory: tf.Tensor,  # int32
        s: tf.Tensor,
        y: tf.Tensor,
    ):
        ys64 = self._dot64(y, s)
        cond_update = ys64 > tf.constant(1e-10, tf.float64)

        s_store = tf.cast(s, s_flat_mem.dtype)
        y_store = tf.cast(y, y_flat_mem.dtype)

        def memory_append():
            return (
                tf.tensor_scatter_nd_update(s_flat_mem, [[idx_memory]], [s_store]),
                tf.tensor_scatter_nd_update(y_flat_mem, [[idx_memory]], [y_store]),
                idx_memory + tf.constant(1, idx_memory.dtype),
            )

        def memory_circ_add():
            return (
                tf.concat([s_flat_mem[1:], tf.expand_dims(s_store, 0)], axis=0),
                tf.concat([y_flat_mem[1:], tf.expand_dims(y_store, 0)], axis=0),
                idx_memory,
            )

        def do_update():
            return tf.cond(
                idx_memory < tf.constant(self.memory, idx_memory.dtype),
                memory_append,
                memory_circ_add,
            )

        return tf.cond(cond_update, do_update, lambda: (s_flat_mem, y_flat_mem, idx_memory))

    # ---------- Memory update (cautious: angle test + Powell damping, with 64-bit reductions) ----------
    @tf.function(reduce_retracing=True)
    def _update_memory_cautious(
        self,
        s_flat_mem: tf.Tensor,
        y_flat_mem: tf.Tensor,
        idx_memory: tf.Tensor,  # int32
        s: tf.Tensor,
        y: tf.Tensor,
    ):
        ys64 = self._dot64(y, s)
        yy64 = self._dot64(y, y)
        ss64 = self._dot64(s, s)

        cos_theta64 = ys64 / tf.sqrt((yy64 + tf.constant(1e-300, tf.float64)) * (ss64 + tf.constant(1e-300, tf.float64)))
        angle_ok = cos_theta64 > tf.constant(1e-3, tf.float64)
        curv_ok = ys64 > tf.constant(1e-10, tf.float64)

        # Cheap B0 ≈ (1/gamma) I, gamma = (y^T s)/(y^T y)
        gamma64 = ys64 / (yy64 + tf.constant(1e-300, tf.float64))
        inv_gamma64 = 1.0 / (gamma64 + tf.constant(1e-300, tf.float64))

        mu64 = tf.constant(0.25, tf.float64)
        sBs64 = inv_gamma64 * ss64

        theta64 = tf.where(
            ys64 >= mu64 * sBs64,
            tf.constant(1.0, tf.float64),
            mu64 * sBs64 / (sBs64 - ys64 + tf.constant(1e-300, tf.float64)),
        )

        # y_damped = theta*y + (1-theta)*(inv_gamma*s), computed in vector dtype then cast for storage
        theta_vec = tf.cast(theta64, y.dtype)
        inv_gamma_vec = tf.cast(inv_gamma64, y.dtype)
        y_damped = theta_vec * y + (1.0 - theta_vec) * (inv_gamma_vec * s)

        pair_ok = angle_ok & curv_ok & (self._dot64(y_damped, s) > tf.constant(1e-10, tf.float64))

        s_store = tf.cast(s, s_flat_mem.dtype)
        y_store = tf.cast(y_damped, y_flat_mem.dtype)

        def memory_append():
            return (
                tf.tensor_scatter_nd_update(s_flat_mem, [[idx_memory]], [s_store]),
                tf.tensor_scatter_nd_update(y_flat_mem, [[idx_memory]], [y_store]),
                idx_memory + tf.constant(1, idx_memory.dtype),
            )

        def memory_circ_add():
            return (
                tf.concat([s_flat_mem[1:], tf.expand_dims(s_store, 0)], axis=0),
                tf.concat([y_flat_mem[1:], tf.expand_dims(y_store, 0)], axis=0),
                idx_memory,
            )

        def do_update():
            return tf.cond(
                idx_memory < tf.constant(self.memory, idx_memory.dtype),
                memory_append,
                memory_circ_add,
            )

        return tf.cond(pair_ok, do_update, lambda: (s_flat_mem, y_flat_mem, idx_memory))

    # ---------- Main loop ----------
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

        # Always compute Δ0 fresh from w0
        delta0 = self._suggest_trust_radius_init(w_flat, c=self.trust_radius_c)

        # Dynamic TR adaptation multiplier (tensor, not a tf.Variable)
        adapt_scale = tf.cast(1.0, w_flat.dtype)

        # Evaluate at initial point
        cost, grad_w = self._get_grad(input)
        grad_w_flat = self.map.flatten_w(grad_w)

        # Track previous cost as a tensor
        prev_cost = tf.identity(cost)

        # L-BFGS memory (optionally float64)
        w_dim = tf.shape(w_flat)[0]
        pairs_dtype = tf.float64 if self.store_pairs_float64 else w_flat.dtype
        idx_memory = tf.constant(0, dtype=tf.int32)
        s_flat_mem = tf.zeros([self.memory, w_dim], dtype=pairs_dtype)
        y_flat_mem = tf.zeros([self.memory, w_dim], dtype=pairs_dtype)

        costs = tf.TensorArray(dtype=w_flat.dtype, size=self.iter_max)

        for it in tf.range(self.iter_max, dtype=tf.int32):
            w_prev = w_flat
            g_prev = grad_w_flat

            # H0 tempering factor: tau in (0,1], warm-up ~5 iters
            tau = tf.constant(1.0, w_flat.dtype) - tf.exp(
                -tf.cast(it, w_flat.dtype) / tf.constant(5.0, w_flat.dtype)
            )

            # Direction
            p_flat = tf.cond(
                idx_memory > 0,
                lambda: self._two_loop_recursion(
                    grad_w_flat, s_flat_mem[:idx_memory], y_flat_mem[:idx_memory], tau
                ),
                lambda: -grad_w_flat,
            )

            # Enforce descent before line search (64-bit check)
            gTp64 = self._dot64(grad_w_flat, p_flat)
            p_flat = tf.cond(gTp64 >= 0.0, lambda: -grad_w_flat, lambda: p_flat)

            # Line search
            alpha = self._line_search(w_flat, p_flat, input)

            if self.use_trust_region:
                # Apply dynamic adaptation multiplier to Δ0
                delta0_eff = delta0 * adapt_scale
                alpha = self._cap_by_trust_region(
                    alpha,
                    p_flat,
                    it,
                    delta0_eff,
                    tf.cast(self.trust_radius_growth, alpha.dtype),
                )
                cond_bad = tf.logical_not(alpha >= tf.constant(0.0, alpha.dtype))
                alpha, p_flat = tf.cond(
                    cond_bad,
                    lambda: self._fallback_when_bad_alpha(
                        p_flat,
                        grad_w_flat,
                        it,
                        delta0_eff,
                        tf.cast(self.trust_radius_growth, w_flat.dtype),
                    ),
                    lambda: (alpha, p_flat),
                )
            else:
                # Only enforce alpha >= alpha_min (no TR, no fallback)
                alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            # Step
            w_flat = w_prev + alpha * p_flat
            self.map.set_w(self.map.unflatten_w(w_flat))

            # New value/grad
            cost, grad_w = self._get_grad(input)
            grad_w_flat = self.map.flatten_w(grad_w)

            # Curvature pair
            s = w_flat - w_prev
            y = grad_w_flat - g_prev

            # Memory update
            if self.cautious_pairs:
                s_flat_mem, y_flat_mem, idx_memory = self._update_memory_cautious(
                    s_flat_mem, y_flat_mem, idx_memory, s, y
                )
            else:
                s_flat_mem, y_flat_mem, idx_memory = self._update_memory_basic(
                    s_flat_mem, y_flat_mem, idx_memory, s, y
                )

            # Dynamic TR adaptation: grow on improvement, shrink otherwise
            improved = cost < prev_cost

            improved = cost < prev_cost
            adapt_scale = tf.cond(
                improved,
                lambda: adapt_scale * tf.constant(1.2, adapt_scale.dtype),
                lambda: adapt_scale * tf.constant(0.5, adapt_scale.dtype),
            )
            prev_cost = cost

            # Bookkeeping / stop
            costs = costs.write(it, cost)
            grad_norm = self._get_grad_norm(grad_w)
            should_stop = self._progress_update(it, cost, grad_norm)
            if should_stop:
                break

        return costs.stack()
