#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# GNU GPL v3

import tensorflow as tf
from typing import Callable, Tuple

from ..mappings import Mapping
from .optimizer import Optimizer
from .line_search import LineSearches, ValueAndGradient
from ..utils import _normalize_precision

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
        *,
        precision: str | tf.dtypes.DType = "double",  # NEW
    ):
        super().__init__(cost_fn, map, print_cost, print_cost_freq)
        self.name = "lbfgs"

        # --- precision config (STRICT; no auto-casting) ---
        self.compute_dtype = _normalize_precision(precision)

        self.line_search = LineSearches[line_search_method]()
        self.iter_max = tf.Variable(iter_max)
        self.alpha_min = tf.Variable(tf.cast(alpha_min, self.compute_dtype), dtype=self.compute_dtype)
        self.memory = memory

        if self.memory < 1:
            raise ValueError("L-BFGS memory must be at least 1")

        # Detect once whether the mapping exposes θ-space bounds
        self._has_box_bounds = hasattr(map, "get_box_bounds_flat")

    def update_parameters(self, iter_max: int, alpha_min: float) -> None:
        self.iter_max.assign(iter_max)
        self.alpha_min.assign(tf.cast(alpha_min, self.compute_dtype))

    # ---------- STRICT dtype checks (kept local to LBFGS) ----------
    @tf.function(reduce_retracing=True)
    def _ensure_precision_strict(self, inputs: tf.Tensor) -> tf.Tensor:
        # Inputs
        tf.debugging.assert_type(
            inputs, self.compute_dtype,
            message=f"[LBFGS] inputs must be {self.compute_dtype.name}"
        )
        # Variables
        for w in self.map.get_w():
            tf.debugging.assert_type(
                w, self.compute_dtype,
                message=f"[LBFGS] mapping variables must be {self.compute_dtype.name}"
            )
        # Bounds (if any)
        if self._has_box_bounds:
            L, U = self.map.get_box_bounds_flat()
            tf.debugging.assert_type(L, self.compute_dtype, message="[LBFGS] lower bounds must match optimizer dtype")
            tf.debugging.assert_type(U, self.compute_dtype, message="[LBFGS] upper bounds must match optimizer dtype")
        return inputs

    # ---------- reduction helpers: now use selected precision (name kept) ----------
    @tf.function(reduce_retracing=True)
    def _dot64(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        # name preserved; operates in self.compute_dtype
        return tf.tensordot(tf.cast(a, self.compute_dtype), tf.cast(b, self.compute_dtype), axes=1)

    @tf.function(reduce_retracing=True)
    def _norm64(self, a: tf.Tensor) -> tf.Tensor:
        # name preserved; operates in self.compute_dtype
        return tf.norm(tf.cast(a, self.compute_dtype))

    # ---------- L-BFGS two-loop (H0 tempering via `tau`) ----------
    @tf.function(reduce_retracing=True)
    def _two_loop_recursion(
        self, grad: tf.Tensor, s_list: tf.Tensor, y_list: tf.Tensor, tau: tf.Tensor
    ) -> tf.Tensor:
        q = grad
        alpha_list = tf.TensorArray(dtype=self.compute_dtype, size=0, dynamic_size=True)
        num_elems = tf.shape(s_list)[0]

        # First loop
        for i in tf.range(num_elems - 1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]
            rho = tf.cast(1.0, self.compute_dtype) / (self._dot64(y_i, s_i) + tf.constant(1e-300, self.compute_dtype))
            alpha_i = rho * self._dot64(s_i, q)
            alpha_list = alpha_list.write(i, alpha_i)
            q = q - tf.cast(alpha_i, q.dtype) * tf.cast(y_i, q.dtype)

        def compute_gamma_fn() -> tf.Tensor:
            last_y = y_list[num_elems - 1]
            last_s = s_list[num_elems - 1]
            ys = self._dot64(last_y, last_s)
            yy = self._dot64(last_y, last_y)
            return ys / (yy + tf.constant(1e-300, self.compute_dtype))

        gamma = tf.cond(
            num_elems > 0, compute_gamma_fn, lambda: tf.cast(1.0, self.compute_dtype)
        )

        # H0 tempering: r = (tau * gamma) * q
        r = tf.cast(tau, q.dtype) * tf.cast(gamma, q.dtype) * q

        # Second loop
        for i in tf.range(num_elems):
            s_i = s_list[i]
            y_i = y_list[i]
            alpha_i = alpha_list.read(i)
            rho = tf.cast(1.0, self.compute_dtype) / (self._dot64(y_i, s_i) + tf.constant(1e-300, self.compute_dtype))
            beta = rho * self._dot64(y_i, r)
            r = r + tf.cast(s_i, r.dtype) * (tf.cast(alpha_i - beta, r.dtype))

        return -r

    # ---------- Box-constraint helpers (used only if mapping provides bounds) ----------
    @tf.function(reduce_retracing=True)
    def _project_box(self, w_flat: tf.Tensor, L_flat: tf.Tensor, U_flat: tf.Tensor) -> tf.Tensor:
        return tf.minimum(tf.maximum(w_flat, L_flat), U_flat)

    @tf.function(reduce_retracing=True)
    def _free_mask(self, w_flat: tf.Tensor, grad_flat: tf.Tensor, L_flat: tf.Tensor, U_flat: tf.Tensor) -> tf.Tensor:
        eps = tf.constant(1e-12, w_flat.dtype)
        inside = tf.logical_and(w_flat > L_flat + eps, w_flat < U_flat - eps)
        atL_in = tf.logical_and(w_flat <= L_flat + eps, grad_flat > 0.0)  # inward from L
        atU_in = tf.logical_and(w_flat >= U_flat - eps, grad_flat < 0.0)  # inward from U
        return tf.logical_or(inside, tf.logical_or(atL_in, atU_in))

    @tf.function(reduce_retracing=True)
    def _alpha_cap_box(self, w_flat: tf.Tensor, p_flat: tf.Tensor, L_flat: tf.Tensor, U_flat: tf.Tensor) -> tf.Tensor:
        inf = tf.constant(1e30, w_flat.dtype)
        num = tf.where(p_flat > 0, U_flat - w_flat, tf.where(p_flat < 0, L_flat - w_flat, tf.zeros_like(w_flat)))
        den = tf.where(p_flat != 0, p_flat, tf.ones_like(p_flat))
        alpha_each = tf.where(p_flat != 0, num / den, inf)
        alpha_each = tf.where(alpha_each >= 0.0, alpha_each, inf)
        return tf.reduce_min(alpha_each)

    # ---------- Line search (unconstrained) ----------
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

    # ---------- Line search (projected-path for box constraints) ----------
    @tf.function
    def _line_search_projected(
        self, w_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor, L_flat: tf.Tensor, U_flat: tf.Tensor
    ) -> tf.Tensor:
        def value_and_gradients_function(alpha: tf.Tensor) -> ValueAndGradient:
            w_backup = self.map.copy_w(self.map.get_w())

            # Projected path: φ(α) = f( Π(w + α p) )
            w_trial = w_flat + alpha * p_flat
            w_alpha = self._project_box(w_trial, L_flat, U_flat)
            w_alpha_unflat = self.map.unflatten_w(w_alpha)

            self.map.set_w(w_alpha_unflat)
            f, grad = self._get_grad(input)
            grad_flat = self.map.flatten_w(grad)

            # Directional derivative along projected path: df = g^T (J ⊙ p)
            eps = tf.constant(1e-12, w_alpha.dtype)
            atL = w_alpha <= (L_flat + eps)
            atU = w_alpha >= (U_flat - eps)
            outward_at_L = tf.logical_and(atL, p_flat < 0.0)
            outward_at_U = tf.logical_and(atU, p_flat > 0.0)
            J = tf.logical_not(tf.logical_or(outward_at_L, outward_at_U))
            p_eff = tf.where(J, p_flat, tf.zeros_like(p_flat))

            df = tf.cast(self._dot64(grad_flat, p_eff), grad_flat.dtype)

            # Restore
            self.map.set_w(w_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(w_flat, p_flat, value_and_gradients_function)

    # ---------- Memory update (basic) ----------
    @tf.function(reduce_retracing=True)
    def _update_memory_basic(
        self,
        s_flat_mem: tf.Tensor,
        y_flat_mem: tf.Tensor,
        idx_memory: tf.Tensor,  # int32
        s: tf.Tensor,
        y: tf.Tensor,
    ):
        ys = self._dot64(y, s)
        cond_update = ys > tf.constant(1e-10, self.compute_dtype)

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

    # ---------- Main loop ----------
    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        # Enforce strict precision for this run (inputs, variables, bounds)
        inputs = self._ensure_precision_strict(inputs)

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
        tf.debugging.assert_type(w_flat, self.compute_dtype, message="[LBFGS] flatten_w must return compute dtype")

        # Optional θ-space bounds (only if mapping provides them)
        if self._has_box_bounds:
            L_flat, U_flat = self.map.get_box_bounds_flat()
            # Strict: no casting; require correct dtype
            tf.debugging.assert_type(L_flat, self.compute_dtype, message="[LBFGS] lower bounds must match dtype")
            tf.debugging.assert_type(U_flat, self.compute_dtype, message="[LBFGS] upper bounds must match dtype")
            # Start feasible
            w_flat = self._project_box(w_flat, L_flat, U_flat)
            self.map.set_w(self.map.unflatten_w(w_flat))
        else:
            # Dummy tensors to satisfy TF signature; never used when _has_box_bounds is False
            L_flat = tf.zeros_like(w_flat)
            U_flat = tf.zeros_like(w_flat)

        # Evaluate at initial point
        cost, grad_w = self._get_grad(input)
        grad_w_flat = self.map.flatten_w(grad_w)
        tf.debugging.assert_type(grad_w_flat, self.compute_dtype, message="[LBFGS] flattened gradient dtype mismatch")

        # L-BFGS memory
        w_dim = tf.shape(w_flat)[0]
        idx_memory = tf.constant(0, dtype=tf.int32)
        s_flat_mem = tf.zeros([self.memory, w_dim], dtype=w_flat.dtype)
        y_flat_mem = tf.zeros([self.memory, w_dim], dtype=w_flat.dtype)

        costs = tf.TensorArray(dtype=w_flat.dtype, size=self.iter_max)

        # define a default mask so Autograph sees it before the loop
        free_mask_all_true = tf.ones_like(w_flat, dtype=tf.bool)

        for it in tf.range(self.iter_max, dtype=tf.int32):
            w_prev = w_flat
            g_prev = grad_w_flat

            # H0 tempering: tau in (0,1]
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

            free_mask = free_mask_all_true
            # If box bounds: mask direction to free set
            if self._has_box_bounds:
                free_mask = self._free_mask(w_flat, grad_w_flat, L_flat, U_flat)
                p_flat = tf.where(free_mask, p_flat, tf.zeros_like(p_flat))

            # Enforce descent
            gTp = self._dot64(grad_w_flat, p_flat)
            p_flat = tf.cond(gTp >= 0.0, lambda: -grad_w_flat, lambda: p_flat)

            # Line search (projected-path if bounded; else original)
            if self._has_box_bounds:
                alpha = self._line_search_projected(w_flat, p_flat, input, L_flat, U_flat)
            else:
                alpha = self._line_search(w_flat, p_flat, input)

            # No trust region: just enforce alpha_min
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            # Box cap (optional but helpful), only if bounded
            if self._has_box_bounds:
                alpha_box = self._alpha_cap_box(w_flat, p_flat, L_flat, U_flat)
                alpha = tf.minimum(alpha, alpha_box)

            # Step and (if needed) project
            w_trial = w_prev + alpha * p_flat
            if self._has_box_bounds:
                w_flat = self._project_box(w_trial, L_flat, U_flat)
            else:
                w_flat = w_trial

            self.map.set_w(self.map.unflatten_w(w_flat))

            # New value/grad
            cost, grad_w = self._get_grad(input)
            grad_w_flat = self.map.flatten_w(grad_w)
            tf.debugging.assert_type(grad_w_flat, self.compute_dtype, message="[LBFGS] flattened gradient dtype mismatch")

            # Curvature pair (clean for projection if needed)
            s = w_flat - w_prev
            y = grad_w_flat - g_prev

            if self._has_box_bounds:
                # zero components where projection altered the step
                proj_delta = w_flat - w_trial
                changed = tf.not_equal(tf.abs(proj_delta) > 0, False)
                s = tf.where(changed, tf.zeros_like(s), s)
                y = tf.where(changed, tf.zeros_like(y), y)
                # also zero on non-free coords (consistent with masked direction)
                s = tf.where(free_mask, s, tf.zeros_like(s))
                y = tf.where(free_mask, y, tf.zeros_like(y))

            # Memory update (always basic)
            s_flat_mem, y_flat_mem, idx_memory = self._update_memory_basic(
                s_flat_mem, y_flat_mem, idx_memory, s, y
            )

            # Bookkeeping / stop
            costs = costs.write(it, cost)
            grad_norm = self._get_grad_norm(grad_w)

            self.map.on_step_end(it)
            
            should_stop = self._progress_update(it, cost, grad_norm)
            if should_stop:
                break

        return costs.stack()
