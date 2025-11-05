#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

from .progress_optimizer import ProgressOptimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus, HaltState, StepState
from igm.utils.math.precision import _normalize_precision
from igm.utils.math.norms import compute_norm


class Optimizer(ABC):

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
    ):
        self.name = ""
        self.cost_fn = cost_fn
        self.map = map
        self.halt = halt
        self.step_state = None
        self.halt_state = None
        self.display = ProgressOptimizer(enabled=print_cost, freq=print_cost_freq)
        self.precision = _normalize_precision(precision)
        self.ord_grad_u = ord_grad_u
        self.ord_grad_w = ord_grad_w

    @abstractmethod
    def update_parameters(self) -> None:
        raise NotImplementedError(
            "❌ The parameters update function is not implemented in this class."
        )

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        criterion_names = self.halt.criterion_names if self.halt else []
        self.display.start(int(self.iter_max), criterion_names)
        self.map.on_minimize_start(int(self.iter_max))
        costs = self.minimize_impl(inputs)
        return costs

    @abstractmethod
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The minimize_impl function is not implemented in this class."
        )

    def _get_grad_norm(
        self, grad_u: list[tf.Tensor], grad_w: list[tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # For w
        grad_w_flat = self.map.flatten_w(grad_w)
        grad_w_norm = compute_norm(grad_w_flat, ord=self.ord_grad_w)
        # For (u,v)
        grad_u_x, grad_u_y = grad_u
        grad_u_flat = tf.sqrt(tf.square(grad_u_x) + tf.square(grad_u_y))
        grad_u_norm = compute_norm(grad_u_flat, ord=self.ord_grad_u)
        return grad_u_norm, grad_w_norm

    @tf.function
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        w = self.map.get_w()
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            for wi in w:
                tape.watch(wi)
            U, V = self.map.get_UV(inputs)
            cost = self.cost_fn(U, V, inputs)
        grad_u = tape.gradient(cost, [U, V])
        grad_w = tape.gradient(cost, w)
        del tape
        return cost, grad_u, grad_w

    def _init_step_state(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        w: Any,
    ) -> None:
        self.step_state = StepState(
            iter=tf.constant(-1, dtype=self.precision),
            u=[U, V],
            w=w,
            cost=tf.constant(np.nan, dtype=self.precision),
            grad_u_norm=tf.constant(np.nan, dtype=self.precision),
            grad_w_norm=tf.constant(np.nan, dtype=self.precision),
        )

    def _update_step_state(
        self,
        iter: tf.Tensor,
        U: tf.Tensor,
        V: tf.Tensor,
        w: Any,
        cost: tf.Tensor,
        grad_u_norm: tf.Tensor,
        grad_w_norm: tf.Tensor,
    ) -> None:
        self.step_state = StepState(iter, [U, V], w, cost, grad_u_norm, grad_w_norm)

    def _check_stopping(self) -> tf.Tensor:
        """Check stopping criteria, update halt_state, and return status"""
        if self.halt is None:
            self.halt_state = HaltState.empty()
        else:
            status, values, satisfied = self.halt.check(
                self.step_state.iter, self.step_state
            )
            self.halt_state = HaltState(status, values, satisfied)

        # Check max iterations
        max_iter_reached = tf.greater_equal(self.step_state.iter + 1, self.iter_max)

        self.halt_state.status = tf.cond(
            tf.not_equal(self.halt_state.status, HaltStatus.CONTINUE.value),
            lambda: self.halt_state.status,
            lambda: tf.cond(
                max_iter_reached,
                lambda: tf.constant(HaltStatus.COMPLETED.value),
                lambda: tf.constant(HaltStatus.CONTINUE.value),
            ),
        )

        return self.halt_state.status

    def _update_display(self) -> None:
        """Update display using halt_state"""

        def update_display(iter_val, cost_val, *crit_data):
            if crit_data:
                n = len(crit_data) // 2
                values = [float(crit_data[i].numpy()) for i in range(n)]
                satisfied = [bool(crit_data[n + i].numpy()) for i in range(n)]
            else:
                values = None
                satisfied = None

            self.display.update(
                int(iter_val.numpy()),
                float(cost_val.numpy()),
                values,
                satisfied,
            )
            return 1.0

        should_update = self.display.should_update(self.step_state.iter)

        # Build args from halt_state
        py_func_args = [self.step_state.iter, self.step_state.cost]
        if self.halt_state.criterion_values and self.halt_state.criterion_satisfied:
            py_func_args.extend(self.halt_state.criterion_values)
            py_func_args.extend(self.halt_state.criterion_satisfied)

        tf.cond(
            should_update,
            lambda: tf.py_function(update_display, py_func_args, tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def _finalize_display(self, halt_status: tf.Tensor) -> None:
        """Finalize display using provided halt_status"""
        if not self.display.enabled:
            return

        should_stop = tf.not_equal(halt_status, HaltStatus.CONTINUE.value)

        def finalize(status: tf.Tensor) -> int:
            status_val = int(status.numpy())
            halt_status_enum = HaltStatus(status_val)
            self.display.stop(halt_status_enum)
            return 0

        tf.cond(
            should_stop,
            lambda: tf.py_function(finalize, [halt_status], tf.int32),
            lambda: tf.constant(0),
        )
