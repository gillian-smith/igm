#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from rich.theme import Theme
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)

from ..mappings import Mapping
from igm.utils.math.precision import _normalize_precision

progress_theme = Theme(
    {
        "label": "bold #e5e7eb",
        "value.cost": "#f59e0b",
        "value.grad": "#06b6d4",
        "value.delta": "#a78bfa",
        "bar.incomplete": "grey35",
        "bar.complete": "#22c55e",
    }
)


class Optimizer(ABC):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        convergence_tolerance: float = 1e-6,
    ):
        self.name = ""
        self.cost_fn = cost_fn
        self.map = map
        self.print_cost = print_cost
        self.print_cost_freq = print_cost_freq
        self.precision = _normalize_precision(precision)
        self.convergence_tolerance = convergence_tolerance

    @abstractmethod
    def update_parameters(self) -> None:
        raise NotImplementedError(
            "❌ The parameters update function is not implemented in this class."
        )

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.print_cost:
            self._progress_setup()
        self.map.on_minimize_start(int(self.iter_max))
        costs = self.minimize_impl(inputs)

        return costs

    @abstractmethod
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The minimize_impl function is not implemented in this class."
        )

    def _get_grad_norm(self, grad_w: list[tf.Tensor]) -> tf.Tensor:
        grad_flat = self.map.flatten_w(grad_w)
                        
        return tf.norm(grad_flat)

    @tf.function
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor]]:
        w = self.map.get_w()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for wi in w:
                tape.watch(wi)
            U, V = self.map.get_UV(inputs)
            processed_inputs = self.map.synchronize_inputs(inputs)
            cost = self.cost_fn(U, V, processed_inputs)

        grad_w = tape.gradient(cost, w)
        del tape
        return cost, grad_w

    def _progress_setup(self) -> None:
        if not self.print_cost:
            return

        self.console = Console(theme=progress_theme)

        self.progress = Progress(
            "🎯",
            BarColumn(
                bar_width=None, style="bar.incomplete", complete_style="bar.complete"
            ),
            MofNCompleteColumn(),
            "[label]•[/]",
            TextColumn("[label]Cost:[/] [value.cost]{task.fields[cost]}"),
            "[label]•[/]",
            TextColumn("[label]Grad:[/] [value.grad]{task.fields[grad_norm]}"),
            "[label]•[/]",
            TextColumn("[label]Time:[/]"),
            TimeElapsedColumn(),
            "[label]•[/]",
            TextColumn("[label]ETA:[/]"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        self.task = self.progress.add_task(
            "progress",
            total=int(self.iter_max.numpy()),
            cost="N/A",
            grad_norm="N/A",
        )

        self.progress.start()

    def _progress_update(self, iter: tf.Tensor, cost: tf.Tensor, grad_norm: tf.Tensor) -> tf.Tensor:

        do_update = tf.logical_and(
            tf.constant(self.print_cost, dtype=tf.bool),
            tf.equal(tf.math.mod(iter, self.print_cost_freq), 0),
        )

        def update(iter: tf.Tensor, cost: tf.Tensor, grad_norm: tf.Tensor) -> float:
            self.progress.update(
                self.task,
                completed=int(iter.numpy()) + 1,
                cost=f"{float(cost.numpy()):.4e}",
                grad_norm=f"{float(grad_norm.numpy()):.4e}",
            )
            return 1.0

        tf.cond(
            do_update,
            lambda: tf.py_function(
                update,
                [iter, cost, grad_norm],
                cost.dtype,
            ),
            lambda: tf.constant(0.0, dtype=cost.dtype),
        )

        # Check stopping criteria
        should_check = tf.greater(iter, 0)
        
        # Get halt criterion result (boolean and message)
        halt, halt_message = self.map.check_halt_criterion(iter, cost)
        
        converged = tf.logical_and(should_check, grad_norm < self.convergence_tolerance)
        max_iter_reached = tf.greater_equal(iter + 1, self.iter_max)  # Check if next iter would exceed max
        
        should_stop = tf.logical_or(tf.logical_or(halt, converged), max_iter_reached)
        
        # Finalize progress and show exit message when stopping
        def finalize_with_reason(halt_val: tf.Tensor, converged_val: tf.Tensor, halt_msg: tf.Tensor) -> int:
            if self.print_cost:
                self.progress.stop()
                
                # Use the passed boolean values (now accessible via .numpy())
                if halt_val.numpy():
                    msg = halt_msg.numpy().decode('utf-8') if halt_msg.numpy() else "Halt criterion met."
                    self.console.print(f"🛑 [bold yellow]Optimization halted![/bold yellow] {msg}")
                elif converged_val.numpy():
                    self.console.print("✅ [bold green]Optimization converged![/bold green] Gradient norm below threshold.")
                else:  # max_iter_reached
                    self.console.print("🏁 [bold blue]Optimization completed![/bold blue] Maximum iterations reached.")
                
                self.console.print()  # Add spacing
            return 0

        # Call finalize when stopping - pass the boolean tensors and halt message as arguments
        tf.cond(
            should_stop,
            lambda: tf.py_function(finalize_with_reason, [halt, converged, halt_message], tf.int32),
            lambda: tf.constant(0)
        )
        
        return should_stop
