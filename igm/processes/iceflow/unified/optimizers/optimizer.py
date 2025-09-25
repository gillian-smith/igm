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
    ):
        self.name = ""
        self.cost_fn = cost_fn
        self.map = map
        self.print_cost = print_cost
        self.print_cost_freq = print_cost_freq

    @abstractmethod
    def update_parameters(self) -> None:
        raise NotImplementedError(
            "❌ The parameters update function is not implemented in this class."
        )

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.print_cost:
            self._progress_setup()

        costs = self.minimize_impl(inputs)

        if self.print_cost:
            self._progress_finalize()

        return costs

    @abstractmethod
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The minimize_impl function is not implemented in this class."
        )

    @tf.function
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        w = self.map.get_w()
        with tf.GradientTape(persistent=True) as tape:
            U, V = self.map.get_UV(inputs)
            cost = self.cost_fn(U, V, inputs)
        grad_u = tape.gradient(cost, [U, V])
        grad_w = tape.gradient(cost, w)
        del tape
        return cost, grad_u, grad_w

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
        )

        self.progress.start()

    def _progress_update(self, iter: tf.Tensor, cost: tf.Tensor) -> tf.Tensor:

        do_update = tf.logical_and(
            tf.constant(self.print_cost, dtype=tf.bool),
            tf.equal(tf.math.mod(iter, self.print_cost_freq), 0),
        )

        def update(iter: tf.Tensor, cost: tf.Tensor) -> float:
            self.progress.update(
                self.task,
                completed=int(iter.numpy()) + 1,
                cost=f"{float(cost.numpy()):.4e}",
            )
            return 1.0

        tf.cond(
            do_update,
            lambda: tf.py_function(
                update,
                [iter, cost],
                cost.dtype,
            ),
            lambda: tf.constant(0.0, dtype=cost.dtype),
        )

    def _progress_finalize(self) -> None:
        if not self.print_cost:
            return

        self.progress.stop()
