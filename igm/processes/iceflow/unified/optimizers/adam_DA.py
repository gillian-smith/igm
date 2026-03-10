#!/usr/bin/env python3
# Copyright (C) 2021-2025
# GNU GPL v3

from __future__ import annotations
import tensorflow as tf
from typing import Any, Callable, Tuple

from ..mappings import Mapping
from .adam import OptimizerAdam  # base Adam

# ---- pretty progress (same theme as your LBFGS DA) ----
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


class OptimizerAdamDataAssimilation(OptimizerAdam):
    """
    Adam specialization for data assimilation.

    Adds:
      - storage & display of data/physics cost components
      - rich progress bar with the same look/feel as your LBFGS DA
    """

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        precision: str,
        print_cost: bool = True,
        print_cost_freq: int = 10,
        lr: float = 1e-3,
        iter_max: int = int(1e5),
        lr_decay: float = 0.0,
        lr_decay_steps: int = 1000,
        batch_size: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            cost_fn=cost_fn,
            map=map,
            print_cost=print_cost,
            print_cost_freq=print_cost_freq,
            precision=precision,
            lr=lr,
            iter_max=iter_max,
            lr_decay=lr_decay,
            lr_decay_steps=lr_decay_steps,
            batch_size=batch_size,
        )

        self.name = "ADAM_Data_Assimilation"

    # identical shape/signature as your LBFGS-DA version
    @tf.function
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        w = self.map.get_theta()
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            for wi in w:
                tape.watch(wi)
            U, V = self.map.get_UV(inputs)
            processed_inputs = self.map.synchronize_inputs(inputs)
            # expect cost_fn to return (total, data, physics)
            cost, data_cost, reg_cost = self.cost_fn(U, V, processed_inputs)

        grad_u = tape.gradient(cost, [U, V])
        grad_theta = tape.gradient(cost, w)
        del tape
        return cost, grad_u, grad_theta
