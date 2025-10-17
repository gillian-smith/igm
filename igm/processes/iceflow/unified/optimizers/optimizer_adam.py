#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable

from ..mappings import Mapping
from .optimizer import Optimizer

tf.config.optimizer.set_jit(True)


class OptimizerAdam(Optimizer):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        print_cost: bool,
        print_cost_freq: int,
        precision: str,
        lr: float = 1e-3,
        iter_max: int = int(1e5),
        lr_decay: float = 0.0,
        lr_decay_steps: int = 1000,
    ):
        super().__init__(cost_fn, map, print_cost, print_cost_freq, precision)
        self.name = "adam"
        self.print_cost = print_cost

        version_tf = int(tf.__version__.split(".")[1])
        if (version_tf <= 10) | (version_tf >= 16):
            module_optimizer = tf.keras.optimizers
        else:
            module_optimizer = tf.keras.optimizers.legacy

        if lr_decay > 0.0:
            schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=lr_decay_steps,
                decay_rate=lr_decay,
            )
            self.optim_adam = module_optimizer.Adam(learning_rate=schedule)
        else:
            self.iter_max = tf.Variable(iter_max)
            self.optim_adam = module_optimizer.Adam(learning_rate=tf.Variable(lr))

    def update_parameters(
        self, iter_max: int, lr: float, lr_decay: float, lr_decay_steps: int
    ) -> None:
        self.iter_max.assign(iter_max)
        self.optim_adam.learning_rate.assign(lr)
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:

        # Initial state
        w = self.map.get_w()
        U, V = self.map.get_UV(inputs[0, :, :, :])
        n_batches = inputs.shape[0]

        costs = tf.TensorArray(dtype=self.precision, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            batch_costs = tf.TensorArray(dtype=self.precision, size=n_batches)

            batch_grad_norms = tf.TensorArray(dtype=self.precision, size=n_batches)

            for i in tf.range(n_batches):
                input = inputs[i, :, :, :, :]

                # Save previous solution
                U_prev = tf.identity(U)
                V_prev = tf.identity(V)

                # Compute cost and grad
                cost, grad_w = self._get_grad(input)
                
                # Compute and store gradient norm for this batch
                grad_norm = self._get_grad_norm(grad_w)
                batch_grad_norms = batch_grad_norms.write(i, grad_norm)

                # Apply Adam descent
                self.optim_adam.apply_gradients(zip(grad_w, w))

                # Post-process
                U, V = self.map.get_UV(input)

                # Store cost for this batch
                batch_costs = batch_costs.write(i, cost)

            iter_cost = tf.reduce_mean(batch_costs.stack())
            avg_grad_norm = tf.reduce_mean(batch_grad_norms.stack())

            costs = costs.write(iter, iter_cost)

            self.map.on_step_end(iter)

            should_stop = self._progress_update(iter, iter_cost, avg_grad_norm)

            # Early stopping check
            if should_stop:
                break

        return costs.stack()
