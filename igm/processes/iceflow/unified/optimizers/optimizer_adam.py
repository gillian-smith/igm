#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Optional

from .optimizer import Optimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus

tf.config.optimizer.set_jit(True)


class OptimizerAdam(Optimizer):

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
        lr: float = 1e-3,
        iter_max: int = int(1e5),
        lr_decay: float = 0.0,
        lr_decay_steps: int = 1000,
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
        self.name = "adam"

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

        n_batches = inputs.shape[0]

        # State variables
        w = self.map.get_w()
        U, V = self.map.get_UV(inputs[0, :, :, :])
        self._init_step_state(U, V, w)

        # Accessory variables
        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=self.precision, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            cost_sum = tf.constant(0.0, dtype=self.precision)
            grad_u_norm_sum = tf.constant(0.0, dtype=self.precision)
            grad_w_norm_sum = tf.constant(0.0, dtype=self.precision)

            for b in tf.range(n_batches):
                input = inputs[b, :, :, :, :]

                cost, grad_u, grad_w = self._get_grad(input)
                self.optim_adam.apply_gradients(zip(grad_w, w))

                grad_u_norm, grad_w_norm = self._get_grad_norm(grad_u, grad_w)

                cost_sum = cost_sum + cost
                grad_u_norm_sum = grad_u_norm_sum + grad_u_norm
                grad_w_norm_sum = grad_w_norm_sum + grad_w_norm

            cost_avg = cost_sum / n_batches
            grad_u_norm_avg = grad_u_norm_sum / n_batches
            grad_w_norm_avg = grad_w_norm_sum / n_batches

            # TODO: check if this is necessary
            self.map.on_step_end(iter)

            costs = costs.write(iter, cost_avg)

            U, V = self.map.get_UV(input)

            self._update_step_state(
                iter, U, V, w, cost_avg, grad_u_norm_avg, grad_w_norm_avg
            )
            halt_status = self._check_stopping()
            self._update_display()

            iter_last = iter

            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break

        self._finalize_display(halt_status)

        return costs.stack()[: iter_last + 1]
