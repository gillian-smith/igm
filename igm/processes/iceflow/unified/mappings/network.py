#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import List, Tuple

from .mapping import Mapping
from ..bcs import BoundaryCondition
from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV


class MappingNetwork(Mapping):
    def __init__(
        self,
        bcs: List[BoundaryCondition],
        network: tf.keras.Model,
        normalizer: tf.keras.layers.Layer,
        Nz: tf.Tensor,
        output_scale: tf.Tensor = 1.0,
        precision: str = "float32",
    ):
        super().__init__(bcs, precision)
        self.network = network
        self.output_scale = output_scale
        self.network.input_normalizer = normalizer
        self.shapes = [w.shape for w in network.trainable_variables]
        self.sizes = [tf.reduce_prod(s) for s in self.shapes]
        self.Nz = Nz

        # Patience-based halt criterion variables
        self.patience = 500  # Default patience value; can be overridden
        self.best_cost = tf.Variable(
            float("inf"), trainable=False, name="best_cost", dtype=self.precision
        )
        self.patience_counter = tf.Variable(0, trainable=False, name="patience_counter")
        self.cost_initialized = tf.Variable(
            False, trainable=False, name="cost_initialized"
        )
        self.minimize_call_started = tf.Variable(
            False, trainable=False, name="minimize_call_started"
        )

    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        Y = self.network(self.inputs) * self.output_scale
        U, V = Y_to_UV(self.Nz, Y)
        return U, V

    def copy_theta(self, theta: list[tf.Variable]) -> list[tf.Tensor]:
        return [theta_i.read_value() for theta_i in theta]

    def copy_theta_flat(self, theta_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(theta_flat)

    def get_theta(self) -> list[tf.Variable]:
        return self.network.trainable_variables

    def set_theta(self, theta: list[tf.Tensor]) -> None:
        for var, val in zip(self.network.trainable_variables, theta):
            var.assign(val)

    def flatten_theta(self, theta: list[tf.Variable | tf.Tensor]) -> tf.Tensor:
        theta_flat = [tf.reshape(theta_i, [-1]) for theta_i in theta]
        return tf.concat(theta_flat, axis=0)

    def unflatten_theta(self, theta_flat: tf.Tensor) -> list[tf.Tensor]:
        splits = tf.split(theta_flat, self.sizes)
        return [tf.reshape(t, s) for t, s in zip(splits, self.shapes)]

    def check_halt_criterion(
        self, iteration: int, cost: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Check patience-based halt criterion.
        Always resets patience variables at iteration 0 (start of each minimize call).
        Returns: (halt_boolean, halt_message_string)
        """

        # Always reset patience variables when iteration is 0 (start of new minimize call)
        def reset_and_initialize():
            self.best_cost.assign(cost)
            self.patience_counter.assign(0)
            self.cost_initialized.assign(True)
            self.minimize_call_started.assign(True)
            return tf.constant(False, dtype=tf.bool), tf.constant("", dtype=tf.string)

        def check_patience():
            # Check if current cost is better than best cost
            cost_improved = tf.less(cost, self.best_cost)

            def update_best():
                self.best_cost.assign(cost)
                self.patience_counter.assign(0)
                return tf.constant(False, dtype=tf.bool), tf.constant(
                    "", dtype=tf.string
                )

            def increment_patience():
                self.patience_counter.assign_add(1)
                should_halt = tf.greater_equal(self.patience_counter, self.patience)

                # Create halt message - use simple string to avoid tf.strings.format issues
                halt_message = tf.cond(
                    should_halt,
                    lambda: tf.constant(
                        "Patience exhausted! No improvement for specified iterations.",
                        dtype=tf.string,
                    ),
                    lambda: tf.constant("", dtype=tf.string),
                )

                return should_halt, halt_message

            return tf.cond(cost_improved, update_best, increment_patience)

        # Always reset at iteration 0, otherwise check patience
        return tf.cond(tf.equal(iteration, 0), reset_and_initialize, check_patience)
