#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import List, Optional
from enum import Enum, auto

from .criteria import Criterion
from .metrics.metric import StepState


class HaltStatus(Enum):
    CONTINUE = auto()
    SUCCESS = auto()
    FAILURE = auto()


class Halt:
    def __init__(
        self,
        crit_success: Optional[List[Criterion]] = None,
        crit_failure: Optional[List[Criterion]] = None,
        freq: int = 1,
    ):
        self.crit_success = crit_success or []
        self.crit_failure = crit_failure or []
        self.freq = freq

    @tf.function
    def check(self, iter: tf.Tensor, step_state: StepState) -> tf.Tensor:

        do_check = tf.equal(tf.math.mod(iter, self.freq), 0)

        def check_criteria():
            failure = tf.constant(False)

            for crit in self.crit_failure:
                crit_met = tf.logical_and(
                    tf.logical_not(failure), crit.check(step_state)
                )
                failure = tf.logical_or(failure, crit_met)

            def check_failure():
                for crit in self.crit_success:
                    crit.metric.reset_metric_prev()

                return tf.constant(HaltStatus.FAILURE.value)

            def check_success():
                success = tf.constant(False)

                for crit in self.crit_success:
                    crit_met = tf.logical_and(
                        tf.logical_not(success), crit.check(step_state)
                    )
                    success = tf.logical_or(success, crit_met)

                if success:
                    for crit in self.crit_success:
                        crit.metric.reset_metric_prev()

                return tf.cond(
                    success,
                    lambda: tf.constant(HaltStatus.SUCCESS.value),
                    lambda: tf.constant(HaltStatus.CONTINUE.value),
                )

            return tf.cond(failure, check_failure, check_success)

        return tf.cond(
            do_check,
            check_criteria,
            lambda: tf.constant(HaltStatus.CONTINUE.value),
        )
