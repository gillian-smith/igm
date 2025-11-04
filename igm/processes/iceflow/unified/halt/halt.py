#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from typing import List, Optional
from enum import Enum, auto
from dataclasses import dataclass

from .criteria import Criterion
from .step_state import StepState


class HaltStatus(Enum):
    CONTINUE = auto()
    SUCCESS = auto()
    FAILURE = auto()
    COMPLETED = auto()


@dataclass
class HaltState:
    status: tf.Tensor
    criterion_values: List[tf.Tensor]
    criterion_satisfied: List[tf.Tensor]

    @staticmethod
    def empty():
        return HaltState(
            status=tf.constant(HaltStatus.CONTINUE.value),
            criterion_values=[],
            criterion_satisfied=[],
        )


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
        self.criterion_names = self._build_criterion_names()

    def _build_criterion_names(self) -> List[str]:
        names = []
        for crit in self.crit_success:
            names.append(crit.name)
        return names

    def check(self, iter: tf.Tensor, step_state: StepState):
        do_check = tf.equal(tf.math.mod(iter, self.freq), 0)

        def check_criteria():
            success_values = []
            success_satisfied = []

            # Failure criteria (checked but not returned for display)
            failure = tf.constant(False)
            for crit in self.crit_failure:
                is_sat, val = crit.check(step_state)
                failure = tf.logical_or(failure, is_sat)

            # Success criteria (checked and returned for display)
            success = tf.constant(False)
            for crit in self.crit_success:
                is_sat, val = crit.check(step_state)
                success_values.append(val)
                success_satisfied.append(is_sat)
                success = tf.logical_or(success, is_sat)

            # Determine status
            if failure:
                for crit in self.crit_success:
                    crit.reset()
                for crit in self.crit_failure:
                    crit.reset()
                return (
                    tf.constant(HaltStatus.FAILURE.value),
                    success_values,
                    success_satisfied,
                )

            if success:
                for crit in self.crit_success:
                    crit.reset()
                for crit in self.crit_failure:
                    crit.reset()
                return (
                    tf.constant(HaltStatus.SUCCESS.value),
                    success_values,
                    success_satisfied,
                )

            return (
                tf.constant(HaltStatus.CONTINUE.value),
                success_values,
                success_satisfied,
            )

        def no_check():
            n_success_crit = len(self.crit_success)
            return (
                tf.constant(HaltStatus.CONTINUE.value),
                [tf.constant(np.nan)] * n_success_crit,
                [tf.constant(False)] * n_success_crit,
            )

        return tf.cond(do_check, check_criteria, no_check)
