#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from .criterion import Criterion
from ..metrics.metric import Metric, StepState


class CriterionAbsTol(Criterion):

    def __init__(self, metric: Metric, tol: float):
        super().__init__(metric)
        self.tol = tol

    def check(self, step_state: StepState) -> tf.Tensor:
        metric_value = self.metric.compute(step_state)
        return tf.greater(self.tol, metric_value)
