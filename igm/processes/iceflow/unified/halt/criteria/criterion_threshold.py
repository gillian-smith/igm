#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState


class CriterionThreshold(Criterion):

    def __init__(self, metric: Metric, threshold: float):
        super().__init__(metric)
        self.threshold = threshold

    def check(self, step_state: StepState) -> tf.Tensor:
        metric_value = self.metric.compute(step_state)
        return tf.reduce_any(tf.abs(metric_value) > self.threshold)
