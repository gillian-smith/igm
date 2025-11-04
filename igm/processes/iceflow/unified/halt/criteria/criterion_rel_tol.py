#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from typing import Tuple

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState
from igm.utils.math.norms import compute_norm


class CriterionRelTol(Criterion):

    def __init__(self, metric: Metric, tol: float, ord: str):
        super().__init__(metric)
        self.tol = tol
        self.ord = ord
        self.name = "rel_tol"

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        metric_value = self.metric.compute(step_state)

        def first_iteration():
            self.metric.save_metric(metric_value)
            return tf.constant(False), tf.constant(np.nan)

        def compute_relative_change():
            metric_prev = self.metric.get_metric_prev()
            num = compute_norm(metric_value - metric_prev, ord=self.ord)
            denom = compute_norm(metric_prev, ord=self.ord) + 1e-12
            relative_change = num / denom

            is_satisfied = tf.less(relative_change, self.tol)
            self.metric.save_metric(metric_value)
            return is_satisfied, relative_change

        return tf.cond(
            self.metric.has_metric_prev(), compute_relative_change, first_iteration
        )
