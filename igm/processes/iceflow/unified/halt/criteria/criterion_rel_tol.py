#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from .criterion import Criterion
from ..metrics.metric import Metric, StepState
from igm.igm.utils.math.norms import compute_norm


class CriterionRelTol(Criterion):

    def __init__(
        self,
        metric: Metric,
        tol: float,
        ord: str = "l2",
        eps: float = 1e-10,
    ):
        super().__init__(metric)
        self.tol = tol
        self.ord = ord
        self.eps = eps

    def check(self, step_state: StepState) -> tf.Tensor:
        metric_value = self.metric.compute(step_state)
        prev_value = self.metric.get_metric_prev()

        if prev_value is None:
            return tf.constant(False)

        diff = metric_value - prev_value
        diff_norm = compute_norm(diff, ord=self.ord)
        metric_norm = compute_norm(metric_value, ord=self.ord)
        rel_change = diff_norm / (metric_norm + self.eps)

        self.metric.save_metric(metric_value)

        return tf.greater(self.tol, rel_change)
