#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from .metric import Metric, StepState
from igm.utils.math.norms import compute_norm


class MetricGradW(Metric):

    def __init__(self, ord: str = "l2"):
        super().__init__()
        self.ord = ord

    def compute_impl(self, step_state: StepState) -> tf.Tensor:
        return compute_norm(step_state.grad_w, ord=self.ord)
