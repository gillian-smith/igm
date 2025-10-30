#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..step_state import StepState


class Metric(ABC):

    def __init__(
        self,
        normalize_factor: Optional[float] = None,
        normalize_function: Optional[Callable[[StepState], tf.Tensor]] = None,
    ):
        self.metric_prev = None
        self.normalize_factor = normalize_factor
        self.normalize_function = normalize_function

    @abstractmethod
    def compute_impl(self, step_state: StepState) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The compute method is not implemented in this class."
        )

    def compute(self, step_state: StepState) -> tf.Tensor:
        metric_value = self.compute_impl(step_state)

        if self.normalize_function is not None:
            return metric_value / self.normalize_function(step_state)
        elif self.normalize_factor is not None:
            return metric_value / self.normalize_factor
        else:
            return metric_value

    def save_metric(self, metric: tf.Tensor) -> None:
        self.metric_prev = metric

    def get_metric_prev(self) -> tf.Tensor:
        return self.metric_prev

    def reset_metric_prev(self) -> None:
        self.metric_prev = None
