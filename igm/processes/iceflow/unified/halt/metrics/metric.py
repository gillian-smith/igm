#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from ..step_state import StepState
from igm.utils.math.precision import _normalize_precision


TypeMetric = Union[tf.Tensor, Tuple[tf.Tensor, ...]]


class Metric(ABC):

    def __init__(
        self,
        dtype: str = "float32",
        normalize_factor: Optional[float] = None,
    ):
        dtype = _normalize_precision(dtype)
        self.metric_prev = tf.Variable(
            initial_value=tf.zeros([], dtype=dtype),
            dtype=dtype,
            trainable=False,
            validate_shape=False,
            shape=tf.TensorShape(None),
        )
        self.metric_prev_init = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.normalize_factor = normalize_factor

    @abstractmethod
    def compute_impl(self, step_state: StepState) -> TypeMetric:
        raise NotImplementedError(
            "❌ The compute method is not implemented in this class."
        )

    def compute(self, step_state) -> TypeMetric:
        metric_value = self.compute_impl(step_state)

        if self.normalize_factor is None:
            return metric_value

        return tf.nest.map_structure(
            lambda m: m / tf.cast(self.normalize_factor, m.dtype), metric_value
        )

    def save_metric(self, metric: TypeMetric) -> None:
        self.metric_prev.assign(metric)
        self.metric_prev_init.assign(True)

    def get_metric_prev(self) -> TypeMetric:
        return self.metric_prev

    def has_metric_prev(self) -> TypeMetric:
        return self.metric_prev_init

    def reset_metric_prev(self) -> None:
        self.metric_prev_init.assign(False)
