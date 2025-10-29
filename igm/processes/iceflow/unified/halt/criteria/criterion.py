#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod

from ..metrics.metric import Metric, StepState


class Criterion(ABC):

    def __init__(self, metric: Metric):
        self.metric = metric

    @abstractmethod
    def check(self, step_state: StepState) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The check method is not implemented in this class."
        )
