#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple

from ..metrics import Metric
from ..step_state import StepState
from igm.utils.math.precision import _normalize_precision


class Criterion(ABC):

    def __init__(
        self,
        metric: Metric,
        dtype: str = "float32",
    ):
        self.metric = metric
        self.dtype = _normalize_precision(dtype)
        self.name = "crit"

    @abstractmethod
    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError(
            "❌ The check method is not implemented in this class."
        )

    def reset(self) -> None:
        pass
