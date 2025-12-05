#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, List

from ..bcs import BoundaryCondition
from igm.utils.math.precision import _normalize_precision

TV = Union[tf.Tensor, tf.Variable]


class Mapping(ABC):
    def __init__(
        self,
        bcs: List[BoundaryCondition],
        precision: str = "float32",
    ) -> None:
        self.apply_bcs = bcs
        self.precision = _normalize_precision(precision)

    def set_inputs(self, inputs: tf.Tensor) -> None:
        self.inputs = inputs

    @abstractmethod
    def get_UV_impl(self) -> Tuple[TV, TV]:
        pass

    @tf.function(jit_compile=True)
    def get_UV(self, inputs: tf.Tensor) -> Tuple[TV, TV]:
        self.set_inputs(inputs)
        U, V = self.get_UV_impl()
        for apply_bc in self.apply_bcs:
            U, V = apply_bc(U, V)
        return U, V

    @abstractmethod
    def get_w(self) -> Any:
        pass

    @abstractmethod
    def set_w(self, w: Any) -> None:
        pass

    @abstractmethod
    def copy_w(self, w: Any) -> Any:
        pass

    @abstractmethod
    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def flatten_w(self, w: Any) -> tf.Tensor:
        pass

    @abstractmethod
    def unflatten_w(self, w_flat: tf.Tensor) -> Any:
        pass

    @abstractmethod
    def check_halt_criterion(self, iteration: int, cost: tf.Tensor) -> tf.Tensor:
        pass

    def on_minimize_start(self, iter_max: int) -> None:
        pass

    @tf.function
    def on_step_end(self, iteration: tf.Tensor) -> None:
        pass
