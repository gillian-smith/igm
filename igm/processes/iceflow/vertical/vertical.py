#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from igm.utils.math.precision import _normalize_precision

class VerticalDiscr(ABC):
    """
    Abstract vertical discretization.

    Attributes
    ----------
    w : tf.Tensor
        Quadrature weights, shape (Nq,).
    V_q : tf.Tensor
        Map DOFs → values at quadrature points, shape (Nq, Ndof).
    V_q_grad : tf.Tensor
        Map DOFs → vertical gradients at quad points, shape (Nq, Ndof).
    V_q_int : tf.Tensor
        Map DOFs → vertical integral at quad points, shape (Nq, Ndof).
    V_b : tf.Tensor
        Map DOFs → basal value (zeta=0), shape (Ndof,).
    V_s : tf.Tensor
        Map DOFs → surface value (zeta=1), shape (Ndof,).
    V_bar : tf.Tensor
        Map DOFs → vertical average, shape (Ndof,).
    """

    w: tf.Tensor
    V_q: tf.Tensor
    V_q_grad: tf.Tensor
    V_q_int: tf.Tensor
    V_b: tf.Tensor
    V_s: tf.Tensor
    V_bar: tf.Tensor

    def __init__(self, cfg: DictConfig) -> None:
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype = _normalize_precision(precision)
        self._compute_discr(cfg)

    @abstractmethod
    def _compute_discr(self, cfg: DictConfig) -> None:
        raise NotImplementedError(
            "❌ The discretization is not implemented in this class."
        )
