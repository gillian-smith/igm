#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Callable, Optional, Tuple

from igm.utils.math.precision import normalize_precision

from .enthalpy import VerticalDiscrEnthalpy, compute_discr_enthalpy


class VerticalDiscr(ABC):
    """
    Abstract vertical discretization.

    Attributes
    ----------
    w : tf.Tensor
        Quadrature weights, shape (Nq,).
    zeta : tf.Tensor
        Quadrature points in reference element [0, 1], shape (Nq,).
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
    enthalpy : Optional[VerticalDiscrEnthalpy]
        Enthalpy vertical discretization and coupling matrices (if enthalpy is enabled).
    """

    w: tf.Tensor
    zeta: tf.Tensor
    V_q: tf.Tensor
    V_q_grad: tf.Tensor
    V_q_int: tf.Tensor
    V_b: tf.Tensor
    V_s: tf.Tensor
    V_bar: tf.Tensor
    enthalpy: Optional[VerticalDiscrEnthalpy]

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize vertical discretization."""
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype = normalize_precision(precision)
        basis_fct = self._compute_discr(cfg)

        if "enthalpy" in cfg.processes:
            self.enthalpy = compute_discr_enthalpy(cfg, basis_fct, self.dtype)

    @abstractmethod
    def _compute_discr(
        self, cfg: DictConfig
    ) -> Tuple[Callable[[tf.Tensor], tf.Tensor], ...]:
        """Compute discretization matrices. Returns basis functions."""
        raise NotImplementedError(
            "❌ The discretization is not implemented in this class."
        )
