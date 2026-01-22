#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple
from omegaconf import DictConfig

from igm.utils.math.precision import normalize_precision


class HorizontalDiscr(ABC):
    """
    Abstract horizontal discretization.

    Attributes
    ----------
    w_h: tf.Tensor
        Quadrature weights, shape (Nq,).
    """

    w_h: tf.Tensor

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize horizontal discretization."""
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype = normalize_precision(precision)
        self._compute_discr(cfg)

    @abstractmethod
    def _compute_discr(self, cfg: DictConfig) -> None:
        """Compute discretization attributes."""
        raise NotImplementedError(
            "❌ The discretization is not implemented in this class."
        )

    @abstractmethod
    def grad_h(self, X: tf.Tensor, dX: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Gradients at quadrature points.

        Input X: (batch, ...,  Ny, Nx)
        Input dX: (batch, Ny, Nx)
        Output: (batch, Nq, ..., Ny-1, Nx-1)
        """
        raise NotImplementedError(
            "❌ The horizontal gradient is not implemented in this class."
        )

    @abstractmethod
    def interp_h(self, X: tf.Tensor) -> tf.Tensor:
        """
        Interpolate to quadrature points.

        Input X: (batch, ...,  Ny, Nx)
        Input dX: (batch, Ny, Nx)
        Output: (batch, Nq, ..., Ny-1, Nx-1)
        """
        raise NotImplementedError(
            "❌ The interpolation is not implemented in this class."
        )
