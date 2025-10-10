#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from .vertical import VerticalDiscr
from .utils import compute_matrices, compute_gauss_quad
from .utils_legendre import compute_basis, compute_basis_grad, compute_basis_int


class LegendreDiscr(VerticalDiscr):

    def _compute_discr(self, cfg: DictConfig) -> None:
        cfg_numerics = cfg.processes.iceflow.numerics

        Nz = cfg_numerics.Nz

        x_quad, w_quad = compute_gauss_quad(Nz)

        basis_fct = tuple(compute_basis(i) for i in range(Nz))
        basis_fct_grad = tuple(compute_basis_grad(i) for i in range(Nz))
        basis_fct_int = tuple(compute_basis_int(i) for i in range(Nz))

        V_q, V_q_grad, V_q_int, V_b, V_s, V_bar = compute_matrices(
            basis_fct,
            basis_fct_grad,
            basis_fct_int,
            x_quad,
            w_quad,
        )

        self.w = tf.cast(w_quad, self.dtype)
        self.V_q = tf.cast(V_q, self.dtype)
        self.V_q_grad = tf.cast(V_q_grad, self.dtype)
        self.V_q_int = tf.cast(V_q_int, self.dtype)
        self.V_b = tf.cast(V_b, self.dtype)
        self.V_s = tf.cast(V_s, self.dtype)
        self.V_bar = tf.cast(V_bar, self.dtype)
