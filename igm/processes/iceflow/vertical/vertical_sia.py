#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from .utils import compute_matrices, compute_gauss_quad
from .utils_sia import (
    phi_bed,
    phi_surf,
    grad_phi_bed,
    grad_phi_surf,
    int_phi_bed,
    int_phi_surf,
)
from .vertical import VerticalDiscr


class SIADiscr(VerticalDiscr):

    def _compute_discr(self, cfg: DictConfig) -> None:

        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics

        Nz = cfg_numerics.Nz
        n = cfg_physics.exp_glen

        if Nz != 2:
            raise ValueError("❌ SIA vertical basis only supports Nz=2.")

        x_quad, w_quad = compute_gauss_quad(order=5)

        basis_fct = (
            lambda z: phi_bed(z, n),
            lambda z: phi_surf(z, n),
        )
        basis_fct_grad = (
            lambda z: grad_phi_bed(z, n),
            lambda z: grad_phi_surf(z, n),
        )
        basis_fct_int = (
            lambda z: int_phi_bed(z, n),
            lambda z: int_phi_surf(z, n),
        )

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
