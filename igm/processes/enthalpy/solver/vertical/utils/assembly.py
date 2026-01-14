#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple


@tf.function
def assemble_system(
    E: tf.Tensor,
    dt: tf.Tensor,
    dz: tf.Tensor,
    w: tf.Tensor,
    K: tf.Tensor,
    f: tf.Tensor,
    BCB: tf.Tensor,
    VB: tf.Tensor,
    VS: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    TensorFlow function to assemble finite difference system for enthalpy equation.

    Discretizes the 1D vertical enthalpy equation: dE/dt + w·dE/dz = K·d²E/dz² + f
    using implicit time stepping and upwind advection.

    Args:
        E: Current enthalpy field (J kg^-1).
        dt: Time step (s).
        dz: Vertical layer thickness (m).
        w: Vertical velocity field (m s^-1).
        K: Thermal diffusivity field (m^2 s^-1).
        f: Volumetric source term (W kg^-1).
        BCB: Bottom boundary condition flag (1=Neumann, 0=Dirichlet) (-).
        VB: Bottom boundary condition value (J kg^-1 or J kg^-1 m^-1).
        VS: Surface boundary condition value (J kg^-1).

    Returns:
        Tuple of tridiagonal system components (L, M, U, R).
    """
    nz, ny, nx = E.shape
    s = dt * K / (dz * dz)  # Diffusion coefficient

    # Initialize system
    L = tf.zeros((nz - 1, ny, nx))
    M = tf.ones((nz, ny, nx))
    U = tf.zeros((nz - 1, ny, nx))
    R = E + dt * f

    # Assembly: diffusion terms
    M = M + tf.concat([s, tf.zeros((1, ny, nx))], axis=0)
    M = M + tf.concat([tf.zeros((1, ny, nx)), s], axis=0)
    L = L - s
    U = U - s

    # Bottom boundary condition
    M = tf.concat(
        [tf.where(BCB == 1, -tf.ones_like(BCB), tf.ones_like(BCB))[None, :, :], M[1:]],
        axis=0,
    )
    U = tf.concat(
        [tf.where(BCB == 1, tf.ones_like(BCB), tf.zeros_like(BCB))[None, :, :], U[1:]],
        axis=0,
    )
    R = tf.concat([tf.where(BCB == 1, VB * dz[0], VB)[None, :, :], R[1:]], axis=0)

    # Surface boundary condition
    M = tf.concat([M[:-1], tf.ones_like(BCB)[None, :, :]], axis=0)
    L = tf.concat([L[:-1], tf.zeros_like(BCB)[None, :, :]], axis=0)
    R = tf.concat([R[:-1], VS[None, :, :]], axis=0)

    # Upwind advection (implicit)
    wdivdz = dt * (w[1:] + w[:-1]) / (2.0 * dz)
    L = tf.concat([L[:-1] + tf.where(w[1:-1] > 0, -wdivdz[:-1], 0), L[-1:]], axis=0)
    M = tf.concat(
        [M[:1], M[1:-1] + tf.where(w[1:-1] > 0, wdivdz[:-1], -wdivdz[1:]), M[-1:]],
        axis=0,
    )
    U = tf.concat([U[:1], U[1:] + tf.where(w[1:-1] <= 0, wdivdz[1:], 0)], axis=0)

    return L, M, U, R
