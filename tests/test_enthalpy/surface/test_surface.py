#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for surface temperature computation."""

import numpy as np
import tensorflow as tf

from igm.processes.enthalpy.surface.utils import compute_T_s_tf

# Data type
dtype = tf.float32

# Physical constants
T_pmp_ref = tf.constant(273.15, dtype=dtype)


def test_T_s_cold() -> None:
    """Test cold air gives cold surface temperature."""
    T_air = tf.constant([[[-20.0]]], dtype=dtype)
    T_offset = tf.constant(0.0, dtype=dtype)

    T_s = compute_T_s_tf(T_air, T_offset, T_pmp_ref)

    np.testing.assert_allclose(T_s.numpy()[0, 0], 253.15, rtol=1e-5)


def test_T_s_capped() -> None:
    """Test warm air is capped at pressure melting point."""
    T_air = tf.constant([[[10.0]]], dtype=dtype)
    T_offset = tf.constant(0.0, dtype=dtype)

    T_s = compute_T_s_tf(T_air, T_offset, T_pmp_ref)

    np.testing.assert_allclose(T_s.numpy()[0, 0], T_pmp_ref.numpy(), rtol=1e-6)


def test_T_s_offset() -> None:
    """Test temperature offset is applied before capping."""
    T_air = tf.constant([[[-10.0]]], dtype=dtype)
    T_offset = tf.constant(-5.0, dtype=dtype)

    T_s = compute_T_s_tf(T_air, T_offset, T_pmp_ref)

    np.testing.assert_allclose(T_s.numpy()[0, 0], 258.15, rtol=1e-5)


def test_T_s_average() -> None:
    """Test surface temperature averages over first axis."""
    T_air = tf.constant([[[-30.0]], [[-10.0]]], dtype=dtype)
    T_offset = tf.constant(0.0, dtype=dtype)

    T_s = compute_T_s_tf(T_air, T_offset, T_pmp_ref)

    np.testing.assert_allclose(T_s.numpy()[0, 0], 253.15, rtol=1e-5)


def test_T_s_shape() -> None:
    """Test output shape after time averaging."""
    n_time, ny, nx = 4, 3, 5
    T_air = tf.ones((n_time, ny, nx), dtype=dtype) * -10.0
    T_offset = tf.constant(0.0, dtype=dtype)

    T_s = compute_T_s_tf(T_air, T_offset, T_pmp_ref)

    assert T_s.shape == (ny, nx)
