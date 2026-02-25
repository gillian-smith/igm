#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for till friction computations."""

import pytest
import numpy as np
import tensorflow as tf

from igm.processes.enthalpy.till.friction.utils import (
    compute_phi_tf,
    compute_tauc_tf,
    compute_slidingco_tf,
)

# Data type
dtype = tf.float32


@pytest.mark.parametrize(
    "bed,phi_expected",
    [
        (-1000.0, 5.0),
        (-500.0, 17.5),
        (0.0, 30.0),
        (500.0, 30.0),
    ],
)
def test_phi_bed(bed: float, phi_expected: float) -> None:
    """Test friction angle interpolation with bed elevation."""
    bed_tf = tf.constant([[bed]], dtype=dtype)
    bed_min = tf.constant(-1000.0, dtype=dtype)
    bed_max = tf.constant(0.0, dtype=dtype)
    phi_min = tf.constant(5.0, dtype=dtype)
    phi_max = tf.constant(30.0, dtype=dtype)

    phi = compute_phi_tf(bed_tf, bed_min, bed_max, phi_min, phi_max)

    np.testing.assert_allclose(phi.numpy()[0, 0], phi_expected, rtol=1e-5)


def test_phi_shape() -> None:
    """Test output shape matches input."""
    ny, nx = 5, 4
    bed = tf.ones((ny, nx), dtype=dtype) * -500.0
    bed_min = tf.constant(-1000.0, dtype=dtype)
    bed_max = tf.constant(0.0, dtype=dtype)
    phi_min = tf.constant(5.0, dtype=dtype)
    phi_max = tf.constant(30.0, dtype=dtype)

    phi = compute_phi_tf(bed, bed_min, bed_max, phi_min, phi_max)

    assert phi.shape == (ny, nx)


def test_tauc_mohr_coulomb() -> None:
    """Test yield stress follows Mohr-Coulomb criterion."""
    N = tf.constant([[1e6]], dtype=dtype)
    phi = tf.constant([[30.0]], dtype=dtype)
    h_ice = tf.constant([[1000.0]], dtype=dtype)
    tauc_ice_free = tf.constant(1e10, dtype=dtype)
    tauc_min = tf.constant(0.0, dtype=dtype)
    tauc_max = tf.constant(1e10, dtype=dtype)

    tauc = compute_tauc_tf(N, phi, h_ice, tauc_ice_free, tauc_min, tauc_max)

    tauc_expected = 1e6 * np.tan(30.0 * np.pi / 180.0)
    np.testing.assert_allclose(tauc.numpy()[0, 0], tauc_expected, rtol=1e-5)


def test_tauc_ice_free() -> None:
    """Test ice-free areas use tauc_ice_free."""
    N = tf.constant([[1e6]], dtype=dtype)
    phi = tf.constant([[30.0]], dtype=dtype)
    h_ice = tf.constant([[0.0]], dtype=dtype)
    tauc_ice_free = tf.constant(1e10, dtype=dtype)
    tauc_min = tf.constant(0.0, dtype=dtype)
    tauc_max = tf.constant(1e12, dtype=dtype)

    tauc = compute_tauc_tf(N, phi, h_ice, tauc_ice_free, tauc_min, tauc_max)

    np.testing.assert_allclose(tauc.numpy()[0, 0], tauc_ice_free, rtol=1e-5)


def test_tauc_max() -> None:
    """Test yield stress is clamped to bounds."""
    N = tf.constant([[1e6]], dtype=dtype)
    phi = tf.constant([[45.0]], dtype=dtype)
    h_ice = tf.constant([[1000.0]], dtype=dtype)
    tauc_ice_free = tf.constant(1e10, dtype=dtype)
    tauc_min = tf.constant(1e5, dtype=dtype)
    tauc_max = tf.constant(5e5, dtype=dtype)

    tauc = compute_tauc_tf(N, phi, h_ice, tauc_ice_free, tauc_min, tauc_max)

    np.testing.assert_allclose(tauc.numpy()[0, 0], tauc_max, rtol=1e-5)


def test_slidingco_from_tauc() -> None:
    """Test sliding coefficient computation."""
    tauc = tf.constant([[1e5]], dtype=dtype)
    u_ref = tf.constant(100.0, dtype=dtype)
    m = tf.constant(3.0, dtype=dtype)

    C = compute_slidingco_tf(tauc, u_ref, m)

    C_expected = 1e5 * (100.0 ** (-1.0 / 3.0)) * 1e-6
    np.testing.assert_allclose(C.numpy()[0, 0], C_expected, rtol=1e-5)


def test_slidingco_shape() -> None:
    """Test output shape matches input."""
    ny, nx = 5, 4
    tauc = tf.ones((ny, nx), dtype=dtype) * 1e5
    u_ref = tf.constant(100.0, dtype=dtype)
    m = tf.constant(3.0, dtype=dtype)

    C = compute_slidingco_tf(tauc, u_ref, m)

    assert C.shape == (ny, nx)
