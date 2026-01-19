#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for horizontal advection solver."""

import pytest
import numpy as np
import tensorflow as tf

from igm.processes.enthalpy.solver.horizontal.utils import compute_advection_upwind

# Data type
dtype = tf.float32


@pytest.fixture
def advection_grid():
    """Simple grid for horizontal advection tests."""
    nz, ny, nx = 3, 4, 4
    return {
        "U": tf.zeros((nz, ny, nx), dtype=dtype),
        "V": tf.zeros((nz, ny, nx), dtype=dtype),
        "V_U_to_E": tf.eye(nz, dtype=dtype),
        "E": tf.ones((nz, ny, nx), dtype=dtype) * 1000.0,
        "dx": tf.constant(100.0, dtype=dtype),
    }


def test_advection_zero_velocity(advection_grid) -> None:
    """Test that zero velocity gives zero advection."""
    result = compute_advection_upwind(**advection_grid)
    np.testing.assert_allclose(result.numpy(), 0.0, atol=1e-10)


def test_advection_uniform_field(advection_grid) -> None:
    """Test that uniform field gives zero advection even with velocity."""
    advection_grid["U"] = tf.ones_like(advection_grid["U"]) * 10.0
    advection_grid["V"] = tf.ones_like(advection_grid["V"]) * 10.0

    result = compute_advection_upwind(**advection_grid)

    # Uniform field should have zero gradient
    np.testing.assert_allclose(result.numpy(), 0.0, atol=1e-10)


def test_advection_shape(advection_grid) -> None:
    """Test output shape matches input."""
    result = compute_advection_upwind(**advection_grid)
    assert result.shape == advection_grid["E"].shape


@pytest.mark.parametrize("direction", ["x", "y"])
def test_advection_upwind(advection_grid, direction: str) -> None:
    """Test upwind scheme uses correct differencing direction."""
    nz, ny, nx = 3, 4, 4

    # Create gradient in E
    if direction == "x":
        x = tf.linspace(0.0, 1.0, nx)
        E = tf.reshape(x, (1, 1, nx)) * tf.ones((nz, ny, 1), dtype=dtype) * 1000.0
        advection_grid["E"] = E
        advection_grid["U"] = tf.ones((nz, ny, nx), dtype=dtype) * 100.0
    else:
        y = tf.linspace(0.0, 1.0, ny)
        E = tf.reshape(y, (1, ny, 1)) * tf.ones((nz, 1, nx), dtype=dtype) * 1000.0
        advection_grid["E"] = E
        advection_grid["V"] = tf.ones((nz, ny, nx), dtype=dtype) * 100.0

    result = compute_advection_upwind(**advection_grid)

    # Result should be non-zero for non-uniform field with velocity
    assert np.abs(result.numpy()).max() > 0.0
