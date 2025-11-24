#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import pytest
import numpy as np
import tensorflow as tf
from typing import Tuple

from igm.processes.iceflow.unified.bcs import BoundaryConditions
from igm.processes.iceflow.unified.bcs.bcs import (
    BoundaryCondition,
    FrozenBed,
    PeriodicNS,
    PeriodicWE,
)


def create_UV(shape: Tuple[int, ...] = (1, 3, 3, 1)) -> Tuple[tf.Tensor, tf.Tensor]:
    size = np.prod(shape)
    arr = np.arange(size).reshape(shape).astype(np.float32)
    return tf.constant(arr), tf.constant(arr + 100.0)


def test_base_class_abstract() -> None:
    with pytest.raises(TypeError):
        BoundaryCondition()


def test_base_class_apply() -> None:

    class IncompleteBC(BoundaryCondition):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteBC()


def test_registry_names() -> None:
    assert "frozen_bed" in BoundaryConditions
    assert "periodic_ns" in BoundaryConditions
    assert "periodic_we" in BoundaryConditions
    assert BoundaryConditions["frozen_bed"] == FrozenBed
    assert BoundaryConditions["periodic_ns"] == PeriodicNS
    assert BoundaryConditions["periodic_we"] == PeriodicWE


@pytest.mark.parametrize(
    "shape",
    [
        (4, 3, 3, 1),
        (2, 5, 1, 1),
        (1, 1, 7, 2),
        (1, 5, 2, 3),
    ],
)
def test_frozen_bed(shape: Tuple[int, ...]) -> None:
    Nz = shape[1]
    V_b = tf.one_hot(0, Nz)
    bc = BoundaryConditions["frozen_bed"](V_b)
    U_in, V_in = create_UV(shape)
    U_out, V_out = bc.apply(U_in, V_in)

    assert np.allclose(U_out[:, 0, :, :], 0.0)
    assert np.allclose(V_out[:, 0, :, :], 0.0)

    np.testing.assert_array_equal(U_out[:, 1:, :, :], U_in[:, 1:, :, :])
    np.testing.assert_array_equal(V_out[:, 1:, :, :], V_in[:, 1:, :, :])


@pytest.mark.parametrize(
    "shape",
    [
        (4, 3, 3, 1),
        (2, 5, 1, 1),
        (1, 1, 7, 2),
        (1, 5, 2, 3),
    ],
)
def test_periodic_NS(shape: Tuple[int, ...]) -> None:
    Nz = shape[1]
    V_b = tf.one_hot(0, Nz)
    bc = BoundaryConditions["periodic_ns"](V_b)
    U_in, V_in = create_UV(shape)
    U_out, V_out = bc.apply(U_in, V_in)

    np.testing.assert_array_equal(U_out[:, :, -1, :], U_in[:, :, 0, :])
    np.testing.assert_array_equal(V_out[:, :, -1, :], V_in[:, :, 0, :])


@pytest.mark.parametrize(
    "shape",
    [
        (4, 3, 3, 1),
        (2, 5, 1, 1),
        (1, 1, 7, 2),
        (1, 5, 2, 3),
    ],
)
def test_periodic_WE(shape: Tuple[int, ...]) -> None:
    Nz = shape[1]
    V_b = tf.one_hot(0, Nz)
    bc = BoundaryConditions["periodic_we"](V_b)
    U_in, V_in = create_UV(shape)
    U_out, V_out = bc.apply(U_in, V_in)

    np.testing.assert_array_equal(U_out[:, :, :, -1], U_in[:, :, :, 0])
    np.testing.assert_array_equal(V_out[:, :, :, -1], V_in[:, :, :, 0])
