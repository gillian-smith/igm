#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import pytest
import numpy as np
from omegaconf import DictConfig

import igm
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.iceflow.vertical import VerticalDiscrs
from igm.processes.iceflow.vertical.utils import compute_gauss_quad
from igm.processes.iceflow.vertical.vertical_sia import SIADiscr


@pytest.fixture
def cfg() -> DictConfig:
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    cfg.processes.iceflow.numerics.Nz = 2
    cfg.processes.iceflow.physics.exp_glen = 3.0
    return cfg


def test_registry_name() -> None:
    assert "sia" in VerticalDiscrs
    assert VerticalDiscrs["sia"] == SIADiscr


def test_registry_instantiate(cfg: DictConfig) -> None:
    assert isinstance(VerticalDiscrs["sia"](cfg), SIADiscr)


def test_nz(cfg: DictConfig) -> None:
    cfg.processes.iceflow.numerics.Nz = 3
    with pytest.raises(ValueError):
        SIADiscr(cfg)


def test_matrices_shapes(cfg: DictConfig) -> None:
    discr = SIADiscr(cfg)

    assert discr.w.shape == (5,)
    assert discr.V_q.shape == (5, 2)
    assert discr.V_q_grad.shape == (5, 2)
    assert discr.V_q_int.shape == (5, 2)
    assert discr.V_b.shape == (2,)
    assert discr.V_s.shape == (2,)
    assert discr.V_bar.shape == (2,)


def test_matrices_properties(cfg: DictConfig) -> None:
    discr = SIADiscr(cfg)

    sum_rows_V_q = np.sum(discr.V_q.numpy(), axis=1)
    sum_rows_V_q_grad = np.sum(discr.V_q_grad.numpy(), axis=1)

    np.testing.assert_allclose(sum_rows_V_q, 1.0)
    np.testing.assert_allclose(sum_rows_V_q_grad, 0.0)


def test_matrices_example(cfg: DictConfig) -> None:
    cfg.processes.iceflow.physics.exp_glen = 0.0
    discr = SIADiscr(cfg)

    w_computed = discr.w.numpy()
    V_q_computed = discr.V_q.numpy()
    V_q_grad_computed = discr.V_q_grad.numpy()
    V_q_int_computed = discr.V_q_int.numpy()
    V_b_computed = discr.V_b.numpy()
    V_s_computed = discr.V_s.numpy()
    V_bar_computed = discr.V_bar.numpy()

    z_quad, w_quad = compute_gauss_quad(order=5)
    z = z_quad.numpy()
    w = w_quad.numpy()

    w_expected = w
    V_q_expected = np.stack([1.0 - z, z], axis=1)
    V_q_grad_expected = np.tile(np.array([-1.0, 1.0]), (len(z), 1))
    V_q_int_expected = np.stack([z - 0.5 * z**2, 0.5 * z**2], axis=1)
    V_b_expected = np.array([1.0, 0.0])
    V_s_expected = np.array([0.0, 1.0])
    V_bar_expected = np.array([np.sum(w * (1.0 - z)), np.sum(w * z)])

    rtol = 1e-5
    atol = 1e-7

    np.testing.assert_allclose(w_computed, w_expected, rtol, atol)
    np.testing.assert_allclose(V_q_computed, V_q_expected, rtol, atol)
    np.testing.assert_allclose(V_q_grad_computed, V_q_grad_expected, rtol, atol)
    np.testing.assert_allclose(V_q_int_computed, V_q_int_expected, rtol, atol)
    np.testing.assert_allclose(V_b_computed, V_b_expected, rtol, atol)
    np.testing.assert_allclose(V_s_computed, V_s_expected, rtol, atol)
    np.testing.assert_allclose(V_bar_computed, V_bar_expected, rtol, atol)
