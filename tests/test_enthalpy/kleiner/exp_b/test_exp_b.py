#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Test Kleiner Experiment B: Polythermal steady-state."""

import pytest
from tests.test_enthalpy.kleiner.utils import run_experiment_test

pytestmark = pytest.mark.slow


def test_exp_b(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Kleiner Experiment B with Nz=500."""
    run_experiment_test(monkeypatch, experiment="exp_b", Nz=500)
