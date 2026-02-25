#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Test Kleiner Experiment A: Transient thermal evolution."""

import pytest
from tests.test_enthalpy.kleiner.utils import run_experiment_test

pytestmark = pytest.mark.slow


def test_exp_a(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Kleiner Experiment A with dt=200 years, Nz_E=50."""
    run_experiment_test(monkeypatch, experiment="exp_a", dt=200.0, Nz_E=50)
