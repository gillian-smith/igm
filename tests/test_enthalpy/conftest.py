#!/usr/bin/env python3
"""
Pytest configuration for enthalpy benchmark tests.
"""

import pytest


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "exp_a: marks tests for Experiment A (transient)"
    )
    config.addinivalue_line(
        "markers", "exp_b: marks tests for Experiment B (steady-state polythermal)"
    )


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="Enable plotting of test results",
    )


@pytest.fixture(scope="session", autouse=True)
def setup_plotting(request):
    """Setup plotting based on command-line option."""
    import os

    if request.config.getoption("--plot"):
        os.environ["IGM_PLOT_TESTS"] = "true"
    else:
        os.environ.setdefault("IGM_PLOT_TESTS", "false")
