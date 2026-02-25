#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Generic test runner for Kleiner et al. (2015) benchmarks."""

import os
import pytest

from .experiments import (
    setup_experiment_a,
    setup_experiment_b,
    run_simulation_a,
    run_simulation_b,
    extract_results_b,
)
from .validation import validate_experiment_a, validate_experiment_b
from .plots import plot_exp_a, plot_exp_b


def _print_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")


def _try_plot(plot_fn, results) -> None:
    """Attempt to generate plot, warn on failure."""
    try:
        plot_fn(results)
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


def run_experiment_test(
    monkeypatch: pytest.MonkeyPatch,
    experiment: str,
    dt: float = 200.0,
    Nz_E: int = 50,
) -> None:
    """
    Generic test runner for Kleiner enthalpy benchmarks.

    Args:
        monkeypatch: pytest fixture
        experiment: Experiment name (exp_a, exp_b)
        dt: Time step in years (for exp_a)
        Nz_E: Number of vertical levels
    """
    test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", experiment
    )
    monkeypatch.chdir(test_dir)

    _print_header(f"Running Kleiner Experiment {experiment[-1].upper()}")

    if experiment == "exp_a":
        _run_exp_a(dt, Nz_E)
    elif experiment == "exp_b":
        _run_exp_b(Nz_E)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def _run_exp_a(dt: float, Nz_E: int) -> None:
    """Run and validate Experiment A."""
    cfg, state = setup_experiment_a(dt=dt, Nz_E=Nz_E)
    results = run_simulation_a(cfg, state, dt)

    _try_plot(plot_exp_a, results)

    is_valid, errors = validate_experiment_a(results)

    _print_header("VALIDATION SUMMARY")
    for phase, data in errors.items():
        print(f"  {phase}: {'PASS' if data['valid'] else 'FAIL'}")
    print("=" * 70)

    assert is_valid, f"Validation failed: {errors}"


def _run_exp_b(Nz_E: int) -> None:
    """Run and validate Experiment B."""
    cfg, state = setup_experiment_b(Nz_E=Nz_E)
    run_simulation_b(cfg, state)
    results = extract_results_b(state)

    _try_plot(plot_exp_b, results)

    is_valid, errors = validate_experiment_b(results)

    _print_header("VALIDATION SUMMARY")
    for key, data in errors.items():
        if key in ("overall_valid", "full_analytical"):
            continue
        if isinstance(data, dict) and "valid" in data:
            metric = data.get("rmsd", data.get("error", ""))
            print(f"  {key}: {'PASS' if data['valid'] else 'FAIL'} ({metric:.3f})")
    print("=" * 70)

    assert is_valid, f"Validation failed: {errors}"
