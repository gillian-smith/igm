#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Validation wrappers for Kleiner et al. (2015) benchmarks."""

from typing import Tuple, Dict, Any


def validate_experiment_a(results: Dict[str, Any]) -> Tuple[bool, Dict]:
    """
    Validate Experiment A results against analytical solutions.

    Returns:
        (is_valid, errors_dict)
    """
    from tests.test_enthalpy.kleiner.exp_a.analytical_solutions import validate_exp_a

    time = results["time"]
    T_base = results["T_base"]
    melt_rate = results["melt_rate"]
    till_water = results["till_water"]

    validation = validate_exp_a(time, T_base, melt_rate, till_water)

    is_valid = all(
        [
            validation["phase_i"]["valid"],
            validation["phase_ii"]["valid"],
            validation["phase_iii"]["valid"],
        ]
    )

    return is_valid, validation


def validate_experiment_b(results: Dict[str, Any]) -> Tuple[bool, Dict]:
    """
    Validate Experiment B results against analytical solutions.

    Returns:
        (is_valid, errors_dict)
    """
    from tests.test_enthalpy.kleiner.exp_b.analytical_solutions import validate_exp_b

    validation = validate_exp_b(results)
    is_valid = validation["overall_valid"]

    return is_valid, validation
