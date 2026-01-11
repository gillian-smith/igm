"""
Experiment A: Parallel-sided slab (transient)

Tests for the transient enthalpy benchmark experiment from
Kleiner et al. (2015) "Enthalpy benchmark experiments for numerical ice sheet models"
"""

from .analytical_solutions import (
    basal_melt_rate_steady_state,
    basal_temperature_steady_state,
    transient_basal_melt_rate,
    melting_to_freezing_transition_time,
)

from .test_exp_a import (
    test_exp_a_full,
    test_exp_a_phase3_transient,
)

__all__ = [
    'basal_melt_rate_steady_state',
    'basal_temperature_steady_state',
    'transient_basal_melt_rate',
    'melting_to_freezing_transition_time',
    'test_exp_a_full',
    'test_exp_a_phase3_transient',
]
