#!/usr/bin/env python3
"""
Test for Experiment A: Parallel-sided slab (transient)

Based on Kleiner et al. (2015) "Enthalpy benchmark experiments for numerical ice sheet models"
The Cryosphere, 9, 217–228, doi:10.5194/tc-9-217-2015

This experiment tests:
1. Basal boundary condition switching (Neumann/Dirichlet)
2. Basal melt rate calculation
3. Reversibility of the thermal solution
4. Conservation of water volume

This test uses the enthalpy module as a blackbox through its public API
(initialize and update functions), testing the module as it would be used in practice.
"""

import os
import numpy as np
import tensorflow as tf
import pytest
import matplotlib.pyplot as plt

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes import enthalpy
from igm.processes.enthalpy.temperature import compute_temperature

from .analytical_solutions import (
    basal_melt_rate_steady_state,
    basal_temperature_steady_state,
    transient_basal_melt_rate,
)


pytestmark = [pytest.mark.slow, pytest.mark.exp_a]


def _create_test_state(cfg, H, Ts_init, qgeo):
    """Create a simple test state for Experiment A."""
    state = State()

    # Geometry (single column)
    # NOTE: Most fields are Tensors, not Variables - they don't change during the test
    state.thk = H * tf.ones((2, 2), dtype=tf.float32)
    state.dx = tf.constant(10000.0)  # Scalar grid spacing
    state.dX = tf.constant(
        10000.0 * tf.ones((2, 2), dtype=tf.float32)
    )  # Grid spacing tensor for gradients
    state.dt = tf.Variable(0.0)  # Will be updated per timestep
    state.t = tf.Variable(0.0)

    # Velocity fields (all zero for this experiment)
    Nz = cfg.processes.enthalpy.numerics.Nz
    state.U = tf.zeros((Nz, 2, 2), dtype=tf.float32)
    state.V = tf.zeros((Nz, 2, 2), dtype=tf.float32)
    state.W = tf.zeros((Nz, 2, 2), dtype=tf.float32)

    # Surface temperature - Variable because we update it between phases
    state.air_temp = tf.Variable(Ts_init * tf.ones((1, 2, 2), dtype=tf.float32))

    # Bed topography (dummy)
    state.topg = tf.zeros((2, 2), dtype=tf.float32)

    # Geothermal heat flux - Variable for consistency, though not changed in this test
    state.basal_heat_flux = tf.Variable(qgeo * tf.ones((2, 2), dtype=tf.float32))

    return state


def test_exp_a_full():
    """
    Full Experiment A: Three-phase transient simulation.

    Phases:
    - Phase I (0-100 ka): Cold initial phase, Ts = -30°C
    - Phase II (100-150 ka): Warming phase, Ts = -10°C
    - Phase III (150-300 ka): Cooling phase, Ts = -30°C

    Tests reversibility and basal water layer evolution.
    """
    # Time parameters (in years)
    spy = 31556926.0  # Seconds per year
    dt = 200.0
    t_phase1 = 100000.0
    t_phase2 = 50000.0
    t_phase3 = 150000.0

    # Create time array
    t1 = np.arange(0, t_phase1, dt) + dt
    t2 = np.arange(0, t_phase2, dt) + dt
    t3 = np.arange(0, t_phase3, dt) + dt
    tim = np.concatenate([t1, t1[-1] + t2, t1[-1] + t2[-1] + t3])

    # Load and configure
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    # Vertical discretization
    cfg.processes.iceflow.numerics.Nz = 50
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.iceflow.numerics.vert_basis = "lagrange"

    # Match enthalpy vertical settings
    cfg.processes.enthalpy.numerics.Nz = 50
    cfg.processes.enthalpy.numerics.vert_spacing = 1

    # Set enthalpy thermal parameters
    cfg.processes.enthalpy.thermal.K_ratio = (
        1e-5  # Very small temperate ice conductivity
    )
    cfg.processes.enthalpy.surface.T_offset = 0.0  # Use air_temp directly

    # Till hydrology parameters
    cfg.processes.enthalpy.till.hydro.h_water_till_max = (
        200.0  # Allow large water layer
    )
    cfg.processes.enthalpy.till.hydro.drainage_rate = 0.0  # No drainage

    # Experiment parameters
    H = 1000.0  # Ice thickness [m]
    Ts_cold = -30.0 + 273.15  # Cold surface temp [K]
    Ts_warm = -10.0 + 273.15  # Warm surface temp [K]
    qgeo = 0.042  # Geothermal flux [W/m²]

    # Create state
    state = _create_test_state(cfg, H, Ts_cold, qgeo)

    # Initialize enthalpy module
    enthalpy.initialize(cfg, state)

    # Override initial temperature to be isothermal at Ts_cold (cold ice profile)
    # E is set as a Tensor (not Variable) since it's computed by the solver
    c_ice = cfg.processes.enthalpy.thermal.c_ice
    T_ref = cfg.processes.enthalpy.thermal.T_ref
    Nz = cfg.processes.enthalpy.numerics.Nz
    T_init = Ts_cold * tf.ones((Nz, 2, 2), dtype=tf.float32)
    state.E = c_ice * (T_init - T_ref)

    # Compute T and omega from E
    compute_temperature(cfg, state)

    # Track evolution
    results = {
        "time": [],
        "T_base": [],
        "h_water_till": [],
        "basal_melt_rate": [],
        "phase": [],
    }

    # Time integration
    for it, t in enumerate(tim):
        # Determine phase and set surface temperature
        if t <= t_phase1:
            phase = "I"
            state.air_temp.assign(Ts_cold * tf.ones((1, 2, 2)))
        elif t <= t_phase1 + t_phase2:
            phase = "II"
            state.air_temp.assign(Ts_warm * tf.ones((1, 2, 2)))
        else:
            phase = "III"
            state.air_temp.assign(Ts_cold * tf.ones((1, 2, 2)))

        # Update time and timestep
        state.t.assign(t)
        state.dt.assign(dt * spy)  # Convert years to seconds

        # Update enthalpy (blackbox call)
        enthalpy.update(cfg, state)

        # Record evolution
        results["time"].append(t)
        results["T_base"].append((state.T[0, 0, 0] - 273.15).numpy())
        results["h_water_till"].append(state.h_water_till[0, 0].numpy())
        results["basal_melt_rate"].append(state.basal_melt_rate[0, 0].numpy())
        results["phase"].append(phase)

        if it % 100 == 0:
            print(
                f"Phase {phase}, time: {t:8.0f} yr, T_base: {state.T[0, 0, 0].numpy()-273.15:.2f}°C, "
                f"h_water_till: {state.h_water_till[0, 0].numpy():.2f} m, "
                f"basal_melt_rate: {state.basal_melt_rate[0, 0].numpy():.6f} m/a"
            )

    # Convert to arrays
    for key in results:
        if key != "phase":
            results[key] = np.array(results[key])

    # Validation: Check Phase I steady state
    phase1_idx = results["time"] <= t_phase1
    T_base_phase1_final = results["T_base"][phase1_idx][-1]

    # Analytical steady-state basal temperature for Phase I
    k_ice = cfg.processes.enthalpy.thermal.k_ice
    T_base_analytical = basal_temperature_steady_state(Ts_cold, H, qgeo, k_ice) - 273.15

    print(f"\nPhase I final base temperature: {T_base_phase1_final:.2f}°C")
    print(f"Analytical base temperature: {T_base_analytical:.2f}°C")
    assert (
        abs(T_base_phase1_final - T_base_analytical) < 0.1
    ), f"Phase I base temperature mismatch"

    # Validation: Check Phase II steady state
    phase2_idx = (results["time"] > t_phase1) & (results["time"] <= t_phase1 + t_phase2)
    basal_melt_rate_phase2_final = results["basal_melt_rate"][phase2_idx][-1]

    # Analytical steady-state melt rate for Phase II
    T_pmp_ref = cfg.processes.enthalpy.thermal.T_pmp_ref
    L_ice = cfg.processes.enthalpy.thermal.L_ice
    water_density = cfg.processes.enthalpy.drainage.water_density

    basal_melt_rate_analytical = basal_melt_rate_steady_state(
        Ts_warm,
        T_pmp_ref,
        H,
        k_ice,
        qgeo,
        water_density,
        L_ice,
    )

    print(f"\nPhase II final basal_melt_rate: {basal_melt_rate_phase2_final:.6f} m/a")
    print(f"Analytical basal_melt_rate: {basal_melt_rate_analytical:.6f} m/a")
    assert (
        abs(basal_melt_rate_phase2_final - basal_melt_rate_analytical) < 2e-4
    ), f"Phase II basal_melt_rate mismatch: difference = {abs(basal_melt_rate_phase2_final - basal_melt_rate_analytical):.6f}"

    # Validation: Check reversibility (Phase III returns to Phase I state)
    T_base_phase3_final = results["T_base"][-1]
    h_water_till_phase3_final = results["h_water_till"][-1]

    print(f"\nPhase III final base temperature: {T_base_phase3_final:.2f}°C")
    print(f"Phase III final h_water_till: {h_water_till_phase3_final:.2f} m")

    # Optional: plot results
    _plot_exp_a_results(results, t_phase1, t_phase2)

    assert (
        abs(T_base_phase3_final - T_base_analytical) < 0.2
    ), f"Reversibility test failed: final temperature does not match initial"
    assert (
        h_water_till_phase3_final < 1.0
    ), f"Reversibility test failed: till water not fully refrozen"

    print("\n✓ Experiment A passed all validation checks")


@pytest.mark.skip(reason="TMP")
def test_exp_a_phase3_transient():
    """
    Test transient behavior during Phase III (cooling with basal water layer).

    Compares numerical solution with analytical transient solution.
    """
    # Simplified test focusing on Phase IIIa (first 20 ka of cooling phase)
    spy = 31556926.0  # Seconds per year
    dt = 50.0  # Smaller time step for better accuracy
    t_phase3a = 20000.0
    tim = np.arange(0, t_phase3a, dt) + dt

    # Load and configure
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    cfg.processes.iceflow.numerics.Nz = 100  # Higher resolution
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.iceflow.numerics.vert_basis = "lagrange"

    cfg.processes.enthalpy.numerics.Nz = 100
    cfg.processes.enthalpy.numerics.vert_spacing = 1

    cfg.processes.enthalpy.thermal.K_ratio = 1e-5
    cfg.processes.enthalpy.surface.T_offset = 0.0
    cfg.processes.enthalpy.till.hydro.h_water_till_max = 200.0
    cfg.processes.enthalpy.till.hydro.drainage_rate = 0.0

    # Experiment parameters
    H = 1000.0
    Ts_cold = -30.0 + 273.15
    Ts_warm = -10.0 + 273.15
    qgeo = 0.042

    # Create state
    state = _create_test_state(cfg, H, Ts_cold, qgeo)

    # Initialize enthalpy module
    enthalpy.initialize(cfg, state)

    # Set initial temperature: linear profile from warm surface to melting base
    depth = state.enthalpy.vertical_discr.depth
    z_coords = (depth[:, 0, 0] * state.thk[0, 0]).numpy()
    T_pmp_ref = cfg.processes.enthalpy.thermal.T_pmp_ref
    T_init = Ts_warm + (T_pmp_ref - Ts_warm) * (1 - z_coords / H)
    T_init = tf.constant(T_init[:, None, None], dtype=tf.float32)

    c_ice = cfg.processes.enthalpy.thermal.c_ice
    T_ref = cfg.processes.enthalpy.thermal.T_ref
    state.E = c_ice * (T_init - T_ref)

    # Compute T and omega from E
    compute_temperature(cfg, state)

    # Start with some till water (as a Tensor)
    state.h_water_till = 50.0 * tf.ones((2, 2), dtype=tf.float32)

    # Now apply cold surface temperature
    state.air_temp.assign(Ts_cold * tf.ones((1, 2, 2), dtype=tf.float32))

    basal_melt_rates_numerical = []
    times = []

    # Time integration
    for it, t in enumerate(tim):
        state.t.assign(t)
        state.dt.assign(dt * spy)  # Convert years to seconds

        # Update enthalpy (blackbox call)
        enthalpy.update(cfg, state)

        basal_melt_rates_numerical.append(state.basal_melt_rate[0, 0].numpy())
        times.append(t)

    # Compute analytical solution
    k_ice = cfg.processes.enthalpy.thermal.k_ice
    ice_density = cfg.processes.iceflow.physics.ice_density
    c_ice = cfg.processes.enthalpy.thermal.c_ice
    water_density = cfg.processes.enthalpy.drainage.water_density
    L_ice = cfg.processes.enthalpy.thermal.L_ice

    kappa = k_ice / (ice_density * c_ice)
    basal_melt_rates_analytical = transient_basal_melt_rate(
        np.array(times),
        Ts_cold,
        Ts_warm,
        H,
        k_ice,
        qgeo,
        water_density,
        L_ice,
        T_pmp_ref,
        kappa,
    )

    # Compare
    basal_melt_rates_numerical = np.array(basal_melt_rates_numerical)
    rmse = np.sqrt(
        np.mean((basal_melt_rates_numerical - basal_melt_rates_analytical) ** 2)
    )

    print(f"\nPhase IIIa transient test:")
    print(f"RMSE between numerical and analytical: {rmse:.2e} m/a")

    # Relaxed tolerance due to numerical discretization
    assert rmse < 5e-3, f"Transient solution RMSE too large: {rmse}"

    # Optional: plot comparison
    _plot_phase3_comparison(
        times, basal_melt_rates_numerical, basal_melt_rates_analytical
    )

    print("✓ Phase III transient test passed")


def _plot_exp_a_results(results, t_phase1, t_phase2):
    """Plot Experiment A results."""
    plot_enabled = os.environ.get("IGM_PLOT_TESTS", "false").lower() == "true"
    # if not plot_enabled:
    #    return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Temperature
    axes[0].plot(results["time"] / 1000, results["T_base"], "b-", linewidth=1.5)
    axes[0].axvline(
        t_phase1 / 1000, color="r", linestyle="--", alpha=0.5, label="Phase transitions"
    )
    axes[0].axvline((t_phase1 + t_phase2) / 1000, color="r", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Base Temperature [°C]", fontsize=11)
    axes[0].set_xlabel("Time [ka]", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Basal melt rate
    axes[1].plot(
        results["time"] / 1000, results["basal_melt_rate"], "g-", linewidth=1.5
    )
    axes[1].axvline(t_phase1 / 1000, color="r", linestyle="--", alpha=0.5)
    axes[1].axvline((t_phase1 + t_phase2) / 1000, color="r", linestyle="--", alpha=0.5)
    axes[1].axhline(0, color="k", linestyle="-", alpha=0.3)
    axes[1].set_ylabel("Basal Melt Rate [m/a]", fontsize=11)
    axes[1].set_xlabel("Time [ka]", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Till water height
    axes[2].plot(
        results["time"] / 1000, results["h_water_till"], "orange", linewidth=1.5
    )
    axes[2].axvline(t_phase1 / 1000, color="r", linestyle="--", alpha=0.5)
    axes[2].axvline((t_phase1 + t_phase2) / 1000, color="r", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Till Water Height [m]", fontsize=11)
    axes[2].set_xlabel("Time [ka]", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(__file__), "exp_a_results.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Results saved to {out_path}")


def _plot_phase3_comparison(times, numerical, analytical):
    """Plot Phase III transient comparison."""
    plot_enabled = os.environ.get("IGM_PLOT_TESTS", "false").lower() == "true"
    if not plot_enabled:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(times, numerical, "b-", linewidth=1.5, label="Numerical")
    plt.plot(times, analytical, "r--", linewidth=2, label="Analytical")
    plt.axhline(0, color="k", linestyle="-", alpha=0.3)
    plt.xlabel("Time [years]", fontsize=11)
    plt.ylabel("Basal Melt Rate [m/a]", fontsize=11)
    plt.title("Phase IIIa: Transient Basal Melt Rate", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(__file__), "exp_a_phase3_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Comparison saved to {out_path}")


if __name__ == "__main__":
    test_exp_a_full()
    test_exp_a_phase3_transient()
