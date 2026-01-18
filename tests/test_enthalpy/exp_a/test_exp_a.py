#!/usr/bin/env python3
"""
Test Experiment A from Kleiner et al. (2015):
"Enthalpy benchmark experiments for numerical ice sheet models"

Validates transient thermal evolution with three phases:
- Phase I (0-100 ky): Equilibration with cold surface (-30°C)
- Phase II (100-150 ky): Warming pulse (-5°C)
- Phase III (150-300 ky): Return to cold surface (-30°C)
"""

import os
import numpy as np
import tensorflow as tf
import pytest
import matplotlib.pyplot as plt

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.enthalpy import enthalpy
from analytical_solutions import validate_exp_a


pytestmark = pytest.mark.slow


def test_exp_a_transient_thermal_evolution():
    """Test Experiment A with dt=200 years."""
    dt = 200.0

    # Setup and run simulation
    cfg, state = _setup_experiment_a(dt)
    time, T_base, melt_rate, till_water = _run_simulation(cfg, state, dt)

    # Validate and plot results (plot saved even if validation fails)
    try:
        _validate_phases(time, T_base, melt_rate, till_water)
    finally:
        _plot_results(time, T_base, melt_rate, till_water)
        _plot_analytical_comparison(time, T_base, melt_rate, till_water)


def _setup_experiment_a(dt):
    """Initialize configuration and state for Experiment A."""
    # Load configuration
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    # Configure vertical discretization
    Nz_E = 50
    Nz_U = 10
    cfg.processes.iceflow.numerics.Nz = Nz_U
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.enthalpy.numerics.Nz = Nz_E
    cfg.processes.enthalpy.numerics.vert_spacing = 1

    # Configure enthalpy solver for Experiment A
    cfg.processes.enthalpy.thermal.K_ratio = 1.0e-5
    cfg.processes.enthalpy.till.hydro.h_water_till_max = 200.0
    cfg.processes.enthalpy.till.hydro.drainage_rate = 0.0
    cfg.processes.enthalpy.drainage.drain_ice_column = False
    cfg.processes.enthalpy.solver.allow_basal_refreezing = True
    cfg.processes.enthalpy.solver.correct_w_for_melt = False

    # Initialize state
    Ny, Nx = 2, 2
    state = State()
    state.thk = tf.Variable(1000.0 * tf.ones((Ny, Nx)), trainable=False)
    state.topg = tf.Variable(tf.zeros((Ny, Nx)), trainable=False)
    state.usurf = state.topg + state.thk
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(dt, trainable=False)
    state.air_temp = tf.Variable(-30.0 * tf.ones((1, Ny, Nx)), trainable=False)
    state.basal_heat_flux = 0.042 * tf.ones((Ny, Nx))
    state.U = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)
    state.V = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)
    state.W = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)
    state.dx = tf.Variable(1000.0, trainable=False)
    state.dX = tf.Variable(tf.ones((Ny, Nx)) * 1000.0, trainable=False)

    # Initialize enthalpy module with cold ice (-30°C)
    enthalpy.initialize(cfg, state)
    T_init = 243.15 * tf.ones((Nz_E, Ny, Nx))
    state.E = cfg.processes.enthalpy.thermal.c_ice * (
        T_init - cfg.processes.enthalpy.thermal.T_ref
    )
    state.T = T_init
    state.omega = tf.zeros_like(state.E)
    state.h_water_till = tf.zeros((Ny, Nx))

    return cfg, state


def _run_simulation(cfg, state, dt):
    """Run the thermal evolution simulation."""
    ttf = 300000.0  # 300 ky total
    tim = np.arange(0, ttf, dt) + dt

    time, T_base, melt_rate, till_water = [], [], [], []

    print(f"\n{'='*70}")
    print(f"Running Experiment A: dt={dt:.0f} years, duration={ttf/1000:.0f} ky")
    print(f"{'='*70}")
    print(
        f"{'Time':>8} | {'Phase':^10} | {'T_surf':>7} | {'T_base':>7} | {'Melt':>9} | {'Till':>6}"
    )
    print(
        f"{'[ky]':>8} | {' ':^10} | {'[°C]':>7} | {'[°C]':>7} | {'[m/y]':>9} | {'[m]':>6}"
    )
    print(f"{'-'*70}")

    for it, t in enumerate(tim):
        state.t.assign(t)

        # Apply surface temperature forcing
        T_surface = -5.0 if 100000.0 <= t < 150000.0 else -30.0
        state.air_temp.assign(T_surface * tf.ones((1, 2, 2)))

        # Update enthalpy
        enthalpy.update(cfg, state)

        # Record results
        time.append(t)
        T_base.append(state.T[0, 0, 0].numpy() - 273.15)
        melt_rate.append(state.basal_melt_rate[0, 0].numpy())
        till_water.append(state.h_water_till[0, 0].numpy())

        # Determine phase
        if t < 100000.0:
            phase = "I"
        elif t < 150000.0:
            phase = "II"
        else:
            phase = "III"

        # Print every 50 timesteps
        if it % 50 == 0:
            print(
                f"{t/1000:8.1f} | {phase:^10} | {T_surface:7.1f} | {T_base[-1]:7.2f} | "
                f"{melt_rate[-1]:9.6f} | {till_water[-1]:6.2f}"
            )

    print(f"{'-'*70}\n")
    return (np.array(time), np.array(T_base), np.array(melt_rate), np.array(till_water))


def _validate_phases(time, T_base, melt_rate, till_water):
    """Validate thermal evolution across three phases."""
    pmp_base = -0.7  # Pressure melting point at 1000m depth

    print("\n" + "=" * 60)
    print("PHASE VALIDATION")
    print("=" * 60)

    # Phase I: Equilibration (0-100 ky)
    print("\nPhase I: Equilibration (0-100 ky)")
    mask_i = time < 100000.0
    T_i_init, T_i_final = T_base[mask_i][0], T_base[mask_i][-1]
    warming_i = T_i_final - T_i_init

    print(f"  T_base: {T_i_init:.2f}°C → {T_i_final:.2f}°C (Δ={warming_i:.2f}°C)")
    print(f"  Max melt rate: {melt_rate[mask_i].max():.6f} m/y")

    assert (
        warming_i > 0
    ), f"Base should warm from geothermal flux (got {warming_i:.2f}°C)"
    assert (
        T_i_final < pmp_base
    ), f"Base should stay below PMP (got {T_i_final:.2f}°C vs {pmp_base:.2f}°C)"
    assert (
        melt_rate[mask_i].max() < 0.001
    ), f"Melt rate should be minimal (got {melt_rate[mask_i].max():.6f} m/y)"
    print("  ✓ Phase I passed")

    # Phase II: Warming pulse (100-150 ky)
    print("\nPhase II: Warming Pulse (100-150 ky)")
    mask_ii = (time >= 100000.0) & (time < 150000.0)
    T_ii_init, T_ii_final = T_base[mask_ii][0], T_base[mask_ii][-1]
    warming_ii = T_ii_final - T_ii_init
    melt_ii_final = melt_rate[mask_ii][-1]
    till_ii_final = till_water[mask_ii][-1]

    print(f"  T_base: {T_ii_init:.2f}°C → {T_ii_final:.2f}°C (Δ={warming_ii:.2f}°C)")
    print(f"  Final melt rate: {melt_ii_final:.6f} m/y")
    print(f"  Final till water: {till_ii_final:.2f} m")

    warming_rate_i = warming_i / 100.0
    warming_rate_ii = warming_ii / 50.0
    assert (
        warming_rate_ii >= warming_rate_i * 0.9
    ), f"Warming rate Phase II should match Phase I ({warming_rate_ii:.3f} vs {warming_rate_i:.3f}°C/ky)"
    assert (
        T_ii_final > T_i_final
    ), f"Phase II final T should exceed Phase I ({T_ii_final:.2f}°C vs {T_i_final:.2f}°C)"
    assert (
        T_ii_final > pmp_base - 1.0
    ), f"Base should approach PMP ({T_ii_final:.2f}°C vs {pmp_base:.2f}°C)"

    if T_ii_final >= pmp_base:
        assert (
            melt_ii_final > 0
        ), f"Melt should occur at PMP (got {melt_ii_final:.6f} m/y)"
        assert (
            till_ii_final > 0
        ), f"Till water should accumulate (got {till_ii_final:.2f} m)"
    print("  ✓ Phase II passed")

    # Phase III: Cooling (150-300 ky)
    print("\nPhase III: Cooling (150-300 ky)")
    mask_iii = time >= 150000.0
    T_iii_init, T_iii_final = T_base[mask_iii][0], T_base[mask_iii][-1]
    cooling_iii = T_iii_init - T_iii_final
    melt_iii_final = melt_rate[mask_iii][-1]

    print(
        f"  T_base: {T_iii_init:.2f}°C → {T_iii_final:.2f}°C (Δ={-cooling_iii:.2f}°C)"
    )
    print(f"  Final melt rate: {melt_iii_final:.6f} m/y")

    assert (
        cooling_iii >= -0.1
    ), f"Base should not warm significantly (got {cooling_iii:.2f}°C cooling)"
    assert (
        T_iii_final <= T_ii_final
    ), f"Phase III final T should not exceed Phase II ({T_iii_final:.2f}°C vs {T_ii_final:.2f}°C)"
    assert (
        melt_iii_final <= melt_ii_final
    ), f"Melt rate should decrease ({melt_iii_final:.6f} vs {melt_ii_final:.6f} m/y)"
    print("  ✓ Phase III passed")

    # Analytical validation
    print("\n" + "=" * 60)
    print("ANALYTICAL VALIDATION")
    print("=" * 60)

    analytical = validate_exp_a(time, T_base, melt_rate, till_water)

    print(f"\nPhase I: T_base comparison")
    print(f"  Numerical: {analytical['phase_i']['T_numerical']:.2f}°C")
    print(f"  Analytical: {analytical['phase_i']['T_analytical']:.2f}°C")
    print(f"  Error: {analytical['phase_i']['error']:.3f} K")
    assert analytical["phase_i"]["valid"], "Phase I analytical validation failed"
    print("  ✓ Analytical validation passed")

    print(f"\nPhase II: Melt rate comparison")
    print(f"  Numerical: {analytical['phase_ii']['melt_numerical']:.6f} m/y")
    print(f"  Analytical: {analytical['phase_ii']['melt_analytical']:.6f} m/y")
    print(f"  Error: {analytical['phase_ii']['error']:.6f} m/y")
    assert analytical["phase_ii"]["valid"], "Phase II analytical validation failed"
    print("  ✓ Analytical validation passed")

    print(f"\nPhase III: Transient melt rate comparison")
    print(f"  Mean relative error: {analytical['phase_iii']['mean_error']:.2%}")
    assert analytical["phase_iii"]["valid"], "Phase III analytical validation failed"
    print("  ✓ Analytical validation passed")

    print("\n" + "=" * 60)
    print("ALL PHASES VALIDATED SUCCESSFULLY")
    print("=" * 60)


def _plot_results(time, T_base, melt_rate, till_water):
    """Generate diagnostic plots."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    time_ky = time / 1000

    # Temperature
    ax1.plot(time_ky, T_base, "b-", linewidth=2)
    ax1.axhline(-0.7, color="gray", linestyle=":", alpha=0.5, label="PMP")
    ax1.axvline(100, color="r", linestyle="--", alpha=0.5, label="Phase transitions")
    ax1.axvline(150, color="r", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Base Temperature [°C]")
    ax1.set_title("Experiment A: Transient Thermal Evolution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Melt rate
    ax2.plot(time_ky, melt_rate, "b-", linewidth=2)
    ax2.axvline(100, color="r", linestyle="--", alpha=0.5)
    ax2.axvline(150, color="r", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Basal Melt Rate [m/y]")
    ax2.grid(True, alpha=0.3)

    # Till water
    ax3.plot(time_ky, till_water, "g-", linewidth=2)
    ax3.axvline(100, color="r", linestyle="--", alpha=0.5)
    ax3.axvline(150, color="r", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Till Water Height [m]")
    ax3.set_xlabel("Time [ky]")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = "exp_a.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"\n✓ Results saved to {output_file}")


def _plot_analytical_comparison(time, T_base, melt_rate, till_water):
    """Generate Phase III analytical comparison plot."""
    analytical = validate_exp_a(time, T_base, melt_rate, till_water)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Phase III: Transient melt rate comparison
    melt_numerical = analytical["phase_iii"]["melt_numerical"]
    melt_analytical = analytical["phase_iii"]["melt_analytical"]
    time_iii = analytical["phase_iii"]["time"]

    # Skip first point if it's anomalous
    start_idx = (
        1
        if melt_analytical[0] < 0 or abs(melt_analytical[0] - melt_analytical[1]) > 0.01
        else 0
    )

    # Hard cutoff at 225 ky (75 ky into Phase III)
    max_time_iii = 75000.0  # years since start of Phase III
    time_cutoff_mask = time_iii <= max_time_iii
    if time_cutoff_mask.sum() > 0:
        end_idx = np.where(time_cutoff_mask)[0][-1] + 1
    else:
        end_idx = len(time_iii)

    # Extract the region to plot
    time_iii_plot = time_iii[start_idx:end_idx] / 1000 + 150
    melt_num_plot = melt_numerical[start_idx:end_idx]
    melt_ana_plot = melt_analytical[start_idx:end_idx]

    # Melt rate comparison
    ax1.plot(time_iii_plot, melt_num_plot, "b-", linewidth=2, label="Numerical")
    ax1.plot(
        time_iii_plot, melt_ana_plot, "r--", linewidth=2, label="Analytical", alpha=0.7
    )
    ax1.set_xlabel("Time [ky]")
    ax1.set_ylabel("Basal Melt Rate [m/y]")
    ax1.set_title(
        f'Phase III: Transient Melt Rate\nMean error (validated region): {analytical["phase_iii"]["mean_error"]:.2%}'
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Absolute error over time (same time range)
    valid_mask = (melt_ana_plot > 1e-6) & (melt_num_plot > 1e-6)
    if valid_mask.sum() > 0:
        abs_error = np.abs(melt_num_plot - melt_ana_plot)
        ax2.plot(
            time_iii_plot[valid_mask], abs_error[valid_mask] * 1000, "k-", linewidth=2
        )
    ax2.set_xlabel("Time [ky]")
    ax2.set_ylabel("Absolute Error [mm/y]")
    ax2.set_title("Phase III: Absolute Error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = "exp_a_analytical_comparison.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✓ Analytical comparison saved to {output_file}")


if __name__ == "__main__":
    test_exp_a_transient_thermal_evolution()
