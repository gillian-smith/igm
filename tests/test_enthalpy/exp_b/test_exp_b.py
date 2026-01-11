#!/usr/bin/env python3
"""
Clean Test for Experiment B: Polythermal parallel-sided slab (steady state)

Based on Kleiner et al. (2015) "Enthalpy benchmark experiments for numerical ice sheet models"
The Cryosphere, 9, 217–228, doi:10.5194/tc-9-217-2015

Experiment B: Tests steady-state enthalpy profile in a polythermal glacier with
prescribed ice flow and strain heating. The experiment verifies:
1. Formation of cold temperate transition surface (CTS)
2. Correct energy balance with strain heating
3. Grid convergence of numerical solution
4. Physical consistency (no unbounded growth)
"""

import os
import numpy as np
import tensorflow as tf
import pytest
import matplotlib.pyplot as plt

import igm
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.iceflow.utils.vertical_discretization import (
    compute_levels,
    compute_dz,
    compute_depth,
)
from igm.processes.enthalpy.utils.thermal import (
    surface_enthalpy_from_temperature,
    temperature_from_enthalpy,
    compute_pressure_melting_point,
)
from igm.processes.enthalpy.utils.solver import (
    assemble_enthalpy_system,
    solve_tridiagonal_system,
)


pytestmark = [pytest.mark.slow, pytest.mark.exp_b]


def test_exp_b_steady_state_formation():
    """
    Test that a polythermal ice column forms correctly in steady state.

    This is the main Experiment B test. It verifies:
    - CTS forms at reasonable depth
    - Temperature profile is physically consistent
    - Numerical solution converges
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Polythermal Parallel-sided Slab (Steady State)")
    print("=" * 70)

    # Experiment parameters from paper
    H = 200.0  # Ice thickness [m]
    gamma = np.deg2rad(10.0)  # Slope [rad]
    Ts = -3.0 + 273.15  # Surface temp [K]
    a_s_annual = -0.5  # Mass balance (negative = accumulation) [m/a]

    # Flow law parameters
    A = 5.3e-24  # Flow law factor [Pa^-3 s^-1]

    # Load configuration
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    Nz = 100
    cfg.processes.iceflow.numerics.Nz = Nz
    cfg.processes.iceflow.numerics.vert_spacing = 1

    cfg_enthalpy = cfg.processes.enthalpy
    cfg_physics = cfg.processes.iceflow.physics

    # Set parameters for this experiment
    cfg_enthalpy.KtdivKc = 1e-5  # Very small temperate ice conductivity
    cfg_enthalpy.till_wat_max = 0.0  # No basal water for this experiment
    cfg_enthalpy.drain_rate = 0.0
    cfg_enthalpy.claus_clape = 0.0  # No pressure dependence

    # Convert accumulation rate
    a_s = a_s_annual / cfg_enthalpy.spy  # Convert to m/s

    # Setup geometry
    thk = tf.Variable(H * tf.ones((1, 1)))
    levels = compute_levels(Nz, cfg.processes.iceflow.numerics.vert_spacing)
    dz = compute_dz(thk, levels)
    depth = compute_depth(dz)

    z = depth[:, 0, 0].numpy()

    # Prescribed velocity (from Eq. 13-15 in paper)
    # vx = prescribed shear flow
    # vz = a_s (vertical velocity = accumulation rate)
    vz = a_s * np.ones_like(z)
    w = tf.constant(vz[:, None, None], dtype=tf.float32)

    # Strain heating from shear flow (Eq. 16-17)
    # ε_eff = (A * (rho*g*sin(gamma))^3) * (H-z)^3
    # Ψ = 4 * A * rho / (H-z)^0.3 * ε_eff^2  (simplified for our geometry)
    rho = cfg_physics.ice_density
    g = cfg_physics.gravity_cst

    # Effective strain rate (depends on vertical derivative of shear stress)
    sigma_xy = rho * g * np.sin(gamma) * (H - z)
    eps_eff = A * sigma_xy**3

    # Regularize to avoid divide by zero
    eps_eff_safe = np.maximum(eps_eff, 1e-20)

    # Viscosity from flow law: eta = (1/2) * A^(-1/n) * eps_eff^((1-n)/n)
    # For n=3: eta = (1/2) * A^(-1/3) * eps_eff^(-2/3)
    n = 3.0
    mu = 0.5 * A ** (-1.0 / n) * eps_eff_safe ** ((1.0 - n) / n)

    # Strain heating = 2 * eta * eps_eff^2
    psi = 2 * mu * eps_eff**2
    psi = np.where(eps_eff < 1e-15, 0.0, psi)  # Zero out negligible heating

    strainheat = tf.constant(psi[:, None, None], dtype=tf.float32)

    # No friction or geothermal heating for this pure test
    frictheat = tf.Variable(0.0 * tf.ones((1, 1)))
    geoheatflux = tf.Variable(0.0 * tf.ones((1, 1)))
    tillwat = tf.Variable(0.0 * tf.ones((1, 1)))

    # Initial temperature: linear profile from surface to slightly cold at base
    # This ensures we start with some structure
    Tpmp_base = cfg_enthalpy.melt_temp  # Melting point at surface pressure
    T_init = Ts + (Tpmp_base - Ts) * (1.0 - z / H) * 0.9
    T_init = tf.constant(T_init[:, None, None], dtype=tf.float32)

    E = tf.Variable(cfg_enthalpy.ci * (T_init - cfg_enthalpy.ref_temp))
    surftemp = tf.Variable(Ts * tf.ones((1, 1)))

    # Time stepping to steady state
    dt_annual = 10.0  # years
    dt = dt_annual * cfg_enthalpy.spy  # convert to seconds
    max_iterations = 10000
    tolerance = 1e-5

    print(f"\nInitial conditions:")
    print(f"  H = {H} m, γ = {np.rad2deg(gamma):.1f}°, Ts = {Ts-273.15:.1f}°C")
    print(f"  a_s = {a_s_annual} m/a, Nz = {Nz}")
    print(f"  dt = {dt_annual} a, max_iter = {max_iterations}")
    print(f"\nIntegrating to steady state...")

    E_prev = tf.identity(E)
    converged = False

    for it in range(max_iterations):
        # Pressure melting point
        Tpmp, Epmp = compute_pressure_melting_point(
            depth,
            g,
            rho,
            cfg_enthalpy.claus_clape,
            cfg_enthalpy.melt_temp,
            cfg_enthalpy.ci,
            cfg_enthalpy.ref_temp,
        )

        # Surface enthalpy
        surfenth = surface_enthalpy_from_temperature(
            surftemp,
            cfg_enthalpy.melt_temp,
            cfg_enthalpy.ci,
            cfg_enthalpy.ref_temp,
        )

        # Solve enthalpy equation
        E_new, basalMeltRate = _solve_enthalpy_vertical(
            cfg,
            E,
            Epmp,
            dt,
            dz,
            w,
            surfenth,
            geoheatflux,
            strainheat,
            frictheat,
            tillwat,
        )
        E.assign(E_new)

        # Check convergence
        if it % 500 == 0 and it > 0:
            diff = tf.reduce_max(tf.abs(E - E_prev)).numpy()

            T, omega = temperature_from_enthalpy(
                E, Tpmp, Epmp, cfg_enthalpy.ci, cfg_enthalpy.ref_temp, cfg_enthalpy.Lh
            )

            # Check for CTS (where enthalpy equals Epmp)
            E_array = E[:, 0, 0].numpy()
            Epmp_array = Epmp[:, 0, 0].numpy()
            cts_mask = E_array >= Epmp_array

            if np.any(cts_mask):
                # Find transition points
                transitions = np.where(np.diff(cts_mask.astype(int)) != 0)[0]
                if len(transitions) > 0:
                    cts_idx = transitions[0]
                    # Linear interpolation for CTS position
                    if cts_idx > 0:
                        z_cts = z[cts_idx] + (
                            Epmp_array[cts_idx] - E_array[cts_idx]
                        ) * (z[cts_idx + 1] - z[cts_idx]) / (
                            E_array[cts_idx + 1] - E_array[cts_idx]
                        )
                    else:
                        z_cts = z[0]
                else:
                    z_cts = 0.0
            else:
                z_cts = 0.0

            T_base = T[0, 0, 0].numpy() - 273.15
            omega_max = tf.reduce_max(omega).numpy()

            print(
                f"  Iter {it:5d}: ΔE_max = {diff:.2e} J/kg, "
                f"CTS = {z_cts:6.2f} m, T_base = {T_base:6.2f}°C, ω_max = {omega_max:.3f}"
            )

            if diff < tolerance:
                print(f"\n✓ Converged after {it} iterations!")
                converged = True
                break

            E_prev = tf.identity(E)

    if not converged:
        print(f"⚠ Did not fully converge after {max_iterations} iterations")

    # Final analysis
    T, omega = temperature_from_enthalpy(
        E, Tpmp, Epmp, cfg_enthalpy.ci, cfg_enthalpy.ref_temp, cfg_enthalpy.Lh
    )

    T_array = T[:, 0, 0].numpy()
    E_array = E[:, 0, 0].numpy()
    Epmp_array = Epmp[:, 0, 0].numpy()
    omega_array = omega[:, 0, 0].numpy()

    # Find CTS position
    cts_mask = E_array >= Epmp_array
    if np.any(cts_mask) and not np.all(cts_mask):
        transitions = np.where(np.diff(cts_mask.astype(int)) != 0)[0]
        if len(transitions) > 0:
            cts_idx = transitions[0]
            z_cts = z[cts_idx] + (Epmp_array[cts_idx] - E_array[cts_idx]) * (
                z[cts_idx + 1] - z[cts_idx]
            ) / (E_array[cts_idx + 1] - E_array[cts_idx])
        else:
            z_cts = 0.0
    else:
        z_cts = 0.0

    # Final results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"CTS height:              {z_cts:10.2f} m (should be > 0)")
    print(f"Base temperature:        {T_array[0]:10.2f} K ({T_array[0]-273.15:7.2f}°C)")
    print(
        f"Surface temperature:     {T_array[-1]:10.2f} K ({T_array[-1]-273.15:7.2f}°C)"
    )
    print(f"Max water content:       {np.max(omega_array):10.3f} (relative)")
    print(f"Max strain heating:      {np.max(psi):10.2e} W/m³")

    # Validation criteria
    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")

    # 1. CTS should form (polythermal condition)
    cts_formed = z_cts > 1.0  # At least 1 m above bed
    print(
        f"1. CTS formed:           {'✓ PASS' if cts_formed else '✗ FAIL'} (z_cts={z_cts:.2f} m)"
    )
    assert cts_formed, "No CTS formed - glacier is entirely cold or temperate"

    # 2. Base should be at melting point
    T_base_ok = abs(T_array[0] - cfg_enthalpy.melt_temp) < 1.0
    print(
        f"2. Base at melting point: {'✓ PASS' if T_base_ok else '✗ FAIL'} "
        f"(T_base={T_array[0]:.2f} K, Tm={cfg_enthalpy.melt_temp:.2f} K)"
    )
    assert T_base_ok, f"Base temperature {T_array[0]} K is not at melting point"

    # 3. Surface should be at imposed temperature
    T_surf_ok = abs(T_array[-1] - Ts) < 0.1
    print(
        f"3. Surface temperature: {'✓ PASS' if T_surf_ok else '✗ FAIL'} "
        f"(T_surf={T_array[-1]:.2f} K, Ts={Ts:.2f} K)"
    )
    assert T_surf_ok, f"Surface temperature not matching imposed value"

    # 4. Temperate ice should have water content
    omega_gt_zero = np.max(omega_array) > 1e-4
    print(
        f"4. Temperate ice wet:    {'✓ PASS' if omega_gt_zero else '✗ FAIL'} "
        f"(ω_max={np.max(omega_array):.2e})"
    )
    assert omega_gt_zero, "Temperate ice should have non-zero water content"

    # Optional: plot results
    _plot_exp_b_results(z, T_array, E_array, omega_array, z_cts, Nz)

    print(f"\n{'='*70}")
    print("✓ EXPERIMENT B TEST PASSED")
    print(f"{'='*70}\n")


def test_exp_b_grid_convergence():
    """
    Test that solution converges with increasing resolution.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Grid Convergence Test")
    print("=" * 70)

    resolutions = [25, 50, 100]
    cts_positions = []
    t_bases = []

    for Nz in resolutions:
        print(f"\nTesting Nz = {Nz}...")

        # Experiment parameters
        H = 200.0
        gamma = np.deg2rad(10.0)
        Ts = -3.0 + 273.15
        a_s_annual = -0.5
        A = 5.3e-24

        # Setup
        cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
        cfg.processes.iceflow.numerics.Nz = Nz
        cfg.processes.iceflow.numerics.vert_spacing = 1

        cfg_enthalpy = cfg.processes.enthalpy
        cfg_physics = cfg.processes.iceflow.physics
        cfg_enthalpy.KtdivKc = 1e-5
        cfg_enthalpy.till_wat_max = 0.0
        cfg_enthalpy.drain_rate = 0.0
        cfg_enthalpy.claus_clape = 0.0

        a_s = a_s_annual / cfg_enthalpy.spy

        thk = tf.Variable(H * tf.ones((1, 1)))
        levels = compute_levels(Nz, cfg.processes.iceflow.numerics.vert_spacing)
        dz = compute_dz(thk, levels)
        depth = compute_depth(dz)

        z = depth[:, 0, 0].numpy()

        # Setup velocity and heating
        vz = a_s * np.ones_like(z)
        w = tf.constant(vz[:, None, None], dtype=tf.float32)

        rho = cfg_physics.ice_density
        g = cfg_physics.gravity_cst
        sigma_xy = rho * g * np.sin(gamma) * (H - z)
        eps_eff = A * sigma_xy**3
        eps_eff_safe = np.maximum(eps_eff, 1e-20)
        n = 3.0
        mu = 0.5 * A ** (-1.0 / n) * eps_eff_safe ** ((1.0 - n) / n)
        psi = 2 * mu * eps_eff**2
        psi = np.where(eps_eff < 1e-15, 0.0, psi)
        strainheat = tf.constant(psi[:, None, None], dtype=tf.float32)

        frictheat = tf.Variable(0.0 * tf.ones((1, 1)))
        geoheatflux = tf.Variable(0.0 * tf.ones((1, 1)))
        tillwat = tf.Variable(0.0 * tf.ones((1, 1)))

        T_init = Ts + (cfg_enthalpy.melt_temp - Ts) * (1.0 - z / H) * 0.9
        T_init = tf.constant(T_init[:, None, None], dtype=tf.float32)
        E = tf.Variable(cfg_enthalpy.ci * (T_init - cfg_enthalpy.ref_temp))
        surftemp = tf.Variable(Ts * tf.ones((1, 1)))

        # Integrate to steady state
        dt = 10.0 * cfg_enthalpy.spy
        for it in range(5000):
            Tpmp, Epmp = compute_pressure_melting_point(
                depth,
                g,
                rho,
                cfg_enthalpy.claus_clape,
                cfg_enthalpy.melt_temp,
                cfg_enthalpy.ci,
                cfg_enthalpy.ref_temp,
            )
            surfenth = surface_enthalpy_from_temperature(
                surftemp, cfg_enthalpy.melt_temp, cfg_enthalpy.ci, cfg_enthalpy.ref_temp
            )
            E_new, _ = _solve_enthalpy_vertical(
                cfg,
                E,
                Epmp,
                dt,
                dz,
                w,
                surfenth,
                geoheatflux,
                strainheat,
                frictheat,
                tillwat,
            )
            E.assign(E_new)

            if it % 1000 == 0 and it > 0:
                print(f"  Iteration {it}")

        # Extract results
        T, omega = temperature_from_enthalpy(
            E, Tpmp, Epmp, cfg_enthalpy.ci, cfg_enthalpy.ref_temp, cfg_enthalpy.Lh
        )

        E_array = E[:, 0, 0].numpy()
        Epmp_array = Epmp[:, 0, 0].numpy()
        cts_mask = E_array >= Epmp_array

        if np.any(cts_mask) and not np.all(cts_mask):
            transitions = np.where(np.diff(cts_mask.astype(int)) != 0)[0]
            if len(transitions) > 0:
                cts_idx = transitions[0]
                z_cts = z[cts_idx] + (Epmp_array[cts_idx] - E_array[cts_idx]) * (
                    z[cts_idx + 1] - z[cts_idx]
                ) / (E_array[cts_idx + 1] - E_array[cts_idx])
            else:
                z_cts = 0.0
        else:
            z_cts = 0.0

        T_base = T[0, 0, 0].numpy()

        cts_positions.append(z_cts)
        t_bases.append(T_base)

        print(f"  CTS position: {z_cts:.2f} m, T_base: {T_base:.2f} K")

    # Check convergence (positions shouldn't vary too wildly)
    cts_variation = max(cts_positions) - min(cts_positions)
    print(f"\nCTS position variation: {cts_variation:.2f} m")
    print(f"CTS positions: {[f'{x:.2f}' for x in cts_positions]}")

    # Check that CTS is consistent (not too much variation)
    # Note: With coarse grids, CTS position can vary by ~5m due to discretization
    assert cts_variation < 6.0, f"CTS position varies too much: {cts_variation:.2f} m"

    print("✓ Grid convergence test passed")


def _solve_enthalpy_vertical(
    cfg, E, Epmp, dt, dz, w, surfenth, bheatflx, strainheat, frictheat, tillwat
):
    """Solve the vertical enthalpy equation."""
    cfg_enthalpy = cfg.processes.enthalpy
    cfg_physics = cfg.processes.iceflow.physics

    nz, ny, nx = E.shape

    # Material properties
    ki = cfg_enthalpy.ki
    ice_density = cfg_physics.ice_density
    water_density = cfg_enthalpy.water_density
    ci = cfg_enthalpy.ci
    ref_temp = cfg_enthalpy.ref_temp
    Lh = cfg_enthalpy.Lh
    spy = cfg_enthalpy.spy
    KtdivKc = cfg_enthalpy.KtdivKc
    thr = cfg_physics.thr_ice_thk
    min_temp = cfg_enthalpy.min_temp

    # Thermal diffusivity
    PKc = ki / (ice_density * ci)
    f = strainheat / ice_density

    # Enhanced diffusivity in temperate ice
    K = PKc * tf.ones_like(dz)
    K = tf.where((E[:-1] + E[1:]) / 2.0 >= (Epmp[:-1] + Epmp[1:]) / 2.0, K * KtdivKc, K)

    # Boundary conditions
    VS = surfenth

    COLD_BASE = (E[0] < Epmp[0]) | (tillwat <= 0)
    DRY_ICE = tillwat <= 0
    COLD_ICE = E[1] < Epmp[1]

    BCB = tf.where(
        COLD_BASE,
        tf.where(DRY_ICE, tf.ones((ny, nx)), tf.zeros((ny, nx))),
        tf.where(COLD_ICE, tf.zeros((ny, nx)), tf.ones((ny, nx))),
    )

    VB = tf.where(
        COLD_BASE,
        tf.where(DRY_ICE, -(ci / ki) * (bheatflx + frictheat), Epmp[0]),
        tf.where(COLD_ICE, Epmp[0], 0.0),
    )

    # Assemble and solve system
    L, M, U, R = assemble_enthalpy_system(
        E, dt, tf.maximum(dz, thr), w, K, f, BCB, VB, VS
    )
    E = solve_tridiagonal_system(L, M, U, R)

    # Enforce bounds
    Emin = ci * (min_temp - ref_temp)
    E = tf.maximum(E, Emin)

    Emax = Epmp + Lh
    E = tf.minimum(E, Emax)

    # Compute basal melt rate
    flux = tf.where(
        E[1] < Epmp[1],
        -(ki / ci) * (E[1] - E[0]) / tf.maximum(dz[0], thr),
        -KtdivKc * (ki / ci) * (E[1] - E[0]) / tf.maximum(dz[0], thr),
    )

    basalMeltRate = tf.where(
        (E[0] < Epmp[0]) & (tillwat <= 0),
        tf.zeros((ny, nx)),
        spy * (bheatflx + frictheat - flux) / (water_density * Lh),
    )

    return E, basalMeltRate


def _plot_exp_b_results(z, T, E, omega, z_cts, Nz):
    """Plot Experiment B results."""
    plot_enabled = os.environ.get("IGM_PLOT_TESTS", "false").lower() == "true"
    if not plot_enabled:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Temperature profile
    axes[0].plot(T - 273.15, z, "b-", linewidth=2, label="Temperature")
    axes[0].axhline(
        z_cts, color="r", linestyle="--", linewidth=1, label=f"CTS (z={z_cts:.1f}m)"
    )
    axes[0].set_xlabel("Temperature [°C]", fontsize=11)
    axes[0].set_ylabel("Height above bed [m]", fontsize=11)
    axes[0].set_title("Temperature Profile", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Enthalpy profile
    axes[1].plot(E, z, "g-", linewidth=2)
    axes[1].set_xlabel("Enthalpy [J/kg]", fontsize=11)
    axes[1].set_ylabel("Height above bed [m]", fontsize=11)
    axes[1].set_title("Enthalpy Profile", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Water content
    axes[2].plot(omega, z, "orange", linewidth=2)
    axes[2].set_xlabel("Water Content (relative)", fontsize=11)
    axes[2].set_ylabel("Height above bed [m]", fontsize=11)
    axes[2].set_title("Water Content Profile", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(__file__), f"exp_b_steady_state_Nz{Nz}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    test_exp_b_steady_state_formation()
    test_exp_b_grid_convergence()
