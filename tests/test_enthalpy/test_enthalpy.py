#!/usr/bin/env python3

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


pytestmark = pytest.mark.slow


def test_enthalpy():
    """
    Test enthalpy evolution for a simple vertical column.

    This test simulates thermal evolution with:
    - Initial cold ice (-30°C)
    - Surface warming period (100-150 ky)
    - Geothermal heat flux

    Expected: Base temperature should warm significantly.
    """
    # Time parameters
    ttf = 150000.0  # Total time [years]
    dt = 200.0  # Time step [years]
    tim = np.arange(0, ttf, dt) + dt

    # Load configuration
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    cfg.processes.iceflow.numerics.Nz = 50
    cfg.processes.iceflow.numerics.vert_spacing = 1

    # Set enthalpy parameters
    cfg_enthalpy = cfg.processes.enthalpy
    cfg_enthalpy.KtdivKc = 1e-5
    cfg_enthalpy.till_wat_max = 200.0
    cfg_enthalpy.drain_rate = 0.0

    # Initialize geometry
    thk = tf.Variable(1000.0 * tf.ones((1, 1)))

    # Vertical discretization
    levels = compute_levels(
        cfg.processes.iceflow.numerics.Nz,
        cfg.processes.iceflow.numerics.vert_spacing,
    )
    dz = compute_dz(thk, levels)
    depth = compute_depth(dz)

    # Heat sources
    strainheat = tf.Variable(tf.zeros((cfg.processes.iceflow.numerics.Nz, 1, 1)))
    frictheat = tf.Variable(0.0 * tf.ones((1, 1)))
    geoheatflux = tf.Variable(0.042 * tf.ones((1, 1)))  # [W/m²]
    tillwat = tf.Variable(0.0 * tf.ones((1, 1)))

    # Initial temperature field
    T_init = (-30.0 + 273.15) * tf.ones((cfg.processes.iceflow.numerics.Nz, 1, 1))
    E = tf.Variable(cfg_enthalpy.ci * (T_init - cfg_enthalpy.ref_temp))
    w = tf.Variable(tf.zeros_like(E))

    # Track evolution
    TB = []  # Base temperature
    HW = []  # Till water height
    MR = []  # Melt rate

    surftemp = tf.Variable((-30.0 + 273.15) * tf.ones((1, 1)))

    # Time integration
    for it, t in enumerate(tim):
        # Surface temperature forcing
        if 100000.0 <= t < 150000.0:
            surftemp.assign((-5.0 + 273.15) * tf.ones((1, 1)))
        else:
            surftemp.assign((-30.0 + 273.15) * tf.ones((1, 1)))

        # Compute pressure melting point
        Tpmp, Epmp = compute_pressure_melting_point(
            depth,
            cfg.processes.iceflow.physics.gravity_cst,
            cfg.processes.iceflow.physics.ice_density,
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
        E_new, basalMeltRate = _solve_vertical_enthalpy_test(
            cfg,
            E,
            Epmp,
            dt * cfg_enthalpy.spy,
            dz,
            w,
            surfenth,
            geoheatflux,
            strainheat,
            frictheat,
            tillwat,
        )
        E.assign(E_new)

        # Convert to temperature
        T, omega = temperature_from_enthalpy(
            E,
            Tpmp,
            Epmp,
            cfg_enthalpy.ci,
            cfg_enthalpy.ref_temp,
            cfg_enthalpy.Lh,
        )

        # Update till water
        tillwat = tillwat + dt * (basalMeltRate - cfg_enthalpy.drain_rate)
        tillwat = tf.clip_by_value(tillwat, 0.0, cfg_enthalpy.till_wat_max)

        # Record evolution
        TB.append(T[0] - 273.15)
        HW.append(tillwat)
        MR.append(basalMeltRate)

        if it % 100 == 0:
            print(
                f"time: {t:8.0f}, T_base: {T[0, 0, 0].numpy():.2f}, "
                f"tillwat: {tillwat[0, 0].numpy():.4f}, "
                f"meltrate: {basalMeltRate[0, 0].numpy():.6f}"
            )

    # Final temperature
    last_temp = np.stack(TB)[:, 0, 0][-1]
    print(f"\nFinal base temperature: {last_temp:.2f} °C")

    # Validation: base should warm significantly
    assert last_temp > -2.0, f"Base temperature {last_temp} is too cold"

    # Optional: plot results
    _plot_results_if_requested(tim, TB, MR, HW)


def _solve_vertical_enthalpy_test(
    cfg, E, Epmp, dt, dz, w, surfenth, bheatflx, strainheat, frictheat, tillwat
):
    """Simplified version of solve_vertical_enthalpy for testing."""
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

    # BC flags: 1=Neumann, 0=Dirichlet
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

    # Compute basal heat flux
    flux = tf.where(
        E[1] < Epmp[1],
        -(ki / ci) * (E[1] - E[0]) / tf.maximum(dz[0], thr),
        -KtdivKc * (ki / ci) * (E[1] - E[0]) / tf.maximum(dz[0], thr),
    )

    # Basal melt rate [m/y]
    basalMeltRate = tf.where(
        (E[0] < Epmp[0]) & (tillwat <= 0),
        tf.zeros((ny, nx)),
        spy * (bheatflx + frictheat - flux) / (water_density * Lh),
    )

    return E, basalMeltRate


def _plot_results_if_requested(tim, TB, MR, HW):
    """Plot results if enabled."""
    plot_enabled = False  # Set to True to generate plots

    if not plot_enabled:
        return

    fig = plt.figure(figsize=(8, 8))

    plt.subplot(311)
    plt.plot(tim, np.stack(TB)[:, 0, 0])
    plt.ylabel("Base Temperature [°C]")
    plt.xlabel("Time [years]")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(tim, np.stack(MR)[:, 0, 0])
    plt.ylabel("Basal Melt Rate [m/y]")
    plt.xlabel("Time [years]")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(tim, np.stack(HW)[:, 0, 0])
    plt.ylabel("Till Water Height [m]")
    plt.xlabel("Time [years]")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("enthalpy_test_results.png", dpi=150)
    plt.close()
    print("Results saved to enthalpy_test_results.png")


if __name__ == "__main__":
    test_enthalpy()
