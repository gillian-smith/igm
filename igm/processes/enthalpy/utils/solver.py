#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Tuple
import tensorflow as tf
from omegaconf import DictConfig

from .advection import compute_upwind_advection
from igm.common.core import State


@tf.function
def drainageFunc(
    omega: tf.Tensor,
    threshold_1: float,
    threshold_2: float,
    threshold_3: float,
) -> tf.Tensor:
    """
    Compute drainage rate as function of water content.

    Following Greve (1997) and Aschwanden et al. (2012).

    Args:
        omega: Water content [dimensionless]
        threshold_1: First drainage threshold
        threshold_2: Second drainage threshold
        threshold_3: Third drainage threshold

    Returns:
        Drainage rate [y⁻¹]
    """
    return tf.where(
        omega <= threshold_1,
        0.0,
        tf.where(
            omega <= threshold_2,
            0.5 * omega - 0.005,
            tf.where(
                omega <= threshold_3,
                4.5 * omega - 0.085,
                0.05,
            ),
        ),
    )


@tf.function
def solve_tridiagonal_system(
    L: tf.Tensor,
    M: tf.Tensor,
    U: tf.Tensor,
    R: tf.Tensor,
) -> tf.Tensor:
    """
    Solve tridiagonal system using Thomas Algorithm (TDMA).

    Args:
        L: Lower diagonal [nz-1, ny, nx]
        M: Main diagonal [nz, ny, nx]
        U: Upper diagonal [nz-1, ny, nx]
        R: Right-hand side [nz, ny, nx]

    Returns:
        Solution [nz, ny, nx]
    """
    nz = tf.shape(M)[0]

    w = tf.TensorArray(dtype=tf.float32, size=nz - 1)
    g = tf.TensorArray(dtype=tf.float32, size=nz)
    p = tf.TensorArray(dtype=tf.float32, size=nz)

    # Forward sweep
    w = w.write(0, U[0] / M[0])
    g = g.write(0, R[0] / M[0])

    for i in tf.range(1, nz - 1):
        w = w.write(i, U[i] / (M[i] - L[i - 1] * w.read(i - 1)))

    for i in tf.range(1, nz):
        g = g.write(
            i, (R[i] - L[i - 1] * g.read(i - 1)) / (M[i] - L[i - 1] * w.read(i - 1))
        )

    # Backward substitution
    p = p.write(nz - 1, g.read(nz - 1))

    for i in tf.range(nz - 2, -1, -1):
        p = p.write(i, g.read(i) - w.read(i) * p.read(i + 1))

    return p.stack()


@tf.function
def assemble_enthalpy_system(
    E: tf.Tensor,
    dt: tf.Tensor,
    dz: tf.Tensor,
    w: tf.Tensor,
    K: tf.Tensor,
    f: tf.Tensor,
    BCB: tf.Tensor,
    VB: tf.Tensor,
    VS: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Assemble finite difference system for enthalpy equation.

    Solves: dE/dt + w·dE/dz = K·d²E/dz² + f

    Args:
        E: Current enthalpy [J/kg]
        dt: Time step [s]
        dz: Layer thickness [m]
        w: Vertical velocity [m/s]
        K: Thermal diffusivity [m²/s]
        f: Source term [W/kg]
        BCB: Bottom BC flag (1=Neumann, 0=Dirichlet)
        VB: Bottom BC value
        VS: Surface BC value

    Returns:
        Tuple of (L, M, U, R) - tridiagonal system components
    """
    nz, ny, nx = E.shape
    s = dt * K / (dz * dz)  # Diffusion coefficient

    # Initialize system
    L = tf.zeros((nz - 1, ny, nx))
    M = tf.ones((nz, ny, nx))
    U = tf.zeros((nz - 1, ny, nx))
    R = E + dt * f

    # Assembly: diffusion terms
    M = M + tf.concat([s, tf.zeros((1, ny, nx))], axis=0)
    M = M + tf.concat([tf.zeros((1, ny, nx)), s], axis=0)
    L = L - s
    U = U - s

    # Bottom boundary condition
    M = tf.concat(
        [tf.where(BCB == 1, -tf.ones_like(BCB), tf.ones_like(BCB))[None, :, :], M[1:]],
        axis=0,
    )
    U = tf.concat(
        [tf.where(BCB == 1, tf.ones_like(BCB), tf.zeros_like(BCB))[None, :, :], U[1:]],
        axis=0,
    )
    R = tf.concat([tf.where(BCB == 1, VB * dz[0], VB)[None, :, :], R[1:]], axis=0)

    # Surface boundary condition
    M = tf.concat([M[:-1], tf.ones_like(BCB)[None, :, :]], axis=0)
    L = tf.concat([L[:-1], tf.zeros_like(BCB)[None, :, :]], axis=0)
    R = tf.concat([R[:-1], VS[None, :, :]], axis=0)

    # Upwind advection (implicit)
    wdivdz = dt * (w[1:] + w[:-1]) / (2.0 * dz)
    L = tf.concat([L[:-1] + tf.where(w[1:-1] > 0, -wdivdz[:-1], 0), L[-1:]], axis=0)
    M = tf.concat(
        [M[:1], M[1:-1] + tf.where(w[1:-1] > 0, wdivdz[:-1], -wdivdz[1:]), M[-1:]],
        axis=0,
    )
    U = tf.concat([U[:1], U[1:] + tf.where(w[1:-1] <= 0, wdivdz[1:], 0)], axis=0)

    return L, M, U, R


def solve_enthalpy_equation(
    cfg: DictConfig,
    state: State,
    E: tf.Tensor,
    Epmp: tf.Tensor,
    dt: tf.Tensor,
    dz: tf.Tensor,
    Wc: tf.Tensor,
    surfenth: tf.Tensor,
    bheatflx: tf.Tensor,
    strainheat: tf.Tensor,
    frictheat: tf.Tensor,
    tillwat: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Solve enthalpy equation and compute basal melt rate.

    Args:
        cfg: Configuration object
        state: State object (for U, V, dx)
        E: Current enthalpy [J/kg]
        Epmp: Pressure melting point enthalpy [J/kg]
        dt: Time step [s]
        dz: Layer thickness [m]
        Wc: Corrected vertical velocity [m/s]
        surfenth: Surface enthalpy [J/kg]
        bheatflx: Basal heat flux [W/m²]
        strainheat: Strain heating rate [W/m³]
        frictheat: Friction heating rate [W/m²]
        tillwat: Till water content [m]

    Returns:
        Tuple of (E [J/kg], basalMeltRate [m/y])
    """
    cfg_enthalpy = cfg.processes.enthalpy

    # Horizontal advection (explicit)
    E = E - dt * compute_upwind_advection(
        state.U / cfg_enthalpy.spy,
        state.V / cfg_enthalpy.spy,
        E,
        state.dx,
    )

    # Solve vertical diffusion-advection (implicit)
    E, basalMeltRate = _solve_vertical_enthalpy(
        cfg, E, Epmp, dt, dz, Wc, surfenth, bheatflx, strainheat, frictheat, tillwat
    )

    return E, basalMeltRate


def _solve_vertical_enthalpy(
    cfg: DictConfig,
    E: tf.Tensor,
    Epmp: tf.Tensor,
    dt: tf.Tensor,
    dz: tf.Tensor,
    w: tf.Tensor,
    surfenth: tf.Tensor,
    bheatflx: tf.Tensor,
    strainheat: tf.Tensor,
    frictheat: tf.Tensor,
    tillwat: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Solve vertical enthalpy equation with appropriate boundary conditions."""
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
    PKc = ki / (ice_density * ci)  # [m²/s]
    f = strainheat / ice_density  # [W/kg]

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

    Emax = Epmp + Lh  # omega = 1
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

    # Drainage along ice column
    if (dt > 0) and cfg_enthalpy.drain_ice_column:
        E, basalMeltRate = _apply_drainage(
            E,
            Epmp,
            dt,
            dz,
            basalMeltRate,
            Lh,
            ice_density,
            water_density,
            spy,
            cfg_enthalpy.target_water_fraction,
            cfg_enthalpy.drainage_omega_threshold_1,
            cfg_enthalpy.drainage_omega_threshold_2,
            cfg_enthalpy.drainage_omega_threshold_3,
        )

    return E, basalMeltRate


@tf.function
def _apply_drainage(
    E: tf.Tensor,
    Epmp: tf.Tensor,
    dt: tf.Tensor,
    dz: tf.Tensor,
    basalMeltRate: tf.Tensor,
    Lh: float,
    ice_density: float,
    water_density: float,
    spy: float,
    target_water_fraction: float,
    threshold_1: float,
    threshold_2: float,
    threshold_3: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply water drainage through ice column."""
    # Cell-centered layer thickness
    DZ = tf.concat([dz[0:1], dz[:-1] + dz[1:], dz[-1:]], axis=0) / 2.0

    # Water content
    omega = tf.maximum((E - Epmp) / Lh, 0.0)

    # Identify cells to drain
    CD = omega > target_water_fraction

    # Drainage fraction
    fraction_drained = (
        drainageFunc(omega, threshold_1, threshold_2, threshold_3) * dt / spy
    )
    fraction_drained = tf.minimum(fraction_drained, omega - target_water_fraction)

    # Drained water thickness
    H_drained = tf.where(CD, fraction_drained * DZ, 0.0)  # [m]

    # Update enthalpy
    E = tf.where(CD, E - fraction_drained * Lh, E)

    # Total drained water
    H_total_drained = tf.reduce_sum(H_drained, axis=0)  # [m]

    # Add to basal melt rate
    basalMeltRate = (
        basalMeltRate + (spy / dt) * (ice_density / water_density) * H_total_drained
    )

    return E, basalMeltRate
