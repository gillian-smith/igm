#!/usr/bin/env python3
"""
Analytical solutions for Experiment A: Parallel-sided slab (transient)

Based on Kleiner et al. (2015) "Enthalpy benchmark experiments for numerical ice sheet models"
The Cryosphere, 9, 217–228, doi:10.5194/tc-9-217-2015

Experiment A tests the implementation of basal boundary conditions and melting/refreezing
during transient simulations.
"""

import numpy as np


def basal_melt_rate_steady_state(Ts, Tpmp, H, ki, qgeo, rho_w, L):
    """
    Compute steady-state basal melt rate for Experiment A.

    This applies when the base is at pressure melting point due to a basal water layer.

    Parameters
    ----------
    Ts : float
        Surface temperature [K]
    Tpmp : float
        Pressure melting point temperature at base [K]
    H : float
        Ice thickness [m]
    ki : float
        Thermal conductivity [W m^-1 K^-1]
    qgeo : float
        Geothermal heat flux [W m^-2]
    rho_w : float
        Water density [kg m^-3]
    L : float
        Latent heat of fusion [J kg^-1]

    Returns
    -------
    ab : float
        Basal melt rate [m a^-1 water equivalent]
    """
    # Equation (18) from Kleiner et al. (2015)
    ab = (1.0 / (rho_w * L)) * (qgeo + ki * (Ts - Tpmp) / H)

    # Convert from m/s to m/a
    spy = 31556926.0  # seconds per year
    ab_annual = ab * spy

    return ab_annual


def basal_temperature_steady_state(Ts, H, qgeo, ki):
    """
    Compute steady-state basal temperature for cold ice conditions.

    Parameters
    ----------
    Ts : float
        Surface temperature [K]
    H : float
        Ice thickness [m]
    qgeo : float
        Geothermal heat flux [W m^-2]
    ki : float
        Thermal conductivity [W m^-1 K^-1]

    Returns
    -------
    Tb : float
        Basal temperature [K]
    """
    # From linear temperature profile with geothermal flux at base
    Tb = Ts + H * qgeo / ki

    return Tb


def transient_basal_melt_rate(t, Ts_cold, Ts_warm, H, ki, qgeo, rho_w, L, Tpmp, kappa, n_terms=25):
    """
    Compute transient basal melt rate during cooling phase (Phase IIIa).

    This analytical solution applies when a basal water layer exists and the base
    is at pressure melting point, while the surface temperature has been changed.

    Parameters
    ----------
    t : float or array
        Time since start of cooling phase [years]
    Ts_cold : float
        Cold surface temperature [K]
    Ts_warm : float
        Warm surface temperature [K]
    H : float
        Ice thickness [m]
    ki : float
        Thermal conductivity [W m^-1 K^-1]
    qgeo : float
        Geothermal heat flux [W m^-2]
    rho_w : float
        Water density [kg m^-3]
    L : float
        Latent heat of fusion [J kg^-1]
    Tpmp : float
        Pressure melting point at base [K]
    kappa : float
        Thermal diffusivity [m^2 s^-1]
    n_terms : int, optional
        Number of Fourier series terms (default: 25)

    Returns
    -------
    ab : float or array
        Basal melt rate [m a^-1 water equivalent]
    """
    spy = 31556926.0  # seconds per year
    t_s = t * spy  # Convert years to seconds

    # Temperature gradient from steady-state equilibrium profile
    dT_eq_dz = (Ts_cold - Tpmp) / H

    # Fourier series for temperature deviation
    dT_dz_deviation = 0.0
    for n in range(1, n_terms + 1):
        lambda_n = -kappa * (n * np.pi / H) ** 2
        An = (-1) ** (n + 1) * 2 * (Ts_warm - Ts_cold) / (n * np.pi)
        dT_dz_deviation += (n * np.pi / H) * An * np.exp(lambda_n * t_s)

    # Total temperature gradient at base
    dT_dz = dT_eq_dz + dT_dz_deviation

    # Basal melt rate (Equation A14-A15 from Kleiner et al. 2015)
    qi = -ki * dT_dz  # Heat flux in ice at base
    ab = (1.0 / (rho_w * L)) * (qgeo - qi)

    # Convert from m/s to m/a
    ab_annual = ab * spy

    return ab_annual


def melting_to_freezing_transition_time(Ts_cold, Ts_warm, H, ki, qgeo, Tpmp, kappa, n_terms=25):
    """
    Estimate the time when melting transitions to freezing during cooling phase.

    This is when the basal melt rate crosses zero.

    Parameters
    ----------
    Ts_cold : float
        Cold surface temperature [K]
    Ts_warm : float
        Warm surface temperature [K]
    H : float
        Ice thickness [m]
    ki : float
        Thermal conductivity [W m^-1 K^-1]
    qgeo : float
        Geothermal heat flux [W m^-2]
    Tpmp : float
        Pressure melting point at base [K]
    kappa : float
        Thermal diffusivity [m^2 s^-1]
    n_terms : int, optional
        Number of Fourier series terms (default: 25)

    Returns
    -------
    t_transition : float
        Transition time [years]
    """
    # Search for zero crossing
    t_test = np.linspace(0, 10000, 10000)  # Test up to 10,000 years

    # Dummy values for rho_w and L (they cancel out at the zero crossing)
    rho_w = 1000.0
    L = 3.34e5

    melt_rates = transient_basal_melt_rate(
        t_test, Ts_cold, Ts_warm, H, ki, qgeo, rho_w, L, Tpmp, kappa, n_terms
    )

    # Find first zero crossing
    sign_changes = np.where(np.diff(np.sign(melt_rates)))[0]

    if len(sign_changes) > 0:
        t_transition = t_test[sign_changes[0]]
    else:
        t_transition = None

    return t_transition
