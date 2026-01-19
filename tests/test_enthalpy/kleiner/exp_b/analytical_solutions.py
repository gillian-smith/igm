#!/usr/bin/env python3
"""
Analytical solutions for Experiment B from Kleiner et al. (2015):
"Enthalpy benchmark experiments for numerical ice sheet models"
The Cryosphere, 9, 217–228, doi:10.5194/tc-9-217-2015

Implements the analytical solution from Appendix A2 for a polythermal
parallel-sided slab with strain heating.
"""

import numpy as np
from scipy.optimize import fsolve


class ExperimentBAnalytical:
    """Analytical solution for polythermal parallel-sided slab."""

    def __init__(
        self,
        H=200.0,
        gamma=4.0 * np.pi / 180,
        k_ice=2.1,
        c_ice=2009.0,
        rho_ice=910.0,
        T_surface=-3.0,
        A=5.3e-24,
        a_perp=0.2,  # m/a - CORRECTED from 0.3
        g=9.81,
        T_ref=223.15,
        T_0=273.15,
        L=3.35e5,
    ):
        """
        Initialize physical parameters for Experiment B.

        Parameters:
            H: Ice thickness [m]
            gamma: Surface slope [radians]
            k_ice: Thermal conductivity [W/(m·K)]
            c_ice: Specific heat capacity [J/(kg·K)]
            rho_ice: Ice density [kg/m³]
            T_surface: Surface temperature [°C]
            A: Flow rate factor [Pa⁻³ s⁻¹]
            a_perp: Accumulation rate [m/a]
            g: Gravitational acceleration [m/s²]
            T_ref: Reference temperature [K]
            T_0: Melting point at standard pressure [K]
            L: Latent heat of fusion [J/kg]
        """
        self.H = H
        self.gamma = gamma
        self.k_ice = k_ice
        self.c_ice = c_ice
        self.rho = rho_ice
        self.T_surface = T_surface
        self.A = A
        self.a_perp = a_perp / 31556926.0  # Convert m/a to m/s
        self.g = g
        self.T_ref = T_ref
        self.T_0 = T_0
        self.L = L

        # Derived quantities
        self.D = k_ice / (rho_ice * c_ice)  # Thermal diffusivity [m²/s]
        self.M = H * self.a_perp
        self.K = 2 * A * (rho_ice * g * np.sin(gamma)) ** 4 * H**6 / rho_ice

        # Boundary enthalpies
        self.E_surf = c_ice * (T_0 + T_surface - T_ref)
        self.E_pmp = c_ice * (T_0 - T_ref)

    def _compute_particular_coefficients(self):
        """Compute coefficients a_k for particular solution E_p(ζ)."""
        D, M, K = self.D, self.M, self.K

        # Solve from highest to lowest power
        a5 = -K / (5 * M)
        a4 = (4 * K - 20 * D * a5) / (4 * M)
        a3 = (-6 * K - 12 * D * a4) / (3 * M)
        a2 = (4 * K - 6 * D * a3) / (2 * M)
        a1 = (-K - 2 * D * a2) / M

        return np.array([a1, a2, a3, a4, a5])

    def _eval_particular(self, zeta, coeffs):
        """Evaluate particular solution at given zeta."""
        return sum(coeffs[i] * zeta ** (i + 1) for i in range(5))

    def _eval_particular_deriv(self, zeta, coeffs):
        """Evaluate derivative of particular solution."""
        return sum((i + 1) * coeffs[i] * zeta**i for i in range(5))

    def _solve_cts(self, coeffs):
        """Solve for CTS position and constants c1, c2."""
        D, M = self.D, self.M

        def system(x):
            zeta_m, c1, c2 = x

            # Surface BC: E(ζ=1) = E_surf
            eq1 = (
                c1 * np.exp(-M / D)
                + c2
                + self._eval_particular(1.0, coeffs)
                - self.E_surf
            )

            # CTS continuity: E(ζ=ζ_m) = E_pmp
            eq2 = (
                c1 * np.exp(-M * zeta_m / D)
                + c2
                + self._eval_particular(zeta_m, coeffs)
                - self.E_pmp
            )

            # CTS gradient: dE/dζ|_ζm⁺ = 0 (for K0=0)
            eq3 = -c1 * (M / D) * np.exp(-M * zeta_m / D) + self._eval_particular_deriv(
                zeta_m, coeffs
            )

            return [eq1, eq2, eq3]

        # Initial guess
        x0 = [0.1, 1000.0, self.E_pmp - 1000.0]
        solution = fsolve(system, x0)

        return solution[0], solution[1], solution[2]

    def solve_steady_state(self, n_points=501):
        """
        Solve for steady-state enthalpy profile.

        Returns:
            E: Enthalpy profile [J/kg]
            T: Temperature profile [°C]
            omega: Water content profile [%]
            z: Height coordinate [m]
            z_cts: CTS position [m]
        """
        # Compute particular solution coefficients
        coeffs = self._compute_particular_coefficients()

        # Solve for CTS position and constants
        zeta_m, c1, c2 = self._solve_cts(coeffs)

        # Create coordinate array
        zeta = np.linspace(0, 1, n_points)
        E = np.zeros(n_points)
        T = np.zeros(n_points)
        omega = np.zeros(n_points)

        for i, z in enumerate(zeta):
            if z < zeta_m:
                # Temperate ice
                E[i] = self.E_pmp + (self.K / (5 * self.M)) * (
                    (1 - z) ** 5 - (1 - zeta_m) ** 5
                )
                T[i] = self.T_0
                omega[i] = (E[i] - self.E_pmp) / self.L * 100
            else:
                # Cold ice
                E[i] = (
                    c1 * np.exp(-self.M * z / self.D)
                    + c2
                    + self._eval_particular(z, coeffs)
                )
                T[i] = (E[i] / self.c_ice) + self.T_ref
                omega[i] = 0.0

        return E, T - 273.15, omega, zeta * self.H, zeta_m * self.H

    def validate_cts_position(self, z_cts_numerical, tolerance=5.0):
        """
        Validate CTS position.

        Parameters:
            z_cts_numerical: Numerical CTS position [m]
            tolerance: Acceptable error [m] (default: 5.0 m)

        Returns:
            (is_valid, z_cts_analytical, error)
        """
        _, _, _, _, z_cts_analytical = self.solve_steady_state()
        error = abs(z_cts_numerical - z_cts_analytical)
        is_valid = error < tolerance

        return is_valid, z_cts_analytical, error

    def validate_temperature_profile(self, z_num, T_num, tolerance=0.5):
        """
        Validate temperature profile using RMSD.

        Parameters:
            z_num: Numerical height coordinates [m]
            T_num: Numerical temperature [°C]
            tolerance: RMSD tolerance [°C]

        Returns:
            (is_valid, T_analytical, rmsd)
        """
        E_ana, T_ana, _, z_ana, _ = self.solve_steady_state(n_points=len(z_num))

        # Interpolate analytical solution to numerical grid
        T_ana_interp = np.interp(z_num, z_ana, T_ana)

        # Compute RMSD
        rmsd = np.sqrt(np.mean((T_num - T_ana_interp) ** 2))
        is_valid = rmsd < tolerance

        return is_valid, T_ana_interp, rmsd

    def validate_enthalpy_profile(self, z_num, E_num, tolerance=300.0):
        """
        Validate enthalpy profile using RMSD.

        Parameters:
            z_num: Numerical height coordinates [m]
            E_num: Numerical enthalpy [J/kg]
            tolerance: RMSD tolerance [J/kg] (default: 300.0 J/kg)

        Returns:
            (is_valid, E_analytical, rmsd)
        """
        E_ana, _, _, z_ana, _ = self.solve_steady_state(n_points=len(z_num))

        # Interpolate analytical solution to numerical grid
        E_ana_interp = np.interp(z_num, z_ana, E_ana)

        # Compute RMSD
        rmsd = np.sqrt(np.mean((E_num - E_ana_interp) ** 2))
        is_valid = rmsd < tolerance

        return is_valid, E_ana_interp, rmsd

    def validate_water_content(self, z_num, omega_num, tolerance=0.5):
        """
        Validate water content profile.

        Parameters:
            z_num: Numerical height coordinates [m]
            omega_num: Numerical water content [%]
            tolerance: RMSD tolerance [%]

        Returns:
            (is_valid, omega_analytical, rmsd)
        """
        _, _, omega_ana, z_ana, _ = self.solve_steady_state(n_points=len(z_num))

        # Interpolate analytical solution to numerical grid
        omega_ana_interp = np.interp(z_num, z_ana, omega_ana)

        # Compute RMSD
        rmsd = np.sqrt(np.mean((omega_num - omega_ana_interp) ** 2))
        is_valid = rmsd < tolerance

        return is_valid, omega_ana_interp, rmsd


def validate_exp_b(results):
    """
    Validate Experiment B results against analytical solution.

    Parameters:
        results: Dictionary containing numerical results with keys:
            'z': Height coordinates [m]
            'E': Enthalpy [J/kg]
            'T': Temperature [°C]
            'omega': Water content [%]
            'cts_position': CTS position [m]

    Returns:
        Dictionary with validation results
    """
    analytical = ExperimentBAnalytical()

    # Extract numerical results
    z_num = results["z"]
    E_num = results["E"]
    T_num = results["T"]
    omega_num = results["omega"]
    cts_num = results["cts_position"]

    # Get full analytical solution
    E_ana, T_ana, omega_ana, z_ana, cts_ana = analytical.solve_steady_state()

    # Validate each component
    is_valid_cts, cts_ana_check, cts_error = analytical.validate_cts_position(cts_num)
    is_valid_T, T_ana_interp, T_rmsd = analytical.validate_temperature_profile(
        z_num, T_num
    )
    is_valid_E, E_ana_interp, E_rmsd = analytical.validate_enthalpy_profile(
        z_num, E_num
    )
    is_valid_omega, omega_ana_interp, omega_rmsd = analytical.validate_water_content(
        z_num, omega_num
    )

    # Overall validation
    overall_valid = is_valid_cts and is_valid_T and is_valid_E

    return {
        "overall_valid": overall_valid,
        "cts": {
            "valid": is_valid_cts,
            "numerical": cts_num,
            "analytical": cts_ana,
            "error": cts_error,
        },
        "temperature": {
            "valid": is_valid_T,
            "numerical": T_num,
            "analytical": T_ana_interp,
            "rmsd": T_rmsd,
        },
        "enthalpy": {
            "valid": is_valid_E,
            "numerical": E_num,
            "analytical": E_ana_interp,
            "rmsd": E_rmsd,
        },
        "water_content": {
            "valid": is_valid_omega,
            "numerical": omega_num,
            "analytical": omega_ana_interp,
            "rmsd": omega_rmsd,
        },
        "full_analytical": {
            "z": z_ana,
            "E": E_ana,
            "T": T_ana,
            "omega": omega_ana,
            "cts": cts_ana,
        },
    }


if __name__ == "__main__":
    # Test the analytical solution
    print("=" * 70)
    print("EXPERIMENT B ANALYTICAL SOLUTION")
    print("=" * 70)

    analytical = ExperimentBAnalytical()
    E, T, omega, z, z_cts = analytical.solve_steady_state()

    print(f"\nCTS Position:         {z_cts:.2f} m")
    print(f"Base Temperature:     {T[0]:.2f}°C")
    print(f"Surface Temperature:  {T[-1]:.2f}°C")
    print(f"Max Water Content:    {omega.max():.2f}%")
    print(f"Base Enthalpy:        {E[0]/1000:.2f} kJ/kg")
    print(f"Surface Enthalpy:     {E[-1]/1000:.2f} kJ/kg")

    print("\n" + "=" * 70)
    print("Validation against paper (Kleiner et al. 2015):")
    print("=" * 70)
    print(f"CTS Position:  {z_cts:.2f} m (expected ~19 m)")
    print(f"Error:         {abs(z_cts - 19.0):.2f} m")
    print(f"Status:        {'✓ PASS' if abs(z_cts - 19.0) < 1.0 else '⚠ CHECK'}")
    print("=" * 70)
