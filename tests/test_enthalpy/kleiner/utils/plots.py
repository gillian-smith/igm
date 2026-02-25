#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Plotting utilities for Kleiner et al. (2015) benchmarks."""

import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_exp_a(results: Dict[str, Any]) -> None:
    """Generate Experiment A diagnostic plots."""
    time_ky = results["time"] / 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.align_ylabels(axes)

    # Base temperature
    axes[0].plot(time_ky, results["T_base"], "-", linewidth=2.0)
    axes[0].set_ylabel("Base temperature (°C)", fontsize=15)
    axes[0].set_title(
        "Experiment A (Kleiner et al., 2015)", fontsize=16, fontweight="bold"
    )

    # Basal melt rate
    axes[1].plot(time_ky, results["melt_rate"], "-", linewidth=2.0)
    axes[1].set_ylabel("Basal melt rate (m/yr)", fontsize=15)

    # Till water
    axes[2].plot(time_ky, results["till_water"], "-", linewidth=2.0)
    axes[2].set_ylabel("Till water height (m)", fontsize=15)
    axes[2].set_xlabel("Time (ky)", fontsize=15)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=15)
        ax.set_xlim(min(time_ky), max(time_ky))

    plt.tight_layout()
    plt.savefig("exp_a.png", dpi=150)
    plt.close()


def plot_exp_b(results: Dict[str, Any]) -> None:
    """Generate Experiment B diagnostic plots."""
    z = results["z"]
    cts = results["cts_position"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # Enthalpy
    axes[0].plot(results["E"] / 1000, z, "-", lw=2)
    axes[0].set_xlabel("Enthalpy (kJ/kg)", fontsize=15)
    axes[0].set_ylabel("Height (m)", fontsize=15)
    if cts > 0.0:
        axes[0].axhline(cts, color="k", ls="--", label=f"CTS: {cts:.1f} m")
        axes[0].legend(fontsize=15)

    # Temperature
    axes[1].plot(results["T"], z, "-", lw=2)
    axes[1].set_xlabel("Temperature (°C)", fontsize=15)
    axes[1].set_title(
        "Experiment B (Kleiner et al., 2015)", fontsize=16, fontweight="bold"
    )
    if cts > 0.0:
        axes[1].axhline(cts, color="k", ls="--")

    # Water content
    axes[2].plot(results["omega"], z, "-", lw=2)
    axes[2].set_xlabel("Water content (%)", fontsize=15)
    if cts > 0.0:
        axes[2].axhline(cts, color="k", ls="--")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=15)
        ax.set_ylim(min(z), max(z))

    plt.tight_layout()
    plt.savefig("exp_b.png", dpi=150)
    plt.close()
