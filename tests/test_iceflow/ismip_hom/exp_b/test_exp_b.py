#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import pytest
import numpy as np
import xarray as xr
from tests.test_iceflow.ismip_hom.utils import run_igm_unified, plot_comparison


@pytest.mark.parametrize(
    "length,mapping,optimizer",
    [
        (length, mapping, optimizer)
        for length in [5, 10, 20, 40, 80, 160]
        for mapping in ["identity", "network"]
        for optimizer in ["adam", "lbfgs"]
    ],
)
def test_exp_b_unified(
    monkeypatch: pytest.MonkeyPatch, length: int, mapping: str, optimizer: str
) -> None:
    # Run simulation
    run_igm_unified(monkeypatch, length, mapping, optimizer)

    # Simulation results
    path_results_igm = f"outputs/{length}km/unified/{mapping}/{optimizer}/results.nc"

    with xr.open_dataset(path_results_igm) as file:
        values = file["velsurf_mag"].values
        v_igm = values[0, 0]
        x_igm = np.linspace(0.0, 1.0, len(v_igm))

    # Reference results
    file_ref = np.loadtxt(f"../data/oga/oga1b{length:03d}.txt")
    x_ref = file_ref[:, 0]
    v_ref = file_ref[:, 1]

    # Plot
    title = f"ISMIP-HOM | EXP-B | L={length}km | unified/{mapping}/{optimizer}"
    filename = f"exp_b_unified_{length}km_{mapping}_{optimizer}.pdf"
    plot_comparison(x_ref, v_ref, x_igm, v_igm, title, filename)
