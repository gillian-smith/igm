#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import pytest
import numpy as np
import xarray as xr
from tests.test_iceflow.ismip_hom.utils import run_igm_unified, plot_comparison


@pytest.mark.parametrize(
    "mapping,optimizer",
    [
        (mapping, optimizer)
        for mapping in ["identity", "network"]
        for optimizer in ["adam", "lbfgs"]
    ],
)
def test_exp_e_2_unified(
    monkeypatch: pytest.MonkeyPatch, mapping: str, optimizer: str
) -> None:
    # Run simulation
    run_igm_unified(monkeypatch, mapping, optimizer)

    # Simulation results
    path_results_igm = os.path.join(
        "outputs", "unified", mapping, optimizer, "results.nc"
    )

    with xr.open_dataset(path_results_igm) as file:
        values = file["velsurf_mag"].values
        v_igm = values[0, 0]
        x_igm = np.linspace(0.0, 1.0, len(v_igm))

    # Reference results
    path_file_ref = os.path.join("..", "data", "oga", "oga1e001.txt")
    try:
        file_ref = np.loadtxt(path_file_ref)
    except FileNotFoundError:
        assert False, (
            f"❌ The file <{path_file_ref}> is not available. "
            + "Please run <igm/tests/get_data.sh> to download it."
        )
    x_ref = file_ref[:, 0]
    v_ref = file_ref[:, 1]

    # Plot
    title = f"ISMIP-HOM | EXP-E-2 | unified/{mapping}/{optimizer}"
    filename = f"exp_e_2_unified_{mapping}_{optimizer}.pdf"
    plot_comparison(x_ref, v_ref, x_igm, v_igm, title, filename)
