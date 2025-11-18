#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import pytest
import numpy as np
import xarray as xr
from tests.test_iceflow.ismip_hom.utils import run_igm_unified, plot_comparison


@pytest.mark.parametrize(
    "length,mapping,optimizer",
    [
        (length, mapping, optimizer)
        for length in [160]
        for mapping in ["network"]
        for optimizer in ["lbfgs"]
    ],
)
def test_exp_c_unified(
    monkeypatch: pytest.MonkeyPatch, length: int, mapping: str, optimizer: str
) -> None:
    # Run simulation
    run_igm_unified(monkeypatch, mapping, optimizer, length)

    # Simulation results
    path_results_igm = os.path.join(
        "outputs", f"{length}km", "unified", mapping, optimizer, "results.nc"
    )

    with xr.open_dataset(path_results_igm) as file:
        values = file["velsurf_mag"].values
        ny = values.shape[1]
        v_igm = values[0, int(0.25 * ny)]
        x_igm = np.linspace(0.0, 1.0, len(v_igm))

    # Reference results
    path_file_ref = os.path.join("..", "data", "oga", f"oga1c{length:03d}.txt")
    try:
        file_ref = np.loadtxt(path_file_ref)
    except FileNotFoundError:
        assert False, (
            f"❌ The file <{path_file_ref}> is not available. "
            + "Please run <igm/tests/get_data.sh> to download it."
        )
    idx = file_ref[:, 1] == 0.25
    x_ref = file_ref[idx, 0]
    v_ref = np.hypot(file_ref[idx, 2], file_ref[idx, 3])

    # Plot
    title = f"ISMIP-HOM | EXP-C | L={length}km | unified/{mapping}/{optimizer}"
    filename = f"exp_c_unified_{length}km_{mapping}_{optimizer}.pdf"
    plot_comparison(x_ref, v_ref, x_igm, v_igm, title, filename)
