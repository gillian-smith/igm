#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import pytest
import sys
from igm.igm_run import main
from typing import Optional


def run_igm_unified(
    monkeypatch: pytest.MonkeyPatch,
    mapping: str = "identity",
    optimizer: str = "adam",
    length: Optional[int] = None,
) -> None:

    argv = [
        "igm_run.py",
        f"+experiment=params_{optimizer}",
        "processes.iceflow.method=unified",
        f"processes.iceflow.unified.mapping={mapping}",
        f"processes.iceflow.unified.optimizer={optimizer}",
    ]

    if length is not None:
        argv.append(f"inputs.init_state.L={length * 1e3}")
        path_run_dir = os.path.join(
            "outputs", f"{length}km", "unified", mapping, optimizer
        )
    else:
        path_run_dir = os.path.join("outputs", "unified", mapping, optimizer)

    argv.append(f"hydra.run.dir={path_run_dir}")

    if length is not None:
        argv.append(f"inputs.init_state.L={length * 1e3}")

    if mapping == "identity":
        argv.append("processes.iceflow.unified.adam.lr_init=0.9")

    monkeypatch.setattr(sys, "argv", argv)
    main()
