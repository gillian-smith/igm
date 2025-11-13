#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import pytest
import sys
from igm.igm_run import main


def run_igm_unified(
    monkeypatch: pytest.MonkeyPatch, length: int, mapping: str, optimizer: str
) -> None:
    argv = [
        "igm_run.py",
        f"+experiment=params_{optimizer}",
        f"inputs.init_state.L={length * 1e3}",
        "processes.iceflow.method=unified",
        f"processes.iceflow.unified.mapping={mapping}",
        f"processes.iceflow.unified.optimizer={optimizer}",
        f"hydra.run.dir=outputs/{length}km/unified/{mapping}/{optimizer}",
    ]
    if mapping == "identity":
        argv.append("processes.iceflow.unified.lr_init=0.9")

    monkeypatch.setattr(sys, "argv", argv)
    main()
