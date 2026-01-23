#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.emulate.emulator import (
    initialize_iceflow_emulator,
    update_iceflow_emulator,
)
from igm.processes.iceflow.solve.solve import (
    initialize_iceflow_solver,
    update_iceflow_solved,
)
from igm.processes.iceflow.utils.velocities import get_misfit


def initialize_iceflow_diagnostic(cfg: DictConfig, state: State) -> None:
    """Initialize the diagnostic mode for the iceflow module."""

    # Initialize emulator
    initialize_iceflow_emulator(cfg, state)

    # Initialize solver
    initialize_iceflow_solver(cfg, state)

    # Initialize metrics
    state.iceflow.diag_metrics = []


def update_iceflow_diagnostic(cfg: DictConfig, state: State) -> None:
    """Update the diagnostic mode for the iceflow module."""

    # Get emulator results
    update_iceflow_emulator(cfg, state)
    U_emulator, V_emulator, cost_emulator = state.U, state.V, state.cost_emulator

    # Get solver results
    update_iceflow_solved(cfg, state)
    U_solver, V_solver, cost_solver = state.U, state.V, state.cost_solver

    # Save diagnostics periodically
    cfg_diag = cfg.processes.iceflow.diagnostic
    if state.it % cfg_diag.save_freq != 0:
        return

    volume = tf.reduce_sum(state.thk) * tf.pow(state.dx, 2.0) / 1e9
    error_L1, error_L2 = get_misfit(
        U_solver,
        V_solver,
        U_emulator,
        V_emulator,
        state.iceflow.discr_v.V_bar,
        state.thk,
    )

    metrics = [
        state.it,
        state.t.numpy(),
        volume.numpy(),
        len(cost_emulator.numpy()),
        len(cost_solver.numpy()),
        cost_emulator.numpy()[-1],
        cost_solver.numpy()[-1],
        error_L1.numpy(),
        error_L2.numpy(),
    ]

    state.iceflow.diag_metrics.append(metrics)

    np.savetxt(
        cfg_diag.filename_metrics,
        state.iceflow.diag_metrics,
        delimiter=",",
        fmt="%10.3f",
        header="it,time,volume,it_emulator,it_solver,cost_emulator,cost_solver,error_L1,error_L2",
        comments="",
    )
