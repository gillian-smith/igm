#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
 Quick notes about the code below:

 The goal of this module is to compute the ice flow velocity field
 using a deep-learning emulator of the Blatter-Pattyn model.

 The aim of this module is
   - to initialize the ice flow and its emulator in init_iceflow
   - to update the ice flow and its emulator in update_iceflow

In update_iceflow, we compute/update with function _update_iceflow_emulated,
and retraine the iceflow emaultor in function _update_iceflow_emulator

- in _update_iceflow_emulated, we baiscially gather together all input fields
of the emulator and stack all in a single tensor X, then we compute the output
with Y = iceflow_model(X), and finally we split Y into U and V

- in _update_iceflow_emulator, we retrain the emulator. For that purpose, we
iteratively (usually we do only one iteration) compute the output of the emulator,
compute the energy associated with the state of the emulator, and compute the
gradient of the energy with respect to the emulator parameters. Then we update
the emulator parameters with the gradient descent method (Adam optimizer).
Because this step may be memory consuming, we split the computation in several
patches of size cfg.processes.iceflow.emulator.framesizemax. This permits to
retrain the emulator on large size arrays.

Alternatively, one can solve the Blatter-Pattyn model using a solver using
function _update_iceflow_solved. Doing so is not very different to retrain the
emulator as we minmize the same energy, however, with different controls,
namely directly the velocity field U and V instead of the emulator parameters.
"""
from omegaconf import DictConfig

from igm.common.core import State
from igm.processes.iceflow.emulate.emulator import (
    update_iceflow_emulator,
    initialize_iceflow_emulator,
)
from igm.processes.iceflow.solve.solve import (
    initialize_iceflow_solver,
    update_iceflow_solved,
)
from igm.processes.iceflow.diagnostic.diagnostic import (
    initialize_iceflow_diagnostic,
    update_iceflow_diagnostic,
)
from igm.processes.iceflow.unified.unified import (
    initialize_iceflow_unified,
    update_iceflow_unified,
)
from igm.processes.iceflow.emulate.utils import save_iceflow_model
from igm.processes.iceflow.utils.init import initialize_iceflow_fields
from igm.processes.iceflow.utils.vertical_discretization import define_vertical_weight
from igm.processes.iceflow.vertical import VerticalDiscrs


class Iceflow:
    pass


def initialize(cfg: DictConfig, state: State) -> None:

    # Make sure this function is only called once
    if getattr(state, "iceflow_initialized", False):
        return
    state._iceflow_initialized = True

    # Create ice flow object
    state.iceflow = Iceflow()

    # Initialize ice-flow fields: U, V, slidingco, arrhenius
    initialize_iceflow_fields(cfg, state)

    # Initialize vertical discretization
    cfg_numerics = cfg.processes.iceflow.numerics

    vertical_basis = cfg_numerics.vert_basis.lower()
    vertical_discr = VerticalDiscrs[vertical_basis](cfg)
    state.iceflow.vertical_discr = vertical_discr

    state.vert_weight = define_vertical_weight(
        cfg_numerics.Nz, cfg_numerics.vert_spacing
    )

    # Initialize ice-flow method
    iceflow_method = cfg.processes.iceflow.method.lower()

    if iceflow_method == "emulated":
        initialize_iceflow = initialize_iceflow_emulator
    elif iceflow_method == "solved":
        initialize_iceflow = initialize_iceflow_solver
    elif iceflow_method == "diagnostic":
        initialize_iceflow = initialize_iceflow_diagnostic
    elif iceflow_method == "unified":
        initialize_iceflow = initialize_iceflow_unified
    else:
        raise ValueError(f"❌ Unknown ice flow method: <{iceflow_method}>.")

    initialize_iceflow(cfg, state)


def update(cfg: DictConfig, state: State) -> None:

    # Logger
    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at iteration : " + str(state.it))

    # Update ice-flow method
    iceflow_method = cfg.processes.iceflow.method.lower()

    if iceflow_method == "emulated":
        update_iceflow = update_iceflow_emulator
    elif iceflow_method == "solved":
        update_iceflow = update_iceflow_solved
    elif iceflow_method == "diagnostic":
        update_iceflow = update_iceflow_diagnostic
    elif iceflow_method == "unified":
        update_iceflow = update_iceflow_unified
    else:
        raise ValueError(f"❌ Unknown ice flow method: <{iceflow_method}>.")

    update_iceflow(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:

    # Save emulated model
    iceflow_method = cfg.processes.iceflow.method.lower()
    if iceflow_method == "emulated":
        if cfg.processes.iceflow.emulator.save_model:
            save_iceflow_model(cfg, state)
