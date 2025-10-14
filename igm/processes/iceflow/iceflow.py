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
import tensorflow as tf
import warnings

from igm.processes.iceflow.emulate.emulated import (
    update_iceflow_emulated,
)
from igm.processes.iceflow.emulate.emulator import (
    update_iceflow_emulator,
    initialize_iceflow_emulator,
)
from igm.processes.iceflow.emulate.utils import save_iceflow_model

from igm.processes.iceflow.utils.misc import initialize_iceflow_fields
from igm.processes.iceflow.utils.data_preprocessing import (
    compute_PAD,
    get_fieldin,
)

from igm.processes.iceflow.utils.vertical_discretization import define_vertical_weight

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
from igm.processes.iceflow.vertical import VerticalDiscrs


class Iceflow:
    pass


def initialize(cfg, state):

    # Make sure this function is only called once
    if hasattr(state, "was_initialize_iceflow_already_called"):
        return
    else:
        state.was_initialize_iceflow_already_called = True

    # Create ice flow object
    iceflow = Iceflow()
    state.iceflow = iceflow

    # Parameters aliases
    cfg_emulator = cfg.processes.iceflow.emulator
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    # deinfe the fields of the ice flow such a U, V, but also sliding coefficient, arrhenius, ectt
    initialize_iceflow_fields(cfg, state)

    # Initialize vertical discretization
    vertical_basis = cfg_numerics.vert_basis.lower()
    vertical_discr = VerticalDiscrs[vertical_basis](cfg)
    state.iceflow.vertical_discr = vertical_discr

    # Set vertical weights
    state.vert_weight = define_vertical_weight(
        cfg_numerics.Nz, cfg_numerics.vert_spacing
    )

    # padding is necessary when using U-net emulator
    state.PAD = compute_PAD(
        cfg_emulator.network.multiple_window_size,
        state.thk.shape[1],
        state.thk.shape[0],
    )

    # Set ice-flow method
    iceflow_method = cfg.processes.iceflow.method.lower()

    if iceflow_method == "emulated":
        # define the emulator, and the optimizer
        initialize_iceflow_emulator(cfg, state)
    elif iceflow_method == "solved":
        # define the solver, and the optimizer
        initialize_iceflow_solver(cfg, state)
    elif iceflow_method == "diagnostic":
        # define the second velocity field
        initialize_iceflow_diagnostic(cfg, state)
    elif iceflow_method == "unified":
        # define the velocity through a mapping
        initialize_iceflow_unified(cfg, state)
    else:
        raise ValueError(f"❌ Unknown ice flow method: <{iceflow_method}>.")

    if not iceflow_method in ["solved", "unified"]:

        fieldin = get_fieldin(cfg, state)

        update_iceflow_emulator(
            cfg,
            state,
            fieldin,
            initial=True,
            it=0,
        )

        update_iceflow_emulated(cfg, state, fieldin)

    assert (cfg_emulator.exclude_borders == 0) | (
        cfg_emulator.network.multiple_window_size == 0
    )
    # if (cfg.processes.iceflow.emulator.exclude_borders==0) and (cfg.processes.iceflow.emulator.network.multiple_window_size==0):
    # raise ValueError("The emulator must exclude borders or use multiple windows, otherwise it will not work properly.")


def update(cfg, state):

    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at iteration : " + str(state.it))

    iceflow_method = cfg.processes.iceflow.method.lower()

    if iceflow_method in ["emulated", "diagnostic"]:

        fieldin = get_fieldin(cfg, state)

        update_iceflow_emulator(
            cfg,
            state,
            fieldin,
            initial=False,
            it=state.it,
        )

        update_iceflow_emulated(cfg, state, fieldin)

        if iceflow_method == "diagnostic":
            update_iceflow_diagnostic(cfg, state)

    elif iceflow_method == "solved":
        update_iceflow_solved(cfg, state)

    elif iceflow_method == "unified":
        update_iceflow_unified(cfg, state)

    else:
        raise ValueError(f"❌ Unknown ice flow method: <{iceflow_method}>.")


def finalize(cfg, state):

    if cfg.processes.iceflow.emulator.save_model:
        save_iceflow_model(cfg, state)
