#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
import tensorflow as tf

from igm.common.core import State
from igm.processes.iceflow.data_preparation import preparation_params as prep
from igm.processes.iceflow.energy.utils import get_energy_components

from .evaluator import evaluate_iceflow, EvaluatorParams, get_evaluator_params_args
from .solver import solve_iceflow, SolverParams, get_solver_params_args


def initialize_iceflow_solver(cfg: DictConfig, state: State) -> None:

    if not hasattr(cfg, "processes"):
        raise AttributeError("❌ <cfg.processes> does not exist.")
    if not hasattr(cfg.processes, "iceflow"):
        raise AttributeError("❌ <cfg.processes.iceflow> does not exist.")
    if not hasattr(state, "thk"):
        raise AttributeError("❌ <state.thk> does not exist.")

    # Initialize optimizer
    optimizer_name = cfg.processes.iceflow.solver.optimizer.lower()
    learning_rate = cfg.processes.iceflow.solver.step_size

    if optimizer_name == "adam":
        version_tf = int(tf.__version__.split(".")[1])
        if (version_tf <= 10) | (version_tf >= 16):
            module_optimizer = tf.keras.optimizers
        else:
            module_optimizer = tf.keras.optimizers.legacy

        state.optimizer = module_optimizer.Adam(learning_rate=learning_rate)
    else:
        state.optimizer = None

    # Initialize energy components
    state.iceflow.energy_components = get_energy_components(cfg)

    # Evaluator params
    evaluator_params_args = get_evaluator_params_args(cfg)
    evaluator_params = EvaluatorParams(**evaluator_params_args)
    state.iceflow.evaluator_params = evaluator_params

    # Solver params
    preparation_params_args = prep.get_input_params_args(cfg)
    preparation_params = prep.PreparationParams(**preparation_params_args)

    input_height = state.thk.shape[0]
    input_width = state.thk.shape[1]

    Ny, Nx, _, batch_size, _ = prep.calculate_expected_dimensions(
        input_height, input_width, preparation_params
    )

    solver_params_args = get_solver_params_args(cfg, Nx, Ny, batch_size)
    solver_params = SolverParams(**solver_params_args)
    state.iceflow.solver_params = solver_params


def update_iceflow_solved(cfg: DictConfig, state: State) -> None:

    # Solve ice flow
    solve_iceflow(cfg, state)

    # Evaluate ice flow
    evaluate_iceflow(cfg, state)
