#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common.core import State
from .bcs import BoundaryConditions
from igm.common import print_model_with_inputs, print_model_with_inputs_detailed
from .mappings import Mappings, InterfaceMappings
from .optimizers import Optimizers, InterfaceOptimizers
from .evaluator import EvaluatorParams, get_evaluator_params_args, evaluate_iceflow
from .solver import solve_iceflow
from .utils import get_cost_fn, _print_data_preparation_summary
from igm.common import print_model_with_inputs_detailed
from igm.processes.iceflow.data_preparation.config import (
    PreparationParams,
    get_input_params_args,
)
from igm.processes.iceflow.data_preparation.patching import OverlapPatching
from igm.processes.iceflow.data_preparation.batch_builder import TrainingBatchBuilder
from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_state_to_X,
    X_to_fieldin,
)


def initialize_iceflow_unified(cfg: DictConfig, state: State) -> None:
    """Initialize iceflow module in unified mode."""

    # Initialize training set
    preparation_params_args = get_input_params_args(cfg)
    preparation_params = PreparationParams(**preparation_params_args)

    state.iceflow.preparation_params = preparation_params
    X = fieldin_state_to_X(cfg, state)
    fieldin_dict = X_to_fieldin(
        X,
        fieldin_names=preparation_params.fieldin_names,
        dim_arrhenius=cfg.processes.iceflow.physics.dim_arrhenius,
        Nz=cfg.processes.iceflow.numerics.Nz,
    )

    state.iceflow.patching = OverlapPatching(
        patch_size=preparation_params.patch_size,
        overlap=preparation_params.overlap,
        fieldin=X,
    )
    num_patches = state.iceflow.patching.num_patches  # int
    patch_H, patch_W, patch_C = (
        state.iceflow.patching.patch_shape
    )  # patch spatial dimensions

    # Initialize mapping
    mapping_name = cfg.processes.iceflow.unified.mapping
    mapping_args = InterfaceMappings[mapping_name].get_mapping_args(cfg, state)
    mapping = Mappings[mapping_name](**mapping_args)
    state.iceflow.mapping = mapping

    # Initialize optimizer
    optimizer_name = cfg.processes.iceflow.unified.optimizer
    optimizer_args = InterfaceOptimizers[optimizer_name].get_optimizer_args(
        cfg=cfg, cost_fn=get_cost_fn(cfg, state), map=mapping
    )
    optimizer = Optimizers[optimizer_name](**optimizer_args)
    state.iceflow.optimizer = optimizer

    sampler = TrainingBatchBuilder(
        preparation_params=preparation_params,
        fieldin_names=preparation_params.fieldin_names,
        patch_shape=(patch_H, patch_W, patch_C),
        num_patches=num_patches,
    )
    optimizer.sampler = sampler

    _print_data_preparation_summary(
        prep=preparation_params,
        X=X,
        patching=state.iceflow.patching,
        sampler=sampler,
    )

    # Evaluator params
    evaluator_params_args = get_evaluator_params_args(cfg)
    evaluator_params = EvaluatorParams(**evaluator_params_args)
    state.iceflow.evaluator_params = evaluator_params

    if (
        cfg.processes.iceflow.unified.mapping == "network"
        and cfg.processes.iceflow.unified.network.print_summary
    ):
        print_model_with_inputs_detailed(
            model=state.iceflow_model,
            input_data=fieldin_dict,
            cfg_inputs=cfg.processes.iceflow.unified.inputs,
            normalization_method=cfg.processes.iceflow.unified.normalization.method,
        )

    # Solve once
    solve_iceflow(cfg, state, init=True)

    # Evaluate once
    evaluate_iceflow(cfg, state)


def update_iceflow_unified(cfg: DictConfig, state: State) -> None:
    """Update iceflow module in unified mode."""

    # Solve ice flow
    solve_iceflow(cfg, state)

    # Evalute ice flow
    evaluate_iceflow(cfg, state)
