#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
import tensorflow as tf

from igm.common.core import State

from ..optimizers import InterfaceOptimizers, Status
from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_state_to_X,
)


def get_status(cfg: DictConfig, state: State, init: bool = False) -> Status:

    cfg_unified = cfg.processes.iceflow.unified
    warm_up_it = cfg_unified.warm_up_it
    retrain_freq = cfg_unified.retrain_freq

    if init:
        status = Status.INIT
    elif state.it <= warm_up_it:
        status = Status.WARM_UP
    elif retrain_freq > 0 and state.it > 0 and state.it % retrain_freq == 0:
        status = Status.DEFAULT
    else:
        status = Status.IDLE

    return status


def get_solver_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Returns [N, H, W, C] patches (sampler handles batching/augmentation)."""
    fieldin = fieldin_state_to_X(cfg, state)

    # Create patches using the patching object
    patches = state.iceflow.patching.generate_patches(fieldin)
    
    return patches


def solve_iceflow(cfg: DictConfig, state: State, init: bool = False) -> None:

    # Get status: should we optimize again?
    status = get_status(cfg, state, init)

    # Get optimizer
    optimizer = state.iceflow.optimizer

    # Set optimizer parameters
    set_optimizer_params = InterfaceOptimizers[optimizer.name].set_optimizer_params
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # Optimize and save cost
    if do_solve:
        inputs = get_solver_inputs_from_state(cfg, state)
        state.cost = optimizer.minimize(inputs)
