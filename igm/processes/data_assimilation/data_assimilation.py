#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
import tensorflow as tf

from .utils import compute_rms_std_optimization
from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings
from igm.processes.iceflow.unified.optimizers import Optimizers, InterfaceOptimizers


from igm.processes.iceflow.emulate.emulator import update_iceflow_emulator
from igm.processes.iceflow.utils.data_preprocessing import get_fieldin

from igm.processes.iceflow.data_preparation.data_preprocessing_tensor import (
    create_input_tensor_from_fieldin,
)

from igm.processes.iceflow import initialize as iceflow_initialize

class DataAssimilation:
    pass

def data_assimilation_initialize(cfg, state):

    cfg_da = cfg.processes.data_assimilation

    # Initialize mapping
    mapping_args = InterfaceMappings["data_assimilation"].get_mapping_args(cfg, state)
    mapping = Mappings["data_assimilation"](**mapping_args)

    # Initialize optimizer
    optimizer_name = cfg_da.optimizer
    optimizer_args = InterfaceOptimizers[optimizer_name].get_optimizer_args(
        cfg=cfg, cost_fn=get_cost_fn(cfg, state), map=mapping
    )

    optimizer = Optimizers[optimizer_name](**optimizer_args)
    state.data_assimilation = DataAssimilation()
    state.data_assimilation.optimizer = optimizer

def get_cost_fn(cfg, state):
    
    def cost_function(U, V, inputs):

        U_reshape = U[0,-1,:,:] # surface velocity, assumes no patching!
        V_reshape = V[0,-1,:,:]

        velsurf    = tf.stack([U_reshape, V_reshape], axis=-1)
        velsurfobs = tf.stack([state.uvelsurfobs, state.vvelsurfobs], axis=-1)

        REL = tf.expand_dims((tf.norm(velsurfobs, axis=-1) >= cfg.processes.data_assimilation.fitting.velsurfobs_thr), axis=-1)
        ACT = ~tf.math.is_nan(velsurfobs) 

        cost = 0.5 * tf.reduce_mean(
            ((velsurfobs[ACT & REL] - velsurf[ACT & REL]) / cfg.processes.data_assimilation.fitting.velsurfobs_std) ** 2
        )
        return cost
    
    return cost_function



def initialize(cfg, state):
    iceflow_initialize(cfg, state)  # initialize the iceflow model
    data_assimilation_initialize(cfg, state)  # initialize the optimizer

    fieldin = get_fieldin(cfg, state)
    inputs = create_input_tensor_from_fieldin(
        fieldin, state.iceflow.patching, state.iceflow.preparation_params
    )

    state.cost = state.data_assimilation.optimizer.minimize(inputs)

    # Update state with optimized values using the mapping method
    state.data_assimilation.optimizer.map.update_state_fields(state)

def update(cfg, state):

    fieldin = get_fieldin(cfg, state)
    inputs = create_input_tensor_from_fieldin(
        fieldin, state.iceflow.patching, state.iceflow.preparation_params
    )

    state.cost = state.iceflow.optimizer.minimize(inputs)


    state.cost = state.data_assimilation.optimizer.minimize(inputs)

    # Update state with optimized values using the mapping method
    state.data_assimilation.optimizer.map.update_state_fields(state)


def finalize(cfg, state):
    pass
