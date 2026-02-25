#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
from igm.processes.iceflow.unified.evaluator import evaluate_iceflow

from .outputs.output_ncdf import update_ncdf_optimize

from .utils import initial_thickness
from igm.utils.math.precision import normalize_precision

from igm.processes.iceflow.unified.mappings.data_assimilation import MappingDataAssimilation
from igm.processes.iceflow.unified.mappings.interfaces.data_assimilation import InterfaceDataAssimilation
from igm.processes.iceflow.unified.optimizers.lbfgs_DA import OptimizerLBFGSBoundsDA
from igm.processes.iceflow.unified.optimizers.interfaces import InterfaceLBFGS

from igm.processes.iceflow.data_preparation.batch_builder import TrainingBatchBuilder
from igm.processes.data_assimilation_SR.objective import build_objective_from_cfg

class DataAssimilation:
    def __init__(self):
        self.map = None
        self.opt = None
        self.cost_fn = None
        self.maxiter = 0
        self.out_freq = 0
        self.result = None


def get_cost_and_obj(cfg, state, da_map):
    objective = build_objective_from_cfg(cfg, state, da_map)

    def cost_function(U, V, inputs):
        total, misfit, reg, _ = objective(U, V, inputs)
        return total, misfit, reg

    return cost_function, objective

def data_assimilation_initialize(cfg, state):
    cfg_da = cfg.processes.data_assimilation_SR
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    da = DataAssimilation()

    # Initial thickness and slidingco guess (for now keep this simple, could be improved in the future with more sophisticated initializations)
    # mainly I just want to avoid cheating by unintentionally using the true thickness as initial guess
    thk0 = initial_thickness(
        s=state.usurf,
        u=state.uvelsurfobs,
        v=state.vvelsurfobs,
        mask=state.icemask,
        dx=state.dX[0, 0],
        dy=state.dX[0, 0],
    )
    slidingco0 = np.zeros_like(thk0) + cfg.processes.iceflow.physics.init_slidingco 

    state.uvelsurfobs = tf.cast(state.uvelsurfobs, dtype=dtype)
    state.vvelsurfobs = tf.cast(state.vvelsurfobs, dtype=dtype)
    state.thk = tf.convert_to_tensor(thk0, dtype=dtype)
    state.slidingco = tf.convert_to_tensor(slidingco0, dtype=dtype)     
    state.usurf = tf.cast(state.usurf, dtype)
    state.dX = tf.cast(state.dX, dtype) 

    mapping_args = InterfaceDataAssimilation.get_mapping_args(cfg, state)
    da.map = MappingDataAssimilation(**mapping_args)

    da.cost_fn, da.objective = get_cost_and_obj(cfg, state, da.map)

    optimizer_args = InterfaceLBFGS.get_optimizer_args(cfg, da.cost_fn, da.map)
    da.opt = OptimizerLBFGSBoundsDA(**optimizer_args)

    num_patches = state.iceflow.patching.num_patches
    patch_H, patch_W, patch_C = state.iceflow.patching.patch_shape
    sampler = TrainingBatchBuilder(
        preparation_params=state.iceflow.preparation_params,
        fieldin_names=state.iceflow.preparation_params.fieldin_names,
        patch_shape=(patch_H, patch_W, patch_C),
        num_patches=num_patches,
    )
    da.opt.sampler = sampler

    da.maxiter = cfg_da.optimization.nbitmax
    da.out_freq = cfg_da.output.freq

    def _step_callback(it_tf):
        it = int(it_tf.numpy())

        # For outputs (not for gradients): sync state fields for netcdf writing
        da.map.update_state_fields(state)

        X = fieldin_state_to_X(cfg, state)
        inputs = state.iceflow.patching.generate_patches(X)

        U, V = da.map.get_UV(inputs)
        inputs_used = da.map.inputs if hasattr(da.map, "inputs") else inputs
        total, data, reg = da.cost_fn(U, V, inputs_used)

        state.da_cost_total = float(total.numpy())
        state.da_cost_data  = float(data.numpy())
        state.da_cost_reg   = float(reg.numpy())

        evaluate_iceflow(cfg, state)
        update_ncdf_optimize(cfg, state, it)

    da.map.set_step_callback(_step_callback, out_freq=da.out_freq)
    state.data_assimilation = da

def initialize(cfg, state):
    data_assimilation_initialize(cfg, state)

def update(cfg, state):
    da = state.data_assimilation

    # Initial output (iteration 0)
    da.map.update_state_fields(state)
    evaluate_iceflow(cfg, state)
    update_ncdf_optimize(cfg, state, 0)

    X = fieldin_state_to_X(cfg, state)
    inputs = state.iceflow.patching.generate_patches(X)

    da.opt.minimize(inputs)

    # Final write
    da.map.update_state_fields(state)
    evaluate_iceflow(cfg, state)
    update_ncdf_optimize(cfg, state, int(da.maxiter))


def finalize(cfg, state):
    pass