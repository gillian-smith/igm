#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
from igm.processes.iceflow.unified.evaluator import evaluate_iceflow

from .outputs.output_ncdf import update_ncdf_optimize
from igm.processes.iceflow.utils.velocities import get_velsurf

from .initial_ice_thickness import initial_thickness
from igm.utils.math.precision import normalize_precision

from igm.processes.iceflow.unified.mappings.data_assimilation import MappingDataAssimilation
from igm.processes.iceflow.unified.mappings.interfaces.data_assimilation import InterfaceDataAssimilation
from igm.processes.iceflow.unified.optimizers.lbfgs_DA import OptimizerLBFGSBoundsDA
from igm.processes.iceflow.unified.optimizers.interfaces import InterfaceLBFGS

from igm.processes.iceflow.data_preparation.batch_builder import TrainingBatchBuilder

class DataAssimilation:
    def __init__(self):
        self.map = None
        self.opt = None
        self.cost_fn = None
        self.maxiter = 0
        self.out_freq = 0
        self.result = None

def smoothness_biharmonic(field, dx, lam):
    dtype = field.dtype
    f = field[None, ..., None]  # [1, Ny, Nx, 1]
    k = tf.constant([[0., 1., 0.],
                     [1., -4., 1.],
                     [0., 1., 0.]], dtype=dtype)
    k = k[:, :, None, None]  # [3,3,1,1]
    fpad = tf.pad(f, [[0,0],[1,1],[1,1],[0,0]], mode="REFLECT")
    lap = tf.nn.conv2d(fpad, k, strides=1, padding="VALID")
    lap = tf.cast(lap, dtype)

    dx_squared = tf.reshape(dx * dx, [1, tf.shape(dx)[0], tf.shape(dx)[1], 1])
    dx_squared = tf.cast(dx_squared, dtype)
    lap /= dx_squared

    half = tf.constant(0.5, dtype=dtype)
    return lam * half * tf.reduce_mean(tf.square(lap))


def get_cost_fn_data(cfg, state):
    def cost_function(U, V, inputs):
        dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

        U = U[0]
        V = V[0]

        uvelsurf, vvelsurf = get_velsurf(U, V, state.iceflow.vertical_discr.V_s)

        uobs = tf.cast(state.uvelsurfobs, dtype)
        vobs = tf.cast(state.vvelsurfobs, dtype)

        mask = tf.logical_and(~tf.math.is_nan(uobs), ~tf.math.is_nan(vobs))

        ru = (uobs - uvelsurf) / cfg.processes.SR_DA.fitting.velsurfobs_std
        rv = (vobs - vvelsurf) / cfg.processes.SR_DA.fitting.velsurfobs_std

        # Vector SSE per point, then mean over points
        cost_data = 0.5 * tf.reduce_mean(tf.boolean_mask(ru**2 + rv**2, mask))

        # Regularization based on the (patched) inputs tensor
        current_thk = tf.cast(state.thk, dtype)
        lam = tf.cast(cfg.processes.SR_DA.regularization.thk, dtype)
        dx = tf.cast(state.dX, dtype)
        cost_reg = smoothness_biharmonic(current_thk, dx, lam)

        cost_total = tf.cast(cost_data, dtype) + tf.cast(cost_reg, dtype)
        return cost_total, tf.cast(cost_data, dtype), tf.cast(cost_reg, dtype)

    return cost_function

def data_assimilation_initialize(cfg, state):
    cfg_da = cfg.processes.SR_DA
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    da = DataAssimilation()
    da.cost_fn = get_cost_fn_data(cfg, state)

    # Initial thickness guess (physical)
    thk0 = initial_thickness(
        s=state.usurf,
        u=state.uvelsurfobs,
        v=state.vvelsurfobs,
        mask=state.icemask,
        dx=state.dX[0, 0],
        dy=state.dX[0, 0],
    )
    thk0 = tf.convert_to_tensor(thk0, dtype=dtype)

    # Put initial physical field in state (mapping will read from state to initialize θ)
    state.thk = thk0
    state.usurf = tf.cast(state.usurf, dtype)

    mapping_args = InterfaceDataAssimilation.get_mapping_args(cfg, state)
    da.map = MappingDataAssimilation(**mapping_args)

    optimizer_args = InterfaceLBFGS.get_optimizer_args(
        cfg,
        da.cost_fn,
        da.map,
    )
    da.opt = OptimizerLBFGSBoundsDA(**optimizer_args)

    num_patches = state.iceflow.patching.num_patches  # int
    patch_H, patch_W, patch_C = (
        state.iceflow.patching.patch_shape
    ) 
    sampler = TrainingBatchBuilder(
        preparation_params=state.iceflow.preparation_params,
        fieldin_names=state.iceflow.preparation_params.fieldin_names,
        patch_shape=(patch_H, patch_W, patch_C),
        num_patches=num_patches,
    )
    da.opt.sampler = sampler

    # Iteration controls
    da.maxiter = cfg_da.optimization.nbitmax
    da.out_freq = cfg_da.output.freq

    def _step_callback(it_tf):
        it = int(it_tf.numpy())
        # keep state consistent for writing
        da.map.update_state_fields(state)
        # Build inputs for THIS accepted iterate
        X = fieldin_state_to_X(cfg, state)
        inputs = state.iceflow.patching.generate_patches(X)
        # Forward + cost terms
        U, V = da.map.get_UV(inputs)
        inputs_used = da.map.inputs if hasattr(da.map, "inputs") else inputs
        total, data, reg = da.cost_fn(U, V, inputs_used)

        # Put scalars somewhere stable for the NetCDF writer to read
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

    # Build patches ONCE (mapping will inject θ-dependent fields each evaluation)
    X = fieldin_state_to_X(cfg, state)
    inputs = state.iceflow.patching.generate_patches(X)

    da.opt.minimize(inputs)

    # Final write (ensure state matches final θ)
    da.map.update_state_fields(state)
    evaluate_iceflow(cfg, state)
    update_ncdf_optimize(cfg, state, int(da.maxiter))


def finalize(cfg, state):
    pass