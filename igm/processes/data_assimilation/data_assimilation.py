#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
import tensorflow as tf
import numpy as np
import xarray as xr

from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings
from igm.processes.iceflow.unified.optimizers import Optimizers, InterfaceOptimizers

from igm.processes.iceflow.utils.data_preprocessing import get_fieldin

from igm.processes.iceflow.data_preparation.data_preprocessing_tensor import (
    create_input_tensor_from_fieldin,
)

from igm.processes.iceflow.unified.evaluator import evaluate_iceflow
from igm.processes.iceflow.unified.utils import get_cost_fn


from .outputs.output_ncdf import output_ncdf_optimize_final

class DataAssimilation:
    pass

def data_assimilation_initialize(cfg, state):

    cfg_da = cfg.processes.data_assimilation

    # Initialize mapping
    mapping_key = "combined_data_assimilation"
    mapping_args = InterfaceMappings[mapping_key].get_mapping_args(cfg, state)
    mapping = Mappings[mapping_key](**mapping_args)

    # Initialize optimizer with combined cost function
    optimizer_name = cfg_da.optimizer
    optimizer_args = InterfaceOptimizers[optimizer_name].get_optimizer_args(
        cfg=cfg, cost_fn=get_data_assimilation_cost_fn(cfg, state), map=mapping
    )

    optimizer_name_da = cfg_da.optimizer + "_da"

    optimizer = Optimizers[optimizer_name_da](**optimizer_args)
    state.data_assimilation = DataAssimilation()
    state.data_assimilation.optimizer = optimizer
    state.data_assimilation.cost_components = {}

def smoothness_biharmonic(field, dx, lam):
    # field: [Nx, Ny] float tensor
    f = field[None, ..., None]  # [1, Nx, Ny, 1]
    # Discrete 5-point Laplacian kernel (isotropic on a square grid)
    k = tf.constant([[0., 1., 0.],
                     [1., -4., 1.],
                     [0., 1., 0.]], dtype=field.dtype)
    k = k[:, :, None, None]  # [3,3,1,1]
    # Reflect padding avoids edge bias and keeps shape
    fpad = tf.pad(f, [[0,0],[1,1],[1,1],[0,0]], mode="REFLECT")
    lap = tf.nn.conv2d(fpad, k, strides=1, padding="VALID")
    # Scale for physical spacing (Δx, Δy). For a rectangular grid, divide by Δx² and Δy² appropriately.
    # 5-point assumes Δx=Δy; for Δx≠Δy, you can use separable stencils or scale neighbors differently.
    lap /= (dx*dx)  # if dx==dy; otherwise use a non-isotropic laplacian kernel (see note below)

    # Biharmonic (curvature) penalty: ||∇² h||²
    return lam * 0.5 * tf.reduce_mean(tf.square(lap))


def get_data_assimilation_cost_fn(cfg, state):

    physics_cost_fn = get_cost_fn(cfg, state)

    def data_loss(U, V, inputs):
        U_reshape = U[0,-1,:,:] # surface velocity, assumes no patching!
        V_reshape = V[0,-1,:,:]

        velsurf    = tf.stack([U_reshape, V_reshape], axis=-1)
        # Cast observations to match the precision of the forward pass results
        velsurfobs = tf.stack([tf.cast(state.uvelsurfobs, U_reshape.dtype), 
                              tf.cast(state.vvelsurfobs, U_reshape.dtype)], axis=-1)

        REL = tf.expand_dims((tf.norm(velsurfobs, axis=-1) >= cfg.processes.data_assimilation.fitting.velsurfobs_thr), axis=-1)
        ACT = ~tf.math.is_nan(velsurfobs) 

        cost1 = 0.5 * tf.reduce_mean(
            ((velsurfobs[ACT & REL] - velsurf[ACT & REL]) / cfg.processes.data_assimilation.fitting.velsurfobs_std) ** 2
        )

        return cost1

    def physics_loss(U, V, inputs):
        return physics_cost_fn(U, V, inputs)
    
    def cost_function(U, V, inputs):

        cost1 = data_loss(U, V, inputs)
        cost2 = physics_loss(U, V, inputs)

        # Extract current thickness from inputs (channel 0) instead of state.thk
        # current_thk = inputs[0, :, :, 0]  # [batch, height, width, channels] -> [height, width]
        # lam = tf.cast(cfg.processes.data_assimilation.regularization.thk, current_thk.dtype)
        # REGU_H2 = smoothness_biharmonic(current_thk, state.dx, lam)  # Use current optimized thickness

        cost = cost1 + tf.constant(10000.0, dtype=cost2.dtype) * cost2 # + REGU_H2
        # print both costs
        # tf.print("Data cost:", cost1, "Physics cost:", cost2)  # , "Regu cost:", REGU_H2)
        return cost, cost1, cost2

    return cost_function

def write_progress_to_netcdf(prog: dict, path: str, *, attrs: dict | None = None, complevel: int = 4) -> str:
    """
    prog: {"iteration": [ints], "fields": {name: [np.ndarray (H,W), ...], ...}}
    path: output .nc path
    """

    it = np.asarray(prog["iteration"], dtype=np.int64)
    data_vars = {}
    for name, snaps in prog["fields"].items():
        arr = np.stack(snaps, axis=0)  # (T, H, W)
        H, W = arr.shape[1], arr.shape[2]
        data_vars[name] = xr.DataArray(arr, dims=("time", "y", "x"),
                                       coords={"time": it, "y": np.arange(H), "x": np.arange(W)})

    ds = xr.Dataset(data_vars)
    if attrs: ds.attrs.update(attrs)

    enc = {k: {"zlib": True, "complevel": int(complevel)} for k in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    return path

def initialize(cfg, state):

    data_assimilation_initialize(cfg, state)  # initialize the optimizer


def update(cfg, state):

    fieldin = get_fieldin(cfg, state)
    inputs = create_input_tensor_from_fieldin(
        fieldin, state.iceflow.patching, state.iceflow.preparation_params
    )

    state.cost = state.data_assimilation.optimizer.minimize(inputs)

    prog = state.data_assimilation.optimizer.map.get_progress()
    out = write_progress_to_netcdf(prog, "progress_snapshots.nc", attrs={"source":"LBFGS CombinedDA"})
    print("Wrote", out)

    # Update state with optimized values using the mapping method
    state.data_assimilation.optimizer.map.update_state_fields(state)

    evaluate_iceflow(cfg, state)

    output_ncdf_optimize_final(cfg, state)

def finalize(cfg, state):
    pass
