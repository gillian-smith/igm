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

def add_measurement_noise_uv(uobs, vobs, ice_mask, noise_factor, base_std):
    """
    Add one realization of iid Gaussian noise to u/v observations,
    but ONLY where ice_mask is True and u/v are finite.

    noise_factor: dimensionless multiplier (0 => no noise)
    base_std:      velocity std in observation units (e.g. m/yr)
    """
    noise_factor = float(noise_factor)
    base_std = float(base_std)
    sigma = noise_factor * base_std

    u = np.asarray(uobs)
    v = np.asarray(vobs)
    ice = np.asarray(ice_mask).astype(bool)

    if sigma <= 0.0:
        return u, v, 0.0

    valid = np.isfinite(u) & np.isfinite(v) & ice
    if not np.any(valid):
        return u, v, sigma

    nu = np.random.normal(loc=0.0, scale=sigma, size=u.shape).astype(u.dtype, copy=False)
    nv = np.random.normal(loc=0.0, scale=sigma, size=v.shape).astype(v.dtype, copy=False)

    u_noisy = u.copy()
    v_noisy = v.copy()
    u_noisy[valid] = u_noisy[valid] + nu[valid]
    v_noisy[valid] = v_noisy[valid] + nv[valid]

    return u_noisy, v_noisy, sigma

def masked_mean(x: tf.Tensor, mask: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    m = tf.cast(mask, x.dtype)
    num = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)))
    den = tf.reduce_sum(m) + tf.cast(eps, x.dtype)
    return num / den

def smoothness_biharmonic(field, dx, lam, mask=None, eps=1e-12):
    dtype = field.dtype
    f = field[None, ..., None]  # [1, Ny, Nx, 1]

    k = tf.constant([[0., 1., 0.],
                     [1., -4., 1.],
                     [0., 1., 0.]], dtype=dtype)[:, :, None, None]

    fpad = tf.pad(f, [[0,0],[1,1],[1,1],[0,0]], mode="REFLECT")
    lap = tf.nn.conv2d(fpad, k, strides=1, padding="VALID")  # [1, Ny, Nx, 1]
    dx2 = tf.reshape(dx * dx, [1, tf.shape(dx)[0], tf.shape(dx)[1], 1])
    lap = lap / tf.cast(dx2, dtype)

    lap2 = tf.square(lap)[0, :, :, 0]  # [Ny, Nx]

    if mask is None:
        mean_lap2 = tf.reduce_mean(lap2)
    else:
        mean_lap2 = masked_mean(lap2, tf.cast(mask, tf.bool), eps=eps)

    return tf.cast(lam, dtype) * tf.cast(0.5, dtype) * mean_lap2

def get_cost_fn_data(cfg, state, da_map):
    def cost_function(U, V, inputs):
        dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

        U = U[0]
        V = V[0]

        uvelsurf, vvelsurf = get_velsurf(U, V, state.iceflow.discr_v.V_s)

        uobs = tf.cast(state.uvelsurfobs, dtype)
        vobs = tf.cast(state.vvelsurfobs, dtype)

        valid = tf.math.is_finite(uobs) & tf.math.is_finite(vobs)
        ice = tf.cast(state.icemask, tf.bool)
        active = valid & ice


        std = tf.cast(cfg.processes.SR_DA.fitting.velsurfobs_std, dtype)
        ru = (uobs - uvelsurf) / std
        rv = (vobs - vvelsurf) / std

        res2 = tf.cast(ru**2 + rv**2, dtype)  # x is float dtype

        cost_data = tf.cast(0.5, dtype) * masked_mean(res2, active, eps=1e-12)


        current_thk = da_map.get_physical_field("thk") # this is needed to ensure the gradient tape tracks thk

        lam = tf.cast(cfg.processes.SR_DA.regularization.thk, dtype)
        dx = tf.cast(state.dX, dtype)
        cost_reg = smoothness_biharmonic(current_thk, dx, lam, mask=active, eps=1e-12)

        cost_total = tf.cast(cost_data + cost_reg, dtype)
        
        return cost_total, tf.cast(cost_data, dtype), tf.cast(cost_reg, dtype)

    return cost_function

def data_assimilation_initialize(cfg, state):
    cfg_da = cfg.processes.SR_DA
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    da = DataAssimilation()

    # --- Measurement noise injection (one-time) ---
    noise_factor = float(getattr(cfg_da, "measurement_noise", 0.0))
    base_std = float(cfg_da.fitting.velsurfobs_std)

    u_noisy, v_noisy, sigma = add_measurement_noise_uv(
        state.uvelsurfobs,
        state.vvelsurfobs,
        ice_mask=state.icemask,
        noise_factor=noise_factor,
        base_std=base_std,
    )

    # Overwrite the observations used everywhere downstream (cost, plots, init, etc.)
    state.uvelsurfobs = u_noisy
    state.vvelsurfobs = v_noisy

    # Initial thickness guess (physical)
    thk0 = initial_thickness(
        s=state.usurf,
        u=state.uvelsurfobs,
        v=state.vvelsurfobs,
        mask=state.icemask,
        dx=state.dX[0, 0],
        dy=state.dX[0, 0],
    )
    state.uvelsurfobs = tf.convert_to_tensor(u_noisy, dtype=dtype)
    state.vvelsurfobs = tf.convert_to_tensor(v_noisy, dtype=dtype)
    state.thk = tf.convert_to_tensor(thk0, dtype=dtype)
    state.usurf = tf.cast(state.usurf, dtype)

    # 1) Build mapping first
    mapping_args = InterfaceDataAssimilation.get_mapping_args(cfg, state)
    da.map = MappingDataAssimilation(**mapping_args)

    # 2) Build cost_fn capturing the map
    da.cost_fn = get_cost_fn_data(cfg, state, da.map)

    # 3) Build optimizer after cost_fn exists
    optimizer_args = InterfaceLBFGS.get_optimizer_args(cfg, da.cost_fn, da.map)
    da.opt = OptimizerLBFGSBoundsDA(**optimizer_args)

    # sampler wiring unchanged...
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