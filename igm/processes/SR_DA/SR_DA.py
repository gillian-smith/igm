#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf

# Off-the-shelf optimizer
from scipy.optimize import minimize, Bounds  # pip install scipy

from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
from igm.processes.iceflow.unified.evaluator import evaluate_iceflow

from .outputs.output_ncdf import update_ncdf_optimize
from igm.processes.iceflow.utils.velocities import get_velsurf

from .initial_ice_thickness import initial_thickness
from igm.utils.math.precision import normalize_precision

class DataAssimilation:
    pass


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

        uvelsurf, vvelsurf = get_velsurf(U, V, state.iceflow.vertical_discr.V_b)
        velsurf = tf.stack([uvelsurf, vvelsurf], axis=-1)

        velsurfobs = tf.stack([state.uvelsurfobs, state.vvelsurfobs], axis=-1)

        velsurfobs = tf.cast(velsurfobs, dtype)
        velsurfobs_thr = tf.constant(cfg.processes.SR_DA.fitting.velsurfobs_thr, dtype=dtype)
        velsurfobs_std = tf.constant(cfg.processes.SR_DA.fitting.velsurfobs_std, dtype=dtype)

        REL = tf.expand_dims((tf.norm(velsurfobs, axis=-1) >= velsurfobs_thr), axis=-1)
        ACT = ~tf.math.is_nan(velsurfobs)

        cost1 = 0.5 * tf.reduce_mean(
            ((velsurfobs[ACT & REL] - velsurf[ACT & REL]) / velsurfobs_std) ** 2
        )

        current_thk = inputs[0, :, :, 0]
        lam = tf.cast(cfg.processes.SR_DA.regularization.thk, dtype)
        dx = tf.cast(state.dX, dtype)
        REGU_H2 = smoothness_biharmonic(current_thk, dx, lam)

        cost = tf.cast(cost1, dtype) + tf.cast(REGU_H2, dtype)
        return cost, tf.cast(cost1, dtype), tf.cast(REGU_H2, dtype)

    return cost_function

def _get_thk_spec(cfg):
    """
    Reads the thickness variable spec from:
      cfg.processes.SR_DA.variables = [
        { name: thk, transform: identity|log10, lower_bound: 0.0, upper_bound: 1000.0 },
        ...
      ]
    Returns: (transform: str, lower: float, upper: float)
    """
    vars_cfg = cfg.processes.SR_DA.variables

    def _get(obj, key, default=None):
        if hasattr(obj, "get"):
            return obj.get(key, default)
        return getattr(obj, key, default)

    for v in vars_cfg:
        name = _get(v, "name", None)
        if str(name) != "thk":
            continue

        transform = str(_get(v, "transform", "identity")).lower()
        if transform in ("none", "identity", ""):
            transform = "identity"
        if transform not in ("identity", "log10"):
            raise ValueError(f"Unsupported thk transform: {transform}")

        lower = float(_get(v, "lower_bound", None))
        upper = float(_get(v, "upper_bound", None))
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("thk lower_bound/upper_bound must be finite.")
        if upper <= 0.0:
            raise ValueError("thk upper_bound must be > 0 for both identity and log10 transforms.")
        if upper <= lower:
            raise ValueError("thk upper_bound must be > lower_bound.")

        return transform, lower, upper

    raise ValueError("Could not find SR_DA.variables entry with name: thk")



def data_assimilation_initialize(cfg, state):
    cfg_da = cfg.processes.SR_DA
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    da = DataAssimilation()
    da.cost_fn = get_cost_fn_data(cfg, state)

    transform, thk_min, thk_max = _get_thk_spec(cfg)
    da.transform = transform
    da.thk_min = thk_min
    da.thk_max = thk_max


    thk = initial_thickness(
        s=state.usurf,
        u=state.uvelsurfobs,
        v=state.vvelsurfobs,
        mask=state.icemask,
        dx=state.dX[0,0],
        dy=state.dX[0,0],
    )

    da.thk_var = tf.Variable(thk, trainable=True, dtype=dtype)
    state.thk = da.thk_var  # downstream sees the TF variable
    state.usurf = tf.cast(state.usurf, dtype)

    # initialize zero state.dJdthk as tf tensor
    state.dJdthk = tf.zeros_like(state.thk, dtype=dtype)
    da.maxiter = int(getattr(getattr(cfg_da, "optimization", {}), "maxiter", 10000))
    da.out_freq = int(getattr(getattr(cfg_da, "optimization", {}), "output_freq", 50))

    da.cost_components = {}
    da.result = None

    state.data_assimilation = da



def initialize(cfg, state):
    data_assimilation_initialize(cfg, state)


def update(cfg, state):
    da = state.data_assimilation
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
    Ny, Nx = state.thk.shape

    update_ncdf_optimize(cfg, state, 0)

    # Bounds are in physical thickness units per your config
    thk_min = float(da.thk_min)
    thk_max = float(da.thk_max)

    # log10 cannot represent thk=0 exactly
    eps = 1e-12

    thk0_phys = np.asarray(da.thk_var.numpy())
    thk0_phys = np.clip(thk0_phys, thk_min, thk_max)

    if da.transform == "log10":
        # optimize z = log10(thk) but bounds defined in physical units
        thk0_safe = np.clip(thk0_phys, max(thk_min, eps), thk_max)
        x0 = np.log10(thk0_safe).reshape(-1).astype(np.float64)

        L = np.full_like(x0, np.log10(max(thk_min, eps)), dtype=np.float64)
        U = np.full_like(x0, np.log10(thk_max), dtype=np.float64)
        bounds = Bounds(L, U)
    else:
        # identity: optimize thk directly in physical space
        x0 = thk0_phys.reshape(-1).astype(np.float64)
        L = np.full_like(x0, thk_min, dtype=np.float64)
        U = np.full_like(x0, thk_max, dtype=np.float64)
        bounds = Bounds(L, U)


    it = {"k": 0}

    def fun_and_jac(x_flat):
        # Map SciPy vector -> thickness field (TF tensor)
        x_flat = np.asarray(x_flat, dtype=np.float64)

        if da.transform == "log10":
            z = tf.convert_to_tensor(x_flat.reshape(Ny, Nx), dtype=dtype)
            thk = tf.pow(tf.cast(10.0, dtype), z)
            # Clip to physical bounds (note: if lower=0, this will allow 0, but thk from pow won't hit 0)
            thk = tf.clip_by_value(thk, thk_min, thk_max)
        else:
            thk = tf.convert_to_tensor(x_flat.reshape(Ny, Nx), dtype=dtype)
            thk = tf.clip_by_value(thk, thk_min, thk_max)

        da.thk_var.assign(thk)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(da.thk_var)

            X = fieldin_state_to_X(cfg, state)
            inputs = state.iceflow.patching.generate_patches(X)
            if inputs.shape.rank == 3:
                inputs = inputs[None, ...]

            U, V = state.iceflow.optimizer.map.get_UV(inputs)
            cost_total, cost_data, cost_reg = da.cost_fn(U, V, inputs)

        # Gradient used for optimization (keep this as total unless you want to optimize misfit-only)
        grad_thk_total = tape.gradient(cost_total, da.thk_var)

        # Gradient you want to SAVE (misfit-only)
        grad_thk_misfit = tape.gradient(cost_data, da.thk_var)

        del tape

        if grad_thk_total is None:
            raise RuntimeError("Total gradient w.r.t thickness is None.")
        if grad_thk_misfit is None:
            # This can happen if misfit has zero valid points in ACT&REL (or other disconnect)
            grad_thk_misfit = tf.zeros_like(da.thk_var)

        # Store for output (physical dJ_misfit/dthk)
        da.latest_grad_thk_misfit = tf.stop_gradient(tf.identity(grad_thk_misfit))

        # If optimizing in z=log10(thk), TF already applied chain rule through thk=10**z.
        # But note: grad_thk here is dJ/d(thk_var). We need dJ/dx (x is z or thk).
        if da.transform == "log10":
            grad_x = grad_thk_total * tf.cast(np.log(10.0), dtype) * da.thk_var
        else:
            grad_x = grad_thk_total

        da.cost_components = {
            "total": float(cost_total.numpy()),
            "data": float(cost_data.numpy()),
            "reg": float(cost_reg.numpy()),
        }

        f = float(cost_total.numpy())
        g = grad_x.numpy().reshape(-1).astype(np.float64)
        return f, g

    def callback(xk):
        it["k"] += 1
        if it["k"] % da.out_freq != 0:
            return
        # Re-evaluate at the accepted iterate xk so the stored gradient matches it.
        # This is extra work but avoids saving a line-search trial gradient.
        _f, _g = fun_and_jac(xk)  # updates da.thk_var and da.latest_grad_thk

        # Attach to state so update_ncdf_optimize can write it
        state.dJdthk = da.latest_grad_thk_misfit
        # Update state.thk already held in da.thk_var; just re-evaluate outputs
        evaluate_iceflow(cfg, state)
        update_ncdf_optimize(cfg, state, it["k"])

        # Optional: quick print
        cc = da.cost_components
        print(f"[SciPy] iter={it['k']}  total={cc.get('total', np.nan):.6e}  "
              f"data={cc.get('data', np.nan):.6e}  reg={cc.get('reg', np.nan):.6e}")

    print(f"[SciPy] about to minimize: transform={da.transform}  n={x0.size} "
          f"x0[min,max]=({np.min(x0):.3e},{np.max(x0):.3e})", flush=True)

    # Force one explicit evaluation BEFORE minimize (this catches NaNs / None grads / non-reachability)
    try:
        f0, g0 = fun_and_jac(x0)
        g0 = np.asarray(g0)
        print(f"[SciPy] f0={f0:.6e}  "
              f"g0[inf-norm]={np.linalg.norm(g0, ord=np.inf):.6e}  "
              f"finite(f0)={np.isfinite(f0)}  finite(g0)={np.isfinite(g0).all()}",
              flush=True)
        if not np.isfinite(f0) or not np.isfinite(g0).all():
            print("[SciPy] WARNING: objective or gradient not finite at x0 -> optimizer may exit immediately.",
                  flush=True)
    except Exception as e:
        print("[SciPy] ERROR during initial fun_and_jac(x0):", repr(e), flush=True)
        raise
    
    eps = np.finfo(float).eps  # ~2.22e-16
    try:
        res = minimize(
            fun_and_jac,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                # Much stricter than default (~1e7*eps ≈ 2e-9)
                "ftol": 10 * eps,     # corresponds to factr ~ 10 (very tight)
                "gtol": 1e-12,        # projected gradient tolerance
                "maxiter": da.maxiter,
                "maxfun": 200000,     # sometimes needed if line searches are expensive
                "maxls": 40,
                "iprint": 1,          # more solver chatter
            },
        )
    except Exception as e:
        print("[SciPy] minimize crashed:", repr(e), flush=True)
        raise

    print("[SciPy] result:",
          f"success={res.success} status={res.status} nit={res.nit} nfev={res.nfev} njev={getattr(res,'njev',None)}",
          flush=True)
    print("[SciPy] message:", res.message, flush=True)


    da.result = res

    # Ensure final state is consistent and write last output
    evaluate_iceflow(cfg, state)
    update_ncdf_optimize(cfg, state, it["k"] + 1)


def finalize(cfg, state):
    pass
