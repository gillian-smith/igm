#!/usr/bin/env python3
# Copyright ...
from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .mapping import Mapping
from .transforms import TRANSFORMS
from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV

TV = Union[tf.Tensor, tf.Variable]


@dataclass
class CombinedVariableSpec:
    """
    Parameter we invert in PHYSICAL space, with an internal θ parameterization.

    transform: one of TRANSFORMS keys (e.g., "identity", "log10", "softplus")
    bounds:    Given in PHYSICAL space; converted once to θ-space for the optimizer.
    mask:      Optional boolean mask restricting optimization to a subregion.
               Accepts either:
                 • str  -> dotted path to a tensor on `state` (e.g. state.icemask)
                 • Tensor/Variable -> directly used as the mask
               True/1 = optimized (free), False/0 = frozen (keeps background values).
    """

    name: str
    transform: str = "identity"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    mask: Optional[Union[str, TV]] = None


class MappingCombinedDataAssimilation(Mapping):
    """
    Optimize *both* the emulator network weights and DA controls with a single theta.

      theta = [ network.trainable_variables...,  θ_1, θ_2, ... ]

    θ live in transform space (via TRANSFORMS). Before the forward pass, θ are mapped
    back to PHYSICAL space via transform.to_physical(θ) and written into inputs.
    """

    def __init__(
        self,
        bcs: List[str],
        network: tf.keras.Model,
        Nz: tf.Tensor,
        output_scale: tf.Tensor,
        state: Any,
        variables: List[CombinedVariableSpec],
        field_to_channel: Optional[Dict[str, int]] = None,
        eps: float = 1e-12,
        *,
        precision: str = "single",  # kept: precision support
        store_freq: int = 0,
    ):
        super().__init__(bcs, precision)

        self.store_freq = int(store_freq)
        self._num_vars = None

        # Snapshot state (allocated in on_minimize_start)
        self._snap_idx: tf.Variable | None = None
        self._iter_var: tf.Variable | None = None  # shape [T_cap], int32
        self._fields_var: tf.Variable | None = (
            None  # shape [T_cap, V, H, W], dtype = physical
        )
        self._t_cap: int = 0

        # ----- Core model pieces -----
        self.network = network
        self.Nz = Nz
        self.output_scale = tf.cast(output_scale, self.precision)
        self.eps = float(eps)

        # Ensure network weights already match requested precision
        for v in self.network.trainable_variables:
            if v.dtype != self.precision:
                raise TypeError(
                    f"[CombinedDA] Network variable dtype is {v.dtype.name}, "
                    f"but requested precision is {self.precision.name}. "
                    "Please build/load the network in the same precision."
                )

        # Channel mapping
        self.field_to_channel: Dict[str, int] = field_to_channel

        # ----- Network weights block (first in w) -----
        self._net_vars: List[tf.Variable] = list(self.network.trainable_variables)
        self._net_shapes = [v.shape for v in self._net_vars]
        self._net_sizes = [int(s.num_elements()) for s in self._net_shapes]
        self._net_total = int(sum(self._net_sizes))

        # ----- DA θ block (second in w) -----
        self._specs: List[CombinedVariableSpec] = variables
        self._theta_vars: List[tf.Variable] = []
        self._theta_shapes: List[tf.TensorShape] = []
        self._theta_sizes: List[int] = []
        self._L_theta_list: List[tf.Tensor] = []
        self._U_theta_list: List[tf.Tensor] = []

        # Keep transform objects per variable (so we don't relookup every time)
        self._transform_objs = []

        # Mask bookkeeping (mirror masked DA mapping behavior)
        self._full_shapes: List[tf.TensorShape] = []
        self._mask_bool: List[Optional[tf.Tensor]] = []
        self._mask_indices: List[Optional[tf.Tensor]] = []
        self._mask_backgrounds: List[Optional[tf.Tensor]] = []

        for spec in variables:
            tname = spec.transform.lower()
            if tname not in TRANSFORMS:
                raise ValueError(
                    f"❌ Unknown transform '{spec.transform}' for '{spec.name}'."
                )
            T = TRANSFORMS[tname]()  # instance
            self._transform_objs.append(T)

            # Read physical value from state and cast to compute dtype
            x_phys_ref = getattr(state, spec.name)
            x_phys = tf.cast(x_phys_ref, dtype=self.precision)

            # Resolve mask (optional)
            mask_bool = None
            mask_indices = None
            mask_background = None
            if spec.mask is not None:
                mask_tensor = self._resolve_mask(state, spec.mask)
                mask_bool = tf.cast(mask_tensor, tf.bool)
                if mask_bool.shape != x_phys.shape:
                    raise ValueError(
                        f"❌ Mask for '{spec.name}' has shape {mask_bool.shape}; expected {x_phys.shape}."
                    )
                if not bool(tf.reduce_any(mask_bool)):
                    raise ValueError(
                        f"❌ Mask for '{spec.name}' has no active elements."
                    )
                mask_indices = tf.where(mask_bool)
                # Background in PHYSICAL space: keep original values where mask==False, zeros on True
                mask_background = tf.where(mask_bool, tf.zeros_like(x_phys), x_phys)

            # θ initialization from physical via transform class (full field)
            theta0_full = T.to_theta(x_phys, eps=self.eps)
            self._full_shapes.append(theta0_full.shape)

            # If masked, keep only active entries in θ; else keep full θ
            if mask_bool is not None:
                theta0 = tf.boolean_mask(theta0_full, mask_bool)
            else:
                theta0 = theta0_full

            theta_var = tf.Variable(
                theta0, trainable=True, dtype=self.precision, name=f"theta_{spec.name}"
            )
            self._theta_vars.append(theta_var)
            self._theta_shapes.append(theta_var.shape)
            self._theta_sizes.append(int(theta_var.shape.num_elements()))
            self._mask_bool.append(mask_bool)
            self._mask_indices.append(mask_indices)
            self._mask_backgrounds.append(mask_background)

            # Bounds: PHYSICAL → θ-space via transform (scalar), then broadcast to θ shape
            L_theta_scalar, U_theta_scalar = T.theta_bounds(
                spec.lower_bound, spec.upper_bound, dtype=self.precision, eps=self.eps
            )
            self._L_theta_list.append(tf.fill(theta_var.shape, L_theta_scalar))
            self._U_theta_list.append(tf.fill(theta_var.shape, U_theta_scalar))

        # Cached totals for un/flattening
        self._theta_total = int(sum(self._theta_sizes))

    # --------------------------------------------------------------------------
    # Helpers: mask resolution & θ→full-field (PHYSICAL) reconstruction
    # --------------------------------------------------------------------------
    @staticmethod
    def _resolve_mask(state: Any, mask: Union[str, TV]) -> tf.Tensor:
        if isinstance(mask, str):
            obj = state
            for attr in mask.split("."):
                if not hasattr(obj, attr):
                    raise ValueError(
                        f"❌ Mask path '{mask}' could not be resolved on state."
                    )
                obj = getattr(obj, attr)
            return tf.convert_to_tensor(obj)
        else:
            return tf.convert_to_tensor(mask)

    def _theta_to_full_physical(self, idx: int) -> tf.Tensor:
        """Map current θ (possibly masked) back to a full PHYSICAL grid."""
        T = self._transform_objs[idx]
        theta = self._theta_vars[idx]
        full_shape = self._full_shapes[idx]
        mask_bool = self._mask_bool[idx]

        if mask_bool is None:
            val = T.to_physical(theta)
            val.set_shape(full_shape)
            return val

        updates = T.to_physical(theta)  # (N_active,)
        updates = tf.reshape(updates, [-1])
        background = tf.cast(
            self._mask_backgrounds[idx], updates.dtype
        )  # PHYSICAL background
        indices = self._mask_indices[idx]
        field = tf.tensor_scatter_nd_update(background, indices, updates)
        field.set_shape(full_shape)
        return field

    # --------------------------------------------------------------------------
    # Forward & inputs synchronization
    # --------------------------------------------------------------------------

    @tf.function(jit_compile=True)
    def get_UV(self, inputs: tf.Tensor) -> Tuple[TV, TV]:
        processed_inputs = self.synchronize_inputs(inputs)
        self.set_inputs(processed_inputs)
        U, V = self.get_UV_impl()
        for apply_bc in self.apply_bcs:
            U, V = apply_bc(U, V)
        return U, V

    @tf.function(reduce_retracing=True)
    def synchronize_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.apply_theta_to_inputs(inputs)

    @tf.function(reduce_retracing=True)
    def get_UV_impl(self) -> Tuple[TV, TV]:
        Y = self.network(self.inputs) * self.output_scale
        U, V = Y_to_UV(self.Nz, Y)
        return U, V

    @tf.function(reduce_retracing=True)
    def apply_theta_to_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        updated = inputs
        B, H, W, C = tf.unstack(tf.shape(inputs))

        for i, (spec, T) in enumerate(zip(self._specs, self._transform_objs)):
            # θ → full physical via transform class (handles masked/unmasked)
            phys_full = self._theta_to_full_physical(i)

            # broadcast to batch and make BHWC1
            phys_b = tf.tile(tf.reshape(phys_full, [1, H, W, 1]), [B, 1, 1, 1])

            ch = int(self.field_to_channel[spec.name])

            # Replace channel ch with phys_b using slice + concat (graph-safe)
            left = updated[:, :, :, :ch]
            right = updated[:, :, :, ch + 1 :]
            updated = tf.concat([left, phys_b, right], axis=-1)

        return updated

    # --------------------------------------------------------------------------
    # theta management
    # --------------------------------------------------------------------------
    def _split_blocks(self, theta: List[TV]) -> Tuple[List[TV], List[TV]]:
        return list(theta[: len(self._net_vars)]), list(theta[len(self._net_vars) :])

    def get_theta(self) -> List[tf.Variable]:
        return self._net_vars + self._theta_vars

    def set_theta(self, theta: List[tf.Tensor]) -> None:
        net_vals, theta_vals = self._split_blocks(theta)
        if len(net_vals) != len(self._net_vars) or len(theta_vals) != len(
            self._theta_vars
        ):
            raise ValueError("❌ set_theta received a vector with unexpected block sizes.")
        for v, val in zip(self._net_vars, net_vals):
            v.assign(tf.cast(val, v.dtype))
        for t, val in zip(self._theta_vars, theta_vals):
            t.assign(tf.cast(val, t.dtype))

    def copy_theta(self, theta: List[tf.Variable]) -> List[tf.Tensor]:
        return [theta_i.read_value() for theta_i in theta]

    def copy_theta_flat(self, theta_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(theta_flat)

    def flatten_theta(self, theta: List[Union[tf.Variable, tf.Tensor]]) -> tf.Tensor:
        flats = [tf.reshape(tf.cast(theta_i, self.precision), [-1]) for theta_i in theta]
        return tf.concat(flats, axis=0)

    def unflatten_theta(self, theta_flat: tf.Tensor) -> List[tf.Tensor]:
        # network block
        idx = 0
        net_vals: List[tf.Tensor] = []
        for s, shp, var in zip(self._net_sizes, self._net_shapes, self._net_vars):
            split = tf.reshape(theta_flat[idx : idx + s], shp)
            net_vals.append(tf.cast(split, var.dtype))
            idx += s
        # theta block
        theta_vals: List[tf.Tensor] = []
        for s, shp, var in zip(self._theta_sizes, self._theta_shapes, self._theta_vars):
            split = tf.reshape(theta_flat[idx : idx + s], shp)
            theta_vals.append(tf.cast(split, var.dtype))
            idx += s
        return net_vals + theta_vals

    # --------------------------------------------------------------------------
    # Bounds (θ-space only; network weights unbounded)
    # --------------------------------------------------------------------------
    def get_box_bounds_flat(self) -> Tuple[tf.Tensor, tf.Tensor]:
        if self._net_total > 0:
            neg_inf = tf.constant(-float("inf"), self.precision)
            pos_inf = tf.constant(float("inf"), self.precision)
            net_L = tf.fill([self._net_total], neg_inf)
            net_U = tf.fill([self._net_total], pos_inf)
        else:
            net_L = net_U = tf.zeros([0], dtype=self.precision)

        if self._theta_vars:
            L_th = tf.concat(
                [tf.reshape(Li, [-1]) for Li in self._L_theta_list], axis=0
            )
            U_th = tf.concat(
                [tf.reshape(Ui, [-1]) for Ui in self._U_theta_list], axis=0
            )
            return (
                tf.concat([net_L, tf.cast(L_th, self.precision)], 0),
                tf.concat([net_U, tf.cast(U_th, self.precision)], 0),
            )
        else:
            return net_L, net_U

    # --------------------------------------------------------------------------
    # Optional helper: update State with current physical fields (post-optim)
    # --------------------------------------------------------------------------
    def update_state_fields(self, state: Any) -> None:
        for i, (spec, T) in enumerate(zip(self._specs, self._transform_objs)):
            phys_full = self._theta_to_full_physical(i)
            # Cast back to the original dtype of the state variable to maintain consistency
            original_var = getattr(state, spec.name)
            phys_full_cast = tf.cast(phys_full, original_var.dtype)
            setattr(state, spec.name, phys_full_cast)

    # --------------------------------------------------------------------------
    # Progress storage hooks
    # --------------------------------------------------------------------------

    def on_minimize_start(self, iter_max: int) -> None:
        # Called eagerly. Prepare fixed-size Variables for snapshots.
        self._num_vars = len(self._specs)

        # Infer (H, W) and dtype from a sample physical field
        sample = self._theta_to_full_physical(0)  # (H, W)
        phys_dtype = sample.dtype
        H = (
            int(sample.shape[0])
            if sample.shape[0] is not None
            else int(tf.shape(sample)[0])
        )
        W = (
            int(sample.shape[1])
            if sample.shape[1] is not None
            else int(tf.shape(sample)[1])
        )

        # Compute capacity (no snapshots if disabled)
        if self.store_freq <= 0 or not iter_max or iter_max <= 0:
            self._t_cap = 0
        else:
            self._t_cap = int(iter_max // self.store_freq + 2)

        # Allocate / reset Variables
        self._snap_idx = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="snap_idx"
        )

        if self._t_cap > 0:
            self._iter_var = tf.Variable(
                tf.zeros([self._t_cap], dtype=tf.int32),
                trainable=False,
                name="iter_snapshots",
            )
            self._fields_var = tf.Variable(
                tf.zeros([self._t_cap, self._num_vars, H, W], dtype=phys_dtype),
                trainable=False,
                name="field_snapshots",
            )
        else:
            # Keep placeholders (won’t be used)
            self._iter_var = None
            self._fields_var = None

    @tf.function(experimental_relax_shapes=True)
    def on_step_end(self, iteration: tf.Tensor) -> None:
        sf = tf.constant(self.store_freq, tf.int32)

        def _append():
            idx = self._snap_idx.read_value()

            # Build (V, H, W) stack in TF
            phys_list = [self._theta_to_full_physical(i) for i in range(self._num_vars)]
            stacked = tf.stack(phys_list, axis=0)  # (V, H, W)

            # Scatter into Variables at index idx
            self._iter_var.scatter_nd_update(
                indices=tf.reshape(idx, [1, 1]), updates=tf.reshape(iteration, [1])
            )
            self._fields_var.scatter_nd_update(
                indices=tf.reshape(idx, [1, 1]), updates=tf.expand_dims(stacked, axis=0)
            )
            # Bump counter
            self._snap_idx.assign_add(1)
            return tf.constant(0, tf.int32)

        # Guard everything when disabled or capacity==0
        def _enabled_and_capacity():
            return tf.cond(
                tf.greater(sf, 0),
                lambda: tf.cond(
                    tf.equal(tf.math.floormod(iteration, sf), 0),
                    _append,
                    lambda: tf.constant(0, tf.int32),
                ),
                lambda: tf.constant(0, tf.int32),
            )

        # If you want to also guard on capacity at graph-time:
        if self._t_cap == 0 or self._iter_var is None or self._fields_var is None:
            return
        _ = _enabled_and_capacity()

    def get_progress(self) -> dict:
        # Eager: materialize after optimize()
        if (
            self._t_cap == 0
            or self._snap_idx is None
            or self._iter_var is None
            or self._fields_var is None
        ):
            return {"iteration": [], "fields": {}}

        T = int(self._snap_idx.numpy())
        if T == 0:
            return {"iteration": [], "fields": {spec.name: [] for spec in self._specs}}

        it = self._iter_var.numpy()[:T]  # (T,)
        F = self._fields_var.numpy()[:T, :, :, :]  # (T, V, H, W)
        fields = {spec.name: F[:, k, :, :] for k, spec in enumerate(self._specs)}
        return {"iteration": it, "fields": fields}
