#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), see LICENSE

from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

from .mapping import Mapping
from .transforms import TRANSFORMS, ParameterTransform
from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV


@dataclass
class VariableSpec:
    """
    Which state field we invert and which parameterization we use (in PHYSICAL space).
    Bounds are specified in PHYSICAL space, e.g., meters or Pa·s^n.

    If ``mask`` is provided it must resolve to a tensor in ``state`` with the same
    shape as the target field. Only entries where ``mask`` is ``True`` (or non-zero)
    are exposed to the optimizer; the complement keeps its initial values.
    """

    name: str                         # e.g. "thk", "slidingco"
    transform: str = "identity"       # key in TRANSFORMS (e.g., "identity", "log10", ...) [default + case-insensitive]
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    mask: Optional[str] = None        # dotted path on ``state`` resolving to a tensor mask


class MappingDataAssimilation(Mapping):
    """
    Exposes selected state fields as trainable θ, converts θ→physical via a registered
    ``ParameterTransform`` and runs the shared neural network emulator directly.

    - Bounds are given in PHYSICAL space and converted once into θ-space.
    - ``apply_theta_to_inputs`` patches selected channels of the BHWC inputs on-the-fly.
    """

    def __init__(
        self,
        bcs: List[str],
        network: tf.keras.Model,
        Nz: tf.Tensor,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        output_scale,
        state,
        variables: List[VariableSpec],
        eps: float = 1e-12,
        field_to_channel: Optional[Dict[str, int]] = None,   
        precision: str = "single",           
    ):
        super().__init__(bcs, precision)
        if not variables:
            raise ValueError("❌ DataAssimilation mapping requires at least one variable.")

        self.network = network
        self.Nz = Nz
        self.output_scale = tf.cast(output_scale, self.precision)
        self.vars: List[VariableSpec] = variables
        self.eps = eps
        self.cost_fn = cost_fn

        for v in self.network.trainable_variables:
            if v.dtype != self.precision:
                raise TypeError(
                    f"[DataAssimilation] Network variable dtype is {v.dtype.name}, "
                    f"but requested precision is {self.precision.name}. "
                    "Please build/load the network in the same precision."
                )

        self.field_to_channel: Dict[str, int] = field_to_channel or {
            'thk': 0, 'usurf': 1, 'arrhenius': 2, 'slidingco': 3, 'dX': 4
        }

        self.base_cost_reference = tf.Variable(float("nan"), trainable=False, name="base_cost_reference", dtype=self.precision)
        self._base_cost_tol = tf.constant(0.1, dtype=self.precision)  # 10% threshold
        self._base_cost_eps = tf.constant(1e-12, dtype=self.precision)  # Small epsilon for division

        # Diagnostics used by the halt criterion
        self.base_cost = tf.Variable(0.0, trainable=False, name="base_cost", dtype=self.precision)

        # Ensure state fields are tf.Variable and keep references (for initialization parity).
        self._field_refs: Dict[str, tf.Variable] = {}
        for spec in self.vars:
            field_val = getattr(state, spec.name)
            if isinstance(field_val, tf.Variable):
                self._field_refs[spec.name] = field_val
            else:
                field_var = tf.Variable(field_val, trainable=False, name=f"ref_{spec.name}")
                setattr(state, spec.name, field_var)
                self._field_refs[spec.name] = field_var

        # Build transform objects, initialize θ from physical fields, record shapes/sizes.
        self.transforms: List[ParameterTransform] = []
        self._w: List[tf.Variable] = []
        self._shapes: List[tf.TensorShape] = []
        self._sizes: List[tf.Tensor] = []    
        self._sizes_int: List[Optional[int]] = []   
        self._full_shapes: List[tf.TensorShape] = []
        self._mask_bool: List[Optional[tf.Tensor]] = []
        self._mask_indices: List[Optional[tf.Tensor]] = []
        self._mask_backgrounds: List[Optional[tf.Tensor]] = []

        for spec in self.vars:
            tname = (spec.transform or "identity").lower()
            if tname not in TRANSFORMS:
                raise ValueError(f"❌ Unknown transform '{spec.transform}' for '{spec.name}'.")
            tform = TRANSFORMS[tname]()  # instance
            self.transforms.append(tform)

            x0_var = self._field_refs[spec.name]
            x0 = tf.convert_to_tensor(x0_var)

            mask_bool = None
            mask_indices = None
            mask_background = None
            if spec.mask is not None:
                mask_tensor = self._resolve_mask(state, spec.mask)
                mask_bool = tf.cast(mask_tensor, tf.bool)
                if mask_bool.shape != x0.shape:
                    raise ValueError(
                        f"❌ Mask '{spec.mask}' shape {mask_bool.shape} does not match field '{spec.name}' shape {x0.shape}."
                    )
                if not bool(tf.reduce_any(mask_bool)):
                    raise ValueError(f"❌ Mask '{spec.mask}' for '{spec.name}' has no active elements.")
                mask_indices = tf.where(mask_bool)
                mask_background = tf.where(mask_bool, tf.zeros_like(x0), x0)

            theta0_full = tform.to_theta(x0_var, eps=self.eps)
            self._full_shapes.append(theta0_full.shape)

            if mask_bool is not None:
                theta0 = tf.boolean_mask(theta0_full, mask_bool)
            else:
                theta0 = theta0_full

            # Keep θ in compute precision
            theta = tf.Variable(tf.cast(theta0, self.precision), trainable=True, name=f"theta_{spec.name}")
            self._w.append(theta)
            self._shapes.append(theta.shape)
            self._sizes.append(tf.size(theta))
            self._sizes_int.append(theta.shape.num_elements() if theta.shape.num_elements() is not None else None)
            self._mask_bool.append(mask_bool)
            self._mask_indices.append(mask_indices)
            self._mask_backgrounds.append(mask_background)

        # Precompute θ-space bounds for optimizer consumption.
        self._L_list: List[tf.Tensor] = []
        self._U_list: List[tf.Tensor] = []
        for spec, theta, tform in zip(self.vars, self._w, self.transforms):
            Ls, Us = tform.theta_bounds(spec.lower_bound, spec.upper_bound, dtype=theta.dtype, eps=self.eps)
            self._L_list.append(tf.fill(theta.shape, Ls))
            self._U_list.append(tf.fill(theta.shape, Us))

    # ------- Forward plumbing -------------------------------------------------

    @staticmethod
    def _resolve_mask(state, path: str) -> tf.Tensor:
        obj = state
        for attr in path.split('.'):
            if not hasattr(obj, attr):
                raise ValueError(f"❌ Mask path '{path}' could not be resolved on state.")
            obj = getattr(obj, attr)
        return tf.convert_to_tensor(obj)

    def _theta_to_field(self, idx: int) -> tf.Tensor:
        mask_bool = self._mask_bool[idx]
        tform = self.transforms[idx]
        theta = self._w[idx]
        full_shape = self._full_shapes[idx]

        if mask_bool is None:
            val = tform.to_physical(theta)
            val.set_shape(full_shape)
            return val

        updates = tform.to_physical(theta)
        updates = tf.reshape(updates, [-1])
        background = tf.cast(self._mask_backgrounds[idx], updates.dtype)
        indices = self._mask_indices[idx]
        field = tf.tensor_scatter_nd_update(background, indices, updates)
        field.set_shape(full_shape)
        return field

    @tf.function(reduce_retracing=True)
    def synchronize_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        updated_inputs = self.apply_theta_to_inputs(inputs)
        return updated_inputs

    @tf.function(jit_compile=True)
    def get_UV(self, inputs: tf.Tensor) -> Tuple[TV, TV]:
        processed_inputs = self.synchronize_inputs(inputs)
        self.set_inputs(processed_inputs)
        U, V = self.get_UV_impl()
        for apply_bc in self.apply_bcs:
            U, V = apply_bc(U, V)
        return U, V

    @tf.function(jit_compile=True, reduce_retracing=True)
    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        Y = self.network(self.inputs) * self.output_scale
        U, V = Y_to_UV(self.Nz, Y)

        # Diagnostics for halt criterion use the same cost function as the base mapping
        current_base_cost = self.cost_fn(U, V, self.inputs)
        self.base_cost.assign(current_base_cost)
        return U, V

    # ------- State update -----------------------------------------------------

    def update_state_fields(self, state):
        """Write current physical values back into `state` (post-optimization)."""
        for idx, spec in enumerate(self.vars):
            full_value = self._theta_to_field(idx)
            ref = self._field_refs[spec.name]
            ref.assign(tf.cast(full_value, ref.dtype))
            setattr(state, spec.name, ref)

    # ------- Bounds (θ-space) for optimizer ----------------------------------

    def get_box_bounds_flat(self) -> Tuple[tf.Tensor, tf.Tensor]:
        L_flat = tf.concat([tf.reshape(Li, [-1]) for Li in self._L_list], axis=0)
        U_flat = tf.concat([tf.reshape(Ui, [-1]) for Ui in self._U_list], axis=0)
        return L_flat, U_flat

    # ------- Parameter plumbing ----------------------------------------------

    def get_w(self) -> List[tf.Variable]:
        return self._w

    def set_w(self, w: List[tf.Tensor]) -> None:
        if len(w) != len(self._w):
            raise ValueError("❌ set_w: length mismatch.")
        for var, val in zip(self._w, w):
            var.assign(val)

    def copy_w(self, w: List[tf.Variable]) -> List[tf.Tensor]:
        return [wi.read_value() for wi in w]

    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(w_flat)

    def flatten_w(self, w: List[tf.Variable | tf.Tensor]) -> tf.Tensor:
        flats = []
        for i, wi in enumerate(w):
            if wi is None:
                raise ValueError(f"❌ None gradient for parameter: {self.vars[i].name}")
            flats.append(tf.reshape(wi, [-1]))
        return tf.concat(flats, axis=0)

    def unflatten_w(self, w_flat: tf.Tensor) -> List[tf.Tensor]:
        if all(s is not None for s in self._sizes_int):
            vals: List[tf.Tensor] = []
            idx = 0
            for s_int, shp in zip(self._sizes_int, self._shapes):
                nxt = idx + int(s_int)  # type: ignore[arg-type]
                vals.append(tf.reshape(w_flat[idx:nxt], shp))
                idx = nxt
            return vals
        else:
            splits = tf.split(w_flat, self._sizes)
            return [tf.reshape(t, s) for t, s in zip(splits, self._shapes)]

    # ------- Halt criterion  ------------------------

    def check_halt_criterion(self, iteration: int, cost: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Halt if the base (forward) cost increased by >10% relative to its value at
        the start of the current minimize call. Returns (halt_bool, message).
        """
        reference = self.base_cost_reference.read_value()

        def _set_reference() -> Tuple[tf.Tensor, tf.Tensor]:
            self.base_cost_reference.assign(self.base_cost)
            return tf.constant(False, dtype=tf.bool), tf.constant("", dtype=tf.string)

        def _check_increase() -> Tuple[tf.Tensor, tf.Tensor]:
            rel_increase = (self.base_cost - reference) / (tf.abs(reference) + self._base_cost_eps)
            should_halt = tf.greater(rel_increase, self._base_cost_tol)
            message = tf.cond(
                should_halt,
                lambda: tf.constant("Base cost increased beyond threshold (10.0%)", dtype=tf.string),
                lambda: tf.constant("", dtype=tf.string),
            )
            return should_halt, message

        return tf.cond(tf.math.is_nan(reference), _set_reference, _check_increase)


    def on_minimize_start(self, iter_max: int) -> None:
        """
        Reset the base-cost reference used by the halt criterion.
        Called eagerly at the start of each minimize call.
        """
        reset_value = tf.constant(float("nan"), dtype=self.base_cost_reference.dtype)
        self.base_cost_reference.assign(reset_value)

    # ------- Input channel patching --------------------------------

    @tf.function(reduce_retracing=True)
    def apply_theta_to_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Patch BHWC inputs with current physical-space values for selected fields.
        Channel mapping follows the configured mapping.
        """
        updated = inputs
        B, H, W, C = tf.unstack(tf.shape(inputs))
        for idx, spec in enumerate(self.vars):
            ch = self.field_to_channel.get(spec.name, None)
            if ch is None:
                continue
            val = self._theta_to_field(idx)
            val = tf.cast(val, updated.dtype)
            phys_b = tf.tile(tf.reshape(val, [1, H, W, 1]), [B, 1, 1, 1])

            left = updated[:, :, :, :ch]
            right = updated[:, :, :, ch + 1 :]
            updated = tf.concat([left, phys_b, right], axis=-1)
        return updated
