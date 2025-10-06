#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), see LICENSE

from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

from .mapping import Mapping


@dataclass
class VariableSpec:
    """Which state field we invert and which parameterization we use (in PHYSICAL space)."""
    name: str                         # e.g. "thk", "slidingco"
    transform: str                    # "identity" or "log10"
    lower_bound: Optional[float] = None  # lower bound in PHYSICAL space (None = no bound)
    upper_bound: Optional[float] = None  # upper bound in PHYSICAL space (None = no bound)


class MappingDataAssimilation(Mapping):
    """
    Parameterizes selected state fields as trainables (θ), projects them back to
    physical fields before the forward pass, and delegates U,V to an existing mapping.

    Bounds:
      - Bounds are specified in PHYSICAL space in VariableSpec (e.g., thickness in meters).
      - This class converts those bounds once to θ-space (the space actually optimized).
      - If a bound is None, it becomes ±inf in θ-space (i.e., no constraint for that side).
      - For 'log10' transform: θ = log10(x). If lower_bound <= 0 or None -> θ-L = -inf.
        If an upper_bound <= 0 is provided, it is invalid and will raise an error.

    Notes:
      * transform == "identity"  -> train θ = field,        feed field = θ
      * transform == "log10"     -> train θ = log10(field), feed field = 10**θ
        (positivity enforced; chain rule handled by TF)
    """

    def __init__(
        self,
        bcs: List[str],
        base_mapping,
        base_cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        state,
        variables: List[VariableSpec],
        eps: float = 1e-12,
    ):
        super().__init__(bcs)
        if not variables:
            raise ValueError("❌ DataAssimilation mapping requires at least one variable.")

        self.base_mapping = base_mapping
        self.vars: List[VariableSpec] = variables
        self.eps = eps
        self.base_cost_fn = base_cost_fn

        # Diagnostics (unchanged behavior)
        self.base_cost = tf.Variable(0.0, trainable=False, name="base_cost")
        self.initial_cost = tf.Variable(0.0, trainable=False, name="initial_cost")
        self.cost_initialized = tf.Variable(False, trainable=False, name="cost_initialized")

        # Store field references as Variables for direct updates (kept for consistency)
        self._field_refs: Dict[str, tf.Variable] = {}
        for spec in self.vars:
            field_val = getattr(state, spec.name)
            if isinstance(field_val, tf.Variable):
                self._field_refs[spec.name] = field_val
            else:
                field_var = tf.Variable(field_val, trainable=False, name=f"ref_{spec.name}")
                setattr(state, spec.name, field_var)  # keep State in sync with a Variable
                self._field_refs[spec.name] = field_var

        # Create trainable parameters θ (one per selected field), track shapes/sizes
        self._w: List[tf.Variable] = []
        self._shapes: List[tf.TensorShape] = []
        self._sizes: List[tf.Tensor] = []

        for spec in self.vars:
            field_val = self._field_refs[spec.name]
            if spec.transform == "log10":
                # θ0 = log10(max(x0, eps))
                ln10 = tf.constant(2.302585092994046, field_val.dtype)
                theta0 = tf.math.log(tf.maximum(field_val, tf.cast(self.eps, field_val.dtype))) / ln10
                theta = tf.Variable(theta0, trainable=True, name=f"theta_{spec.name}")
            elif spec.transform == "identity":
                theta = tf.Variable(field_val, trainable=True, name=f"{spec.name}")
            else:
                raise ValueError(f"❌ Unknown transform '{spec.transform}' for '{spec.name}'.")

            self._w.append(theta)
            self._shapes.append(theta.shape)
            self._sizes.append(tf.size(theta))

        # Precompute θ-space bounds for all parameters (for optimizer use)
        self._L_list: List[tf.Tensor] = []
        self._U_list: List[tf.Tensor] = []

        for spec, theta in zip(self.vars, self._w):
            dtype = theta.dtype

            L_phys = spec.lower_bound
            U_phys = spec.upper_bound

            # Convert to θ-space scalars
            if spec.transform == "identity":
                L_theta_scalar = -tf.constant(float("inf"), dtype) if L_phys is None else tf.constant(L_phys, dtype)
                U_theta_scalar =  tf.constant(float("inf"), dtype) if U_phys is None else tf.constant(U_phys, dtype)
            elif spec.transform == "log10":
                ln10 = tf.constant(2.302585092994046, dtype)
                # Lower bound:
                #   - None or <=0  -> -inf (no lower bound in θ; exp ensures positivity anyway)
                #   - >0           -> log10(L)
                if (L_phys is None) or (L_phys <= 0.0):
                    L_theta_scalar = -tf.constant(float("inf"), dtype)
                else:
                    L_theta_scalar = tf.math.log(tf.constant(L_phys, dtype)) / ln10
                # Upper bound:
                #   - None         -> +inf
                #   - <=0          -> invalid (cannot take log), raise
                #   - >0           -> log10(U)
                if U_phys is None:
                    U_theta_scalar = tf.constant(float("inf"), dtype)
                else:
                    if U_phys <= 0.0:
                        raise ValueError(
                            f"❌ Invalid upper_bound ({U_phys}) for log10 transform of '{spec.name}': must be > 0."
                        )
                    U_theta_scalar = tf.math.log(tf.constant(U_phys, dtype)) / ln10
            else:
                # guarded above, but keep static analyzer happy
                raise ValueError(f"❌ Unknown transform '{spec.transform}' for '{spec.name}'.")

            # Broadcast per-variable scalars to the variable's shape (so later we can flatten & concat)
            self._L_list.append(tf.fill(theta.shape, L_theta_scalar))
            self._U_list.append(tf.fill(theta.shape, U_theta_scalar))

    # ------- Mapping API: U,V from current parameters -------------------------

    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Map current θ → physical fields inside the input tensor (channel-wise), then
        delegate to the base mapping's forward (get_UV_impl). No behavior change when
        bounds are None; bounds are not enforced here (optimizer handles them).
        """
        updated_inputs = self.inputs

        # Map from field name to channel index in the input tensor
        field_to_channel = {
            'thk': 0,
            'usurf': 1,
            'arrhenius': 2,
            'slidingco': 3,
            'dX': 4,
            # extend as needed
        }

        for spec, theta in zip(self.vars, self._w):
            if spec.name in field_to_channel:
                if spec.transform == "log10":
                    ln10 = tf.constant(2.302585092994046, theta.dtype)
                    val = tf.exp(ln10 * theta)  # 10**θ
                else:
                    val = theta

                # Expand to [batch, H, W, 1] and replace the appropriate channel
                val_bhwc1 = tf.expand_dims(tf.expand_dims(val, axis=0), axis=-1)
                ch = field_to_channel[spec.name]

                channel_updates = []
                C = updated_inputs.shape[-1]
                for i in range(C):
                    if i == ch:
                        channel_updates.append(val_bhwc1)
                    else:
                        channel_updates.append(updated_inputs[:, :, :, i:i+1])
                updated_inputs = tf.concat(channel_updates, axis=-1)

        # Keep base & self inputs in sync and delegate to base mapping's forward core
        self.base_mapping.set_inputs(updated_inputs)
        self.set_inputs(updated_inputs)
        U, V = self.base_mapping.get_UV_impl()

        # For diagnostics
        current_base_cost = self.base_cost_fn(U, V, updated_inputs)
        self.base_cost.assign(current_base_cost)
        return U, V

    def update_state_fields(self, state):
        """Update state fields with current optimized parameter values (physical space)."""
        for i, var_spec in enumerate(self.vars):
            theta = self._w[i]
            if var_spec.transform == "log10":
                ln10 = tf.constant(2.302585092994046, theta.dtype)
                physical_value = tf.exp(ln10 * theta)
            else:
                physical_value = theta
            setattr(state, var_spec.name, physical_value)

    # ------- Bounds (θ-space) accessors for the optimizer ---------------------

    def get_box_bounds_flat(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return flattened lower/upper bounds in θ-space for all parameters in self._w.
        Bounds are ±inf where the user omitted them in the YAML.
        This is a passive accessor: if the optimizer doesn't use it, nothing changes.
        """
        L_flat = tf.concat([tf.reshape(Li, [-1]) for Li in self._L_list], axis=0)
        U_flat = tf.concat([tf.reshape(Ui, [-1]) for Ui in self._U_list], axis=0)
        return L_flat, U_flat

    # ------- Mapping API: parameters (w) plumbing -----------------------------

    def get_w(self) -> List[tf.Variable]:
        return self._w

    def set_w(self, w: List[tf.Tensor]) -> None:
        if len(w) != len(self._w):
            raise ValueError("❌ set_w: length mismatch.")
        for var, val in zip(self._w, w):
            var.assign(val)

    def copy_w(self, w: List[tf.Variable]) -> List[tf.Tensor]:
        # Return immediate tensor snapshots (no aliasing)
        return [wi.read_value() for wi in w]

    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(w_flat)

    def flatten_w(self, w: List[tf.Variable | tf.Tensor]) -> tf.Tensor:
        flats = []
        for i, wi in enumerate(w):
            if wi is None:
                # Defensive: None gradient would indicate wiring issues upstream
                raise ValueError(f"❌ None gradient for parameter: {self.vars[i].name}")
            flats.append(tf.reshape(wi, [-1]))
        return tf.concat(flats, axis=0)

    def unflatten_w(self, w_flat: tf.Tensor) -> List[tf.Tensor]:
        splits = tf.split(w_flat, self._sizes)
        return [tf.reshape(t, s) for t, s in zip(splits, self._shapes)]

    # ------- Halt criterion  ------------------------------

    def check_halt_criterion(self, iteration: int, cost: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Check base cost increase halt criterion.
        Returns: (halt_boolean, halt_message_string)
        """

        def initialize_cost():
            self.initial_cost.assign(self.base_cost)
            self.cost_initialized.assign(True)
            return tf.constant(False, dtype=tf.bool), tf.constant("", dtype=tf.string)

        def check_cost_increase():
            rel_increase = (self.base_cost - self.initial_cost) / (tf.abs(self.initial_cost) + 1e-12)
            threshold = tf.constant(0.1, dtype=tf.float32)  # 10%
            should_halt = tf.greater(rel_increase, threshold)
            halt_message = tf.cond(
                should_halt,
                lambda: tf.constant("Base cost increased beyond threshold (10.0%)", dtype=tf.string),
                lambda: tf.constant("", dtype=tf.string),
            )
            return should_halt, halt_message

        return tf.cond(self.cost_initialized, check_cost_increase, initialize_cost)
