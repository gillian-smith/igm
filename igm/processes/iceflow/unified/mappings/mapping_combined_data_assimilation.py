#!/usr/bin/env python3
# Copyright ...
from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .mapping import Mapping
from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV
from .transforms import TRANSFORMS          # NEW: use transform classes
from ..utils import _normalize_precision    # (kept from your precision-aware version)

TV = Union[tf.Tensor, tf.Variable]


@dataclass
class CombinedVariableSpec:
    """
    Parameter we invert in PHYSICAL space, with an internal θ parameterization.

    transform: one of TRANSFORMS keys (e.g., "identity", "log10", "softplus")
    bounds:    Given in PHYSICAL space; converted once to θ-space for the optimizer.
    mask:      Not supported in this simplified BHWC-only version (must be None).
    """
    name: str
    transform: str = "identity"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    mask: Optional[TV] = None


class MappingCombinedDataAssimilation(Mapping):
    """
    Optimize *both* the emulator network weights and DA controls with a single w.

      w = [ network.trainable_variables...,  θ_1, θ_2, ... ]

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
        precision: str | tf.dtypes.DType = "double",   # kept: precision support
    ):
        super().__init__(bcs)

        # ----- precision config -----
        self.compute_dtype = _normalize_precision(precision)

        # ----- Core model pieces -----
        self.network = network
        self.Nz = Nz
        self.output_scale = tf.cast(output_scale, self.compute_dtype)
        self.eps = float(eps)

        # Ensure network weights already match requested precision
        for v in self.network.trainable_variables:
            if v.dtype != self.compute_dtype:
                raise TypeError(
                    f"[CombinedDA] Network variable dtype is {v.dtype.name}, "
                    f"but requested precision is {self.compute_dtype.name}. "
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

        for spec in variables:
            tname = spec.transform.lower()
            if tname not in TRANSFORMS:
                raise ValueError(f"❌ Unknown transform '{spec.transform}' for '{spec.name}'.")
            T = TRANSFORMS[tname]()                # instance
            self._transform_objs.append(T)

            # Read physical value from state and cast to compute dtype
            x_phys = getattr(state, spec.name)
            x_phys = tf.convert_to_tensor(x_phys, dtype=self.compute_dtype)

            # θ initialization from physical via transform class
            theta0 = T.to_theta(x_phys, eps=self.eps)
            theta_var = tf.Variable(theta0, trainable=True, dtype=self.compute_dtype, name=f"theta_{spec.name}")
            self._theta_vars.append(theta_var)
            self._theta_shapes.append(theta_var.shape)
            self._theta_sizes.append(int(theta_var.shape.num_elements()))

            # Bounds: PHYSICAL → θ-space via transform
            L_theta_scalar, U_theta_scalar = T.theta_bounds(
                spec.lower_bound, spec.upper_bound, dtype=self.compute_dtype, eps=self.eps
            )
            self._L_theta_list.append(tf.fill(theta_var.shape, L_theta_scalar))
            self._U_theta_list.append(tf.fill(theta_var.shape, U_theta_scalar))

        # Masks explicitly unsupported for now
        for spec in self._specs:
            if spec.mask is not None:
                raise ValueError(
                    f"Mask provided for '{spec.name}', but masks are disabled in the "
                    "simplified combined mapping (BHWC-only, no masks)."
                )

        # Cached totals for un/flattening
        self._theta_total = int(sum(self._theta_sizes))

    # --------------------------------------------------------------------------
    # Forward & inputs synchronization
    # --------------------------------------------------------------------------
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

        for spec, theta, T in zip(self._specs, self._theta_vars, self._transform_objs):
            # θ → physical via transform class
            phys = T.to_physical(theta)

            # broadcast to batch and make BHWC1
            phys_b = tf.tile(tf.reshape(phys, [1, H, W, 1]), [B, 1, 1, 1])

            ch = int(self.field_to_channel[spec.name])

            # Replace channel ch with phys_b using slice + concat (graph-safe)
            left  = updated[:, :, :, :ch]
            right = updated[:, :, :, ch+1:]
            updated = tf.concat([left, phys_b, right], axis=-1)

        return updated

    # --------------------------------------------------------------------------
    # w management
    # --------------------------------------------------------------------------
    def _split_blocks(self, w: List[TV]) -> Tuple[List[TV], List[TV]]:
        return list(w[: len(self._net_vars)]), list(w[len(self._net_vars):])

    def get_w(self) -> List[tf.Variable]:
        return self._net_vars + self._theta_vars

    def set_w(self, w: List[tf.Tensor]) -> None:
        net_vals, theta_vals = self._split_blocks(w)
        if len(net_vals) != len(self._net_vars) or len(theta_vals) != len(self._theta_vars):
            raise ValueError("❌ set_w received a vector with unexpected block sizes.")
        for v, val in zip(self._net_vars, net_vals):
            v.assign(tf.cast(val, v.dtype))
        for t, val in zip(self._theta_vars, theta_vals):
            t.assign(tf.cast(val, t.dtype))

    def copy_w(self, w: List[tf.Variable]) -> List[tf.Tensor]:
        return [wi.read_value() for wi in w]

    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(w_flat)

    def flatten_w(self, w: List[Union[tf.Variable, tf.Tensor]]) -> tf.Tensor:
        flats = [tf.reshape(tf.cast(wi, self.compute_dtype), [-1]) for wi in w]
        return tf.concat(flats, axis=0)

    def unflatten_w(self, w_flat: tf.Tensor) -> List[tf.Tensor]:
        # network block
        idx = 0
        net_vals: List[tf.Tensor] = []
        for s, shp, var in zip(self._net_sizes, self._net_shapes, self._net_vars):
            split = tf.reshape(w_flat[idx: idx + s], shp)
            net_vals.append(tf.cast(split, var.dtype))
            idx += s
        # theta block
        theta_vals: List[tf.Tensor] = []
        for s, shp, var in zip(self._theta_sizes, self._theta_shapes, self._theta_vars):
            split = tf.reshape(w_flat[idx: idx + s], shp)
            theta_vals.append(tf.cast(split, var.dtype))
            idx += s
        return net_vals + theta_vals

    # --------------------------------------------------------------------------
    # Bounds (θ-space only; network weights unbounded)
    # --------------------------------------------------------------------------
    def get_box_bounds_flat(self) -> Tuple[tf.Tensor, tf.Tensor]:
        if self._net_total > 0:
            neg_inf = tf.constant(-float("inf"), self.compute_dtype)
            pos_inf = tf.constant(float("inf"), self.compute_dtype)
            net_L = tf.fill([self._net_total], neg_inf)
            net_U = tf.fill([self._net_total], pos_inf)
        else:
            net_L = net_U = tf.zeros([0], dtype=self.compute_dtype)

        if self._theta_vars:
            L_th = tf.concat([tf.reshape(Li, [-1]) for Li in self._L_theta_list], axis=0)
            U_th = tf.concat([tf.reshape(Ui, [-1]) for Ui in self._U_theta_list], axis=0)
            return (
                tf.concat([net_L, tf.cast(L_th, self.compute_dtype)], 0),
                tf.concat([net_U, tf.cast(U_th, self.compute_dtype)], 0),
            )
        else:
            return net_L, net_U

    # --------------------------------------------------------------------------
    # Optional helper: update State with current physical fields (post-optim)
    # --------------------------------------------------------------------------
    def update_state_fields(self, state: Any) -> None:
        for spec, theta, T in zip(self._specs, self._theta_vars, self._transform_objs):
            phys = T.to_physical(theta)
            setattr(state, spec.name, phys)

    # --------------------------------------------------------------------------
    # Halt criterion (simple default; can be extended later)
    # --------------------------------------------------------------------------
    def check_halt_criterion(self, iteration: int, cost: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.constant(False, tf.bool), tf.constant("", tf.string)
