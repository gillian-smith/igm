#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), see LICENSE

from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .mapping import Mapping



@dataclass
class VariableSpec:
    """Which state field we invert and which parameterization we use."""
    name: str           # e.g. "thk", "slidingco"
    transform: str      # "identity" or "log10"


class MappingDataAssimilation(Mapping):
    """
    Parameterizes selected state fields as trainables (w), projects them back to
    physical fields before the forward pass, and returns U,V by delegating to
    an existing mapping. This ensures consistency with the main iceflow mapping.

    Constructor args (typically provided by InterfaceDataAssimilation):
      - bcs: boundary-condition names (same as other mappings)
      - base_mapping: existing Mapping object to delegate forward evaluation to
      - state: State object used only for initialization (not stored)
      - variables: list[VariableSpec] parsed from Hydra
      - eps: small positive to avoid log(0) in log10 parameterization

    Notes:
      * transform == "identity"  -> train θ = field, feed field = θ
      * transform == "log10"     -> train θ = log10(field), feed field = 10**θ
        (positivity enforced; chain rule handled automatically by TF)
    """

    def __init__(
        self,
        bcs: List[str],
        base_mapping,
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
        
        # Store field references as Variables for direct updates (lightweight approach)
        self._field_refs: Dict[str, tf.Variable] = {}
        for spec in self.vars:
            field_val = getattr(state, spec.name)
            # Always convert to Variable 
            if isinstance(field_val, tf.Variable):
                self._field_refs[spec.name] = field_val  # It's already a Variable
            else:
                # Convert to Variable if it's not already
                field_var = tf.Variable(field_val, trainable=False, name=f"ref_{spec.name}")
                setattr(state, spec.name, field_var)  # Replace field with Variable
                self._field_refs[spec.name] = field_var

        # Create trainable parameters θ (one per selected field)
        self._w: List[tf.Variable] = []
        self._shapes: List[tf.TensorShape] = []
        self._sizes: List[tf.Tensor] = []

        for spec in self.vars:
            field_val = self._field_refs[spec.name]

            if spec.transform == "log10":
                LN10 = tf.constant(2.302585092994046, tf.float32)

                # θ0 = log10(max(x0, eps))
                theta0 = tf.math.log(tf.maximum(field_val, self.eps)) / LN10
                theta = tf.Variable(theta0, trainable=True, name=f"theta_{spec.name}")
                self._w.append(theta)
                self._shapes.append(theta.shape)
                self._sizes.append(tf.size(theta))
            elif spec.transform == "identity":
                theta = tf.Variable(field_val, trainable=True, name=f"{spec.name}")
                self._w.append(theta)
                self._shapes.append(theta.shape)
                self._sizes.append(tf.size(theta))
            else:
                raise ValueError(f"❌ Unknown transform '{spec.transform}' for '{spec.name}'.")

    # ------- Mapping API: U,V from current parameters -------------------------

    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # Map θ -> physical fields via direct Variable assignment, then delegate to base mapping
        
        # Update field Variables with current parameter values
        # for spec, theta in zip(self.vars, self._w):
        #     field_ref = self._field_refs[spec.name]
        #     if spec.transform == "log10":
        #         LN10 = tf.constant(2.302585092994046, tf.float32)

        #         val = tf.exp(LN10 * theta)  # 10**θ
        #     else:  # identity
        #         val = theta
        #     field_ref.assign(val)

        updated_inputs = self.inputs
        
        # Define mapping from field names to input channel indices
        field_to_channel = {
            'thk': 0,
            'usurf': 1, 
            'arrhenius': 2,
            'slidingco': 3,
            'dX': 4,
            # Add other fields as needed
        }
        
        # Update each parameter in the input tensor
        for spec, theta in zip(self.vars, self._w):
            if spec.name in field_to_channel:
                # Transform parameter to physical value
                if spec.transform == "log10":
                    LN10 = tf.constant(2.302585092994046, tf.float32)
                    val = tf.exp(LN10 * theta)  # 10**θ
                else:  # identity
                    val = theta
                
                # Add batch dimension to match input tensor shape
                # updated_inputs has shape [batch, height, width, channels]
                # val has shape [height, width], need to make it [batch, height, width, 1]
                val_with_batch = tf.expand_dims(tf.expand_dims(val, axis=0), axis=-1)  # [1, height, width, 1]
                
                # Get the channel index for this field
                channel_idx = field_to_channel[spec.name]
                
                # Replace the channel in the input tensor
                channel_updates = []
                for i in range(updated_inputs.shape[-1]):
                    if i == channel_idx:
                        channel_updates.append(val_with_batch)
                    else:
                        channel_updates.append(updated_inputs[:, :, :, i:i+1])
                
                updated_inputs = tf.concat(channel_updates, axis=-1)
            
        # Set inputs on base mapping and delegate forward evaluation
        self.base_mapping.set_inputs(updated_inputs)
        U, V = self.base_mapping.get_UV_impl()
        return U, V

    def update_state_fields(self, state):
        """Update state fields with current optimized parameter values."""
    
        for i, var_spec in enumerate(self.vars): 
            field_name = var_spec.name
            theta = self._w[i]  # Use the stored variable directly
            
            # Transform back to physical value
            if var_spec.transform == "log10":
                LN10 = tf.constant(2.302585092994046, tf.float32)
                physical_value = tf.exp(LN10 * theta)
            else:  # identity
                physical_value = theta
            
            # Update the state field
            setattr(state, field_name, physical_value)

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
            if wi is None: # this is meant to catch and warn about None gradients, which would indicate that something is wrong
                param_name = self.vars[i].name if i < len(self.vars) else f"param_{i}"
                print(f"⚠️  None gradient for parameter: {param_name}")
                zero_grad = tf.zeros(self._shapes[i], dtype=tf.float32)
                flats.append(tf.reshape(zero_grad, [-1]))
            else:
                flats.append(tf.reshape(wi, [-1]))
        return tf.concat(flats, axis=0)

    def unflatten_w(self, w_flat: tf.Tensor) -> List[tf.Tensor]:
        splits = tf.split(w_flat, self._sizes)
        return [tf.reshape(t, s) for t, s in zip(splits, self._shapes)]
