#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from igm.utils.math.precision import normalize_precision
from igm.processes.iceflow.utils.velocities import get_velsurf


@dataclass
class DAEvaluationContext:
    """
    This holds references and computed derived fields to avoid recomputing
    key quantities in every term in the DA cost. I can add various other
    derived fields in the future: in particular divflux would probably
    be useful...

    IMPORTANT: any θ-dependent physical field MUST be accessed via da_map
    (ctx.physical(...)) to guarantee gradients flow properly.
    """
    cfg: Any
    state: Any
    da_map: Any
    U_in: tf.Tensor
    V_in: tf.Tensor
    inputs: Any

    _dtype: Optional[tf.DType] = None
    _U: Optional[tf.Tensor] = None
    _V: Optional[tf.Tensor] = None
    _dx: Optional[tf.Tensor] = None

    _uvelsurf: Optional[tf.Tensor] = None
    _vvelsurf: Optional[tf.Tensor] = None

    _mask_cache: Dict[str, tf.Tensor] = None

    def __post_init__(self) -> None:
        self._mask_cache = {}

    @property
    def dtype(self) -> tf.DType:
        if self._dtype is None:
            self._dtype = normalize_precision(self.cfg.processes.iceflow.numerics.precision)
        return self._dtype

    @property
    def U(self) -> tf.Tensor:
        if self._U is None:
            self._U = self.U_in[0]
        return self._U

    @property
    def V(self) -> tf.Tensor:
        if self._V is None:
            self._V = self.V_in[0]
        return self._V

    @property
    def dx(self) -> tf.Tensor:
        if self._dx is None:
            self._dx = tf.cast(self.state.dX, self.dtype)
        return self._dx

    # ---- data access ----

    def obs(self, name: str) -> tf.Tensor:
        """Observation tensor from state by attribute name (e.g. 'uvelsurfobs')."""
        if not hasattr(self.state, name):
            raise AttributeError(f"State has no observation field '{name}'.")
        return tf.cast(getattr(self.state, name), self.dtype)

    def physical(self, name: str) -> tf.Tensor:
        """θ-dependent physical field (e.g. 'thk', 'slidingco') via mapping."""
        return self.da_map.get_physical_field(name)

    def model(self, name: str) -> tf.Tensor:
        """
        Model quantity provider.
        
        Special cases can be added at a future date as necessary e.g. divflux
        """
        if name == "uvelsurf":
            u, _ = self.velsurf()
            return u
        if name == "vvelsurf":
            _, v = self.velsurf()
            return v

        # Default: treat as physical field
        # (works for 'thk', 'slidingco', etc.)
        return self.physical(name)

    def velsurf(self) -> Tuple[tf.Tensor, tf.Tensor]:
        if self._uvelsurf is None or self._vvelsurf is None:
            u, v = get_velsurf(self.U, self.V, self.state.iceflow.discr_v.V_s)
            self._uvelsurf = u
            self._vvelsurf = v
        return self._uvelsurf, self._vvelsurf

    def get_mask(self, mask_name: Optional[str]) -> tf.Tensor:
        """
        Returns a boolean mask tensor.

        Default (mask_name is None): state.icemask
        Override (mask_name is a string): state.<mask_name> (error if missing)
        """
        if mask_name is None:
            return tf.cast(self.state.icemask, tf.bool)

        if not hasattr(self.state, mask_name):
            raise AttributeError(f"State has no mask field '{mask_name}'.")
        return tf.cast(getattr(self.state, mask_name), tf.bool)