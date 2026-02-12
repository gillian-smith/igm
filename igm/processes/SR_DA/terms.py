#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import tensorflow as tf

from .context import DAEvaluationContext
from .penalties import  PENALTY_REGISTRY
from .utils import masked_mean, masked_sum


MISFIT = "misfit"
REGULARIZATION = "regularization"


@dataclass(frozen=True)
class MisfitSpec:
    kind: str
    name: str
    components: Sequence[str]
    obs: Sequence[str]
    std: float
    mask: Optional[str] = None
    eps: float = 1e-12

@dataclass(frozen=True)
class FieldPenaltySpec:
    # field name (e.g. 'thk', 'slidingco')
    name: str
    penalty: str
    lam: float
    mask: Optional[str] = None
    eps: float = 1e-12

    
class CostTerm(ABC):
    name: str
    group: str  # MISFIT / REGULARIZATION

    @abstractmethod
    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        raise NotImplementedError

class GaussianMisfitTerm(CostTerm):
    group = MISFIT

    def __init__(self, spec: MisfitSpec) -> None:
        self.spec = spec
        self.name = f"misfit:{spec.name}"

    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        dtype = ctx.dtype
        std = tf.cast(self.spec.std, dtype)

        # Base mask (default: icemask unless user overrides with a state mask)
        mask = ctx.get_mask(self.spec.mask)

        # Observation availability: important that no data is defined as NaN
        for obs_name in self.spec.obs:
            y = ctx.obs(obs_name)
            mask = mask & tf.math.is_finite(y)

        res2 = None
        for comp_name, obs_name in zip(self.spec.components, self.spec.obs):
            y = ctx.obs(obs_name)
            m = ctx.model(comp_name)
            r = (y - m) / std
            r = tf.where(mask, r, tf.zeros_like(r)) # important! kill NaNs that would pollute gradient
            term = tf.square(tf.cast(r, dtype))
            res2 = term if res2 is None else (res2 + term)

        return tf.cast(0.5, dtype) * masked_sum(tf.cast(res2, dtype), mask)

class FieldPenaltyTerm(CostTerm):
    group = REGULARIZATION

    def __init__(self, spec: FieldPenaltySpec) -> None:
        if spec.penalty not in PENALTY_REGISTRY:
            raise ValueError(f"Unknown penalty '{spec.penalty}'. Available: {list(PENALTY_REGISTRY.keys())}")
        self.spec = spec
        self.name = f"reg:{spec.name}:{spec.penalty}"

    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        dtype = ctx.dtype

        field = ctx.physical(self.spec.name)  # ensures tape tracks θ

        lam = tf.cast(self.spec.lam, dtype)
        mask = ctx.get_mask(self.spec.mask)  # None => icemask

        fn = PENALTY_REGISTRY[self.spec.penalty]
        return fn(field=field, dx=ctx.dx, lam=lam, mask=mask, eps=float(self.spec.eps))

MISFIT_REGISTRY = {
    "gaussian": GaussianMisfitTerm,
}