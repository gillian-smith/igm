# igm/processes/iceflow/unified/mappings/transforms.py
#!/usr/bin/env python3
from __future__ import annotations
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Type

class ParameterTransform(ABC):
    name: str

    @abstractmethod
    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        ...

    @abstractmethod
    def to_physical(self, theta: tf.Tensor) -> tf.Tensor:
        ...

    @abstractmethod
    def theta_bounds(
        self,
        lower_phys: Optional[float],
        upper_phys: Optional[float],
        dtype: tf.dtypes.DType,
        eps: float = 1e-12,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        ...

class IdentityTransform(ParameterTransform):
    name = "identity"
    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        return x_phys
    def to_physical(self, theta: tf.Tensor) -> tf.Tensor:
        return theta
    def theta_bounds(self, lower_phys, upper_phys, dtype, eps: float = 1e-12):
        L = -tf.constant(float("inf"), dtype) if lower_phys is None else tf.constant(lower_phys, dtype)
        U =  tf.constant(float("inf"), dtype) if upper_phys is None else tf.constant(upper_phys, dtype)
        return L, U

class Log10Transform(ParameterTransform):
    name = "log10"
    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        ln10 = tf.constant(2.302585092994046, x_phys.dtype)
        return tf.math.log(tf.maximum(x_phys, tf.cast(eps, x_phys.dtype))) / ln10
    def to_physical(self, theta: tf.Tensor) -> tf.Tensor:
        ln10 = tf.constant(2.302585092994046, theta.dtype)
        return tf.exp(ln10 * theta)
    def theta_bounds(self, lower_phys, upper_phys, dtype, eps: float = 1e-12):
        ln10 = tf.constant(2.302585092994046, dtype)
        if (lower_phys is None) or (lower_phys <= 0.0):
            L = -tf.constant(float("inf"), dtype)
        else:
            L = tf.math.log(tf.constant(lower_phys, dtype)) / ln10
        if upper_phys is None:
            U = tf.constant(float("inf"), dtype)
        else:
            if upper_phys <= 0.0:
                raise ValueError("Upper bound must be > 0 for log10.")
            U = tf.math.log(tf.constant(upper_phys, dtype)) / ln10
        return L, U

class SoftplusTransform(ParameterTransform):
    """Maps ℝ → (0, ∞) with y = softplus(theta) = log(1 + exp(theta))."""
    name = "softplus"

    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        # inverse softplus: theta = log(exp(y) - 1); use expm1 for stability
        y = tf.maximum(x_phys, tf.cast(eps, x_phys.dtype))
        return tf.math.log(tf.math.expm1(y))

    def to_physical(self, theta: tf.Tensor) -> tf.Tensor:
        # forward softplus
        return tf.nn.softplus(theta)

    def theta_bounds(
        self,
        lower_phys: Optional[float],
        upper_phys: Optional[float],
        dtype: tf.dtypes.DType,
        eps: float = 1e-12,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Convert PHYSICAL bounds to theta-space via inverse softplus
        if (lower_phys is None) or (lower_phys <= 0.0):
            L = -tf.constant(float("inf"), dtype)
        else:
            L = tf.math.log(tf.math.expm1(tf.constant(lower_phys, dtype)))
        if upper_phys is None:
            U = tf.constant(float("inf"), dtype)
        else:
            if upper_phys <= 0.0:
                raise ValueError("Upper bound must be > 0 for softplus.")
            U = tf.math.log(tf.math.expm1(tf.constant(upper_phys, dtype)))
        return L, U

# registry
TRANSFORMS: Dict[str, Type[ParameterTransform]] = {
    "identity": IdentityTransform,
    "log10": Log10Transform,
    "softplus": SoftplusTransform,
}