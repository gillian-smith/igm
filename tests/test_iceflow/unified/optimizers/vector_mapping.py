# tests/optimizers/_vector_mapping.py

#!/usr/bin/env python3
import tensorflow as tf
from typing import List, Tuple
from igm.processes.iceflow.unified.mappings.mapping import Mapping

class VectorMappingForLBFGS(Mapping):
    """
    Minimal Mapping for optimizer tests.
    - Parameters: a single trainable vector theta ∈ R^n
    - U,V: dummy tensors (cost_fn will ignore them and read theta directly)
    """
    def __init__(self, n: int, dtype=tf.float32):
        super().__init__(bcs=[])
        self.theta = tf.Variable(tf.zeros([n], dtype=dtype), trainable=True, name="theta")
        self._shapes: List[tf.TensorShape] = [self.theta.shape]
        self._sizes:  List[tf.Tensor]       = [tf.size(self.theta)]

    # --- Forward (dummy) ---
    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # Produce small dummy tensors; shapes are irrelevant for the test cost_fn.
        z = tf.zeros([1, 1, 1, 1], dtype=self.theta.dtype)
        return z, z

    # --- Parameter plumbing ---
    def get_w(self) -> List[tf.Variable]:
        return [self.theta]

    def set_w(self, w: List[tf.Tensor]) -> None:
        if len(w) != 1:
            raise ValueError("TestVectorMapping.set_w: expected a single tensor.")
        self.theta.assign(w[0])

    def copy_w(self, w: List[tf.Variable]) -> List[tf.Tensor]:
        return [w[0].read_value()]

    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(w_flat)

    def flatten_w(self, w: List[tf.Variable | tf.Tensor]) -> tf.Tensor:
        return tf.reshape(w[0], [-1])

    def unflatten_w(self, w_flat: tf.Tensor) -> List[tf.Tensor]:
        return [tf.reshape(w_flat, self.theta.shape)]

    def apply_theta_to_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs

    # --- Halt criterion (none) ---
    def check_halt_criterion(self, iteration: int, cost: tf.Tensor):
        # Keep running unless Optimizer stops by its own criteria
        return tf.constant(False, tf.bool), tf.constant("", tf.string)

# --- Bounded mapping for box-constraint tests --------------------------------

class BoundedVectorMappingForLBFGS(VectorMappingForLBFGS):
    """
    Extends the simple vector mapping with θ-space box bounds to activate
    the projected path line-search and free-mask logic in OptimizerLBFGS.
    """
    def __init__(self, n: int, L: float, U: float, dtype=tf.float64):
        super().__init__(n=n, dtype=dtype)
        L = tf.convert_to_tensor(L, dtype=self.theta.dtype)
        U = tf.convert_to_tensor(U, dtype=self.theta.dtype)
        self._L = tf.fill([n], L)
        self._U = tf.fill([n], U)

    # This method is probed via hasattr(map, "get_box_bounds_flat")
    def get_box_bounds_flat(self):
        return self._L, self._U