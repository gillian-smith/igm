import tensorflow as tf
from .base import Augmentation


class FlipParams(tf.experimental.ExtensionType):
    probability: float


class FlipAugmentation(Augmentation):
    def __init__(self, params: FlipParams):
        self.params = params

    def apply(self, x):
        # Each flip type is decided independently, allowing for both flips
        # Use tf.where for efficient conditional execution without lambda closures
        
        # Apply horizontal flip conditionally
        do_h_flip = tf.random.uniform([]) < self.params.probability
        x = tf.where(do_h_flip, tf.image.flip_left_right(x), x)
        
        # Apply vertical flip conditionally
        do_v_flip = tf.random.uniform([]) < self.params.probability
        x = tf.where(do_v_flip, tf.image.flip_up_down(x), x)
        
        return x
