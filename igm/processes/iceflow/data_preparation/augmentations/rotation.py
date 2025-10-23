import tensorflow as tf
from .base import Augmentation


class RotationParams(tf.experimental.ExtensionType):
    probability: float


class RotationAugmentation(Augmentation):
    def __init__(self, params: RotationParams):
        self.params = params

    def apply(self, x):
        # Check if image is square - only apply rotation to square images
        height = tf.shape(x)[0]
        width = tf.shape(x)[1]
        is_square = tf.equal(height, width)

        # Generate random rotation amount (0, 1, 2, or 3 for 0°, 90°, 180°, 270°)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        
        # Only rotate if probability check passes AND image is square
        should_rotate = tf.logical_and(
            tf.random.uniform([]) < self.params.probability, is_square
        )

        # Use tf.switch_case with static k values for graph compatibility
        # This avoids dynamic k parameter issues with tf.image.rot90
        def rotate():
            return tf.switch_case(
                k,
                branch_fns=[
                    lambda: x,                        # k=0: no rotation
                    lambda: tf.image.rot90(x, k=1),   # k=1: 90° CCW
                    lambda: tf.image.rot90(x, k=2),   # k=2: 180°
                    lambda: tf.image.rot90(x, k=3),   # k=3: 270° CCW
                ]
            )
        
        # Apply rotation only if conditions are met
        return tf.cond(should_rotate, rotate, lambda: x)
