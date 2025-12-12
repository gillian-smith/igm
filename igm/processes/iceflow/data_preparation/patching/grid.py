import tensorflow as tf
from .base import Patching  # or from .patching import Patching if colocated

class GridPatching(Patching):
    """
    Non-overlapping grid patching with fixed patch size.

    - Stride = patch_size
    - Symmetric padding so all patches are [patch_size, patch_size, C]
    - If either H or W < patch_size, return a single full-image patch
    """

    def __init__(self, patch_size: int):
        super().__init__(patch_size)

    @tf.function(reduce_retracing=True)
    def _pad_tensor(self, X: tf.Tensor, padding_h: tf.Tensor, padding_w: tf.Tensor) -> tf.Tensor:
        needs_padding = tf.logical_or(padding_h > 0, padding_w > 0)
        paddings = [[0, padding_h], [0, padding_w], [0, 0]]
        return tf.cond(needs_padding, lambda: tf.pad(X, paddings, mode="SYMMETRIC"), lambda: X)

    @tf.function(reduce_retracing=True)
    def patch_tensor(self, X: tf.Tensor) -> tf.Tensor:
        """
        Returns a tensor of shape (num_patches, patch_size, patch_size, channels)
        when patching is active; otherwise (1, H, W, C) for small images.
        """
        self._validate_input(X)
        height, width = self._get_patch_dimensions(X)

        # Same early-exit rule used elsewhere: if either dim is smaller, don't patch
        if tf.logical_or(self.patch_size > height, self.patch_size > width):
            return tf.expand_dims(X, axis=0)

        # Number of tiles per dimension (ceil division)
        ps = tf.constant(self.patch_size, tf.int32)
        n_y = tf.math.floordiv(height + ps - 1, ps)
        n_x = tf.math.floordiv(width  + ps - 1, ps)

        # Pad to an exact multiple of patch_size
        padding_h = n_y * ps - height
        padding_w = n_x * ps - width
        Xp = self._pad_tensor(X, padding_h, padding_w)

        # Start coordinates (stride = patch_size)
        y_coords = tf.range(n_y, dtype=tf.int32) * ps
        x_coords = tf.range(n_x, dtype=tf.int32) * ps
        y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing="ij")
        y_flat = tf.reshape(y_grid, [-1])
        x_flat = tf.reshape(x_grid, [-1])

        # Extract fixed-size patches using the base helper
        patches = tf.map_fn(
            lambda ij: self._extract_patch(Xp, ij[0], ij[1]),
            tf.stack([y_flat, x_flat], axis=1),
            fn_output_signature=tf.TensorSpec(
                shape=[self.patch_size, self.patch_size, None], dtype=X.dtype
            ),
            parallel_iterations=10,
        )
        return patches
