import tensorflow as tf
from typing import Tuple
from .base import Patching


class OverlapPatching(Patching):
    """
    Patching strategy with configurable overlap between patches.

    """

    def __init__(
        self,
        patch_size: int,
        overlap: float = 0.25,
    ):
        """
        Initialize overlap patching with adaptive approach.

        Args:
            patch_size: Size of each patch (height and width).
            overlap: Target fractional overlap between patches (0.0 to 1.0).
        """
        super().__init__(patch_size)
        if not 0.0 <= overlap < 1.0:
            raise ValueError("Overlap must be in range [0.0, 1.0)")
        self.target_overlap = overlap

    @tf.function(reduce_retracing=True)
    def _calculate_patching_parameters(
        self, height: tf.Tensor, width: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate optimal adaptive patching parameters using TensorFlow operations.

        Args:
            height: Height of input tensor.
            width: Width of input tensor.

        Returns:
            Tuple of (n_patches_h, n_patches_w, stride_h, stride_w, padding_h, padding_w).
        """
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        patch_size_f = tf.cast(self.patch_size, tf.float32)

        # Calculate ideal stride from target overlap
        ideal_stride = patch_size_f * (1.0 - self.target_overlap)

        # Calculate minimum number of patches needed with ideal stride
        min_patches_h = tf.maximum(
            1.0, tf.math.ceil((height_f - patch_size_f) / ideal_stride) + 1.0
        )
        min_patches_w = tf.maximum(
            1.0, tf.math.ceil((width_f - patch_size_f) / ideal_stride) + 1.0
        )

        # Calculate parameters for minimum configuration
        # Use floor to guarantee coverage (stride will never be too large)
        n_patches_h_1 = tf.cast(min_patches_h, tf.int32)
        n_patches_w_1 = tf.cast(min_patches_w, tf.int32)

        stride_h_1 = tf.cond(
            n_patches_h_1 > 1,
            lambda: tf.cast(
                tf.floor((height_f - patch_size_f) / (min_patches_h - 1.0)), tf.int32
            ),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        stride_w_1 = tf.cond(
            n_patches_w_1 > 1,
            lambda: tf.cast(
                tf.floor((width_f - patch_size_f) / (min_patches_w - 1.0)), tf.int32
            ),
            lambda: tf.constant(0, dtype=tf.int32),
        )

        # Calculate padding needed for configuration 1
        last_end_h_1 = tf.cond(
            n_patches_h_1 > 1,
            lambda: (n_patches_h_1 - 1) * stride_h_1 + self.patch_size,
            lambda: self.patch_size,
        )
        last_end_w_1 = tf.cond(
            n_patches_w_1 > 1,
            lambda: (n_patches_w_1 - 1) * stride_w_1 + self.patch_size,
            lambda: self.patch_size,
        )

        padding_h_1 = tf.maximum(0, last_end_h_1 - height)
        padding_w_1 = tf.maximum(0, last_end_w_1 - width)

        # Calculate overlap for configuration 1
        overlap_h_1 = tf.cond(
            stride_h_1 > 0,
            lambda: tf.cast(self.patch_size - stride_h_1, tf.float32) / patch_size_f,
            lambda: 0.0,
        )
        overlap_w_1 = tf.cond(
            stride_w_1 > 0,
            lambda: tf.cast(self.patch_size - stride_w_1, tf.float32) / patch_size_f,
            lambda: 0.0,
        )
        avg_overlap_1 = (overlap_h_1 + overlap_w_1) / 2.0

        # Score for configuration 1
        overlap_penalty_1 = tf.abs(avg_overlap_1 - self.target_overlap)
        padding_penalty_1 = tf.cast(padding_h_1 + padding_w_1, tf.float32) / 100.0
        score_1 = overlap_penalty_1 + padding_penalty_1

        # Calculate parameters for configuration 2
        # Use ideal stride with padding to maintain target overlap more closely
        n_patches_h_2 = n_patches_h_1
        n_patches_w_2 = n_patches_w_1

        # Use ideal stride based on target overlap (may require padding)
        stride_h_2 = tf.cond(
            n_patches_h_2 > 1,
            lambda: tf.cast(tf.round(ideal_stride), tf.int32),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        stride_w_2 = tf.cond(
            n_patches_w_2 > 1,
            lambda: tf.cast(tf.round(ideal_stride), tf.int32),
            lambda: tf.constant(0, dtype=tf.int32),
        )

        # Calculate padding needed for configuration 2
        last_end_h_2 = tf.cond(
            n_patches_h_2 > 1,
            lambda: (n_patches_h_2 - 1) * stride_h_2 + self.patch_size,
            lambda: self.patch_size,
        )
        last_end_w_2 = tf.cond(
            n_patches_w_2 > 1,
            lambda: (n_patches_w_2 - 1) * stride_w_2 + self.patch_size,
            lambda: self.patch_size,
        )

        padding_h_2 = tf.maximum(0, last_end_h_2 - height)
        padding_w_2 = tf.maximum(0, last_end_w_2 - width)

        # Calculate overlap for configuration 2
        overlap_h_2 = tf.cond(
            stride_h_2 > 0,
            lambda: tf.cast(self.patch_size - stride_h_2, tf.float32) / patch_size_f,
            lambda: 0.0,
        )
        overlap_w_2 = tf.cond(
            stride_w_2 > 0,
            lambda: tf.cast(self.patch_size - stride_w_2, tf.float32) / patch_size_f,
            lambda: 0.0,
        )
        avg_overlap_2 = (overlap_h_2 + overlap_w_2) / 2.0

        # Score for configuration 2
        overlap_penalty_2 = tf.abs(avg_overlap_2 - self.target_overlap)
        padding_penalty_2 = tf.cast(padding_h_2 + padding_w_2, tf.float32) / 100.0
        score_2 = overlap_penalty_2 + padding_penalty_2

        # Check if configuration 1 provides full coverage
        # Config 1 only valid if last patch reaches end of image
        config_1_covers_h = last_end_h_1 >= height
        config_1_covers_w = last_end_w_1 >= width
        config_1_has_coverage = tf.logical_and(config_1_covers_h, config_1_covers_w)

        # Choose configuration: prefer config 1 only if it has full coverage AND better score
        use_config_1 = tf.logical_and(config_1_has_coverage, score_1 <= score_2)

        n_patches_h = tf.cond(
            use_config_1, lambda: n_patches_h_1, lambda: n_patches_h_2
        )
        n_patches_w = tf.cond(
            use_config_1, lambda: n_patches_w_1, lambda: n_patches_w_2
        )
        stride_h = tf.cond(use_config_1, lambda: stride_h_1, lambda: stride_h_2)
        stride_w = tf.cond(use_config_1, lambda: stride_w_1, lambda: stride_w_2)
        padding_h = tf.cond(use_config_1, lambda: padding_h_1, lambda: padding_h_2)
        padding_w = tf.cond(use_config_1, lambda: padding_w_1, lambda: padding_w_2)

        return n_patches_h, n_patches_w, stride_h, stride_w, padding_h, padding_w

    @tf.function(reduce_retracing=True)
    def _pad_tensor(
        self, X: tf.Tensor, padding_h: tf.Tensor, padding_w: tf.Tensor
    ) -> tf.Tensor:
        """
        Pad tensor using TensorFlow operations (graph-compatible).

        Args:
            X: Input tensor of shape (height, width, channels).
            padding_h: Height padding amount.
            padding_w: Width padding amount.

        Returns:
            Padded tensor with symmetric padding.
        """
        # Only pad if necessary
        needs_padding = tf.logical_or(padding_h > 0, padding_w > 0)

        paddings = [
            [0, padding_h],  # height padding
            [0, padding_w],  # width padding
            [0, 0],  # no channel padding
        ]

        return tf.cond(
            needs_padding, lambda: tf.pad(X, paddings, mode="SYMMETRIC"), lambda: X
        )

    @tf.function(reduce_retracing=True)
    def patch_tensor(self, X: tf.Tensor) -> tf.Tensor:
        """
        Split input tensor into overlapping patches using hybrid adaptive approach.

        This method uses adaptive patching to achieve 100% coverage with minimal padding
        while maintaining overlap close to the target value. Now fully compatible with
        TensorFlow graph mode.

        Args:
            X: Input tensor of shape (height, width, channels).

        Returns:
            Tensor of patches with shape (num_patches, patch_size, patch_size, channels).
            All patches are stacked along the batch dimension.
        """
        self._validate_input(X)
        height, width = self._get_patch_dimensions(X)

        # Handle case where input is smaller than patch size
        if tf.logical_and(self.patch_size > width, self.patch_size > height):
            return tf.expand_dims(X, axis=0)

        # Use TensorFlow-compatible adaptive approach
        n_patches_y, n_patches_x, stride_y, stride_x, padding_h, padding_w = (
            self._calculate_patching_parameters(height, width)
        )

        # Pad the input tensor if needed
        padded_X = self._pad_tensor(X, padding_h, padding_w)

        # Generate patch coordinates
        y_coords = tf.range(n_patches_y) * stride_y
        x_coords = tf.range(n_patches_x) * stride_x

        # Create mesh grid of coordinates
        y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing="ij")
        y_flat = tf.reshape(y_grid, [-1])
        x_flat = tf.reshape(x_grid, [-1])

        # Extract all patches using tf.map_fn for efficiency
        patches = tf.map_fn(
            lambda coords: self._extract_patch(padded_X, coords[0], coords[1]),
            tf.stack([y_flat, x_flat], axis=1),
            fn_output_signature=tf.TensorSpec(
                shape=[self.patch_size, self.patch_size, None], dtype=X.dtype
            ),
            parallel_iterations=10,
        )

        return patches
