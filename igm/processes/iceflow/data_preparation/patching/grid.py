import tensorflow as tf
from typing import Tuple
from typeguard import typechecked
from .base import Patching


class GridPatching(Patching):
    """
    Patching strategy that divides input into a regular grid without overlap.

    This implementation splits the input tensor into non-overlapping patches
    arranged in a regular grid. All patches are stacked along the batch dimension.
    """

    def __init__(self, patch_size: int):
        """
        Initialize grid patching.

        Args:
            patch_size: Size of each patch (height and width).
        """
        super().__init__(patch_size)

    @tf.function(reduce_retracing=True)
    def _calculate_grid_parameters(
        self, height: tf.Tensor, width: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate grid patching parameters.

        Args:
            height: Height of input tensor.
            width: Width of input tensor.

        Returns:
            Tuple of (n_patches_y, n_patches_x, patch_height, patch_width).
        """
        # Calculate number of patches in each dimension
        n_patches_y = height // self.patch_size + 1
        n_patches_x = width // self.patch_size + 1

        # Calculate actual patch dimensions (may be smaller than patch_size)
        patch_height = height // n_patches_y
        patch_width = width // n_patches_x

        return n_patches_y, n_patches_x, patch_height, patch_width

    @tf.function(reduce_retracing=True)
    def _extract_grid_patch(
        self,
        X: tf.Tensor,
        i: tf.Tensor,
        j: tf.Tensor,
        patch_height: tf.Tensor,
        patch_width: tf.Tensor,
    ) -> tf.Tensor:
        """
        Extract a patch from the regular grid.

        Args:
            X: Input tensor of shape (height, width, channels).
            i: Grid row index.
            j: Grid column index.
            patch_height: Height of each patch.
            patch_width: Width of each patch.

        Returns:
            Extracted patch.
        """
        start_y = j * patch_height
        start_x = i * patch_width
        end_y = start_y + patch_height
        end_x = start_x + patch_width

        return X[start_y:end_y, start_x:end_x, :]

    @tf.function(reduce_retracing=True)
    def _extract_all_grid_patches(
        self,
        X: tf.Tensor,
        n_patches_y: tf.Tensor,
        n_patches_x: tf.Tensor,
        patch_height: tf.Tensor,
        patch_width: tf.Tensor,
    ) -> tf.Tensor:
        """
        Extract all grid patches using TensorFlow operations.

        Args:
            X: Input tensor.
            n_patches_y: Number of patches in y direction.
            n_patches_x: Number of patches in x direction.
            patch_height: Height of each patch.
            patch_width: Width of each patch.

        Returns:
            All patches stacked together.
        """
        # Create coordinate meshes
        i_range = tf.range(n_patches_x)
        j_range = tf.range(n_patches_y)

        # Create all coordinate combinations
        i_grid, j_grid = tf.meshgrid(i_range, j_range, indexing="ij")
        i_flat = tf.reshape(i_grid, [-1])
        j_flat = tf.reshape(j_grid, [-1])

        # Extract patches using map_fn
        coordinates = tf.stack([i_flat, j_flat], axis=1)

        def extract_single_patch(coords):
            i, j = coords[0], coords[1]
            return self._extract_grid_patch(X, i, j, patch_height, patch_width)

        patches = tf.map_fn(
            extract_single_patch,
            coordinates,
            fn_output_signature=tf.TensorSpec(shape=[None, None, None], dtype=X.dtype),
            parallel_iterations=1,  # Set to 1 to avoid warning in eager execution
        )

        return patches

    @typechecked
    def patch_tensor(self, X: tf.Tensor) -> tf.Tensor:
        """
        Split input tensor into grid patches.

        Args:
            X: Input tensor of shape (height, width, channels).

        Returns:
            Tensor of patches with shape (num_patches, patch_height, patch_width, channels).
            All patches are stacked along the batch dimension.
        """
        self._validate_input(X)
        height, width = self._get_patch_dimensions(X)

        n_patches_y, n_patches_x, patch_height, patch_width = (
            self._calculate_grid_parameters(height, width)
        )

        # Extract all patches
        patches = self._extract_all_grid_patches(
            X, n_patches_y, n_patches_x, patch_height, patch_width
        )

        # patches shape: (num_patches, patch_height, patch_width, channels)
        return patches
