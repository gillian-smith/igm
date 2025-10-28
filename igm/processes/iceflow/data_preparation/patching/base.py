import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple
from typeguard import typechecked


class Patching(ABC):
    """
    Abstract base class for tensor patching strategies.

    Patching takes an input tensor with shape (height, width, channels)
    and splits it into smaller patches for processing. Different strategies
    can be used for determining patch layout, overlap, and stacking behavior.

    The framework supports:
    - patch_tensor: main method that splits tensor into patches
    - Configurable patch size and overlap strategies
    - All patches stacked along the batch dimension for efficient processing
    """

    def __init__(self, patch_size: int):
        """
        Initialize base patching.

        Args:
            patch_size: Size of each patch (height and width).
        """
        # Import here to avoid circular imports
        self.patch_size = patch_size

    @abstractmethod
    @typechecked
    def patch_tensor(self, X: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Split input tensor into patches.

        This method must be implemented by subclasses to define the specific
        patching strategy (e.g., with/without overlap, different stacking methods).

        Args:
            X: Input tensor of shape (height, width, channels).
            **kwargs: Additional parameters specific to each implementation.

        Returns:
            Tensor containing patches stacked along the batch dimension.
        """
        pass

    @tf.function(reduce_retracing=True)
    def _validate_input(self, X: tf.Tensor) -> None:
        """
        Validate input tensor shape and properties.

        Args:
            X: Input tensor to validate.

        Raises:
            tf.errors.InvalidArgumentError: If input is invalid.
        """
        tf.debugging.assert_rank(
            X, 3, "Input tensor must be 3D (height, width, channels)"
        )
        tf.debugging.assert_greater(tf.shape(X)[0], 0, "Height must be positive")
        tf.debugging.assert_greater(tf.shape(X)[1], 0, "Width must be positive")
        tf.debugging.assert_greater(tf.shape(X)[2], 0, "Channels must be positive")

    @tf.function(reduce_retracing=True)
    def _get_patch_dimensions(self, X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get height and width from input tensor.

        Args:
            X: Input tensor of shape (height, width, channels).

        Returns:
            Tuple of (height, width) as TensorFlow tensors.
        """
        shape = tf.shape(X)
        return shape[0], shape[1]

    @tf.function(reduce_retracing=True)
    def _extract_patch(
        self, X: tf.Tensor, start_y: tf.Tensor, start_x: tf.Tensor
    ) -> tf.Tensor:
        """
        Extract a single patch from the input tensor.

        Args:
            X: Input tensor of shape (height, width, channels).
            start_y: Starting y coordinate for the patch.
            start_x: Starting x coordinate for the patch.

        Returns:
            Extracted patch of shape (patch_size, patch_size, channels).
        """
        return X[
            start_y : start_y + self.patch_size,
            start_x : start_x + self.patch_size,
            :,
        ]
