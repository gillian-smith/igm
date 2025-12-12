# base.py
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from typeguard import typechecked

class Patching(ABC):
    """
    Abstract base class for tensor patching strategies.

    Lifecycle:

    - Construct with `patch_size` and optionally `fieldin`.
    - If `fieldin` is provided, initialization happens automatically.
    - Otherwise, call `initialize_for_field(sample_X)` manually once in eager mode.

    - During training / DA, call `generate_patches(X)` from inside tf.functions.
      This is graph-friendly and always returns [N, H_p, W_p, C].
    """

    def __init__(self, patch_size: int, fieldin: Optional[tf.Tensor] = None):
        self.patch_size = int(patch_size)

        # Lazily initialised per-domain metadata
        self._initialized: bool = False
        self._needs_patching: bool = True
        self._num_patches: Optional[int] = None
        self._patch_shape: Optional[Tuple[int, int, int]] = None  # (Hp, Wp, C)

        # Auto-initialize if fieldin provided
        if fieldin is not None:
            self.initialize_for_field(fieldin)

    # ------------------------------------------------------------------
    # Public metadata accessors
    # ------------------------------------------------------------------
    @property
    def num_patches(self) -> int:
        if not self._initialized:
            raise RuntimeError(
                "Patching.initialize_for_field() must be called before num_patches."
            )
        return self._num_patches  # type: ignore[return-value]

    @property
    def patch_shape(self) -> Tuple[int, int, int]:
        if not self._initialized:
            raise RuntimeError(
                "Patching.initialize_for_field() must be called before patch_shape."
            )
        return self._patch_shape  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Abstract implementation: "real" patching logic
    # ------------------------------------------------------------------
    @abstractmethod
    @typechecked
    def patch_tensor(self, X: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Split input tensor into patches.

        Subclasses assume that patching *is* required and must always return
        rank-4 [N, Hp, Wp, C].

        The "no patching, just wrap as [1,H,W,C]" case is handled by
        `generate_patches`, not by this method.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _initialize_patching_parameters(
        self, h: int, w: int, c: int, sample_X: tf.Tensor
    ) -> tuple[int, int, int, int]:
        """
        Subclass must compute:
        - cached parameters
        - patch shape (Hp, Wp, Cp)
        - number of patches N

        Must return (Hp, Wp, Cp, N)
        """
        raise NotImplementedError
    # ------------------------------------------------------------------
    # One-time eager initialization per domain
    # ------------------------------------------------------------------
    def initialize_for_field(self, sample_X: tf.Tensor) -> None:
        """
        Overridden to remove the premature call to patch_tensor().

        Subclasses now handle:
        - computing patching parameters
        - computing num_patches
        - computing patch_shape
        """
        sample_X = tf.convert_to_tensor(sample_X)
        self._validate_input(sample_X)

        # Static shape if possible, fallback to dynamic
        shape = sample_X.shape
        if shape.rank != 3 or any(dim is None for dim in shape):
            dyn_shape = tf.shape(sample_X)
            h = int(dyn_shape[0].numpy())
            w = int(dyn_shape[1].numpy())
            c = int(dyn_shape[2].numpy())
        else:
            h = int(shape[0])
            w = int(shape[1])
            c = int(shape[2])

        if h <= 0 or w <= 0 or c <= 0:
            raise ValueError("Invalid shape for sample_X.")

        # Decide if patching is needed
        self._needs_patching = (h > self.patch_size) or (w > self.patch_size)

        # Delegate parameter + metadata initialization to subclass
        Hp, Wp, Cp, N = self._initialize_patching_parameters(h, w, c, sample_X)

        self._patch_shape = (Hp, Wp, Cp)
        self._num_patches = N
        self._initialized = True


    def _patch_or_expand(self, X: tf.Tensor) -> tf.Tensor:
        """
        Internal helper: uses the same branching as generate_patches but
        without checking _initialized.
        """
        if not self._needs_patching:
            X = tf.convert_to_tensor(X)
            self._validate_input(X)
            return X[tf.newaxis, ...]
        else:
            return self.patch_tensor(X)

    # ------------------------------------------------------------------
    # Graph-friendly entry point for training / DA
    # ------------------------------------------------------------------
    @tf.function(reduce_retracing=True)
    def generate_patches(self, X: tf.Tensor) -> tf.Tensor:
        """
        Graph-friendly entry point used during training / DA.

        Requires:
        - `initialize_for_field()` has been called once with a representative
          field for this domain/configuration.

        Behaviour:
        - If `_needs_patching` is True → calls subclass `patch_tensor(X)`.
        - Else → wraps X as a single "patch": [1,H,W,C].

        Returns:
            patches: [N, H_p, W_p, C]
        """
        tf.debugging.assert_rank(
            X, 3, message="Input to generate_patches must have rank 3 [H,W,C]."
        )

        if not self._initialized:
            # This is a static Python check evaluated at trace time.
            raise RuntimeError(
                "Patching.generate_patches() called before initialize_for_field(). "
                "Call initialize_for_field() once in eager mode with a representative "
                "field for this domain."
            )

        if self._needs_patching:
            patches = self.patch_tensor(X)
        else:
            patches = X[tf.newaxis, ...]

        tf.debugging.assert_rank(
            patches, 4, message="Patching must produce rank-4 tensor [N,H,W,C]."
        )

        if self._patch_shape is not None:
            Hp, Wp, Cp = self._patch_shape
            patches = tf.ensure_shape(patches, [None, Hp, Wp, Cp])

        return patches

    # ------------------------------------------------------------------
    # Shared TF helpers (used by subclasses)
    # ------------------------------------------------------------------
    @tf.function(reduce_retracing=True)
    def _validate_input(self, X: tf.Tensor) -> None:
        tf.debugging.assert_rank(
            X, 3, "Input tensor must be 3D (height, width, channels)."
        )
        tf.debugging.assert_greater(tf.shape(X)[0], 0, "Height must be positive.")
        tf.debugging.assert_greater(tf.shape(X)[1], 0, "Width must be positive.")
        tf.debugging.assert_greater(tf.shape(X)[2], 0, "Channels must be positive.")

    @tf.function(reduce_retracing=True)
    def _get_patch_dimensions(self, X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        shape = tf.shape(X)
        return shape[0], shape[1]

    @tf.function(reduce_retracing=True)
    def _extract_patch(
        self, X: tf.Tensor, start_y: tf.Tensor, start_x: tf.Tensor
    ) -> tf.Tensor:
        return X[
            start_y : start_y + self.patch_size,
            start_x : start_x + self.patch_size,
            :,
        ]
