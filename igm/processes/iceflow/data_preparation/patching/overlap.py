# overlap.py

import tensorflow as tf
from typing import Tuple
from .base import Patching


class OverlapPatching(Patching):
    """
    Overlapping patch extraction with guaranteed full coverage.

    Design:
    - Parameters (n_patches, strides, padding) are computed once in eager mode
      inside initialize_for_field via _initialize_patching_parameters.
    - These parameters are stored as plain Python ints.
    - patch_tensor() uses these cached values inside a tf.function in graph mode.
    """

    def __init__(self, patch_size: int, overlap: float = 0.25, fieldin=None):
        if overlap < 0.0 or overlap >= 1.0:
            raise ValueError("Overlap must be in [0.0, 1.0).")

        self.target_overlap = float(overlap)

        # Will be populated in initialize_for_field
        self._n_patches_y: int | None = None
        self._n_patches_x: int | None = None
        self._stride_y: int | None = None
        self._stride_x: int | None = None
        self._padding_h: int | None = None
        self._padding_w: int | None = None

        super().__init__(patch_size, fieldin=fieldin)

    # ----------------------------------------------------------------------
    # Original TF parameter computation (unchanged logic)
    # ----------------------------------------------------------------------
    @tf.function(reduce_retracing=True)
    def _calculate_patching_parameters(
        self, height: tf.Tensor, width: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute patching parameters ensuring full coverage while approximating
        the target overlap. This is the original TF-based logic.
        """
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        patch_size_f = tf.cast(self.patch_size, tf.float32)

        # Ideal stride from target overlap
        ideal_stride = patch_size_f * (1.0 - self.target_overlap)

        # Minimum patches with ideal stride
        min_patches_h = tf.maximum(
            1.0, tf.math.ceil((height_f - patch_size_f) / ideal_stride) + 1.0
        )
        min_patches_w = tf.maximum(
            1.0, tf.math.ceil((width_f - patch_size_f) / ideal_stride) + 1.0
        )

        # --- Configuration 1: coverage-guaranteeing floor stride ---
        n_patches_h_1 = tf.cast(min_patches_h, tf.int32)
        n_patches_w_1 = tf.cast(min_patches_w, tf.int32)

        stride_h_1 = tf.cond(
            n_patches_h_1 > 1,
            lambda: tf.cast(
                tf.floor((height_f - patch_size_f) / (min_patches_h - 1.0)), tf.int32
            ),
            lambda: tf.constant(0, tf.int32),
        )
        stride_w_1 = tf.cond(
            n_patches_w_1 > 1,
            lambda: tf.cast(
                tf.floor((width_f - patch_size_f) / (min_patches_w - 1.0)), tf.int32
            ),
            lambda: tf.constant(0, tf.int32),
        )

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

        overlap_penalty_1 = tf.abs(avg_overlap_1 - self.target_overlap)
        padding_penalty_1 = tf.cast(padding_h_1 + padding_w_1, tf.float32) / 100.0
        score_1 = overlap_penalty_1 + padding_penalty_1

        # --- Configuration 2: ideal stride + padding ---
        n_patches_h_2 = n_patches_h_1
        n_patches_w_2 = n_patches_w_1

        stride_h_2 = tf.cond(
            n_patches_h_2 > 1,
            lambda: tf.cast(tf.round(ideal_stride), tf.int32),
            lambda: tf.constant(0, tf.int32),
        )
        stride_w_2 = tf.cond(
            n_patches_w_2 > 1,
            lambda: tf.cast(tf.round(ideal_stride), tf.int32),
            lambda: tf.constant(0, tf.int32),
        )

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

        overlap_penalty_2 = tf.abs(avg_overlap_2 - self.target_overlap)
        padding_penalty_2 = tf.cast(padding_h_2 + padding_w_2, tf.float32) / 100.0
        score_2 = overlap_penalty_2 + padding_penalty_2

        # Config 1 only valid if it actually reaches the end
        config_1_valid = tf.logical_and(
            last_end_h_1 >= height, last_end_w_1 >= width
        )
        use_config_1 = tf.logical_and(config_1_valid, score_1 <= score_2)

        n_patches_h = tf.cond(use_config_1, lambda: n_patches_h_1, lambda: n_patches_h_2)
        n_patches_w = tf.cond(use_config_1, lambda: n_patches_w_1, lambda: n_patches_w_2)
        stride_h = tf.cond(use_config_1, lambda: stride_h_1, lambda: stride_h_2)
        stride_w = tf.cond(use_config_1, lambda: stride_w_1, lambda: stride_w_2)
        padding_h = tf.cond(use_config_1, lambda: padding_h_1, lambda: padding_h_2)
        padding_w = tf.cond(use_config_1, lambda: padding_w_1, lambda: padding_w_2)

        return n_patches_h, n_patches_w, stride_h, stride_w, padding_h, padding_w

    # ----------------------------------------------------------------------
    # Hook called by Patching.initialize_for_field(...)
    # ----------------------------------------------------------------------
    def _initialize_patching_parameters(
        self, h: int, w: int, c: int, sample_X: tf.Tensor
    ) -> tuple[int, int, int, int]:
        """
        Compute and cache patching parameters BEFORE any patch_tensor() call.

        Returns:
            Hp, Wp, Cp, N  (patch height, width, channels, num_patches)
        """

        # Case 1: no patching needed → single full-image "patch"
        if not self._needs_patching:
            self._n_patches_y = 1
            self._n_patches_x = 1
            self._stride_y = 0
            self._stride_x = 0
            self._padding_h = 0
            self._padding_w = 0

            Hp, Wp, Cp, N = h, w, c, 1
            return Hp, Wp, Cp, N

        # Case 2: use TF logic to compute parameters (once, in eager mode)
        n_py_t, n_px_t, sy_t, sx_t, pad_h_t, pad_w_t = self._calculate_patching_parameters(
            tf.constant(h, tf.int32),
            tf.constant(w, tf.int32),
        )

        # Store as Python ints so tests and plain Python arithmetic work naturally
        self._n_patches_y = int(n_py_t.numpy())
        self._n_patches_x = int(n_px_t.numpy())
        self._stride_y = int(sy_t.numpy())
        self._stride_x = int(sx_t.numpy())
        self._padding_h = int(pad_h_t.numpy())
        self._padding_w = int(pad_w_t.numpy())

        # Patch shape is fixed: patch_size × patch_size × C
        Hp = self.patch_size
        Wp = self.patch_size
        Cp = c
        N = self._n_patches_y * self._n_patches_x

        return Hp, Wp, Cp, N

    # ----------------------------------------------------------------------
    # Graph-mode patch extraction using cached Python-int parameters
    # ----------------------------------------------------------------------
    @tf.function(reduce_retracing=True)
    def patch_tensor(self, X: tf.Tensor) -> tf.Tensor:
        """
        Extract patches using cached layout parameters.

        Behaviour:
        - If _needs_patching is False: returns [1, H, W, C] (handled by base.generate_patches).
        - If True: uses cached (n_patches_y/x, stride_y/x, padding_h/w) computed in eager mode.
        """
        self._validate_input(X)

        # No patching case is handled in generate_patches; here we assume patching is needed.
        if not self._needs_patching:
            return X[tf.newaxis, ...]

        # Pull cached python ints
        n_py = self._n_patches_y
        n_px = self._n_patches_x
        sy = self._stride_y
        sx = self._stride_x
        pad_h = self._padding_h
        pad_w = self._padding_w

        # Safety: ensure initialization happened
        if any(v is None for v in (n_py, n_px, sy, sx, pad_h, pad_w)):
            raise RuntimeError(
                "OverlapPatching.patch_tensor called before initialize_for_field."
            )

        # Padding if needed
        need_pad = tf.logical_or(pad_h > 0, pad_w > 0)
        X = tf.cond(
            need_pad,
            lambda: tf.pad(
                X,
                [[0, pad_h], [0, pad_w], [0, 0]],
                mode="SYMMETRIC",
            ),
            lambda: X,
        )

        # Coordinate grid
        y_coords = tf.range(n_py, dtype=tf.int32) * sy
        x_coords = tf.range(n_px, dtype=tf.int32) * sx
        y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing="ij")
        y_flat = tf.reshape(y_grid, [-1])
        x_flat = tf.reshape(x_grid, [-1])
        coords = tf.stack([y_flat, x_flat], axis=1)

        # Extract patches
        patches = tf.map_fn(
            lambda c: X[c[0] : c[0] + self.patch_size, c[1] : c[1] + self.patch_size, :],
            coords,
            fn_output_signature=tf.TensorSpec(
                shape=[self.patch_size, self.patch_size, None],
                dtype=X.dtype,
            ),
            parallel_iterations=10,
        )

        return patches
