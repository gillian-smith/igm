# batch_builder.py

from __future__ import annotations

from typing import Sequence, Tuple

import tensorflow as tf

from .config import (
    PreparationParams,
    _augs_effective,
    _noise_is_effective,
)
from .augmentation_ops import (
    _get_rotation_augmentation,
    _get_flip_augmentation,
    _get_noise_augmentation,
    _apply_augmentations_to_tensor,
    _split_tensor_into_batches,
    ensure_fixed_tensor_shape,
)


class TrainingBatchBuilder(tf.Module):
    """
    Build per-iteration training batches from a fixed pool of patches.

    Semantics (per call):

    Let N be the number of patches produced by the patching strategy

    • If target_samples > N AND at least one augmentation is *effective*:
        - Always include ALL N original patches (unaugmented)
        - Add (target_samples - N) *augmented* copies
        - Total samples per iteration = target_samples

    • Else:
        - Use ONLY the original N patches (no per-iteration augmentation)
        - Total samples per iteration = N

    Batching:

    - Effective batch size B = min(config.batch_size, total_samples_per_iter)
    - If total_samples_per_iter > B → multiple full batches
      Shape: [M, B, H, W, C], M = floor(total_samples / B)
    - If total_samples_per_iter == B → one batch
      Shape: [1, B, H, W, C]

    """

    def __init__(
        self,
        preparation_params: PreparationParams,
        fieldin_names: Sequence[str],
        patch_shape: Tuple[int, int, int],   # (H, W, C)
        num_patches: int,                    # N from patcher.num_patches
        seed: int = 42,
        name: str = "training_batch_builder",
    ):
        super().__init__(name=name)

        self.prep = preparation_params

        # ----- Static geometry -----
        self.H, self.W, self.C = map(int, patch_shape)
        self.num_patches = int(num_patches)

        # ----- Precision / dtype -----
        if preparation_params.precision == "single":
            self.dtype = tf.float32
        elif preparation_params.precision == "double":
            self.dtype = tf.float64
        else:
            raise ValueError(
                f"Unsupported precision '{preparation_params.precision}', "
                "expected 'single' or 'double'."
            )

        # ----- Augmentation configuration (all eager, Python-level) -----
        # Effective augmentations (noise only if mask actually hits something)
        self.has_rotation = preparation_params.rotation_probability > 0.0
        self.has_flip = preparation_params.flip_probability > 0.0
        self.has_noise = _noise_is_effective(preparation_params)

        self.has_any_aug = (
            self.has_rotation or self.has_flip or self.has_noise
        )

        # High-level decision: do we run in "dynamic augmentation" mode?
        target_samples = int(preparation_params.target_samples)
        self.target_samples = target_samples

        if target_samples <= 0:
            raise ValueError(
                f"target_samples must be > 0, got {target_samples}"
            )

        self.dynamic_augmentation = (target_samples > self.num_patches) and self.has_any_aug

        # Per-iteration sample counts
        if self.dynamic_augmentation:
            # All originals + extra augmented copies
            self.original_samples_per_iter = self.num_patches
            self.augmented_samples_per_iter = target_samples - self.num_patches
            if self.augmented_samples_per_iter <= 0:
                # Should not happen, but guard anyway
                self.augmented_samples_per_iter = 0
                self.dynamic_augmentation = False
            self.total_samples_per_iter = self.original_samples_per_iter + self.augmented_samples_per_iter
        else:
            # Just the originals; no per-iteration augmentation
            self.original_samples_per_iter = self.num_patches
            self.augmented_samples_per_iter = 0
            self.total_samples_per_iter = self.num_patches

        # Effective batch size is *clamped* by the total samples we actually have:
        # - If total_samples_per_iter <= config.batch_size → one batch with all samples
        # - Else → multiple full batches of size config.batch_size
        config_batch_size = int(preparation_params.batch_size)
        if config_batch_size <= 0:
            raise ValueError(
                f"batch_size must be > 0, got {config_batch_size}"
            )

        self.batch_size_effective = min(
            config_batch_size, self.total_samples_per_iter
        )

        # ----- Pre-build augmentation objects (once) -----
        self.rot_aug = (
            _get_rotation_augmentation(preparation_params.rotation_probability)
            if self.has_rotation
            else None
        )
        self.flip_aug = (
            _get_flip_augmentation(preparation_params.flip_probability)
            if self.has_flip
            else None
        )
        self.noise_aug = (
            _get_noise_augmentation(
                preparation_params.noise_type,
                preparation_params.noise_scale,
                fieldin_names,
                preparation_params.noise_channels,
            )
            if self.has_noise
            else None
        )

        # ----- Stateful RNG for reproducible sampling & shuffling -----
        # Seed can be made configurable if you want.
        self.rng = tf.random.Generator.from_seed(seed)

    # ------------------------------------------------------------------
    # Core graph-mode entrypoint
    # ------------------------------------------------------------------
    @tf.function(reduce_retracing=True, jit_compile=False)
    def build_batches(self, patches: tf.Tensor) -> tf.Tensor:
        """
        Build [M, B, H, W, C] batches from [N, H, W, C] patches.

        Args:
            patches: Tensor of shape [N, H, W, C], N must equal num_patches
                     used at initialization.

        Returns:
            Tensor of shape [M, B, H, W, C], where:
                - B = batch_size_effective
                - M = floor(total_samples_per_iter / B)
        """
        # Enforce rank and spatial shape
        tf.debugging.assert_rank(
            patches, 4, message="TrainingBatchBuilder expects [N, H, W, C] patches."
        )

        patches = tf.cast(patches, self.dtype)
        patches = ensure_fixed_tensor_shape(
            patches, (self.H, self.W, self.C)
        )

        # Sanity check: number of patches must match what we were configured for
        N = tf.shape(patches)[0]
        tf.debugging.assert_equal(
            N,
            self.num_patches,
            message=(
                "[TrainingBatchBuilder] Received a different number of patches "
                "than expected from initialization."
            ),
        )

        # -----------------------------
        # 1. Build the per-iteration pool
        # -----------------------------
        if self.dynamic_augmentation:
            # Always include the N originals
            pool = patches

            # Extra augmented copies
            extras_needed = self.augmented_samples_per_iter
            if extras_needed > 0:
                # Sample indices with replacement using the stateful RNG
                idx = self.rng.uniform(
                    shape=[extras_needed],
                    maxval=N,
                    dtype=tf.int32,
                )
                extras = tf.gather(patches, idx)  # [extras_needed, H, W, C]

                # Apply augmentations only to the extras
                extras = _apply_augmentations_to_tensor(
                    extras,
                    self.rot_aug,
                    self.flip_aug,
                    self.noise_aug,
                    self.has_rotation,
                    self.has_flip,
                    self.has_noise,
                    self.dtype,
                )

                pool = tf.concat([pool, extras], axis=0)
        else:
            # No per-iteration augmentation: pool is just the original patches
            pool = patches

        # Shape check: pool size should match what we computed in __init__
        total = tf.shape(pool)[0]
        tf.debugging.assert_equal(
            total,
            self.total_samples_per_iter,
            message=(
                "[TrainingBatchBuilder] Pool size does not match "
                "total_samples_per_iter from initialization."
            ),
        )

        # -----------------------------
        # 2. Shuffle the pool (always)
        # -----------------------------
        # Use stateful RNG to generate a random permutation
        random_vals = self.rng.uniform(
            shape=[total], maxval=1.0, dtype=tf.float32
        )
        perm = tf.argsort(random_vals)
        pool = tf.gather(pool, perm)

        # -----------------------------
        # 3. Split into batches
        # -----------------------------
        training_tensor = _split_tensor_into_batches(
            pool, self.batch_size_effective
        )
        # Static inner shape hint for XLA / shape inference
        training_tensor = tf.ensure_shape(
            training_tensor,
            [None, self.batch_size_effective, self.H, self.W, self.C],
        )

        return training_tensor

    # Convenience: make the module callable
    __call__ = build_batches
