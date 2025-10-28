# Tensor-based data preprocessing for IGM training.
# This module provides tensor-based alternatives to the tf.data.Dataset approach
# for better XLA compatibility and consistent compilation behavior.

from __future__ import annotations

from typing import Any, Dict, Tuple

import math
import tensorflow as tf

from .preparation_params import ( 
    PreparationParams, _single_full_image_patch, _augs_effective,
    _has_rotation, _has_flip, _has_noise, _calculate_effective_batch_size, 
    _to_py_int, _should_upsample_tensor
)

from .preparation_ops import (
    _get_rotation_augmentation, _get_flip_augmentation, _get_noise_augmentation,
    _apply_augmentations_to_tensor, _create_extra_copies,
    _split_tensor_into_batches, ensure_fixed_tensor_shape,
    _print_skip_message, _print_tensor_dimensions
)


def _create_single_batch_tensor(fieldin: tf.Tensor) -> tf.Tensor:
    """Create a training tensor with single batch and single sample from fieldin."""
    return tf.expand_dims(tf.expand_dims(fieldin, 0), 0)


def create_input_tensor_from_fieldin(
    fieldin: tf.Tensor, patching, preparation_params: PreparationParams
) -> tf.Tensor:
    """
    Create a training tensor from field input by first splitting into patches, then applying augmentations.
    Returns a tensor instead of a tf.data.Dataset for consistent XLA compilation.

    IMPORTANT: Augmentations are applied at most once per sample.
    If augmentations are enabled, we upsample first (by replication only),
    then apply a single augmentation pass to the entire pool.

    Args:
        fieldin: Input field tensor of shape [height, width, channels]
        patching: Patching object that implements patch_tensor method
        preparation_params: Parameters containing augmentation and batching settings

    Returns:
        training_tensor: Shape [num_batches, batch_size, height, width, channels]
    """

    # Step 0: Ensure desired precision
    desired_dtype = tf.float32 if preparation_params.precision == "single" else tf.float64
    if fieldin.dtype != desired_dtype:
        fieldin = tf.cast(fieldin, desired_dtype)

    # Early exit #1: explicit skip flag
    if preparation_params.skip_preparation:
        training_tensor = _create_single_batch_tensor(fieldin)
        _print_skip_message(training_tensor, reason="Explicit skip_preparation=True")
        return training_tensor

    # Early exit #2: single full-image patch + no *effective* augmentations
    if _single_full_image_patch(fieldin, preparation_params) and not _augs_effective(preparation_params):
        training_tensor = _create_single_batch_tensor(fieldin)
        _print_skip_message(
            training_tensor,
            reason="Single full-image patch with no effective augmentations",
        )
        return training_tensor

    # Step 1: Split fieldin into patches using the provided patching object
    patches = patching.patch_tensor(fieldin)

    # Step 2: Continue with the existing processing pipeline
    patch_shape = tf.shape(patches)
    dtype = patches.dtype
    num_patches_t = patch_shape[0]
    actual_patch_count = _to_py_int(num_patches_t)  # Python int

    # Decide whether to upsample 
    apply_augmentations_effective = _augs_effective(preparation_params)
    should_upsample = _should_upsample_tensor(
        prep=preparation_params,
        num_patches=actual_patch_count,
        apply_augs=apply_augmentations_effective,
    )

    # CRITICAL: Preserve original patches without augmentation
    # If we need more samples and will augment, separate originals from copies
    if should_upsample:
        # Keep original patches untouched
        original_samples = patches
        
        # Create additional copies for augmentation
        extra_copies, adjusted_target_samples = _create_extra_copies(
            patches,
            preparation_params.target_samples,
            actual_patch_count,
        )
        
        # Apply augmentations ONLY to the extra copies
        if _augs_effective(preparation_params):
            # Get pre-created singleton augmentation objects
            has_rotation = _has_rotation(preparation_params)
            has_flip = _has_flip(preparation_params)
            has_noise = _has_noise(preparation_params)
            
            rotation_aug = _get_rotation_augmentation(preparation_params.rotation_probability) if has_rotation else None
            flip_aug = _get_flip_augmentation(preparation_params.flip_probability) if has_flip else None
            noise_aug = None
            
            if has_noise:
                noise_aug = _get_noise_augmentation(
                    preparation_params.noise_type,
                    preparation_params.noise_scale,
                    preparation_params.fieldin_names,
                    preparation_params.noise_channels
                )
            
            augmented_copies = _apply_augmentations_to_tensor(
                extra_copies,
                rotation_aug,
                flip_aug,
                noise_aug,
                has_rotation,
                has_flip,
                has_noise,
                dtype,
            )
        else:
            augmented_copies = extra_copies
        
        # Combine originals (unaugmented) with augmented copies
        training_samples = tf.concat([original_samples, augmented_copies], axis=0)
    else:
        # No upsampling needed, use patches as-is
        training_samples = patches
        adjusted_target_samples = actual_patch_count

    # Enforce static inner shape to keep XLA happy
    if patches.shape.rank == 4 and all(d is not None for d in patches.shape[1:]):
        training_samples = ensure_fixed_tensor_shape(
            training_samples, (patches.shape[1], patches.shape[2], patches.shape[3])
        )

    # Compute effective batch size as Python int (preserve original API)
    effective_batch_size = _calculate_effective_batch_size(
        preparation_params, adjusted_target_samples
    )

    # Shuffle 
    training_samples = tf.random.shuffle(training_samples)

    # Batch into [num_batches, batch_size, H, W, C]
    training_tensor = _split_tensor_into_batches(training_samples, effective_batch_size)

    _print_tensor_dimensions(
        fieldin, training_tensor, effective_batch_size, preparation_params, actual_patch_count
    )

    return training_tensor


