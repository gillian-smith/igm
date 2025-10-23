# Tensor-based data preprocessing for IGM training.
# This module provides tensor-based alternatives to the tf.data.Dataset approach
# for better XLA compatibility and consistent compilation behavior.

from __future__ import annotations

from typing import Any, Dict, Tuple

import math
import tensorflow as tf

from .augmentations.rotation import RotationAugmentation, RotationParams
from .augmentations.flip import FlipAugmentation, FlipParams
from .augmentations.noise import NoiseAugmentation, NoiseParams
from rich.theme import Theme
from rich.console import Console


# -------------------------
# Local types & small utils
# -------------------------

class PreparationParams(tf.experimental.ExtensionType):
    overlap: float      
    batch_size: int
    patch_size: int
    rotation_probability: float
    flip_probability: float
    noise_type: str
    noise_scale: float
    target_samples: int
    fieldin_names: Tuple[str, ...]
    precision: str                     # 'single' or 'double'
    noise_channels: Tuple[str, ...]
    skip_preparation: bool

def get_input_params_args(cfg) -> Dict[str, Any]:

    cfg_data_preparation = cfg.processes.iceflow.unified.data_preparation

    return {
        "overlap": cfg_data_preparation.overlap,
        "batch_size": cfg_data_preparation.batch_size,
        "patch_size": cfg_data_preparation.patch_size,
        "rotation_probability": cfg_data_preparation.rotation_probability,
        "flip_probability": cfg_data_preparation.flip_probability,
        "noise_type": cfg_data_preparation.noise_type,
        "noise_scale": cfg_data_preparation.noise_scale,
        "target_samples": cfg_data_preparation.target_samples,
        "fieldin_names": cfg.processes.iceflow.emulator.fieldin,
        "precision": cfg.processes.iceflow.numerics.precision,
        "noise_channels": _determine_noise_channels(cfg),
        "skip_preparation": _should_skip_preparation(cfg),
    }

def _determine_noise_channels(cfg) -> Tuple[str, ...]:
    """
    Determine which channels should have noise applied based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        Tuple[str, ...]: Channel names to apply noise to
    """
    if hasattr(cfg.processes, "data_assimilation"):
        # Use control_list fields that are suitable for noise
        noise_channels = []
        for f in cfg.processes.data_assimilation.control_list:
            if f in ["thk", "usurf"]:  # Only apply noise to these fields
                noise_channels.append(f)
        # Convert to tuple for tf.experimental.ExtensionType compatibility
        return tuple(noise_channels) if noise_channels else ("thk", "usurf")
    else:
        # Default noise channels
        return ("thk", "usurf")

def calculate_expected_dimensions(
    input_height: int,
    input_width: int,
    preparation_params: PreparationParams,
) -> tuple:
    """
    Calculate the expected Ny, Nx, num_patches, and effective_batch_size for given input dimensions.
    """

    patch_size = preparation_params.patch_size
    overlap = preparation_params.overlap
    batch_size = preparation_params.batch_size
    target_samples = preparation_params.target_samples

    if patch_size > input_width or patch_size > input_height:
        return (
            input_height,  # Ny = original height
            input_width,   # Nx = original width
            1,             # num_patches = 1
            min(batch_size, max(target_samples, 1)),  # effective_batch_size
            max(target_samples, 1),  # adjusted_target_samples
        )

    # Calculate patch counts
    height_f = float(input_height)
    width_f = float(input_width)
    patch_size_f = float(patch_size)

    # Calculate minimum stride and number of patches
    min_stride = int(patch_size_f * (1.0 - overlap))

    # Calculate number of patches in each direction
    n_patches_y = max(
        1, int(math.ceil((height_f - patch_size_f) / float(min_stride))) + 1
    )
    n_patches_x = max(
        1, int(math.ceil((width_f - patch_size_f) / float(min_stride))) + 1
    )

    num_patches = n_patches_y * n_patches_x

    # Calculate final dimensions and batch size
    # Patch dimensions are always patch_size x patch_size
    Ny = patch_size
    Nx = patch_size

    # Calculate adjusted target samples (must be at least num_patches)
    adjusted_target_samples = max(target_samples, num_patches)

    # Calculate effective batch size (min of batch_size and adjusted_target_samples)
    effective_batch_size = min(batch_size, adjusted_target_samples)

    return Ny, Nx, num_patches, effective_batch_size, adjusted_target_samples


def _should_skip_preparation(cfg) -> bool:
    """
    Determine if data preparation should be skipped.
    """
    return (
        cfg.processes.iceflow.method == "unified" and 
        cfg.processes.iceflow.unified.mapping == "identity"
    )

def create_channel_mask(fieldin_names, noise_channels=None) -> tf.Tensor:
    """
    Create a boolean mask indicating which channels should have noise applied.

    Args:
        fieldin_names: Sequence of field names (e.g., ('thk','usurf',...))
        noise_channels: Sequence of channel names to apply noise to. If None, applies to ('thk', 'usurf').

    Returns:
        tf.Tensor: Boolean mask of shape [num_channels], dtype=bool
    """
    if noise_channels is None:
        noise_channels = ("thk", "usurf")
    mask = [name in noise_channels for name in fieldin_names]
    return tf.constant(mask, dtype=tf.bool)


def _to_py_int(x) -> int:
    """Convert a scalar Tensor or Python numeric to a Python int (eager-safe)."""
    if isinstance(x, tf.Tensor):
        return int(x.numpy())
    return int(x)


# -------------------------
# Console / theming (prints)
# -------------------------

data_prep_theme = Theme(
    {
        "label": "bold #e5e7eb",
        "value.samples": "#f59e0b",
        "value.dimensions": "#06b6d4",
        "value.augmentation": "#a78bfa",
        "value.brackets": "italic #64748b",
        "bar.complete": "#22c55e",
    }
)

# Module-level flag to ensure print functions are only called once
_print_already_done = False

def _print_skip_message(training_tensor: tf.Tensor, reason: str = "Identity mapping"):
    """
    Print a minimal message when data preparation is skipped.
    Converts dynamic shapes to Python ints to avoid Tensor repr in logs.
    Only prints on first call to avoid cluttering output during optimization.
    """
    global _print_already_done
    if _print_already_done:
        return
    _print_already_done = True

    console = Console(theme=data_prep_theme)

    shape = tf.shape(training_tensor)
    num_batches = _to_py_int(shape[0])
    batch_size = _to_py_int(shape[1])
    height = _to_py_int(shape[2])
    width = _to_py_int(shape[3])
    channels = _to_py_int(shape[4])

    console.print()
    console.print("🚫 [label]DATA PREPARATION SKIPPED[/]", justify="center")
    console.print(f"[label]Reason:[/] {reason}")
    console.print(
        f"[label]Output Shape:[/] [value.dimensions][{num_batches}, {batch_size}, {height}, {width}, {channels}][/]"
    )
    console.print()


def _print_tensor_dimensions(
    fieldin: tf.Tensor,
    training_tensor: tf.Tensor,
    effective_batch_size: int,
    preparation_params: PreparationParams,
    actual_patch_count: int,
):
    """
    Compare input fieldin dimensions with output training tensor dimensions and
    explain the transformation (patching vs augmentation).
    Only prints on first call to avoid cluttering output during optimization.
    """
    global _print_already_done
    if _print_already_done:
        return
    _print_already_done = True

    console = Console(theme=data_prep_theme)

    # Input dimensions
    input_shape = tf.shape(fieldin)
    input_h = _to_py_int(input_shape[0])
    input_w = _to_py_int(input_shape[1])
    input_c = _to_py_int(input_shape[2])

    # Output dimensions
    output_shape = tf.shape(training_tensor)
    num_batches = _to_py_int(output_shape[0])
    batch_size = _to_py_int(output_shape[1])
    output_h = _to_py_int(output_shape[2])
    output_w = _to_py_int(output_shape[3])
    output_c = _to_py_int(output_shape[4])

    total_samples = num_batches * batch_size

    # Determine the source of dimensional changes
    was_patched = not (input_h == output_h and input_w == output_w)

    # Additional samples beyond actual patches
    additional_samples = max(0, total_samples - int(actual_patch_count))

    has_augmentations = _augs_effective(preparation_params)

    console.print()
    console.print("📊 [label]DATA PREPARATION SUMMARY[/]", justify="center")

    # Input/Output comparison
    console.print(
        f"[label]Input:[/] [value.dimensions]{input_h} × {input_w} × {input_c}[/] "
        f"[label]→[/] [value.dimensions]{num_batches}[/] [value.brackets](batches)[/] "
        f"[value.dimensions]× {batch_size}[/] [value.brackets](samples)[/] "
        f"[value.dimensions]× {output_h}[/] [value.brackets](height)[/] "
        f"[value.dimensions]× {output_w}[/] [value.brackets](width)[/] "
        f"[value.dimensions]× {output_c}[/] [value.brackets](inputs)[/]"
    )

    if was_patched:
        console.print(
            f"[label]Patching:[/] [value.dimensions]{input_h}×{input_w} → {output_h}×{output_w}[/] "
            f"[label]•[/] [value.samples]{actual_patch_count} patches[/]"
        )
    else:
        console.print(
            f"[label]Patching:[/] None (dimensions preserved) "
            f"[label]•[/] [value.samples]{actual_patch_count} samples[/]"
        )

    # Sample generation summary
    if additional_samples > 0:
        method_icon = "🔄" if has_augmentations else "📋"
        method_text = "Upsampling + Augmentation" if has_augmentations else "Upsampling only"
        console.print(
            f"[label]Generation:[/] {method_icon} [value.samples]+{additional_samples}[/] via {method_text}"
        )

        if has_augmentations:
            aug_parts = []
            if preparation_params.rotation_probability > 0:
                aug_parts.append(f"🔄Rotation({preparation_params.rotation_probability:.2f})")
            if preparation_params.flip_probability > 0:
                aug_parts.append(f"🔀Flip({preparation_params.flip_probability:.2f})")
            if preparation_params.noise_type != "none" and preparation_params.noise_scale > 0:
                aug_parts.append(
                    f"🎲{preparation_params.noise_type.title()}({preparation_params.noise_scale:.3f})"
                )
            aug_str = " [label]•[/] ".join(aug_parts)
            console.print(f"[label]Augmentations:[/] [value.augmentation]{aug_str}[/]")
    else:
        console.print(f"[label]Generation:[/] None (using patches only)")

    # Final simple stats 
    console.print(
        f"[label]Total Samples:[/] [value.samples]{total_samples}[/] "
        f"[label]•[/] [label]Batch Size:[/] [value.samples]{effective_batch_size}[/]"
    )
    console.print(f"[label]Target:[/] [value.samples]{preparation_params.target_samples}[/]")
    console.print()


# ----------------
# Core logic
# ----------------

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

    training_samples = patches

    # Decide whether to upsample 
    apply_augmentations_effective = _augs_effective(preparation_params)
    should_upsample = _should_upsample_tensor(
        preparation_params=preparation_params,
        num_patches=actual_patch_count,
        apply_augmentations=apply_augmentations_effective,
    )

    # If we need more samples and we will augment, replicate first (no augmentation here)
    if should_upsample:
        training_samples, adjusted_target_samples = _upsample_tensor(
            training_samples,
            preparation_params.target_samples,
            preparation_params,
        )
    else:
        adjusted_target_samples = actual_patch_count

    # Apply exactly one augmentation pass if enabled (now over the full pool)
    if _augs_effective(preparation_params):
        # Get pre-created singleton augmentation objects
        has_rotation = _has_rotation(preparation_params)
        has_flip = _has_flip(preparation_params)
        has_noise = _has_noise(preparation_params)
        
        rotation_aug = _get_rotation_augmentation(preparation_params.rotation_probability) if has_rotation else None
        flip_aug = _get_flip_augmentation(preparation_params.flip_probability) if has_flip else None
        noise_aug = None
        
        if has_noise:
            channel_mask = create_channel_mask(
                preparation_params.fieldin_names, preparation_params.noise_channels
            )
            noise_aug = _get_noise_augmentation(
                preparation_params.noise_type,
                preparation_params.noise_scale,
                channel_mask
            )
        
        training_samples = _apply_augmentations_to_tensor(
            training_samples,
            rotation_aug,
            flip_aug,
            noise_aug,
            has_rotation,
            has_flip,
            has_noise,
            dtype,
        )

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


def _single_full_image_patch(fieldin: tf.Tensor, prep: PreparationParams) -> bool:
    """True iff patch_size covers both dimensions, i.e., patching yields a single full-image patch."""
    shape = tf.shape(fieldin)
    h = _to_py_int(shape[0])
    w = _to_py_int(shape[1])
    return (prep.patch_size >= h) and (prep.patch_size >= w)


def _noise_is_effective(prep: PreparationParams) -> bool:
    """True iff noise would actually alter any channel."""
    if prep.noise_type == "none" or prep.noise_scale <= 0:
        return False
    mask = create_channel_mask(prep.fieldin_names, prep.noise_channels)
    return bool(tf.reduce_any(mask).numpy())


def _has_rotation(prep: PreparationParams) -> bool:
    """Check if rotation augmentation is enabled."""
    return prep.rotation_probability > 0.0


def _has_flip(prep: PreparationParams) -> bool:
    """Check if flip augmentation is enabled."""
    return prep.flip_probability > 0.0


def _has_noise(prep: PreparationParams) -> bool:
    """Check if noise augmentation is enabled (config-level check)."""
    return prep.noise_type != "none" and prep.noise_scale > 0.0


def _augs_effective(prep: PreparationParams) -> bool:
    """
    Python-only check: will any aug actually change data?
    Uses mask-aware noise check; suitable for decisions outside @tf.function.
    """
    if _has_rotation(prep):
        return True
    if _has_flip(prep):
        return True
    return _noise_is_effective(prep)


def _should_upsample_tensor(
    preparation_params: PreparationParams, num_patches: int, apply_augmentations: bool
) -> bool:
    """
    Pure-Python decision to keep tracing out of the graph.
    Upsample only if target_samples > num_patches AND augmentations are enabled.
    (Otherwise we'd create identical copies, which we avoid.)
    """
    if preparation_params.target_samples <= num_patches:
        return False
    if not apply_augmentations:
        # Avoid duplicating identical samples without augmentations.
        return False
    return True


def _calculate_effective_batch_size(
    preparation_params: PreparationParams, adjusted_target_samples: int
) -> int:
    """
    Calculate the effective batch size based on parameters and available samples.
    Returns a Python int to preserve original behavior.
    """
    return min(preparation_params.batch_size, adjusted_target_samples)


# Module-level singleton augmentation objects (created once, reused always)
# This completely avoids object creation overhead and retracing issues
_ROTATION_AUGMENTATIONS = {}  # Cache by probability
_FLIP_AUGMENTATIONS = {}  # Cache by probability  
_NOISE_AUGMENTATIONS = {}  # Cache by (type, scale, channel_mask_hash)


def _get_rotation_augmentation(probability: float):
    """Get or create rotation augmentation singleton."""
    if probability not in _ROTATION_AUGMENTATIONS:
        _ROTATION_AUGMENTATIONS[probability] = RotationAugmentation(
            RotationParams(probability=probability)
        )
    return _ROTATION_AUGMENTATIONS[probability]


def _get_flip_augmentation(probability: float):
    """Get or create flip augmentation singleton."""
    if probability not in _FLIP_AUGMENTATIONS:
        _FLIP_AUGMENTATIONS[probability] = FlipAugmentation(
            FlipParams(probability=probability)
        )
    return _FLIP_AUGMENTATIONS[probability]


def _get_noise_augmentation(noise_type: str, noise_scale: float, channel_mask: tf.Tensor):
    """Get or create noise augmentation singleton."""
    # Create hashable key from parameters
    # Note: channel_mask should be stable for same fieldin_names + noise_channels
    mask_hash = hash(tuple(channel_mask.numpy().tolist()))
    cache_key = (noise_type, noise_scale, mask_hash)
    
    if cache_key not in _NOISE_AUGMENTATIONS:
        _NOISE_AUGMENTATIONS[cache_key] = NoiseAugmentation(
            NoiseParams(
                noise_type=noise_type,
                noise_scale=noise_scale,
                channel_mask=channel_mask,
            )
        )
    return _NOISE_AUGMENTATIONS[cache_key]


@tf.function
def _apply_augmentations_to_tensor(
    tensor: tf.Tensor,
    rotation_aug,
    flip_aug,
    noise_aug,
    has_rotation: bool,
    has_flip: bool,
    has_noise: bool,
    dtype: tf.DType,
) -> tf.Tensor:
    """
    Apply augmentations to tensor using pre-created augmentation objects.
    
    Uses @tf.function for performance while avoiding object creation inside the graph.
    Augmentation objects are created once at module level and reused.
    
    Args:
        tensor: Input tensor of shape [num_samples, height, width, channels]
        rotation_aug: Pre-created RotationAugmentation object (or None)
        flip_aug: Pre-created FlipAugmentation object (or None)
        noise_aug: Pre-created NoiseAugmentation object (or None)
        has_rotation: Whether to apply rotation
        has_flip: Whether to apply flip
        has_noise: Whether to apply noise
        dtype: Target dtype for output tensor
        
    Returns:
        Augmented tensor with same shape as input
    """
    
    def apply_to_sample(x):
        """Apply all enabled augmentations sequentially to a single sample."""
        # Apply rotation if enabled
        if has_rotation:
            x = rotation_aug.apply(x)
        
        # Apply flip if enabled
        if has_flip:
            x = flip_aug.apply(x)
        
        # Apply noise if enabled
        if has_noise:
            x = noise_aug.apply(x)
        
        return tf.cast(x, dtype)
    
    # Vectorize across the leading (sample) dimension
    return tf.vectorized_map(apply_to_sample, tensor)


def _upsample_tensor(
    tensor: tf.Tensor,
    target_samples: int,
    preparation_params: PreparationParams,
) -> Tuple[tf.Tensor, int]:
    """
    Upsample tensor to reach target number of samples by generating replicated samples.
    NOTE: No augmentation is performed here. Augmentation is applied exactly once
    later to the whole pool if enabled.

    Returns:
        (final_tensor, adjusted_target_samples:int)
    """
    current_shape = tf.shape(tensor)
    num_samples = _to_py_int(current_shape[0])

    # Ensure target not below existing count (pure Python)
    adjusted_target_samples = max(int(target_samples), num_samples)

    # If we already have enough, truncate and return
    if num_samples >= adjusted_target_samples:
        return tensor[:adjusted_target_samples], adjusted_target_samples

    # Extras needed
    extras_needed = adjusted_target_samples - num_samples

    # Replicate originals once to cover extras, slice exact count
    reps = math.ceil(extras_needed / num_samples)
    replicated = tf.tile(tensor, [reps + 1, 1, 1, 1])[:extras_needed]

    final_tensor = tf.concat([tensor, replicated], axis=0)
    return final_tensor, adjusted_target_samples


@tf.function
def _split_tensor_into_batches(tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
    """
    Split tensor into batches with shape [num_batches, batch_size, height, width, channels].
    Drops the remainder to keep full batches only.
    """
    tf.debugging.assert_greater(batch_size, 0, message="batch_size must be > 0")

    total = tf.shape(tensor)[0]
    num_batches = total // batch_size
    samples_to_use = num_batches * batch_size
    trimmed = tensor[:samples_to_use]

    shape_rest = tf.shape(trimmed)[1:]  # [H, W, C]
    h, w, c = shape_rest[0], shape_rest[1], shape_rest[2]
    return tf.reshape(trimmed, [num_batches, batch_size, h, w, c])


@tf.function
def ensure_fixed_tensor_shape(
    tensor: tf.Tensor, expected_shape: Tuple[int, int, int]
) -> tf.Tensor:
    """
    Ensure tensor has a fixed inner shape [H, W, C] for consistent XLA compilation.
    """
    return tf.ensure_shape(
        tensor, [None, expected_shape[0], expected_shape[1], expected_shape[2]]
    )
