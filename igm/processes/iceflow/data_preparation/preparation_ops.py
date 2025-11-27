from __future__ import annotations
import math, tensorflow as tf
from typing import Tuple
from rich.theme import Theme
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from .config import create_channel_mask, _to_py_int

from .augmentations.rotation import RotationAugmentation, RotationParams
from .augmentations.flip import FlipAugmentation, FlipParams
from .augmentations.noise import NoiseAugmentation, NoiseParams
from .config import _augs_effective


# Rich logging, single-shot guard
data_prep_theme = Theme({
    "label": "bold #e5e7eb",
    "value.samples": "#f59e0b",
    "value.dimensions": "#06b6d4",
    "value.augmentation": "#a78bfa",
    "value.brackets": "italic #64748b",
    "bar.complete": "#22c55e",
})

_print_already_done = False

_ROTATION_AUGMENTATIONS = {}
_FLIP_AUGMENTATIONS = {}
_NOISE_AUGMENTATIONS = {}

def _get_rotation_augmentation(p):
    if p not in _ROTATION_AUGMENTATIONS:
        _ROTATION_AUGMENTATIONS[p] = RotationAugmentation(RotationParams(probability=p))
    return _ROTATION_AUGMENTATIONS[p]

def _get_flip_augmentation(p):
    if p not in _FLIP_AUGMENTATIONS:
        _FLIP_AUGMENTATIONS[p] = FlipAugmentation(FlipParams(probability=p))
    return _FLIP_AUGMENTATIONS[p]

def _get_noise_augmentation(noise_type: str, noise_scale: float, fieldin_names, noise_channels):
    mask = create_channel_mask(fieldin_names, noise_channels)
    key = (noise_type, float(noise_scale), tuple(mask.numpy().tolist()))
    if key not in _NOISE_AUGMENTATIONS:
        _NOISE_AUGMENTATIONS[key] = NoiseAugmentation(NoiseParams(
            noise_type=noise_type, noise_scale=noise_scale, channel_mask=mask
        ))
    return _NOISE_AUGMENTATIONS[key]

@tf.function
def _apply_augmentations_to_tensor(tensor, rotation_aug, flip_aug, noise_aug,
                                   has_rotation: bool, has_flip: bool, has_noise: bool, dtype: tf.DType):
    def apply_to_sample(x):
        if has_rotation: x = rotation_aug.apply(x)
        if has_flip:     x = flip_aug.apply(x)
        if has_noise:    x = noise_aug.apply(x)
        return tf.cast(x, dtype)
    return tf.vectorized_map(apply_to_sample, tensor)

def _create_extra_copies(tensor: tf.Tensor, target_samples: int, num_originals: int) -> Tuple[tf.Tensor, int]:
    adjusted = max(int(target_samples), num_originals)
    extras_needed = adjusted - num_originals
    if extras_needed <= 0:
        z = tf.zeros([0, tf.shape(tensor)[1], tf.shape(tensor)[2], tf.shape(tensor)[3]], dtype=tensor.dtype)
        return z, adjusted
    reps = math.ceil(extras_needed / num_originals)
    replicated = tf.tile(tensor, [reps, 1, 1, 1])[:extras_needed]
    return replicated, adjusted

@tf.function
def _split_tensor_into_batches(tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
    tf.debugging.assert_greater(batch_size, 0, message="batch_size must be > 0")
    total = tf.shape(tensor)[0]
    num_batches = total // batch_size
    trimmed = tensor[: num_batches * batch_size]
    h, w, c = tf.shape(trimmed)[1], tf.shape(trimmed)[2], tf.shape(trimmed)[3]
    return tf.reshape(trimmed, [num_batches, batch_size, h, w, c])

@tf.function
def ensure_fixed_tensor_shape(tensor: tf.Tensor, expected_shape: Tuple[int, int, int]) -> tf.Tensor:
    return tf.ensure_shape(tensor, [None, expected_shape[0], expected_shape[1], expected_shape[2]])


def _print_skip_message(training_tensor: tf.Tensor, reason: str):
    global _print_already_done
    if _print_already_done:
        return
    _print_already_done = True

    console = Console(theme=data_prep_theme)

    # Expect 5D: [num_batches, batch_size, H, W, C]
    training_tensor = tf.convert_to_tensor(training_tensor)
    shape = tf.shape(training_tensor)

    try:
        nb, bs, h, w, c = map(
            _to_py_int,
            (shape[0], shape[1], shape[2], shape[3], shape[4]),
        )
        shape_str = f"[{nb}, {bs}, {h}, {w}, {c}]"
    except Exception:
        # Fallback: don’t crash if rank is unexpected
        shape_str = str(training_tensor.shape)

    table = Table(
        show_header=False,
        border_style="green",
        expand=False,
    )
    table.add_column("Label", style="label")
    table.add_column("Value", style="value.dimensions")

    table.add_row("Reason", reason)
    table.add_row("Output shape", shape_str)

    console.print()
    console.print(
        Panel(
            table,
            title="[bold]Data Preparation Skipped[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

def _print_tensor_dimensions(
    fieldin,
    training_tensor,
    effective_batch_size: int,
    prep,
    actual_patch_count: int,
):
    """
    Rich summary of data-preparation geometry, sampling, and augmentations.

    fieldin:
        Original input field (expected shape [H, W, C])
    training_tensor:
        Final training tensor (expected shape [num_batches, batch_size, H, W, C])
    effective_batch_size:
        Actual batch size used by the sampler
    prep:
        PreparationParams instance
    actual_patch_count:
        Number of patches produced by the patcher (before any up/down-sampling)
    """
    global _print_already_done
    if _print_already_done:
        return
    _print_already_done = True

    console = Console(theme=data_prep_theme)

    # Make sure we are working with tensors
    fieldin = tf.convert_to_tensor(fieldin)
    training_tensor = tf.convert_to_tensor(training_tensor)

    inp = tf.shape(fieldin)
    out = tf.shape(training_tensor)

    # Expect fieldin: [H, W, C]
    ih, iw, ic = _to_py_int(inp[0]), _to_py_int(inp[1]), _to_py_int(inp[2])

    # Expect training_tensor: [num_batches, batch_size, H, W, C]
    nb, bs, oh, ow, oc = map(
        _to_py_int,
        (out[0], out[1], out[2], out[3], out[4]),
    )

    total = nb * bs
    was_patched = not (ih == oh and iw == ow)
    addl_from_patches = max(0, total - int(actual_patch_count))
    has_augs = _augs_effective(prep)

    # ===================== GEOMETRY TABLE =====================
    geom_table = Table(
        title="[bold cyan]Data Prep Geometry[/bold cyan]",
        show_header=False,
        border_style="green",
        title_style="bold cyan",
        expand=False,
    )
    geom_table.add_column("Label", style="label")
    geom_table.add_column("Value", style="value.dimensions")

    geom_table.add_row("Input field", f"{ih} × {iw} × {ic}")
    geom_table.add_row("Patch", f"{oh} × {ow} × {oc}")
    geom_table.add_row("Num. patches", str(actual_patch_count))
    geom_table.add_row(
        "Patching applied",
        "[green]no[/]" if not was_patched else "[bold yellow]yes[/]",
    )

    if actual_patch_count > 0:
        geom_table.caption = (
            f"[label]Patch-derived samples:[/] "
            f"[value.samples]{actual_patch_count}[/]"
        )

    # ================= SAMPLING & BATCHING TABLE ==============
    samp_table = Table(
        title="[bold cyan]Sampling & Batching[/bold cyan]",
        show_header=False,
        border_style="blue",
        title_style="bold cyan",
        expand=False,
    )
    samp_table.add_column("Label", style="label")
    samp_table.add_column("Value", style="value.samples")

    samp_table.add_row("Target samples", str(prep.target_samples))
    samp_table.add_row("Effective samples", str(total))
    samp_table.add_row("Batch size", str(effective_batch_size))
    samp_table.add_row("Num. batches", str(nb))

    # Generation / upsampling info
    if addl_from_patches > 0 or has_augs:
        if addl_from_patches > 0 and has_augs:
            gen_text = (
                f"[value.samples]+{addl_from_patches}[/] via upsampling"
                " + augmentation"
            )
        elif addl_from_patches > 0:
            gen_text = f"[value.samples]+{addl_from_patches}[/] via upsampling"
        else:  # has_augs only
            gen_text = "No extra samples; augmentations applied in-place"
        samp_table.add_row("Generation", gen_text)
    else:
        samp_table.add_row("Generation", "Using patches only")

    # Augmentation details
    if has_augs:
        aug_parts = []
        if getattr(prep, "rotation_probability", 0.0) > 0:
            aug_parts.append(
                f"🔄Rotation({prep.rotation_probability:.2f})"
            )
        if getattr(prep, "flip_probability", 0.0) > 0:
            aug_parts.append(
                f"🔀Flip({prep.flip_probability:.2f})"
            )
        if (
            getattr(prep, "noise_type", "none") != "none"
            and getattr(prep, "noise_scale", 0.0) > 0
        ):
            aug_parts.append(
                f"🎲{prep.noise_type.title()}({prep.noise_scale:.3f})"
            )

        aug_text = " [label]•[/] ".join(aug_parts) if aug_parts else "Enabled"
        samp_table.add_row("Augmentations", f"[value.augmentation]{aug_text}[/]")
    else:
        samp_table.add_row("Augmentations", "None")

    # Caption describing how effective samples relate to target
    if total == prep.target_samples:
        caption = (
            f"[label]Effective samples:[/] [value.samples]{total}[/] "
            "(matches target)"
        )
    else:
        if total > prep.target_samples:
            reason_text = "more than target (patching / upsampling)"
        else:
            reason_text = "less than target (subsampling / batching)"

        caption = (
            f"[label]Effective samples:[/] [value.samples]{total}[/] "
            f"(target [value.samples]{prep.target_samples}[/]; {reason_text})"
        )

    samp_table.caption = caption

    # ================= COMBINED PANEL OUTPUT ==================
    body = Columns([geom_table, samp_table], expand=True, equal=False)

    console.print()
    console.print(
        Panel(
            body,
            title="[bold]Data Preparation Summary[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()
