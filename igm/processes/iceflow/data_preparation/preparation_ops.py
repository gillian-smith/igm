from __future__ import annotations
import math, tensorflow as tf
from typing import Tuple
from rich.theme import Theme
from rich.console import Console
from .preparation_params import create_channel_mask, _to_py_int

from .augmentations.rotation import RotationAugmentation, RotationParams
from .augmentations.flip import FlipAugmentation, FlipParams
from .augmentations.noise import NoiseAugmentation, NoiseParams
from .preparation_params import _augs_effective


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
    if _print_already_done: return
    _print_already_done = True
    console = Console(theme=data_prep_theme)
    shape = tf.shape(training_tensor)
    console.print()
    console.print("🚫 [label]DATA PREPARATION SKIPPED[/]", justify="center")
    console.print(f"[label]Reason:[/] {reason}")
    console.print(f"[label]Output Shape:[/] [value.dimensions][{_to_py_int(shape[0])}, {_to_py_int(shape[1])}, {_to_py_int(shape[2])}, {_to_py_int(shape[3])}, {_to_py_int(shape[4])}][/]")
    console.print()

def _print_tensor_dimensions(fieldin, training_tensor, effective_batch_size, prep, actual_patch_count):
    global _print_already_done
    if _print_already_done: return
    _print_already_done = True
    console = Console(theme=data_prep_theme)
    inp = tf.shape(fieldin); out = tf.shape(training_tensor)
    ih, iw, ic = _to_py_int(inp[0]), _to_py_int(inp[1]), _to_py_int(inp[2])
    nb, bs, oh, ow, oc = map(_to_py_int, [out[0], out[1], out[2], out[3], out[4]])
    total = nb * bs
    was_patched = not (ih == oh and iw == ow)
    addl = max(0, total - int(actual_patch_count))
    has_augs = _augs_effective(prep)
    console.print(); console.print("📊 [label]DATA PREPARATION SUMMARY[/]", justify="center")
    console.print(f"[label]Input:[/] [value.dimensions]{ih} × {iw} × {ic}[/] [label]→[/] [value.dimensions]{nb}[/] [value.brackets](batches)[/] × [value.dimensions]{bs}[/] [value.brackets](samples)[/] × [value.dimensions]{oh}[/] [value.brackets](height)[/] × [value.dimensions]{ow}[/] [value.brackets](width)[/] × [value.dimensions]{oc}[/] [value.brackets](inputs)[/]")
    if was_patched:
        console.print(f"[label]Patching:[/] [value.dimensions]{ih}×{iw} → {oh}×{ow}[/] [label]•[/] [value.samples]{actual_patch_count} patches[/]")
    else:
        console.print(f"[label]Patching:[/] None (dimensions preserved) [label]•[/] [value.samples]{actual_patch_count} samples[/]")
    if addl > 0:
        method_icon = "🔄" if has_augs else "📋"
        method_text = "Upsampling + Augmentation" if has_augs else "Upsampling only"
        console.print(f"[label]Generation:[/] {method_icon} [value.samples]+{addl}[/] via {method_text}")
        if has_augs:
            parts = []
            if prep.rotation_probability > 0: parts.append(f"🔄Rotation({prep.rotation_probability:.2f})")
            if prep.flip_probability > 0: parts.append(f"🔀Flip({prep.flip_probability:.2f})")
            if prep.noise_type != 'none' and prep.noise_scale > 0: parts.append(f"🎲{prep.noise_type.title()}({prep.noise_scale:.3f})")
            if parts: console.print(f"[label]Augmentations:[/] [value.augmentation]{' [label]•[/] '.join(parts)}[/]")
    else:
        console.print(f"[label]Generation:[/] None (using patches only)")
    console.print(f"[label]Total Samples:[/] [value.samples]{total}[/] [label]•[/] [label]Batch Size:[/] [value.samples]{effective_batch_size}[/]")
    console.print(f"[label]Target:[/] [value.samples]{prep.target_samples}[/]"); console.print()