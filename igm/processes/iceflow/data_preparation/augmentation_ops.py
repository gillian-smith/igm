from __future__ import annotations
import tensorflow as tf
from typing import Tuple

from .config import create_channel_mask

from .augmentations.rotation import RotationAugmentation, RotationParams
from .augmentations.flip import FlipAugmentation, FlipParams
from .augmentations.noise import NoiseAugmentation, NoiseParams


# Singleton caches for augmentation objects
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
