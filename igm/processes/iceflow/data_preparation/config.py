from __future__ import annotations
import math, warnings
import tensorflow as tf
from typing import Any, Dict, Tuple


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

def get_input_params_args(cfg) -> Dict[str, Any]:

    cfg_data_preparation = cfg.processes.iceflow.unified.data_preparation

    # Validate and clamp noise_scale to prevent negative values
    noise_scale = cfg_data_preparation.noise_scale
    if noise_scale > 1.0:
        warnings.warn(
            f"noise_scale={noise_scale:.3f} exceeds maximum safe value of 1.0. "
            f"Values greater than 1.0 can produce negative results. "
            f"Clamping to 1.0.",
            UserWarning,
            stacklevel=2
        )
        noise_scale = 1.0
    elif noise_scale < 0.0:
        warnings.warn(
            f"noise_scale={noise_scale:.3f} is negative. Clamping to 0.0.",
            UserWarning,
            stacklevel=2
        )
        noise_scale = 0.0

    return {
        "overlap": cfg_data_preparation.overlap,
        "batch_size": cfg_data_preparation.batch_size,
        "patch_size": cfg_data_preparation.patch_size,
        "rotation_probability": cfg_data_preparation.rotation_probability,
        "flip_probability": cfg_data_preparation.flip_probability,
        "noise_type": cfg_data_preparation.noise_type,
        "noise_scale": noise_scale,
        "target_samples": cfg_data_preparation.target_samples,
        "fieldin_names": cfg.processes.iceflow.emulator.fieldin,
        "precision": cfg.processes.iceflow.numerics.precision,
        "noise_channels": _determine_noise_channels(cfg),
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
        return ("thk", "usurf","arrhenius","slidingco")


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

def _has_rotation(prep): return prep.rotation_probability > 0.0
def _has_flip(prep):     return prep.flip_probability > 0.0
def _has_noise(prep):    return (prep.noise_type != "none") and (prep.noise_scale > 0.0)

def _noise_is_effective(prep: PreparationParams) -> bool:
    if prep.noise_type == "none" or prep.noise_scale <= 0: return False
    mask = create_channel_mask(prep.fieldin_names, prep.noise_channels)
    return bool(tf.reduce_any(mask).numpy())

def _augs_effective(prep: PreparationParams) -> bool:
    return _has_rotation(prep) or _has_flip(prep) or _noise_is_effective(prep)
