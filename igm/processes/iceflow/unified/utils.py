#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Callable

from igm.common.core import State
from igm.processes.iceflow.energy.energy import iceflow_energy_UV
from igm.processes.iceflow.energy.utils import get_energy_components
from igm.processes.iceflow.data_preparation.config import PreparationParams
from igm.processes.iceflow.data_preparation.patching import OverlapPatching
from igm.processes.iceflow.data_preparation.batch_builder import TrainingBatchBuilder
from igm.processes.iceflow.data_preparation.preparation_ops import (
    _print_skip_message,
    _print_tensor_dimensions,
)
from igm.processes.iceflow.data_preparation.config import _augs_effective


def get_cost_fn(
    cfg: DictConfig, state: State
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
    """Create cost function for ice flow optimization."""

    cfg_unified = cfg.processes.iceflow.unified
    cfg_physics = cfg.processes.iceflow.physics
    cfg_numerics = cfg.processes.iceflow.numerics

    energy_components = get_energy_components(cfg)

    def cost_fn(U: tf.Tensor, V: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        """Cost function from velocity fields and inputs."""
        nonstaggered_energy, staggered_energy = iceflow_energy_UV(
            Nz=cfg_numerics.Nz,
            dim_arrhenius=cfg_physics.dim_arrhenius,
            staggered_grid=cfg_numerics.staggered_grid,
            inputs_names=tuple(cfg_unified.inputs),
            inputs=input,
            U=U,
            V=V,
            vert_disc=state.iceflow.vertical_discr,
            energy_components=energy_components,
        )

        energy_mean_staggered = tf.reduce_mean(staggered_energy, axis=[1, 2, 3])
        energy_mean_nonstaggered = tf.reduce_mean(nonstaggered_energy, axis=[1, 2, 3])

        total_energy = tf.reduce_sum(energy_mean_nonstaggered, axis=0) + tf.reduce_sum(
            energy_mean_staggered, axis=0
        )

        return total_energy

    return cost_fn

def _print_data_preparation_summary(
    prep: PreparationParams,
    X: tf.Tensor,
    patching: OverlapPatching,
    sampler: TrainingBatchBuilder,
) -> None:
    """
    One-shot rich logging of data-preparation geometry.

    Uses:
    - X: full input field before patching
    - patching: overlap patcher (for num_patches)
    - sampler: training batch builder (for total samples / batch size)

    Relies on the global guard inside _print_tensor_dimensions / _print_skip_message,
    so this is safe to call multiple times; it will only print once.
    """

    total = sampler.total_samples_per_iter
    B = sampler.batch_size_effective
    if total <= 0 or B <= 0:
        return

    # Match behaviour of _split_tensor_into_batches: full batches only.
    num_batches = total // B
    if num_batches <= 0:
        return

    # ----- Build "fieldin" as the *original* field -----
    fieldin = tf.convert_to_tensor(X)

    # If X has a leading sample dimension, strip it: expect [H, W, C]
    if fieldin.shape.rank == 4:
        fieldin = fieldin[0]

    # Safety: we need 3D [H, W, C] for the summary logic
    if fieldin.shape.rank != 3:
        # If this ever happens, better to bail quietly than crash at import time
        return

    ih = fieldin.shape[0]
    iw = fieldin.shape[1]

    # ----- Build dummy training tensor with final batch geometry -----
    Hp, Wp, Cp = sampler.H, sampler.W, sampler.C  # patch geometry
    training_tensor = tf.zeros(
        [num_batches, B, Hp, Wp, Cp],
        dtype=fieldin.dtype,
    )

    num_patches = int(patching.num_patches)
    has_augs = _augs_effective(prep)

    # "No-op" condition:
    # - only one patch,
    # - patch covers full domain,
    # - no effective augmentations,
    # - no up/down-sampling (total == num_patches == target_samples)
    no_patching = (num_patches == 1) and (Hp == ih) and (Wp == iw)
    no_sampling_change = (total == num_patches == int(prep.target_samples))

    if no_patching and (not has_augs) and no_sampling_change:
        _print_skip_message(
            training_tensor,
            "No patching, no augmentation, and no up/down-sampling",
        )
    else:
        _print_tensor_dimensions(
            fieldin,
            training_tensor,
            sampler.batch_size_effective,
            prep,
            num_patches,
        )
