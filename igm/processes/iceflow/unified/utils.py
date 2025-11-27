#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Callable

from igm.common.core import State
from igm.processes.iceflow.energy.energy import iceflow_energy_UV
from igm.processes.iceflow.energy.utils import get_energy_components


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
