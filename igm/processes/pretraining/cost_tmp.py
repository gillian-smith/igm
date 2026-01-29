#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Callable

from igm.common import State
from igm.processes.iceflow.energy.energy import iceflow_energy_UV
from igm.processes.iceflow.energy.utils import get_energy_components

from igm.processes.iceflow.utils.data_preprocessing import X_to_fieldin
import logging
logger = logging.getLogger(__name__)

def get_cost_fn(cfg, state):

    # always-needed stuff
    cfg_unified  = cfg.processes.iceflow.unified
    cfg_physics  = cfg.processes.iceflow.physics
    cfg_numerics = cfg.processes.iceflow.numerics
    energy_components = get_energy_components(cfg)

    net_inputs_names = tuple(cfg_unified.inputs)

    slidingco0 = cfg_physics.init_slidingco
    arrhenius0 = cfg_physics.init_arrhenius
    dX0 = 200.0  # default value if not provided in inputs

    missing = ("dX", "slidingco", "arrhenius")
    energy_inputs_names = net_inputs_names + tuple(n for n in missing if n not in net_inputs_names)

    # Warn ONCE (not per-iteration) when defaults will be used
    net_lower = {n.lower() for n in net_inputs_names}
    if "dx" not in net_lower:
        logger.warning("Pretraining: 'dX' not in cfg.processes.iceflow.unified.inputs -> using constant dX = state.x[1]-state.x[0].")
    if "slidingco" not in net_lower:
        logger.warning("Pretraining: 'slidingco' not in inputs -> using constant slidingco = cfg.processes.iceflow.physics.init_slidingco (%s).", slidingco0)
    if "arrhenius" not in net_lower:
        logger.warning(
            "Pretraining: 'arrhenius' not in inputs -> using constant arrhenius = cfg.processes.iceflow.physics.init_arrhenius (%s)%s.",
            arrhenius0,
            " expanded to Nz channels (dim_arrhenius=3)" if cfg_physics.dim_arrhenius == 3 else "",
        )
    
    def _safe_unpack_net(X_net: tf.Tensor) -> dict:
        # Uses static channel count (trace-time); avoids out-of-bounds strided_slice
        C = X_net.shape[-1]
        if C is None:
            # If channels are truly dynamic, you should assert instead of guessing.
            tf.debugging.assert_rank(X_net, 4, message="Expected inputs [B,Ny,Nx,C].")
            return X_to_fieldin(
                X=X_net,
                fieldin_names=list(net_inputs_names),
                dim_arrhenius=cfg_physics.dim_arrhenius,
                Nz=cfg_numerics.Nz,
            )

        field = {}
        idx = 0
        missing_in_tensor = []

        for name in net_inputs_names:
            low = name.lower()

            if low == "arrhenius" and cfg_physics.dim_arrhenius == 3:
                need = cfg_numerics.Nz
                if idx + need <= C:
                    field[name] = tf.experimental.numpy.moveaxis(
                        X_net[..., idx:idx+need], [-1], [1]
                    )  # [B,Nz,Ny,Nx]
                    idx += need
                else:
                    missing_in_tensor.append(name)
            else:
                if idx < C:
                    field[name] = X_net[..., idx]  # [B,Ny,Nx]
                    idx += 1
                else:
                    missing_in_tensor.append(name)

        if missing_in_tensor:
            logger.warning(
                "Pretraining: inputs tensor has %d channels but cfg lists more. "
                "These configured inputs are missing in the tensor and will use defaults/zeros: %s",
                C, missing_in_tensor
            )

        if idx < C:
            logger.warning(
                "Pretraining: inputs tensor has %d channels but cfg consumed %d. Extra channels will be ignored.",
                C, idx
            )

        return field

    def _make_energy_inputs(X_net: tf.Tensor) -> tf.Tensor:
        """
        X_net: [B, Ny, Nx, C_net] exactly what the network expects.
        Returns: X_energy with channels matching energy_inputs_names (and arrhenius expanded if dim_arrhenius==3).
        """

        # Parse only what is actually present in the network tensor
        field_net = _safe_unpack_net(X_net)

        shp = tf.shape(X_net)
        B, Ny, Nx = shp[0], shp[1], shp[2]
        dtype = X_net.dtype

        dX  = tf.cast(dX0, dtype)
        sc0 = tf.cast(slidingco0, dtype)
        ar0 = tf.cast(arrhenius0, dtype)

        def _chan(name: str) -> tf.Tensor:
            key = name
            low = name.lower()

            if key in field_net:
                if low == "arrhenius" and cfg_physics.dim_arrhenius == 3:
                    # field_net['arrhenius'] is [B, Nz, Ny, Nx]; pack back to [B, Ny, Nx, Nz]
                    return tf.experimental.numpy.moveaxis(field_net[key], [1], [-1])
                return field_net[key][..., tf.newaxis]  # [B,Ny,Nx,1]

            # Missing -> fill with constants
            if low == "dx":
                return tf.fill([B, Ny, Nx, 1], dX)
            if low == "slidingco":
                return tf.fill([B, Ny, Nx, 1], sc0)
            if low == "arrhenius":
                if cfg_physics.dim_arrhenius == 3:
                    return tf.fill([B, Ny, Nx, cfg_numerics.Nz], ar0)  # [B,Ny,Nx,Nz]
                return tf.fill([B, Ny, Nx, 1], ar0)

            # Safe fallback: zeros (but you may prefer to raise for unknown names)
            return tf.zeros([B, Ny, Nx, 1], dtype=dtype)

        return tf.concat([_chan(nm) for nm in energy_inputs_names], axis=-1)

    def cost_fn(U: tf.Tensor, V: tf.Tensor, input_net: tf.Tensor) -> tf.Tensor:
        input_energy = _make_energy_inputs(input_net)

        energy = iceflow_energy_UV(
            Nz=cfg_numerics.Nz,
            dim_arrhenius=cfg_physics.dim_arrhenius,
            inputs_names=list(energy_inputs_names),
            inputs=input_energy,
            U=U,
            V=V,
            discr_h=state.iceflow.discr_h,
            discr_v=state.iceflow.discr_v,
            energy_components=energy_components,
        )

        energy_mean = tf.reduce_mean(energy, axis=[1, 2, 3])
        total_energy = tf.reduce_sum(energy_mean, axis=0)

        return total_energy
    return cost_fn