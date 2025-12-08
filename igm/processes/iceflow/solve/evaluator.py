#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict
from omegaconf import DictConfig

from igm.common.core import State
from igm.processes.iceflow.utils.velocities import (
    get_velbase,
    get_velsurf,
    get_velbar,
    clip_max_velbar,
)


class EvaluatorParams(tf.experimental.ExtensionType):
    Nz: int
    force_max_velbar: float
    dim_arrhenius: int


def get_evaluator_params_args(cfg: DictConfig) -> Dict[str, Any]:

    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    return {
        "dim_arrhenius": cfg_physics.dim_arrhenius,
        "Nz": cfg_numerics.Nz,
        "force_max_velbar": cfg.processes.iceflow.force_max_velbar,
    }


def get_kwargs_from_state(state: State) -> Dict[str, Any]:

    return {
        "thk": state.thk,
        "V_bar": state.iceflow.vertical_discr.V_bar,
        "V_b": state.iceflow.vertical_discr.V_b,
        "V_s": state.iceflow.vertical_discr.V_s,
    }


def get_evaluator_inputs_from_state(_: DictConfig, state: State) -> tf.Tensor:

    return tf.stack([state.U, state.V], axis=0)


@tf.function(jit_compile=False)
# @tf.function(jit_compile=True)
def evaluator_iceflow(
    inputs: tf.Tensor, parameters: EvaluatorParams, **kwargs: Dict[str, Any]
) -> Dict[str, tf.Tensor]:

    # Compute velocity from mapping
    U, V = tf.unstack(inputs, axis=0)
    # U, V = U[0], V[0]

    # Post-processing of velocity fields
    U = tf.where(kwargs["thk"] > 0.0, U, 0.0)
    V = tf.where(kwargs["thk"] > 0.0, V, 0.0)

    if parameters.force_max_velbar > 0.0:
        U, V = clip_max_velbar(U, V, kwargs["V_bar"], parameters.force_max_velbar)

    # Retrieve derived quantities from velocity fields
    uvelbase, vvelbase = get_velbase(U, V, kwargs["V_b"])
    uvelsurf, vvelsurf = get_velsurf(U, V, kwargs["V_s"])
    ubar, vbar = get_velbar(U, V, kwargs["V_bar"])

    return {
        "U": U,
        "V": V,
        "uvelbase": uvelbase,
        "vvelbase": vvelbase,
        "uvelsurf": uvelsurf,
        "vvelsurf": vvelsurf,
        "ubar": ubar,
        "vbar": vbar,
    }


def evaluate_iceflow(cfg: DictConfig, state: State) -> None:

    print("eval?")
    # Get inputs for mapping
    inputs = get_evaluator_inputs_from_state(cfg, state)

    # Get kwargs for evaluator
    kwargs = get_kwargs_from_state(state)

    # Evaluate ice-flow model
    evaluator_params = state.iceflow.evaluator_params
    update = evaluator_iceflow(inputs, evaluator_params, **kwargs)

    # Update velocity state
    for key, value in update.items():
        setattr(state, key, value)
