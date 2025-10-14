#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict, Tuple

from igm.processes.iceflow.energy.utils import stag4h, stag2v, psia
from igm.utils.gradient.compute_gradient import compute_gradient
from .energy import EnergyComponent
from igm.processes.iceflow.vertical import VerticalDiscr


class GravityComponent(EnergyComponent):

    def __init__(self, params):
        self.params = params

    def cost(self, U, V, fieldin, vert_disc, staggered_grid):
        return cost_gravity(U, V, fieldin, vert_disc, staggered_grid, self.params)


class GravityParams(tf.experimental.ExtensionType):

    exp_glen: float
    ice_density: float
    gravity_cst: float
    force_negative_gravitational_energy: bool
    vert_basis: str


def get_gravity_params_args(cfg) -> Dict[str, Any]:

    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    return {
        "exp_glen": cfg_physics.exp_glen,
        "ice_density": cfg_physics.ice_density,
        "gravity_cst": cfg_physics.gravity_cst,
        "force_negative_gravitational_energy": cfg_physics.force_negative_gravitational_energy,
        "vert_basis": cfg_numerics.vert_basis,
    }


def cost_gravity(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict,
    vert_disc: VerticalDiscr,
    staggered_grid: bool,
    gravity_params: GravityParams,
) -> tf.Tensor:

    thk, usurf, dX = fieldin["thk"], fieldin["usurf"], fieldin["dX"]

    V_q = vert_disc.V_q
    w = vert_disc.w

    ice_density = gravity_params.ice_density
    gravity_cst = gravity_params.gravity_cst
    fnge = gravity_params.force_negative_gravitational_energy

    return _cost(
        U,
        V,
        usurf,
        dX,
        thk,
        ice_density,
        gravity_cst,
        fnge,
        V_q,
        w,
        staggered_grid,
    )


@tf.function()
def _cost(
    U,
    V,
    usurf,
    dX,
    thk,
    ice_density,
    gravity_cst,
    fnge,
    V_q,
    w,
    staggered_grid,
):

    slopsurfx, slopsurfy = compute_gradient(usurf, dX, dX, staggered_grid)

    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        thk = stag4h(thk)

    U = tf.einsum("ij,bjkl->bikl", V_q, U)
    V = tf.einsum("ij,bjkl->bikl", V_q, V)

    uds = U * slopsurfx[:, None, :, :] + V * slopsurfy[:, None, :, :]

    if fnge:
        uds = tf.minimum(uds, 0.0)  # force non-postiveness

    #    uds = tf.where(thk[:, None, :, :]>0, uds, 0.0)

    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    return (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * thk
        * tf.reduce_sum(w[None, :, None, None] * uds, axis=1)
    )
