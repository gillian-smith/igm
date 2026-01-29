#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from igm.utils.grad.grad import grad_xy
from igm.processes.iceflow.utils.velocities import get_velbase


def compute_vertical_velocity_legendre(cfg, state):

    vertical_discr = state.iceflow.discr_v

    sloptopgx, sloptopgy = grad_xy(state.topg, state.dX, state.dX, False, "extrapolate")

    uvelbase, vvelbase = get_velbase(state.U, state.V, vertical_discr.V_b)

    wvelbase = uvelbase * sloptopgx + vvelbase * sloptopgy  # Lagrange basis

    dUdx, _ = grad_xy(state.U, state.dX, state.dX, False)
    _, dVdy = grad_xy(state.V, state.dX, state.dX, False)

    # Lagrange basis
    WLA = (
        wvelbase[None, ...]
        - tf.tensordot(vertical_discr.V_q_int, dUdx + dVdy, axes=[[1], [0]])
        * state.thk[None, ...]
    )

    # Legendre basis
    return tf.einsum(
        "ji,jkl->ikl",
        vertical_discr.V_q,
        WLA * state.dzeta[:, None, None],
    )
