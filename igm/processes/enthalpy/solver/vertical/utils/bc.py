#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple


@tf.function()
def compute_bc(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    E_s: tf.Tensor,
    h_water_till: tf.Tensor,
    dEdz_dry: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    TensorFlow function to compute boundary conditions for the enthalpy solver.

    Determines the type and value of basal boundary condition based on thermal
    state (cold/temperate) and till water availability (dry/wet).

    Args:
        E: Enthalpy field (J kg^-1).
        E_pmp: Pressure melting point enthalpy (J kg^-1).
        E_s: Surface enthalpy (J kg^-1).
        h_water_till: Till water layer thickness (m).
        dEdz_dry: Enthalpy gradient for dry bed Neumann condition (J kg^-1 m^-1).

    Returns:
        Tuple of (BCB, VB, VS):
        - BCB: Basal BC type flag (1=Neumann, 0=Dirichlet) (-).
        - VB: Basal BC value (J kg^-1 for Dirichlet, J kg^-1 m^-1 for Neumann).
        - VS: Surface Dirichlet BC value (J kg^-1).
    """
    shape_2d = E_s.shape

    BASE_COLD = (E[0] < E_pmp[0]) | (h_water_till <= 0.0)
    BASE_DRY = h_water_till <= 0.0
    ICE_COLD = E[1] < E_pmp[1]

    VS = E_s

    BCB = tf.where(
        BASE_COLD,
        tf.where(BASE_DRY, tf.ones(shape_2d), tf.zeros(shape_2d)),
        tf.where(ICE_COLD, tf.zeros(shape_2d), tf.ones(shape_2d)),
    )

    VB = tf.where(
        BASE_COLD,
        tf.where(
            BASE_DRY,
            dEdz_dry,
            E_pmp[0],
        ),
        tf.where(ICE_COLD, E_pmp[0], 0.0),
    )

    return BCB, VB, VS
