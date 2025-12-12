#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.utils.grad.grad import grad_xy


def compute_strain_heating(
    cfg: DictConfig,
    U: tf.Tensor,
    V: tf.Tensor,
    arrhenius: tf.Tensor,
    dx: float,
    dz: tf.Tensor,
) -> tf.Tensor:
    """
    Compute volumetric strain heating rate.

    Args:
        cfg: Configuration object
        U: X-velocity [m/y]
        V: Y-velocity [m/y]
        arrhenius: Arrhenius factor [MPa⁻³ y⁻¹]
        dx: Horizontal grid spacing [m]
        dz: Vertical layer thickness [m]

    Returns:
        Strain heating rate [W/m³]
    """
    cfg_enthalpy = cfg.processes.enthalpy
    cfg_physics = cfg.processes.iceflow.physics

    # Convert to m/s
    U_si = U / cfg_enthalpy.spy
    V_si = V / cfg_enthalpy.spy

    return compute_strainheat_tf(
        U_si,
        V_si,
        arrhenius,
        dx,
        dz,
        cfg_physics.exp_glen,
        cfg_physics.thr_ice_thk,
        cfg_physics.dim_arrhenius,
        cfg_enthalpy.spy,
    )


@tf.function
def compute_strainheat_tf(
    U: tf.Tensor,
    V: tf.Tensor,
    arrhenius: tf.Tensor,
    dx: float,
    dz: tf.Tensor,
    exp_glen: float,
    thr: float,
    dim_arrhenius: int,
    spy: float,
) -> tf.Tensor:
    """
    Compute volumetric strain heating rate from velocity field.

    Args:
        U: X-velocity [m/s]
        V: Y-velocity [m/s]
        arrhenius: Arrhenius factor [MPa⁻³ y⁻¹]
        dx: Horizontal grid spacing [m]
        dz: Vertical layer thickness [m]
        exp_glen: Glen's flow law exponent
        thr: Thickness threshold [m]
        dim_arrhenius: Arrhenius dimension (2 or 3)
        spy: Seconds per year [s/y]

    Returns:
        Strain heating rate [W/m³]
    """
    # Pad velocities symmetrically
    Ui = tf.pad(U, [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")
    Uj = tf.pad(U, [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")
    Uk = tf.pad(U, [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")

    Vi = tf.pad(V, [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")
    Vj = tf.pad(V, [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")
    Vk = tf.pad(V, [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")

    # Vertical spacing for derivatives
    DZ2 = tf.concat([dz[0:1], dz[:-1] + dz[1:], dz[-1:]], axis=0)

    # Compute strain rate components
    Exx = (Ui[:, :, 2:] - Ui[:, :, :-2]) / (2 * dx)
    Eyy = (Vj[:, 2:, :] - Vj[:, :-2, :]) / (2 * dx)
    Ezz = -Exx - Eyy

    Exy = 0.5 * (
        (Vi[:, :, 2:] - Vi[:, :, :-2]) / (2 * dx)
        + (Uj[:, 2:, :] - Uj[:, :-2, :]) / (2 * dx)
    )
    Exz = 0.5 * (Uk[2:, :, :] - Uk[:-2, :, :]) / tf.maximum(DZ2, thr)
    Eyz = 0.5 * (Vk[2:, :, :] - Vk[:-2, :, :]) / tf.maximum(DZ2, thr)

    # Effective strain rate
    strainrate = tf.sqrt(
        0.5 * (Exx**2 + Eyy**2 + Ezz**2 + 2 * (Exy**2 + Exz**2 + Eyz**2))
    )

    # Set to zero where layers are too thin
    strainrate = tf.where(DZ2 > 1, strainrate, 0.0)

    # Convert arrhenius units: MPa⁻³ y⁻¹ to Pa⁻³ s⁻¹
    unit_conversion = 1e18 * spy

    if dim_arrhenius == 2:
        arrhenius_expanded = tf.expand_dims(arrhenius, axis=0)
        return (arrhenius_expanded / unit_conversion) ** (
            -1.0 / exp_glen
        ) * strainrate ** (1.0 + 1.0 / exp_glen)
    else:
        return (arrhenius / unit_conversion) ** (-1.0 / exp_glen) * strainrate ** (
            1.0 + 1.0 / exp_glen
        )


def compute_friction_heating(
    cfg: DictConfig,
    U: tf.Tensor,
    V: tf.Tensor,
    slidingco: tf.Tensor,
    topg: tf.Tensor,
    dx: float,
) -> tf.Tensor:
    """
    Compute basal friction heating rate.

    Args:
        cfg: Configuration object
        U: X-velocity [m/y]
        V: Y-velocity [m/y]
        slidingco: Sliding coefficient [MPa·m⁻¹/ⁿ·y¹/ⁿ]
        topg: Bed topography [m]
        dx: Horizontal grid spacing [m]

    Returns:
        Friction heating rate [W/m²]
    """
    cfg_enthalpy = cfg.processes.enthalpy
    cfg_physics = cfg.processes.iceflow.physics

    # Convert to m/s
    U_si = U / cfg_enthalpy.spy
    V_si = V / cfg_enthalpy.spy

    return compute_frictheat_tf(
        U_si,
        V_si,
        slidingco,
        topg,
        dx,
        cfg_physics.sliding.weertman.exponent,
        cfg_enthalpy.spy,
    )


@tf.function
def compute_frictheat_tf(
    U: tf.Tensor,
    V: tf.Tensor,
    slidingco: tf.Tensor,
    topg: tf.Tensor,
    dx: float,
    exp_weertman: float,
    spy: float,
) -> tf.Tensor:
    """
    Compute basal friction heating from sliding velocity.

    Args:
        U: X-velocity [m/s]
        V: Y-velocity [m/s]
        slidingco: Sliding coefficient [MPa·m⁻¹/ⁿ·y¹/ⁿ]
        topg: Bed topography [m]
        dx: Horizontal grid spacing [m]
        exp_weertman: Weertman sliding law exponent
        spy: Seconds per year [s/y]

    Returns:
        Friction heating rate [W/m²]
    """
    # Bed slope
    sloptopgx, sloptopgy = grad_xy(topg, dx, dx, False, "extrapolate")

    # Vertical velocity component at base
    wvelbase = U[0] * sloptopgx + V[0] * sloptopgy

    # Total basal velocity
    ub = tf.sqrt(U[0] ** 2 + V[0] ** 2 + wvelbase**2)

    # Friction heating: τ·u
    # Unit conversion: MPa·m⁻¹/ⁿ·y¹/ⁿ → Pa·m/s
    return (
        (slidingco * 1e6)
        * (spy) ** (1.0 / exp_weertman)
        * ub ** (1.0 / exp_weertman + 1)
    )
