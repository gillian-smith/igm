#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.grad.grad import grad_xy, pad_x, pad_y, pad_z


def compute_strain_heat(cfg: DictConfig, state: State) -> tf.Tensor:

    cfg_physics = cfg.processes.iceflow.physics

    n = cfg_physics.exp_glen
    h_min = cfg_physics.thr_ice_thk

    dzeta = state.enthalpy.vertical_discr.dzeta
    dz = dzeta * state.thk[None, ...]

    return compute_strain_heat_tf(
        state.U,
        state.V,
        state.arrhenius,
        state.dx,
        dz,
        n,
        h_min,
    )


# TODO: correct for coordinate-following coordinates?


@tf.function
def compute_strain_heat_tf(
    U: tf.Tensor,
    V: tf.Tensor,
    arrhenius: tf.Tensor,
    dx: tf.Tensor,
    dz: tf.Tensor,
    n: tf.Tensor,
    h_min: tf.Tensor,
    mode_pad_xy: str = "symmetric",
    mode_pad_z: str = "extrapolate",
) -> tf.Tensor:

    spy = 31556926.0

    U_si = U / spy
    V_si = V / spy

    # Pad velocities in x, y, z directions
    Ui = pad_x(U_si, mode=mode_pad_xy)
    Uj = pad_y(U_si, mode=mode_pad_xy)
    Uk = pad_z(U_si, mode=mode_pad_z)

    Vi = pad_x(V_si, mode=mode_pad_xy)
    Vj = pad_y(V_si, mode=mode_pad_xy)
    Vk = pad_z(V_si, mode=mode_pad_z)

    dz_padded = pad_z(dz, mode="symmetric")
    DZ2 = dz_padded[:-1, :, :] + dz_padded[1:, :, :]

    # Compute strain rate components
    Exx = (Ui[:, :, 2:] - Ui[:, :, :-2]) / (2 * dx)
    Eyy = (Vj[:, 2:, :] - Vj[:, :-2, :]) / (2 * dx)
    Ezz = -Exx - Eyy

    Exy = 0.5 * (
        (Vi[:, :, 2:] - Vi[:, :, :-2]) / (2 * dx)
        + (Uj[:, 2:, :] - Uj[:, :-2, :]) / (2 * dx)
    )

    # Vertical shear strain rates
    Exz = 0.5 * (Uk[2:, :, :] - Uk[:-2, :, :]) / tf.maximum(DZ2, h_min)
    Eyz = 0.5 * (Vk[2:, :, :] - Vk[:-2, :, :]) / tf.maximum(DZ2, h_min)

    # Effective strain rate
    strainrate = tf.sqrt(
        0.5 * (Exx**2 + Eyy**2 + Ezz**2 + 2 * (Exy**2 + Exz**2 + Eyz**2))
    )

    # Convert arrhenius units: MPa⁻³ y⁻¹ to Pa⁻³ s⁻¹
    unit_conversion = 1e18 * spy

    dim_arrhenius = arrhenius.ndim

    arrhenius_expanded = (
        tf.expand_dims(arrhenius, axis=0) if dim_arrhenius == 2 else arrhenius
    )

    return (
        2.0
        * (arrhenius_expanded / unit_conversion) ** (-1.0 / n)
        * strainrate ** (1.0 + 1.0 / n)
    )


def compute_friction_heat(cfg: DictConfig, state: State) -> tf.Tensor:

    cfg_physics = cfg.processes.iceflow.physics
    m = cfg_physics.sliding.weertman.exponent

    return compute_friction_heat_tf(
        state.U,
        state.V,
        state.slidingco,
        state.topg,
        state.dX,
        m,
    )


@tf.function
def compute_friction_heat_tf(
    U: tf.Tensor,
    V: tf.Tensor,
    C: tf.Tensor,
    b: tf.Tensor,
    dx: tf.Tensor,
    m: tf.Tensor,
) -> tf.Tensor:

    spy = 31556926.0

    U_si = U / spy
    V_si = V / spy

    # Bed slope
    dbdx, dbdy = grad_xy(b, dx, dx, False, "extrapolate")

    # Vertical velocity component at base
    wvelbase = U_si[0] * dbdx + V_si[0] * dbdy

    # Total basal velocity
    ub = tf.sqrt(U_si[0] ** 2 + V_si[0] ** 2 + wvelbase**2)

    # Friction heating: τ·u
    # Unit conversion: MPa·m⁻¹/ⁿ·y¹/ⁿ → Pa·m/s
    return C * ub ** (1.0 / m + 1.0) * 1e6 * spy ** (1.0 / m)
