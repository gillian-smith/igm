from typing import Optional, Tuple
import tensorflow as tf

from igm.utils.math.getmag import getmag


def get_velbase_1(U: tf.Tensor, V_b: tf.Tensor) -> tf.Tensor:
    return tf.einsum("j,...jkl->...kl", V_b, U)


@tf.function(jit_compile=True)
def get_velbase(
    U: tf.Tensor, V: tf.Tensor, V_b: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return get_velbase_1(U, V_b), get_velbase_1(V, V_b)


def get_velsurf_1(U: tf.Tensor, V_s: tf.Tensor) -> tf.Tensor:
    return tf.einsum("j,...jkl->...kl", V_s, U)


@tf.function(jit_compile=True)
def get_velsurf(
    U: tf.Tensor, V: tf.Tensor, V_s: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return get_velsurf_1(U, V_s), get_velsurf_1(V, V_s)


def get_velbar_1(U: tf.Tensor, V_bar: tf.Tensor) -> tf.Tensor:
    return tf.einsum("j,...jkl->...kl", V_bar, U)


@tf.function(jit_compile=True)
def get_velbar(
    U: tf.Tensor, V: tf.Tensor, V_bar: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return get_velbar_1(U, V_bar), get_velbar_1(V, V_bar)


@tf.function(jit_compile=True)
def boundvel(velbar_mag: tf.Tensor, U: tf.Tensor, velbar_mag_max: float) -> tf.Tensor:
    return tf.where(velbar_mag >= velbar_mag_max, velbar_mag_max * (U / velbar_mag), U)


@tf.function(jit_compile=True)
def clip_max_velbar(
    U: tf.Tensor, V: tf.Tensor, V_bar: tf.Tensor, velbar_mag_max: float
) -> Tuple[tf.Tensor, tf.Tensor]:

    velbar_x, velbar_y = get_velbar(U, V, V_bar)
    velbar_mag = getmag(velbar_x, velbar_y)

    U_clipped = boundvel(velbar_mag, U, velbar_mag_max)
    V_clipped = boundvel(velbar_mag, V, velbar_mag_max)

    return U_clipped, V_clipped


@tf.function(jit_compile=True)
def get_misfit(
    U1: tf.Tensor,
    V1: tf.Tensor,
    U2: tf.Tensor,
    V2: tf.Tensor,
    V_bar: tf.Tensor,
    thk: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:

    delta_velbar_x, delta_velbar_y = get_velbar(U1 - U2, V1 - V2, V_bar)

    delta_velbar_mag = getmag(delta_velbar_x, delta_velbar_y)

    mask = tf.ones_like(delta_velbar_mag)
    if thk is not None:
        mask = tf.cast(thk > 0.0, delta_velbar_mag.dtype)

    mask_sum = tf.reduce_sum(mask)
    error_L1 = tf.reduce_sum(mask * delta_velbar_mag) / mask_sum
    error_L2 = tf.sqrt(tf.reduce_sum(mask * tf.square(delta_velbar_mag)) / mask_sum)

    return error_L1, error_L2
