#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf

from igm.utils.math.precision import _normalize_precision

def cnn(cfg, nb_inputs, nb_outputs, input_normalizer=None):
    """
    Routine serve to build a convolutional neural network
    """

    precision = cfg.processes.iceflow.numerics.precision
    dtype = _normalize_precision(precision)

    inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs], dtype=dtype)

    if input_normalizer is not None:
        inputs = input_normalizer(inputs)

    use_batch_norm = (
        hasattr(cfg.processes.iceflow.emulator.network, "batch_norm")
        and cfg.processes.iceflow.emulator.network.batch_norm
    )

    use_residual = (
        hasattr(cfg.processes.iceflow.emulator.network, "residual")
        and cfg.processes.iceflow.emulator.network.residual
    )

    use_separable = (
        hasattr(cfg.processes.iceflow.emulator.network, "separable")
        and cfg.processes.iceflow.emulator.network.separable
    )

    if hasattr(cfg.processes.iceflow.emulator.network, "l2_reg"):
        kernel_regularizer = tf.keras.regularizers.l2(
            cfg.processes.iceflow.emulator.network.l2_reg
        )
    else:
        kernel_regularizer = None

    conv = inputs

    if cfg.processes.iceflow.emulator.network.activation.lower() == "leakyrelu":
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        activation = tf.keras.layers.Activation(
            cfg.processes.iceflow.emulator.network.activation
        )

    for i in range(int(cfg.processes.iceflow.emulator.network.nb_layers)):

        residual_in = conv

        if use_separable:
            conv = tf.keras.layers.SeparableConv2D(
                filters=cfg.processes.iceflow.emulator.network.nb_out_filter,
                kernel_size=(cfg.processes.iceflow.emulator.network.conv_ker_size,) * 2,
                depthwise_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
                pointwise_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
                padding="same",
                depthwise_regularizer=kernel_regularizer,
                pointwise_regularizer=kernel_regularizer,
                dtype=dtype,
            )(conv)

        else:
            conv = tf.keras.layers.Conv2D(
                filters=cfg.processes.iceflow.emulator.network.nb_out_filter,
                kernel_size=(
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                ),
                kernel_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
                padding="same",
                kernel_regularizer=kernel_regularizer,
                dtype=dtype,
            )(conv)

        if use_batch_norm:
            conv = tf.keras.layers.BatchNormalization(dtype=dtype)(conv)

        conv = activation(conv)

        if cfg.processes.iceflow.emulator.network.dropout_rate > 0:
            conv = tf.keras.layers.Dropout(
                cfg.processes.iceflow.emulator.network.dropout_rate
            )(conv)

        if use_residual and i % 2 == 1 and conv.shape[-1] == residual_in.shape[-1]:
            conv = tf.keras.layers.Add()([conv, residual_in])

    if cfg.processes.iceflow.emulator.network.cnn3d_for_vertical:

        conv = tf.expand_dims(conv, axis=1)

        for i in range(int(np.log(cfg.processes.iceflow.numerics.Nz) / np.log(2))):

            conv = tf.keras.layers.Conv3D(
                filters=cfg.processes.iceflow.emulator.network.nb_out_filter
                / (2 ** (i + 1)),
                kernel_size=(
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                ),
                padding="same",
                dtype=dtype,
            )(conv)

            conv = tf.keras.layers.UpSampling3D(size=(2, 1, 1))(conv)

        conv = tf.transpose(
            tf.concat([conv[:, :, :, :, 0], conv[:, :, :, :, 1]], axis=1),
            perm=[0, 2, 3, 1],
        )

    outputs = conv

    outputs = tf.keras.layers.Conv2D(
        filters=nb_outputs,
        kernel_size=(
            1,
            1,
        ),
        kernel_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
        activation=None,
        dtype=dtype,
    )(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def unet(cfg, nb_inputs, nb_outputs):
    """
    Routine serve to define a UNET network from keras_unet_collection
    """

    from keras_unet_collection import models

    layers = np.arange(int(cfg.processes.iceflow.emulator.network.nb_blocks))

    number_of_filters = [
        cfg.processes.iceflow.emulator.network.nb_out_filter * 2 ** (layers[i])
        for i in range(len(layers))
    ]

    return models.unet_2d(
        (None, None, nb_inputs),
        number_of_filters,
        n_labels=nb_outputs,
        stack_num_down=2,
        stack_num_up=2,
        activation=cfg.processes.iceflow.emulator.network.activation,
        output_activation=None,
        batch_norm=False,
        pool="max",
        unpool=False,
        name="unet",
    )

class FixedAffineNormalizer(tf.keras.layers.Layer):
    """
    Fixed, non-adapting per-channel affine transform.

    Applies, for each input channel c:

        x_norm[..., c] = (x[..., c] - offsets[c]) / scales[c]

    There is no `adapt` step and the layer is non-trainable.
    """

    def __init__(self, scales, offsets=None, dtype=tf.float32, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        # Convert to tensors; we let broadcasting handle dimensions.
        self.scales = tf.constant(scales, dtype=dtype)
        if offsets is None:
            offsets = np.zeros_like(scales)
        self.offsets = tf.constant(offsets, dtype=dtype)

    def call(self, x):
        # x has shape [N, H, W, C]; scales/offsets are [C].
        return (x - self.offsets) / self.scales


def build_norm_layer(cfg, nb_inputs, scales, offsets=None):

    if len(scales) != nb_inputs:
        raise ValueError(
            f"Expected input scales of length {nb_inputs}, got {len(scales)}"
        )

    precision = cfg.processes.iceflow.numerics.precision
    dtype = _normalize_precision(precision)  # e.g., tf.float32 or tf.float64
    np_dtype = dtype.as_numpy_dtype

    scales = np.asarray(scales, dtype=np_dtype)

    if np.any(scales == 0.0):
        raise ValueError("All input scales must be non-zero.")

    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np_dtype)
        if offsets.shape != scales.shape:
            raise ValueError(
                f"Offsets and scales must have the same shape; "
                f"got {offsets.shape} and {scales.shape}."
            )
    else:
        offsets = np.zeros_like(scales)

    # Just wrap the constants in a non-trainable layer.
    return FixedAffineNormalizer(scales=scales, offsets=offsets, dtype=dtype)

