#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf

from igm.utils.math.precision import _normalize_precision


def cnn_og(cfg, nb_inputs, nb_outputs, input_normalizer=None):
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

class CNNSkip(tf.keras.Model):
    """
    Simple convolutional neural network with skip connection.
    """

    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(CNNSkip, self).__init__()

        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = _normalize_precision(precision)

        self.input_normalizer = input_normalizer

        # Build convolutional layers (unchanged)
        self.conv_layers = []
        self.activations = []
        n_layers = int(cfg.processes.iceflow.emulator.network.nb_layers)
        n_filters = cfg.processes.iceflow.emulator.network.nb_out_filter

        for _ in range(n_layers):
            layer = tf.keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(5, 5),
                padding='same',
                dtype=self.dtype_model,
            )
            activation = tf.keras.layers.Activation(
                cfg.processes.iceflow.emulator.network.activation
            )
            self.conv_layers.append(layer)
            self.activations.append(activation)

        # Projection layer for skip connection (1×1 conv)
        self.skip_proj = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(1, 1),
            padding='same',
            dtype=self.dtype_model,
        )

        # Output layer (unchanged)
        self.output_layer = tf.keras.layers.Conv2D(
            filters=nb_outputs,
            kernel_size=(1, 1),
            activation=None,
            dtype=self.dtype_model,
        )

        self.build(input_shape=[None, None, None, nb_inputs])

    def call(self, inputs):
        x = inputs

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        # Store skip connection
        skip = self.skip_proj(x)

        # Main path
        for conv, activation in zip(self.conv_layers, self.activations):
            x = conv(x)
            x = activation(x)

        # Add skip connection
        x = x + skip

        # Output layer
        outputs = self.output_layer(x)

        return outputs


class CNN(tf.keras.Model):
    """
    Simple convolutional neural network
    """
    
    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(CNN, self).__init__()
        
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = _normalize_precision(precision)
        
        self.input_normalizer = input_normalizer
        
        # Build convolutional layers
        self.conv_layers = []
        self.activations = []
        for i in range(int(cfg.processes.iceflow.emulator.network.nb_layers)):
            layer = tf.keras.layers.Conv2D(
                filters=cfg.processes.iceflow.emulator.network.nb_out_filter,
                kernel_size=(5, 5),  # Simple kernels
                padding='same',
                dtype=self.dtype_model,
            )
            activation = tf.keras.layers.Activation(
                cfg.processes.iceflow.emulator.network.activation
            )
            self.conv_layers.append(layer)
            self.activations.append(activation)
        
        # Output layer
        self.output_layer = tf.keras.layers.Conv2D(
            filters=nb_outputs,
            kernel_size=(1, 1),
            activation=None,
            dtype=self.dtype_model,
        )
        
        # Build the model immediately
        self.build(input_shape=[None, None, None, nb_inputs])
    
    def call(self, inputs):
        x = inputs
        
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)
        
        # Pass through convolutional layers
        for conv, activation in zip(self.conv_layers, self.activations):
            x = conv(x)
            x = activation(x)
        
        # Output layer
        outputs = self.output_layer(x)
        
        return outputs
class MLP(tf.keras.Model):
    """
    Simple multi-layer perceptron (fully connected network)
    """
    
    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(MLP, self).__init__()
        
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = _normalize_precision(precision)
        
        self.input_normalizer = input_normalizer
        
        # Build hidden layers
        self.hidden_layers = []

        for i in range(int(cfg.processes.iceflow.emulator.network.nb_layers)):
            layer = tf.keras.layers.Dense(
                units=cfg.processes.iceflow.emulator.network.nb_out_filter,
                activation=cfg.processes.iceflow.emulator.network.activation,
                dtype=self.dtype_model,
            )
            self.hidden_layers.append(layer)
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            units=nb_outputs,
            activation=None,
            dtype=self.dtype_model,
        )
        
        self.build(input_shape=[None, None, None, nb_inputs])
    
    def call(self, inputs):
        x = inputs
        
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output layer
        outputs = self.output_layer(x)
        
        return outputs
    
    # def __repr__(self):
        # return f"mlp(num_layers={len(self.hidden_layers)}, dtype={self.dtype_model})"

class FourierMLP(tf.keras.Model):
    """
    MLP with proper coordinate handling for spatial problems
    """
    
    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(FourierMLP, self).__init__()
        
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = _normalize_precision(precision)
        
        self.input_normalizer = input_normalizer
        self.nb_inputs = nb_inputs
        
        nb_filters = cfg.processes.iceflow.emulator.network.nb_out_filter
        nb_layers = int(cfg.processes.iceflow.emulator.network.nb_layers)
        activation = cfg.processes.iceflow.emulator.network.activation
        
        # Fourier features (crucial for spatial learning)
        self.fourier_scale = 1.0  # Adjust this if needed
        self.fourier_dim = 256
        
        # +2 for normalized (x,y) coordinates
        coord_input_dim = nb_inputs + 2
        
        self.B = tf.Variable(
            tf.random.normal([coord_input_dim, self.fourier_dim]) * self.fourier_scale,
            trainable=False,
            dtype=self.dtype_model
        )
        
        # MLP layers
        self.dense_layers = []
        input_dim = self.fourier_dim * 2
        
        # First layer handles Fourier features
        self.dense_layers.append(
            tf.keras.layers.Dense(
                nb_filters,
                activation=activation,
                kernel_initializer='glorot_uniform',
                dtype=self.dtype_model
            )
        )
        
        # Hidden layers with residual connections
        for i in range(nb_layers - 1):
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    nb_filters,
                    activation=activation,
                    kernel_initializer='glorot_uniform',
                    dtype=self.dtype_model
                )
            )
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            nb_outputs,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            dtype=self.dtype_model
        )
        
        self.build(input_shape=[None, None, None, nb_inputs])
    
    def call(self, inputs):
        x = inputs
        
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)
        
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        
        # Create normalized coordinate grid
        # CRITICAL: Normalize to [-1, 1] for both dimensions
        y_coords = tf.linspace(-1.0, 1.0, height)
        x_coords = tf.linspace(-1.0, 1.0, width)
        
        xx, yy = tf.meshgrid(x_coords, y_coords)
        
        # Expand to match batch size
        xx = tf.cast(xx, self.dtype_model)
        yy = tf.cast(yy, self.dtype_model)
        xx = tf.tile(xx[None, :, :, None], [batch_size, 1, 1, 1])
        yy = tf.tile(yy[None, :, :, None], [batch_size, 1, 1, 1])
        
        # Concatenate coordinates with input features
        x = tf.concat([x, xx, yy], axis=-1)
        
        # Fourier features
        x_proj = tf.matmul(x, self.B)
        x = tf.concat([tf.sin(2 * np.pi * x_proj), tf.cos(2 * np.pi * x_proj)], axis=-1)
        
        # MLP with residual connections
        for i, dense in enumerate(self.dense_layers):
            if i == 0:
                x = dense(x)
            else:
                residual = x
                x = dense(x) + residual
        
        outputs = self.output_layer(x)
        
        return outputs
    
class UNET_inital(tf.keras.Model):
    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(UNET_inital, self).__init__()
        
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = _normalize_precision(precision)
        
        self.input_normalizer = input_normalizer
        
        nb_filters = cfg.processes.iceflow.emulator.network.nb_out_filter
        activation = cfg.processes.iceflow.emulator.network.activation
        
        # Use He initialization for better gradient flow
        initializer = tf.keras.initializers.HeNormal()
        
        # Encoder
        self.enc_conv1 = tf.keras.layers.Conv2D(nb_filters, 3, padding='same', activation=activation, 
                                                 kernel_initializer=initializer, dtype=self.dtype_model)
        self.enc_conv2 = tf.keras.layers.Conv2D(nb_filters, 3, padding='same', activation=activation,
                                                 kernel_initializer=initializer, dtype=self.dtype_model)
        self.pool1 = tf.keras.layers.Conv2D(nb_filters, 3, strides=2, padding='same',
                                             kernel_initializer=initializer, dtype=self.dtype_model)
        
        self.enc_conv3 = tf.keras.layers.Conv2D(nb_filters*2, 3, padding='same', activation=activation,
                                                 kernel_initializer=initializer, dtype=self.dtype_model)
        self.enc_conv4 = tf.keras.layers.Conv2D(nb_filters*2, 3, padding='same', activation=activation,
                                                 kernel_initializer=initializer, dtype=self.dtype_model)
        self.pool2 = tf.keras.layers.Conv2D(nb_filters*2, 3, strides=2, padding='same',
                                             kernel_initializer=initializer, dtype=self.dtype_model)
        
        # Bottleneck
        self.bottleneck1 = tf.keras.layers.Conv2D(nb_filters*4, 3, padding='same', activation=activation,
                                                   kernel_initializer=initializer, dtype=self.dtype_model)
        self.bottleneck2 = tf.keras.layers.Conv2D(nb_filters*4, 3, padding='same', activation=activation,
                                                   kernel_initializer=initializer, dtype=self.dtype_model)
        
        # Decoder
        self.up1 = tf.keras.layers.Conv2DTranspose(nb_filters*2, 3, strides=2, padding='same',
                                                    kernel_initializer=initializer, dtype=self.dtype_model)
        self.dec_conv1 = tf.keras.layers.Conv2D(nb_filters*2, 3, padding='same', activation=activation,
                                                 kernel_initializer=initializer, dtype=self.dtype_model)
        self.dec_conv2 = tf.keras.layers.Conv2D(nb_filters*2, 3, padding='same', activation=activation,
                                                 kernel_initializer=initializer, dtype=self.dtype_model)
        
        self.up2 = tf.keras.layers.Conv2DTranspose(nb_filters, 3, strides=2, padding='same',
                                                    kernel_initializer=initializer, dtype=self.dtype_model)
        self.dec_conv3 = tf.keras.layers.Conv2D(nb_filters, 3, padding='same', activation=activation,
                                                 kernel_initializer=initializer, dtype=self.dtype_model)
        self.dec_conv4 = tf.keras.layers.Conv2D(nb_filters, 3, padding='same', activation=activation,
                                                 kernel_initializer=initializer, dtype=self.dtype_model)
        
        # Output - use smaller initialization for output layer
        self.output_layer = tf.keras.layers.Conv2D(nb_outputs, 1, activation=None,
                                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                                    dtype=self.dtype_model)
        
        self.build(input_shape=[None, None, None, nb_inputs])
    
    def call(self, inputs):
        x = inputs
        
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)
        
        # Encoder
        skip1 = self.enc_conv1(x)
        skip1 = self.enc_conv2(skip1)
        x = self.pool1(skip1)
        
        skip2 = self.enc_conv3(x)
        skip2 = self.enc_conv4(skip2)
        x = self.pool2(skip2)
        
        # Bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        
        # Decoder
        x = self.up1(x)
        x = tf.keras.layers.Concatenate()([x, skip2])
        x = self.dec_conv1(x)
        x = self.dec_conv2(x)
        
        x = self.up2(x)
        x = tf.keras.layers.Concatenate()([x, skip1])
        x = self.dec_conv3(x)
        x = self.dec_conv4(x)
        
        outputs = self.output_layer(x)
        
        return outputs

class FNO(tf.keras.Model):
    """
    Simplified Fourier Neural Operator - operates in frequency domain for global information
    """
    
    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(FNO, self).__init__()
        
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = _normalize_precision(precision)
        
        self.input_normalizer = input_normalizer
        
        nb_filters = cfg.processes.iceflow.emulator.network.nb_out_filter
        nb_layers = int(cfg.processes.iceflow.emulator.network.nb_layers)
        
        # Lifting layer: project input to hidden dimension
        self.lift = tf.keras.layers.Conv2D(nb_filters, 1, dtype=self.dtype_model)
        
        # Fourier layers
        self.fourier_layers = []
        self.conv_layers = []
        for i in range(nb_layers):
            # We'll use Conv2D as a simplified spectral convolution
            self.fourier_layers.append(
                tf.keras.layers.Conv2D(nb_filters, 1, dtype=self.dtype_model)
            )
            self.conv_layers.append(
                tf.keras.layers.Conv2D(nb_filters, 1, dtype=self.dtype_model)
            )
        
        self.activation = tf.keras.layers.Activation(
            cfg.processes.iceflow.emulator.network.activation
        )
        
        # Projection layer: project back to output dimension
        self.project = tf.keras.layers.Conv2D(nb_outputs, 1, activation=None, dtype=self.dtype_model)
        
        self.build(input_shape=[None, None, None, nb_inputs])
    
    def call(self, inputs):
        x = inputs
        
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)
        
        # Lift to higher dimension
        x = self.lift(x)
        
        # Fourier layers
        for fourier_layer, conv_layer in zip(self.fourier_layers, self.conv_layers):
            residual = x
            
            # FFT to frequency domain
            x_freq = tf.signal.fft2d(tf.cast(x, tf.complex64))
            
            # Apply spectral convolution (simplified: using real part)
            x_freq_real = tf.cast(tf.math.real(x_freq), self.dtype_model)
            x_freq_imag = tf.cast(tf.math.imag(x_freq), self.dtype_model)
            
            # Process in frequency domain
            x_freq_processed = fourier_layer(x_freq_real)
            
            # IFFT back to spatial domain
            x_freq_complex = tf.cast(x_freq_processed, tf.complex64)
            x = tf.cast(tf.math.real(tf.signal.ifft2d(x_freq_complex)), self.dtype_model)
            
            # Add skip connection in spatial domain
            x = x + conv_layer(residual)
            x = self.activation(x)
        
        # Project to output
        outputs = self.project(x)
        
        return outputs

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


class ManualNormalizationLayer(tf.keras.layers.Layer):
    """
    Normalization layer with manually specified mean and variance for each channel.
    """
    def __init__(self, scales, variances=None, epsilon=1e-6, dtype='float32', **kwargs):
        """
        Args:
            scales: Array of mean values for each channel
            variances: Array of variance values for each channel (optional, defaults to ones)
            epsilon: Small constant for numerical stability
            dtype: Data type for the layer
        """
        super(ManualNormalizationLayer, self).__init__(dtype=dtype, **kwargs)
        self.epsilon = epsilon
        self.scales = np.array(scales, dtype=dtype)
        
        # Default variances to ones if not provided
        if variances is None:
            self.variances = np.ones_like(self.scales, dtype=dtype)
        else:
            self.variances = np.array(variances, dtype=dtype)
            
        # Validate shapes match
        if len(self.scales) != len(self.variances):
            raise ValueError(
                f"scales and variances must have same length, "
                f"got {len(self.scales)} and {len(self.variances)}"
            )
    
    def build(self, input_shape):
        """Build the layer - validate input channels and create weight tensors."""
        nb_channels = input_shape[-1]
        
        # Validate that scales match input channels
        if len(self.scales) != nb_channels:
            raise ValueError(
                f"Expected scales of length {nb_channels}, got {len(self.scales)}"
            )
        
        # Create non-trainable weights for mean and variance
        # Shape: [1, 1, 1, nb_channels] for broadcasting over [batch, height, width, channels]
        self.mean = self.add_weight(
            name='mean',
            shape=(1, 1, 1, nb_channels),
            initializer=tf.constant_initializer(self.scales.reshape(1, 1, 1, -1)),
            trainable=False,
            dtype=self.dtype
        )
        
        self.variance = self.add_weight(
            name='variance',
            shape=(1, 1, 1, nb_channels),
            initializer=tf.constant_initializer(self.variances.reshape(1, 1, 1, -1)),
            trainable=False,
            dtype=self.dtype
        )
        
        super(ManualNormalizationLayer, self).build(input_shape)
    
    def call(self, inputs):
        """Apply normalization: (x - mean) / std"""
        std = tf.sqrt(self.variance + self.epsilon)
        normalized = (inputs - self.mean) / std
        return normalized
    
    def get_config(self):
        """Enable serialization."""
        config = super(ManualNormalizationLayer, self).get_config()
        config.update({
            'scales': self.scales.tolist(),
            'variances': self.variances.tolist(),
            'epsilon': self.epsilon,
        })
        return config

class StandardizationLayer(tf.keras.layers.Layer):
    """
    Options for normalization strategies in PDE problems.
    """
    def __init__(self, mode='channel', epsilon=1e-6, **kwargs):
        """
        Args:
            mode: 'spatial' - normalize each sample/channel over spatial dims
                  'channel' - normalize each channel over batch and spatial dims
                  'global' - normalize over everything (batch, spatial, channels)
        """
        super(StandardizationLayer, self).__init__(**kwargs)
        self.mode = mode
        self.epsilon = epsilon
    
    def call(self, inputs):
        if self.mode == 'spatial':
            # Normalize each sample/channel over spatial dimensions [H, W]
            mean = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=[1, 2], keepdims=True)
        elif self.mode == 'channel':
            # Normalize each channel over batch and spatial [N, H, W]
            mean = tf.reduce_mean(inputs, axis=[0, 1, 2], keepdims=True)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=[0, 1, 2], keepdims=True)
        elif self.mode == 'global':
            # Normalize over everything
            mean = tf.reduce_mean(inputs)
            variance = tf.reduce_mean(tf.square(inputs - mean))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        std = tf.sqrt(variance + self.epsilon)
        normalized = (inputs - mean) / std
        
        return normalized
