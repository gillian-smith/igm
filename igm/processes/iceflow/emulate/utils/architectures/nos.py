import tensorflow as tf

from igm.utils.math.precision import _normalize_precision


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
        self.project = tf.keras.layers.Conv2D(
            nb_outputs, 1, activation=None, dtype=self.dtype_model
        )

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
            x = tf.cast(
                tf.math.real(tf.signal.ifft2d(x_freq_complex)), self.dtype_model
            )

            # Add skip connection in spatial domain
            x = x + conv_layer(residual)
            x = self.activation(x)

        # Project to output
        outputs = self.project(x)

        return outputs
