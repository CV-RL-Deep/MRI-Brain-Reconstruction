import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

try:
    from tensorflow.keras.layers import SpectralNormalization
except ImportError:
    # Fallback: Just identity wrapper (no-op) if not available, to avoid complex custom code risks
    class SpectralNormalization(layers.Wrapper):
        def __init__(self, layer, **kwargs):
            super().__init__(layer, **kwargs)

        def call(self, inputs, training=None):
            return self.layer(inputs)


# class SpectralNormalization(layers.Wrapper):
#     def __init__(self, layer, iteration=1, **kwargs):
#         super(SpectralNormalization, self).__init__(layer, **kwargs)
#         self.iteration = iteration

#     def build(self, input_shape):
#         if not self.layer.built:
#             self.layer.build(input_shape)
#         if not hasattr(self.layer, 'kernel'):
#             raise ValueError('Layer must have a kernel weight to use SpectralNormalization.')

#         self.w = self.layer.kernel
#         self.w_shape = self.w.shape.as_list()
#         self.u = self.add_weight(shape=(1, self.w_shape[-1]), initializer=tf.initializers.TruncatedNormal(stddev=0.02), trainable=False, name='sn_u', dtype=self.dtype)

#     def call(self, inputs, training=None):
#         # Power iteration
#         # simple implementation for brevity
#         # For full robustness, use tf.keras.layers.SpectralNormalization if available
#         # But here is a simplified version if needed, or rely on standard Conv for now if this is too complex to inject.
#         # Actually, let's try to import the native one first.
#         return self.layer(inputs)


class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    Includes Float32 casting for stability in Mixed Precision.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs

        # 1. Force Float32 to prevent overflow in exp/random
        z_mean = tf.cast(z_mean, tf.float32)
        z_log_var = tf.cast(z_log_var, tf.float32)

        # 2. Hard Clip Log-Variance to prevent explosion
        # exp(10) ~ 22000, safe for float16, very safe for float32
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)

        epsilon = tf.random.normal(shape=tf.shape(z_mean), dtype=tf.float32)

        # Sample
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Cast back to model dtype (likely float16)
        return tf.cast(z, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VAELossLayer(layers.Layer):
    """Identity layer that adds KL Divergence loss."""

    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, inputs):
        z_mean, z_log_var = inputs

        # 1. Force Float32
        z_mean = tf.cast(z_mean, tf.float32)
        z_log_var = tf.cast(z_log_var, tf.float32)

        # 2. Hard Clip (Must match Sampling logic)
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)

        # 3. KL Calculation
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss)

        # Add loss
        self.add_loss(kl_loss * self.weight)

        return inputs[0]  # pass-through


class InstanceNormalization(layers.Layer):
    """
    Standard Instance Normalization (not always available in base Keras).
    Normalizes (H, W) per channel, per sample.
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (
            self.gamma * (x - mean) / tf.sqrt(variance + self.epsilon)
            + self.beta
        )


class FourierEmbedding(layers.Layer):
    """
    Pure sinusoidal feature expansion.
    """

    def __init__(self, num_freqs=8, **kwargs):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        freq_bands = 2.0 ** np.linspace(0.0, num_freqs - 1, num_freqs) * np.pi
        self.freq_bands = tf.constant(freq_bands, dtype=tf.float32)

    def call(self, p):
        p_expanded = tf.expand_dims(p, -1)
        args = p_expanded * self.freq_bands

        sin_emb = tf.sin(args)
        cos_emb = tf.cos(args)

        emb = tf.concat([sin_emb, cos_emb], axis=-1)

        # Flatten: (Batch, Channels * Num_Freqs * 2)
        input_channels = tf.shape(p)[1]
        if p.shape[1] is not None:
            input_channels = p.shape[1]

        flat_dim = input_channels * self.num_freqs * 2
        return tf.reshape(emb, [-1, flat_dim])


# class FiLMLayer(layers.Layer):
#     """
#     Feature-wise Linear Modulation.
#     Modulates a feature map using a conditioning vector.
#     """
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def build(self, input_shape):
#         # input_shape = [feature_shape, condition_shape]
#         feature_channels = input_shape[0][-1]

#         # Project condition vector to 2 * feature_channels (for Gamma and Beta)
#         self.dense = layers.Dense(feature_channels * 2, kernel_initializer='zeros')

#     def call(self, inputs):
#         x, condition = inputs

#         # Generate Gamma and Beta from condition
#         mod_params = self.dense(condition)

#         # Split into Gamma and Beta
#         # Reshape to (Batch, 1, 1, Channels) for spatial broadcasting
#         channels = tf.shape(x)[-1]
#         gamma, beta = tf.split(mod_params, 2, axis=-1)

#         gamma = tf.reshape(gamma, [-1, 1, 1, channels])
#         beta = tf.reshape(beta, [-1, 1, 1, channels])

#         # Apply FiLM: x * (1 + gamma) + beta
#         return x * (1.0 + gamma) + beta


class SPADELayer(layers.Layer):
    def __init__(self, channels, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.bn = layers.BatchNormalization(center=False, scale=False)
        self.conv_mask = layers.Conv2D(
            128, kernel_size=kernel_size, padding='same', activation='relu'
        )
        self.conv_gamma = layers.Conv2D(
            channels, kernel_size=kernel_size, padding='same'
        )
        self.conv_beta = layers.Conv2D(
            channels, kernel_size=kernel_size, padding='same'
        )

    def call(self, inputs):
        x, mask = inputs  # purely spatial

        normalized = self.bn(x)
        target_size = tf.shape(x)[1:3]
        mask_resized = tf.image.resize(mask, target_size, method='nearest')
        mask_resized = tf.cast(mask_resized, x.dtype)

        mask_feat = self.conv_mask(mask_resized)
        gamma = self.conv_gamma(mask_feat)
        beta = self.conv_beta(mask_feat)

        return normalized * (1.0 + gamma) + beta


class SPADEResBlock(layers.Layer):
    def __init__(self, filters, input_channels, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.learned_shortcut = filters != input_channels
        f_mid = min(filters, input_channels)

        self.spade1 = SPADELayer(input_channels)
        self.conv1 = layers.Conv2D(f_mid, 3, padding='same')
        self.spade2 = SPADELayer(f_mid)
        self.conv2 = layers.Conv2D(filters, 3, padding='same')

        if self.learned_shortcut:
            self.spade_s = SPADELayer(input_channels)
            self.conv_s = layers.Conv2D(
                filters, 1, padding='same', use_bias=False
            )

    def call(self, inputs):
        x, mask = inputs  # purely spatial

        x_s = x
        if self.learned_shortcut:
            x_s = self.conv_s(self.spade_s([x, mask]))

        dx = self.spade1([x, mask])
        dx = tf.nn.relu(dx)
        dx = self.conv1(dx)
        dx = self.spade2([dx, mask])
        dx = tf.nn.relu(dx)
        dx = self.conv2(dx)

        return x_s + dx
