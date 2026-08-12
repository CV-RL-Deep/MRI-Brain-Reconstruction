import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers, models, applications

from src.models.layers import (SpectralNormalization, Sampling, VAELossLayer,
                               InstanceNormalization, FourierEmbedding,
                               SPADEResBlock)
from configs.config import Config


class DiscriminatorBuilder:
    """
    Builds a PatchGAN Discriminator.
    Input: [Reconstructed_Image, Condition_Stack]
    Output: Patch Map of Real/Fake scores.
    """
    @staticmethod
    def build(input_shape, condition_shape, base_filters=64):
        img_input = layers.Input(shape=input_shape, name='img_input')
        cond_input = layers.Input(shape=condition_shape, name='cond_input')

        # Concat Image + Condition (History + Mask)
        x = layers.Concatenate()([img_input, cond_input])

        # Store intermediate features for Feature Matching Loss
        features = []

        # C64
        # x = layers.Conv2D(base_filters, 4, strides=2, padding='same')(x)
        # x = layers.LeakyReLU(0.2)(x)
        x = SpectralNormalization(layers.Conv2D(base_filters, 4, strides=2,
                                                padding='same'))(x)
        x = layers.LeakyReLU(0.2)(x)
        features.append(x)

        # C128
        # x = layers.Conv2D(base_filters*2, 4, strides=2, padding='same')(x)
        # x = layers.GroupNormalization(groups=-1)(x)
        # x = layers.LeakyReLU(0.2)(x)
        x = SpectralNormalization(layers.Conv2D(base_filters*2, 4, strides=2,
                                                padding='same'))(x)
        x = InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        features.append(x)

        # C256
        # x = layers.Conv2D(base_filters*4, 4, strides=2, padding='same')(x)
        # x = layers.GroupNormalization(groups=-1)(x)
        # x = layers.LeakyReLU(0.2)(x)
        x = SpectralNormalization(layers.Conv2D(base_filters*4, 4, strides=2,
                                                padding='same'))(x)
        x = InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        features.append(x)

        # C512 (Stride 1)
        # x = layers.Conv2D(base_filters*8, 4, strides=1, padding='same')(x)
        # x = layers.GroupNormalization(groups=-1)(x)
        # x = layers.LeakyReLU(0.2)(x)
        x = SpectralNormalization(layers.Conv2D(base_filters*8, 4, strides=1,
                                                padding='same'))(x)
        x = InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        features.append(x)

        # Output Map (1 channel)
        outputs = layers.Conv2D(1, 4, strides=1, padding='same')(x)

        return models.Model([img_input, cond_input], [outputs] + features,
                            name="discriminator")


class ModelBuilder:
    """
    Factory to instantiate different neural network architectures.
    Currently supports: U-Net (Custom & EfficientNet backbones).
    Future support: VAE, GAN.
    """
    @staticmethod
    def build(config: Config):
        arch = config.model.architecture.lower()

        if arch == 'unet':
            return ModelBuilder._build_unet(config)
        elif arch == 'spade':
            return ModelBuilder._build_spade_generator(config)
        elif arch == 'vae':
            return ModelBuilder._build_spade_vae_generator(config)
        elif arch == 'gan':
            raise NotImplementedError("GAN architecture not yet implemented.")
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    @staticmethod
    def conv_block(x, filters, activation='gelu'):
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters, 3, padding='same',
                          kernel_initializer='he_normal')(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        return x

    @staticmethod
    def attention_gate(x, skip, filters):
        """Additive Attention Gate"""
        g = layers.Conv2D(filters, 1, padding='same')(x)
        g = layers.BatchNormalization()(g)
        s = layers.Conv2D(filters, 1, padding='same')(skip)
        s = layers.BatchNormalization()(s)

        psi = layers.Activation('relu')(layers.Add()([g, s]))
        psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(psi)
        return layers.Multiply()([skip, psi])

    @staticmethod
    def up_block(x, skip, filters, activation='gelu', use_attention=False):
        """
        Upsampling block using PixelShuffle (Sub-pixel Convolution) for sharpness.
        Replaces standard UpSampling2D to avoid blur/checkerboard artifacts.
        """
        # 1. PixelShuffle Upsampling
        # To upscale 2x, we need 4x channels.
        # We assume 'x' currently has 'in_filters'. We want 'filters' output.
        # So we project to filters * 4

        # x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        # x = layers.Conv2D(filters, 3, padding='same')(x) # adjust channels
        x = layers.Conv2D(filters * 4, kernel_size=1, padding='same',
                          kernel_initializer='he_normal')(x)
        # PixelShuffle trick: tf.nn.depth_to_space
        x = layers.Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)

        # x is now (2H, 2W, filters)

        # 2. Skip Connection
        if use_attention and skip is not None:
            # Attention gates act on the skip connection
            skip = ModelBuilder.attention_gate(x, skip, filters // 2)

        if skip is not None:
            x = layers.Concatenate()([x, skip])

        # 3. Refinement Convolution
        x = ModelBuilder.conv_block(x, filters, activation)
        return x

    @staticmethod
    def get_efficientnet_encoder(input_tensor, backbone_name):
        """Returns the model and the skip connection tensors."""
        # Map names to constructors
        ctors = {
            'B0': applications.EfficientNetB0, 'B1': applications.EfficientNetB1,
            'B2': applications.EfficientNetB2, 'B3': applications.EfficientNetB3,
            'B4': applications.EfficientNetB4, 'B5': applications.EfficientNetB5,
            'B6': applications.EfficientNetB6, 'B7': applications.EfficientNetB7,
        }
        # Standard EfficientNet naming convention for block activation layers
        # Key: Backbone Name -> List of layer names for [Skip 1/2, Skip 1/4, Skip 1/8, Skip 1/16]
        # Skip 1/32 comes from bridge output
        skip_names = {
            'B0': ['block2a_expand_activation', 'block3a_expand_activation',
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B1': ['block2a_expand_activation', 'block3a_expand_activation',
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B2': ['block2a_expand_activation', 'block3a_expand_activation',
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B3': ['block2a_expand_activation', 'block3a_expand_activation',
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B4': ['block2a_expand_activation', 'block3a_expand_activation',
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B5': ['block2a_expand_activation', 'block3a_expand_activation',
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B6': ['block2a_expand_activation', 'block3a_expand_activation',
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B7': ['block2a_expand_activation', 'block3a_expand_activation',
                   'block4a_expand_activation', 'block6a_expand_activation'],
        }

        # B0-B7 share similar early block naming for the expand activation.
        # However, deeper networks repeat blocks. We target the FIRST block of each stage.
        # Actually, for U-Net skips, we usually want the LAST block of a stage (deepest features at that res).
        # But 'expand_activation' of the FIRST block of the NEXT stage works too (pre-downsample).
        # The list above targets the start of stage 2, 3, 4, 6.
        # This has worked for B0. For B4+, resolutions might shift.
        # Verified for standard Keras EfficientNet implementation: These names are stable
        names = skip_names.get(backbone_name, skip_names['B0'])
        base = ctors[backbone_name](
            name=f"encoder_{backbone_name}",
            include_top=False,
            weights=None,
            input_tensor=input_tensor
        )

        try:
            skips = [base.get_layer(n).output for n in names]
        except ValueError as e:
            # Fallback for debugging if layer names shift in newer TF versions
            print(f"Error finding skip layers for {backbone_name}. Available layers:")
            raise e
        return base.output, skips

    @staticmethod
    def _build_spade_generator(config: Config):
        arch = config.model.architecture.lower()
        N = config.data.neighborhood

        # 1. Inputs
        history_input = layers.Input(shape=(*config.data.padded_size, N),
                                     name='history_input')
        mask_input = layers.Input(shape=(*config.data.padded_size, 1),
                                  name='mask_input')

        # p_rel_input = layers.Input(shape=(N,), name='p_rel_input')
        # p_abs_input = layers.Input(shape=(1,), name='p_abs_input')

        base_f = config.model.base_filters

        # 2. Positional Encodings
        # p_rel_emb = FourierEmbedding(hidden_dim=256, num_freqs=8, name='p_rel_embed')(p_rel_input)
        # p_abs_emb = FourierEmbedding(hidden_dim=256, num_freqs=8, name='p_abs_embed')(p_abs_input)
        p_history_input = layers.Input(shape=(N,), name='p_history_input')
        p_abs_input = layers.Input(shape=(1,), name='p_abs_input')

        # 4. Encoder (Processes History Images)
        decoder_filters = [base_f * 8, base_f * 4, base_f * 2, base_f]

        if config.model.backbone:
            bridge, skips = ModelBuilder.get_efficientnet_encoder(history_input,
                                                                  config.model.backbone)
        else:
            s1 = ModelBuilder.conv_block(history_input, base_f)
            p1 = layers.MaxPooling2D()(s1)
            s2 = ModelBuilder.conv_block(p1, base_f * 2)
            p2 = layers.MaxPooling2D()(s2)
            s3 = ModelBuilder.conv_block(p2, base_f * 4)
            p3 = layers.MaxPooling2D()(s3)
            s4 = ModelBuilder.conv_block(p3, base_f * 8)
            bridge = layers.MaxPooling2D()(s4)
            bridge = ModelBuilder.conv_block(bridge, base_f * 16)
            skips = [s4, s3, s2, s1]

        # 4. Bottleneck Injection (Absolute Position)
        # Modulates the global bridge features using the exact slice depth
        # x = FiLMLayer(name='bottleneck_film_abs')([bridge, p_abs_emb])

        # -------------------------------
        # # --- THE BOTTLENECK INJECTION ---
        # # 5. Two-Layer Adapter (MLP) with Zero-Weight Initialization
        # bridge_channels = K.int_shape(bridge)[-1]

        # # --- THE POSITIONAL PIPELINE ---
        # # 2. Glue input (N) and target (1) absolute positions into a single vector
        # p_combined = layers.Concatenate(axis=-1)([p_history_input, p_abs_input])

        # # 3. Fourier Encoding
        # fourier_feats = FourierEmbedding(num_freqs=8)(p_combined)

        # # Hidden layer
        # emb = layers.Dense(256, activation='swish', name='pos_mlp_hidden')(fourier_feats)

        # # Projection layer (ZERO INITIALIZED)
        # emb = layers.Dense(bridge_channels,
        #                    kernel_initializer='zeros',
        #                    bias_initializer='zeros',
        #                    name='pos_mlp_out')(emb)

        # # Reshape to (Batch, 1, 1, Channels) to broadcast spatially
        # emb_spatial = layers.Reshape((1, 1, bridge_channels))(emb)

        # # 6. Inject via ADDITION
        # x = layers.Add(name='bottleneck_pos_add')([bridge, emb_spatial])
        # # --------------------------------

        # --- THE CONDITIONAL POSITIONAL PIPELINE ---
        if config.model.use_positional_encoding:
            # Glue input (N) and target (1) absolute positions into a single vector
            p_combined = layers.Concatenate(axis=-1)([p_history_input, p_abs_input])
            fourier_feats = FourierEmbedding(num_freqs=8)(p_combined)

            bridge_channels = K.int_shape(bridge)[-1]
            emb = layers.Dense(256, activation='swish', name='pos_mlp_hidden')(fourier_feats)
            emb = layers.Dense(bridge_channels,
                               kernel_initializer='zeros',
                               bias_initializer='zeros',
                               name='pos_mlp_out')(emb)

            emb_spatial = layers.Reshape((1, 1, bridge_channels))(emb)
            x = layers.Add(name='bottleneck_pos_add')([bridge, emb_spatial])
        else:
            # Completely bypass positional encoding. Model becomes shift-invariant
            x = bridge
        # -------------------------------------------

        # 7. Decoder with SPADE and Skip Modulation
        for i, skip in enumerate(reversed(skips)):
            filters = decoder_filters[i] if i < len(decoder_filters) else base_f

            x = layers.Conv2D(filters * 4, 1, padding='same')(x)
            x = layers.Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)

            # if skip is not None:
            #     # 8. Skip Connection Injection (Relative Position)
            #     # Modulates the local skip features using the sequence trajectory
            #     modulated_skip = FiLMLayer(name=f'skip_film_rel_{i}')([skip, p_rel_emb])

            #     x = layers.Concatenate()([x, modulated_skip])
            if skip is not None:
                x = layers.Concatenate()([x, skip])

            # 8. SPADE Block (Conditioned strictly on Spatial Mask)
            # SPADE Block receives ONLY the feature map and spatial mask
            x = SPADEResBlock(filters, 1)([x, mask_input])

        # 9. Final PixelShuffle
        if config.model.backbone:
            current_channels = K.int_shape(x)[-1]
            x = layers.Conv2D(current_channels * 4, 1, padding='same')(x)
            x = layers.Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)

        # outputs = layers.Conv2D(1, 1, activation='sigmoid', name='reconstruction')(x)
        # Feature switch
        if config.model.num_hypotheses == 1:
            outputs = layers.Conv2D(1, 1, activation='sigmoid', name='reconstruction')(x)
        else:
            # Outputs shape: (Batch, H, W, M)
            outputs = layers.Conv2D(
                config.model.num_hypotheses,
                1,
                activation='sigmoid',
                name='reconstruction_mhp'
            )(x)

        # Return Multi-Input Model
        return models.Model(
            inputs=[history_input, mask_input, p_history_input, p_abs_input],
            outputs=outputs,
            name=f"{arch}_{config.model.backbone}"
        )

    @staticmethod
    def _build_spade_vae_generator(config: Config):
        arch = config.model.architecture.lower()

        # TODO: assert we're using target layer mask
        # 1. Inputs
        # Image Input: History [t-N ... t-1]
        # img_shape = (*config.data.padded_size, config.model.input_channels)
        # img_input = layers.Input(shape=img_shape, name='history_input')
        # Input: [History (N) + Mask (1)]
        total_channels = config.model.input_channels
        full_input = layers.Input(shape=(*config.data.padded_size, total_channels),
                                  name='full_input')

        # Split:
        # History: Channels [0 ... N-1]
        # Mask: Channel [-1] (assuming input_last_target_mask=True)
        N = config.data.neighborhood

        # Mask Input: Target Geometry [t] (Binary Mask)
        # Note: SPADE usually takes a 1-channel label map or one-hot.
        # We use 1-channel binary mask
        # mask_shape = (*config.data.padded_size, 1)
        # mask_input = layers.Input(shape=mask_shape, name='mask_input')
        # Slicing using Lambda layers to support serialization

        # History (first N channels) -> Goes to Encoder
        history_input = layers.Lambda(lambda x: x[..., :N], name='split_history')(full_input)

        # Mask (last channel) -> Goes to SPADE
        mask_input = layers.Lambda(lambda x: x[..., N:], name='split_mask')(full_input)

        base_f = config.model.base_filters

        # Encoder uses History
        bridge, skips = ModelBuilder.get_efficientnet_encoder(history_input,
                                                              config.model.backbone)

        # 2. Encoder (Processes History Images)
        if config.model.backbone:
            # bridge, skips = ModelBuilder.get_efficientnet_encoder(inputs, config.model.backbone)
            bridge, skips = ModelBuilder.get_efficientnet_encoder(history_input,
                                                                  config.model.backbone)
            # Decoder filters logic: scaling down from bridge
            # Bridge (B0=1280, B3=1536) -> 4 * base_f -> 2 * base_f -> ...
            decoder_filters = [base_f * 8, base_f * 4, base_f * 2, base_f]
        else:
            # Custom Encoder using base_f
            # s1 = ModelBuilder.conv_block(inputs, base_f)
            s1 = ModelBuilder.conv_block(history_input, base_f)
            p1 = layers.MaxPooling2D()(s1)

            s2 = ModelBuilder.conv_block(p1, base_f * 2)
            p2 = layers.MaxPooling2D()(s2)

            s3 = ModelBuilder.conv_block(p2, base_f * 4)
            p3 = layers.MaxPooling2D()(s3)

            s4 = ModelBuilder.conv_block(p3, base_f * 8)
            bridge = layers.MaxPooling2D()(s4)

            bridge = ModelBuilder.conv_block(bridge, base_f * 16)
            skips = [s4, s3, s2, s1]
            decoder_filters = [base_f * 8, base_f * 4, base_f * 2, base_f]

        # --- VAE BOTTLENECK START ---
        # 1. Project Bridge to latent parameters
        # Keep spatial dims (e.g. 5x5). Filters = latent_dim.
        # We can reuse base_filters * 16 (e.g. 1024) or reduce it.
        latent_dim = bridge.shape[-1]

        z_mean = layers.Conv2D(latent_dim, 1, padding='same', name='z_mean')(bridge)
        z_log_var = layers.Conv2D(latent_dim, 1, padding='same', name='z_log_var')(bridge)

        # 2. Re-parameterization Trick (Sampling)
        z = Sampling(name='z_sampling')([z_mean, z_log_var])

        # 3. Inject KL Loss (Internal to model)
        # This adds the loss to model.losses automatically
        VAELossLayer(weight=config.train.weight_kl)([z_mean, z_log_var])

        # 4. Decoder with SPADE (Conditioned on Mask)
        # Decoder uses Mask for SPADE
        # x = bridge
        # 4. Decoder starts from sampled Z
        x = z
        # --- VAE BOTTLENECK END ---

        for i, skip in enumerate(reversed(skips)):
            filters = decoder_filters[i] if i < len(decoder_filters) else base_f

            # Use PixelShuffle
            # x = layers.Conv2D(x.shape[-1] * 4, 1, padding='same')(x)
            x = layers.Conv2D(filters * 4, 1, padding='same')(x)
            x = layers.Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)

            if skip is not None:
                x = layers.Concatenate()([x, skip])

            # Use SPADE Block
            # Input to SPADE is always the Raw Input Stack (inputs)
            # CRITICAL CHANGE:
            # SPADE Block conditioned on 'mask_input', NOT 'img_input'
            x = SPADEResBlock(filters, 1)([x, mask_input]) # 1 channel mask

        # Final
        if config.model.backbone:
            # Same PixelShuffle logic
            # x = layers.Conv2D(x.shape[-1] * 4, 1, padding='same')(x)
            # Safe logic using K.int_shape
            current_channels = K.int_shape(x)[-1]
            x = layers.Conv2D(current_channels * 4, 1, padding='same')(x)
            x = layers.Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)

        outputs = layers.Conv2D(1, 1, activation='sigmoid', name='reconstruction')(x)
        return models.Model(inputs=full_input, outputs=outputs,
                            name=f"{arch}_{config.model.backbone}")
        # return models.Model(inputs=full_input, outputs=outputs,
        #                     name=f"spade_vae_{config.model.backbone}")

    @staticmethod
    def _build_unet(config: Config):
        arch = config.model.architecture.lower()

        # Copying the core build logic for completeness of context:
        # input_shape = (*config.data.padded_size, config.model.input_channels)
        # inputs = layers.Input(shape=input_shape, name='input_image')
        # x = inputs
        N = config.data.neighborhood

        # 1. Standardize Inputs to match the Data Pipeline (Dictionary format)
        history_input = layers.Input(shape=(*config.data.padded_size, N), name='history_input')
        mask_input = layers.Input(shape=(*config.data.padded_size, 1), name='mask_input')
        p_history_input = layers.Input(shape=(N,), name='p_history_input')
        p_abs_input = layers.Input(shape=(1,), name='p_abs_input')

        # 2. Concatenate history and mask for standard U-Net processing
        if config.data.input_last_target_mask:
            x = layers.Concatenate(axis=-1)([history_input, mask_input])
        else:
            x = history_input

        inputs_list =[history_input, mask_input, p_history_input, p_abs_input]

        if config.model.use_stem:
            x = layers.Conv2D(32, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(config.model.activation)(x)

        if config.model.backbone:
            bridge, skips = ModelBuilder.get_efficientnet_encoder(x, config.model.backbone)
            # Decoder filters logic: scaling down from bridge
            # B0 bridge is 1280.
            # We want to smooth the transition
            decoder_filters = [192, 80, 40, 24]
            # decoder_filters = [256, 128, 64, 32]
        else:
            s1 = ModelBuilder.conv_block(x, 32)
            p1 = layers.MaxPooling2D()(s1)
            s2 = ModelBuilder.conv_block(p1, 64)
            p2 = layers.MaxPooling2D()(s2)
            s3 = ModelBuilder.conv_block(p2, 128)
            p3 = layers.MaxPooling2D()(s3)
            s4 = ModelBuilder.conv_block(p3, 256)
            bridge = layers.MaxPooling2D()(s4)
            bridge = ModelBuilder.conv_block(bridge, 512)
            skips = [s4, s3, s2, s1]
            decoder_filters = [256, 128, 64, 32]

        x = bridge
        for i, skip in enumerate(reversed(skips)):
            filters = decoder_filters[i] if i < len(decoder_filters) else 16
            x = ModelBuilder.up_block(x, skip, filters, config.model.activation,
                                      config.model.use_attention)

        if config.model.backbone:
             x = layers.UpSampling2D((2,2), interpolation='bilinear')(x)

        # outputs = layers.Conv2D(1, 1, activation='sigmoid', name='reconstruction')(x)
        # 3. Feature switch to match output shapes across ablations
        if config.model.num_hypotheses == 1:
            outputs = layers.Conv2D(1, 1, activation='sigmoid', name='reconstruction')(x)
        else:
            outputs = layers.Conv2D(
                config.model.num_hypotheses,
                1,
                activation='sigmoid',
                name='reconstruction_mhp'
            )(x)

        return models.Model(inputs=inputs_list, outputs=outputs,
                            name=f"{arch}_{config.model.backbone}")
