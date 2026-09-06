import tensorflow as tf

from ..configs.config import Config
from ..core.env import KAGGLE
from ..core.utils import logger
from ..models.builder import DiscriminatorBuilder
from ..models.losses import (
    SpatiallyWeightedL1Loss,
    CompositeLoss,
    VanillaL1Loss,
    RelaxedMHPLossWrapper,
    SpectralLoss,
    get_perceptual_loss,
)
from .callbacks import (
    TrainingVisualizer,
    BufferUpdateCallback,
    GeneratorCheckpoint,
)
from ..evaluation.metrics import (
    OracleMAE,
    OracleMSE,
    OracleSSIM,
    OraclePSNR,
    GradientSharpnessMetric,
)


class SPADEGANTrainer(tf.keras.Model):
    def __init__(self, generator, discriminator, config, extra_metrics=None):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.cfg = config

        # Reuse existing loss classes
        self.spatial_l1 = SpatiallyWeightedL1Loss(config)
        # Conditional initialization
        if self.cfg.train.weight_perceptual >= 0.01:
            self.perceptual = get_perceptual_loss(config)
        else:
            self.perceptual = None
        self.spectral = SpectralLoss()

        # --- FIX: Convert GAN weights to dynamic TF Variables ---
        self.w_gan = tf.Variable(
            self.cfg.train.weight_gan, dtype=tf.float32, trainable=False
        )
        self.w_fm = tf.Variable(
            self.cfg.train.weight_fm, dtype=tf.float32, trainable=False
        )
        self.w_l1 = tf.Variable(
            self.cfg.train.weight_l1, dtype=tf.float32, trainable=False
        )
        self.w_perc = tf.Variable(
            self.cfg.train.weight_perceptual, dtype=tf.float32, trainable=False
        )
        self.w_spec = tf.Variable(
            self.cfg.train.weight_spectral, dtype=tf.float32, trainable=False
        )
        self.w_kl = tf.Variable(
            self.cfg.train.weight_kl, dtype=tf.float32, trainable=False
        )

        # Trackers
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.l1_tracker = tf.keras.metrics.Mean(name="l1_loss")
        self.gan_tracker = tf.keras.metrics.Mean(name="gan_loss")
        self.perc_tracker = tf.keras.metrics.Mean(name="perc_loss")
        self.spec_tracker = tf.keras.metrics.Mean(name="spec_loss")
        if self.cfg.model.architecture == 'vae':
            self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

        # Extra Metrics (MAE, SSIM, etc.)
        self.extra_metrics = extra_metrics if extra_metrics else []

    def compile(self, g_optimizer, d_optimizer, **kwargs):
        super().compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        return [
            self.g_loss_tracker,
            self.d_loss_tracker,
            self.l1_tracker,
            self.gan_tracker,
            self.perc_tracker,
            self.spec_tracker,
        ] + self.extra_metrics

    def train_step(self, data):
        x, y = data
        real_img = y[..., 0:1]

        # 1. Train Discriminator
        with tf.GradientTape() as tape:
            fake_img = self.generator(x, training=True)

            # D returns list: [validity, feat1, feat2, ...]
            d_real_out = self.discriminator([real_img, x], training=True)
            d_fake_out = self.discriminator([fake_img, x], training=True)

            # The first element is the validity score
            d_real_logits = d_real_out[0]
            d_fake_logits = d_fake_out[0]

            # Fix: Cast to float32 for stable loss calc
            d_real_logits = tf.cast(d_real_logits, tf.float32)
            d_fake_logits = tf.cast(d_fake_logits, tf.float32)

            # Hinge Loss
            d_loss = tf.reduce_mean(
                tf.nn.relu(1.0 - d_real_logits)
            ) + tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))

            # Loss scaling for Mixed Precision (if using standard Optimizer.minimize it's auto,
            # but with manual GradientTape we often need explicit scaling if not handled by optimizer wrapper)
            # Modern Keras optimizers usually handle this if 'jit_compile=True' or standard fit.
            # But let's assume standard float32 calc is enough

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights)
        )

        # 2. Train Generator
        with tf.GradientTape() as tape:
            fake_img = self.generator(x, training=True)
            d_fake_out = self.discriminator([fake_img, x], training=False)
            d_real_out = self.discriminator(
                [real_img, x], training=False
            )  # get real features too

            d_fake_logits = d_fake_out[0]

            # A. GAN Loss
            g_gan_loss = -tf.reduce_mean(d_fake_logits)

            # B. Feature Matching Loss (NEW)
            g_fm_loss = 0.0
            # Iterate over features (index 1 onwards)
            for real_feat, fake_feat in zip(d_real_out[1:], d_fake_out[1:]):
                g_fm_loss += tf.reduce_mean(tf.abs(real_feat - fake_feat))

            # C. Spatial L1 Loss
            g_l1_loss = self.spatial_l1(y, fake_img)

            # D. Perceptual Loss (Conditional Bypass)
            if self.perceptual is not None:
                g_perc_loss = self.perceptual(real_img, fake_img)
            else:
                g_perc_loss = tf.constant(0.0, dtype=tf.float32)

            # E. Spectral Loss
            g_spec_loss = self.spectral(y, fake_img)

            # Ensure all losses are float32 before weighted sum
            g_gan_loss = tf.cast(g_gan_loss, tf.float32)
            g_fm_loss = tf.cast(g_fm_loss, tf.float32)
            g_l1_loss = tf.cast(g_l1_loss, tf.float32)
            g_perc_loss = tf.cast(g_perc_loss, tf.float32)
            g_spec_loss = tf.cast(g_spec_loss, tf.float32)

            # Total
            # --- FIX: Multiply by dynamic TF Variables instead of cfg floats ---
            total_g_loss = (
                (g_gan_loss * self.w_gan)
                + (g_fm_loss * self.w_fm)
                + (g_l1_loss * self.w_l1)
                + (g_perc_loss * self.w_perc)
                + (g_spec_loss * self.w_spec)
            )

            # FIX: Add Model Internal Losses (KL Divergence)
            if self.cfg.model.architecture == 'vae' and self.generator.losses:
                # Sum up all regularization losses (KL)
                kl_loss_sum = tf.reduce_sum(self.generator.losses)
                # total_g_loss += kl_loss_sum
                # Multiply KL by dynamic weight
                total_g_loss += kl_loss_sum * self.w_kl

                # Optional: Track it
                self.kl_tracker.update_state(
                    kl_loss_sum
                )  # (if there is a tracker)

        g_grads = tape.gradient(total_g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_weights)
        )

        # Metrics
        self.g_loss_tracker.update_state(total_g_loss)
        self.d_loss_tracker.update_state(d_loss)
        self.l1_tracker.update_state(g_l1_loss)
        self.gan_tracker.update_state(g_gan_loss)
        self.perc_tracker.update_state(g_perc_loss)
        self.spec_tracker.update_state(g_spec_loss)

        # Update Extra Metrics (MAE, SSIM)
        # They compare y (Real) vs fake_img
        for m in self.extra_metrics:
            m.update_state(y, fake_img)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Validation Logic
        x, y = data
        real_img = y[..., 0:1]
        fake_img = self.generator(x, training=False)

        # Calculate L1 only for validation tracking
        l1 = self.spatial_l1(y, fake_img)
        self.l1_tracker.update_state(l1)

        # Update Extra Metrics
        for m in self.extra_metrics:
            m.update_state(y, fake_img)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        # Used for validation/inference
        return self.generator(inputs)


class Trainer:
    def __init__(
        self,
        config: Config,
        model,
        train_ds,
        val_ds,
        data_manager,
        train_steps=None,
        val_steps=None,
    ):
        self.cfg = config
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.manager = data_manager
        self.train_steps = train_steps
        self.val_steps = val_steps

    # def train(self, generator_ref=None, compile_model=True):
    def train(self, compile_model=True):
        logger.info(f"Starting training for {self.cfg.train.epochs} epochs.")

        callbacks_list = []
        # Only add visualizer if we are NOT in batch mode (saves massive time and RAM)
        if not self.cfg.batch_mode:
            callbacks_list.append(
                TrainingVisualizer(
                    self.val_ds,
                    self.cfg,
                    frequency=max(1, self.cfg.train.epochs // 5),
                )
            )
        # Only attach the heavy Buffer callback if hallucination is actually enabled!
        if self.cfg.aug.prob_hallucination_max > 0.0:
            callbacks_list.append(
                BufferUpdateCallback(
                    self.manager, samples_per_epoch=128 if KAGGLE else 384
                )
            )

        # LOGIC SWITCH: Check both Architecture AND Training Mode
        is_spade_arch = self.cfg.model.architecture in ['spade', 'vae']
        is_gan_mode = self.cfg.train.gan_mode

        if is_gan_mode and is_spade_arch:
            logger.info(">>> Mode: SPADE GAN Training (Adversarial)")

            # Callbacking
            callbacks_list.append(
                # Custom Checkpoint for Generator
                GeneratorCheckpoint(
                    f"{self.cfg.model.name}_best.keras",
                    monitor='val_l1_loss',
                    save_best_only=True,
                    mode='min',
                )
            )

            # 1. Build Discriminator
            # Input shape: Image (1ch) + Condition (N+Mask ch)
            img_shape = (*self.cfg.data.padded_size, 1)
            cond_shape = (
                *self.cfg.data.padded_size,
                self.cfg.model.input_channels,
            )

            # Note: DiscriminatorBuilder must be defined in Cell 8
            # --- FIX: Only build and compile the GAN wrappers if compiling! ---
            if compile_model:
                img_shape = (*self.cfg.data.padded_size, 1)
                cond_shape = (
                    *self.cfg.data.padded_size,
                    self.cfg.model.input_channels,
                )

                # Build Discriminator
                d_model = DiscriminatorBuilder.build(img_shape, cond_shape)

                extra_metrics = []
                clean_metrics = []

                # Wrap in GAN Trainer
                gan_model = SPADEGANTrainer(
                    self.model, d_model, self.cfg, extra_metrics=clean_metrics
                )

                gan_model.compile(
                    g_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.cfg.train.learning_rate_g, beta_1=0.5
                    ),
                    d_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.cfg.train.learning_rate_d, beta_1=0.5
                    ),
                    metrics=extra_metrics,
                )
                # Replace self.model with the GAN wrapper so .fit() works
                self.model = gan_model

            # 5. Train
            # Train (self.model is now either the newly built gan_model,
            # or the pre-existing gan_model passed from HPMEngine)
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=self.cfg.train.epochs,
                steps_per_epoch=self.train_steps,
                validation_steps=self.val_steps,
                callbacks=callbacks_list,
                verbose=2 ** int(self.cfg.batch_mode),
            )

            # Point self.model back to the trained generator for downstream inference
            # (Note: self.model is the SPADEGANTrainer wrapper here)
            self.model = self.model.generator
        else:
            # REGRESSION MODE
            # Works for both 'unet' AND 'spade' (if gan_mode=False)
            arch_name = self.cfg.model.architecture.upper()
            logger.info(
                f">>> Mode: Standard Regression Training ({arch_name} Generator)"
            )

            # Callbacking
            callbacks_list.append(
                # Checkpoint: Save model with lowest validation loss (or L1 for GAN)
                tf.keras.callbacks.ModelCheckpoint(
                    f"{self.cfg.model.name}_best.keras",
                    save_best_only=True,
                    monitor='val_loss',
                    # If SPADE, we track l1_loss. If Regression, we track total loss
                    # monitor='val_l1_loss' if is_spade_arch else 'val_loss',
                    save_weights_only=False,  # we want to save full model
                ),
            )

            # --- FIX: Only compile if the flag is True! ---
            if compile_model:
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.cfg.train.learning_rate
                )
                # Select Loss
                if getattr(self.cfg.train, 'use_unmasked_loss', False):
                    loss_fn = RelaxedMHPLossWrapper(
                        base_loss_fn=VanillaL1Loss(),
                        num_hypotheses=self.cfg.model.num_hypotheses,
                        epsilon=self.cfg.model.mhp_epsilon,
                    )
                elif self.cfg.train.use_spatial_loss:
                    loss_fn = SpatiallyWeightedL1Loss(self.cfg)
                else:
                    # loss_fn = CompositeLoss(self.cfg)
                    # Wrap it in the MHP logic
                    loss_fn = RelaxedMHPLossWrapper(
                        base_loss_fn=CompositeLoss(self.cfg),
                        num_hypotheses=self.cfg.model.num_hypotheses,
                        epsilon=self.cfg.model.mhp_epsilon,
                    )
                metrics = [
                    # 'mae', 'mse', SSIMMetric(), PSNRMetric(),
                    # GradientSharpnessMetric()
                    OracleMAE(),  # OracleMAEMetric(),
                    OracleMSE(),  # OracleMSEMetric(),
                    OracleSSIM(),  # SSIMMetric(),
                    OraclePSNR(),  # PSNRMetric(),
                    GradientSharpnessMetric(),
                ]
                self.model.compile(
                    optimizer=optimizer, loss=loss_fn, metrics=metrics
                )

            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=self.cfg.train.epochs,
                steps_per_epoch=self.train_steps,
                validation_steps=self.val_steps,
                callbacks=callbacks_list,
                verbose=2 ** int(self.cfg.batch_mode),
            )

        return history
