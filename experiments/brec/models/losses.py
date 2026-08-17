import tensorflow as tf

from tensorflow.keras import applications, models

from src.core.utils import logger
from configs.config import Config


class PerceptualLoss(tf.keras.losses.Loss):
    """
    Computes distance in EfficientNet feature space.
    Reusable component for U-Net and GAN training.
    """
    def __init__(self, name='perceptual_loss', backbone='effnet',
                 weights='imagenet', input_shape=(None, None, 3)):
        # tf.keras.backend.clear_session() # may cause memory leak
        super().__init__(name=name)

        self.backbone = backbone

        if self.backbone == 'vgg':
            logger.info("Initializing VGG19 Perceptual Model...")

            # Use VGG19. It avoids conflict with EfficientNet generator backbones.
            # VGG expects (224, 224, 3) but works with (None, None, 3)
            base = applications.VGG19(
                include_top=False, weights=weights, # input_shape=(None, None, 3)
            )

        else:
            # --- Eager Initialization of Perceptual Model ---
            # We initialize this HERE (in eager mode) so it's ready before model.fit() builds the graph.
            # Otherwise, loading weights inside the training loop causes 'InaccessibleTensorError'
            logger.info("Initializing EfficientNetB0 Perceptual Model...")

            # Load backbone without top layers
            base = applications.EfficientNetB0(
                # name='effnet_perceptual',
                include_top=False,
                weights=weights, # 'imagenet'
            )

        base.trainable = False

        if self.backbone == 'vgg':
            # Standard layers for style/texture loss:
            # block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1
            layer_names = [
                'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
            ]
            outputs = [base.get_layer(name).output for name in layer_names]

            self.model = models.Model(base.input, outputs, name="extractor_vgg")
        else:
            # Select specific feature layers for texture comparison
            # 'block2a' and 'block3a' capture low-to-mid level features (edges, textures):
            # 'block2a' (low-level texture) and 'block3a' (mid-level shape)
            outputs = [
                base.get_layer(n).output for n in [
                    'block2a_expand_activation', 'block3a_expand_activation'
                ]
            ]

            self.model = models.Model(base.input, outputs,
                                      name="extractor_effnet")

    def call(self, y_true, y_pred):
        # Expects single channel 0-1 inputs
        # Convert to 3-channel RGB 0-255 for EfficientNet
        true_rgb = tf.image.grayscale_to_rgb(y_true) * 255.0
        pred_rgb = tf.image.grayscale_to_rgb(y_pred) * 255.0

        # VGG Preprocessing: expects 0-255 BGR, but we have 0-1 RGB.
        # Simple shift: scale to 0-255.
        # Keras VGG preprocess_input does mean subtraction, let's do a rough approx
        # or use the official function if possible
        if self.backbone == 'vgg':
            # Official preprocessing (handles mean subtraction)
            true_rgb = applications.vgg19.preprocess_input(true_rgb)
            pred_rgb = applications.vgg19.preprocess_input(pred_rgb)

        feats_true = self.model(true_rgb)
        feats_pred = self.model(pred_rgb)

        loss = 0.0
        for f_t, f_p in zip(feats_true, feats_pred):
            loss += tf.reduce_mean(tf.abs(f_t - f_p))

        return loss


_GLOBAL_PERCEPTUAL_LOSS = None


def get_perceptual_loss(config: Config):
    global _GLOBAL_PERCEPTUAL_LOSS
    if _GLOBAL_PERCEPTUAL_LOSS is None:
        _GLOBAL_PERCEPTUAL_LOSS = PerceptualLoss(
            backbone=config.train.perceptual_backbone,
            weights=config.train.perceptual_init,
            input_shape=(*config.data.padded_size, 3)
        )
    return _GLOBAL_PERCEPTUAL_LOSS


class SpectralLoss(tf.keras.losses.Loss):
    """
    Computes the L1 distance between the Log-Magnitude of the Fourier Transforms.
    Forces the model to match the frequency distribution (sharpness/texture) of the target.
    """
    def __init__(self, name="spectral_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # 1. Cast to float32 (FFT requires it, and mixed precision might be float16)
        y_true_f32 = tf.cast(y_true[..., 0], tf.float32) # take 1st channel (Image)
        y_pred_f32 = tf.cast(y_pred[..., 0], tf.float32)

        # 2. Compute 2D Real FFT
        # Output shape: (Batch, H, W/2 + 1) complex64
        fft_true = tf.signal.rfft2d(y_true_f32)
        fft_pred = tf.signal.rfft2d(y_pred_f32)

        # 3. Compute Magnitude
        mag_true = tf.abs(fft_true)
        mag_pred = tf.abs(fft_pred)

        # 4. Log Scaling (Critical for balancing low/high freqs)
        # Add epsilon or 1.0 to avoid log(0)
        log_true = tf.math.log(mag_true + 1.0)
        log_pred = tf.math.log(mag_pred + 1.0)

        # 5. L1 Distance in Frequency Domain
        return tf.reduce_mean(tf.abs(log_true - log_pred))


class FocalFrequencyLoss(tf.keras.losses.Loss):
    """
    Focal Frequency Loss (FFL) for Image Reconstruction.
    Based on: https://arxiv.org/pdf/2012.12821
    Penalizes both magnitude and phase differences in the complex frequency domain,
    dynamically up-weighting frequencies where the model performs poorly.
    """
    def __init__(self, alpha: float = 1.0, name: str = "focal_frequency_loss"):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # 1. Cast to complex64 for FFT
        # Assuming inputs are (B, H, W, 1), we strip the channel dim for 2D FFT
        y_true_c = tf.cast(y_true[..., 0], tf.complex64)
        y_pred_c = tf.cast(y_pred[..., 0], tf.complex64)

        # 2. Compute 2D Complex FFT
        fft_true = tf.signal.fft2d(y_true_c)
        fft_pred = tf.signal.fft2d(y_pred_c)

        # 3. Shift zero-frequency component to center
        fft_true = tf.signal.fftshift(fft_true, axes=[1, 2])
        fft_pred = tf.signal.fftshift(fft_pred, axes=[1, 2])

        # 4. Calculate complex difference matrix d(u,v) = |F_r - F_f|^2
        # tf.abs on complex returns the magnitude (sqrt(real^2 + imag^2)).
        # Squaring it gives the squared distance
        diff_matrix = tf.square(tf.abs(fft_true - fft_pred))

        # 5. Normalize difference matrix to [0, 1] per image in the batch
        max_diff = tf.reduce_max(diff_matrix, axis=[1, 2], keepdims=True)
        # Safe division to prevent NaNs
        diff_norm = tf.math.divide_no_nan(diff_matrix, max_diff)

        # 6. Calculate focal weights w(u,v) = d_norm(u,v)^alpha
        weight_matrix = tf.pow(diff_norm, self.alpha)

        # 7. Compute final weighted frequency loss
        loss = tf.reduce_mean(weight_matrix * diff_matrix)

        return loss


class CompositeLoss(tf.keras.losses.Loss):
    def __init__(self, config: Config, name="composite_loss"):
        super().__init__(name=name)

        self.cfg = config.train

        # Only initialize the heavy perceptual model if we are actually going to use it!
        if self.cfg.lambda_perceptual >= 0.01:
            self.perceptual = get_perceptual_loss(config)
        else:
            self.perceptual = None

        self.spectral = SpectralLoss()

        # FIX: Convert static floats to dynamic TensorFlow Variables
        self.w_healthy = tf.Variable(self.cfg.lambda_healthy, dtype=tf.float32,
                                     trainable=False)
        self.w_tumor = tf.Variable(self.cfg.lambda_tumor, dtype=tf.float32,
                                   trainable=False)
        self.w_bg = tf.Variable(self.cfg.lambda_background, dtype=tf.float32,
                                trainable=False)
        self.w_grad = tf.Variable(self.cfg.lambda_grad, dtype=tf.float32,
                                  trainable=False)
        self.w_perc = tf.Variable(self.cfg.lambda_perceptual, dtype=tf.float32,
                                  trainable=False)
        self.w_spec = tf.Variable(self.cfg.lambda_spectral, dtype=tf.float32,
                                  trainable=False)

    def call(self, y_true, y_pred):
        """
        y_true: (B, H, W, 4) -> [Image, TumorMask, DetailedMask, BrainMask]
        y_pred: (B, H, W, 1)
        """
        # Unpack
        gt_img = y_true[..., 0:1]
        tumor_mask = y_true[..., 1:2]
        brain_mask = y_true[..., 3:4]

        # 1. Pixel Losses (MAE)
        mae = tf.abs(gt_img - y_pred)

        # Healthy Tissue (Inside brain, outside tumor)
        # Fix: Robust Division
        healthy_mask = tf.maximum(0.0, brain_mask - tumor_mask)
        loss_healthy = tf.math.divide_no_nan(
            tf.reduce_sum(mae * healthy_mask),
            tf.reduce_sum(healthy_mask)# + 1e-8
        )

        # Tumor Region (The Inpainting Target - Critical)
        # Fix: Robust Division prevents NaN when batch has no tumor
        loss_tumor = tf.math.divide_no_nan(
            tf.reduce_sum(mae * tumor_mask),
            tf.reduce_sum(tumor_mask)# + 1e-8
        )

        # Background
        bg_mask = 1.0 - brain_mask
        loss_bg = tf.math.divide_no_nan(
            tf.reduce_sum(mae * bg_mask),
            tf.reduce_sum(bg_mask)# + 1e-8
        )

        # 2. Gradient Loss (Sharpness)
        dy_true, dx_true = tf.image.image_gradients(gt_img)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        grad_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred) +
                                   tf.abs(dx_true - dx_pred))

        # 3. Perceptual Loss (Conditional Bypass)
        if self.perceptual is not None:
            perc_loss = self.perceptual(gt_img, y_pred)
        else:
            perc_loss = tf.constant(0.0, dtype=tf.float32)

        # 4. Spectral Loss
        spec_loss = self.spectral(y_true, y_pred)

        # Combine weighted losses
        # FIX: Multiply by the dynamic Variables
        total = (self.w_healthy * loss_healthy +
                 self.w_tumor * loss_tumor +
                 self.w_bg * loss_bg +
                 self.w_grad * grad_loss +
                 self.w_perc * perc_loss +
                 self.w_spec * spec_loss)

        return total


class SpatiallyWeightedL1Loss(tf.keras.losses.Loss):
    """
    L1 Loss that applies heavy penalties to Brain and Tumor regions.
    Ported from SPADE GAN logic: weights = 1 (Bg) + 14 (Brain) + 10 (Tumor).
    """
    def __init__(self, config: Config, name="spatial_l1_loss"):
        super().__init__(name=name)
        self.w_brain_add = 14.0 # base 1 + 14 = 15
        self.w_tumor_add = 10.0 # 15 + 10 = 25 for tumor regions

    def call(self, y_true, y_pred):
        # Unpack
        gt_img = y_true[..., 0:1]
        tumor_mask = y_true[..., 1:2]
        brain_mask = y_true[..., 3:4]

        # Absolute Error
        l1_error = tf.abs(gt_img - y_pred)

        # Construct Weight Map
        # Start with Background (1.0)
        weights = tf.ones_like(l1_error)

        # Add Brain Penalty
        weights = weights + (brain_mask * self.w_brain_add)

        # Add Tumor Penalty
        weights = weights + (tumor_mask * self.w_tumor_add)

        # Weighted Mean
        return tf.reduce_mean(l1_error * weights)


class VanillaL1Loss(tf.keras.losses.Loss):
    """
    Standard L1 Loss without any spatial or region-based weighting.
    Used to demonstrate unbounded spatial drift in unconstrained baseline models.
    """
    def __init__(self, name="vanilla_l1_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Extract only the target image slice (Channel 0)
        gt_img = y_true[..., 0:1]

        # Strip M-channel metadata if multiple hypotheses are somehow activated
        if y_pred.shape[-1] is not None and y_pred.shape[-1] > 1:
            pred_img = y_pred[..., 0:1]
        else:
            pred_img = y_pred

        return tf.reduce_mean(tf.abs(gt_img - pred_img))


class RelaxedMHPLossWrapper(tf.keras.losses.Loss):
    def __init__(self, base_loss_fn, num_hypotheses=1, epsilon=0.05, **kwargs):
        # We handle reduction manually
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.base_loss_fn = base_loss_fn
        self.M = num_hypotheses
        self.epsilon = epsilon

        if self.M > 1:
            self.winner_weight = 1.0 - epsilon
            self.loser_weight = epsilon / (self.M - 1)

    def call(self, y_true, y_pred):
        # BASELINE FALLBACK:
        if self.M == 1:
            return self.base_loss_fn(y_true, y_pred)

        # MULTIPLE HYPOTHESES LOGIC:
        # y_true shape: (B, H, W, 1)
        # y_pred shape: (B, H, W, M)

        # 1. Unstack the M predictions into a list of M tensors of shape (B, H, W)
        preds = tf.unstack(y_pred, axis=-1)

        # 2. Calculate base CompositeLoss for each hypothesis
        # self.base_loss_fn should return shape (B,)
        losses =[]
        for p in preds:
            p_expanded = tf.expand_dims(p, -1) # Restore channel dim -> (B, H, W, 1)
            losses.append(self.base_loss_fn(y_true, p_expanded))

        # Stack losses -> Shape: (M, Batch)
        losses_tensor = tf.stack(losses, axis=0)

        # 3. Find the best hypothesis for each batch item (The Oracle)
        best_indices = tf.argmin(losses_tensor, axis=0) # shape: (Batch,)

        # 4. Create masks for winners and losers
        winner_mask = tf.one_hot(best_indices, depth=self.M) # shape: (Batch, M)
        winner_mask = tf.transpose(winner_mask)              # shape: (M, Batch)

        # 5. Apply Epsilon Relaxation
        weights = (winner_mask * self.winner_weight) + ((1.0 - winner_mask) * self.loser_weight)

        # 6. Weight the losses and sum across the M dimension
        relaxed_losses = tf.reduce_sum(losses_tensor * weights, axis=0) # shape: (Batch,)

        return relaxed_losses
