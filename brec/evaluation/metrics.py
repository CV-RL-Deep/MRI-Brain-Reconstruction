import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


@tf.function
def get_oracle_prediction(y_true, y_pred):
    """
    Extracts the best hypothesis (Oracle) from M predictions using physical gathering.
    This strips M-channel metadata to prevent SSIM XLA errors.
    """
    # 1. Slice GT Image immediately
    gt_img = y_true[..., 0:1]

    # 2. Get Hypotheses Count
    # Using .shape[-1] handles static, tf.shape handles dynamic
    M = y_pred.shape[-1]
    if M is None:
        M = tf.shape(y_pred)[-1]

    if M == 1:
        return y_pred

    # 3. Find the index of the best hypothesis for each batch item
    # mae_per_hyp shape: (Batch, M)
    mae_per_hyp = tf.reduce_mean(tf.abs(gt_img - y_pred), axis=[1, 2])
    best_indices = tf.argmin(mae_per_hyp, axis=1)  # (Batch,)

    # 4. PHYSICAL GATHER (The Fix)
    # Instead of one-hot multiplication, we gather the slices.
    # This forces the compiler to treat the result as a fresh 1-channel tensor.
    batch_size = tf.shape(y_pred)[0]
    batch_range = tf.range(batch_size, dtype=best_indices.dtype)

    # Create indices for gather_nd: [[batch_0, hyp_idx], [batch_1, hyp_idx], ...]
    indices = tf.stack([batch_range, best_indices], axis=1)

    # Transpose y_pred to (Batch, M, H, W) to gather across M
    y_trans = tf.transpose(y_pred, [0, 3, 1, 2])
    oracle_slices = tf.gather_nd(y_trans, indices)  # result: (Batch, H, W)

    # Restore channel dimension
    return tf.expand_dims(oracle_slices, axis=-1)  # result: (Batch, H, W, 1)


class GradientSharpnessMetric(tf.keras.metrics.Metric):
    def __init__(self, name='grad_diff', **kwargs):
        super().__init__(name=name, **kwargs)
        self.diff_sum = self.add_weight(name='diff_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        gt_img = y_true[..., 0:1]
        oracle_pred = get_oracle_prediction(y_true, y_pred)  # <--- MHP FIX
        dy_true, dx_true = tf.image.image_gradients(gt_img)
        dy_pred, dx_pred = tf.image.image_gradients(oracle_pred)
        grad_diff = tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred)
        batch_mean = tf.reduce_mean(grad_diff, axis=(1, 2, 3))
        self.diff_sum.assign_add(tf.reduce_sum(batch_mean))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.diff_sum / self.count

    def reset_state(self):
        self.diff_sum.assign(0.0)
        self.count.assign(0.0)


class OracleMAE(tf.keras.metrics.Metric):
    def __init__(self, name="oracle_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tracker = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        gt_img = y_true[..., 0:1]  # <--- CRITICAL SLICE
        oracle_pred = get_oracle_prediction(y_true, y_pred)

        # Calculate MAE
        batch_mae = tf.reduce_mean(tf.abs(gt_img - oracle_pred), axis=[1, 2, 3])
        self.tracker.update_state(batch_mae)

    def result(self):
        return self.tracker.result()

    def reset_states(self):
        self.tracker.reset_states()


class OracleMSE(tf.keras.metrics.Metric):
    def __init__(self, name="oracle_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tracker = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        gt_img = y_true[..., 0:1]  # <--- CRITICAL SLICE
        oracle_pred = get_oracle_prediction(y_true, y_pred)

        # Calculate MSE
        batch_mse = tf.reduce_mean(
            tf.square(gt_img - oracle_pred), axis=[1, 2, 3]
        )
        self.tracker.update_state(batch_mse)

    def result(self):
        return self.tracker.result()

    def reset_states(self):
        self.tracker.reset_states()


class OraclePSNR(tf.keras.metrics.Metric):
    def __init__(self, name="oracle_psnr", max_val=1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.tracker = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        gt_img = y_true[..., 0:1]  # <--- CRITICAL SLICE
        oracle_pred = get_oracle_prediction(y_true, y_pred)

        # We tell the graph compiler that the channel dimension is exactly 1.
        # The None values represent the dynamic Batch, Height, and Width
        # oracle_pred.set_shape([None, None, None, 1])
        # gt_img.set_shape([None, None, None, 1])

        # Calculate PSNR
        psnr_values = tf.image.psnr(gt_img, oracle_pred, max_val=self.max_val)
        self.tracker.update_state(psnr_values)

    def result(self):
        return self.tracker.result()

    def reset_states(self):
        self.tracker.reset_states()


class OracleSSIM(tf.keras.metrics.Metric):
    def __init__(self, name="oracle_ssim", max_val=1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = tf.cast(max_val, tf.float32)
        self.tracker = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 1. Cast everything to float32 (SSIM is very sensitive to this)
        gt_img = tf.cast(y_true[..., 0:1], tf.float32)
        oracle_pred = tf.cast(get_oracle_prediction(y_true, y_pred), tf.float32)

        # 2. THE DYNAMIC RESHAPE (No hardcoding)
        # We fetch the dynamic shape and force the channel to 1.
        # This acts as a "hard reset" for the graph compiler.
        dyn_shape = tf.shape(oracle_pred)
        gt_img = tf.reshape(
            gt_img, [dyn_shape[0], dyn_shape[1], dyn_shape[2], 1]
        )
        oracle_pred = tf.reshape(
            oracle_pred, [dyn_shape[0], dyn_shape[1], dyn_shape[2], 1]
        )

        # 3. Calculate SSIM
        ssim_values = tf.image.ssim(gt_img, oracle_pred, max_val=self.max_val)

        self.tracker.update_state(ssim_values)

    def result(self):
        return self.tracker.result()

    def reset_states(self):
        self.tracker.reset_states()


class Evaluator:
    """
    Handles calculation of metrics and plotting of results.
    """

    @staticmethod
    def calculate_region_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
    ):
        """
        Calculates MAE, MSE, PSNR, SSIM for a specific region defined by 'mask'.
        """
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return {'mae': 0.0, 'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0}

        true_region = y_true[mask_bool]
        pred_region = y_pred[mask_bool]

        mae = np.mean(np.abs(true_region - pred_region))
        mse = np.mean(np.square(true_region - pred_region))

        # PSNR
        psnr_val = (
            psnr(true_region, pred_region, data_range=1.0) if mse > 0 else 100.0
        )

        # SSIM (Calculated on bounding box of the region to be valid)
        rows, cols = np.where(mask_bool)
        r_min, r_max = np.min(rows), np.max(rows)
        c_min, c_max = np.min(cols), np.max(cols)

        if (r_max - r_min < 7) or (c_max - c_min < 7):
            ssim_val = 0.0
        else:
            # Crop to bbox
            bbox_true = y_true[r_min:r_max, c_min:c_max]
            bbox_pred = y_pred[r_min:r_max, c_min:c_max]
            ssim_val = ssim(bbox_true, bbox_pred, data_range=1.0)

        return {'mae': mae, 'mse': mse, 'psnr': psnr_val, 'ssim': ssim_val}

    @staticmethod
    def plot_ortho_slices(volume: np.ndarray, title: str = "Orthogonal Views"):
        """Plots Center slices for Axial, Coronal, and Sagittal planes."""
        c_x, c_y, c_z = np.array(volume.shape) // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        # Axial (XY)
        axes[0].imshow(volume[c_x, :, :], cmap='bone')
        axes[0].set_title(f"Axial (Slice {c_x})")

        # Coronal (XZ) -> In numpy (Z, Y, X), this is (:, Y, :)
        axes[1].imshow(np.rot90(volume[:, c_y, :]), cmap='bone')
        axes[1].set_title(f"Coronal (Slice {c_y})")

        # Sagittal (YZ) -> (:, :, X)
        axes[2].imshow(np.rot90(volume[:, :, c_z]), cmap='bone')
        axes[2].set_title(f"Sagittal (Slice {c_z})")

        for ax in axes:
            ax.axis('off')
        plt.show()

    @staticmethod
    def visualize_restoration(gt, pred, mask=None, slice_idx=None):
        """Side-by-side comparison of GT, Pred, and Difference."""
        if slice_idx is None:
            slice_idx = gt.shape[0] // 2

        g_slice = gt[slice_idx]
        p_slice = pred[slice_idx]
        diff = np.abs(g_slice - p_slice)

        cols = 4 if mask is not None else 3
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

        axes[0].imshow(g_slice, cmap='bone')
        axes[0].set_title("Ground Truth")

        axes[1].imshow(p_slice, cmap='bone')
        axes[1].set_title("Restoration")

        axes[2].imshow(diff, cmap='inferno', vmin=0, vmax=0.3)
        axes[2].set_title("Error Map")

        if mask is not None:
            # Overlay mask on GT
            axes[3].imshow(g_slice, cmap='bone')
            axes[3].contour(mask[slice_idx], levels=[0.5], colors='red')
            axes[3].set_title("Tumor Mask Overlay")

        for ax in axes:
            ax.axis('off')
        plt.show()
