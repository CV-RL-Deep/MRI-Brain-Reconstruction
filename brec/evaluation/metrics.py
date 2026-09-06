import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from ..core.utils import logger


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


class Pillar1Diagnostics:
    """
    Mathematical calibration and empirical validation tools for Pillar 1.
    """

    @staticmethod
    def estimate_baseline_innovation_variance(
        model: tf.keras.Model,
        val_generator,
        num_batches: int = 30,
    ) -> float:
        """
        Measures baseline single-step residual innovation standard deviation:
        sigma_0 = sqrt(E[ (x_t* - f_theta(C_t*))^2 ]) on clean validation pairs.
        """
        residuals = []
        for i, (x_batch, y_batch) in enumerate(val_generator):
            if i >= num_batches:
                break
            preds = model(x_batch, training=False)
            gt = y_batch[..., 0:1]
            if preds.shape[-1] > 1:
                preds = preds[..., 0:1]
            diff = (gt - preds).numpy().flatten()
            residuals.extend(diff)

        sigma_0 = float(np.std(residuals))
        logger.info(
            f"Empirically Estimated Baseline Innovation sigma_0: {sigma_0:.6f}"
        )
        return sigma_0

    @staticmethod
    def estimate_spatial_diffusion_coefficient(
        volumes: list[np.ndarray],
        delta_z: float = 1.0,
    ) -> float:
        """
        Estimates D_spatial from clean ground-truth slice sequences using discrete 2D Laplacians:
        D_spatial = E_z ||laplacian(x_z) - laplacian(x_{z-delta_z})||_2^2 / (2 * delta_z * E_z ||laplacian(x_z)||_2^2)
        """
        from scipy.ndimage import laplace

        numerator_sum = 0.0
        denominator_sum = 0.0

        for vol in volumes:
            Z = vol.shape[0]
            for z in range(1, Z):
                slice_curr = vol[z]
                slice_prev = vol[z - 1]

                # Background exclusion mask
                mask = (slice_curr > 0.01) | (slice_prev > 0.01)
                if np.sum(mask) < 100:
                    continue

                lap_curr = laplace(slice_curr)
                lap_prev = laplace(slice_prev)

                diff_sq = np.sum((lap_curr[mask] - lap_prev[mask]) ** 2)
                energy_sq = np.sum((lap_curr[mask]) ** 2)

                numerator_sum += diff_sq
                denominator_sum += energy_sq

        if denominator_sum < 1e-12:
            return 0.045

        d_spatial = float(numerator_sum / (2.0 * delta_z * denominator_sum))
        logger.info(
            f"Empirically Estimated Spatial Diffusion Coefficient D_spatial: {d_spatial:.6f}"
        )
        return d_spatial

    @staticmethod
    def estimate_empirical_lipschitz(
        model: tf.keras.Model,
        val_batch: dict,
        perturbation_std: float = 0.02,
        num_trials: int = 10,
    ) -> float:
        """
        Evaluates empirical Lipschitz constant ratio:
        L_hat = ||f_theta(C + Delta C) - f_theta(C)||_2 / ||Delta C||_2
        """
        ratios = []
        hist_input = val_batch["history_input"]

        for _ in range(num_trials):
            delta_c = tf.random.normal(
                tf.shape(hist_input), mean=0.0, stddev=perturbation_std
            )
            perturbed_batch = {k: tf.identity(v) for k, v in val_batch.items()}
            perturbed_batch["history_input"] = hist_input + delta_c

            out_clean = model(val_batch, training=False)
            out_perturbed = model(perturbed_batch, training=False)

            if out_clean.shape[-1] > 1:
                out_clean = out_clean[..., 0:1]
                out_perturbed = out_perturbed[..., 0:1]

            norm_delta_out = tf.norm(
                tf.reshape(out_perturbed - out_clean, [-1])
            )
            norm_delta_in = tf.norm(tf.reshape(delta_c, [-1]))

            ratio = float(norm_delta_out / tf.maximum(norm_delta_in, 1e-12))
            ratios.append(ratio)

        l_hat = float(np.mean(ratios))
        logger.info(f"Measured Empirical Lipschitz Constant L_hat: {l_hat:.4f}")
        return l_hat

    @staticmethod
    def compute_stepwise_error_variance(
        gt_vol: np.ndarray,
        pred_vol: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> list[dict]:
        """
        Calculates per-step spatial error variance Var(delta_k) along the rollout trajectory.
        """
        results = []
        for i, z_idx in enumerate(range(start_idx, end_idx)):
            k = i + 1
            gt_slice = gt_vol[z_idx]
            pr_slice = pred_vol[z_idx]

            mask = gt_slice > 0.01
            if not np.any(mask):
                continue

            delta = (gt_slice[mask] - pr_slice[mask]).flatten()
            variance = float(np.var(delta))
            mae = float(np.mean(np.abs(delta)))

            results.append(
                {
                    "Rollout_Step_k": k,
                    "Z_Index": z_idx,
                    "Error_Variance": variance,
                    "MAE": mae,
                }
            )
        return results
