import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calc_focal_frequency_error(
    gt_slice: np.ndarray, pred_slice: np.ndarray, alpha: float = 1.0
) -> float:
    """
    Computes the Focal Frequency Loss (FFL) as an evaluation metric.
    Numpy equivalent of the TF implementation to prevent memory leaks during inference.
    """
    # 1. Compute 2D FFT (Complex domain)
    fft_gt = np.fft.fft2(gt_slice)
    fft_pr = np.fft.fft2(pred_slice)

    # 2. Shift zero-frequency component to center
    fft_gt = np.fft.fftshift(fft_gt)
    fft_pr = np.fft.fftshift(fft_pr)

    # 3. Calculate complex difference matrix d(u,v) = |F_r - F_f|^2
    diff_matrix = np.abs(fft_gt - fft_pr) ** 2

    # 4. Normalize difference matrix to [0, 1] for focal weighting
    max_diff = np.max(diff_matrix)
    if max_diff < 1e-8:
        return 0.0

    diff_norm = diff_matrix / max_diff

    # 5. Calculate focal weights w(u,v) = d_norm(u,v)^alpha
    weight_matrix = diff_norm**alpha

    # 6. Compute final weighted frequency loss
    ffl = np.mean(weight_matrix * diff_matrix)

    return float(ffl)


def calc_gradient_sharpness_error(gt_slice, pred_slice):
    """Numpy equivalent of your custom GradientSharpnessMetric"""
    dy_gt, dx_gt = np.gradient(gt_slice)
    dy_pr, dx_pr = np.gradient(pred_slice)
    grad_diff = np.abs(dy_gt - dy_pr) + np.abs(dx_gt - dx_pr)
    return float(np.mean(grad_diff))


def calculate_step_metrics(
    gt_vol, pred_vol, start_idx, end_idx, direction, ablation_name, vol_id
):
    """Evaluates an autoregressive rollout step-by-step (k)."""
    metrics_list = []
    iter_range = (
        range(start_idx, end_idx)
        if direction == 'forward'
        else range(end_idx - 1, start_idx - 1, -1)
    )

    for i, z_idx in enumerate(iter_range):
        k = i + 1  # rollout step (1, 2, 3...)
        gt_slice = gt_vol[z_idx]
        pr_slice = pred_vol[z_idx]

        mae = float(np.mean(np.abs(gt_slice - pr_slice)))
        mse = float(np.mean(np.square(gt_slice - pr_slice)))
        psnr_val = (
            float(psnr(gt_slice, pr_slice, data_range=1.0))
            if mse > 0
            else 100.0
        )
        ssim_val = float(ssim(gt_slice, pr_slice, data_range=1.0))
        grad_err = calc_gradient_sharpness_error(gt_slice, pr_slice)
        ffl_err = calc_focal_frequency_error(gt_slice, pr_slice, alpha=1.0)

        metrics_list.append(
            {
                'Ablation': ablation_name,
                'Volume_ID': vol_id,
                'Direction': direction,
                'Rollout_Step_k': k,
                'Z_Index': z_idx,
                'SSIM': ssim_val,
                'PSNR': psnr_val,
                'MAE': mae,
                'Grad_Error': grad_err,
                'FFL': ffl_err,  # <--- NEW LOGIC
            }
        )
    return metrics_list
