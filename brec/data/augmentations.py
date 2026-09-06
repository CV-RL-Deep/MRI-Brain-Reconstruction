import random

from typing import Tuple, Dict, Any

import numpy as np

from scipy.ndimage import gaussian_filter, zoom, rotate

from ..configs.config import AugmentationConfig


class AugmentationLogic:
    """
    Fast NumPy/SciPy implementations for CPU-based Data Generators.
    Eliminates TF-Eager execution overhead during data loading.
    """

    @staticmethod
    def apply_geometric(input_stack, target_slice, target_masks, config):
        """
        Applies consistent Flip and Rotation to inputs and targets.
        input_stack: (C, H, W)
        target_slice: (H, W)
        target_masks: list of (H, W) masks (tumor, brain_mask)
        """
        # 1. Flip Left-Right (Axis 2)
        if random.random() < config.prob_flip:
            # Numpy flip is fast
            input_stack = np.flip(input_stack, axis=2)
            target_slice = np.flip(target_slice, axis=1)
            target_masks = [np.flip(m, axis=1) for m in target_masks]

        # 2. Rotation
        if random.random() < config.prob_rotate:
            angle = random.uniform(-config.rotate_range, config.rotate_range)

            # Rotate Input (C, H, W) - Rotate axes 1, 2
            input_stack = rotate(
                input_stack,
                angle,
                axes=(1, 2),
                reshape=False,
                order=1,
                mode='constant',
                cval=0.0,
            )

            # Rotate Target (H, W) - Standard
            target_slice = rotate(
                target_slice,
                angle,
                axes=(0, 1),
                reshape=False,
                order=1,
                mode='constant',
                cval=0.0,
            )

            # Rotate Masks (H, W) - Nearest Neighbor
            target_masks = [
                rotate(
                    m,
                    angle,
                    axes=(0, 1),
                    reshape=False,
                    order=0,
                    mode='constant',
                    cval=0.0,
                )
                for m in target_masks
            ]

        return input_stack, target_slice, target_masks

    @staticmethod
    def apply_autoregressive_corruption(
        input_stack: np.ndarray, config: AugmentationConfig
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
        stack = input_stack.copy()
        flags = {'blur_applied': False, 'noise_applied': False}
        num_slices = stack.shape[0]  # [t-N ... t-1]

        # Decide if the SEQUENCE is corrupted starting from the most recent
        # We iterate backwards: t-1 -> t-N

        current_mode = None

        # Check t-1 (Most recent history)
        if random.random() < config.prob_autoregressive_candidate:
            current_mode = random.choice(['blur', 'noise', 'both'])
        else:
            return stack, flags  # clean sequence

        # Apply cascade
        # Slice index: num_slices-1 is t-1
        for i in range(num_slices - 1, -1, -1):
            # Apply current mode
            slice_data = stack[i]

            # Apply based on current mode
            apply_blur = (current_mode == 'blur') or (current_mode == 'both')
            apply_noise = (current_mode == 'noise') or (current_mode == 'both')

            if apply_blur:
                slice_data = gaussian_filter(
                    slice_data, sigma=config.blur_sigma
                )
                flags['blur_applied'] = True

            if apply_noise:
                mask = (slice_data > 0.01).astype(np.float32)
                noise = np.random.normal(0, config.noise_std, slice_data.shape)
                slice_data = slice_data + (noise * mask)
                flags['noise_applied'] = True

            stack[i] = np.clip(slice_data, 0.0, 1.0)

            # Degrade mode for next (older) slice
            # If current is 'both', next can stay 'both' or drop to 'blur'/'noise'
            # If current is single, it stays single
            if current_mode == 'both':
                current_mode = random.choice(['blur', 'noise', 'both'])

            # Check if cascade stops (error started at t-X, so t-(X-1) is clean)
            if random.random() > config.prob_autoregressive_candidate:
                break

        return stack, flags

    @staticmethod
    def apply_tumor_void(
        input_stack: np.ndarray,
        manager: Any,  # changed type hint
        brats_files: list,
        config: AugmentationConfig,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Injects a BraTS tumor mask as a void.
        Note: This is legacy logic used by Static Generator. Active Gen uses direct method below.
        """
        if not brats_files:
            return input_stack, np.zeros_like(input_stack[0]), False

        # 1. Fetch Mask
        b_subj = random.choice(brats_files)

        # Check if manager is Active or Static/Legacy
        if hasattr(manager, 'get_brats_pair'):
            _, seg_vol = manager.get_brats_pair(b_subj['t1'], b_subj['seg'])
        else:
            # StaticLoader doesn't have get_brats_pair, it has get_volume('brats')
            # But this method is designed for file-based access.
            # If using StaticLoader, we should use apply_tumor_void_direct.
            # This method is kept for compatibility if цу revert to file-based logic
            return input_stack, np.zeros_like(input_stack[0]), False

        if seg_vol is None:
            return input_stack, np.zeros_like(input_stack[0]), False

        # 2. Select Random Crop
        z_input, h_in, w_in = input_stack.shape
        z_seg = seg_vol.shape[0]

        if z_seg <= z_input:
            return input_stack, np.zeros_like(input_stack[0]), False

        start_z = random.randint(0, z_seg - z_input - 1)
        seg_stack = seg_vol[start_z : start_z + z_input + 1]

        if not np.any(np.isin(seg_stack, config.brats_labels_for_void)):
            return input_stack, np.zeros_like(input_stack[0]), False

        # 3. Resize
        scale_h = h_in / seg_stack.shape[1]
        scale_w = w_in / seg_stack.shape[2]
        seg_resized = zoom(seg_stack, (1, scale_h, scale_w), order=0)

        # 4. Crop/Pad safety
        if seg_resized.shape[1:] != (h_in, w_in):
            temp = np.zeros((z_input + 1, h_in, w_in), dtype=seg_resized.dtype)
            min_h = min(seg_resized.shape[1], h_in)
            min_w = min(seg_resized.shape[2], w_in)
            temp[:, :min_h, :min_w] = seg_resized[:, :min_h, :min_w]
            seg_resized = temp

        # 5. Masking
        input_seg = seg_resized[:z_input]
        target_seg = seg_resized[z_input]

        is_tumor = np.isin(input_seg, config.brats_labels_for_void)
        input_stack[is_tumor] = 0.0
        target_tumor_mask = np.isin(
            target_seg, config.brats_labels_for_void
        ).astype(np.float32)

        return input_stack, target_tumor_mask, True

    @staticmethod
    def apply_tumor_void_direct(
        input_stack: np.ndarray,
        seg_stack: np.ndarray,
        config: AugmentationConfig,
    ):
        """
        Applies a pre-extracted N+1 segmentation stack as a void to the input stack.
        seg_stack shape: (N+1, H_seg, W_seg)
        input_stack shape: (N, H_in, W_in)
        """
        z_input, h_in, w_in = input_stack.shape

        # 1. Verify sizes match our logic (seg_stack should be exactly 1 slice larger for the target)
        if seg_stack.shape[0] != z_input + 1:
            return input_stack, np.zeros_like(input_stack[0]), False

        # 2. Fast exit if there are no tumor labels in the provided stack at all
        if not np.any(np.isin(seg_stack, config.brats_labels_for_void)):
            return input_stack, np.zeros_like(input_stack[0]), False

        # 3. Spatial Resizing (Matching bounding boxes across different brains)
        scale_h = h_in / seg_stack.shape[1]
        scale_w = w_in / seg_stack.shape[2]

        # Use nearest neighbor (order=0) to preserve integer label classes
        seg_resized = zoom(seg_stack, (1, scale_h, scale_w), order=0)

        # 4. Safety padding/cropping to guarantee exact (N+1, H_in, W_in) shape
        if seg_resized.shape[1:] != (h_in, w_in):
            temp = np.zeros((z_input + 1, h_in, w_in), dtype=seg_resized.dtype)
            min_h = min(seg_resized.shape[1], h_in)
            min_w = min(seg_resized.shape[2], w_in)
            temp[:, :min_h, :min_w] = seg_resized[:, :min_h, :min_w]
            seg_resized = temp

        # 5. Split into Context (t-N ... t-1) and Target (t)
        input_seg = seg_resized[:z_input]
        target_seg = seg_resized[z_input]

        target_tumor_mask = np.isin(
            target_seg, config.brats_labels_for_void
        ).astype(np.float32)

        # FIX: Abort if the target slice has no tumor pixels.
        # (Otherwise the model learns to erase an input without reconstructing
        # anything on the target)
        if not np.any(target_tumor_mask):
            return input_stack, np.zeros_like(input_stack[0]), False

        # --- THE FIX: TISSUE OVERLAP CHECK ---
        is_tumor = np.isin(input_seg, config.brats_labels_for_void)

        # A. Create a rough mask of actual brain tissue in the input stack
        brain_tissue = input_stack > 0.01

        # B. Check where the tumor mask actually intersects with brain tissue
        actual_void_overlap = is_tumor & brain_tissue

        # C. Require a meaningful visible void (e.g., at least 10 pixels erased)
        # This prevents both the "Air Void" and the "Surprise Tumor"
        if np.sum(actual_void_overlap) < 10:
            return input_stack, np.zeros_like(input_stack[0]), False

        # -------------------------------------

        # 6. Apply the void
        input_stack = (
            input_stack.copy()
        )  # Only copy right before modifying to save RAM
        is_tumor = np.isin(input_seg, config.brats_labels_for_void)
        input_stack[is_tumor] = 0.0

        return input_stack, target_tumor_mask, True
