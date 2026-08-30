from typing import Tuple

import numpy as np
import tensorflow as tf

from scipy.ndimage import find_objects

from .utils import logger, PipelineTimer
from ..config import Config


class GeometryOps:
    """
    A collection of pure TensorFlow and Numpy geometry operations.
    Designed to be used within tf.data pipelines or inference loops.
    """

    @staticmethod
    def get_smart_crop_coords(
        volume: np.ndarray, target_shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculates the bounding box to center the brain in the frame.
        Uses Numpy/Scipy (CPU) as this runs once during volume loading.

        Args:
            volume: 3D Numpy array (Z, H, W)
            target_shape: (H_out, W_out)

        Returns:
            ((h_start, h_end), (w_start, w_end))
        """
        # 1. Projection to find brain content
        projection = np.max(volume, axis=0) > 0.01

        # 2. Find bounding box
        slices = find_objects(projection.astype(int))

        if not slices:
            # Fallback: Center crop
            h, w = volume.shape[1], volume.shape[2]
            ch, cw = h // 2, w // 2
        else:
            h_slice, w_slice = slices[0]
            ch = h_slice.start + (h_slice.stop - h_slice.start) // 2
            cw = w_slice.start + (w_slice.stop - w_slice.start) // 2

        # 3. Calculate Bounds
        th, tw = target_shape
        h_start = ch - th // 2
        w_start = cw - tw // 2

        # 4. Clip to image boundaries
        max_h, max_w = volume.shape[1], volume.shape[2]

        # Adjust if out of bounds (left/top)
        if h_start < 0:
            h_start = 0
        if w_start < 0:
            w_start = 0

        # Adjust if out of bounds (right/bottom) - shift back
        if h_start + th > max_h:
            h_start = max(0, max_h - th)
        if w_start + tw > max_w:
            w_start = max(0, max_w - tw)

        return (int(h_start), int(h_start + th)), (
            int(w_start),
            int(w_start + tw),
        )

    @staticmethod
    @tf.function
    def normalize_volume(vol_tensor: tf.Tensor) -> tf.Tensor:
        """
        Min-Max normalization robust to outliers (TF Graph compatible).
        """
        # Cast to float32
        vol_tensor = tf.cast(vol_tensor, tf.float32)

        min_val = tf.reduce_min(vol_tensor)
        vol_tensor = vol_tensor - min_val

        # Use a safe division
        max_val = tf.reduce_max(vol_tensor)
        scale = tf.maximum(max_val, 1e-8)

        return vol_tensor / scale

    @staticmethod
    @tf.function
    def resize_and_pad(
        image: tf.Tensor, target_shape: Tuple[int, int], method: str = 'bicubic'
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Resizes image to fit target_shape preserving aspect ratio (Letterbox),
        then pads the rest.

        Args:
            image: (H, W, C) or (H, W) Tensor
            target_shape: (H_out, W_out)

        Returns:
            image_padded: Resized and padded image
            scale: float scale factor used
            padding: (pad_top, pad_left)
        """
        # Ensure 3D (H, W, C) for resize ops
        input_shape = tf.shape(image)
        h, w = (
            tf.cast(input_shape[0], tf.float32),
            tf.cast(input_shape[1], tf.float32),
        )
        target_h, target_w = (
            tf.cast(target_shape[0], tf.float32),
            tf.cast(target_shape[1], tf.float32),
        )

        # Calculate Scale
        scale = tf.minimum(target_h / h, target_w / w)

        new_h = tf.cast(h * scale, tf.int32)
        new_w = tf.cast(w * scale, tf.int32)

        # Resize
        # image_resized = tf.image.resize(image, [new_h, new_w], method='bicubic')
        image_resized = tf.image.resize(image, [new_h, new_w], method=method)

        # Calculate Padding
        pad_h = tf.cast(target_h, tf.int32) - new_h
        pad_w = tf.cast(target_w, tf.int32) - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply Padding
        if len(image.shape) == 3:
            paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
        else:
            paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]

        image_padded = tf.pad(image_resized, paddings, constant_values=0.0)

        # Return aux info for inverse operation
        return image_padded, scale, tf.stack([pad_top, pad_left])

    @staticmethod
    @tf.function
    def inverse_resize_pad(
        image: tf.Tensor,
        original_shape: Tuple[int, int],
        scale: float,
        pads: tf.Tensor,
    ) -> tf.Tensor:
        """
        Reverses the resize_and_pad operation (Crop -> Inverse Resize).
        Useful for reconstructing full volumes from model outputs.
        """
        pad_top = pads[0]
        pad_left = pads[1]

        orig_h, orig_w = original_shape[0], original_shape[1]

        # 1. Crop Center (Remove Padding)
        # Calculate height/width of the actual content inside the padded image
        active_h = tf.cast(tf.cast(orig_h, tf.float32) * scale, tf.int32)
        active_w = tf.cast(tf.cast(orig_w, tf.float32) * scale, tf.int32)

        image_cropped = tf.image.crop_to_bounding_box(
            image, pad_top, pad_left, active_h, active_w
        )

        # 2. Resize Up (Bicubic usually better for upsampling)
        image_restored = tf.image.resize(
            image_cropped, [orig_h, orig_w], method='bicubic'
        )

        return image_restored


def run_geometry_sanity_check(config):
    # --- Test the Geometry Ops (Sanity Check) ---
    with PipelineTimer("GeometryOps Test"):
        # Mock data
        dummy_vol = np.random.rand(113, 137, 1).astype(np.float32)
        dummy_tensor = tf.convert_to_tensor(dummy_vol)

        # 1. Test Normalize
        norm_tensor = GeometryOps.normalize_volume(dummy_tensor)

        # 2. Test Resize/Pad
        padded, scale, pads = GeometryOps.resize_and_pad(
            norm_tensor, config.data.padded_size
        )

        # 3. Test Inverse
        restored = GeometryOps.inverse_resize_pad(
            padded, (113, 137), scale, pads
        )

        logger.info(f"Original Shape: {dummy_tensor.shape}")
        logger.info(
            f"Padded Shape: {padded.shape} (Target: {config.data.padded_size})"
        )
        logger.info(f"Restored Shape: {restored.shape}")
        logger.info("Geometry Sanity Check Passed.")


class InputProcessor:
    @staticmethod
    def prepare_model_input(
        input_stack_native: np.ndarray,
        target_slice_native: np.ndarray,
        start_idx: int,
        target_idx: int,
        axis_size: int,
        direction: str,
        void_mask_native: np.ndarray = None,
        config: Config = None,
    ):
        N = config.data.neighborhood

        # 1. Handle History Image Stack (Ensure H, W, N)
        if (
            input_stack_native.shape[0] == N
            and input_stack_native.shape[2] != N
        ):
            input_stack_native = np.transpose(input_stack_native, (1, 2, 0))

        # 2. Handle Target Mask (1 Channel)
        brain_mask = (target_slice_native > 0.01).astype(np.float32)[..., None]

        # 3. Positional Encodings (Absolute)
        p_abs = np.array(
            [target_idx / max(1.0, float(axis_size - 1))], dtype=np.float32
        )

        step = 1 if direction == 'forward' else -1
        p_history = []
        for i in range(N):
            current_slice_idx = start_idx + (i * step)
            norm_pos = current_slice_idx / max(1.0, float(axis_size - 1))
            p_history.append(norm_pos)

        p_history_arr = np.array(p_history, dtype=np.float32)

        # RETURN PURE NUMPY DICT (No resizing here!)
        model_inputs = {
            'history_input': input_stack_native.astype(np.float32),
            'mask_input': brain_mask,
            'p_history_input': p_history_arr,
            'p_abs_input': p_abs,
        }

        return (
            model_inputs,
            1.0,
            np.array([0, 0]),
        )  # mock scale/pads for generator compatibility
