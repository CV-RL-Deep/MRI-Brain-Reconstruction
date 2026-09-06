from collections import deque

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from ..configs.config import Config
from ..core.utils import logger
from ..core.geometry import GeometryOps, InputProcessor


class VolumeReconstructor:
    """
    Engine for applying the trained model to full 3D volumes.
    Supports Autoregressive and Bidirectional strategies.
    """

    def __init__(self, model, config: Config):
        self.model = model
        self.cfg = config

    def _predict_slice(
        self,
        input_stack_native,
        target_slice_native,
        start_idx,
        target_idx,
        axis_size,
        direction,
        mask_volume=None,
    ):

        void_mask = (
            (mask_volume[target_idx] > 0).astype(np.float32)
            if mask_volume is not None
            else None
        )

        model_inputs, scale, pads = InputProcessor.prepare_model_input(
            input_stack_native,
            target_slice_native,
            start_idx,
            target_idx,
            axis_size,
            direction,
            void_mask_native=void_mask,
            config=self.cfg,
        )

        hist_pad, scale, pads = GeometryOps.resize_and_pad(
            tf.convert_to_tensor(model_inputs['history_input']),
            self.cfg.data.padded_size,
            'bicubic',
        )
        mask_pad, _, _ = GeometryOps.resize_and_pad(
            tf.convert_to_tensor(model_inputs['mask_input']),
            self.cfg.data.padded_size,
            'nearest',
        )
        model_inputs['history_input'] = hist_pad
        model_inputs['mask_input'] = mask_pad

        x_batch = {k: tf.expand_dims(v, 0) for k, v in model_inputs.items()}

        # Returns shape (1, H_pad, W_pad, M)
        pred_padded_raw = self.model(x_batch, training=False)[0]

        native_h, native_w = target_slice_native.shape

        # Inverse resize ALL M hypotheses at once!
        # Returns shape (H_native, W_native, M)
        pred_restored = GeometryOps.inverse_resize_pad(
            pred_padded_raw, (native_h, native_w), scale, pads
        ).numpy()

        # --- DIAGNOSTIC ---
        if np.max(pred_restored) < 1e-6:
            logger.warning(
                f"⚠️ Reconstructor output is EMPTY! Input max: {np.max(input_stack_native):.4f}"
            )
        # ------------------

        return pred_restored

    def autoregressive_restore(
        self,
        volume: np.ndarray,
        start_idx: int,
        end_idx: int,
        direction: str = 'forward',
        mask_volume: np.ndarray = None,
    ) -> np.ndarray:
        """
        Restores slices. If mask_volume is provided, uses 'Teacher Forcing'
        to keep healthy tissue (outside mask) perfect, inpainting only inside mask.
        mask_volume: (Z, H, W) binary mask where 1=Tumor/Void, 0=Healthy.
        """
        axis_size = volume.shape[0]
        recon_vol = np.zeros_like(volume)
        N = self.cfg.data.neighborhood

        if direction == 'forward':
            iter_range = range(start_idx, end_idx)
            # Context: [t-N ... t-1]
            # Transpose to (H, W, N) for InputProcessor consistency
            initial_context = [
                volume[i] for i in range(start_idx - N, start_idx)
            ]
            recon_vol[:start_idx] = volume[:start_idx]
        else:
            iter_range = range(end_idx - 1, start_idx - 1, -1)
            # Context: [t+N ... t+1] (Spatial High to Low)
            # For backward prediction at 't', we need t+1, t+2, t+3.
            # Generator flip logic: t+1 becomes "index 0", t+3 "index 2"?
            # Let's trust the queue order
            initial_context = [volume[i] for i in range(end_idx, end_idx + N)]
            # If we iterate backwards (decreasing index), the "immediate previous" is index+1.
            # Queue append logic handles the shift.
            # Important: Reverse initial context so popleft/append works linearly?
            # Let's keep it simple: List order [Closest ... Farthest] or [Farthest ... Closest]?
            # Standard: [t-3, t-2, t-1]. Append t. New: [t-2, t-1, t].
            # Backward: [t+3, t+2, t+1]. Append t. New: [t+2, t+1, t].
            # So we load [t+3, t+2, t+1] naturally
            initial_context.reverse()  # [t+1, t+2, t+3] -> [t+3, t+2, t+1] order in RAM?
            # Actually, standard range gives [100, 101, 102].
            # We want buffer to end with 100 (t+1).
            # So input [102, 101, 100]
            recon_vol[end_idx:] = volume[end_idx:]

        buffer = deque(initial_context, maxlen=N)

        # Safety: Ensure iter_range is valid for the buffer size N
        valid_iter = []
        for i in iter_range:
            # For forward: need [i-N...i-1]. So i-N must be >= 0
            # For backward: need [i+1...i+N]. So i+N must be < axis_size
            if direction == 'forward' and (i - N) < 0:
                continue
            if direction == 'backward' and (i + N) >= axis_size:
                continue
            valid_iter.append(i)

        for i in tqdm(
            valid_iter,
            desc=f"{direction.title()} Pass",
            leave=False,
            disable=self.cfg.batch_mode,
        ):  # use valid_iter instead of iter_range
            # Stack from buffer -> (N, H, W)
            stack_n_h_w = np.stack(list(buffer), axis=0)

            # ----- Zero out the tumor in the context stack! -----
            if mask_volume is not None:
                ctx_indices = (
                    [i - N + k for k in range(N)]
                    if direction == 'forward'
                    else [i + N - k for k in range(N)]
                )
                for buf_idx, z_idx in enumerate(ctx_indices):
                    if 0 <= z_idx < axis_size:
                        m = (mask_volume[z_idx] > 0).astype(np.float32)
                        stack_n_h_w[buf_idx] *= 1.0 - m
            # -----------------------------------------------------

            # InputProcessor expects (H, W, N), we transpose
            stack_h_w_n = np.transpose(stack_n_h_w, (1, 2, 0))

            # Target Slice (Ground Truth Geometry)
            target_slice = volume[i]

            # Calculate true start index for positional encoding
            # Forward: Context is [i-N, ..., i-1]. Start is i-N.
            # Backward: Context is [i+N, ..., i+1]. Start is i+N.
            true_start_idx = (i - N) if direction == 'forward' else (i + N)

            # Predict returns (H, W, M)
            preds_all = self._predict_slice(
                stack_h_w_n,
                target_slice,
                true_start_idx,
                i,
                axis_size,
                direction,
                mask_volume,
            )

            # FAST GREEDY SELECTION: Just take Hypothesis 0
            # (In MHP, the heads specialize, so sticking to Head 0 ensures consistent style)
            pred_slice = (
                preds_all[..., 0]
                if self.cfg.model.num_hypotheses > 1
                else preds_all[..., 0]
            )

            # --- TEACHER FORCING LOGIC ---
            if mask_volume is not None:
                # Get tumor mask for current slice 'i'
                # Ensure mask is binary (0/1)
                m = (mask_volume[i] > 0).astype(np.float32)

                # Blend: Keep GT where mask is 0, Use Pred where mask is 1
                # Inpaint ONLY inside the tumor
                final_slice = (pred_slice * m) + (target_slice * (1.0 - m))
            else:
                # Pure Autoregression
                final_slice = pred_slice

            recon_vol[i] = final_slice
            buffer.append(final_slice)

        return recon_vol

    def beam_search_restore(
        self,
        volume: np.ndarray,
        start_idx: int,
        end_idx: int,
        direction: str = 'forward',
        mask_volume: np.ndarray = None,
        beam_width: int = 3,
    ) -> np.ndarray:
        axis_size = volume.shape[0]
        N = self.cfg.data.neighborhood
        M = self.cfg.model.num_hypotheses

        # Fallback to greedy if M=1
        if M == 1:
            return self.autoregressive_restore(
                volume, start_idx, end_idx, direction, mask_volume
            )

        if direction == 'forward':
            iter_range = range(start_idx, end_idx)
            initial_context = [
                volume[i] for i in range(start_idx - N, start_idx)
            ]
        else:
            iter_range = range(end_idx - 1, start_idx - 1, -1)
            initial_context = [volume[i] for i in range(end_idx, end_idx + N)]
            initial_context.reverse()

        # A beam is: (cumulative_cost, buffer_list, generated_slices_list)
        beams = [(0.0, initial_context, [])]

        for i in tqdm(
            iter_range,
            desc=f"{direction.title()} Beam Search",
            leave=False,
            disable=self.cfg.batch_mode,
        ):
            new_beams = []
            target_slice = volume[i]
            true_start_idx = (i - N) if direction == 'forward' else (i + N)

            # Next index to look ahead (for cost calculation)
            next_i = (i + 1) if direction == 'forward' else (i - 1)
            can_lookahead = (direction == 'forward' and next_i < end_idx) or (
                direction == 'backward' and next_i >= start_idx
            )

            # 1. Expand each active beam
            for cost, buffer, gen_slices in beams:
                stack_h_w_n = np.transpose(np.stack(buffer, axis=0), (1, 2, 0))

                # Get M hypotheses for current step
                preds_all = self._predict_slice(
                    stack_h_w_n,
                    target_slice,
                    true_start_idx,
                    i,
                    axis_size,
                    direction,
                    mask_volume,
                )

                # 2. Score each hypothesis
                for m in range(M):
                    hyp_slice = preds_all[..., m]

                    # Apply Teacher Forcing if mask exists
                    if mask_volume is not None:
                        mask_m = (mask_volume[i] > 0).astype(np.float32)
                        final_slice = (hyp_slice * mask_m) + (
                            target_slice * (1.0 - mask_m)
                        )
                    else:
                        final_slice = hyp_slice

                    step_cost = 0.0

                    # --- THE RL LOOKAHEAD (Cost Calculation) ---
                    if can_lookahead:
                        temp_buffer = buffer[1:] + [final_slice]
                        temp_stack = np.transpose(
                            np.stack(temp_buffer, axis=0), (1, 2, 0)
                        )
                        next_target = volume[next_i]
                        next_start_idx = (
                            (next_i - N)
                            if direction == 'forward'
                            else (next_i + N)
                        )

                        # Peek at t+1
                        future_preds = self._predict_slice(
                            temp_stack,
                            next_target,
                            next_start_idx,
                            next_i,
                            axis_size,
                            direction,
                            mask_volume,
                        )
                        # Variance across the M heads for t+1
                        # High variance = the model is confused by our `final_slice` choice!
                        future_variance = np.var(future_preds, axis=-1)
                        step_cost = float(np.mean(future_variance))

                    new_cost = cost + step_cost
                    new_beams.append(
                        (
                            new_cost,
                            buffer[1:] + [final_slice],
                            gen_slices + [final_slice],
                        )
                    )

            # 3. Prune the beams
            new_beams.sort(key=lambda x: x[0])
            beams = new_beams[:beam_width]

        # Reconstruction is complete. The best trajectory is the one with the lowest cost.
        best_trajectory = beams[0][2]

        recon_vol = volume.copy()
        if direction == 'forward':
            recon_vol[start_idx:end_idx] = best_trajectory
        else:
            # For backward, the generated slices were appended in reverse order
            recon_vol[start_idx:end_idx] = list(reversed(best_trajectory))

        return recon_vol

    def bidirectional_restore(
        self,
        volume: np.ndarray,
        mask: np.ndarray = None,
        masked_inference: bool = False,
        use_beam_search: bool = True,
    ) -> np.ndarray:
        """
        masked_inference: If True, uses the mask to enforce ground truth outside the tumor.
        """
        N = self.cfg.data.neighborhood

        if mask is not None:
            has_tumor = np.any(np.isin(mask, [1, 2, 4]), axis=(1, 2))
            indices = np.where(has_tumor)[0]
            if len(indices) == 0:
                return volume

            # CRITICAL FIX: Clamp to ensure N context slices always exist
            start = max(N, indices[0] - 2)
            end = min(volume.shape[0] - N, indices[-1] + 3)

            # Safety check: if tumor is so huge/close to edges that start >= end
            if start >= end:
                print(
                    "Warning: Tumor extends too far to edges for context window. Skipping."
                )
                return volume
        else:
            start = N
            end = volume.shape[0] - N

        # Pass 'mask' to autoregressive_restore only if masked_inference is True
        inference_mask = mask if masked_inference else None

        # Choose the restoration engine
        restore_fn = (
            self.beam_search_restore
            if use_beam_search
            else self.autoregressive_restore
        )

        recon_fwd = restore_fn(
            volume, start, end, 'forward', mask_volume=inference_mask
        )
        recon_bwd = restore_fn(
            volume, start, end, 'backward', mask_volume=inference_mask
        )

        recon_merged = volume.copy()
        recon_merged[start:end] = (
            recon_fwd[start:end] + recon_bwd[start:end]
        ) / 2.0

        return recon_merged
