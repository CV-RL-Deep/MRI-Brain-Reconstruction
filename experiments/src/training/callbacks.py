import os
import random

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from src.core.utils import logger
from src.core.geometry import GeometryOps, InputProcessor
from src.data.generators import GeneratorBase
from src.evaluation.metrics import get_oracle_prediction


class TrainingVisualizer(tf.keras.callbacks.Callback):
    """
    Visualizes reconstruction performance on a fixed validation batch.
    Plots: Input (t-1), Ground Truth (t), Prediction (t), Error Map.
    """
    def __init__(self, val_dataset, config, frequency=5, num_samples=3):
        super().__init__()
        self.cfg = config  # store config
        self.frequency = max(1, frequency)
        self.num_samples = num_samples

        # Take one batch for consistent visualization
        # self.vis_batch = next(iter(val_dataset.take(1)))
        # --- FIX: Support both tf.data.Dataset and Keras Sequence ---
        if hasattr(val_dataset, 'take'):
            # It's a tf.data.Dataset
            self.vis_batch = next(iter(val_dataset.take(1)))
        else:
            # It's a Keras Sequence (like HpoValidSequence)
            # Just grab the 0th batch directly!
            self.vis_batch = val_dataset[0]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch == 0) or ((epoch + 1) % self.frequency == 0):
            # Print the first epoch for comparison
            self._plot_results(epoch + 1)

    def _plot_results(self, epoch):
        # Slice the batch to only what we need to plot
        # This saves VRAM during the predict call
        input_batch, target_batch = self.vis_batch

        # Take only N samples
        n = self.num_samples
        # Slice the batch (TensorFlow handles slicing dictionaries of tensors)
        input_micro_batch = {k: v[:n] for k, v in input_batch.items()}
        target_micro_batch = target_batch[:n]

        # Predict on smaller batch
        # pred_batch = self.model(input_micro_batch, training=False)
        # 1. Raw MHP Prediction
        pred_batch_raw = self.model(input_micro_batch, training=False)

        # 2. Extract Oracle & Uncertainty
        pred_batch_oracle = get_oracle_prediction(target_micro_batch,
                                                  pred_batch_raw)

        if self.cfg.model.num_hypotheses > 1:
            uncertainty_batch = tf.math.reduce_variance(pred_batch_raw, axis=-1,
                                                        keepdims=True)
            # Normalize to 0-1 for nice visualization
            uncertainty_batch = uncertainty_batch / (tf.reduce_max(uncertainty_batch,
                                                                   axis=[1,2,3],
                                                                   keepdims=True) + 1e-8)
        else:
            uncertainty_batch = tf.zeros_like(pred_batch_oracle)

        # Extract the image data from the dictionary for plotting
        # 'history_input' contains the t-N...t-1 slices
        inputs = input_micro_batch['history_input'].numpy()
        targets = target_micro_batch.numpy()

        # Config for plot
        # cols = 4
        cols = 5  # added column for Uncertainty
        rows = self.num_samples
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
        fig.suptitle(f"Epoch {epoch} Visualization", fontsize=16)

        for i in range(self.num_samples):
            # Dynamic Indexing Logic:
            # Input channels are [t-N, ... t-2, t-1].
            # If input_last_target_mask is True, there is an extra channel at the end.
            # However, the image stack always comes first.
            # Therefore, the index of 't-1' is always (neighborhood - 1)
            idx_t_minus_1 = self.cfg.data.neighborhood - 1

            # Handle if channel is mask (binary)
            img_t_minus_1 = inputs[i, :, :, idx_t_minus_1]

            # 2. GT
            img_gt = targets[i, :, :, 0]

            # 3. Pred
            # img_pred = pred_batch[i, :, :, 0]
            # Using Oracle Prediction and Uncertainty map
            img_pred = pred_batch_oracle[i, :, :, 0].numpy()
            img_unc = uncertainty_batch[i, :, :, 0].numpy()

            # 4. Diff
            diff = np.abs(img_gt - img_pred)

            # Plot
            ax = axes[i] if rows > 1 else axes

            ax[0].imshow(img_t_minus_1, cmap='bone')
            ax[0].set_title("Input (t-1)")
            ax[0].axis('off')

            ax[1].imshow(img_gt, cmap='bone')
            ax[1].set_title("Target (t)")
            ax[1].axis('off')

            ax[2].imshow(img_pred, cmap='bone')
            ax[2].set_title(f"Pred (MSE = {np.mean(diff ** 2):.4f})")
            ax[2].axis('off')

            ax[3].imshow(diff, cmap='inferno', vmin=0, vmax=0.3)
            ax[3].set_title("Error |Pred - Target|")
            ax[3].axis('off')

            # New Uncertainty Column
            ax[4].imshow(img_unc, cmap='magma')
            ax[4].set_title("Uncertainty (Variance)")
            ax[4].axis('off')

        plt.tight_layout()
        plt.show()

        plt.close(fig) # try to release RAM


class BufferUpdateCallback(tf.keras.callbacks.Callback):
    """
    Updates the Hallucination Buffer using the Active Data Manager.
    Samples directly from the currently loaded pool in RAM.
    """
    def __init__(self, manager, samples_per_epoch=512, rollout_depth=3):
        super().__init__()
        self.manager = manager
        self.samples = samples_per_epoch
        self.cfg = manager.cfg
        self.rollout_depth = rollout_depth

        self.w_ixi = self.cfg.data.ixi_sampling_weight
        self.w_ixi = max(0.1, min(0.9, self.w_ixi))
        self.w_brats = 1.0 - self.w_ixi

    def on_epoch_end(self, epoch, logs=None):
        warmup = max(1, self.cfg.aug.prob_hallucination_warmup)
        current_prob = min(
            self.cfg.aug.prob_hallucination_max,
            ((epoch + 1) / warmup) * self.cfg.aug.prob_hallucination_max
        )

        self.cfg.aug.prob_hallucination_replay = current_prob

        # Gate Optimization: don't populate buffer if it's not being used
        if current_prob < 1e-4:
            return

        logger.debug(f"Populating hallucination buffer (Next Epoch Prob: {current_prob:.2f})")

        batch_size = self.cfg.train.batch_size // 2
        total_batches = max(1, self.samples // batch_size)

        num_samples_processed = 0

        for _ in range(total_batches):
            contexts =[]

            # A. Fill Initial Contexts
            for _ in range(batch_size):
                if random.random() < self.w_ixi:
                    vol = self.manager.get_volume('ixi')
                else:
                    vol = self.manager.get_volume('brats')

                if vol is None: continue

                axis = random.choice(self.cfg.data.projections)
                candidates = vol.indices['clean'][axis]
                N = self.cfg.data.neighborhood

                if len(candidates) < N + self.rollout_depth + 2: continue

                # Pick start so we have room to roll out
                start_idx = random.choice(candidates[:-self.rollout_depth])
                dim_size = vol.shape[axis]
                if start_idx + N >= dim_size: continue

                full_stack = GeneratorBase._get_stack(vol.t1, axis, start_idx, N)
                if full_stack is None: continue

                stack = full_stack[:-1].copy()
                input_stack = np.transpose(stack, (1, 2, 0))

                direction = 'forward'

                if axis == 0:   view = vol.t1
                elif axis == 1: view = np.moveaxis(vol.t1, 1, 0)
                else:           view = np.moveaxis(vol.t1, 2, 0)

                contexts.append({
                    'stack': input_stack, # (H, W, N)
                    'idx': start_idx + N, # the index we are about to predict
                    'path': vol.path,
                    'axis': axis,         # Storing axis for the key
                    'view': view,
                    'native_h': view.shape[1],
                    'native_w': view.shape[2],
                    'dead': False         # Flag to stop NaN propagation
                })

            if not contexts: continue

            # B. Perform Rollout
            for step in range(self.rollout_depth):
                batch_x = []
                valid_indices =[]

                for i, ctx in enumerate(contexts):
                    # Skip dead contexts (e.g., hit a NaN previously) or out of bounds
                    if ctx['dead'] or ctx['idx'] >= ctx['view'].shape[0]:
                        continue

                    target_slice = ctx['view'][ctx['idx']]

                    true_start_idx = (
                        (ctx['idx'] - N) if direction == 'forward'
                        else min(ctx['idx'] + N, ctx['view'].shape[0] - 1)
                    )

                    model_inputs, scale, pads = InputProcessor.prepare_model_input(
                        input_stack_native=ctx['stack'],
                        target_slice_native=target_slice,
                        start_idx=true_start_idx,
                        target_idx=ctx['idx'],
                        axis_size=ctx['view'].shape[0],
                        direction=direction,
                        config=self.cfg
                    )

                    hist_pad, scale, pads = GeometryOps.resize_and_pad(
                        tf.convert_to_tensor(model_inputs['history_input']),
                        self.cfg.data.padded_size, 'bicubic'
                    )
                    mask_pad, _, _ = GeometryOps.resize_and_pad(
                        tf.convert_to_tensor(model_inputs['mask_input']),
                        self.cfg.data.padded_size, 'nearest'
                    )
                    model_inputs['history_input'] = hist_pad
                    model_inputs['mask_input'] = mask_pad

                    batch_x.append(model_inputs)
                    valid_indices.append(i)

                if not batch_x: break

                try:
                    keys = batch_x[0].keys()
                    inputs = {
                        k: tf.stack(
                            [sample[k] for sample in batch_x]
                        ) for k in keys
                    }
                except Exception as e:
                    logger.error(f"BufferUpdateCallback Batching Failed: {e}")
                    break

                # Predict
                # preds = self.model(inputs, training=False)
                preds_raw = self.model(inputs, training=False)

                # MHP Fix: Collapse hypotheses to a single stable image for the rollout
                if self.cfg.model.num_hypotheses > 1:
                    # preds = tf.reduce_mean(preds_raw, axis=-1, keepdims=True)
                    # Pick a random hypothesis index [0, M-1] (enhanced robustness)
                    idx = tf.random.uniform(
                        shape=[], minval=0, maxval=self.cfg.model.num_hypotheses, dtype=tf.int32
                    )
                    preds = preds_raw[..., idx:idx+1]
                else:
                    preds = preds_raw

                # C. Update Contexts & Store
                for k, ctx_idx in enumerate(valid_indices):
                    ctx = contexts[ctx_idx]
                    pred_padded = preds[k]

                    _, scale, pads = GeometryOps.resize_and_pad(
                        tf.zeros((ctx['native_h'], ctx['native_w'], 1)),
                        self.cfg.data.padded_size
                    )

                    pred_restored = GeometryOps.inverse_resize_pad(
                        pred_padded, (ctx['native_h'], ctx['native_w']),
                        scale, pads
                    ).numpy()[:, :, 0]

                    # NaN Check ("Poison Control")
                    if np.isnan(pred_restored).any() or np.isinf(pred_restored).any():
                        logger.warning(f"⚠️ NaN detected in Hallucination Rollout! Killing sequence for {ctx['path']}")
                        ctx['dead'] = True  # Permanently disable this sequence for the rest of the rollout
                        continue

                    # STORE RESULT USING PROPER AXIS-AWARE KEY
                    self.manager.update_hallucination(
                        (ctx['path'], ctx['axis'], ctx['idx']), pred_restored
                    )
                    num_samples_processed += 1

                    # ROLL THE STACK
                    new_stack = np.concatenate([ctx['stack'][..., 1:],
                                                pred_restored[..., None]],
                                               axis=-1)
                    ctx['stack'] = new_stack
                    ctx['idx'] += 1

                # --- CRITICAL LEAK FIX: Clear Eager Tensors ---
                # Forces TF memory pool to release these before the next step
                if 'inputs' in locals(): del inputs
                if 'preds_raw' in locals(): del preds_raw
                if 'preds' in locals(): del preds
                del batch_x, valid_indices
                # ----------------------------------------------

        logger.debug(f"Hallucination buffer updated: {num_samples_processed} samples")


class GeneratorCheckpoint(tf.keras.callbacks.Callback):
    """
    Saves ONLY the Generator sub-model from the GAN Trainer.
    """
    def __init__(self, filepath, monitor='val_l1_loss', save_best_only=True, mode='min'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = np.inf if mode == 'min' else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None: return

        if self.save_best_only:
            if self.mode == 'min':
                improved = current < self.best
            else:
                improved = current > self.best

            if improved:
                self.best = current
                # Save the GENERATOR, not self.model (which is the Trainer)
                self.model.generator.save(self.filepath) # .keras format
                logger.info(f"\nEpoch {epoch + 1}: Generator saved to {self.filepath}")
        else:
            self.model.generator.save(self.filepath)
