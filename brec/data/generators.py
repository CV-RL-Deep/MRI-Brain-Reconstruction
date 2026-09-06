import os
import random
import time

import numpy as np
import tensorflow as tf

from .augmentations import AugmentationLogic
from .cache import DataLoaderBase
from ..configs.config import Config
from ..core.utils import logger
from ..core.geometry import GeometryOps, InputProcessor


class GeneratorBase(tf.keras.utils.Sequence):
    def __init__(self, manager: DataLoaderBase, **kwargs):
        super().__init__(
            **kwargs
        )  # make Keras 3 happy (it is not happy anyway)
        self.manager = manager
        self.cfg = manager.cfg

    def __len__(self):
        return 1000

    @staticmethod
    def _get_stack(vol_data, axis, start_idx, neighborhood):
        """Helper to extract a 2.5D stack along any axis."""
        # Handle None data (e.g. missing seg) by returning None
        if vol_data is None:
            return None

        if axis == 0:
            view = vol_data
        elif axis == 1:
            view = np.moveaxis(vol_data, 1, 0)
        else:
            view = np.moveaxis(vol_data, 2, 0)

        # Slicing
        try:
            return view[start_idx : start_idx + neighborhood + 1]
        except Exception:
            return None

    @staticmethod
    # def _pick_start_index(candidates, axis_size, neighborhood):
    def _pick_start_index(vol, axis, neighborhood, mode='clean'):
        """
        Selects a start index that includes boundary conditions (empty-to-brain).
        Expands the candidate range by 'neighborhood' size.
        """
        axis_size = vol.shape[axis]

        # 1. Define what is ALLOWED and what is FORBIDDEN based on the mode
        if mode == 'clean':
            invalid_set = set(
                vol.indices['tumor'][axis]
            )  # MUST NOT touch these
            required_set = set(
                vol.indices['any'][axis]
            )  # MUST touch brain tissue
        elif mode == 'tumor':
            invalid_set = set()  # Nothing is forbidden
            required_set = set(vol.indices['tumor'][axis])  # MUST touch tumor
        else:
            return None

        valid_starts = []

        # 2. Slide a window across the entire axis
        for s in range(axis_size - neighborhood):
            seq = set(range(s, s + neighborhood + 1))

            # 3. STRICT CHECK: Does this specific sequence violate our rules?
            if not seq.intersection(invalid_set) and seq.intersection(
                required_set
            ):
                valid_starts.append(s)

        if not valid_starts:
            return None

        return random.choice(valid_starts)


class IXIActiveGenerator(GeneratorBase):
    def __call__(self):
        while True:
            # A. Get Volume
            vol = self.manager.get_volume('ixi')
            if vol is None:
                time.sleep(0.05)
                continue

            # B. Pick Projection
            axis = random.choice(self.cfg.data.projections)
            candidates = vol.indices['clean'][axis]
            N = self.cfg.data.neighborhood

            if len(candidates) < N + 2:
                continue

            # Pick Slice
            # start_idx = self._pick_start_index(candidates, vol.shape[axis], N)
            start_idx = self._pick_start_index(vol, axis, N, mode='clean')
            if start_idx is None:
                continue

            # C. Extract Stack
            stack = self._get_stack(vol.t1, axis, start_idx, N)
            if stack is None:
                continue

            # Detach from Cache
            stack = stack.copy()

            # Randomly Flip Temporal Order (Bidirectional Training)
            is_backward = False
            if random.random() < 0.5:
                stack = np.flip(stack, axis=0)  # flip along slice dimension
                is_backward = True

            # Record spatial index of target
            direction_label = 'backward' if is_backward else 'forward'
            real_target_idx = start_idx if is_backward else (start_idx + N)

            self.manager.stats.record(
                'ixi',
                os.path.basename(vol.path),
                axis,
                real_target_idx,
                direction_label,
            )

            input_stack = stack[:-1]
            target_slice = stack[-1]

            info = {
                'glioma_applied': False,
                'blur_applied': False,
                'noise_applied': False,
            }

            # Initialize empty masks
            target_tumor_mask = np.zeros_like(target_slice)
            detailed_mask = np.zeros_like(target_slice)

            # --- 1. Tumor Voiding (Anatomically matched) ---
            if random.random() < self.cfg.aug.prob_glioma:
                # Calculate absolute normalized depth of the target slice
                target_norm_pos = real_target_idx / max(
                    1.0, float(vol.shape[axis] - 1.0)
                )

                # Query manager for an exact anatomical match
                seg_stack = self.manager.get_matched_glioma_stack(
                    axis, target_norm_pos, N
                )

                if seg_stack is not None:
                    # Ensure we match the flip state so the temporal sequence of the void aligns
                    if is_backward:
                        seg_stack = np.flip(seg_stack, axis=0)

                    input_stack, target_tumor_mask, applied = (
                        AugmentationLogic.apply_tumor_void_direct(
                            input_stack, seg_stack, self.cfg.aug
                        )
                    )
                    if applied:
                        info['glioma_applied'] = True

            # Pre-calculate brain mask so it can be geometrically augmented
            brain_mask = (target_slice > 0.01).astype(np.float32)

            # --- 2. Geometric Augmentation (FIXED MASK ALIGNMENT) ---
            # Pack all spatial masks into the array so they rotate with the image
            masks_to_rotate = [target_tumor_mask, detailed_mask, brain_mask]

            input_stack, target_slice, rotated_masks = (
                AugmentationLogic.apply_geometric(
                    input_stack, target_slice, masks_to_rotate, self.cfg.aug
                )
            )

            # Unpack the synchronized masks
            target_tumor_mask, detailed_mask, brain_mask = rotated_masks

            # --- 3. Autoregressive Noise (Photometric) ---
            input_stack, flags = (
                AugmentationLogic.apply_autoregressive_corruption(
                    input_stack, self.cfg.aug
                )
            )
            info.update(flags)

            # --- 4. Hallucination (FIXED AXIS AND INDEX) ---
            if random.random() < self.cfg.aug.prob_hallucination_replay:
                # Get the spatial index of the most recent slice in the history (input_stack[-1])
                h_idx = (start_idx + 1) if is_backward else (start_idx + N - 1)

                # Use strict Axis-aware dictionary key
                h_key = (vol.path, axis, h_idx)

                hal = self.manager.get_hallucination(h_key)
                if hal is not None and hal.shape == input_stack[-1].shape:
                    input_stack[-1] = hal
                hallucination_proc = True
            else:
                hallucination_proc = False

            # --- EDA TRACKING INFO ---
            # Map temporal sequence back to absolute spatial brain indices
            if is_backward:
                # Sequence was flipped. Original physical was [start ... start+N]
                # Target is start. Inputs are start+N, start+N-1, ..., start+1
                spatial_indices = [start_idx + N - k for k in range(N)] + [
                    start_idx
                ]
            else:
                # Target is start+N. Inputs are start, start+1, ..., start+N-1
                spatial_indices = [start_idx + k for k in range(N)] + [
                    start_idx + N
                ]

            info.update(
                {
                    'volume_path': vol.path,
                    'axis_size': vol.shape[axis],
                    'axis': axis,
                    'direction': direction_label,
                    'spatial_indices': spatial_indices,
                    'target_tumor_mask': target_tumor_mask,  # to draw contours
                    'hallucination_proc': hallucination_proc,
                    'mode': getattr(
                        self, 'mode', 'clean'
                    ),  # identify IXI/BraTS mode
                }
            )
            # -------------------------

            # --- D. Formatting ---
            true_start_idx = (start_idx + N) if is_backward else start_idx

            model_inputs, _, _ = InputProcessor.prepare_model_input(
                input_stack_native=input_stack,
                target_slice_native=target_slice,
                start_idx=true_start_idx,
                target_idx=real_target_idx,
                axis_size=vol.shape[axis],
                direction=direction_label,
                void_mask_native=target_tumor_mask
                if np.any(target_tumor_mask)
                else None,
                config=self.cfg,
            )

            y = np.stack(
                [target_slice, target_tumor_mask, detailed_mask, brain_mask],
                axis=-1,
            )

            yield model_inputs, y, info


class BraTSActiveGenerator(GeneratorBase):
    def __init__(self, manager, mode='clean', **kwargs):
        super().__init__(manager, **kwargs)
        self.mode = mode

    def __call__(self):
        while True:
            vol = self.manager.get_volume('brats')
            if vol is None:
                time.sleep(0.05)
                continue

            axis = random.choice(self.cfg.data.projections)

            if self.mode == 'tumor':
                if vol.seg is None:
                    continue
                candidates = vol.indices['tumor'][axis]
            else:
                candidates = vol.indices['clean'][axis]

            N = self.cfg.data.neighborhood
            if len(candidates) == 0:
                continue

            if self.mode == 'tumor':
                target_idx = random.choice(candidates)
                start_idx = target_idx - N
            else:
                # start_idx = self._pick_start_index(candidates, vol.shape[axis], N)
                start_idx = self._pick_start_index(vol, axis, N, mode=self.mode)

            if (
                start_idx is None
                or start_idx < 0
                or start_idx + N >= vol.shape[axis]
            ):
                continue

            # B. Extract Stacks
            t1_stack = self._get_stack(vol.t1, axis, start_idx, N)
            seg_stack_native = (
                self._get_stack(vol.seg, axis, start_idx, N)
                if vol.seg is not None
                else None
            )

            if t1_stack is None:
                continue

            is_backward = False
            if random.random() < 0.5:
                t1_stack = np.flip(t1_stack, axis=0)
                if seg_stack_native is not None:
                    seg_stack_native = np.flip(seg_stack_native, axis=0)
                is_backward = True

            direction_label = 'backward' if is_backward else 'forward'
            real_target_idx = start_idx if is_backward else (start_idx + N)

            self.manager.stats.record(
                'brats',
                os.path.basename(vol.path),
                axis,
                real_target_idx,
                direction_label,
            )

            input_stack = t1_stack[:-1].copy()
            target_slice = t1_stack[-1].copy()

            info = {
                'glioma_applied': False,
                'blur_applied': False,
                'noise_applied': False,
            }

            target_tumor_mask = np.zeros_like(target_slice)
            detailed_mask = np.zeros_like(target_slice)

            # --- 1. Tumor Voiding (Context Aware) ---
            if self.mode == 'tumor' and seg_stack_native is not None:
                # In tumor mode, we use the volume's own actual tumor!
                input_stack, target_tumor_mask, applied = (
                    AugmentationLogic.apply_tumor_void_direct(
                        input_stack, seg_stack_native, self.cfg.aug
                    )
                )
                if applied:
                    info['glioma_applied'] = True
                    detailed_mask = seg_stack_native[-1].astype(np.float32)
            else:
                # In clean mode, we query the manager to inject an artificial void just like IXI
                if random.random() < self.cfg.aug.prob_glioma:
                    target_norm_pos = real_target_idx / max(
                        1.0, float(vol.shape[axis] - 1.0)
                    )
                    seg_stack = self.manager.get_matched_glioma_stack(
                        axis, target_norm_pos, N
                    )

                    if seg_stack is not None:
                        if is_backward:
                            seg_stack = np.flip(seg_stack, axis=0)

                        input_stack, target_tumor_mask, applied = (
                            AugmentationLogic.apply_tumor_void_direct(
                                input_stack, seg_stack, self.cfg.aug
                            )
                        )
                        if applied:
                            info['glioma_applied'] = True

            # Pre-calculate brain mask
            brain_mask = (target_slice > 0.01).astype(np.float32)

            # --- 2. Geometric Augmentation (FIXED MASK ALIGNMENT) ---
            masks_to_rotate = [target_tumor_mask, detailed_mask, brain_mask]

            input_stack, target_slice, rotated_masks = (
                AugmentationLogic.apply_geometric(
                    input_stack, target_slice, masks_to_rotate, self.cfg.aug
                )
            )
            target_tumor_mask, detailed_mask, brain_mask = rotated_masks

            # --- 3. Autoregressive Noise ---
            input_stack, flags = (
                AugmentationLogic.apply_autoregressive_corruption(
                    input_stack, self.cfg.aug
                )
            )
            info.update(flags)

            # --- 4. Hallucination (FIXED AXIS AND INDEX) ---
            if random.random() < self.cfg.aug.prob_hallucination_replay:
                h_idx = (start_idx + 1) if is_backward else (start_idx + N - 1)
                h_key = (vol.path, axis, h_idx)

                hal = self.manager.get_hallucination(h_key)
                if hal is not None and hal.shape == input_stack[-1].shape:
                    input_stack[-1] = hal
                hallucination_proc = True
            else:
                hallucination_proc = False

            # --- EDA TRACKING INFO ---
            # Map temporal sequence back to absolute spatial brain indices
            if is_backward:
                # Sequence was flipped. Original physical was [start ... start+N]
                # Target is start. Inputs are start+N, start+N-1, ..., start+1
                spatial_indices = [start_idx + N - k for k in range(N)] + [
                    start_idx
                ]
            else:
                # Target is start+N. Inputs are start, start+1, ..., start+N-1
                spatial_indices = [start_idx + k for k in range(N)] + [
                    start_idx + N
                ]

            info.update(
                {
                    'volume_path': vol.path,
                    'axis_size': vol.shape[axis],
                    'axis': axis,
                    'direction': direction_label,
                    'spatial_indices': spatial_indices,
                    'target_tumor_mask': target_tumor_mask,  # to draw contours
                    'hallucination_proc': hallucination_proc,
                    'mode': getattr(
                        self, 'mode', 'clean'
                    ),  # identify IXI/BraTS mode
                }
            )
            # -------------------------

            # --- D. Formatting ---
            true_start_idx = (start_idx + N) if is_backward else start_idx

            model_inputs, _, _ = InputProcessor.prepare_model_input(
                input_stack_native=input_stack,
                target_slice_native=target_slice,
                start_idx=true_start_idx,
                target_idx=real_target_idx,
                axis_size=vol.shape[axis],
                direction=direction_label,
                void_mask_native=target_tumor_mask
                if np.any(target_tumor_mask)
                else None,
                config=self.cfg,
            )

            y = np.stack(
                [target_slice, target_tumor_mask, detailed_mask, brain_mask],
                axis=-1,
            )
            yield model_inputs, y, info


class SequentialValidationGenerator(GeneratorBase):
    """
    Finite, deterministic generator for accurate Validation tracking.
    Iterates over both IXI and BraTS RAM pools.
    """

    def __init__(
        self, manager: DataLoaderBase, max_slices_per_vol=10, **kwargs
    ):
        super().__init__(
            manager, **kwargs
        )  # <--- ADD **kwargs (Keras happiness)
        self.max_slices = max_slices_per_vol

    def __call__(self):
        ixi_pool = self.manager.pool.get('ixi', {})
        brats_pool = self.manager.pool.get('brats', {})
        ixi_vols = (
            list(ixi_pool.values())
            if isinstance(ixi_pool, dict)
            else list(ixi_pool)
        )
        brats_vols = (
            list(brats_pool.values())
            if isinstance(brats_pool, dict)
            else list(brats_pool)
        )
        volumes = [('ixi', v) for v in ixi_vols] + [
            ('brats', v) for v in brats_vols
        ]

        for ds_type, vol in volumes:
            axis = 0
            candidates = vol.indices['clean'][axis]
            N = self.cfg.data.neighborhood
            is_backward = False
            if len(candidates) < N + 2:
                continue

            mid = len(candidates) // 2
            half_window = self.max_slices // 2
            start_c = max(0, mid - half_window)
            end_c = min(len(candidates), mid + half_window)

            for target_idx in candidates[start_c:end_c]:
                start_idx = target_idx - N
                if start_idx < 0 or start_idx + N >= vol.shape[axis]:
                    continue

                # NEW: RIGOROUS VALIDATION SAFETY CHECK
                # Ensure the validation sequence is strictly clean
                seq_set = set(range(start_idx, start_idx + N + 1))
                if seq_set.intersection(set(vol.indices['tumor'][axis])):
                    continue  # skip this sequence, it crossed into a tumor!

                stack = self._get_stack(vol.t1, axis, start_idx, N)
                if stack is None:
                    continue

                seg_stack = None
                if ds_type == 'brats' and vol.seg is not None:
                    seg_stack = self._get_stack(vol.seg, axis, start_idx, N)

                input_stack = stack[:-1].copy()
                target_slice = stack[-1].copy()

                if seg_stack is not None:
                    target_tumor_mask = np.isin(
                        seg_stack[-1], self.cfg.aug.brats_labels_for_void
                    ).astype(np.float32)
                    detailed_mask = seg_stack[-1].astype(np.float32)
                else:
                    target_tumor_mask = np.zeros_like(target_slice)
                    detailed_mask = np.zeros_like(target_slice)

                # --- FIX: Convert to Dictionary Format ---
                model_inputs, _, _ = InputProcessor.prepare_model_input(
                    input_stack_native=input_stack,
                    target_slice_native=target_slice,
                    start_idx=start_idx,
                    target_idx=target_idx,
                    axis_size=vol.shape[axis],
                    direction='forward',
                    void_mask_native=target_tumor_mask
                    if np.any(target_tumor_mask)
                    else None,
                    config=self.cfg,
                )

                brain_mask = (target_slice > 0.01).astype(np.float32)
                y = np.stack(
                    [
                        target_slice,
                        target_tumor_mask,
                        detailed_mask,
                        brain_mask,
                    ],
                    axis=-1,
                )
                info = {
                    'glioma_applied': False,
                    'blur_applied': False,
                    'noise_applied': False,
                }

            # --- EDA TRACKING INFO ---
            # Map temporal sequence back to absolute spatial brain indices
            if is_backward:
                # Sequence was flipped. Original physical was [start ... start+N]
                # Target is start. Inputs are start+N, start+N-1, ..., start+1
                spatial_indices = [start_idx + N - k for k in range(N)] + [
                    start_idx
                ]
                direction_label = 'backward'
            else:
                # Target is start+N. Inputs are start, start+1, ..., start+N-1
                spatial_indices = [start_idx + k for k in range(N)] + [
                    start_idx + N
                ]
                direction_label = 'forward'

            info.update(
                {
                    'volume_path': vol.path,
                    'axis_size': vol.shape[axis],
                    'axis': axis,
                    'direction': direction_label,
                    'spatial_indices': spatial_indices,
                    'target_tumor_mask': target_tumor_mask,  # to draw contours
                    'hallucination_proc': False,
                    'mode': getattr(
                        self, 'mode', 'clean'
                    ),  # identify IXI/BraTS mode
                }
            )
            # -------------------------

            yield model_inputs, y, info


class HpoTrainSequence(tf.keras.utils.Sequence):
    """
    Pure Python data pipeline. Bypasses tf.data C++ memory leaks.
    """

    def __init__(
        self,
        ixi_gen_instance,
        brats_gen_instance,
        config,
        steps_per_epoch,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )  # make Keras 3 happy (it is not happy anyway)
        self.ixi_iter = iter(ixi_gen_instance())
        self.brats_iter = iter(brats_gen_instance())
        self.cfg = config
        self.steps = steps_per_epoch
        self.batch_size = config.train.batch_size

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        w_ixi = max(0.1, min(0.9, self.cfg.data.ixi_sampling_weight))

        for _ in range(self.batch_size):
            # 1. Randomly sample
            if random.random() < w_ixi:
                x, y, _ = next(self.ixi_iter)
            else:
                x, y, _ = next(self.brats_iter)

            # 2. Convert to tensors
            x_t = tf.convert_to_tensor(x, dtype=tf.float32)
            y_t = tf.convert_to_tensor(y, dtype=tf.float32)

            # 3. Apply geometry ops dynamically
            xp, _, _ = GeometryOps.resize_and_pad(
                x_t, self.cfg.data.padded_size, 'bicubic'
            )
            yp, _, _ = GeometryOps.resize_and_pad(
                y_t, self.cfg.data.padded_size, 'nearest'
            )

            batch_x.append(xp)
            batch_y.append(yp)

        return tf.stack(batch_x), tf.stack(batch_y)


class HpoValidSequence(tf.keras.utils.Sequence):
    """
    Eagerly materializes the validation set into RAM.
    Zero-overhead, deterministic evaluation.
    """

    def __init__(self, val_gen_instance, config, **kwargs):
        super().__init__(
            **kwargs
        )  # make Keras 3 happy (it is not happy anyway)
        self.cfg = config
        self.batch_size = config.train.batch_size

        # Extract all slices into a list exactly once! (~150 MB footprint)
        self.data = list(val_gen_instance())

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x, batch_y = [], []

        for x, y, _ in batch_data:
            x_t = tf.convert_to_tensor(x, dtype=tf.float32)
            y_t = tf.convert_to_tensor(y, dtype=tf.float32)

            xp, _, _ = GeometryOps.resize_and_pad(
                x_t, self.cfg.data.padded_size, 'bicubic'
            )
            yp, _, _ = GeometryOps.resize_and_pad(
                y_t, self.cfg.data.padded_size, 'nearest'
            )

            batch_x.append(xp)
            batch_y.append(yp)

        return tf.stack(batch_x), tf.stack(batch_y)


def create_tf_dataset(
    generator: GeneratorBase,
    config: Config,
    is_training: bool = True,
    include_info: bool = False,
):
    h, w = config.data.padded_size
    N = config.data.neighborhood

    # 1. Signature for Model Inputs (X)
    x_sig = {
        'history_input': tf.TensorSpec(shape=(None, None, N), dtype=tf.float32),
        'mask_input': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
        'p_history_input': tf.TensorSpec(shape=(N,), dtype=tf.float32),
        'p_abs_input': tf.TensorSpec(shape=(1,), dtype=tf.float32),
    }

    # 2. Signature for Rich EDA Info Dictionary
    # This precisely matches the 11 keys we are yielding for debugging
    info_sig = {
        "glioma_applied": tf.TensorSpec(shape=(), dtype=tf.bool),
        "blur_applied": tf.TensorSpec(shape=(), dtype=tf.bool),
        "noise_applied": tf.TensorSpec(shape=(), dtype=tf.bool),
        "volume_path": tf.TensorSpec(shape=(), dtype=tf.string),
        "axis_size": tf.TensorSpec(shape=(), dtype=tf.int32),
        "axis": tf.TensorSpec(shape=(), dtype=tf.int32),
        "direction": tf.TensorSpec(shape=(), dtype=tf.string),
        "spatial_indices": tf.TensorSpec(
            shape=(None,), dtype=tf.int32
        ),  # 1D variable length list
        "target_tumor_mask": tf.TensorSpec(
            shape=(None, None), dtype=tf.float32
        ),  # 2D native mask
        "hallucination_proc": tf.TensorSpec(shape=(), dtype=tf.bool),
        "mode": tf.TensorSpec(shape=(), dtype=tf.string),
    }

    # 3. Full Yield Signature
    output_sig = (
        x_sig,
        tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),  # targets (Y)
        info_sig,
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_sig)

    def map_fn(x_dict, y, info):
        # 2. PERFORM RESIZING HERE (Multi-threaded C++)
        hist_pad, _, _ = GeometryOps.resize_and_pad(
            x_dict['history_input'], (h, w), method='bicubic'
        )
        mask_pad, _, _ = GeometryOps.resize_and_pad(
            x_dict['mask_input'], (h, w), method='nearest'
        )
        y_padded, _, _ = GeometryOps.resize_and_pad(y, (h, w), method='nearest')

        # Enforce static shapes for the compiler
        hist_pad.set_shape((h, w, N))
        mask_pad.set_shape((h, w, 1))
        y_padded.set_shape((h, w, 4))

        x_dict_out = {
            'history_input': hist_pad,
            'mask_input': mask_pad,
            'p_history_input': x_dict['p_history_input'],
            'p_abs_input': x_dict['p_abs_input'],
        }

        if include_info:
            return x_dict_out, y_padded, info
        else:
            return x_dict_out, y_padded

    # 3. Apply the map function concurrently
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if not is_training:
        batch_size = 1 if include_info else config.train.batch_size
        ds = ds.batch(batch_size)

    return ds


def get_training_dataset(ixi_gen, brats_gen, config):
    w_ixi = config.data.ixi_sampling_weight

    # Check for pure scenarios to avoid TF Graph sampling bugs
    if w_ixi >= 1.0 - 1e-8:
        logger.info("Dataset Mix: 100% IXI Pipeline")
        ds = create_tf_dataset(
            ixi_gen, config, is_training=True, include_info=False
        ).repeat()
    elif w_ixi < 1e-8:
        logger.info("Dataset Mix: 100% BraTS Pipeline")
        ds = create_tf_dataset(
            brats_gen, config, is_training=True, include_info=False
        ).repeat()
    else:
        logger.info(
            f"Dataset Mix: {w_ixi * 100:.1f}% IXI, {(1.0 - w_ixi) * 100:.1f}% BraTS"
        )
        ds_ixi = create_tf_dataset(
            ixi_gen, config, is_training=True, include_info=False
        ).repeat()
        ds_brats = create_tf_dataset(
            brats_gen, config, is_training=True, include_info=False
        ).repeat()
        ds = tf.data.Dataset.sample_from_datasets(
            [ds_ixi, ds_brats], weights=[w_ixi, 1.0 - w_ixi]
        )

    ds = ds.batch(config.train.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
