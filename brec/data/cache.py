import os
import threading
import random
import time

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from typing import Optional

import nibabel as nib
import numpy as np


# --- 0. Statistics Container ---
class DataStatistics:
    """
    Tracks data access patterns to verify sampling uniformity.
    """

    def __init__(self):
        # Counter Key: (dataset_name, volume_id, axis, slice_idx)
        # We track direction separately or fold it in
        self.hits = Counter()
        self.volume_map = {}  # id -> path (for sorting)
        self.lock = threading.Lock()

    def record(self, dataset, vol_id, axis, slice_idx, direction):
        with self.lock:
            self.hits[(dataset, vol_id, axis, slice_idx, direction)] += 1

    def register_volume(self, vol_id, path):
        self.volume_map[vol_id] = path

    def reset(self):
        self.stats.clear()


# --- 1. Unified Container ---
class MedicalVolume:
    """Unified container for IXI and BraTS data."""

    def __init__(
        self, t1_data: np.ndarray, seg_data: np.ndarray = None, path: str = ""
    ):
        self.t1 = t1_data
        self.seg = seg_data
        self.path = path
        self.shape = t1_data.shape

        self.indices = {
            'any': {0: [], 1: [], 2: []},
            'clean': {0: [], 1: [], 2: []},
            'tumor': {0: [], 1: [], 2: []},
        }
        self._analyze()

    def _analyze(self):
        for axis in [0, 1, 2]:
            reduce_axes = tuple(i for i in range(3) if i != axis)
            tissue_map = np.sum(self.t1, axis=reduce_axes) > (
                self.shape[reduce_axes[0]] * self.shape[reduce_axes[1]] * 0.01
            )
            self.indices['any'][axis] = np.where(tissue_map)[0]

            if self.seg is not None:
                tumor_map = np.max(self.seg, axis=reduce_axes) > 0
                self.indices['tumor'][axis] = np.where(tumor_map)[0]
                self.indices['clean'][axis] = np.where(
                    tissue_map & (~tumor_map)
                )[0]
            else:
                self.indices['clean'][axis] = self.indices['any'][axis]


# --- 2. The Universal Loader ---
class VolumeLoader:
    """Single source of truth for loading and preprocessing."""

    @staticmethod
    def load(t1_path: str, seg_path: str = None) -> Optional[MedicalVolume]:
        try:
            # 1. Load & Enforce Canonical
            t1_img = nib.load(t1_path)
            t1_img = nib.as_closest_canonical(t1_img)
            t1_data = t1_img.get_fdata(dtype=np.float32)

            # 2. Transpose (X,Y,Z) -> (Z,Y,X)
            t1_data = np.transpose(t1_data, (2, 1, 0))

            # 3. Load Seg
            seg_data = None
            if seg_path:
                seg_img = nib.load(seg_path)
                seg_img = nib.as_closest_canonical(seg_img)
                seg_data = seg_img.get_fdata(dtype=np.float32)
                seg_data = np.transpose(seg_data, (2, 1, 0))

            # 4. Crop to Content
            mask = t1_data > np.mean(t1_data) * 0.1
            if np.any(mask):
                coords = np.argwhere(mask)
                z_min, y_min, x_min = coords.min(axis=0)
                z_max, y_max, x_max = coords.max(axis=0) + 1

                t1_data = t1_data[z_min:z_max, y_min:y_max, x_min:x_max]
                if seg_data is not None:
                    seg_data = seg_data[z_min:z_max, y_min:y_max, x_min:x_max]

            # 5. Normalize
            p99 = np.percentile(t1_data, 99)
            if p99 > 0:
                t1_data = np.clip(t1_data, 0, p99) / p99

            return MedicalVolume(t1_data, seg_data, path=t1_path)

        except Exception as e:
            # logger.warning(f"Load Failed {t1_path}: {e}") # logger needs import or pass
            return None


# --- 3. Manager Base ---
class DataLoaderBase(ABC):
    def __init__(self, config):
        self.cfg = config
        self.hallucination_buffer = OrderedDict()
        self.lock = threading.Lock()
        self.stats = DataStatistics()
        self.pool = {}  # initialized here to guarantee existence for subclasses

    @abstractmethod
    def get_volume(self, collection: str) -> Optional[MedicalVolume]:
        pass

    def update_hallucination(self, key, data):
        with self.lock:
            self.hallucination_buffer[key] = data
            if (
                len(self.hallucination_buffer)
                > self.cfg.data.hallucination_buffer_size
            ):
                self.hallucination_buffer.popitem(last=False)

    def get_hallucination(self, key):
        with self.lock:
            return self.hallucination_buffer.get(key)

    def reset_buffer(self):
        """Clears the hallucination buffer to prevent state leakage between HPO trials."""
        with self.lock:
            self.hallucination_buffer.clear()

    def get_matched_glioma_stack(
        self,
        axis: int,
        target_norm_pos: float,
        neighborhood: int,
        tolerance: float = 0.05,
    ) -> Optional[np.ndarray]:
        """
        Searches the RAM pool for a BraTS volume containing a glioma at the specified
        normalized anatomical position (+/- tolerance) along the given projection axis.
        Returns the (N+1, H, W) segmentation stack, or None if no match is found.
        """
        with self.lock:
            # Safely grab the current BraTS pool regardless of ActiveLoader (dict) or StaticLoader (list)
            brats_pool = self.pool.get('brats', {})
            if isinstance(brats_pool, dict):
                brats_vols = list(brats_pool.values())
            else:
                brats_vols = list(brats_pool)

        if not brats_vols:
            return None

        # Shuffle volumes to ensure we don't stamp the exact same tumor every time
        random.shuffle(brats_vols)

        for vol in brats_vols:
            if vol.seg is None:
                continue

            # Look up which indices actually contain a tumor for this projection axis
            tumor_indices = vol.indices['tumor'][axis]
            if len(tumor_indices) == 0:
                continue

            axis_size = vol.shape[axis]
            candidate_starts = []

            # Find matching slices within tolerance
            for t_idx in tumor_indices:
                norm_pos = t_idx / max(1.0, float(axis_size - 1))

                if abs(norm_pos - target_norm_pos) <= tolerance:
                    start_idx = t_idx - neighborhood

                    # Verify sequence boundaries: we need [t_idx - neighborhood, t_idx]
                    if (
                        start_idx >= 0
                        and (start_idx + neighborhood) < axis_size
                    ):
                        candidate_starts.append(start_idx)

            if candidate_starts:
                # Pick a random valid start position from this specific volume
                start_idx = random.choice(candidate_starts)

                # Orient the view according to the projection axis
                if axis == 0:
                    view = vol.seg
                elif axis == 1:
                    view = np.moveaxis(vol.seg, 1, 0)
                else:
                    view = np.moveaxis(vol.seg, 2, 0)

                try:
                    # Extract precisely N + 1 slices
                    seg_stack = view[start_idx : start_idx + neighborhood + 1]
                    if seg_stack.shape[0] == neighborhood + 1:
                        return seg_stack
                except Exception as e:
                    # Catch boundary slicing errors silently and try the next volume
                    continue

        # No volume in the current RAM pool had a tumor at this specific relative depth
        return None


# --- 4. Static Loader ---
class StaticLoader(DataLoaderBase):
    def __init__(self, config, ixi_files, brats_list):
        super().__init__(config)
        self.pool = {'ixi': [], 'brats': []}

        # Load sequentially
        for f in ixi_files:
            vol = VolumeLoader.load(f)
            if vol:
                self.pool['ixi'].append(vol)

        for b in brats_list:
            vol = VolumeLoader.load(b['t1'], b['seg'])
            if vol:
                self.pool['brats'].append(vol)

    def get_volume(self, collection: str):
        if not self.pool[collection]:
            return None
        return random.choice(self.pool[collection])


# --- 5. Active Loader ---
class ActiveLoader(DataLoaderBase):
    def __init__(self, config, ixi_files, brats_list):
        super().__init__(config)
        self.files = {}
        if ixi_files:
            self.files['ixi'] = ixi_files
        if brats_list:
            self.files['brats'] = brats_list

        self.pool = {k: {} for k in self.files.keys()}
        self.keys = {k: [] for k in self.files.keys()}
        self.active = False
        self.thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        if self.active:
            return
        self.active = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        # Warmup
        while len(self.pool['ixi']) < min(
            5, self.cfg.data.ixi_cache_size // 10
        ):
            time.sleep(1)

    def stop(self):
        self.active = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_volume(self, collection):
        with self.lock:
            if not self.keys[collection]:
                return None
            key = random.choice(self.keys[collection])
            return self.pool[collection][key]

    def _worker(self):
        # 0. Helper to extract ID based on dataset type
        def get_id(item):
            return (
                item['id'] if isinstance(item, dict) else os.path.basename(item)
            )

        # 1. Prepare index counters only for active datasets
        indices = {k: 0 for k in self.files.keys()}
        for k in self.files:
            random.shuffle(self.files[k])

        while self.active:
            # 2. Load from each available dataset to keep the cache full.
            # We don't need modulo math here;
            # the Dataset pipeline handles the probability weights
            for col in self.files.keys():
                self._load_next(col, indices, key_fn=get_id)

            # 3. Simple throttler
            if (
                sum(len(p) for p in self.pool.values())
                >= self.cfg.data.ixi_cache_size + self.cfg.data.brats_cache_size
            ):
                time.sleep(0.1)
            else:
                time.sleep(0.01)

    def _load_next(self, col, indices, key_fn):
        idx = indices[col]
        item = self.files[col][idx]
        indices[col] = (idx + 1) % len(self.files[col])

        if col == 'ixi':
            args = (item, None)
        else:
            args = (item['t1'], item['seg'])

        vol = VolumeLoader.load(*args)
        if vol:
            key = key_fn(item)
            with self.lock:
                limit = (
                    self.cfg.data.ixi_cache_size
                    if col == 'ixi'
                    else self.cfg.data.brats_cache_size
                )
                if len(self.pool[col]) >= limit:
                    rem = self.keys[col].pop(0)
                    del self.pool[col][rem]
                self.pool[col][key] = vol
                self.keys[col].append(key)
