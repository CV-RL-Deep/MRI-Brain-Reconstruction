import os
import glob
import re

from typing import List, Dict, Tuple

from tqdm import tqdm

import nibabel as nib

from src.core.utils import logger


# Constants for file filtering (from original script)
EXCLUDE_FILES = set([
    'wmIXI338-HH-1971-MADisoTFE1_-s3T188_-0301-00003-000001-01.nii',
    'wmIXI357-HH-2076-MADisoTFE1_-s3T199_-0301-00003-000001-01.nii',
    'wmIXI402-Guys-0961-MPRAGESEN_-s413_-0301-00003-000001-01.nii',
    'wmIXI423-IOP-0974-SAGFSPGR_-sIXI00_-0003-00001-000001-01.nii',
])

def get_subject_id_from_path(path: str) -> str:
    """Extracts subject ID (e.g., IXI306) from filename."""
    filename = os.path.basename(path)
    match = re.search(r'IXI\d{3}', filename)
    if match:
        return match.group(0)
    return filename

def find_t1_files(root: str) -> List[str]:
    """Recursive search for wm*.nii files (IXI/OASIS)."""
    # Pattern matches the Kaggle 'RawT1' pattern from your original script
    pattern = os.path.join(root, '**', 'wm*.nii')
    matches = glob.glob(pattern, recursive=True)

    valid_matches = [
        p for p in matches
        if os.path.getsize(p) > 1024
        and os.path.basename(p) not in EXCLUDE_FILES
    ]
    logger.info(f"Found {len(valid_matches)} valid T1 volumes in {root}")
    return sorted(valid_matches)

def get_brats_subjects(root: str) -> List[Dict]:
    """Scans BraTS directory for T1/Seg pairs."""
    subjects = []
    # Search for T1 files
    t1_files = glob.glob(os.path.join(root, '**', '*_t1.nii'), recursive=True)

    for t1_path in sorted(t1_files):
        if os.path.getsize(t1_path) <= 1024: continue

        folder = os.path.dirname(t1_path)
        subject_id = os.path.basename(t1_path).replace('_t1.nii', '')

        # Standard BraTS naming: ID_seg.nii
        seg_path = os.path.join(folder, f"{subject_id}_seg.nii")

        if not os.path.exists(seg_path):
            # Fallback search
            alts = glob.glob(os.path.join(folder, '*seg*.nii'))
            seg_path = alts[0] if alts else None

        subjects.append({
            'id': subject_id,
            't1': t1_path,
            'seg': seg_path
        })

    logger.info(f"Found {len(subjects)} BraTS subjects in {root}")
    return subjects

def make_slice_index_list(vol_paths: List[str], neighborhood: int = 3,
                          batch_mode=False) -> List[Tuple[str, int, str]]:
    """
    Scans headers to build a list of valid (Volume, SliceIndex, Direction) samples.
    """
    idxs = []
    logger.info(f"Indexing slices for {len(vol_paths)} volumes...")

    for p in tqdm(vol_paths, desc="Indexing", disable=batch_mode):
        try:
            # Fast header check for Z-dim
            # Note: We rely on the DataManager's logic that Z is usually the smallest dim < 128
            # But header read is strictly (X,Y,Z).
            # If our heuristic in DataManager transposes, we need to know WHICH dim becomes Z
            img = nib.load(p)
            shape = img.shape

            # Replicate DataManager heuristic to find Z dimension size
            z_dim = shape[2] # Default NIfTI
            if len(shape) == 3 and shape[2] < shape[0] and shape[2] < 128:
                z_dim = shape[2] # Z is last (standard)
            elif shape[0] < 128:
                z_dim = shape[0] # Z is first
            else:
                z_dim = shape[2] # Fallback

        except Exception as e:
            logger.warning(f"Skipping {os.path.basename(p)}: {e}")
            continue

        # Forward samples: [i, i+1, i+2] -> predict i+3
        # Valid if i+3 < z_dim
        # So max i = z_dim - 1 - neighborhood
        limit_fwd = z_dim - neighborhood - 1
        for i in range(limit_fwd + 1):
            idxs.append((p, i, 'forward'))

        # Backward samples: [i, i-1, i-2] -> predict i-3
        # Valid if i-3 >= 0
        # So min i = neighborhood
        for i in range(neighborhood, z_dim):
            idxs.append((p, i, 'backward'))

    return idxs
