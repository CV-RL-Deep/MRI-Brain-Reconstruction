import gc
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from ..core.utils import logger
from ..configs.config import Config


class SynthSegEvaluator:
    """
    Orchestrates the downstream anatomical validation using SynthSeg.
    Operates on the saved .npy files from the ablation study.
    """

    def __init__(self, config: Config):
        self.results_dir = config.data.results_dir
        self.nifti_dir = os.path.join(self.results_dir, "nifti")
        self.seg_dir = os.path.join(self.results_dir, "synthseg_masks")

        os.makedirs(self.nifti_dir, exist_ok=True)
        os.makedirs(self.seg_dir, exist_ok=True)

        # FreeSurfer / SynthSeg standard labels
        self.macro_regions = {
            'Ventricles': [4, 43],  # Left/Right Lateral Ventricles
            'Deep_GM': [
                10,
                49,
                11,
                50,
                12,
                51,
                13,
                52,
            ],  # Thalamus, Caudate, Putamen, Pallidum
            'White_Matter': [2, 41],  # Cerebral White Matter
            'Cortex': [3, 42],  # Cerebral Cortex
        }

    def setup(self):
        # """Clones SynthSeg if not present."""
        # if not os.path.exists("SynthSeg"):
        #     logger.info("Cloning SynthSeg repository...")
        #     # subprocess.run(["git", "clone", "https://github.com/BBillot/SynthSeg.git"], check=True)
        #     subprocess.run(["git", "clone", "https://github.com/ValV/SynthSeg.git"], check=True)
        # else:
        #     logger.info("SynthSeg repository found.")
        """Prepares SynthSeg environment and ensures models/data are downloaded."""
        # 1. Ensure the package is installed
        # (Assuming you already ran %pip install git+https://github.com/ValV/SynthSeg.git)

        # 2. Trigger the model download logic provided by the library itself
        from SynthSeg.cli import get_model_dir

        logger.info("Verifying SynthSeg model weights...")
        try:
            get_model_dir()  # This will download to user_cache_dir if missing
        except Exception as e:
            logger.error(f"Failed to download SynthSeg models: {e}")
            raise

        logger.info("SynthSeg environment ready.")

    def convert_to_nifti(self):
        """Converts ablation .npy files to .nii.gz for SynthSeg."""
        # Only convert if NIfTI folder is empty
        nifti_files = glob.glob(os.path.join(self.nifti_dir, "*.nii.gz"))
        if len(nifti_files) > 0:
            logger.info(
                f"NIfTI files already exist in {self.nifti_dir}. Skipping conversion."
            )
            return

        logger.info("Converting .npy volumes to .nii.gz...")
        npy_files = glob.glob(os.path.join(self.results_dir, "*.npy"))

        if not npy_files:
            raise FileNotFoundError(
                f"No .npy files found in {self.results_dir}. Did you run the ablation study?"
            )

        for f in tqdm(npy_files, desc="NIfTI Conversion"):
            vol = np.load(f)
            # SynthSeg is resolution/contrast agnostic. An identity affine is perfectly fine.
            nii = nib.Nifti1Image(vol, np.eye(4))
            out_name = os.path.basename(f).replace('.npy', '.nii.gz')
            nib.save(nii, os.path.join(self.nifti_dir, out_name))

    def run_prediction(self):
        """Executes SynthSeg using the library import with explicit defaults."""
        logger.info("Running SynthSeg prediction (This may take a while)...")

        # A. Clear the main model from GPU to free up VRAM for SynthSeg
        if hasattr(self, 'model'):
            del self.model

        tf.keras.backend.clear_session()
        gc.collect()

        # B. Now run the library call
        from SynthSeg.predict_synthseg import predict
        from SynthSeg.cli import get_model_dir
        import SynthSeg  # import the package to find its location

        # 1. Locate the package directory dynamically
        synthseg_pkg_path = os.path.dirname(os.path.dirname(SynthSeg.__file__))

        # 2. Construct the labels path relative to the package location
        # Based on your find command, the labels are in src/SynthSeg/data/labels_classes_priors/
        labels_path = os.path.join(
            synthseg_pkg_path,
            'SynthSeg',
            "data",
            "labels_classes_priors",
            "synthseg_segmentation_labels_2.0.npy",
        )

        # 3. Locate the model
        model_dir = get_model_dir()
        path_model_seg = os.path.join(model_dir, 'synthseg_2.0.h5')

        # if not os.path.exists(labels_path):
        #     raise FileNotFoundError(f"Labels file not found at {labels_path}. Check SynthSeg's data folder.")
        # 4. Verify
        if not os.path.exists(labels_path):
            logger.warning(
                f"'{labels_path}' NOT FOUND! Falling back to './SynthSeg'..."
            )
            # Fallback: Check if it's in the cloned folder directly
            labels_path = "./SynthSeg/src/SynthSeg/data/labels_classes_priors/synthseg_segmentation_labels_2.0.npy"
            if not os.path.exists(labels_path):
                raise FileNotFoundError(
                    f"Could not locate labels at {labels_path}"
                )
        else:
            logger.info(f"Using '{labels_path}' as labels path.")

        logger.info(f"Running SynthSeg prediction...")

        # These are the values the library expects to see for standard segmentation
        predict(
            path_images=self.nifti_dir,
            path_segmentations=self.seg_dir,
            path_model_segmentation=path_model_seg,
            labels_segmentation=labels_path,
            robust=False,
            fast=True,
            v1=False,
            n_neutral_labels=None,  # 19
            labels_denoiser=os.path.join(
                os.path.dirname(labels_path), 'synthseg_denoiser_labels_2.0.npy'
            ),
            path_posteriors=None,
            path_resampled=None,
            path_volumes=None,
            do_parcellation=False,
            path_model_parcellation=None,
            labels_parcellation=None,
            path_qc_scores=None,
            path_model_qc=None,
            labels_qc=None,
            cropping=None,
        )

        logger.info("SynthSeg prediction complete!")

    def _dice_coef(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.sum(mask1 & mask2)
        vol1 = np.sum(mask1)
        vol2 = np.sum(mask2)
        if vol1 + vol2 == 0:
            return 1.0
        return float(2.0 * intersection / (vol1 + vol2))

    def calculate_dsc(self):
        """Computes DSC between Ground Truth and Ablation predictions."""
        logger.info("Calculating Anatomical DSC metrics...")

        # Find all Ground Truth segmentations
        gt_files = glob.glob(os.path.join(self.seg_dir, "gt_*.nii.gz"))
        ablations = [
            'baseline',
            'spade',
            'buffer',
            'full',
            'ours_full_masked',
            'monai_3d_ldm',
            'monai_2d_ldm',
            'monai_vqgan',
        ]

        results = []

        for gt_path in tqdm(gt_files, desc="Scoring Volumes"):
            vol_id = (
                os.path.basename(gt_path)
                .replace('gt_', '')
                .replace('.nii.gz', '')
            )
            gt_seg = nib.load(gt_path).get_fdata()

            for ab in ablations:
                pred_path = os.path.join(self.seg_dir, f"{ab}_{vol_id}.nii.gz")
                if not os.path.exists(pred_path):
                    continue

                pred_seg = nib.load(pred_path).get_fdata()

                for region_name, labels in self.macro_regions.items():
                    # Create binary masks for the macro region
                    gt_mask = np.isin(gt_seg, labels)
                    pred_mask = np.isin(pred_seg, labels)

                    dsc = self._dice_coef(gt_mask, pred_mask)

                    results.append(
                        {
                            'Volume_ID': vol_id,
                            'Ablation': ab,
                            'Region': region_name.replace('_', ' '),
                            'DSC': dsc,
                        }
                    )

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "anatomical_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Anatomical metrics saved to {csv_path}")
