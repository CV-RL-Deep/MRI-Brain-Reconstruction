import os
import glob
import subprocess

import numpy as np

import nibabel as nib
import pandas as pd

from tqdm import tqdm

from brec.core.utils import logger
from configs.config import Config


class FastSurferEvaluator:
    """
    Orchestrates downstream anatomical validation using FastSurfer.
    FastSurfer is a pure PyTorch pipeline that segments T1 MRIs into 95 classes.
    """
    def __init__(self, config: Config):
        self.results_dir = config.data.results_dir
        self.nifti_dir = os.path.join(self.results_dir, "nifti")
        self.seg_dir = os.path.join(self.results_dir, "fastsurfer_masks")

        os.makedirs(self.nifti_dir, exist_ok=True)
        os.makedirs(self.seg_dir, exist_ok=True)

        # FreeSurfer DKT Atlas Labels
        self.macro_regions = {
            'Ventricles': [4, 43], # Left/Right Lateral Ventricles
            'Deep_GM':[10, 49, 11, 50, 12, 51, 13, 52], # Thalamus, Caudate, Putamen, Pallidum
            'White_Matter': [2, 41], # Cerebral White Matter
            'Cortex': list(range(1000, 1036)) + list(range(2000, 2036)) # DKT Cortical regions
        }

    def setup(self):
        """Clones FastSurfer and installs PyTorch dependencies."""
        if not os.path.exists("FastSurfer"):
            logger.info("Cloning FastSurfer...")
            subprocess.run(["git", "clone", # "--branch", "v2.4.2",
                            "https://github.com/Deep-MI/FastSurfer.git"],
                           check=True)

            # logger.info("Installing FastSurfer dependencies...")
            # subprocess.run(["pip", "install", "-r", "FastSurfer/requirements.txt"], check=True)
        else:
            logger.info("FastSurfer repository found.")

        # logger.info("Installing FastSurfer...")
        # subprocess.run(["pip", "install", "-e", "FastSurfer"], check=True)
        logger.info("Installing FastSurfer dependencies...")
        subprocess.run(["pip", "install", "yacs"], check=True)

    def convert_to_nifti(self):
        """Converts ablation .npy files to .nii.gz for FastSurfer."""
        nifti_files = sorted(glob.glob(os.path.join(self.nifti_dir, "*.nii.gz")))
        if len(nifti_files) > 0:
            logger.info(f"NIfTI files already exist in {self.nifti_dir}. Skipping conversion.")
            return

        logger.info("Converting .npy volumes to .nii.gz...")
        npy_files = sorted(glob.glob(os.path.join(self.results_dir, "*.npy")))
        
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {self.results_dir}. Run ablation mode first.")
            
        # for f in tqdm(npy_files, desc="NIfTI Conversion"):
        #     vol = np.load(f)
        #     # FastSurfer handles arbitrary resolutions via its conform module.
        #     nii = nib.Nifti1Image(vol, np.eye(4))
        #     out_name = os.path.basename(f).replace('.npy', '.nii.gz')
        #     nib.save(nii, os.path.join(self.nifti_dir, out_name))
        for f in tqdm(npy_files, desc="NIfTI Conversion"):
            vol = np.load(f) # shape is currently (Z, Y, X) and range is [0.0, 1.0]

            # 1. Restore Canonical RAS+ Orientation (X, Y, Z)
            vol_xyz = np.transpose(vol, (2, 1, 0))

            # 2. Restore MRI Intensity Range (0-255)
            # FastSurfer casts to UCHAR. If we pass [0, 1], it becomes a binary silhouette!
            vol_scaled = np.clip(vol_xyz * 255.0, 0, 255).astype(np.uint8)

            # 3. Save with Identity Affine
            nii = nib.Nifti1Image(vol_scaled, np.eye(4))
            out_name = os.path.basename(f).replace('.npy', '.nii.gz')
            nib.save(nii, os.path.join(self.nifti_dir, out_name))

    def run_prediction(self):
        """Executes FastSurfer segmentation."""
        logger.info("Running FastSurfer prediction...")
        nifti_files = sorted(glob.glob(os.path.join(self.nifti_dir, "*.nii.gz")))

        env = os.environ.copy()

        # Ensure FastSurfer's internal imports resolve correctly
        env["PYTHONPATH"] = os.path.abspath("FastSurfer") + ":" + env.get("PYTHONPATH", "")
        # The bash script relies on this environment variable
        env["FASTSURFER_HOME"] = os.path.abspath("FastSurfer")

        # Create a dummy FreeSurfer license to bypass the bash script's strict checks
        license_path = os.path.abspath("dummy_fs_license.txt")
        if not os.path.exists(license_path):
            with open(license_path, "w") as f:
                f.write("dummy@dummy.com\n12345\n12345\n12345\n")

        for f in tqdm(nifti_files, desc="FastSurfer Inference"):
            sid = os.path.basename(f).replace('.nii.gz', '')
            out_file = os.path.join(self.seg_dir, sid, "mri", "aparc.DKTatlas+aseg.deep.mgz")

            if os.path.exists(out_file):
                continue

            cmd =[
                "bash", "FastSurfer/run_fastsurfer.sh",
                "--allow_root",
                "--t1", os.path.abspath(f),
                "--seg_only",                           # skip surface reconstruction (no license needed)
                "--sd", os.path.abspath(self.seg_dir),  # subject directory root
                "--sid", sid,                           # subject ID
                "--device", "cuda",                     # force GPU
                "--batch", "16",                        # massive speedup for the neural network
                "--threads", "4"                        # speedup for CPU conforming steps
            ]
            
            # result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            # if result.returncode != 0:
            #     logger.error(f"FastSurfer Failed for {sid}:\n{result.stderr}")

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Log BOTH stdout and stderr to prevent silent failures
                logger.error(f"FastSurfer Failed for {sid}:"
                             f"\n--- STDERR ---\n{result.stderr}"
                             f"\n--- STDOUT ---\n{result.stdout}")

    def _dice_coef(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.sum(mask1 & mask2)
        vol1 = np.sum(mask1)
        vol2 = np.sum(mask2)
        if vol1 + vol2 == 0: return 1.0
        return float(2.0 * intersection / (vol1 + vol2))

    def calculate_dsc(self):
        """Computes DSC between Ground Truth and Ablation predictions."""
        logger.info("Calculating Anatomical DSC metrics...")
        
        # Find all Ground Truth segmentations
        # gt_sids =[os.path.basename(f).replace('.nii.gz', '') for f in glob.glob(os.path.join(self.nifti_dir, "gt_*.nii.gz"))]
        gt_sids =[os.path.basename(f).replace('.nii.gz', '') for f in glob.glob(os.path.join(self.nifti_dir, "gt_*.nii.gz"))]
        ablations = ['baseline', 'spade', 'buffer', 'full', 'ours_full_masked',
                     'monai_3d_ldm', 'monai_2d_ldm', 'monai_vqgan']
        
        results =[]
        
        for gt_sid in tqdm(gt_sids, desc="Scoring Volumes"):
            vol_id = gt_sid.replace('gt_', '')
            # FastSurfer saves outputs in .mgz format
            gt_path = os.path.join(self.seg_dir, gt_sid, "mri",
                                   "aparc.DKTatlas+aseg.deep.mgz")
            
            if not os.path.exists(gt_path):
                continue
                
            gt_seg = nib.load(gt_path).get_fdata()
            
            for ab in ablations:
                pred_sid = f"{ab}_{vol_id}"
                pred_path = os.path.join(self.seg_dir, pred_sid, "mri",
                                         "aparc.DKTatlas+aseg.deep.mgz")
                
                if not os.path.exists(pred_path): 
                    continue
                    
                pred_seg = nib.load(pred_path).get_fdata()
                
                # FastSurfer automatically conforms images to 256x256x256 1mm isotropic. 
                # The arrays match perfectly because they underwent the exact same spatial transformations
                for region_name, labels in self.macro_regions.items():
                    gt_mask = np.isin(gt_seg, labels)
                    pred_mask = np.isin(pred_seg, labels)
                    
                    dsc = self._dice_coef(gt_mask, pred_mask)
                    
                    results.append({
                        'Volume_ID': vol_id,
                        'Ablation': ab,
                        'Region': region_name.replace('_', ' '),
                        'DSC': dsc
                    })
                    
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "anatomical_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Anatomical metrics saved to {csv_path}")
