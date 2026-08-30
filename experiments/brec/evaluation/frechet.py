import os
import glob

import numpy as np
import pandas as pd

import torch
from scipy import linalg

import torchvision.models.video as video_models

from tqdm import tqdm

from brec.core.utils import logger
from configs.config import Config


class FrechetEvaluator:
    """
    Computes 3D-FID and FVD (Fréchet Video Distance) to measure 3D structural 
    realism and Z-axis continuity.
    """
    def __init__(self, config: Config):
        self.results_dir = config.data.results_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing 3D Feature Extractors on {self.device}...")
        
        # 1. FVD Extractor: I3D (Inception 3D) pre-trained on Kinetics-400
        # Standard for measuring sequential (Z-axis) smoothness
        self.i3d = video_models.mvit_v1_b(weights=video_models.MViT_V1_B_Weights.DEFAULT)
        self.i3d.head = torch.nn.Identity() # Strip classification head to get raw features
        self.i3d.eval().to(self.device)
        
        # 2. 3D-FID Extractor: 3D ResNet
        # Ideally MedicalNet, but TorchVision's R3D_18 is natively available and highly effective
        self.r3d = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
        self.r3d.fc = torch.nn.Identity()
        self.r3d.eval().to(self.device)

    def _extract_features(self, vol: np.ndarray, model: torch.nn.Module,
                          target_size=(16, 224, 224)) -> np.ndarray:
        # """
        # Converts a (Z, H, W) volume to (B, C, T, H, W) and extracts 3D features.
        # """
        """
        Iterates over the full (Z, H, W) volume using a sliding window, 
        extracting 3D features for every 16-slice chunk.
        """
        # 1. Resize/Crop to standard 3D input (e.g., 16 frames/slices, 112x112 spatial)
        # We sample a block from the center to represent the core brain structure
        Z, H, W = vol.shape
        # z_start = max(0, Z // 2 - 8)
        chunk_size = 16
        stride = 8  # 50% overlap for dense Z-axis coverage
        features_list =[]

        for z_start in range(0, max(1, Z - chunk_size + 1), stride):
            # Extract 16 slices
            vol_clip = vol[z_start:z_start + 16]
            vol_clip = vol[z_start:z_start+chunk_size]

            # if vol_clip.shape[0] < 16:
            #     # Pad if too small
            #     pad_width = 16 - vol_clip.shape[0]
            #     vol_clip = np.pad(vol_clip, ((0, pad_width), (0, 0), (0, 0)), mode='edge')
            # Pad if we hit the end of the volume and it's smaller than 16 slices
            if vol_clip.shape[0] < chunk_size:
                pad_width = chunk_size - vol_clip.shape[0]
                vol_clip = np.pad(vol_clip, ((0, pad_width), (0, 0), (0, 0)), mode='edge')

            # Skip chunks that are pure background (air) to avoid polluting the distribution
            if np.mean(vol_clip) < 0.01:
                continue

            # 2. Format for PyTorch Video Models: (Batch, Channels, Time/Z, Height, Width)
            # Convert grayscale to RGB by repeating channels
            vol_tensor = torch.tensor(vol_clip, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            vol_tensor = vol_tensor.repeat(1, 3, 1, 1, 1) # (1, 3, 16, H, W)

            # Interpolate spatial dimensions to 112x112 (standard for R3D/I3D)
            # vol_tensor = torch.nn.functional.interpolate(vol_tensor, size=(16, 112, 112), mode='trilinear')
            # Interpolate spatial dimensions to the exact target size required by the model
            vol_tensor = torch.nn.functional.interpolate(vol_tensor, size=target_size, mode='trilinear')

            # Normalize to ImageNet/Kinetics stats
            vol_tensor = (vol_tensor - 0.45) / 0.225

            with torch.no_grad():
                features = model(vol_tensor.to(self.device))

        return features.cpu().numpy().flatten()

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Fréchet Distance."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def evaluate(self):
        logger.info("Calculating 3D-FID and FVD...")
        
        gt_files = glob.glob(os.path.join(self.results_dir, "gt_*.npy"))
        if not gt_files:
            logger.error(f"No Ground Truth (gt_*.npy) files found in {self.results_dir}. Aborting.")
            return
        ablations = ['baseline', 'spade', 'buffer', 'full', 'ours_full_masked',
                     'monai_3d_ldm', 'monai_2d_ldm', 'monai_vqgan']
        
        # Dictionaries to store feature vectors
        features_r3d = {ab:[] for ab in ablations}
        features_r3d['gt'] = []
        
        features_i3d = {ab:[] for ab in ablations}
        features_i3d['gt'] =[]
        
        # 1. Extract Features
        for gt_path in tqdm(gt_files, desc="Extracting 3D Features"):
            vol_id = os.path.basename(gt_path).replace('gt_', '').replace('.npy', '')
            gt_vol = np.load(gt_path)

            # Extract GT features
            # features_r3d['gt'].append(self._extract_features(gt_vol, self.r3d))
            # features_i3d['gt'].append(self._extract_features(gt_vol, self.i3d))
            # Extract GT features (R3D expects 112x112, MViT expects 224x224)
            # features_r3d['gt'].append(self._extract_features(
            #     gt_vol, self.r3d, target_size=(16, 112, 112)))
            # features_i3d['gt'].append(self._extract_features(
            #     gt_vol, self.i3d, target_size=(16, 224, 224)))
            # Use .extend() because _extract_features now returns a list of chunks
            features_r3d['gt'].extend(self._extract_features(
                gt_vol, self.r3d, target_size=(16, 112, 112)))
            features_i3d['gt'].extend(self._extract_features(
                gt_vol, self.i3d, target_size=(16, 224, 224)))

            for ab in ablations:
                pred_path = os.path.join(self.results_dir, f"{ab}_{vol_id}.npy")
                if not os.path.exists(pred_path): continue

                pred_vol = np.load(pred_path)
                # features_r3d[ab].append(self._extract_features(pred_vol, self.r3d))
                # features_i3d[ab].append(self._extract_features(pred_vol, self.i3d))
                # features_r3d[ab].append(self._extract_features(
                #     pred_vol, self.r3d, target_size=(16, 112, 112)))
                # features_i3d[ab].append(self._extract_features(
                #     pred_vol, self.i3d, target_size=(16, 224, 224)))
                features_r3d[ab].extend(self._extract_features(
                    pred_vol, self.r3d, target_size=(16, 112, 112)))
                features_i3d[ab].extend(self._extract_features(
                    pred_vol, self.i3d, target_size=(16, 224, 224)))

        # 2. Compute Statistics and Fréchet Distance
        results =[]
        
        # Ground Truth Stats
        mu_gt_r3d, sig_gt_r3d = np.mean(features_r3d['gt'], axis=0), np.cov(features_r3d['gt'], rowvar=False)
        mu_gt_i3d, sig_gt_i3d = np.mean(features_i3d['gt'], axis=0), np.cov(features_i3d['gt'], rowvar=False)
        
        for ab in ablations:
            if not features_r3d[ab]: continue
                
            mu_ab_r3d, sig_ab_r3d = np.mean(features_r3d[ab], axis=0), np.cov(features_r3d[ab], rowvar=False)
            mu_ab_i3d, sig_ab_i3d = np.mean(features_i3d[ab], axis=0), np.cov(features_i3d[ab], rowvar=False)
            
            fid_3d = self._calculate_frechet_distance(mu_gt_r3d, sig_gt_r3d, mu_ab_r3d, sig_ab_r3d)
            fvd = self._calculate_frechet_distance(mu_gt_i3d, sig_gt_i3d, mu_ab_i3d, sig_ab_i3d)
            
            results.append({
                'Ablation': ab,
                '3D-FID': fid_3d,
                'FVD': fvd
            })

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "frechet_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Fréchet metrics saved to {csv_path}")
        print(df.to_string(index=False))
