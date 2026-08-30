import json
import os

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional

import numpy as np

from brec.core.env import KAGGLE, PATH_DATA_IXI, PATH_DATA_BRATS
from brec.core.utils import logger


@dataclass
class ModelConfig:
    """Hyperparameters for the Neural Network Architecture."""

    name: str = '2p5d'  # prefix, e. g. 2p5d_unet, 2p5d_spade, etc
    architecture: str = (
        'spade'  # 'unet', 'spade', 'vae' etc (TODO: 'vae', 'gan', etc)
    )
    backbone: Optional[str] = (
        'B0'  # 'B0'...'B7' for EfficientNet, or None for custom
    )
    use_attention: bool = False  # use Attention Gates in Decoder
    use_stem: bool = False  # use initial Conv stem
    dropout_rate: float = 0.1
    activation: str = 'gelu'
    # Input channels: Neighborhood (slices) + 1 (Mask, optional)
    input_channels: int = 3 + 0
    base_filters: int = 64  # replaces hardcoded 32/64 starts
    use_positional_encoding: bool = False
    num_hypotheses: int = 1  # set to 1 for baseline, >1 (e.g., 4) for MHP
    mhp_epsilon: float = 0.05  # relaxation parameter to prevent dead heads (https://arxiv.org/abs/1612.00197)


@dataclass
class DataConfig:
    """Settings for Data Loading, Caching, and Preprocessing."""

    # Paths
    data_root_ixi: str = PATH_DATA_IXI
    data_root_brats: str = PATH_DATA_BRATS
    ixi_sampling_weight: float = 1.0  # 70% IXI, 30% BraTS by default

    # Dimensions
    # The native patch size found in EDA (approximate)
    native_patch_size: Tuple[int, int] = (113, 137)
    # The size fed into the model (multiples of 32 for UNet/EffNet)
    # padded_size: Tuple[int, int] = (128, 160)
    # Increased to square to allow safe rotation
    padded_size: Tuple[int, int] = (160, 160)
    # Number of slices for 2.5D context (Must be odd: t-1, t, t+1)
    neighborhood: int = 3
    # Whether to append the target mask as an input channel
    input_last_target_mask: bool = True

    # RAM Management
    # Capacity: How many full volumes to keep in RAM at once.
    # Increasing this improves diversity but requires more RAM

    # Projections to train on.
    # 0: Axial (standard), 1: Coronal, 2: Sagittal
    projections: Tuple[int, ...] = (0,)  # (0, 1, 2) for all three

    # Caching (RAM Management)
    # BraTS takes ~2x RAM per volume due to segmentation masks
    ixi_cache_size: int = 120 if KAGGLE else 240
    brats_cache_size: int = 120 if KAGGLE else 240
    # Number of slices in the replay buffer
    # On Kaggle we sample 128 * 3 * 5 = 1920 (last 5 epochs);
    # on cluster we sample 384 * 3 * 7 = 8064 (last 7 epochs)
    hallucination_buffer_size: int = (
        1920 if KAGGLE else 8064
    )  # 5760 for the last 5 epochs

    # ---> NEW: Centralized Ablation Path <---
    weights_dir: str = None
    preload_dir: str = None
    results_dir: str = 'ablation_results'
    figures_dir: str = 'paper_figures'


@dataclass
class AugmentationConfig:
    """Probabilities and params for Data Augmentation."""

    # Tumor Injection
    prob_glioma: float = 0.65
    brats_labels_to_keep: Tuple[int, ...] = (
        0,
    )  # keep background, replace others
    brats_labels_for_void: Tuple[int, ...] = (1, 2, 4)  # tumor labels

    # Photometric / Corruption
    prob_autoregressive_candidate: float = 0.65  # chance a slice is corruptible
    # prob_blur: float = 0.25 # asymmetric blur
    # prob_noise: float = 0.25 # asymmetric noise
    blur_sigma: float = 0.85
    noise_std: float = 0.025
    prob_hallucination_max: float = (
        0.65  # maximum possible hallucination probability
    )
    prob_hallucination_replay: float = 0.0  # starts at 0, ramps up via callback
    prob_hallucination_warmup: int = 5  # epochs to increase aug prob to max

    # Geometric
    prob_flip: float = 0.5  # horizontal flip
    prob_rotate: float = 0.35  # rotation
    rotate_range: float = (
        3.5  # degrees (+/-) 45 is too aggressive for brain fitting
    )


@dataclass
class TrainingConfig:
    """Hyperparameters for the Training Loop."""

    batch_size: int = 64
    epochs: int = 60 if KAGGLE else 60  # 80 for U-Net
    # U-Net
    learning_rate: float = np.pi * 1e-4
    # Loss Weights (CompositeLoss)
    # --- Balanced (HPO) coefficients ---
    lambda_tumor: float = 0.225  # MAE inside tumor
    lambda_healthy: float = 0.268  # MAE healthy tissue
    lambda_grad: float = 0.416  # Gradient loss (sharpness)
    lambda_background: float = 0.727  # MAE background
    lambda_perceptual: float = 0.0003  # VGG/EffNet feature loss
    lambda_spectral: float = 0.078  # Fourier feature loss
    # --- Texture-forced (heuristic) coefficients ---
    # lambda_tumor: float = 0.385       # MAE inside tumor
    # lambda_healthy: float = 0.415     # MAE healthy tissue
    # lambda_grad: float = 1.000        # Gradient loss (sharpness)
    # lambda_background: float = 0.215  # MAE background
    # lambda_perceptual: float = 0.0003 # VGG/EffNet feature loss
    # lambda_spectral: float = 0.105    # Fourier feature loss
    # Alternative
    use_spatial_loss: bool = (
        False  # toggle to switch between CompositeLoss and SpatialL1 (U-Net)
    )
    use_unmasked_loss: bool = (
        False  # config option to strip spatial constraints from loss
    )
    # TRAINING MODE FLAG
    # If True: Uses SPADEGANTrainer (Discriminator + Adversarial Loss)
    # If False: Uses Standard model.fit (L1/Composite Loss only)
    gan_mode: bool = False
    # GAN Weights
    learning_rate_g: float = 1e-4  # generator
    learning_rate_d: float = 1e-4  # discriminator
    weight_gan: float = 1.5  # adversarial
    weight_fm: float = 0.45  # feature matching
    weight_l1: float = 1.75  # spatial reconstruction
    weight_perceptual: float = 0.25  # VGG/EffNet perceptual
    weight_spectral: float = 0.25  # VGG/EffNet perceptual
    weight_kl: float = (
        0.05  # standard starting point for VAE-GANs (0.01 to 0.1)
    )
    # Loss
    perceptual_backbone: str = 'effnet'  # 'vgg' or 'effnet'
    perceptual_init: str = 'imagenet'  # 'imagenet' or None
    # Model saving and resuming
    resume_training: str = None


@dataclass
class HPOConfig:
    n_trials: int = 12 if not KAGGLE else 12
    # Increase steps for better statistical significance
    train_steps_per_trial: int = 200
    val_steps_per_trial: int = 50

    # --- New: HPO Data Supply Proportions ---
    base_train_volumes: int = 160  # total training volumes in RAM during HPO
    base_val_volumes: int = 40  # total validation volumes in RAM during HPO
    base_eval_volumes: int = (
        20  # total evaluation volumes for Autoregressive scoring
    )

    # --- New: HPO Evaluation Metrics ---
    ar_rollout_steps: int = 5  # steps to predict into the future for AR score
    metric_ar_weight: float = (
        0.65  # weight of AR score vs. One-Shot Validation Loss (0.0 to 1.0)
    )

    # Output
    best_params_file: str = "best_hyperparams.json"

    # Distributed HPO settings
    study_name: str = "scratchnet"
    storage_dir: str = "./databases"  # HPO storage where to save sqlite files
    database_name: str = "hpo_final.db"
    # database_preload: str = '/kaggle/input/notebooks/valvex/brain-reconstruction-brats/hpo_final.db'
    database_preload: str = ''
    tracemalloc: bool = False


@dataclass
class Config:
    """Master Configuration Object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugmentationConfig = field(default_factory=AugmentationConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)

    run_type: str = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive')
    batch_mode: bool = False  # Interactive by default
    seed: int = 42

    def __post_init__(self):
        self.model.name = f"{self.model.name}_{self.model.architecture}"
        self.batch_mode = self.run_type != 'Interactive'
        # Interactive mode adjustment
        if self.run_type == 'Interactive':
            logger.info("⚠️ Interactive mode detected. Reducing Epochs.")
            self.train.epochs = 5  # debug
        # Update channel count based on data settings
        self.model.input_channels = self.data.neighborhood + int(
            self.data.input_last_target_mask
        )
        if not KAGGLE:
            self.hpo.database_preload = ''

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        # Helper to recursively map dict to dataclasses
        # (Simplified logic, assuming standard structure)
        cfg = cls()
        for section, params in data.items():
            if hasattr(cfg, section):
                sec_obj = getattr(cfg, section)
                for k, v in params.items():
                    setattr(sec_obj, k, v)
        return cfg


# Instantiate the global config
CFG = Config()

logger.info(
    f"Configuration loaded. Model Input Shape: {CFG.data.padded_size} x {CFG.model.input_channels}"
)
