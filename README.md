# Bounding Exposure Bias and Spatial Drift in Domain-Specific 2.5D Medical Image Synthesis

[![Paper](https://img.shields.io/badge/Paper-Accepted-success.svg)](#)
[![ICIVC](https://img.shields.io/badge/Venue-ICIVC%202026-blue.svg)](http://www.icivc.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Evaluators-blue.svg)](https://pytorch.org)

Official implementation of the paper **"Bounding Exposure Bias and Spatial Drift in Domain-Specific 2.5D Medical Image Synthesis"**, accepted at the **11th International Conference on Image, Vision and Computing (ICIVC 2026)**.

This repository implements **2.5D SPADE-AR**, a computationally efficient framework for sequential 3D medical volume reconstruction and synthesis. The model addresses unconstrained autoregressive drift (exposure bias) by utilizing a dynamic, continuous pixel-space Dataset Aggregation (DAgger) paradigm via a Hallucination Replay Buffer, combined with decoupled Absolute Fourier Positional Encodings.

---

## Key Features

1. **Continuous-Space DAgger (Hallucination Replay):** Bypasses Backpropagation Through Time (BPTT) memory limitations by continuously updating an interleaved `Hallucination Replay Buffer` using $K$-step unconstrained online rollouts. The network learns a contractive projection mapping back to the clean data manifold.
2. **Decoupled Local & Global Grounding:** Enforces local cross-sectional macro-structures using Spatially-Adaptive Normalization (SPADE) blocks while keeping the encoder strictly blind to spatial layout shortcuts. Global positioning along the sequential Z-axis is established by injecting **Absolute Fourier Positional Encodings** at the bottleneck.
3. **Set-Theoretic Mask-Aware Composite Loss:** Formulates localized Mean Absolute Error boundaries ($\Omega_{\text{bg}}$, $\Omega_{\text{tumor}}$, $\Omega_{\text{healthy}}$) combined with a scale-normalized 2D Real FFT log-magnitude Spectral Loss and a spatial gradient loss.
4. **Decoupling/Decoding Strategies:** Built-in support for greedy Autoregressive, RL-guided **Beam Search**, and **Bidirectional** context merging restoration.
5. **Pareto-Efficient Sub-1GB VRAM footprint:** Matches or exceeds 3D Latent Diffusion Model (LDM) anatomical fidelity (SynthSeg/FastSurfer DSC) and sequential continuity (FVD) using a fraction of the computational footprint.

---

## Workflow Execution: All-in-One Notebook

To facilitate running on resource-constrained clinics or free cloud compute accelerators, the entire pipeline is structured as a self-contained, multi-backend notebook:

📂 **`experiments/autoregressive.ipynb`**

This notebook is pre-configured to run seamlessly across three target environments (Local, Kaggle, Colab) by detecting path structures automatically:

### 1. Cloud Dataset Paths
```python
# Automatically resolved in the notebook base configuration cell
KAGGLE = True # auto-detected via KAGGLE_URL_BASE environment flag

# Dataset mount points
PATH_DATA_IXI = '/kaggle/input/preprocessed-oasis-and-epilepsy-and-ixi'
PATH_DATA_BRATS = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
```

### 2. XLA Path Auto-Configuration
In Conda and cloud notebook environments, the codebase resolves GPU JIT compilation paths dynamically to avoid silent fallback to CPU training:
```python
# Locates libdevice.10.bc dynamically and configures NVVM paths
configure_xla_paths()
```

---

## Getting Started

### Prerequisites & Cloud Run Setup

To run the pipeline inside cloud nodes (Kaggle or Google Colab), ensure the following datasets and dependencies are mounted:

#### A. Mount Datasets on Kaggle:
1. Search and add `preprocessed-oasis-and-epilepsy-and-ixi` to input.
2. Search and add `brats20-dataset-training-validation` to input.

#### B. Execute Pip Installs (First cell of the notebook):
```bash
# Installs custom deep validation evaluators
pip install git+https://github.com/ValV/SynthSeg.git
pip install -e ./SynthSeg
```

---

## Configuration & Pipeline Settings

Parameters can be edited inline inside the configuration dataclasses of **`experiments/autoregressive.ipynb`**:

* **`mode` selection:** Set to `'all'` (runs full pipeline), `'ablation'` (runs 4 CVPR core ablations), `'hpo'` (launches Optuna hyperparameter optimization), or `'eda'` (renders volume slice distribution dashboards).
* **`CFG.run_type` detection:** Interactively toggles execution levels. If in **Interactive Mode**, the notebook reduces rollout lengths and optimization steps to protect against cloud timeout limitations.

---

## Architectural & Algorithmic Formulations

### Set-Theoretic Mask-Aware Loss
The composite loss is strictly evaluated on the target spatial grid slice index $t$:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{bg}}\mathcal{L}_{\text{bg}}(\Omega \setminus M_t) + \lambda_{\text{tumor}}\mathcal{L}_{\text{tumor}}(V_t) + \lambda_{\text{healthy}}\mathcal{L}_{\text{healthy}}(M_t \setminus V_t) + \lambda_{\text{grad}}\mathcal{L}_{\text{grad}} + \lambda_{\text{spec}}\mathcal{L}_{\text{spec}}$$

### Spectral Loss
We penalize spectral distribution discrepancies across a single-side coordinate grid using a scale-normalized $L_1$ log-magnitude difference:

$$\mathcal{L}_{\text{spec}} = \mathbb{E}_{x_t \sim \mathcal{D}} \left[ \frac{1}{|\mathcal{F}_t|} \left\| \log(|\mathcal{F}(\hat{x}_t)| + 1) - \log(|\mathcal{F}(x_t)| + 1) \right\|_1 \right]$$

---

## Citation

If you find this work or codebase useful for your research, please cite our conference paper:

```bibtex
@inproceedings{valeyev2026bounding,
  title={Bounding Exposure Bias and Spatial Drift in Domain-Specific 2.5D Medical Image Synthesis},
  author={Valeyev, Vladimir and Zubanenko, Aleksey and Tomilov, Ivan and Gusarova, Natalia and Vatian, Aleksandra},
  booktitle={Proceedings of the 2026 11th International Conference on Image, Vision and Computing (ICIVC)},
  year={2026}
}
```
