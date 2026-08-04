# Brain Reconstruction (2.5D Autoregressive SPADE)

Refactored into a package from the single `brain-reconstruction-brats.ipynb` notebook.
The notebook is now a thin driver: it only imports the package, runs the XLA
bootstrap, and calls the pipeline functions — all logic lives in `.py` modules.

## Project structure

```
notebook/
├── configs/
│   ├── __init__.py
│   └── config.py            # Dataclasses: ModelConfig, DataConfig, AugmentationConfig,
│                            # TrainingConfig, HPOConfig, Config + global CFG instance
├── src/
│   ├── core/
│   │   ├── env.py           # KAGGLE detection, XLA path configuration (must run before TF import)
│   │   ├── utils.py         # logger, PipelineTimer, InferenceProfiler, MemoryTelemetry,
│   │   │                    # enforce_gpu_presence, set_global_seeds, telemetry
│   │   └── geometry.py      # GeometryOps (normalize/resize-pad/inverse), InputProcessor,
│   │                        # run_geometry_sanity_check
│   ├── data/
│   │   ├── files.py         # File discovery: EXCLUDE_FILES, find_t1_files, get_brats_subjects,
│   │   │                    # make_slice_index_list
│   │   ├── cache.py         # DataStatistics, MedicalVolume, VolumeLoader, DataLoaderBase,
│   │   │                    # StaticLoader, ActiveLoader (RAM pool + hallucination buffer)
│   │   ├── augmentations.py # AugmentationLogic: geometric, autoregressive corruption, tumor voids
│   │   └── generators.py    # GeneratorBase, IXIActiveGenerator, BraTSActiveGenerator,
│   │                        # SequentialValidationGenerator, HpoTrainSequence, HpoValidSequence,
│   │                        # create_tf_dataset, get_training_dataset
│   ├── models/
│   │   ├── layers.py        # SpectralNormalization, Sampling, VAELossLayer, InstanceNormalization,
│   │   │                    # FourierEmbedding, SPADELayer, SPADEResBlock
│   │   ├── builder.py       # ModelBuilder (unet / spade / vae), DiscriminatorBuilder
│   │   └── losses.py        # PerceptualLoss (+singleton), SpectralLoss, FocalFrequencyLoss,
│   │                        # CompositeLoss, SpatiallyWeightedL1Loss, VanillaL1Loss,
│   │                        # RelaxedMHPLossWrapper
│   ├── training/
│   │   ├── callbacks.py     # TrainingVisualizer, BufferUpdateCallback, GeneratorCheckpoint
│   │   └── trainer.py       # SPADEGANTrainer, Trainer
│   ├── inference/
│   │   └── reconstructor.py # VolumeReconstructor: autoregressive / beam-search / bidirectional restore
│   ├── evaluation/
│   │   ├── metrics.py       # get_oracle_prediction, Oracle* metrics, GradientSharpnessMetric, Evaluator
│   │   ├── visualizer.py    # analyze_dataset_geometry, VolumeDashboard, VisualizationSuite,
│   │   │                    # display_random_dashboards
│   │   ├── synthseg.py      # SynthSegEvaluator (anatomical DSC)
│   │   ├── frechet.py       # FrechetEvaluator (3D-FID / FVD)
│   │   ├── fastsurfer.py    # FastSurferEvaluator (anatomical DSC)
│   │   └── monai_sota.py    # MonaiSotaEvaluator (MONAI 3D/2D LDM + VQ-GAN comparisons)
│   └── hpo/
│       ├── engine.py        # compute_autoregressive_score, HPMEngine (Optuna)
│       ├── merge.py         # merge_hpo_databases
│       └── analysis.py      # analyze_hpo_results (Plotly reports)
├── main.py                  # Pipeline entry point: main(mode), run_pipeline, run_eda_mode,
│                            # run_ablation_study, get_ablation_configs, step-metric helpers
├── cvpr_plots.py            # CVPR figure generation (Fig1-Fig6 + supplementary)
└── brain-reconstruction-brats.ipynb  # Thin driver: imports + calls
```

## Configuration (`configs/config.py`)

The global config is instantiated once at import time as `CFG = Config()` and used
by the whole pipeline. Every section is a dataclass:

| Section | Role |
|---|---|
| `ModelConfig` | Architecture (`unet`/`spade`/`vae`), EfficientNet backbone (`B0`–`B7`), SPADE decoder, positional encoding, number of hypotheses (MHP). |
| `DataConfig` | Data roots (IXI/BraTS), native/padded patch sizes, neighborhood (N slices, default 3), projections, RAM cache sizes, hallucination buffer size, weights/results/figures directories. |
| `AugmentationConfig` | Tumor void injection, blur/noise corruption cascade, hallucination replay probability, flip/rotation. |
| `TrainingConfig` | Batch size, epochs, learning rate, composite-loss weights, GAN mode + weights, perceptual backbone, resume path. |
| `HPOConfig` | Optuna trials, train/val steps per trial, AR metric weight, storage/database names. |
| `Config` | Master object holding all sections; `run_type`, `batch_mode`, `seed`, `to_dict/save/load`. |

`Config.__post_init__` derives the model name, batch mode, epoch budget
(Interactive mode drops epochs to 5), input channel count, and clears the HPO
preload database outside Kaggle.

## How it works (core concepts)

1. **2.5D autoregressive reconstruction.** The model predicts slice `t` from a
   stack of N previous slices `[t-N … t-1]` (2.5D context) plus a binary target
   mask and positional encodings. During inference, predictions feed back into
   the context (`VolumeReconstructor.autoregressive_restore`), rolling out slice
   by slice in either direction.

2. **Data pipeline.** `ActiveLoader` keeps a RAM pool of loaded volumes (IXI +
   BraTS) in a background thread; generators sample random slices, inject real
   tumor voids from BraTS, apply geometric/photometric augmentation, and stream
   samples through `create_tf_dataset`. A hallucination replay buffer stores
   earlier model outputs and re-injects them to stabilize long rollouts.

3. **Models & losses.** `ModelBuilder` builds an EfficientNet encoder with a
   SPADE decoder (or plain U-Net); `CompositeLoss` blends region-weighted MAE
   (healthy/tumor/background), gradient, perceptual, and spectral Fourier terms.
   `SpatiallyWeightedL1Loss` and GAN training (`SPADEGANTrainer`) are available.

4. **Evaluation.** Per-step rollout metrics (SSIM/PSNR/MAE/grad/FFL) are saved
   during ablation runs, and downstream validation uses SynthSeg or FastSurfer
   (DSC), 3D-FID/FVD (FrechetEvaluator), and MONAI generative baselines
   (`MonaiSotaEvaluator`). `cvpr_plots.py` renders the paper figures.

5. **HPO.** `HPMEngine` keeps a fixed RAM subset, reuses one compiled model
   graph, and scores trials with a blend of one-shot validation loss and an
   autoregressive rollout metric via Optuna.

## Running

```bash
# Full ablation study + CVPR figures (default)
python main.py

# Or via the notebook: set mode and run the final cell
mode = 'ablation'   # 'train' | 'hpo' | 'eda' | 'ablation' | 'sota' | 'dsc' | 'frechet' | 'cvpr' | 'all'
main(mode=mode)
```

Individual pipeline steps can also be invoked directly from notebook cells, e.g.
`run_pipeline(mode='train')`, `run_ablation_study(CFG)`,
`run_cvpr_rendering(render_supp=True)`.

### Environment notes

- XLA configuration (`configure_xla_paths`) must run **before** TensorFlow is
  imported; the bootstrap lives in `src/core/env.py` (no TF import there) and is
  called by the first notebook cell / by `import src.core.env`.
- `KAGGLE_URL_BASE=1` selects Kaggle data paths and skips the GPU-presence guard;
  otherwise a GPU is enforced at import time.
