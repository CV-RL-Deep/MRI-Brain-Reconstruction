import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import glob
import random

from brec.core.env import KAGGLE, configure_xla_paths

if not KAGGLE:
    configure_xla_paths()

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from brec.configs.config import CFG, Config
from brec.core.utils import logger, PipelineTimer, InferenceProfiler
from brec.core.geometry import GeometryOps
from brec.data.files import find_t1_files, get_brats_subjects
from brec.data.cache import ActiveLoader, StaticLoader, VolumeLoader
from brec.data.generators import (
    IXIActiveGenerator,
    BraTSActiveGenerator,
    SequentialValidationGenerator,
    create_tf_dataset,
    get_training_dataset,
)
from brec.inference.reconstructor import VolumeReconstructor
from brec.evaluation.visualizer import (
    analyze_dataset_geometry,
    VisualizationSuite,
    VolumeDashboard,
)
from brec.hpo.engine import HPMEngine
from brec.hpo.merge import merge_hpo_databases
from brec.hpo.analysis import analyze_hpo_results
from brec.models import losses as _losses
from brec.models.builder import ModelBuilder
from brec.training.trainer import Trainer
from brec.utils import calculate_step_metrics


def get_ablation_configs(base_cfg):
    """Generates the 4 specific states for the CVPR ablation study."""
    import copy

    configs = {}

    # 0. Vanilla Drift U-Net (No Spatial Input, No Spatial Loss, No SPADE, No Buffer, No PE)
    cfg_drift = copy.deepcopy(base_cfg)
    cfg_drift.model.architecture = 'unet'
    cfg_drift.data.input_last_target_mask = False
    cfg_drift.train.use_unmasked_loss = True
    cfg_drift.aug.prob_hallucination_max = 0.0
    cfg_drift.aug.prob_hallucination_replay = 0.0
    cfg_drift.aug.prob_glioma = 0.0
    cfg_drift.aug.prob_autoregressive_candidate = 0.0
    cfg_drift.model.use_positional_encoding = False
    configs['drift_vanilla'] = cfg_drift

    # 1. Baseline U-Net (No SPADE, No Buffer, No PE)
    cfg_base = copy.deepcopy(base_cfg)
    cfg_base.model.architecture = 'unet'
    cfg_base.aug.prob_hallucination_max = 0.0
    cfg_base.aug.prob_hallucination_replay = 0.0
    cfg_base.model.use_positional_encoding = False
    configs['baseline'] = cfg_base

    # 2. + SPADE (SPADE on, No Buffer, No PE)
    cfg_spade = copy.deepcopy(base_cfg)
    cfg_spade.model.architecture = 'spade'
    cfg_spade.aug.prob_hallucination_max = 0.0
    cfg_spade.aug.prob_hallucination_replay = 0.0
    cfg_spade.model.use_positional_encoding = False
    configs['spade'] = cfg_spade

    # 3. + Buffer (SPADE on, Buffer on, No PE)
    cfg_buffer = copy.deepcopy(base_cfg)
    cfg_buffer.model.architecture = 'spade'
    cfg_buffer.aug.prob_hallucination_max = 0.65
    cfg_buffer.model.use_positional_encoding = False
    configs['buffer'] = cfg_buffer

    # 4. Full Model (+ PE)
    cfg_full = copy.deepcopy(base_cfg)
    cfg_full.model.architecture = 'spade'
    cfg_full.aug.prob_hallucination_max = 0.65
    cfg_full.model.use_positional_encoding = True
    configs['full'] = cfg_full

    return configs


def run_eda_mode(
    config, train_files, train_brats, input_only=False, downsample=2
):
    logger.info(">>> STARTING EDA DASHBOARD MODE (PIL Optimized) <<<")

    manager = ActiveLoader(config, train_files, train_brats)
    manager.start()

    gen_ixi = iter(IXIActiveGenerator(manager)())
    gen_brats_clean = iter(BraTSActiveGenerator(manager, mode='clean')())
    gen_brats_tumor = iter(BraTSActiveGenerator(manager, mode='tumor')())

    dashboards = {}

    w_ixi = max(0.1, min(0.9, config.data.ixi_sampling_weight))
    w_brats_tumor = 0.20  # FIXME: 20% slices with tumor assumption

    batch_size = config.train.batch_size
    # FIXME: average 130 slices contain brain tissue assumption
    train_steps = max(100, (len(train_files) * 130) // batch_size)
    # Fetch actual epochs from config
    epochs = config.train.epochs
    total_samples = train_steps * batch_size * epochs

    logger.info(f"Simulating {total_samples} discrete samples...")

    for _ in tqdm(range(total_samples), desc="Simulating Epoch Flow"):
        # 1. Pipeline Probability Flow
        if random.random() < w_ixi:
            dir_name = 'eda_ixi_clean'
            model_inputs, y, info = next(gen_ixi)
        else:
            if random.random() < w_brats_tumor:
                dir_name = 'eda_brats_tumor'
                model_inputs, y, info = next(gen_brats_tumor)
            else:
                dir_name = 'eda_brats_clean'
                model_inputs, y, info = next(gen_brats_clean)

        # 2. Enforce Strict 160x160 Padding (Fixes Slice Mismatches)
        hist_pad, _, _ = GeometryOps.resize_and_pad(
            tf.convert_to_tensor(model_inputs['history_input']),
            config.data.padded_size,
            'bicubic',
        )
        y_pad, _, _ = GeometryOps.resize_and_pad(
            tf.convert_to_tensor(y), config.data.padded_size, 'nearest'
        )

        hist_np = hist_pad.numpy()
        y_np = y_pad.numpy()

        # 3. Dashboard Routing
        path = info['volume_path']
        axis = info['axis']
        dash_key = (dir_name, path, axis)

        if dash_key not in dashboards:
            dashboards[dash_key] = VolumeDashboard(
                path, info['axis_size'], axis, dir_name, downsample=downsample
            )
        dash = dashboards[dash_key]

        # NEW: Increment the sequence Z-index for this specific brain
        dash.z_counter += 1
        current_z = dash.z_counter

        N = config.data.neighborhood
        spatial_indices = info['spatial_indices']

        # Target tumor mask is channel 1 of padded Y
        padded_tumor_mask = y_np[:, :, 1]

        # Update Input Slices
        for i in range(N):
            img_slice = hist_np[:, :, i]
            s_idx = spatial_indices[i]
            role = f"T-{N - i}"
            dash.update(
                s_idx,
                img_slice,
                role,
                info,
                z_index=current_z,
                is_target=False,
                tumor_mask=padded_tumor_mask,
            )

        # Update Target Slice
        if not input_only:
            target_img = y_np[:, :, 0]
            target_s_idx = spatial_indices[-1]
            dash.update(
                target_s_idx,
                target_img,
                "T",
                info,
                z_index=current_z,
                is_target=True,
                tumor_mask=padded_tumor_mask,
            )

    VisualizationSuite.plot_sampling_statistics(manager)

    manager.stop()

    logger.info(
        f"Rendering {len(dashboards)} unique volume dashboards to disk..."
    )
    for dash in tqdm(dashboards.values(), desc="Rendering Images"):
        dash.render(input_only=input_only)

    logger.info("EDA Dashboard Generation Complete!")


def run_pipeline(mode='train', test_mode=False):
    config = CFG

    # Override for Quick Kaggle Test
    test_mode = config.run_type == 'Interactive'
    if test_mode:
        logger.info(
            ">>> TEST MODE ENABLED: Reducing HPO parameters for sanity check."
        )
        # config.train.epochs = 2
        config.hpo.n_trials = 2
        config.hpo.train_steps_per_trial = 10
        config.hpo.val_steps_per_trial = 5
        # config.data.ixi_cache_size = 4 # tiny pool

    # 0. Load Overrides
    if mode == 'train' and os.path.exists("best_hyperparams.json"):
        logger.info("Loading optimized hyperparameters...")
        config = Config.load("best_hyperparams.json")

    with PipelineTimer("1. File Discovery & Split"):
        # 1. Discover Files (Required before Manager init)
        t1_files = find_t1_files(config.data.data_root_ixi)
        brats_list = get_brats_subjects(config.data.data_root_brats)

        print(">>> Checking IXI:")
        analyze_dataset_geometry(t1_files, num_samples=2)

        print("\n>>> Checking BraTS:")
        analyze_dataset_geometry(brats_list, num_samples=2)

        # 2. Split
        random.shuffle(t1_files)
        # split = int(0.9 * len(t1_files))
        # train_files, val_files = t1_files[:split], t1_files[split:]
        random.shuffle(t1_files)
        ixi_split = int(0.9 * len(t1_files))
        train_files, val_files = t1_files[:ixi_split], t1_files[ixi_split:]

        # 2b. Split BraTS (Fixing the blind validation bug)
        random.shuffle(brats_list)
        brats_split = int(0.9 * len(brats_list))
        train_brats, val_brats = (
            brats_list[:brats_split],
            brats_list[brats_split:],
        )

        # Calculate Steps (Heuristic based on volume size ~130 slices)
        # Since we use random sampling, we define an "epoch" arbitrarily to match dataset size
        avg_slices = 130
        train_steps = max(
            100, (len(train_files) * avg_slices) // config.train.batch_size
        )
        val_steps = max(
            50, (len(val_files) * avg_slices) // config.train.batch_size
        )

        if mode == 'train':
            logger.info(f"Train Steps: {train_steps}, Val Steps: {val_steps}")

    if mode == 'hpo':
        # logger.info(">>> STARTING HPO MODE")
        # run_hpo_session(20, config, t1_files, brats_list)
        # return
        logger.info(">>> STARTING HPO MODE (Autoregressive Metric)")
        # run_hpo_session(20, config, train_files, brats_list)
        # Instantiate Engine
        hpo_engine = HPMEngine(config, train_files, brats_list)
        hpo_engine.run()
        # FIXME: decouple HPO merge and analyze, do not run in parallel
        merge_hpo_databases(
            config.hpo.storage_dir,
            config.hpo.study_name,
            config.hpo.database_name,
        )
        analyze_hpo_results(
            f"sqlite:///{config.hpo.database_name}"
        )  # "sqlite:///hpo_final.db"
        return

    if mode == 'eda':
        run_eda_mode(config, train_files, train_brats, input_only=False)
        return

    # --- Validation Loader (Static) ---
    # We load validation data into RAM once.
    # If val set is too big, use a smaller subset or switch to ActiveLoader logic for Val too.
    # For safety with 240 volumes, loading 10% (24 vols) is fine
    # val_loader = StaticLoader(config, val_files, [])
    val_loader = StaticLoader(config, val_files, val_brats)

    # --- Training Loader (Active) ---
    # Context manager ensures thread cleanup
    # with ActiveLoader(config, train_files, brats_list) as train_loader:
    # --- Training Loader (Active) ---
    with ActiveLoader(config, train_files, train_brats) as train_loader:
        # 3. Setup Generators
        with PipelineTimer("2. Generator & Dataset Setup"):
            # Training Generators (Infinite)
            # Note: We use the new class names 'IXIActiveGenerator'
            train_ixi = IXIActiveGenerator(train_loader)
            train_brats = BraTSActiveGenerator(train_loader, mode='clean')

            train_ds = get_training_dataset(train_ixi, train_brats, config)

            # Validation Generator (Reuse ActiveGen class but with StaticLoader)
            # It works because they share the 'get_volume' interface
            # val_gen = IXIActiveGenerator(val_loader)
            # val_ds = create_tf_dataset(val_gen, config, is_training=False).take(val_steps)
            # Use the Unbiased, Deterministic Validation Generator
            val_gen = SequentialValidationGenerator(
                val_loader, max_slices_per_vol=15
            )
            # Remove .take() because it is natively finite now!
            val_ds = create_tf_dataset(val_gen, config, is_training=False)
            # Since val_ds is finite, tell Trainer to iterate until exhaustion
            val_steps = None  # FIXME: refine logic flow

            # Build Model
            model = ModelBuilder.build(config)

            # FIX: Pass the model to the loader so the background thread can
            # generate hallucinations without pausing the training loop!
            train_loader.model = model

            # Trainer
            trainer = Trainer(
                config,
                model,
                train_ds,
                val_ds,
                train_loader,
                train_steps=train_steps,
                val_steps=val_steps,
            )

        # 4. Pre-Training Visualization
        with PipelineTimer("3. visualization (Data Augmentation Check)"):
            # We create a specific visualization dataset using the Training Generators
            # This enables "hunting" for rare augmentations

            # Create finite datasets for sampling
            viz_ixi = create_tf_dataset(
                train_ixi, config, is_training=True, include_info=True
            )
            viz_brats = create_tf_dataset(
                train_brats, config, is_training=True, include_info=True
            )

            # Mix them (Weighted)
            # Use Configurable Weights
            w_ixi = config.data.ixi_sampling_weight
            # Clip to safety
            w_ixi = max(0.1, min(0.9, w_ixi))
            w_brats = 1.0 - w_ixi
            # Use repeat() to allow 'hunting' through many samples
            viz_ds = tf.data.Dataset.sample_from_datasets(
                [viz_ixi.repeat(), viz_brats.repeat()], weights=[w_ixi, w_brats]
            )

            # CRITICAL: Batch the viz dataset, otherwise plotter crashes on scalars
            viz_ds = viz_ds.batch(1)

            VisualizationSuite.plot_augmentations(
                viz_ds, num_groups=2, batch_mode=config.batch_mode
            )

        # 5. Training Loop
        name_model = f"{config.model.name}_best.keras"
        if mode == 'viz' and os.path.exists(name_model):
            logger.info(">>> STARTING VISUALIZATION/ABLATION MODE")
            logger.info(f"Loading Pretrained Model [{name_model}]...")
            model.load_weights(name_model)
            history = None
        else:
            logger.info(">>> STARTING LONG-RUN TRAINING MODE")

            # --- RESTORED: RESUME TRAINING LOGIC ---
            if config.train.resume_training:
                if os.path.exists(config.train.resume_training):
                    logger.info(
                        f"♻️ Resuming training from '{config.train.resume_training}'..."
                    )
                    # Load weights into the generator before the Trainer potentially wraps it in GAN mode
                    model.load_weights(config.train.resume_training)
                else:
                    logger.warning(
                        f"⚠️ Resume path '{config.train.resume_training}' not found. Starting from scratch..."
                    )
            # ---------------------------------------

            history = trainer.train()

        # 6. Post-Training Analysis
        with PipelineTimer("4. Post-Training Analysis"):
            # Necessary visualizations for 'train' and 'vis' modes
            if history:
                VisualizationSuite.plot_training_history(history)

            VisualizationSuite.plot_hallucination_buffer(train_loader)

            # Use Val DS for best/worst
            VisualizationSuite.analyze_best_worst(model, val_ds, num_samples=5)

            # Reconstructions
            reconstructor = VolumeReconstructor(model, config)

            # Autoregressive on a real validation file
            # Note: val_loader has 'pool'. We can grab a path from there
            if val_files:
                VisualizationSuite.plot_autoregressive_performance(
                    reconstructor,
                    {'id': None, 't1': val_files[0], 'seg': None},
                    val_loader,
                )
            # Autoregressive on BraTS
            if brats_list:
                VisualizationSuite.plot_autoregressive_performance(
                    reconstructor, brats_list[0], train_loader
                )
                VisualizationSuite.plot_autoregressive_performance(
                    reconstructor,
                    brats_list[0],
                    train_loader,
                    masked_inference=True,
                )

            # Bidirectional on a real validation file
            # Note: val_loader has 'pool'. We can grab a path from there
            if val_files:
                VisualizationSuite.plot_bidirectional(
                    reconstructor,
                    {'id': None, 't1': val_files[0], 'seg': None},
                    val_loader,
                )
            # Bidirectional on BraTS
            if brats_list:
                # We need a BraTS file. train_loader has them
                VisualizationSuite.plot_bidirectional(
                    reconstructor, brats_list[0], train_loader
                )
                VisualizationSuite.plot_bidirectional(
                    reconstructor,
                    brats_list[0],
                    train_loader,
                    masked_inference=True,
                )

            # Data sampling statistics
            VisualizationSuite.plot_sampling_statistics(train_loader)
            if mode == 'viz':
                # TODO: extensive visualizations
                pass


def run_ablation_study_(
    base_config, train_files, val_files, train_brats, val_brats
):
    logger.info(">>> STARTING CVPR ABLATION STUDY <<<")
    configs = get_ablation_configs(base_config)

    weights_dir = base_config.data.weights_dir
    results_dir = base_config.data.results_dir
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []

    # --- KAGGLE INTERACTIVE PROTECTIONS ---
    is_interactive = base_config.run_type == 'Interactive'
    n_eval_vols = 2 if is_interactive else 10
    rollout_span = (
        5 if is_interactive else 20
    )  # 10 steps total interactively vs 40 steps for prod

    if KAGGLE:
        # Prevent System RAM OOM by drastically shrinking the active caching pools
        base_config.data.ixi_cache_size = 100
        base_config.data.brats_cache_size = 100
    # --------------------------------------

    # 1. Lock in a fixed set of Test Volumes for fair comparison
    test_ixi = val_files[:n_eval_vols]
    test_brats = val_brats[:n_eval_vols]

    # 2. Prevent Kaggle Batch OOM: Cap static validation pool globally for Kaggle
    max_val_vols = 20 if KAGGLE else 50
    val_loader = StaticLoader(
        base_config, val_files[:max_val_vols], val_brats[:max_val_vols]
    )

    for ab_name, cfg in configs.items():
        logger.info(f"=============================================")
        logger.info(f"   ABLATION PHASE: {ab_name.upper()}")
        logger.info(f"=============================================")

        # model_weights_path = f"{ABLATION_DIR}/model_{ab_name}.keras"
        primary_weights_path = os.path.join(
            weights_dir, f"model_{ab_name}.keras"
        )
        local_weights_path = os.path.join(results_dir, f"model_{ab_name}.keras")

        # We must rebuild the active loader because the config dictates hallucination probabilities
        with ActiveLoader(cfg, train_files, train_brats) as train_loader:
            # --- MODEL BUILDING & TRAINING ---
            model = ModelBuilder.build(cfg)
            train_loader.model = model  # attach for Buffer Updates

            if os.path.exists(primary_weights_path):
                logger.info(
                    f"✅ Found existing weights in dataset for {ab_name}. Skipping training!"
                )
                logger.debug(f"Model weights = {primary_weights_path}")
                model.load_weights(primary_weights_path)
            elif os.path.exists(local_weights_path):
                logger.info(
                    f"✅ Found locally trained weights for {ab_name}. Skipping training!"
                )
                logger.debug(f"Model weights = {local_weights_path}")
                model.load_weights(local_weights_path)
            else:
                logger.info(f"⚙️ Training {ab_name} model from scratch...")
                # Setup Training pipeline natively
                train_ixi = IXIActiveGenerator(train_loader)
                train_brats_gen = BraTSActiveGenerator(
                    train_loader, mode='clean'
                )
                train_ds = get_training_dataset(train_ixi, train_brats_gen, cfg)

                val_gen = SequentialValidationGenerator(
                    val_loader, max_slices_per_vol=15
                )
                val_ds = create_tf_dataset(val_gen, cfg, is_training=False)

                # Heuristic steps based on dataset size (Scale down for Interactive)
                train_steps = (
                    max(10, (len(train_files) * 130) // cfg.train.batch_size)
                    if is_interactive
                    else max(
                        50, (len(train_files) * 130) // cfg.train.batch_size
                    )
                )

                trainer = Trainer(
                    cfg,
                    model,
                    train_ds,
                    val_ds,
                    train_loader,
                    train_steps=train_steps,
                    val_steps=None,
                )
                trainer.train(compile_model=True)

                logger.info(f"💾 Saving {ab_name} model weights...")
                model.save(
                    local_weights_path
                )  # save full model to local writable dir

            # --- FIX: Kill the loader's background thread before inference ---
            train_loader.stop()

            # --- INFERENCE & ARTIFACT GENERATION (Phase 3) ---
            logger.info(f"🔍 Running Autoregressive Inference for {ab_name}...")
            reconstructor = VolumeReconstructor(model, cfg)
            reconstructor.cfg.batch_mode = (
                True  # <--- FIX: silences the internal TQDM spam!
            )

            # Helper to run eval on a specific pool
            def evaluate_pool(pool, dataset_name):
                profiler = InferenceProfiler(ab_name, results_dir)

                for vol_obj in tqdm(
                    pool, desc=f"Evaluating {dataset_name} ({ab_name})"
                ):
                    vol = vol_obj.t1
                    vol_id = (
                        os.path.basename(vol_obj.path)
                        .replace('.nii.gz', '')
                        .replace('.nii', '')
                    )

                    # Target a rollout near the center of the brain
                    center = vol.shape[0] // 2
                    start, end = center - rollout_span, center + rollout_span

                    if end - start < (rollout_span * 2) or start < 0:
                        continue

                    # 1. Forward Pass (No mask -> Pure Autoregression)
                    profiler.start()
                    recon_vol = reconstructor.autoregressive_restore(
                        vol, start, end, 'forward', mask_volume=None
                    )
                    profiler.stop_and_log(vol_id)

                    # 2. Extract Per-Step Metrics
                    metrics = calculate_step_metrics(
                        vol, recon_vol, start, end, 'forward', ab_name, vol_id
                    )
                    all_metrics.extend(metrics)

                    # 3. Save Raw NPY Volume (For Phase 4 Image Matrices)
                    np.save(
                        os.path.join(results_dir, f"{ab_name}_{vol_id}.npy"),
                        recon_vol,
                    )

                    # Save Ground Truth (Only needs to be done once per volume, but overwriting is safe/fast)
                    np.save(os.path.join(results_dir, f"gt_{vol_id}.npy"), vol)

            # Evaluate on IXI and BraTS pools loaded in the static val_loader
            # --- EVALUATION ROUTING (Respecting Dataset Weights) ---
            if cfg.data.ixi_sampling_weight > 1e-8:
                evaluate_pool(
                    [v for v in val_loader.pool['ixi'] if v.path in test_ixi],
                    "IXI",
                )

            if cfg.data.ixi_sampling_weight < 1.0 - 1e-8:
                evaluate_pool(
                    [
                        v
                        for v in val_loader.pool['brats']
                        if v.path in [b['t1'] for b in test_brats]
                    ],
                    "BraTS",
                )

            # --- CRITICAL LEAK FIX: Wipe the RAM cache for the next ablation ---
            train_loader.pool.clear()
            train_loader.keys.clear()
            train_loader.files.clear()
            # -------------------------------------------------------

        # --- CRITICAL OOM FIX: Deep Memory Cleanup ---
        # 1. Delete local references to destroy the datasets and iterators
        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        if 'train_ds' in locals():
            del train_ds
        if 'val_ds' in locals():
            del val_ds
        if 'reconstructor' in locals():
            del reconstructor

        # 2. Destroy the Perceptual Loss Singleton so it doesn't bleed into the next model
        _losses._GLOBAL_PERCEPTUAL_LOSS = None

        # 3. Force garbage collection and purge Keras backend
        gc.collect()
        tf.keras.backend.clear_session()
        gc.collect()
        # ---------------------------------------------

        # Clean up TF Graph before next ablation to prevent OOM
        tf.keras.backend.clear_session()

    # Save all metrics to a single CSV for Phase 4 plotting
    df = pd.DataFrame(all_metrics)
    # csv_path = f"{ABLATION_DIR}/ablation_metrics.csv"
    csv_path = os.path.join(results_dir, "ablation_metrics.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Ablation study complete! Metrics saved to {csv_path}")


def run_ablation_study(CFG):
    # Setup files (reusing existing logic from run_pipeline)
    t1_files = find_t1_files(CFG.data.data_root_ixi)
    brats_list = get_brats_subjects(CFG.data.data_root_brats)

    random.seed(42)
    random.shuffle(t1_files)
    ixi_split = int(0.9 * len(t1_files))
    t_files, v_files = t1_files[:ixi_split], t1_files[ixi_split:]

    random.shuffle(brats_list)
    brats_split = int(0.9 * len(brats_list))
    t_brats, v_brats = brats_list[:brats_split], brats_list[brats_split:]

    # Run the study
    run_ablation_study_(CFG, t_files, v_files, t_brats, v_brats)


def main(mode='ablation', weights_dir=None, preload_dir=None):
    """
    Initializes configuration and launches a specific pipeline based on the `mode` argument.
    Args:
        mode (str): Pipeline execution mode. Defaults to 'ablation'.
            Supported modes:
            - 'all': Runs the full master pipeline (Ablation -> SOTA -> Frechet -> DSC -> CVPR).
            - 'ablation': Runs the CVPR ablation study (train/inference for 5 models).
            - 'sota': Evaluates our model against SOTA (MONAI 3D LDM, VQ-GAN).
            - 'dsc': Calculates anatomical Dice metrics using FastSurfer.
            - 'frechet': Calculates 3D-FID and FVD metrics.
            - 'cvpr': Preloads previous results and generates final CVPR paper plots.
            - 'train': Standard training (routed to run_pipeline).
            - 'hpo': Hyperparameter optimization (routed to run_pipeline).
            - 'eda': Exploratory Data Analysis dashboards (routed to run_pipeline).
            - 'viz': Visualization mode using pretrained model (routed to run_pipeline).
        weights_dir (str, optional): Path to pretrained weights. Defaults to None (falls back to 'ablations').
        preload_dir (str, optional): Path to previous run results. Defaults to None (falls back to 'ablations-latest').
    """

    if weights_dir:
        CFG.data.weights_dir = weights_dir
    elif not CFG.data.weights_dir:
        CFG.data.weights_dir = 'ablations'

    if preload_dir:
        CFG.data.preload_dir = preload_dir
    elif not CFG.data.preload_dir:
        CFG.data.preload_dir = 'ablations-latest'

    try:
        # Lazy heavy imports (the notebook's own import cells cover them; Python caches the import)
        import cvpr_plots
        from cvpr_plots import (
            run_cvpr_rendering,
            generate_figure_4_anatomical_dsc,
            generate_figure_5_frechet_distances,
        )
        from brec.evaluation.monai_sota import MonaiSotaEvaluator
        from brec.evaluation.frechet import FrechetEvaluator
        from brec.evaluation.fastsurfer import FastSurferEvaluator
        from brec.evaluation.synthseg import SynthSegEvaluator

        if mode == 'all':
            logger.info(
                ">>> 🚀 INITIATING FULL MASTER EVALUATION PIPELINE 🚀 <<<"
            )

            # --- 0. PRELOAD PREVIOUS RESULTS (RESUMABILITY) ---
            if os.path.exists(CFG.data.preload_dir):
                logger.info(
                    f"📥 Preloading existing results from {CFG.data.preload_dir}..."
                )
                os.makedirs(CFG.data.results_dir, exist_ok=True)
                import shutil

                # Direct 1:1 copy from the read-only mount to our writable results directory
                shutil.copytree(
                    CFG.data.preload_dir,
                    CFG.data.results_dir,
                    dirs_exist_ok=True,
                )
                logger.info(
                    "✅ Preload complete. Existing volumes will be skipped!"
                )

            # --- 1. ABLATION STUDY (Our Models) ---
            logger.info(
                "\n" + "=" * 50 + "\n PHASE 1: ABLATION STUDY \n" + "=" * 50
            )
            run_ablation_study(CFG)

            # --- 2. SOTA EVALUATION (MONAI 3D LDM) ---
            logger.info(
                "\n"
                + "=" * 50
                + "\n PHASE 2: MONAI SOTA INFERENCE \n"
                + "=" * 50
            )
            t1_files = find_t1_files(CFG.data.data_root_ixi)
            brats_list = get_brats_subjects(CFG.data.data_root_brats)

            random.seed(42)
            random.shuffle(t1_files)
            ixi_split = int(0.9 * len(t1_files))
            _, v_files = t1_files[:ixi_split], t1_files[ixi_split:]

            brats_pool = []
            for b in brats_list[:20]:
                v = VolumeLoader.load(b['t1'], b['seg'])
                if v:
                    brats_pool.append(v)

            sota_eval = MonaiSotaEvaluator(CFG)
            sota_eval.setup()
            num_sota_vols = 5 if CFG.run_type == 'Interactive' else 20
            # sota_eval.evaluate_masked_inpainting(v_files, brats_pool, num_volumes=num_sota_vols)
            sota_eval.evaluate_sota_models(
                v_files, brats_pool, num_volumes=num_sota_vols
            )

            # --- 3. FRÉCHET METRICS (3D-FID & FVD) ---
            logger.info(
                "\n" + "=" * 50 + "\n PHASE 3: FRÉCHET METRICS \n" + "=" * 50
            )
            frechet_eval = FrechetEvaluator(CFG)
            frechet_eval.evaluate()

            # --- 4. ANATOMICAL VALIDATION (FastSurfer DSC) ---
            logger.info(
                "\n"
                + "=" * 50
                + "\n PHASE 4: ANATOMICAL VALIDATION \n"
                + "=" * 50
            )
            dsc_eval = FastSurferEvaluator(CFG)
            dsc_eval.setup()
            dsc_eval.convert_to_nifti()
            dsc_eval.run_prediction()
            dsc_eval.calculate_dsc()
            # --- 5. CVPR PLOTTING ---
            logger.info(
                "\n" + "=" * 50 + "\n PHASE 5: CVPR PLOTTING \n" + "=" * 50
            )
            run_cvpr_rendering(render_supp=False)

            logger.info(
                ">>> 🎉 MASTER PIPELINE COMPLETE! CHECK 'paper_figures' DIRECTORY 🎉 <<<"
            )

        elif mode == 'ablation':
            # Run the study
            run_ablation_study(CFG)
            run_cvpr_rendering(render_supp=True)
        elif mode == 'dsc':
            # evaluator = SynthSegEvaluator(CFG)
            evaluator = FastSurferEvaluator(CFG)
            evaluator.setup()

            # Check if we have data to process
            npy_files = glob.glob(os.path.join(CFG.data.results_dir, "*.npy"))
            if not npy_files:
                logger.warning(
                    f"No .npy files found in '{CFG.data.results_dir}'. Running 'ablation' mode..."
                )
                run_ablation_study(CFG)

            evaluator.convert_to_nifti()
            evaluator.run_prediction()
            evaluator.calculate_dsc()
            # Generate the specific plot
            generate_figure_4_anatomical_dsc()
        elif mode == 'frechet':
            # Run 3D-FID and FVD evaluation
            evaluator = FrechetEvaluator(CFG)

            # Check if we have data to process
            npy_files = glob.glob(os.path.join(CFG.data.results_dir, "*.npy"))
            if not npy_files:
                logger.warning(
                    f"No .npy files found in '{CFG.data.results_dir}'. Running 'ablation' mode..."
                )
                run_ablation_study(CFG)

            evaluator.evaluate()
            generate_figure_5_frechet_distances()
        elif mode == 'sota':
            # 1. Setup Data
            t1_files = find_t1_files(CFG.data.data_root_ixi)
            brats_list = get_brats_subjects(CFG.data.data_root_brats)

            random.seed(42)
            random.shuffle(t1_files)
            ixi_split = int(0.9 * len(t1_files))
            _, v_files = t1_files[:ixi_split], t1_files[ixi_split:]

            # Load BraTS pool for realistic masks
            brats_pool = []
            for b in brats_list[:20]:
                v = VolumeLoader.load(b['t1'], b['seg'])
                if v:
                    brats_pool.append(v)

            # 2. Run MONAI SOTA
            sota_eval = MonaiSotaEvaluator(CFG)
            sota_eval.setup()
            sota_eval.evaluate_masked_inpainting(
                v_files,
                brats_pool,
                num_volumes=5 if CFG.run_type == 'Interactive' else 20,
            )

        elif mode == 'cvpr':
            # --- 0. PRELOAD PREVIOUS RESULTS (RESUMABILITY) ---
            if os.path.exists(CFG.data.preload_dir):
                logger.info(
                    f"📥 Preloading existing results from {CFG.data.preload_dir}..."
                )
                os.makedirs(CFG.data.results_dir, exist_ok=True)
                import shutil

                # Direct 1:1 copy from the read-only mount to our writable results directory
                shutil.copytree(
                    CFG.data.preload_dir,
                    CFG.data.results_dir,
                    dirs_exist_ok=True,
                )
                logger.info(
                    "✅ Preload complete. Existing volumes will be skipped!"
                )

            run_cvpr_rendering(render_supp=True)
        else:
            run_pipeline(mode=mode)

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    import argparse

    main()
