import os
import copy
import gc
import uuid
import random
import shutil
import tracemalloc

import numpy as np
import optuna
import tensorflow as tf

from brec.core.utils import logger, telemetry
from configs.config import Config
from brec.data.cache import StaticLoader
from brec.data.generators import (IXIActiveGenerator, BraTSActiveGenerator,
                                 SequentialValidationGenerator, HpoTrainSequence,
                                 HpoValidSequence)
from brec.models.builder import ModelBuilder
from brec.models.losses import CompositeLoss, get_perceptual_loss
from brec.training.trainer import Trainer
from brec.evaluation.metrics import (OracleMAE, OracleMSE, OracleSSIM, OraclePSNR,
                                    GradientSharpnessMetric)
from brec.inference.reconstructor import VolumeReconstructor


def compute_autoregressive_score(model, config, val_loader, ixi_files,
                                 brats_files, n_steps=5):
    """
    Calculates the mean MAE over a multi-step autoregressive rollout.
    This is the 'True' metric we want to optimize.
    """
    reconstructor = VolumeReconstructor(model, config)

    # 1. Grab pre-loaded MedicalVolume objects directly from RAM
    test_volumes = []

    for path in ixi_files:
        # Match the path string to the loaded object in the list
        vol = next((v for v in val_loader.pool['ixi'] if v.path == path), None)
        if vol: test_volumes.append(vol)

    for d in brats_files:
        vol = next((v for v in val_loader.pool['brats'] if v.path == d['t1']),
                   None)
        if vol: test_volumes.append(vol)

    total_mae = 0.0
    total_slices = 0

    for vol_obj in test_volumes:
        # 2. Bypass disk entirely
        vol = vol_obj.t1

        # Determine Range (Middle of brain)
        # We need n_steps. Let's start slightly before center
        center = vol.shape[0] // 2
        start = center - (n_steps // 2)
        end = start + n_steps

        # Bounds check
        if start < 0 or end > vol.shape[0]: continue

        # Run Pure Autoregression (Hardest task)
        # We don't use Teacher Forcing (mask) here because we want to measure stability
        recon = reconstructor.autoregressive_restore(vol, start, end, 'forward',
                                                     mask_volume=None)

        # Calculate MAE over the generated sequence
        # GT vs Prediction
        gt_seq = vol[start:end]
        pred_seq = recon[start:end]

        mae = np.mean(np.abs(gt_seq - pred_seq))

        total_mae += mae
        total_slices += 1

    if total_slices == 0: return float('inf')

    return total_mae / total_slices


class HPMEngine:
    """
    Ultra-Stable HPO Engine.
    Uses Static RAM allocation and a Single-Dataset paradigm.
    Zero disk I/O, Zero background threads, Zero memory fragmentation.
    """
    def __init__(self, config: Config, train_files: list, brats_files: list):
        self.cfg = config
        self.tracemem = self.cfg.hpo.tracemalloc

        # FIXED REPRESENTATIVE SUBSETS (e.g., 150 IXI, 150 BraTS)
        # Random seed ensures consistent subset across runs
        random.seed(42)

        # Shuffle before slicing
        random.shuffle(train_files)
        random.shuffle(brats_files)

        # 1. Calculate dynamic proportional splits based on global generator weights
        w_ixi = self.cfg.data.ixi_sampling_weight
        w_ixi = max(0.1, min(0.9, w_ixi)) # safety clip
        w_brats = 1.0 - w_ixi

        # Number of volumes per subset
        n_train_ixi = int(self.cfg.hpo.base_train_volumes * w_ixi)
        n_train_brats = int(self.cfg.hpo.base_train_volumes * w_brats)

        n_valid_ixi = int(self.cfg.hpo.base_val_volumes * w_ixi)
        n_valid_brats = int(self.cfg.hpo.base_val_volumes * w_brats)

        n_eval_ixi = int(self.cfg.hpo.base_eval_volumes * w_ixi)
        n_eval_brats = int(self.cfg.hpo.base_eval_volumes * w_brats)

        # 2. Slice datasets safely
        self.train_ixi = train_files[:n_train_ixi]
        self.train_brats = brats_files[:n_train_brats]

        self.val_ixi = train_files[n_train_ixi : n_train_ixi + n_valid_ixi]
        self.val_brats = brats_files[n_train_brats : n_train_brats + n_valid_brats]

        # Evaluation subset for the AR Metric (sliced from Val pool so it's unseen in training)
        self.eval_ixi = self.val_ixi[:n_eval_ixi]
        self.eval_brats = self.val_brats[:n_eval_brats]

        logger.info(
            f"HPO Volumes Loaded -> Train: {len(self.train_ixi)} IXI / {len(self.train_brats)} BraTS"
            f" | Valid: {len(self.val_ixi)} IXI / {len(self.val_brats)} BraTS"
        )

    def _objective(self, trial):
        # -1. Start the memory profiler at the very beginning of the trial
        if self.tracemem:
            tracemalloc.start()
        # TODO: refactor global telemetry instance
        telemetry.log("TRIALSTART", trial.number)

        logger.info(f">>> Starting trial #{trial.number:03d}...")
        # 0. Clean Slate Keras Graph
        # tf.keras.backend.clear_session()
        # gc.collect()

        # A. RESTORE PRISTINE MODEL WEIGHTS
        self.model.set_weights(self.initial_weights)

        # B. ZERO OUT OPTIMIZER MOMENTUM (Prevents contamination between trials)
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            # for var in self.model.optimizer.variables():
            #     var.assign(tf.zeros_like(var))
            # FIX: In Keras 3, optimizer.variables is a list property, not a method!
            opt_vars = self.model.optimizer.variables
            if callable(opt_vars):
                # Fallback for Keras 2 just in case
                opt_vars = opt_vars()

            for var in opt_vars:
                var.assign(tf.zeros_like(var))

        # C. INJECT NEW HYPERPARAMETERS DYNAMICALLY (No recompilation!)
        if not self.cfg.train.gan_mode:
            lr_inline = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
            # K.set_value(self.model.optimizer.learning_rate, lr_inline)
            # --- FIX: Keras 3 robust learning rate injection ---
            if hasattr(self.model.optimizer.learning_rate, 'assign'):
                # It is a tf.Variable (Standard Keras 3 behavior)
                self.model.optimizer.learning_rate.assign(lr_inline)
            else:
                # It is a raw property (Fallback)
                self.model.optimizer.learning_rate = lr_inline
        else:
            # (GAN logic remains the same, updating both optimizers)
            lr_g_inline = trial.suggest_float('lr_g', 5e-5, 5e-4, log=True)
            lr_d_inline = trial.suggest_float('lr_d', 5e-5, 5e-4, log=True)

            if hasattr(self.trainer.model.g_optimizer.learning_rate, 'assign'):
                self.trainer.model.g_optimizer.learning_rate.assign(lr_g_inline)
                self.trainer.model.d_optimizer.learning_rate.assign(lr_d_inline)
            else:
                self.trainer.model.g_optimizer.learning_rate = lr_g_inline
                self.trainer.model.d_optimizer.learning_rate = lr_d_inline

        # 1. Update self.cfg IN-PLACE
        if not self.cfg.train.gan_mode:
            # self.cfg.train.learning_rate = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
            ...  # skip learning rate for deterministic approach
            if not self.cfg.train.use_spatial_loss:
                #     --- FIX: Update the actual TF Variables on the compiled loss function ---
                self.loss_fn.w_tumor.assign(trial.suggest_float('l_tumor', 0.0, 1.0))
                self.loss_fn.w_healthy.assign(trial.suggest_float('l_healthy', 0.0, 1.0))
                self.loss_fn.w_bg.assign(trial.suggest_float('l_background', 0.0, 1.0))
                self.loss_fn.w_grad.assign(trial.suggest_float('l_grad', 0.0, 1.0))
                self.loss_fn.w_perc.assign(trial.suggest_float('l_perceptual', 0.0, 1.0))
                self.loss_fn.w_spec.assign(trial.suggest_float('l_spectral', 0.0, 1.0))
        else:
            # self.cfg.train.learning_rate_g = trial.suggest_float('lr_g', 5e-5, 5e-4, log=True)
            # self.cfg.train.learning_rate_d = trial.suggest_float('lr_d', 5e-5, 5e-4, log=True)
            # --- FIX: Inject GAN Loss Weights dynamically ---
            # In GAN mode, self.trainer.model is the SPADEGANTrainer
            self.trainer.model.w_l1.assign(trial.suggest_float('w_l1', 0.0, 1.0))
            self.trainer.model.w_perc.assign(trial.suggest_float('w_perc', 0.0, 1.0))
            self.trainer.model.w_spec.assign(trial.suggest_float('w_spec', 0.0, 1.0))
            self.trainer.model.w_gan.assign(trial.suggest_float('w_gan', 0.0, 1.0))
            self.trainer.model.w_fm.assign(trial.suggest_float('w_fm', 0.0, 1.0))

            if self.cfg.model.architecture == 'vae':
                self.trainer.model.w_kl.assign(trial.suggest_float('w_kl', 0.0, 0.1))

        if not self.cfg.train.gan_mode and not self.cfg.train.use_spatial_loss:
            ...
        #     self.cfg.train.lambda_tumor = trial.suggest_float('l_tumor', 0.0, 1.0)
        #     self.cfg.train.lambda_healthy = trial.suggest_float('l_healthy', 0.0, 1.0)
        #     self.cfg.train.lambda_background = trial.suggest_float('l_background', 0.0, 1.0)
        #     self.cfg.train.lambda_grad = trial.suggest_float('l_grad', 0.0, 1.0)
        #     self.cfg.train.lambda_perceptual = trial.suggest_float('l_perceptual', 0.0, 1.0)
        #     self.cfg.train.lambda_spectral = trial.suggest_float('l_spectral', 0.0, 1.0)
        elif not self.cfg.train.use_spatial_loss:
            ...
        #     self.cfg.train.weight_l1 = trial.suggest_float('w_l1', 0.0, 1.0)
        #     self.cfg.train.weight_perceptual = trial.suggest_float('w_perc', 0.0, 1.0)
        #     self.cfg.train.weight_spectral = trial.suggest_float('w_spec', 0.0, 1.0)
        #     self.cfg.train.weight_gan = trial.suggest_float('w_gan', 0.0, 1.0)
        #     self.cfg.train.weight_fm = trial.suggest_float('w_fm', 0.0, 1.0)

        # Short training for HPO
        self.cfg.train.epochs = 5

        # 2. Create FINITE Training Dataset
        # FIX: The `.take()` forces the infinite generator to stop, destroying the C++ zombie threads
        # total_train_batches = self.cfg.hpo.train_steps_per_trial * self.cfg.train.epochs
        # finite_train_ds = self.train_ds.take(total_train_batches)

        # 2. Create Pure Python Training Dataset INSIDE the objective
        # This completely bypasses the C++ tf.data memory leak
        train_ds = HpoTrainSequence(
            self.train_gen_ixi,
            self.train_gen_brats,
            self.cfg,
            self.cfg.hpo.train_steps_per_trial
        )

        combined_score = float('inf')
        ar_score = float('inf')
        os_score = float('inf')

        try:
            # 3. Build & Train (Uses globally defined datasets)
            # model = ModelBuilder.build(self.cfg)

            # trainer = Trainer(self.cfg, model, finite_train_ds, self.val_ds, self.train_loader,
            #                   train_steps=self.cfg.hpo.train_steps_per_trial,
            #                   val_steps=None)
            # trainer = Trainer(self.cfg, model, train_ds, self.val_ds_sequence, self.train_loader,
            #                   train_steps=self.cfg.hpo.train_steps_per_trial,
            #                   val_steps=None)
            # --- FIX: Do NOT build the model here. Do NOT recreate the Trainer! ---
            # Just inject the new sequence into the global trainer.
            self.trainer.train_ds = train_ds
            self.trainer.train_steps = self.cfg.hpo.train_steps_per_trial

            # history = self.trainer.train(generator_ref=None)
            # Train the global model WITHOUT recompiling
            history = self.trainer.train(compile_model=False)

            # 4. Extract 1-Shot Validation Metric
            os_score = float('inf')
            if history and 'val_loss' in history.history:
                os_score = history.history['val_loss'][-1]
            elif history and 'val_l1_loss' in history.history: # fallback for GAN mode
                os_score = history.history['val_l1_loss'][-1]

            # 5. Score using RAM Pool (Zero Disk I/O)
            if np.isnan(history.history['loss'][-1]):
                ar_score = float('inf')
            else:
                # Calculate AR Score reading directly from val_loader.pool
                # --- FIX: Evaluate the global self.model! ---
                ar_score = compute_autoregressive_score(
                    self.model, self.cfg, self.val_loader, self.eval_ixi,
                    self.eval_brats, n_steps=self.cfg.hpo.ar_rollout_steps
                )

            # 6. Blend the Metrics
            w_ar = self.cfg.hpo.metric_ar_weight
            w_os = 1.0 - w_ar

            # Avoid nan propagation in multiplication
            if np.isnan(ar_score) or np.isnan(os_score):
                combined_score = float('inf')
            else:
                combined_score = (ar_score * w_ar) + (os_score * w_os)

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            # ar_score = float('inf')
            combined_score = float('inf')

        finally:
            # Delete local references so Graph can be destroyed
            # del model
            # del trainer
            del train_ds
            if 'history' in locals(): del history
            tf.keras.backend.clear_session()
            gc.collect()

            # 2. Take a snapshot of RAM exactly before the trial ends
            if self.tracemem:
                snapshot = tracemalloc.take_snapshot()

                # 3. Print the top 10 memory-hogging lines of code
                top_stats = snapshot.statistics('lineno')
                print(f"\n--- [PROFILER] TOP 10 RAM ALLOCATIONS (TRIAL {trial.number}) ---")
                for stat in top_stats[:10]:
                    print(stat)

                tracemalloc.stop()

            telemetry.log("TRIALSTOP", trial.number)
            # logger.info(f"Trial #{trial.number:03d} finished with Score: {ar_score:.4f}")
            logger.info(
                f"Trial #{trial.number:03d} finished with Combined Score: {combined_score:.4f} (AR: {ar_score:.4f}, OS: {os_score:.4f})"
            )

        # return ar_score
        return combined_score

    def run(self):
        os.makedirs(self.cfg.hpo.storage_dir, exist_ok=True)

        # Define paths
        unique_id = str(uuid.uuid4())[:8]
        # db_name = f"{self.cfg.hpo.storage_dir}/hpo_{unique_id}.db"
        db_name = os.path.join(self.cfg.hpo.storage_dir, f"hpo_{unique_id}.db")
        # final_db_name = os.path.join(self.cfg.hpo.storage_dir, self.cfg.hpo.database_name)
        final_db_name = self.cfg.hpo.database_name
        # --- CRITICAL FIX: WARM START OPTUNA ---
        # If the merged database exists from previous runs, copy it to our worker's unique DB
        if os.path.isfile(self.cfg.hpo.database_preload) and not os.path.exists(final_db_name):
            logger.info(f"Found preload database: {self.cfg.hpo.database_preload}")
            shutil.copy(self.cfg.hpo.database_preload, final_db_name)
        # Optuna will load the history and continue the TPE search intelligently
        if os.path.exists(final_db_name):
            shutil.copy(final_db_name, db_name)
            logger.info(f"Loaded previous Optuna history from sqlite:///{final_db_name}")
        else:
            logger.info("No previous history found. Starting fresh.")

        storage_url = f"sqlite:///{db_name}"

        logger.info(f"Starting Static-RAM HPO Node. Storage: {storage_url}")

        # 1. LOAD ALL DATA INTO RAM ONCE
        logger.info(f"Allocating {len(self.train_ixi)} IXI and {len(self.train_brats)} BraTS volumes to RAM...")

        self.val_loader = StaticLoader(self.cfg, self.val_ixi, self.val_brats)
        self.train_loader = StaticLoader(self.cfg, self.train_ixi, self.train_brats)

        # Disable Hallucination updating during HPO since StaticLoader doesn't have a background thread
        self.cfg.aug.prob_hallucination_replay = 0.0
        # Disable hullucination buffer at all, since it's impact insignificant during HPO
        self.cfg.aug.prob_hallucination_max = 0.0
        # self.cfg.aug.prob_hallucination_replay = 0.0

        # 2. CREATE TF.DATA DATASETS EXACTLY ONCE
        # logger.info("Initializing global tf.data pipelines...")
        logger.info("Initializing python generators...")

        # These just yield numpy arrays from RAM. No TF logic
        self.train_gen_ixi = IXIActiveGenerator(self.train_loader)
        self.train_gen_brats = BraTSActiveGenerator(self.train_loader, mode='clean')

        # Convert all validation data into a static Sequence once.
        logger.info("Materializing Validation Sequence...")
        val_gen = SequentialValidationGenerator(self.val_loader, max_slices_per_vol=10)
        self.val_ds_sequence = HpoValidSequence(val_gen, self.cfg)

        # 3. PRE-WARM SINGLETON LOSS
        # Forces EfficientNet to download/build once in the global context
        # logger.info("Pre-warming Global Perceptual Loss Singleton...")
        # import src.models.losses as losses_module
        # losses_module.get_perceptual_loss(self.cfg)
        # get_perceptual_loss(self.cfg)
        # PRE-WARM SINGLETON LOSS (only if needed)
        if self.cfg.train.lambda_perceptual >= 0.01 or self.cfg.train.weight_perceptual >= 0.01:
            logger.info("Pre-warming Global Perceptual Loss Singleton...")
            get_perceptual_loss(self.cfg)
        else:
            logger.info("Perceptual Loss weight < 0.01. Bypassing EfficientNet initialization.")

        # --- THE FIX: COMPILE THE C++ GRAPH EXACTLY ONCE ---
        logger.info("Building and Compiling Model Graph...")
        self.model = ModelBuilder.build(self.cfg)

        # Save pristine weights
        self.initial_weights = self.model.get_weights()

        # Instantiate Loss and Trainer once globally
        self.loss_fn = CompositeLoss(self.cfg)

        # We manually compile the model here so the Trainer doesn't recreate the graph
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.train.learning_rate)
        metrics = [
            # 'mae', 'mse', SSIMMetric(), PSNRMetric(), GradientSharpnessMetric()
            OracleMAE(), # OracleMAEMetric(),
            OracleMSE(), # OracleMSEMetric(),
            OracleSSIM(), # SSIMMetric(),
            OraclePSNR(), # PSNRMetric(),
            GradientSharpnessMetric()
        ]
        self.model.compile(optimizer=optimizer, loss=self.loss_fn, metrics=metrics)

        # --- FIX: Pass None for train_ds during global initialization ---
        self.trainer = Trainer(self.cfg, self.model, None, self.val_ds_sequence,
                               self.train_loader,
                               train_steps=self.cfg.hpo.train_steps_per_trial,
                               val_steps=None)

        # 4. RUN OPTUNA
        study = optuna.create_study(
            study_name=self.cfg.hpo.study_name,
            storage=storage_url,
            direction='minimize',
            load_if_exists=True
        )

        study.optimize(
            self._objective,
            n_trials=self.cfg.hpo.n_trials
        )

        logger.info(f"Best Params: {study.best_params}")
        self._save_best_config(study.best_params)

    def _save_best_config(self, best_params):
        # if 'lr' in best_params: self.cfg.train.learning_rate = best_params['lr']
        if 'l_tumor' in best_params: self.cfg.train.lambda_tumor = best_params['l_tumor']
        if 'l_healthy' in best_params: self.cfg.train.lambda_healthy = best_params['l_healthy']
        if 'l_background' in best_params: self.cfg.train.lambda_background = best_params['l_background']
        if 'l_grad' in best_params: self.cfg.train.lambda_grad = best_params['l_grad']
        if 'l_perceptual' in best_params: self.cfg.train.lambda_perceptual = best_params['l_perceptual']
        if 'l_spectral' in best_params: self.cfg.train.lambda_spectral = best_params['l_spectral']
        if 'lr_g' in best_params: self.cfg.train.learning_rate_g = best_params['lr_g']
        if 'lr_d' in best_params: self.cfg.train.learning_rate_d = best_params['lr_d']
        if 'w_l1' in best_params: self.cfg.train.weight_l1 = best_params['w_l1']
        if 'w_perc' in best_params: self.cfg.train.weight_perceptual = best_params['w_perc']
        if 'w_spec' in best_params: self.cfg.train.weight_spectral = best_params['w_spec']
        if 'w_gan' in best_params: self.cfg.train.weight_gan = best_params['w_gan']
        if 'w_fm' in best_params: self.cfg.train.weight_fm = best_params['w_fm']

        # FIX: Restore full RAM capacity for standard training before saving!
        # TODO: de-hardcode
        self.cfg.data.ixi_cache_size = 120
        self.cfg.data.brats_cache_size = 120

        path = self.cfg.hpo.best_params_file
        self.cfg.save(path)
        logger.info(f"Saved optimized config to {path}")
