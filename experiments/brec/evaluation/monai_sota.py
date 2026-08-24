import os
import glob
import gc
import subprocess

import numpy as np

import tensorflow as tf

from brec.core.utils import logger, InferenceProfiler
from brec.data.cache import VolumeLoader
from brec.evaluation.visualizer import VisualizationSuite
from brec.models.builder import ModelBuilder
from brec.inference.reconstructor import VolumeReconstructor
from configs.config import Config


class MonaiSotaEvaluator:
    """
    Evaluates MONAI Generative SOTA models and Our Full Model on Masked Inpainting.
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.results_dir = config.data.results_dir
        self.bundle_dir = os.path.join(self.results_dir, "monai_bundles")
        os.makedirs(self.bundle_dir, exist_ok=True)
        
        self.device = None
        self.autoencoder_3d = None
        self.unet_3d = None
        self.scheduler_3d = None
        
        self.unet_2d = None
        self.scheduler_2d = None

    def setup(self):
        logger.info("Setting up MONAI Generative ecosystem...")
        try:
            import monai
            import generative
        except ImportError:
            logger.info("Force-installing MONAI without touching Kaggle's CUDA/Torch deps...")
            subprocess.run([
                "pip", "install", "--no-deps", "monai", "monai-generative", "einops"
            ], check=True)
            
        import torch
        from monai.bundle import download
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Download 3D LDM Bundle
        self.bundle_3d = "brain_image_synthesis_latent_diffusion_model"
        if not os.path.exists(os.path.join(self.bundle_dir, self.bundle_3d)):
            logger.info("Downloading MONAI 3D LDM Bundle...")
            download(name=self.bundle_3d, bundle_dir=self.bundle_dir)
            
        # 2. Download 2D Diffusion Bundle (BraTS Axial)
        self.bundle_2d = "brats_mri_axial_slices_generative_diffusion"
        if not os.path.exists(os.path.join(self.bundle_dir, self.bundle_2d)):
            logger.info("Downloading MONAI 2D Diffusion Bundle...")
            download(name=self.bundle_2d, bundle_dir=self.bundle_dir)
            
        logger.info("MONAI SOTA bundles ready.")

    def _load_3d_models(self):
        if self.autoencoder_3d is not None: return
        import torch
        from monai.bundle import ConfigParser
        from generative.networks.schedulers import DDIMScheduler
        
        logger.info("Loading 3D LDM weights into VRAM...")
        bundle_path = os.path.join(self.bundle_dir, self.bundle_3d)
        parser = ConfigParser()
        parser.read_config(os.path.join(bundle_path, "configs", "inference.json"))
        
        try:
            self.autoencoder_3d = parser.get_parsed_content("autoencoder_def")
            self.unet_3d = parser.get_parsed_content("diffusion_def")
        except:
            self.autoencoder_3d = parser.get_parsed_content("autoencoder")
            self.unet_3d = parser.get_parsed_content("diffusion")
            
        self.scheduler_3d = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=False)
        
        models_dir = os.path.join(bundle_path, "models")
        ae_sd = torch.load(os.path.join(models_dir, "autoencoder.pt"), map_location=self.device)
        
        unet_weights =[f for f in glob.glob(os.path.join(models_dir, "*.pt")) if "autoencoder" not in f][0]
        unet_sd = torch.load(unet_weights, map_location=self.device)
        
        translated_ae_sd = {k.replace(".conv.conv.", ".postconv.conv.") if "decoder." in k else k: v for k, v in ae_sd.items()}
        translated_unet_sd = {k.replace(".to_out.0.", ".out_proj."): v for k, v in unet_sd.items()}
        
        self.autoencoder_3d.load_state_dict(translated_ae_sd, strict=False)
        self.unet_3d.load_state_dict(translated_unet_sd, strict=False)
        self.autoencoder_3d.eval().to(self.device)
        self.unet_3d.eval().to(self.device)

    def _load_2d_models(self):
        if self.unet_2d is not None: return
        import torch
        from monai.bundle import ConfigParser
        from generative.networks.schedulers import DDIMScheduler
        
        logger.info("Loading 2D Diffusion weights into VRAM...")
        bundle_path = os.path.join(self.bundle_dir, self.bundle_2d)
        parser = ConfigParser()
        parser.read_config(os.path.join(bundle_path, "configs", "inference.json"))
        
        try:
            self.unet_2d = parser.get_parsed_content("network_def")
        except:
            self.unet_2d = parser.get_parsed_content("network")
            
        self.scheduler_2d = DDIMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=False)
        
        models_dir = os.path.join(bundle_path, "models")
        unet_weights = glob.glob(os.path.join(models_dir, "*.pt"))[0]
        self.unet_2d.load_state_dict(torch.load(unet_weights, map_location=self.device), strict=False)
        self.unet_2d.eval().to(self.device)

    def _repaint_3d_latent(self, vol_tensor, mask_tensor, num_inference_steps=50, repaint_jumps=2):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            latent_gt = self.autoencoder_3d.encode_stage_2_inputs(vol_tensor)
            
        latent_mask = F.interpolate(mask_tensor, size=latent_gt.shape[2:], mode='nearest')
        latent_pred = torch.randn_like(latent_gt).to(self.device)
        
        dummy_cond = torch.zeros((1, 4, *latent_gt.shape[2:]), dtype=latent_gt.dtype, device=self.device)
        dummy_context = torch.zeros((1, 1, 4), dtype=latent_gt.dtype, device=self.device)
        
        self.scheduler_3d.set_timesteps(num_inference_steps)
        
        with torch.cuda.amp.autocast():
            for t in self.scheduler_3d.timesteps:
                for jump in range(repaint_jumps):
                    with torch.no_grad():
                        unet_input = torch.cat([latent_pred, dummy_cond], dim=1)
                        step_out = self.unet_3d(x=unet_input, timesteps=torch.tensor([t]).to(self.device), context=dummy_context)
                        
                    noise_pred = step_out[0] if isinstance(step_out, tuple) else step_out
                    step_res = self.scheduler_3d.step(noise_pred, t, latent_pred)
                    latent_pred = step_res[0] if isinstance(step_res, tuple) else step_res
                    
                    noise = torch.randn_like(latent_gt).to(self.device)
                    known_latent_t = self.scheduler_3d.add_noise(latent_gt, noise, torch.tensor([t], dtype=torch.long, device=self.device))
                    latent_pred = known_latent_t * (1.0 - latent_mask) + latent_pred * latent_mask
                    
        with torch.no_grad():
            return self.autoencoder_3d.decode_stage_2_outputs(latent_pred)

    def _repaint_2d_pixel(self, slice_tensor, mask_tensor, num_inference_steps=50):
        import torch
        # 2D DDPM operates in pixel space (no autoencoder)
        img_pred = torch.randn_like(slice_tensor).to(self.device)
        self.scheduler_2d.set_timesteps(num_inference_steps)

        batch_size = slice_tensor.shape[0]

        with torch.cuda.amp.autocast():
            for t in self.scheduler_2d.timesteps:
                # Broadcast timestep to match batch size
                t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    # noise_pred = self.unet_2d(x=img_pred, timesteps=torch.tensor([t]).to(self.device))
                    # FIX: Pass the broadcasted t_tensor to the UNet, not the scalar!
                    noise_pred = self.unet_2d(x=img_pred, timesteps=t_tensor)

                noise_pred = noise_pred[0] if isinstance(noise_pred, tuple) else noise_pred
                step_res = self.scheduler_2d.step(noise_pred, t, img_pred)
                img_pred = step_res[0] if isinstance(step_res, tuple) else step_res
                
                noise = torch.randn_like(slice_tensor).to(self.device)
                # known_img_t = self.scheduler_2d.add_noise(
                #     slice_tensor, noise, torch.tensor([t], dtype=torch.long, device=self.device)
                # )
                known_img_t = self.scheduler_2d.add_noise(
                    slice_tensor, noise, t_tensor
                )
                img_pred = known_img_t * (1.0 - mask_tensor) + img_pred * mask_tensor
                
        return img_pred

    def evaluate_sota_models(self, ixi_files: list, brats_pool: list, num_volumes: int = 10):
        import torch
        import torch.nn.functional as F
        
        test_files = ixi_files[:num_volumes]

        # Pre-generate and save shared masks and GT for all SOTA/Ours comparisons
        logger.info("Pre-generating shared evaluation masks...")
        eval_data =[]
        for f in test_files:
            vol_id = os.path.basename(f).replace('.nii.gz', '').replace('.nii', '')
            vol_obj = VolumeLoader.load(f)
            if not vol_obj: continue

            center = vol_obj.t1.shape[0] // 2
            mask_np = VisualizationSuite._sample_real_brats_mask(
                vol_obj.t1, brats_pool, target_z=center
            )

            # Ensure tumor is actually in the evaluation window
            rollout_span = 5 if self.cfg.run_type == 'Interactive' else 20
            eval_window_mask = mask_np[center - rollout_span : center + rollout_span]

            if np.sum(eval_window_mask) < 10: 
                logger.warning(f"Skipping {vol_id}: Tumor injection failed or missed evaluation window.")
                continue

            # Save GT and Mask so FrechetEvaluator and plots can use them
            np.save(os.path.join(self.results_dir, f"gt_{vol_id}.npy"), vol_obj.t1)
            np.save(os.path.join(self.results_dir, f"mask_{vol_id}.npy"), mask_np)

            eval_data.append((vol_id, vol_obj, mask_np))

        if not eval_data:
            logger.error("No valid volumes could be prepared for SOTA evaluation!")
            return

        # --- 1. OUR FULL MODEL (MASKED) ---
        logger.info("Running Ours (Full) in Masked Mode...")
        model_ours = ModelBuilder.build(self.cfg)
        weights_path = os.path.join(self.cfg.data.weights_dir, "model_full.keras")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(self.cfg.data.results_dir, "model_full.keras")

        # model_ours.load_weights(weights_path)
        try:
            model_ours.load_weights(weights_path)
        except Exception as e:
            logger.error(f"Could not load ours_full weights: {e}")

        reconstructor = VolumeReconstructor(model_ours, self.cfg)
        reconstructor.cfg.batch_mode = True
        prof_ours = InferenceProfiler("ours_full_masked", self.results_dir)

        # for f in tqdm(test_files, desc="Ours Masked"):
        #     vol_id = os.path.basename(f).replace('.nii.gz', '').replace('.nii', '')
        #     out_path = os.path.join(self.results_dir, f"ours_full_masked_{vol_id}.npy")
        #     if os.path.exists(out_path): continue

        #     vol_obj = VolumeLoader.load(f)
        #     mask_np = VisualizationSuite._sample_real_brats_mask(vol_obj.t1, brats_pool)
        #     if np.sum(mask_np) < 10: continue
        for vol_id, vol_obj, mask_np in tqdm(eval_data, desc="Ours Masked"):
            out_path = os.path.join(self.results_dir, f"ours_full_masked_{vol_id}.npy")
            if os.path.exists(out_path): continue
            
            prof_ours.start()
            recon_vol = reconstructor.autoregressive_restore(vol_obj.t1, 0, vol_obj.t1.shape[0], 'forward', mask_volume=mask_np)
            prof_ours.stop_and_log(vol_id)
            np.save(out_path, recon_vol)

        del model_ours
        tf.keras.backend.clear_session()
        gc.collect()

        # --- 2. MONAI 3D LDM & 3D VQ-GAN ---
        self._load_3d_models()
        prof_3d_ldm = InferenceProfiler("monai_3d_ldm", self.results_dir)
        prof_vqgan = InferenceProfiler("monai_vqgan", self.results_dir)
        
        # for f in tqdm(test_files, desc="MONAI 3D Inference"):
        #     vol_id = os.path.basename(f).replace('.nii.gz', '').replace('.nii', '')
        #     out_ldm = os.path.join(self.results_dir, f"monai_3d_ldm_{vol_id}.npy")
        #     out_vq = os.path.join(self.results_dir, f"monai_vqgan_{vol_id}.npy")

        #     vol_obj = VolumeLoader.load(f)
        #     mask_np = VisualizationSuite._sample_real_brats_mask(vol_obj.t1, brats_pool)
        #     if np.sum(mask_np) < 10: continue
        for vol_id, vol_obj, mask_np in tqdm(eval_data, desc="MONAI 3D Inference"):
            out_ldm = os.path.join(self.results_dir, f"monai_3d_ldm_{vol_id}.npy")
            out_vq = os.path.join(self.results_dir, f"monai_vqgan_{vol_id}.npy")

            if os.path.exists(out_ldm) and os.path.exists(out_vq): continue

            vol_np = (vol_obj.t1 * 2.0) - 1.0 
            vol_t = torch.tensor(vol_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            mask_t = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

            pad_z, pad_y, pad_x = max(0, 128 - vol_t.shape[2]), max(0, 160 - vol_t.shape[3]), max(0, 160 - vol_t.shape[4])
            vol_t_pad = F.pad(vol_t, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=-1.0)
            mask_t_pad = F.pad(mask_t, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=0.0)

            # VQ-GAN Inference (Upper Bound)
            if not os.path.exists(out_vq):
                prof_vqgan.start()
                with torch.no_grad():
                    latent = self.autoencoder_3d.encode_stage_2_inputs(vol_t_pad)
                    pred_vq_pad = self.autoencoder_3d.decode_stage_2_outputs(latent)
                prof_vqgan.stop_and_log(vol_id)

                pred_vq = pred_vq_pad[:, :, :vol_t.shape[2], :vol_t.shape[3], :vol_t.shape[4]]
                pred_vq_np = np.clip((pred_vq.squeeze().cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
                np.save(out_vq, (pred_vq_np * mask_np) + (vol_obj.t1 * (1.0 - mask_np)))

            # 3D LDM Inference
            if not os.path.exists(out_ldm):
                prof_3d_ldm.start()
                pred_ldm_pad = self._repaint_3d_latent(vol_t_pad, mask_t_pad, num_inference_steps=50)
                prof_3d_ldm.stop_and_log(vol_id)

                pred_ldm = pred_ldm_pad[:, :, :vol_t.shape[2], :vol_t.shape[3], :vol_t.shape[4]]
                pred_ldm_np = np.clip((pred_ldm.squeeze().cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
                np.save(out_ldm, (pred_ldm_np * mask_np) + (vol_obj.t1 * (1.0 - mask_np)))

        # Clear VRAM
        del self.autoencoder_3d, self.unet_3d
        self.autoencoder_3d = None
        torch.cuda.empty_cache()
        gc.collect()

        # --- 3. MONAI 2D LDM ---
        self._load_2d_models()
        prof_2d_ldm = InferenceProfiler("monai_2d_ldm", self.results_dir)
        
        # for f in tqdm(test_files, desc="MONAI 2D Inference"):
        #     vol_id = os.path.basename(f).replace('.nii.gz', '').replace('.nii', '')
        #     out_2d = os.path.join(self.results_dir, f"monai_2d_ldm_{vol_id}.npy")
        #     if os.path.exists(out_2d): continue
            
        #     vol_obj = VolumeLoader.load(f)
        #     mask_np = VisualizationSuite._sample_real_brats_mask(vol_obj.t1, brats_pool)
        #     if np.sum(mask_np) < 10: continue
        for vol_id, vol_obj, mask_np in tqdm(eval_data, desc="MONAI 2D Inference"):
            out_2d = os.path.join(self.results_dir, f"monai_2d_ldm_{vol_id}.npy")
            if os.path.exists(out_2d): continue

            vol_np = (vol_obj.t1 * 2.0) - 1.0 
            # pred_vol_np = np.zeros_like(vol_np)
            pred_vol_np = vol_np.copy() # default to GT

            prof_2d_ldm.start()

            # 2D models expect 240x240, we pad slices individually
            # for z in range(vol_np.shape[0]):
            #     if np.sum(mask_np[z]) == 0:
            #         pred_vol_np[z] = vol_np[z]
            #         continue

            #     slice_t = torch.tensor(vol_np[z], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            #     mask_t = torch.tensor(mask_np[z], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

            #     pad_y, pad_x = max(0, 240 - slice_t.shape[2]), max(0, 240 - slice_t.shape[3])
            #     slice_pad = F.pad(slice_t, (0, pad_x, 0, pad_y), mode='constant', value=-1.0)
            #     mask_pad = F.pad(mask_t, (0, pad_x, 0, pad_y), mode='constant', value=0.0)

            #     pred_pad = self._repaint_2d_pixel(slice_pad, mask_pad, num_inference_steps=50)
            #     pred_vol_np[z] = pred_pad[0, 0, :slice_t.shape[2], :slice_t.shape[3]].cpu().numpy()

            # Find all slices that actually have a tumor mask to inpaint
            active_z = [z for z in range(vol_np.shape[0]) if np.sum(mask_np[z]) > 0]

            if active_z:
                # # Stack them into a single batch: (B, 1, H, W)
                # slices_t = torch.tensor(vol_np[active_z], dtype=torch.float32).unsqueeze(1).to(self.device)
                # masks_t = torch.tensor(mask_np[active_z], dtype=torch.float32).unsqueeze(1).to(self.device)

                # # 2D models expect 240x240, pad the whole batch
                # pad_y, pad_x = max(0, 240 - slices_t.shape[2]), max(0, 240 - slices_t.shape[3])
                # slices_pad = F.pad(slices_t, (0, pad_x, 0, pad_y), mode='constant', value=-1.0)
                # masks_pad = F.pad(masks_t, (0, pad_x, 0, pad_y), mode='constant', value=0.0)

                # # Run RePaint ONCE for the entire batch
                # preds_pad = self._repaint_2d_pixel(slices_pad, masks_pad, num_inference_steps=50)

                # # Unpack the batch back into the volume
                # for i, z in enumerate(active_z):
                #     pred_vol_np[z] = preds_pad[i, 0, :slices_t.shape[2], :slices_t.shape[3]].cpu().numpy()

                # Process in safe mini-batches to prevent 50+ GB VRAM explosions
                mini_batch_size = 4

                for b_start in range(0, len(active_z), mini_batch_size):
                    chunk_z = active_z[b_start : b_start + mini_batch_size]

                    # Stack them into a mini-batch: (B, 1, H, W)
                    slices_t = torch.tensor(vol_np[chunk_z], dtype=torch.float32).unsqueeze(1).to(self.device)
                    masks_t = torch.tensor(mask_np[chunk_z], dtype=torch.float32).unsqueeze(1).to(self.device)

                    # 2D models expect 240x240, pad the mini-batch
                    pad_y, pad_x = max(0, 240 - slices_t.shape[2]), max(0, 240 - slices_t.shape[3])
                    slices_pad = F.pad(slices_t, (0, pad_x, 0, pad_y), mode='constant', value=-1.0)
                    masks_pad = F.pad(masks_t, (0, pad_x, 0, pad_y), mode='constant', value=0.0)

                    # Run RePaint for the mini-batch
                    preds_pad = self._repaint_2d_pixel(slices_pad, masks_pad, num_inference_steps=50)

                    # Unpack the mini-batch back into the volume
                    for i, z in enumerate(chunk_z):
                        pred_vol_np[z] = preds_pad[i, 0, :slices_t.shape[2], :slices_t.shape[3]].cpu().numpy()

                    # Clear VRAM after each mini-batch
                    torch.cuda.empty_cache()

            prof_2d_ldm.stop_and_log(vol_id)

            pred_final_np = np.clip((pred_vol_np + 1.0) / 2.0, 0.0, 1.0)
            np.save(out_2d, (pred_final_np * mask_np) + (vol_obj.t1 * (1.0 - mask_np)))
