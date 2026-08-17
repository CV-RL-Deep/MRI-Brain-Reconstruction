import os
import math
import random
import glob

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from collections import Counter

import nibabel as nib

from tqdm import tqdm

from src.core.utils import logger
from src.data.cache import VolumeLoader
from configs.config import Config


def analyze_dataset_geometry(data_list, num_samples=3):
    """
    Loads samples, enforces Canonical Orientation (RAS+), and plots views.
    Works for both IXI (list of str) and BraTS (list of dicts).
    """
    print(f"--- Analyzing {num_samples} samples ---")

    # Handle mixed input types
    valid_list = [d['t1'] if isinstance(d, dict) else d for d in data_list]
    valid_list = [p for p in valid_list if p] # filter None

    samples = random.sample(valid_list, min(len(valid_list), num_samples))

    for path in samples:
        filename = os.path.basename(path)
        print(f"\nFile: {filename}")

        # 1. Load Raw
        img = nib.load(path)
        print(f"  Raw Shape: {img.shape}")

        # 2. Enforce Canonical (RAS+)
        # This aligns axes to: 0=Left-Right, 1=Posterior-Anterior, 2=Inferior-Superior
        canonical_img = nib.as_closest_canonical(img)
        can_data = canonical_img.get_fdata(dtype=np.float32)
        print(f"  Canonical (RAS+): {can_data.shape} [X=Sag, Y=Cor, Z=Axial]")

        # 3. Transpose to Logical (Z, Y, X) for Training
        # We want the "Depth" (Slices) to be the first axis.
        # Usually Axial (Z) is the slice axis.
        # Transpose (2, 1, 0) -> (Axial, Coronal, Sagittal)
        std_data = np.transpose(can_data, (2, 1, 0))
        print(f"  Training Input (Z,Y,X): {std_data.shape}")

        # 4. Crop Analysis
        mask = std_data > np.mean(std_data) * 0.1
        if np.any(mask):
            coords = np.argwhere(mask)
            bbox_shape = coords.max(axis=0) - coords.min(axis=0) + 1
            print(f"  Brain Content Size: {bbox_shape}")

        # 5. Visualize
        c_z, c_y, c_x = np.array(std_data.shape) // 2

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{filename} (Canonical -> Transposed 2,1,0)", fontsize=10, y=0.98)

        # View 0: Slice along Axis 0 (Z-axis / Axial)
        # Result is Y-X plane (Coronal-Sagittal axes) -> Axial Image
        axes[0].imshow(std_data[c_z, :, :], cmap='bone', aspect='equal')
        axes[0].set_title(f"Axis 0 (Index {c_z})\nPlane: Axial")

        # View 1: Slice along Axis 1 (Y-axis / Coronal)
        # Result is Z-X plane
        axes[1].imshow(std_data[:, c_y, :], cmap='bone', aspect='equal') # no rotation needed if layout is Z-Y-X
        axes[1].set_title(f"Axis 1 (Index {c_y})\nPlane: Coronal")

        # View 2: Slice along Axis 2 (X-axis / Sagittal)
        # Result is Z-Y plane
        axes[2].imshow(std_data[:, :, c_x], cmap='bone', aspect='equal')
        axes[2].set_title(f"Axis 2 (Index {c_x})\nPlane: Sagittal")

        # FIX: rect=[left, bottom, right, top]
        # Restricts subplots to lower 90% of figure, leaving room for suptitle
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.show()


class VolumeDashboard:
    def __init__(self, volume_path, axis_size, axis, dataset_dir, downsample=2):
        self.volume_path = volume_path
        self.axis_size = axis_size
        self.axis = axis
        self.dataset_dir = dataset_dir

        # Grid parameters
        self.ds = downsample
        self.cell_sz = 160 // self.ds
        self.cols = 12
        self.rows = math.ceil(self.axis_size / self.cols)

        # Preallocate entire grid as a black RGB image
        self.grid_w = self.cols * self.cell_sz
        self.grid_h = self.rows * self.cell_sz
        self.image = Image.new("RGB", (self.grid_w, self.grid_h), (0, 0, 0))

        self.counts = {i: 0 for i in range(axis_size)}
        self.active_indices = set()

        # NEW: Track how many sequences have been stamped onto this brain
        self.z_counter = 0

    def update(self, slice_idx, img_arr, role, info, z_index, is_target=False,
               tumor_mask=None):
        if slice_idx < 0 or slice_idx >= self.axis_size: return

        self.counts[slice_idx] += 1
        # count = self.counts[slice_idx]
        self.active_indices.add(slice_idx)

        # Normalize image to 0-255 uint8 RGB
        img_ds = img_arr[::self.ds, ::self.ds]
        img_u8 = np.clip(img_ds * 255.0, 0, 255).astype(np.uint8)

        # Create base cell image
        cell_img = Image.fromarray(img_u8).convert("RGB")
        cell_draw = ImageDraw.Draw(cell_img)

        # 1. Bevels (Inner Border)
        bevel_color = (0, 255, 0) # Green (Healthy)
        if not is_target and info.get('hallucination_proc'):
            bevel_color = (128, 0, 128) # Purple
        elif info.get('glioma_applied'):
            bevel_color = (255, 0, 0) # Red

        cell_draw.rectangle([0, 0, self.cell_sz-1, self.cell_sz-1],
                            outline=bevel_color, width=max(1, 2//self.ds))

        # 2. Contours (Using scipy to find boundary)
        if tumor_mask is not None and np.any(tumor_mask):
            draw_contour = False
            c_color = (0, 0, 255) # Blue

            if is_target:
                # Target slices NEVER get white contours. Only red if it's a real tumor
                if info['mode'] == 'tumor':
                    draw_contour = True
                    c_color = (255, 0, 0) # Red
            else:
                # Input slices get contours if they were augmented
                if info['glioma_applied']:
                    draw_contour = True
                    c_color = (
                        0, 0, 255
                    ) if info['mode'] == 'clean' else (
                        255, 0, 0
                    )

            if draw_contour:
                mask_ds = tumor_mask[::self.ds, ::self.ds]
                mask_bool = mask_ds > 0.5
                boundary = mask_bool ^ binary_dilation(mask_bool)

                # Stamp contour using RGBA overlay
                overlay = np.zeros((self.cell_sz, self.cell_sz, 4),
                                   dtype=np.uint8)
                overlay[boundary] =[*c_color, 255]
                cell_img.paste(Image.fromarray(overlay), (0,0),
                               Image.fromarray(overlay))

        # 3. Dots for Blur/Noise (Bottom Left)
        dot_color = None
        is_blur = info.get('blur_applied', False) if not is_target else False
        is_noise = info.get('noise_applied', False) if not is_target else False

        if is_blur and is_noise: dot_color = (255, 255, 255) # White
        elif is_blur: dot_color = (0, 255, 255) # Cyan
        elif is_noise: dot_color = (255, 255, 0) # Yellow

        if dot_color:
            r = max(2, 4 // self.ds)
            x, y = int(self.cell_sz * 0.1), int(self.cell_sz * 0.9)
            cell_draw.ellipse([x-r, y-r, x+r, y+r], fill=dot_color)

        # 4. Text Overlays
        # Top-Left: Norm Pos
        norm_pos = f"{slice_idx / max(1, self.axis_size - 1):.3f}"
        cell_draw.text((2, 2), norm_pos, fill=(255, 255, 255))

        # Top-Right: Role + Direction (e.g., "T-1 ->")
        dir_arrow = "<" if info['direction'] == 'forward' else ">"
        role_str = f"{role} {dir_arrow}"
        role_color = (0, 255, 255) if is_target else (255, 165, 0)

        # Right align text
        try: tw = cell_draw.textlength(role_str)
        except AttributeError: tw = cell_draw.textsize(role_str)[0] # legacy compat
        cell_draw.text((self.cell_sz - tw - 2, 2), role_str, fill=role_color)

        # Bottom-Right: Z-Index (Sequence ID)
        z_str = f"z={z_index}"
        try: tw = cell_draw.textlength(z_str)
        except AttributeError: tw = cell_draw.textsize(z_str)[0]
        cell_draw.text((self.cell_sz - tw - 2, self.cell_sz - 12), z_str,
                       fill=(0, 255, 0))

        # 5. Paste directly into Master Grid
        col = slice_idx % self.cols
        row = slice_idx // self.cols
        self.image.paste(cell_img, (col * self.cell_sz, row * self.cell_sz))

    def render(self, input_only=False):
        # Draw placeholder text on untouched indices
        draw = ImageDraw.Draw(self.image)

        # Get global max count for normalization
        max_count = max(self.counts.values()) if self.counts else 1
        max_count = max(1, max_count)

        for i in range(self.axis_size):
            col = i % self.cols
            row = i // self.cols
            x0 = col * self.cell_sz
            y0 = row * self.cell_sz

            if i not in self.active_indices:
                # col = i % self.cols
                # row = i // self.cols
                draw.text((col * self.cell_sz + 2, row * self.cell_sz + 2),
                          f"{i / max(1, self.axis_size - 1):.3f}",
                          fill=(100, 100, 100))
            else:
                # --- INFERNO HEATMAP PATCH (per-slice bottom right corner of each) ---
                count = self.counts[i]
                norm_val = count / max_count

                # Get RGB from colormap
                rgba = cm.inferno(norm_val)
                r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)

                rect_w, rect_h = 32, 14
                rx0 = x0 + self.cell_sz - rect_w - 2
                ry0 = y0 + self.cell_sz - rect_h - 2

                # High contrast text (black text on bright colors, white on dark)
                text_col = (0, 0, 0) if norm_val > 0.7 else (255, 255, 255)
                count_str = f"n={count}"

                try: tw = draw.textlength(count_str)
                except AttributeError: tw = draw.textsize(count_str)[0]

        # 2. --- GLOBAL BIRD'S-EYE HEATMAP IN THE LAST CELL ---
        # Find the absolute last cell of the entire dashboard grid
        last_cell_idx = (self.rows * self.cols) - 1
        lc_col = last_cell_idx % self.cols
        lc_row = last_cell_idx // self.cols

        lx0 = lc_col * self.cell_sz
        ly0 = lc_row * self.cell_sz

        # Draw a subtle border and title for the mini-heatmap
        draw.rectangle([lx0, ly0, lx0 + self.cell_sz - 1, ly0 + self.cell_sz - 1],
                       outline=(100, 100, 100))
        draw.text((lx0 + 4, ly0 + 2), "Global Sampling\nHeatmap",
                  fill=(255, 255, 255))

        # Calculate miniature block sizes
        # We map the mini-heatmap to the exact same column/row structure as the main dashboard
        y_offset = 28 # leave room for the title text
        mini_w = self.cell_sz / self.cols
        mini_h = (self.cell_sz - y_offset) / self.rows

        for i in range(self.axis_size):
            m_col = i % self.cols
            m_row = i // self.cols

            # Coordinates for this specific miniature block
            mx0 = lx0 + (m_col * mini_w)
            my0 = ly0 + y_offset + (m_row * mini_h)
            mx1 = mx0 + mini_w
            my1 = my0 + mini_h

            if i in self.active_indices:
                count = self.counts[i]
                norm_val = count / max_count
                rgba = cm.inferno(norm_val)
                r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                draw.rectangle([mx0, my0, mx1, my1], fill=(r, g, b))
            else:
                # If not sampled, draw it dark gray
                draw.rectangle(
                    [mx0, my0, mx1, my1], fill=(30, 30, 30), outline=(50, 50, 50)
                )

        # Save to disk
        axis_names = ['axial', 'coronal', 'sagittal']
        filename = os.path.basename(self.volume_path)
        base_name = filename.replace('.nii.gz', '').replace('.nii', '')

        save_path = os.path.join(self.dataset_dir, axis_names[self.axis],
                                 f"{base_name}.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.image.save(save_path)


class VisualizationSuite:
    @staticmethod
    def _load_vol_direct(path):
        """Deprecated: Use VolumeLoader.load instead."""
        # This wrapper keeps API compatibility for internal calls if needed,
        # but calls the canonical loader
        vol = VolumeLoader.load(path)
        return vol.t1 if vol else None

    @staticmethod
    def _sample_real_brats_mask(target_volume: np.ndarray, brats_pool: list, target_z: int = None) -> np.ndarray:
        """
        Extracts a real 3D tumor shape from a random BraTS volume and injects it 
        into the target IXI volume. If target_z is provided, centers the tumor at that slice.
        """
        mask = np.zeros_like(target_volume, dtype=np.float32)

        # 1. Find valid brain tissue coordinates in the target IXI volume
        target_coords = np.argwhere(target_volume > 0.1)
        if len(target_coords) == 0:
            return mask

        # 2. Pick a random BraTS volume that actually has a segmentation mask
        valid_brats =[v for v in brats_pool if v.seg is not None]
        if not valid_brats:
            # logger.warning("No BraTS volumes with masks found. Falling back to empty mask.")
            logger.warning("⚠️ No BraTS volumes with masks found. Falling back to synthetic mask to prevent Teacher Forcing zero-error curves.")
            return VisualizationSuite._generate_synthetic_mask(target_volume, max_radius=18)
            return mask

        source_brats = random.choice(valid_brats)

        # 3. Extract the real tumor (labels 1, 2, 4)
        tumor_mask = np.isin(source_brats.seg, [1, 2, 4]).astype(np.float32)
        tumor_coords = np.argwhere(tumor_mask > 0)

        if len(tumor_coords) == 0:
            return mask

        # 4. Crop the real tumor to its 3D bounding box
        z_min, y_min, x_min = tumor_coords.min(axis=0)
        z_max, y_max, x_max = tumor_coords.max(axis=0) + 1
        tumor_crop = tumor_mask[z_min:z_max, y_min:y_max, x_min:x_max]

        t_z, t_y, t_x = tumor_crop.shape

        # 5. Inject into target volume
        # To avoid unnatural centering, we force the tumor into the left or right hemisphere.
        # Assuming shape is (Z, Y, X), index 2 is the Sagittal (Left/Right) axis.
        mid_x = target_volume.shape[2] // 2

        # Pick a hemisphere and filter valid coordinates (adding a 10-pixel margin from the exact center)
        if random.random() < 0.5:
            hemisphere_coords =[c for c in target_coords if c[2] < mid_x - 10]
        else:
            hemisphere_coords = [c for c in target_coords if c[2] > mid_x + 10]

        # Fallback to all coords if the tumor is too big for the strict hemisphere split
        if len(hemisphere_coords) == 0:
            hemisphere_coords = target_coords

        # Try up to 50 times to find a center point where the tumor fits inside the bounds
        for _ in range(50):
            # cz, cy, cx = target_coords[random.randint(0, len(target_coords) - 1)]
            cz, cy, cx = hemisphere_coords[random.randint(0, len(hemisphere_coords) - 1)]

            if target_z is not None:
                cz = target_z  # force Z-center

            # Calculate injection bounds
            sz = cz - t_z // 2
            sy = cy - t_y // 2
            sx = cx - t_x // 2

            ez = sz + t_z
            ey = sy + t_y
            ex = sx + t_x

            # Check bounds
            if (sz >= 0 and sy >= 0 and sx >= 0 and 
                ez < target_volume.shape[0] and 
                ey < target_volume.shape[1] and 
                ex < target_volume.shape[2]):

                # Inject the real shape
                mask[sz:ez, sy:ey, sx:ex] = tumor_crop
                break

        # 6. Safety clip: Ensure the tumor doesn't spill into the background/air
        brain_mask = (target_volume > 0.01).astype(np.float32)
        mask = mask * brain_mask

        return mask

    @staticmethod
    def _generate_synthetic_mask(volume: np.ndarray, max_radius=20) -> np.ndarray:
        """Creates a realistic 3D Gaussian blob (simulated glioma) strictly inside the brain mass."""
        mask = np.zeros_like(volume, dtype=np.float32)

        # 1. Find valid brain tissue coordinates to place the tumor
        # Use > 0.1 to avoid placing the center on the extreme edges/skull
        coords = np.argwhere(volume > 0.1)
        if len(coords) == 0:
            return mask

        # 2. Pick a random center inside the brain
        center_idx = random.randint(0, len(coords) - 1)
        cz, cy, cx = coords[center_idx]

        # 3. Randomize elliptical shape (Gaussian covariance proxies)
        rz = random.uniform(max_radius * 0.4, max_radius * 0.8)
        ry = random.uniform(max_radius * 0.6, max_radius * 1.0)
        rx = random.uniform(max_radius * 0.6, max_radius * 1.0)

        # 4. Generate 3D grid
        z, y, x = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]

        # 5. Gaussian falloff equation
        dist = (
            ((z - cz) ** 2 / rz ** 2) +
            ((y - cy) ** 2 / ry ** 2) +
            ((x - cx) ** 2 / rx ** 2)
        )
        blob = np.exp(-dist)

        # 6. Threshold to create a solid mask (simulating an enhancing core)
        mask[blob > 0.3] = 1.0 # TODO: sum and clip

        # 7. Safety clip: Ensure the tumor doesn't spill into the background/air
        brain_mask = (volume > 0.01).astype(np.float32)
        mask = mask * brain_mask

        return mask

    @staticmethod
    def plot_augmentations(dataset, num_groups=2, batch_mode=False):
        """
        Point 1 & 6: Hunts for specific augmentation types (Glioma, Blur, Noise).
        """
        print("\n--- [1] Data Augmentation Visualization (Hunting Mode) ---")

        samples = {
            "glioma_only": [],
            "blur_only": [],
            "noise_only": [],
            "blur_and_noise": []
        }

        criteria = {
            "glioma_only": lambda s: s["glioma_applied"] and not s["blur_applied"] and not s["noise_applied"],
            "blur_only": lambda s: not s["glioma_applied"] and s["blur_applied"] and not s["noise_applied"],
            "noise_only": lambda s: not s["glioma_applied"] and not s["blur_applied"] and s["noise_applied"],
            "blur_and_noise": lambda s: not s["glioma_applied"] and s["blur_applied"] and s["noise_applied"],
        }

        print("Searching dataset for augmentation examples...")
        count = 0
        stop = False
        counts = Counter()

        # Limit search to avoid hanging forever
        for x_batch, y_batch, info_batch in tqdm(dataset.take(1000), total=1000,
                                                 disable=batch_mode):
            if stop: break

            # Robust unbatching for info dict
            i_batch = [
                dict(zip(info_batch, map(lambda a: a.numpy(), v)))
                for v in zip(*info_batch.values())
            ]

            batch_size = y_batch.shape[0]
            for b in range(batch_size):
                # Extract dictionary of tensors for this specific batch element
                x = {k: v[b] for k, v in x_batch.items()}
                y = y_batch[b]
                info = i_batch[b]
                # Only cast the specific boolean augmentation flags to integers
                aug_flags =['glioma_applied', 'blur_applied', 'noise_applied']
                counts.update({k: int(info[k]) for k in aug_flags if k in info})
                count += 1

                for key, func in criteria.items():
                    if len(samples[key]) < num_groups and func(info):
                        samples[key].append({'x': x, 'y': y})

                if all(len(v) >= num_groups for v in samples.values()):
                    stop = True
                    break

        # --- Plotting ---
        for k in range(num_groups):
            print(f"\n--- Augmentation Group {k + 1} ---")

            # Determine grid size based on first available sample
            first_sample = next((v[0] for v in samples.values() if v), None)
            first_sample = next((v[0] for v in samples.values() if v), None)
            if not first_sample: continue

            # Access 'history_input' from the dictionary
            num_inputs = first_sample['x']['history_input'].shape[-1]
            # num_inputs = first_sample['x'].shape[-1]
            num_rows = len(samples)
            fig, axes = plt.subplots(num_rows, num_inputs + 1, figsize=(15, 3 * num_rows))
            if num_rows == 1: axes = [axes]

            for i, (key, sample_list) in enumerate(samples.items()):
                if k >= len(sample_list):
                    # Placeholder if missing
                    ax = axes[i, 0] if num_rows > 1 else axes[0]
                    ax.text(0.5, 0.5, f"Missing: {key}", ha='center', color='red')
                    for j in range(num_inputs + 1):
                        (axes[i, j] if num_rows > 1 else axes[j]).axis('off')
                    continue

                s = sample_list[k]
                x_data, y_data = s['x'], s['y']

                # Row Title
                ax_first = axes[i, 0] if num_rows > 1 else axes[0]
                ax_first.set_ylabel(key.replace('_', ' ').title(), fontsize=12,
                                    weight='bold')

                # Plot Inputs
                for j in range(num_inputs):
                    ax = axes[i, j] if num_rows > 1 else axes[j]
                    # ax.imshow(x_data[:, :, j], cmap='bone', vmin=0.0, vmax=1.0)
                    ax.imshow(x_data['history_input'][:, :, j], cmap='bone',
                              vmin=0.0, vmax=1.0)
                    # if i == 0: ax.set_title(f"Input {j} (t - {num_inputs - j})")
                    if i == 0: ax.set_title(f"Input {j}")
                    ax.set_xticks([]); ax.set_yticks([])

                # Plot Target
                ax = axes[i, num_inputs] if num_rows > 1 else axes[num_inputs]
                ax.imshow(y_data[:, :, 0], cmap='bone', vmin=0.0, vmax=1.0)
                # if i == 0: ax.set_title("Target (t)")
                if i == 0: ax.set_title("Target")
                ax.axis('off')

            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_training_history(history):
        print("\n--- [3] Training History ---")
        metrics = history.history
        # Find any valid metric to determine epoch count
        any_key = next(iter(metrics.keys()))
        epochs = range(1, len(metrics[any_key]) + 1)

        # Identify base metrics (exclude 'val_' prefix)
        base_metrics = [k for k in metrics.keys() if not k.startswith('val_')]

        # Determine grid
        n_metrics = len(base_metrics)
        cols = min(n_metrics, 3)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_metrics == 1: axes = [axes]
        axes = np.array(axes).flatten()

        for i, metric in enumerate(base_metrics):
            ax = axes[i]

            # Train curve
            ax.plot(epochs, metrics[metric], 'b-o', label=f'Train {metric}')

            # Val curve
            val_key = f'val_{metric}'
            if val_key in metrics:
                ax.plot(epochs, metrics[val_key], 'r--s', label=f'Valid {metric}')

            ax.set_title(metric.upper())
            ax.set_xlabel("Epochs")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_hallucination_buffer(manager, num_samples=3):
        print("\n--- [4] Hallucination Buffer ---")
        buffer = manager.hallucination_buffer
        if not buffer:
            print("Buffer Empty.")
            return

        keys = random.sample(list(buffer.keys()), min(len(buffer), num_samples))
        fig, axes = plt.subplots(len(keys), 3, figsize=(12, 4 * len(keys)))
        if len(keys) == 1: axes = [axes]

        for i, key in enumerate(keys):
            path, axis, idx = key
            pred = buffer[key]

            # Try to load GT
            vol = VisualizationSuite._load_vol_direct(path)

            # Handle index logic:
            # If loaded directly, index is absolute. Buffer usually stores absolute.
            # But check bounds
            if vol is not None and idx < vol.shape[0]:
                gt = vol[idx]
            else:
                gt = np.zeros_like(pred)

            axes[i][0].imshow(gt, cmap='bone'); axes[i][0].set_title("Ground Truth")
            axes[i][1].imshow(pred, cmap='bone'); axes[i][1].set_title("Prediction")
            axes[i][2].imshow(np.abs(gt-pred), cmap='inferno'); axes[i][2].set_title("Error")
            for ax in axes[i]: ax.axis('off')
        plt.show()

    @staticmethod
    def analyze_best_worst(model, val_dataset, num_samples=3):
        print("\n--- [5] Best & Worst Predictions ---")
        best_samples = []
        worst_samples = []

        # Scan dataset
        # Limit to 50 batches for speed
        # --- Support both tf.data.Dataset and Keras Sequence ---
        if hasattr(val_dataset, 'take'):
            # tf.data.Dataset
            iterator = val_dataset.take(50)
        else:
            # Keras Sequence: slice the first 50 batches
            max_batches = min(50, len(val_dataset))
            iterator = [val_dataset[i] for i in range(max_batches)]
        # for x_batch, y_batch in val_dataset.take(50):
        for x_batch, y_batch in iterator:
            preds = model(x_batch, training=False)
            inputs = x_batch.numpy() if hasattr(x_batch, 'numpy') else x_batch
            inputs = x_batch['history_input'].numpy(
            ) if hasattr(x_batch['history_input'], 'numpy') else x_batch['history_input']
            targets = y_batch.numpy() if hasattr(y_batch, 'numpy') else y_batch

            # Calculate MSE per sample in batch
            batch_mse = np.mean(np.square(targets[..., 0] - preds[..., 0]),
                                axis=(1, 2))

            for i in range(targets.shape[0]):
                sample = {
                    'mse': batch_mse[i],
                    'input': inputs[i],
                    'gt': targets[i, ..., 0],
                    'pred': preds[i, ..., 0]
                }

                # Update Best
                best_samples.append(sample)
                best_samples.sort(key=lambda s: s['mse'])
                if len(best_samples) > num_samples: best_samples.pop()

                # Update Worst
                worst_samples.append(sample)
                worst_samples.sort(key=lambda s: s['mse'], reverse=True)
                if len(worst_samples) > num_samples: worst_samples.pop()

        # Sort final lists
        worst_samples.sort(key=lambda s: s['mse'])

        # --- Helper Plotter ---
        def visualize_group(samples, title):
            if not samples: return

            n_vis = len(samples)

            # Determine how many columns needed:
            # Inputs (N) + GT + Pred + Diff(GT) + Diff(Input)
            # Input shape: (H, W, C).
            # If mask attached, C = N+1.
            # We want to show first N channels as slices

            # Assuming standard config where C=3 or 4
            num_input_channels = samples[0]['input'].shape[-1]
            # Heuristic: If mask channel exists, it's the last one.
            # We only show image slices.
            # Let's show ALL input channels for completeness, or just N.
            # Let's show up to 3 input slices to keep plot sane
            num_input_show = min(num_input_channels, 3)

            cols = num_input_show + 4

            fig, axes = plt.subplots(n_vis, cols, figsize=(3 * cols, 3.75 * n_vis))
            fig.suptitle(title, fontsize=16, y=0.98)
            if n_vis == 1: axes = [axes]

            for i, s in enumerate(samples):
                ax_row = axes[i] if n_vis > 1 else axes

                # 1. Inputs
                for j in range(num_input_show):
                    ax = ax_row[j]
                    ax.imshow(s['input'][..., j], cmap='bone')
                    ax.set_title(f"Input {j} (t - {num_input_channels - j})")
                    ax.axis('off')

                # 2. GT
                ax = ax_row[num_input_show]
                ax.imshow(s['gt'], cmap='bone')
                ax.set_title("Target (t)")
                ax.axis('off')

                # 3. Pred
                ax = ax_row[num_input_show + 1]
                ax.imshow(s['pred'], cmap='bone')
                ax.set_title(f"Pred (MSE: {s['mse']:.5f})")
                ax.axis('off')

                # 4. Diff vs GT
                ax = ax_row[num_input_show + 2]
                diff_gt = np.abs(s['gt'] - s['pred'])
                ax.imshow(diff_gt, cmap='inferno', vmin=0, vmax=0.3)
                ax.set_title("Error |Pred - Target|")
                ax.axis('off')

                # 5. Diff vs Last Input (Temporal Change)
                # Last input slice is at index: num_input_channels - 1 (or -2 if mask)
                # Let's assume input image slice is at index 2 (if N=3)
                idx_last_input = 2 if num_input_channels >= 3 else 0
                last_input = s['input'][..., idx_last_input]

                ax = ax_row[num_input_show + 3]
                diff_in = np.abs(s['pred'] - last_input)
                ax.imshow(diff_in, cmap='inferno', vmin=0, vmax=0.3)
                ax.set_title(f"Diff |Pred - Input {j}|")
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        if best_samples: visualize_group(best_samples, "Best Predictions (Lowest Error)")
        if worst_samples: visualize_group(worst_samples, "Worst Predictions (Highest Error)")

    @staticmethod
    def plot_autoregressive_performance(reconstructor, file_info, manager, masked_inference=False):
        # filename = os.path.basename(volume_path)
        filename = os.path.basename(file_info['t1'])
        print(f"\n--- [7] Autoregressive Analysis: {filename} ---")

        # 1. Load
        # vol = VisualizationSuite._load_vol_direct(volume_path)
        vol_obj = VolumeLoader.load(file_info['t1'], file_info['seg'])
        # if vol is None: return
        if vol_obj is None: return

        # Get volume and mask
        vol = vol_obj.t1
        mask_vol = vol_obj.seg
        if mask_vol is None and masked_inference:
            print(f"WARNING: masked inference is impossible, NO MASK!")
            masked_inference = False
        if not masked_inference:
            # Reset the mask if we want pure autoregressive
            mask_vol = None

        # 2. Ranges
        # Load Mask if provided (for BraTS)
        # If IXI, mask_vol is None, so we just run pure AR.
        # But user requested "Masked Autoregressive" visualization.
        # For IXI (Healthy), we don't have a tumor mask to hide behind.
        # So "Masked AR" only makes sense for BraTS or if we inject a fake mask
        center = vol.shape[0] // 2
        span = 40
        fwd_end = min(vol.shape[0], center + span)
        bwd_start = max(0, center - span)

        if fwd_end - bwd_start < 20: return

        # 3. Reconstruct
        if mask_vol is not None:
            print("Running Forward Masked Inference (Teacher Forcing)...")
        else:
            print("Running Forward Pure Autoregression Inference...")
        recon_fwd = reconstructor.autoregressive_restore(vol, center, fwd_end,
                                                         'forward',
                                                         mask_volume=mask_vol)
        if mask_vol is not None:
            print("Running Backward Masked Inference (Teacher Forcing)...")
        else:
            print("Running Backward Pure Autoregression Inference...")
        recon_bwd = reconstructor.autoregressive_restore(vol, bwd_start, center,
                                                         'backward',
                                                         mask_volume=mask_vol)

        # 4. Error Curves
        fwd_mse = np.mean(np.square(vol[center:fwd_end] - recon_fwd[center:fwd_end]), axis=(1, 2))
        bwd_mse = np.mean(np.square(vol[bwd_start:center] - recon_bwd[bwd_start:center]), axis=(1, 2))

        plt.figure(figsize=(14, 6))
        plt.plot(np.arange(bwd_start, center), bwd_mse, 'g-o', label='Backward (<-)')
        plt.plot(np.arange(center, fwd_end), fwd_mse, 'r-o', label='Forward (->)')
        plt.axvline(x=center, color='b', linestyle='--', label='Start (Center Slice)')

        if mask_vol is not None:
            plt.title(f"Bidirectional (Masked) Error Accumulation: {filename}")
        else:
            plt.title(f"Bidirectional Autoregressive Error Accumulation: {filename}")

        plt.xlabel("Slice Index")
        plt.ylabel("Mean Squared Error")
        plt.legend(); plt.grid(True, alpha=0.3); plt.show()

        # 5. Detailed Visuals (Input Stack + Mask + Pred)
        # We show 3 points: Backward Tail, Center, Forward Tail
        idx_show = [
            # bwd_start + 5,
            center - (center - bwd_start) // 2,
            center - 3,
            center - 1,
            center,
            center + 1,
            center + 3,
            center + (fwd_end - center) // 2,
            # fwd_end - 5
        ]
        # labels = ["Backward Tail", "Center (Anchor)", "Forward Tail"]

        N = reconstructor.cfg.data.neighborhood
        cols = N + 4 # inputs(N) + Mask + GT + Pred + Diff
        rows = len(idx_show)

        fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 4.2 * rows)) # increase height
        if mask_vol is not None:
            fig.suptitle(f"Reconstruction Details (Masked): {filename}", y=0.96,
                         fontsize=16)
        else:
            fig.suptitle(f"Reconstruction Details: {filename}", y=0.96,
                         fontsize=16)

        for i, idx in enumerate(idx_show):
            idx = int(max(0, min(idx, vol.shape[0] - 1)))

            # Select Source
            if idx < center:
                img_r = recon_bwd[idx]; name="Backward"
                context_vol = recon_bwd
                # Backward Model Input Order: [t+3, t+2, t+1]
                ctx_indices = [idx + N - k for k in range(N)] # [idx+3, idx+2, idx+1]
            elif idx > center:
                img_r = recon_fwd[idx]; name="Forward"
                context_vol = recon_fwd
                # Forward Model Input Order: [t-3, t-2, t-1]
                ctx_indices = [idx - N + k for k in range(N)] # [idx-3, idx-2, idx-1]
            else:
                img_r = vol[idx]; name="Anchor"
                context_vol = vol
                ctx_indices = [idx - N + k for k in range(N)]

            # 1. Plot Context (Model Input Order)
            for j in range(N):
                c_idx = ctx_indices[j]
                ax = axes[i][j]
                if 0 <= c_idx < vol.shape[0]:
                    ax.imshow(context_vol[c_idx], cmap='bone')
                    ax.set_title(f"In[{j}]\nSlice {c_idx}")
                else:
                    ax.set_title("OOB")
                ax.axis('off')

            # 2. Mask
            ax = axes[i][N]
            mask = (vol[idx] > 0.01).astype(np.float32)
            ax.imshow(mask, cmap='gray')
            ax.set_title("Mask")
            ax.axis('off')

            # 3. GT
            ax = axes[i][N+1]
            ax.imshow(vol[idx], cmap='bone')
            ax.set_title(f"GT {idx}")
            ax.axis('off')

            # 4. Pred
            ax = axes[i][N+2]
            ax.imshow(img_r, cmap='bone')
            ax.set_title("Pred")
            ax.axis('off')

            # 5. Diff
            ax = axes[i][N+3]
            mse = np.mean((vol[idx]-img_r)**2)
            ax.imshow(np.abs(vol[idx]-img_r), cmap='inferno', vmin=0, vmax=0.3)
            ax.set_title(f"Err {mse:.4f}")
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_bidirectional(reconstructor, brats_file, manager, num_vis_slices=5,
                           masked_inference=False):
        """
        Visualizes bidirectional autoregressive inference on a BraTS tumor volume.
        Matches the original pipeline's visualization style.
        """
        print(f"\n--- [9] Bidirectional Inpainting: {brats_file['id']} ---")

        # 1. Load Data using Universal Loader (Canonical Orientation & Crop)
        vol_obj = VolumeLoader.load(brats_file['t1'], brats_file['seg'])
        if vol_obj is None:
            print("Failed to load volume!")
            return

        t1 = vol_obj.t1
        seg = vol_obj.seg

        if masked_inference and seg is None:
            print("WARNING: No real mask found. Generating synthetic spherical mask for IXI evaluation.")
            seg = VisualizationSuite._generate_synthetic_mask(t1, radius=16)

        # 2. Determine Range based on Tumor
        # Assuming Axis 0 is Axial (Slice) after VolumeLoader processing
        start, end = 0, t1.shape[0]

        if seg is not None:
            # Find slices containing tumor labels (1, 2, 4)
            # Sum spatial dims (1, 2) to get tumor pixels per slice
            tumor_presence = np.any(np.isin(seg, [1, 2, 4]), axis=(1, 2))
            indices = np.where(tumor_presence)[0]

            if len(indices) == 0:
                print("No tumor found in segmentation mask.")
                return

            # Add padding context
            pad = 2
            start = max(0, indices[0] - pad)
            end = min(t1.shape[0], indices[-1] + pad + 1)
        else:
            print("No segmentation mask available. Using center volume.")
            center = t1.shape[0] // 2
            start = center - num_vis_slices * 3
            end = center + num_vis_slices * 3

        print(f"Tumor Range found: Slice {start} to {end} (Total {end-start} slices)")

        # Run masked on demand, otherwise as possible
        mask = seg if masked_inference else None

        # 3. Run Forward Pass (Bottom -> Top)
        # Context comes from [start-N ... start]
        recon_fwd = reconstructor.autoregressive_restore(t1, start, end, 'forward',
                                                         mask_volume=mask)

        # 4. Run Backward Pass (Top -> Bottom)
        # Context comes from [end ... end+N]
        recon_bwd = reconstructor.autoregressive_restore(t1, start, end, 'backward',
                                                         mask_volume=mask)

        # 5. Visualization Selection
        # Select N indices evenly spaced within the range
        if end - start < num_vis_slices:
            vis_indices = range(start, end)
        else:
            vis_indices = np.linspace(start, end-1, num=num_vis_slices, dtype=int)

        # Setup Plot
        n_rows = len(vis_indices)
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
        if not masked_inference:
            fig.suptitle(f"Bidirectional Autoregressive Inpainting: {brats_file['id']}",
                         fontsize=16, y=0.98)
        else:
            fig.suptitle(f"Bidirectional (Masked) Inpainting: {brats_file['id']}",
                         fontsize=16, y=0.98)

        cols = [
            "Ground Truth",
            "Forward (Bot->Top)",
            "Backward (Top->Bot)",
            "Fwd Diff",
            "Bwd Diff"
        ]

        # Handle single row case
        if n_rows == 1:
            axes = [axes]
            header_ax = axes[0]
        else:
            header_ax = axes[0]

        # Set Column Titles
        for ax, col in zip(header_ax, cols):
            ax.set_title(col, fontsize=12, weight='bold')

        for i, idx in enumerate(vis_indices):
            row_axes = axes[i]

            # Data
            gt_slice = t1[idx]
            seg_slice = seg[idx] if seg is not None else np.zeros_like(gt_slice)
            fwd_slice = recon_fwd[idx]
            bwd_slice = recon_bwd[idx]

            # 1. Ground Truth + Red Tumor Outline
            row_axes[0].imshow(gt_slice, cmap='bone')
            if np.sum(seg_slice) > 0:
                # Overlay tumor contour
                row_axes[0].contour(seg_slice, levels=[0.5], colors='red',
                                    linewidths=0.5)
            row_axes[0].set_ylabel(f"Slice {idx}", fontsize=12)

            # 2. Forward Prediction
            row_axes[1].imshow(fwd_slice, cmap='bone')

            # 3. Backward Prediction
            row_axes[2].imshow(bwd_slice, cmap='bone')

            # 4. Forward Difference (Heatmap)
            diff_fwd = np.abs(gt_slice - fwd_slice)
            row_axes[3].imshow(diff_fwd, cmap='inferno', vmin=0, vmax=0.3)

            # 5. Backward Difference (Heatmap)
            diff_bwd = np.abs(gt_slice - bwd_slice)
            row_axes[4].imshow(diff_bwd, cmap='inferno', vmin=0, vmax=0.3)

            # Remove ticks
            for ax in row_axes:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_sampling_statistics(manager):
        print("\n--- [Stats] Data Sampling Distribution ---")
        stats = manager.stats.hits
        if not stats:
            print("No statistics recorded.")
            return

        # Organize data
        # Structure: data[dataset][axis] = {slice_idx: count}
        # Aggregate per volume for global check

        # 1. Volume Hits (Histogram)
        vol_counts = Counter()
        for (d, v, a, s, _), count in stats.items():
            vol_counts[f"{d}_{v}"] += count

        # Plot Top 50 volumes
        # FIX: Ensure we don't request more samples than actually exist
        total_hits = sum(vol_counts.values())
        if total_hits == 0:
            print("No sampling statistics to plot.")
            return

        k_samples = min(50, total_hits)

        # Plot Random 50 Sampled Volumes (Weighted by frequency)
        top_vols = random.sample(tuple(vol_counts.keys()),
                                 counts=tuple(vol_counts.values()), k=k_samples)

        labels, values = zip(*Counter(top_vols).items())
        # top_vols = vol_counts.most_common(50)
        # top_vols = random.sample(tuple(vol_counts.keys()),
        #                          counts=tuple(vol_counts.values()), k=50)
        # labels, values = zip(*top_vols)
        # labels, values = zip(*Counter(top_vols).items())

        plt.figure(figsize=(15, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=90, fontsize=8)
        plt.title("50 Random Sampled Volumes")
        plt.show()

        # 2. Slice Distribution per Dataset (The "Fence" check)
        # We aggregate all volumes together to see global slice preference
        slice_counts = {'ixi': {0: Counter(), 1: Counter(), 2: Counter()},
                        'brats': {0: Counter(), 1: Counter(), 2: Counter()}}

        for (d, v, a, s, _), count in stats.items():
            slice_counts[d][a][s] += count

        # Plot
        for dataset in ['ixi', 'brats']:
            fig, axes = plt.subplots(3, 1, figsize=(15, 9))
            fig.suptitle(f"{dataset.upper()} Slice Distribution (Aggregated)",
                         fontsize=16)

            axes_names = ['Axial (0)', 'Coronal (1)', 'Sagittal (2)']
            for axis in [0, 1, 2]:
                data = slice_counts[dataset][axis]
                if not data:
                    axes[axis].text(0.5, 0.5, "No Data", ha='center')
                    axes[axis].set_title(axes_names[axis])
                    continue

                # Fill curve
                max_slice = max(data.keys())
                x = range(max_slice + 1)
                y = [data[i] for i in x]

                axes[axis].bar(x, y, width=1.0)
                axes[axis].set_title(axes_names[axis])
                axes[axis].set_xlabel("Slice Index")
                axes[axis].set_ylabel("Hits")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # 3. Direction Balance
        dir_counts = {'ixi': Counter(), 'brats': Counter()}
        for (d, v, a, s, direction), count in stats.items():
            dir_counts[d][direction] += count

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle("Direction Balance (Forward vs Backward)", fontsize=14)

        for i, ds in enumerate(['ixi', 'brats']):
            counts = dir_counts[ds]
            if counts:
                axes[i].pie(counts.values(), labels=counts.keys(),
                            colors=plt.cm.Accent.colors, autopct='%1.1f%%')  # Pastel1, Set1, Dark2
                axes[i].set_title(ds.upper())
            else:
                axes[i].text(0.5, 0.5, "No Data")

        plt.show()


def display_random_dashboards(base_dirs=None):
    """
    Searches the specified EDA directories (or auto-detects all 'eda_*' folders),
    picks one random dashboard per dataset per axis, and displays it in native resolution.
    """
    from IPython.display import display, Image, HTML

    print(">>> Searching for generated EDA dashboards...\n")

    # Auto-detect directories if not provided
    if base_dirs is None:
        base_dirs =[
            d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('eda_')
        ]

    if not base_dirs:
        print("No 'eda_' directories found. Did the simulation finish?")
        return

    axes = ['axial', 'coronal', 'sagittal']

    for dset in sorted(base_dirs):
        for axis in axes:
            # Search for JPGs in the specific dataset/axis folder
            search_path = os.path.join(dset, axis, "*.jpg")
            files = glob.glob(search_path)

            if not files:
                continue

            # Pick one random dashboard
            chosen_file = random.choice(files)
            filename = os.path.basename(chosen_file)

            # Print a nice formatted HTML header
            header = f"""
            <hr>
            <h3 style='margin-bottom: 5px;'>
                Dataset: <span style='color: #E67E22;'>{dset}</span> | 
                Projection: <span style='color: #2ECC71;'>{axis.capitalize()}</span>
            </h3>
            <p style='margin-top: 0px; color: #7F8C8D;'>File: {filename}</p>
            """
            display(HTML(header))

            # Display the actual image natively
            display(Image(filename=chosen_file))
