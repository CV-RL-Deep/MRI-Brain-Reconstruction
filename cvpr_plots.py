import os
import glob
import shutil
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from configs.config import CFG
from brec.data.files import find_t1_files, get_brats_subjects
from main import get_ablation_configs
from brec.data.cache import VolumeLoader
from brec.models.builder import ModelBuilder
from brec.inference.reconstructor import VolumeReconstructor
from brec.evaluation.visualizer import VisualizationSuite


# 1. Config-Driven Pathing & Auto-Unzip
# Safely pull the paths from the global config
weights_dir = CFG.data.weights_dir
results_dir = CFG.data.results_dir
figures_dir = CFG.data.figures_dir

# Handle zipped datasets (common when uploading artifacts to Kaggle)
if os.path.exists(weights_dir + '.zip'):
    print(f"Extracting {weights_dir}.zip to local './ablations'...")
    with zipfile.ZipFile(weights_dir + '.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    weights_dir = 'ablations'
elif not os.path.exists(weights_dir):
    print(f"Warning: {weights_dir} not found. Defaulting to local 'ablations'.")
    weights_dir = 'ablations'

print(f"Using weights_dir: {weights_dir}")
print(f"Using results_dir: {results_dir}")
os.makedirs(figures_dir, exist_ok=True)


# 2. Strict CVPR Aesthetics (No System LaTeX Dependency)
plt.rcParams.update({
    "text.usetex": False,          # Bypasses the 'latex not found' crash
    "font.family": "serif",        # Serif to match Times
    "mathtext.fontset": "stix",    # STIX matches Times New Roman math rendering perfectly
    "font.size": 8,                # Base size
    "axes.titlesize": 8,           # Title size
    "axes.labelsize": 8,           # Axis label size
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,          # Compact legend
    "figure.titlesize": 9,
    "pdf.fonttype": 42,            # Embed fonts
    "ps.fonttype": 42
})

# Standardized color palette for consistency across all charts
PALETTE = {
    # Ablation Track
    'drift_vanilla': '#94a3b8',    # Slate Gray (Unbounded Drift representation)
    'baseline': '#E63946',         # Red
    'spade': '#F4A261',            # Orange
    'buffer': '#2A9D8F',           # Green
    'full': '#264653',             # Dark Blue
    # SOTA Track
    'ours_full_masked': '#1D3557', # Navy Blue
    'monai_3d_ldm': '#8E44AD',     # Purple
    'monai_2d_ldm': '#E67E22',     # Carrot Orange
    'monai_vqgan': '#7F8C8D'       # Gray (Upper Bound)
}

LABELS = {
    # Ablation Track
    'drift_vanilla': 'Vanilla U-Net (Unbounded Drift)',
    'baseline': 'Baseline U-Net',
    'spade': '+ SPADE',
    'buffer': '+ SPADE + Buffer',
    'full': '+ SPADE + Buffer + PE (Final)',
    # SOTA Track
    'ours_full_masked': 'Ours (2.5D SPADE AR)',
    'monai_3d_ldm': 'MONAI 3D LDM',
    'monai_2d_ldm': 'MONAI 2D LDM',
    # 'monai_vqgan': 'MONAI 3D VQ-GAN (Upper Bound)'
    'monai_vqgan': 'MONAI 3D VQ-GAN'
}


def generate_figure_1_drift_curve():
    """Generates Figure 1: The Quantitative Drift Curves (SSIM and FFL)."""
    print("Generating Figure 1: Drift Curves (SSIM & FFL)...")

    # csv_path = os.path.join(ABLATION_DIR, 'ablation_metrics.csv')
    csv_path = os.path.join(results_dir, 'ablation_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Only look at forward generation for simplicity in the chart
    df = df[df['Direction'] == 'forward']

    is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive'
    max_k = 10 if is_interactive else 40

    # --- PLOT 1: SSIM (Spatial Degradation) ---
    # CVPR Single Column width = 3.25 inches
    fig_ssim, ax_ssim = plt.subplots(figsize=(3.25, 2.5))

    sns.lineplot(
        data=df, x='Rollout_Step_k', y='SSIM', hue='Ablation',
        palette=PALETTE, linewidth=1.5, ax=ax_ssim
    )

    ax_ssim.set_xlim(1, max_k)
    ax_ssim.set_ylim(0.2, 1.0)
    ax_ssim.set_xlabel(r'Autoregressive Rollout Step ($k$)')
    ax_ssim.set_ylabel(r'Structural Similarity (SSIM) $\uparrow$')
    ax_ssim.grid(True, linestyle='--', alpha=0.6)

    handles, labels = ax_ssim.get_legend_handles_labels()
    safe_labels =[LABELS.get(l, l) for l in labels]
    ax_ssim.legend(handles, safe_labels, title="", loc='lower left',
                   fontsize=6, framealpha=0.75)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'Fig1_DriftCurve_SSIM.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig_ssim)
    print("  -> Saved Fig1_DriftCurve_SSIM.pdf")

    # --- PLOT 2: FFL (Frequency/Texture Degradation) ---
    # We only plot FFL if it exists in the dataframe (backwards compatibility check)
    if 'FFL' in df.columns:
        fig_ffl, ax_ffl = plt.subplots(figsize=(3.25, 2.5))

        sns.lineplot(
            data=df, x='Rollout_Step_k', y='FFL', hue='Ablation',
            palette=PALETTE, linewidth=1.5, ax=ax_ffl
        )

        ax_ffl.set_xlim(1, max_k)
        # FFL is unscaled, let Seaborn auto-scale the Y-axis but force bottom to 0
        ax_ffl.set_ylim(bottom=0.0)
        ax_ffl.set_xlabel(r'Autoregressive Rollout Step ($k$)')
        ax_ffl.set_ylabel(r'Focal Frequency Loss (FFL) $\downarrow$')
        ax_ffl.grid(True, linestyle='--', alpha=0.6)

        handles, labels = ax_ffl.get_legend_handles_labels()
        safe_labels =[LABELS.get(l, l) for l in labels]
        ax_ffl.legend(handles, safe_labels, title="", loc='upper left',
                      fontsize=6, framealpha=0.75)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'Fig1_DriftCurve_FFL.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig_ffl)
        print("  -> Saved Fig1_DriftCurve_FFL.pdf")
    else:
        print("  -> Skipping FFL plot (FFL metric not found in CSV. Re-run ablation inference).")


def generate_figure_2_evolution_matrix():
    """Generates Figure 2: The Qualitative Evolution Matrix."""
    print("Generating Figure 2: Evolution Matrix...")

    # Find a target volume to visualize (preferably an IXI file for pure brain drift)
    npy_files = glob.glob(os.path.join(results_dir, 'full_*.npy'))
    if not npy_files:
        # print("Error: No ablation numpy arrays found.")
        print(f"Error: No ablation numpy arrays found in '{results_dir}'.")
        return

    # Extract Vol ID from the first 'full_...' file
    is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive'
    # Loop over up to 5 volumes to ensure we can verify the phenomenon isn't an anomaly
    for vol_idx, npy_file in enumerate(npy_files[:5 if is_interactive else None]):
        vol_id = os.path.basename(npy_file).replace('full_', '').replace('.npy', '')

        # Load all models and Ground Truth for this volume
        vols = {}
        # for ab in['baseline', 'spade', 'buffer', 'full']:
        #     path = os.path.join(results_dir, f'{ab}_{vol_id}.npy')
        #     if os.path.exists(path):
        #         vols[ab] = np.load(path)
        for ab in ['drift_vanilla', 'baseline', 'spade', 'buffer', 'full']:
            path = os.path.join(results_dir, f'{ab}_{vol_id}.npy')
            if os.path.exists(path):
                vols[ab] = np.load(path)

        # Locate the original NIfTI file to extract real Ground Truth
        # gt_vol = None
        # search_paths = glob.glob(
        #     f"/kaggle/input/**/{vol_id}", recursive=True
        # ) + glob.glob(
        #     f"data/**/{vol_id}", recursive=True
        # )
        # if search_paths:
        #     gt_obj = VolumeLoader.load(search_paths[0])
        #     if gt_obj:
        #         gt_vol = gt_obj.t1
        # Locate the pre-saved Ground Truth from the results directory
        gt_vol = None
        gt_path = os.path.join(results_dir, f"gt_{vol_id}.npy")
        if os.path.exists(gt_path):
            gt_vol = np.load(gt_path)
        else:
            # Fallback search if pre-saved .npy is missing
            search_paths = glob.glob(
                f"/kaggle/input/**/{vol_id}", recursive=True
            ) + glob.glob(
                f"data/**/{vol_id}", recursive=True
            )
            if search_paths:
                gt_obj = VolumeLoader.load(search_paths[0])
                if gt_obj:
                    gt_vol = gt_obj.t1

        # Scan and dynamically render only existing ablation runs
        row_keys = []
        for ab in ['drift_vanilla', 'baseline', 'spade', 'buffer', 'full']:
            if ab in vols:
                row_keys.append(ab)

        # row_keys = ['baseline', 'spade', 'buffer', 'full']
        if gt_vol is not None:
            vols['gt'] = gt_vol
            LABELS['gt'] = 'Ground Truth'
            row_keys.append('gt')

        # THE FIX: Dynamically adapt sequence steps to Interactive vs Batch mode limits
        is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive'
        rollout_span = 5 if is_interactive else 20
        start_idx = vols['full'].shape[0] // 2 - rollout_span

        # Let's slice at relative k steps:
        k_steps =[1, 4, 7, 9] if is_interactive else [1, 10, 20, 39]

        # 3.25" width, height scales with rows. Hspace adjusted to fit labels above images.
        fig, axes = plt.subplots(len(row_keys), len(k_steps),
                                 figsize=(3.25, 1.365 * len(row_keys)),
                                 gridspec_kw={'wspace': 0.035, 'hspace': 0.035})

        for r, ab in enumerate(row_keys):
            for c, k in enumerate(k_steps):
                ax = axes[r, c]
                z_idx = start_idx + k

                if ab in vols:
                    # Use vmin/vmax to prevent normalization clipping
                    ax.imshow(vols[ab][start_idx + k], cmap='bone', vmin=0, vmax=1)
                ax.axis('off')

                # Move Ablation label ABOVE the row (Left aligned) for compactness
                if c == 0:
                    safe_label = LABELS.get(ab, ab)#.replace('_', r'\_')
                    ax.text(0.1, 1.175, safe_label, transform=ax.transAxes,
                            ha='left', va='bottom', fontsize=8, weight='bold')

                # Show t-steps on every image to save vertical space
                ax.set_title(f"$t={k}$", fontsize=8, pad=2)

        plt.savefig(
            os.path.join(figures_dir, f"Fig2_EvolutionMatrix_vol{vol_idx + 1:03d}.pdf"),
            dpi=300, bbox_inches='tight', pad_inches=0.01
        )
        plt.close()
        print(f"  -> Saved Fig2_EvolutionMatrix_vol{vol_idx + 1:03d}.pdf")


def generate_figure_3_pe_proof():
    """Generates Figure 3: Z-Axis Proof showing ventricles appearing/disappearing."""
    print("Generating Figure 3: Positional Encoding Proof...")

    # We want to compare 'buffer' (No PE) vs 'full' (+PE)
    npy_files = glob.glob(os.path.join(results_dir, 'full_*.npy'))
    if not npy_files: return

    is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive'
    # Loop over up to 5 volumes to ensure we can verify the phenomenon isn't an anomaly
    for vol_idx, npy_file in enumerate(npy_files[:5 if is_interactive else None]):
        vol_id = os.path.basename(npy_file).replace('full_', '').replace('.npy', '')

        try:
            vol_buffer = np.load(os.path.join(results_dir, f'buffer_{vol_id}.npy'))
            vol_full = np.load(os.path.join(results_dir, f'full_{vol_id}.npy'))
        except Exception as e:
            print(f"Error loading arrays for Fig 3: {e}")
            return

        # Choose a slice late in the rollout near the top of the head
        # where ventricles should NOT exist but 'buffer' hallucinates them.
        # Safely target the very last generated slice instead of an out-of-bounds hardcoded index
        rollout_span = 5 if is_interactive else 20
        z_idx = (vol_full.shape[0] // 2) + (rollout_span - 1)

        # 1. Debugging: Check for NaNs/Infs which cause black screens
        # print(f"Debug: Slice {z_idx} Max Value: {vol_buffer[z_idx].max():.4f}")

        # Load Real Ground Truth
        # gt_slice = vol_full[z_idx] # fallback
        # search_paths = glob.glob(
        #     f"/kaggle/input/**/{vol_id}", recursive=True
        # ) + glob.glob(
        #     f"data/**/{vol_id}", recursive=True
        # )
        # if search_paths:
        #     gt_obj = VolumeLoader.load(search_paths[0])
        #     if gt_obj:
        #         gt_slice = gt_obj.t1[z_idx]
        # Load Real Ground Truth directly from pre-saved volume
        gt_slice = None
        gt_path = os.path.join(results_dir, f"gt_{vol_id}.npy")
        if os.path.exists(gt_path):
            gt_vol = np.load(gt_path)
            gt_slice = gt_vol[z_idx]
        else:
            # Fallback searches if pre-saved .npy is missing
            search_paths = glob.glob(
                f"/kaggle/input/**/{vol_id}", recursive=True
            ) + glob.glob(
                f"data/**/{vol_id}", recursive=True
            )
            if search_paths:
                gt_obj = VolumeLoader.load(search_paths[0])
                if gt_obj:
                    gt_slice = gt_obj.t1[z_idx]

        # If all GT loading paths fail, fallback to vol_full as a last resort
        if gt_slice is None:
            gt_slice = vol_full[z_idx]

        # Calculate MSEs
        mse_buffer = np.mean((gt_slice - vol_buffer[z_idx]) ** 2)
        mse_full = np.mean((gt_slice - vol_full[z_idx]) ** 2)

        # Use consistent vmin/vmax
        # 2x2 Grid for 3.25 inches (Much more readable)
        fig, axes = plt.subplots(2, 2, figsize=(3.25, 3.25 * (1.0 * gt_slice.shape[0] /
                                                              gt_slice.shape[1])),
                                 gridspec_kw={'wspace': 0.15,
                                              'hspace': 0.15})
        axes = axes.flatten()

        # 1. Model w/out PE
        axes[0].imshow(vol_buffer[z_idx], cmap='bone', vmin=0, vmax=1)
        axes[0].set_title("No Pos. Encoding", pad=2)

        # 2. Model w/ PE
        axes[1].imshow(vol_full[z_idx], cmap='bone', vmin=0, vmax=1)
        axes[1].set_title("With Pos. Encoding", pad=2)

        # 3. Error Map (No PE)
        err_buffer = np.abs(gt_slice - vol_buffer[z_idx])
        axes[2].imshow(err_buffer, cmap='inferno', vmin=0, vmax=0.3)
        axes[2].set_title(f"Error MSE: {mse_buffer:.4f}", pad=2)

        # 4. Error Map (PE)
        err_full = np.abs(gt_slice - vol_full[z_idx])
        axes[3].imshow(err_full, cmap='inferno', vmin=0, vmax=0.3)
        axes[3].set_title(f"Error MSE: {mse_full:.4f}", pad=2)

        for ax in axes:
            # ax.axis('off')
            ax.set_xticks([]); ax.set_yticks([])

        plt.savefig(
            os.path.join(figures_dir, f"Fig3_PE_Proof_vol{vol_idx + 1:03d}.pdf"),
            dpi=300, bbox_inches='tight', pad_inches=0.01
        )
        plt.close()
        print(f"  -> Saved Fig3_PE_Proof_vol{vol_idx + 1:03d}.pdf")


def generate_figure_4_anatomical_dsc():
    """Generates Figure 4: Downstream Anatomical Validation (DSC Bar Chart)."""
    print("Generating Figure 4: Anatomical Validation (SynthSeg DSC)...")

    csv_path = os.path.join(results_dir, 'anatomical_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Did you run the 'dsc' mode?")
        return

    df = pd.read_csv(csv_path)

    # Double column width = 6.875 inches
    fig, ax = plt.subplots(figsize=(6.875, 2.5))

    # Grouped bar chart
    sns.barplot(
        data=df,
        x='Region',
        y='DSC',
        hue='Ablation',
        palette=PALETTE,
        errorbar='ci', # shows 95% confidence interval
        capsize=0.05,
        errwidth=1.0,
        ax=ax
    )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Anatomical Macro-Region')
    ax.set_ylabel(r'Dice Similarity Coefficient (DSC) $\uparrow$')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Fix legend
    handles, labels = ax.get_legend_handles_labels()
    safe_labels =[LABELS.get(l, l) for l in labels]
    ax.legend(handles, safe_labels, title="", loc='upper left',
              fontsize=7, framealpha=0.85, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'Fig4_Anatomical_DSC.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved Fig4_Anatomical_DSC.pdf")


def generate_figure_5_frechet_distances():
    """Generates Figure 5: 3D-FID and FVD Metrics."""
    print("Generating Figure 5: 3D-FID and FVD...")

    csv_path = os.path.join(results_dir, 'frechet_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Did you run the 'frechet' mode?")
        return

    df = pd.read_csv(csv_path)
    
    # Double column width = 6.875 inches
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.5),
                             gridspec_kw={'wspace': 0.35})

    sns.barplot(
        data=df, x='Ablation', y='3D-FID', palette=PALETTE, ax=axes[0]
    )
    axes[0].set_xlabel('')
    axes[0].set_ylabel(r'3D-FID $\downarrow$')
    axes[0].set_title('3D Structural Realism', fontsize=9)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[0].set_xticklabels([LABELS.get(l.get_text(), l.get_text())
                             for l in axes[0].get_xticklabels()], 
                            rotation=45, ha='right', fontsize=7)

    # Plot 2: FVD
    sns.barplot(
        data=df, x='Ablation', y='FVD', palette=PALETTE, ax=axes[1]
    )
    axes[1].set_xlabel('')
    axes[1].set_ylabel(r'Fréchet Video Distance (FVD) $\downarrow$')
    axes[1].set_title('Z-Axis Sequential Continuity', fontsize=9)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[1].set_xticklabels([LABELS.get(l.get_text(), l.get_text())
                             for l in axes[1].get_xticklabels()], 
                            rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    # plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(figures_dir, 'Fig5_Frechet_Distances.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved Fig5_Frechet_Distances.pdf")


def generate_figure_6_pareto(combined=True):
    """Generates Figure 6: Pareto Fronts (Efficiency vs Fidelity)."""
    print("Generating Figure 6: Pareto Fronts...")

    perf_csv = os.path.join(results_dir, 'inference_performance.csv')
    fid_csv = os.path.join(results_dir, 'frechet_metrics.csv')
    dsc_csv = os.path.join(results_dir, 'anatomical_metrics.csv')
    abl_csv = os.path.join(results_dir, 'ablation_metrics.csv')

    if not os.path.exists(perf_csv):
        print("Error: Missing performance CSV.")
        return

    # 1. Load and aggregate Performance Data (Mean Time and VRAM per model)
    df_perf = pd.read_csv(perf_csv)
    df_master = df_perf.groupby('Model').agg({
        'Time_Seconds': 'mean',
        'Peak_VRAM_GB': 'mean'
    }).reset_index()

    # 2. Load and aggregate Fréchet Data
    if os.path.exists(fid_csv):
        df_fid = pd.read_csv(fid_csv).groupby('Ablation').mean(numeric_only=True).reset_index()
        df_master = pd.merge(df_master, df_fid, left_on='Model',
                             right_on='Ablation', how='left').drop(columns=['Ablation'])

    # 3. Load and aggregate Anatomical DSC Data
    if os.path.exists(dsc_csv):
        df_dsc = pd.read_csv(dsc_csv).groupby('Ablation').agg({'DSC': 'mean'}).reset_index()
        df_master = pd.merge(df_master, df_dsc, left_on='Model',
                             right_on='Ablation', how='left').drop(columns=['Ablation'])

    # 4. Load and aggregate Spatial Metrics (SSIM, MAE)
    if os.path.exists(abl_csv):
        df_abl = pd.read_csv(abl_csv).groupby('Ablation').agg({'SSIM': 'mean',
                                                               'MAE': 'mean'}).reset_index()
        df_master = pd.merge(df_master, df_abl, left_on='Model',
                             right_on='Ablation', how='left').drop(columns=['Ablation'])

    # Filter to only include the SOTA comparison track
    sota_models =['ours_full_masked', 'monai_3d_ldm', 'monai_2d_ldm', 'monai_vqgan']
    df_merged = df_master[df_master['Model'].isin(sota_models)].copy()

    if df_merged.empty:
        print("ERROR: df_merged is empty! Check if the Model names in inference_performance.csv match Ablation names.")
        return

    # Update palette to include MONAI
    local_palette = PALETTE.copy()
    # local_palette['monai_3d_ldm'] = '#8E44AD' # Purple for SOTA

    local_labels = LABELS.copy()
    # local_labels['monai_3d_ldm'] = 'MONAI 3D LDM (SOTA)'

    # Define the metrics we want to plot and their Y-axis labels
    metrics_to_plot =[
        ('DSC', r'Mean Anatomical DSC $\uparrow$'),
        ('3D-FID', r'3D-FID (Structural Error) $\downarrow$'),
        ('FVD', r'Fréchet Video Distance (FVD) $\downarrow$'),
        # ('SSIM', r'Structural Similarity (SSIM) $\uparrow$'),
        # ('MAE', r'Mean Absolute Error (MAE) $\downarrow$')
    ]

    available_metrics = [(col, lbl) for col, lbl in metrics_to_plot
                         if col in df_merged.columns]
    # logger.debug(f"merged dataframe columns = {df_merged.columns}")
    # logger.debug(f"available metrics = {available_metrics}")

    if combined and available_metrics:
        n_rows = len(available_metrics)
        # fig, axes = plt.subplots(n_rows, 2, figsize=(6.875, 2.5 * n_rows),
        #                          gridspec_kw={'wspace': 0.35, 'hspace': 0.4})
        # FIX: Increased height multiplier from 2.5 to 3.2 for square aspect ratio
        fig, axes = plt.subplots(n_rows, 2, figsize=(6.875, 3.25 * n_rows),
                                 gridspec_kw={'wspace': 0.35, 'hspace': 0.4})
        if n_rows == 1: axes = np.expand_dims(axes, axis=0)

        for i, (metric_col, metric_label) in enumerate(available_metrics):
            df_plot = df_merged.dropna(subset=[metric_col])
            
            # Plot 1: Time vs Metric
            sns.scatterplot(
                data=df_plot, x='Time_Seconds', y=metric_col, hue='Model', 
                palette=local_palette, s=150, edgecolor='black',
                ax=axes[i, 0], legend=False
            )
            axes[i, 0].set_xlabel(r'Inference Time per Volume (Seconds) $\downarrow$')
            axes[i, 0].set_ylabel(metric_label)
            if i == 0: axes[i, 0].set_title('Compute Time vs. Fidelity', fontsize=9)
            axes[i, 0].grid(True, linestyle='--', alpha=0.6)

            # Plot 2: VRAM vs Metric
            sns.scatterplot(
                data=df_plot, x='Peak_VRAM_GB', y=metric_col, hue='Model', 
                palette=local_palette, s=150, edgecolor='black', ax=axes[i, 1]
            )
            axes[i, 1].set_xlabel(r'Peak VRAM (GB) $\downarrow$')
            axes[i, 1].set_ylabel(metric_label)
            if i == 0: axes[i, 1].set_title('Memory Footprint vs. Fidelity', fontsize=9)
            axes[i, 1].grid(True, linestyle='--', alpha=0.6)

            # Remove individual legends
            if axes[i, 1].get_legend() is not None:
                axes[i, 1].legend().remove()

        # Add single legend at bottom
        handles, labels = (axes[0, 1].get_legend_handles_labels()
                           if axes[0, 1].get_legend_handles_labels()[0]
                           else axes[0, 0].get_legend_handles_labels())
        safe_labels =[local_labels.get(l, l) for l in labels]
        fig.legend(handles, safe_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                   fontsize=7, framealpha=0.85, ncol=2)

        plt.savefig(os.path.join(figures_dir, 'Fig6_Pareto_Combined.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  -> Saved Fig6_Pareto_Combined.pdf")
    else:
        # Fallback to individual plots if combined is False
        for metric_col, metric_label in available_metrics:
            # if metric_col not in df_merged.columns:
            #     continue
    
            # Drop models that don't have this specific metric calculated
            df_plot = df_merged.dropna(subset=[metric_col])
            # if df_plot.empty:
            #     continue

            # Double column width, increased wspace to prevent Y-axis overlap
            fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.75),
                                     gridspec_kw={'wspace': 0.35})
            # FIX: Increased height multiplier from 2.75 to 3.2 for square aspect ratio
            # fig, axes = plt.subplots(n_rows, 2, figsize=(6.875, 3.2),
            #                          gridspec_kw={'wspace': 0.35, 'hspace': 0.4})

            # Plot 1: Time vs Metric
            sns.scatterplot(data=df_plot, x='Time_Seconds',
                            y=metric_col, hue='Model', palette=local_palette,
                            s=150, edgecolor='black', ax=axes[0], legend=False)
            axes[0].set_xlabel(r'Inference Time per Volume (Seconds) $\downarrow$');
            axes[0].set_ylabel(metric_label);
            axes[0].set_title(f'Compute Time vs. {metric_col}', fontsize=9);
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # Plot 2: VRAM vs Metric
            sns.scatterplot(data=df_plot, x='Peak_VRAM_GB',
                            y=metric_col, hue='Model', palette=local_palette,
                            s=150, edgecolor='black', ax=axes[1])
            axes[1].set_xlabel(r'Peak VRAM (GB) $\downarrow$');
            axes[1].set_ylabel(metric_label);
            axes[1].set_title(f'Memory Footprint vs. {metric_col}', fontsize=9);
            axes[1].grid(True, linestyle='--', alpha=0.6)

            # Remove the default legend from the second axis
            handles, labels = axes[1].get_legend_handles_labels()
            axes[1].legend().remove()

            # Create a single, centered legend BELOW the plots
            safe_labels =[local_labels.get(l, l) for l in labels]
            fig.legend(handles, safe_labels, loc='lower center',
                       bbox_to_anchor=(0.5, -0.15), fontsize=7,
                       framealpha=0.85, ncol=2)

            # Use pad to ensure the figure boundaries respect the external legend
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'Fig6_Pareto_{metric_col}.pdf'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved Fig6_Pareto_{metric_col}.pdf")


def generate_supp_bidirectional():
    print("Generating Supp: Bidirectional Inpainting...")

    # Check both weights_dir and results_dir for epoch weights
    weight_files = glob.glob(os.path.join(weights_dir, '..', '*-???e.keras'))

    if not weight_files:
        weight_files = glob.glob(os.path.join(results_dir, '..', '*-???e.keras'))

    if not weight_files:
        print("  -> No epoch specific weights found. Skipping Supp Bidirectional.")
        return

    brats_list = get_brats_subjects(CFG.data.data_root_brats)
    if not brats_list: return
    test_brats = brats_list[0]

    vol_obj = VolumeLoader.load(test_brats['t1'], test_brats['seg'])
    if not vol_obj: return
    t1 = vol_obj.t1

    # Fallback ID parsing to fix the "None" title bug
    vol_id = test_brats.get('id')
    if not vol_id:
        vol_id = os.path.basename(test_brats['t1']).replace('_t1.nii.gz', '').replace('.nii', '')

    num_vis_slices = 5
    center = t1.shape[0] // 2
    start = center - num_vis_slices * 3
    end = center + num_vis_slices * 3
    vis_indices = np.linspace(start, end-1, num=num_vis_slices, dtype=int)

    model = ModelBuilder.build(CFG)
    reconstructor = VolumeReconstructor(model, CFG)
    reconstructor.cfg.batch_mode = True # silence TQDM

    for weight_path in sorted(weight_files):
        epoch_id = os.path.basename(weight_path).replace('.keras', '')
        print(f"  -> Processing {epoch_id}...")

        try:
            model.load_weights(weight_path)
        except:
            print(f"FAILED to load weights: {os.path.abspath(weight_path)}...")
            continue

        recon_fwd = reconstructor.autoregressive_restore(t1, start, end, 'forward')
        recon_bwd = reconstructor.autoregressive_restore(t1, start, end, 'backward')

        # Double Column width = 6.875 inches
        fig, axes = plt.subplots(num_vis_slices, 5, figsize=(6.875, 1.2 * num_vis_slices),
                                 gridspec_kw={'wspace': 0.02, 'hspace': 0.05})

        safe_title = f"Bidirectional Autoregressive Inpainting: {vol_id} ({epoch_id})"#.replace('_', r'\_')
        fig.suptitle(safe_title, fontsize=11, weight='bold', y=0.98)

        cols =["Ground Truth", "Forward", "Backward", "Fwd Diff", "Bwd Diff"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col, fontsize=9)

        for i, idx in enumerate(vis_indices):
            gt_slice = t1[idx]
            axes[i, 0].imshow(gt_slice, cmap='bone', vmin=0, vmax=1)
            axes[i, 0].set_ylabel(f"Slice {idx}", fontsize=9)

            axes[i, 1].imshow(recon_fwd[idx], cmap='bone', vmin=0, vmax=1)
            axes[i, 2].imshow(recon_bwd[idx], cmap='bone', vmin=0, vmax=1)
            axes[i, 3].imshow(np.abs(gt_slice - recon_fwd[idx]), cmap='inferno', vmin=0, vmax=0.3)
            axes[i, 4].imshow(np.abs(gt_slice - recon_bwd[idx]), cmap='inferno', vmin=0, vmax=0.3)

            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])

        # Note: dpi=150 is used to keep supplementary PDF file size low for Overleaf
        plt.savefig(os.path.join(figures_dir, f'Supp_Bidirectional_{epoch_id}.pdf'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def generate_supp_ablation_masked_ar():
    print("Generating Supp: Ablation Masked AR...")
    ixi_files = find_t1_files(CFG.data.data_root_ixi)
    if not ixi_files: return

    vol_obj = VolumeLoader.load(ixi_files[0])
    if not vol_obj: return

    vol = vol_obj.t1
    vol_id = os.path.basename(ixi_files[0]).replace('.nii.gz', '').replace('.nii', '')

    # Generate synthetic Glioma
    # synthetic_mask = VisualizationSuite._generate_synthetic_mask(vol, max_radius=18)

    # Generate realistic Glioma from BraTS pool
    # We need access to the BraTS pool. Since this function is called inside the script,
    # we can rebuild a quick static loader pool for BraTS, or pass it in.
    # For now, let's load a few BraTS files to sample from
    brats_list = get_brats_subjects(CFG.data.data_root_brats)
    brats_pool =[]
    for b in brats_list[:10]: # load 10 for variety
        v = VolumeLoader.load(b['t1'], b['seg'])
        if v: brats_pool.append(v)

    center = vol.shape[0] // 2
    # FORCE TUMOR TO CENTER
    synthetic_mask = VisualizationSuite._sample_real_brats_mask(vol, brats_pool, target_z=center)
    # synthetic_mask = VisualizationSuite._sample_real_brats_mask(vol, brats_pool)

    is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive'

    rollout_span = 10 if is_interactive else 30
    start = max(0, center - rollout_span)
    end = min(vol.shape[0], center + rollout_span)

    configs = get_ablation_configs(CFG)

    for ab_name, cfg in configs.items():
        # weight_path = os.path.join(ABLATION_DIR, f'model_{ab_name}.keras')
        weight_path = os.path.join(weights_dir, f'model_{ab_name}.keras')
        if not os.path.exists(weight_path):
            weight_path = os.path.join(results_dir, f'model_{ab_name}.keras')

        if not os.path.exists(weight_path): continue
        print(f"  -> Processing Masked AR for {os.path.abspath(ab_name)}...")

        model = ModelBuilder.build(cfg)
        # model.load_weights(weight_path)
        try:
            model.load_weights(weight_path)
        except:
            print(f"FAILED to load weights: {weight_path}...")
            continue

        reconstructor = VolumeReconstructor(model, cfg)
        reconstructor.cfg.batch_mode = True # silence TQDM

        recon_fwd = reconstructor.autoregressive_restore(
            vol, start, end, 'forward', mask_volume=synthetic_mask
        )
        recon_bwd = reconstructor.autoregressive_restore(
            vol, start, end, 'backward', mask_volume=synthetic_mask
        )

        # 1. Error Curves (Double column)
        fwd_mse = np.mean(np.square(vol[start:end] - recon_fwd[start:end]), axis=(1, 2))
        bwd_mse = np.mean(np.square(vol[start:end] - recon_bwd[start:end]), axis=(1, 2))

        fig, ax = plt.subplots(figsize=(6.875, 3.0))
        ax.plot(np.arange(start, end), bwd_mse, 'g-o', label='Backward (<-)', markersize=3)
        ax.plot(np.arange(start, end), fwd_mse, 'r-o', label='Forward (->)', markersize=3)
        # ax.axvline(x=center, color='b', linestyle='--', label='Start (Center Slice)')
        ax.axvline(x=center, color='b', linestyle='--', label='Tumor Center')

        safe_ab = LABELS.get(ab_name, ab_name).replace('_', r'\_')
        safe_vol_id = vol_id.replace('_', r'\_') 

        ax.set_title(f"Masked Error Accumulation [{safe_ab}] - {safe_vol_id}")
        ax.set_xlabel("Slice Index")
        ax.set_ylabel("Mean Squared Error")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figures_dir, f'Supp_Masked_Curve_{ab_name}.pdf'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Reconstruction Details (Fixing the Anchor BUG)
        idx_show =[
            start + 2,                      # Near start
            center - 1,                     # Just before center
            center,                         # Center (First predicted slice going forward)
            center + 1,                     # Just after center
            end - 3                         # Near end
        ]

        N = cfg.data.neighborhood
        cols = N + 4
        rows = len(idx_show)

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(6.875, 1.2 * rows),
                                 gridspec_kw={'wspace': 0.02,
                                              'hspace': 0.05})
        fig.suptitle(f"Reconstruction Details [{safe_ab}]",
                     fontsize=11, weight='bold', y=0.98)

        for i, idx in enumerate(idx_show):
            idx = int(max(0, min(idx, vol.shape[0] - 1)))

            # If it's the center, it IS a predicted slice in the forward pass
            if idx < center:
                img_r = recon_bwd[idx]
                context_vol = recon_bwd
                ctx_indices =[idx + N - k for k in range(N)]
            else:
                # Includes idx == center
                img_r = recon_fwd[idx]
                context_vol = recon_fwd
                ctx_indices =[idx - N + k for k in range(N)]

            # Context
            for j in range(N):
                c_idx = ctx_indices[j]
                ax = axes[i, j]
                if 0 <= c_idx < vol.shape[0]:
                    ctx_slice = context_vol[c_idx].copy()
                    # Apply the void visually so the plot reflects what the model actually saw
                    if synthetic_mask is not None:
                        m = (synthetic_mask[c_idx] > 0).astype(np.float32)
                        ctx_slice *= (1.0 - m)
                    ax.imshow(ctx_slice, cmap='bone', vmin=0, vmax=1)
                    if i == 0: ax.set_title(f"Ctx {j + 1}", fontsize=8)
                ax.axis('off')

            # Mask
            mask_slice = (synthetic_mask[idx] > 0.01).astype(np.float32)
            axes[i, N].imshow(mask_slice, cmap='gray')
            if i == 0: axes[i, N].set_title("Mask", fontsize=8)
            axes[i, N].axis('off')

            # GT
            axes[i, N + 1].imshow(vol[idx], cmap='bone', vmin=0, vmax=1)
            if i == 0: axes[i, N+1].set_title("GT", fontsize=8)
            axes[i, N + 1].axis('off')

            # Pred
            axes[i, N + 2].imshow(img_r, cmap='bone', vmin=0, vmax=1)
            if i == 0: axes[i, N+2].set_title("Pred", fontsize=8)
            axes[i, N + 2].axis('off')

            # Diff
            mse = np.mean((vol[idx] - img_r)**2)
            axes[i, N + 3].imshow(np.abs(vol[idx] - img_r), cmap='inferno',
                                  vmin=0, vmax=0.2)
            if i == 0: axes[i, N+3].set_title("Error", fontsize=8)
            axes[i, N + 3].axis('off')

            # Row label
            label = "Center (Pred)" if idx == center else f"Slice {idx}"
            axes[i, 0].text(-0.1, 0.5, label, va='center', ha='right', rotation=90,
                            transform=axes[i, 0].transAxes, fontsize=8)

        plt.savefig(os.path.join(figures_dir, f'Supp_Masked_Details_{ab_name}.pdf'),
                    dpi=150, bbox_inches='tight')
        plt.close()


import copy


def generate_supp_scaling_failure(render_errors=True):
    print("Generating Supp: Scaling Failure (B0 vs B1)...")

    # Resolve paths to the two different ablation directories
    b0_weights = os.path.abspath(os.path.join(weights_dir, '..', 'ablations-B0', 'model_full.keras'))
    b1_weights = os.path.abspath(os.path.join(weights_dir, '..', 'ablations-B1', 'model_full.keras'))

    if not os.path.exists(b0_weights) or not os.path.exists(b1_weights):
        print(f"  -> B0 or B1 weights not found. Skipping scaling failure plot.\n"
              f"     Checked: {b0_weights}\n     and: {b1_weights}")
        return

    # brats_list = get_brats_subjects(CFG.data.data_root_brats)
    # if not brats_list: return
    # test_brats = brats_list[0]

    # Use IXI dataset for pure healthy tissue drift evaluation
    ixi_files = find_t1_files(CFG.data.data_root_ixi)
    if not ixi_files: return
    # test_ixi = ixi_files[0]

    is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive'
    num_eval_vols = 5 if is_interactive else 10

    # vol_obj = VolumeLoader.load(test_brats['t1'], test_brats['seg'])
    for vol_idx, test_ixi in enumerate(ixi_files[:num_eval_vols]):
        vol_obj = VolumeLoader.load(test_ixi)
        if not vol_obj: continue
        vol = vol_obj.t1
        # vol_id = test_brats.get('id', 'Unknown')
        vol_id = os.path.basename(test_ixi).replace('.nii.gz', '').replace('.nii', '')
    
        center = vol.shape[0] // 2
        rollout_span = 10
        start = max(0, center - rollout_span)
        end = min(vol.shape[0], center + rollout_span)
    
        # Setup B0 Reconstructor
        cfg_b0 = copy.deepcopy(CFG)
        cfg_b0.model.backbone = 'B0'
        model_b0 = ModelBuilder.build(cfg_b0)
        model_b0.load_weights(b0_weights)
        recon_b0 = VolumeReconstructor(model_b0, cfg_b0)
        recon_b0.cfg.batch_mode = True
    
        # Setup B1 Reconstructor
        cfg_b1 = copy.deepcopy(CFG)
        cfg_b1.model.backbone = 'B1'
        model_b1 = ModelBuilder.build(cfg_b1)
        model_b1.load_weights(b1_weights)
        recon_b1 = VolumeReconstructor(model_b1, cfg_b1)
        recon_b1.cfg.batch_mode = True

        print("  -> Running B0 and B1 inference on CPU...")
        # Force execution on CPU to avoid allocating VRAM during plotting phase
        with tf.device('/CPU:0'):
            pred_b0 = recon_b0.autoregressive_restore(vol, start, end, 'forward')
            pred_b1 = recon_b1.autoregressive_restore(vol, start, end, 'forward')

        # Plotting logic
        num_vis_slices = 5
        vis_indices = np.linspace(start + 2, end - 2, num=num_vis_slices, dtype=int)

        # fig, axes = plt.subplots(num_vis_slices, 5, figsize=(6.875, 1.2 * num_vis_slices),
        #                          gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
        # FIX: Single CVPR column width (3.25 inches). Tighter layout
        num_cols = 3 + 2 * int(render_errors)
        fig, axes = plt.subplots(num_vis_slices, num_cols,
                                 figsize=(3.25, 0.75 * num_vis_slices), 
                                 gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

        # fig.suptitle(f"Scaling Failure Mode (B0 vs B1): {vol_id}",
        #              fontsize=11, weight='bold', y=0.98)
        safe_vol_id = vol_id.replace('_', r'\_')
        fig.suptitle(f"Scaling Failure (B0 vs B1):\n{safe_vol_id}",
                     fontsize=8, weight='bold', y=1.0)

        # cols = ["Ground Truth", "B0 Pred", "B1 Pred", "B0 Error", "B1 Error"]
        # Abbreviated columns to fit 3.25 inches
        cols = ["GT", "B0", "B1", "|B0-GT|", "|B1-GT|"]
        for ax, col in zip(axes[0], cols[:num_cols]):
            ax.set_title(col, fontsize=9)
    
        for i, idx in enumerate(vis_indices):
            gt_slice = vol[idx]
            p_b0 = pred_b0[idx]
            p_b1 = pred_b1[idx]
    
            axes[i, 0].imshow(gt_slice, cmap='bone', vmin=0, vmax=1)
            axes[i, 0].set_ylabel(f"Slice {idx}", fontsize=9)
    
            axes[i, 1].imshow(p_b0, cmap='bone', vmin=0, vmax=1)
            axes[i, 2].imshow(p_b1, cmap='bone', vmin=0, vmax=1)
    
            if render_errors:
                axes[i, 3].imshow(np.abs(gt_slice - p_b0), cmap='inferno', vmin=0, vmax=0.3)
                axes[i, 4].imshow(np.abs(gt_slice - p_b1), cmap='inferno', vmin=0, vmax=0.3)
    
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
    
        plt.savefig(os.path.join(figures_dir, f'Supp_Scaling_Failure_vol{vol_idx + 1:03d}.pdf'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved Supp_Scaling_Failure_vol{vol_idx + 1:03d}.pdf")


def run_cvpr_rendering(render_supp=True, render_bidirectional=False):
    print("--- Starting CVPR Visualization Generation ---")
    generate_figure_1_drift_curve()
    generate_figure_2_evolution_matrix()
    generate_figure_3_pe_proof()
    generate_figure_4_anatomical_dsc()
    generate_figure_5_frechet_distances()
    generate_figure_6_pareto()
    if render_supp:
        if render_bidirectional:
            generate_supp_bidirectional()
        generate_supp_ablation_masked_ar()
        generate_supp_scaling_failure(render_errors=True)
    else:
        print("Skipping Supplementary Figures to save time (render_supp=False).")
    print("--- All Figures Generated Successfully! ---")
