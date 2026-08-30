import os
import glob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def configure_xla_paths():
    """
    Locates libdevice.10.bc in the Conda environment and sets XLA_FLAGS.
    Fixes 'libdevice not found' errors in Conda environments.
    """
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        return # Not in conda, assume system paths work

    # Common locations for libdevice in Conda envs
    # It often moves around depending on nvcc/cudatoolkit version
    candidates = [
        f"{conda_prefix}/lib/libdevice.10.bc",
        f"{conda_prefix}/lib/nvidia/cuda_nvvm/libdevice/libdevice.10.bc",
        f"{conda_prefix}/nvvm/libdevice/libdevice.10.bc"
    ]

    # Recursive search if standard paths fail (slower but robust)
    if not any(os.path.exists(p) for p in candidates):
        found = glob.glob(f"{conda_prefix}/**/libdevice.10.bc", recursive=True)
        if found:
            candidates = found

    for path in candidates:
        if os.path.exists(path):
            cuda_dir = os.path.dirname(os.path.dirname(os.path.dirname(path))) # usually goes up to where 'bin' and 'lib' are

            # Often XLA wants the root of the CUDA installation (where /nvvm/libdevice lives)
            # Strategy: Point --xla_gpu_cuda_data_dir to the folder containing 'nvvm' or 'lib/nvidia'

            # Simplest for TF 2.x in Conda:
            # Point to the specific directory containing the libdevice file is not supported by the flag directly,
            # it wants the CUDA_DIR.

            # BUT, we can just disable XLA if we can't find it, OR simply set the flag to the conda prefix
            # which usually works if the layout is standard.

            print(f"✅ Found libdevice at: {path}")

            # This flag tells XLA where to look for CUDA libraries
            os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={conda_prefix}"

            # Sometimes LD_LIBRARY_PATH needs help too
            lib_path = os.environ.get('LD_LIBRARY_PATH', '')
            if f"{conda_prefix}/lib" not in lib_path:
                os.environ['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:{lib_path}"

            return

    print("⚠️ WARNING: libdevice.10.bc not found in Conda environment. XLA might fail.")
    # If not found, safer to disable XLA to prevent crash
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'


KAGGLE = bool(os.environ.get('KAGGLE_URL_BASE'))

if KAGGLE:
    print(f"Kaggle has XLA paths configured.")

if KAGGLE:
    PATH_DATA_IXI = '/kaggle/input/preprocessed-oasis-and-epilepsy-and-ixi'
    PATH_DATA_BRATS = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
else:
    PATH_DATA_IXI = 'data/ixi'
    PATH_DATA_BRATS = 'data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
