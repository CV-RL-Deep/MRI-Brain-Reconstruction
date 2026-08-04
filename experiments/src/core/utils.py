import json
import logging
import os
import random
import sys
import time

import psutil
import subprocess
import csv

from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from platform import python_version
from typing import Tuple, List, Optional, Union, Any, Dict

import nibabel as nib
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy.ndimage import find_objects
from tensorflow.keras import mixed_precision

import torch

from src.core.env import KAGGLE


# --- Robust Logging Configuration ---
def setup_logger():
    logger = logging.getLogger("RefactoredPipeline")
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                      datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Also force Tensorflow to be quiet unless it's an error
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    return logger

logger = setup_logger()

# --- Profiling / Debugging Utilities ---

class PipelineTimer:
    """
    A context manager to measure execution time of code blocks.
    """
    def __init__(self, name: str, active: bool = True):
        self.name = name
        self.active = active

    def __enter__(self):
        if self.active:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            duration = (time.perf_counter() - self.start) # sec # * 1000  # ms
            if duration > 1.0:
                 logger.info(f"⏱️ {self.name}: {duration:.2f} sec") # changed debug->info to ensure visibility


class InferenceProfiler:
    """
    Tracks Execution Time and Peak VRAM during Inference for Pareto plotting.
    Supports both TensorFlow and PyTorch backends.
    """
    def __init__(self, model_name: str, results_dir: str):
        self.model_name = model_name
        self.results_dir = results_dir
        self.csv_path = os.path.join(results_dir, "inference_performance.csv")
        self.start_time = 0

        # Initialize CSV
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Model', 'Volume_ID', 'Time_Seconds', 'Peak_VRAM_GB'])

    def start(self):
        # Reset Peak Memory Stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
        except: pass

        self.start_time = time.perf_counter()

    def stop_and_log(self, vol_id: str):
        elapsed = time.perf_counter() - self.start_time

        # Gather Peak VRAM
        vram_gb = 0.0

        # Check PyTorch VRAM
        if torch.cuda.is_available():
            pt_vram = torch.cuda.max_memory_allocated() / (1024**3)
            vram_gb = max(vram_gb, pt_vram)

        # Check TensorFlow VRAM
        try:
            tf_info = tf.config.experimental.get_memory_info('GPU:0')
            tf_vram = tf_info['peak'] / (1024**3)
            vram_gb = max(vram_gb, tf_vram)
        except: pass

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.model_name, vol_id, f"{elapsed:.2f}", f"{vram_gb:.2f}"])

        logger.info(f"[{self.model_name}] {vol_id} -> Time: {elapsed:.1f}s | Peak VRAM: {vram_gb:.2f}GB")


class MemoryTelemetry:
    def __init__(self, filename="memory_telemetry.csv"):
        self.filename = filename
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Event', 'Trial', 'RAM', 'VRAM'])

    def log(self, event: str, trial_num: int):
        # 1. System RAM (GB)
        ram_gb = psutil.virtual_memory().used / (1024 ** 3)

        # 2. GPU VRAM (GB)
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            # Sum VRAM across all GPUs (if multiple exist)
            vram_mb = sum(int(x.strip()) for x in result.strip().split('\n'))
            vram_gb = vram_mb / 1024.0
        except Exception:
            vram_mb = -1 # fallback if nvidia-smi fails
            vram_gb = -1

        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([event, trial_num, f"{ram_gb:.2f}", vram_gb])


def enforce_gpu_presence():
    """Protects against silent fallback to CPU training which destroys productivity."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError(
            "🚨 FATAL: No GPUs detected by TensorFlow! "
            "Aborting execution to prevent a massive waste of time on CPU."
        )
    logger.info(f"✅ Hardware Protection Pass: Verified {len(gpus)} GPU(s) available.")


def set_global_seeds(seed: int = 42, use_fp16=False, use_dmem=False):
    """Sets random seeds for reproducibility across Python, Numpy, and TF."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Global seed set to {seed}")

    # Enable Mixed Precision for VRAM efficiency
    if use_fp16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    logger.info(f"Mixed Precision Policy = {mixed_precision.global_policy()}")

    # TITAN RTX Tensor cores workaround
    if use_dmem:
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                for gpu in gpus:
                    logger.info(f"Setting dynamic memory growth for {gpu}...")
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)


# Initialize environment
if not KAGGLE:
    enforce_gpu_presence()  # <--- DROP IN HERE
set_global_seeds(42)
logger.info(f"Process ID = {os.getpid()}")
logger.info(f"Python version = {python_version()}")

# Instantiate globally
telemetry = MemoryTelemetry()
