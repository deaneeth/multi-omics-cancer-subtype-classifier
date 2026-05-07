"""
MLOmics Utility Module
======================
Core utility functions used across the entire project.
- set_seeds(): Reproducibility (MUST be called first in every script/notebook)
- load_config(): Load config.yaml (single source of truth for all hyperparams)
- log_experiment(): Append experiment results to experiment_log.csv
- get_device(): Detect and return the best available compute device
"""

import os
import csv
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import torch

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. set_seeds — Full reproducibility across all libraries
# ---------------------------------------------------------------------------
def set_seeds(seed: int = 42) -> None:
    """Set random seeds for full reproducibility.

    CRITICAL: Call this as the FIRST line after imports in every script/notebook.
    Sets PYTHONHASHSEED, random, numpy, torch (CPU + CUDA), and cuDNN flags.

    Args:
        seed: Random seed value. Default 42 per project convention.
    """
    # Python hash seed — must be set before any hashing occurs
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python stdlib
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU future-proofing

    # cuDNN determinism — sacrifices speed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"All random seeds set to {seed}")


# ---------------------------------------------------------------------------
# 2. load_config — Single source of truth for all hyperparameters
# ---------------------------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    """Load the project YAML configuration file.

    Args:
        path: Path to config.yaml (relative to project root or absolute).
              Default is 'config.yaml' in the current working directory.

    Returns:
        dict: Parsed configuration dictionary with all project settings.

    Raises:
        FileNotFoundError: If config.yaml does not exist at the given path.
        yaml.YAMLError: If the YAML file is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            f"Ensure you are running from the project root directory."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path.resolve()}")
    return config


# ---------------------------------------------------------------------------
# 3. log_experiment — Append results to experiment_log.csv
# ---------------------------------------------------------------------------
# Canonical CSV header order
EXPERIMENT_LOG_COLUMNS = [
    "timestamp",
    "experiment_name",
    "cancer_type",
    "model_type",
    "features",
    "n_folds",
    "fold",
    "precision",
    "recall",
    "f1",
    "nmi",
    "ari",
    "notes",
    "artifact_path",
]


def log_experiment(config_path: str = "experiment_log.csv", **kwargs) -> None:
    """Append one row to the experiment log CSV.

    Automatically adds a UTC timestamp. Any keyword argument whose name
    matches one of the canonical columns will be placed in the correct
    column; unrecognised keys are silently ignored.

    Args:
        config_path: Path to the experiment_log.csv file.
                     Default is 'experiment_log.csv' in the project root.
        **kwargs: Experiment metadata. Common keys include:
            experiment_name, cancer_type, model_type, features,
            n_folds, fold, precision, recall, f1, nmi, ari,
            notes, artifact_path.

    Example:
        >>> log_experiment(
        ...     experiment_name="xgb_baseline_brca",
        ...     cancer_type="GS-BRCA",
        ...     model_type="XGBoost",
        ...     f1=0.87,
        ...     notes="5-fold CV, Top features"
        ... )
    """
    log_path = Path(config_path)

    # Build the row dict — fill missing columns with empty strings
    row = {col: "" for col in EXPERIMENT_LOG_COLUMNS}
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for key, value in kwargs.items():
        if key in EXPERIMENT_LOG_COLUMNS:
            row[key] = value

    # Check if file exists and has content to decide on writing header
    file_exists = log_path.exists() and log_path.stat().st_size > 0

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(
        f"Experiment logged: {kwargs.get('experiment_name', 'unnamed')} "
        f"→ {log_path.resolve()}"
    )


# ---------------------------------------------------------------------------
# 4. get_device — Detect best available compute device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Returns torch.device('cuda') if a CUDA-capable GPU is available,
    otherwise torch.device('cpu'). Prints the selected device.

    Returns:
        torch.device: The selected compute device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Using device: cuda ({gpu_name}, {gpu_mem:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    logger.info(f"Compute device selected: {device}")
    return device


# ---------------------------------------------------------------------------
# 5. Shared training helpers
# ---------------------------------------------------------------------------

def compute_class_weights(y_train: np.ndarray, device: "torch.device") -> "torch.Tensor":
    """Return inverse-frequency class weights as a float32 tensor on device."""
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts.astype(np.float64)
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def dataframes_to_numpy(data_dict: dict) -> dict:
    """Convert {modality: DataFrame} to {modality: np.ndarray float32}."""
    return {name: df.values.astype(np.float32) for name, df in data_dict.items()}


def compute_kegg_coverage(pathway_mapping_path: str, cancer: str) -> dict:
    """Return KEGG pathway coverage stats for one cancer type.

    Reads the pre-computed `coverage_pct`, `n_mapped_features`, and
    `n_features` directly from the JSON artifact instead of recomputing,
    so the returned value always matches what the model actually used.

    Args:
        pathway_mapping_path: Path to pathway_gene_mapping.json.
        cancer: Cancer key in the JSON, e.g. 'BRCA' or 'COAD'.

    Returns:
        dict with keys: coverage_pct, n_mapped, n_total, n_pathways.
    """
    import json
    with open(pathway_mapping_path) as f:
        mapping = json.load(f)
    d = mapping[cancer.upper()]
    return {
        "coverage_pct": d["coverage_pct"],
        "n_mapped": d["n_mapped_features"],
        "n_total": d["n_features"],
        "n_pathways": d["n_pathways_with_hits"],
    }
