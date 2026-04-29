"""
MLOmics Evaluation Module
==========================
Shared evaluation metrics for all models (XGBoost, RF, Fusion).
Metrics: Precision, Recall, F1 (macro), NMI, ARI, Accuracy.

Used by training scripts and notebooks to ensure consistent
metric computation across the entire project.
"""

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Metric names in display order (primary metrics first, accuracy last)
METRIC_NAMES = ["precision", "recall", "f1", "nmi", "ari", "accuracy"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics for a single fold.

    Args:
        y_true: Ground truth labels (integer-encoded).
        y_pred: Predicted labels (integer-encoded).

    Returns:
        Dict with keys: precision, recall, f1, nmi, ari, accuracy.
        All values are floats in [0, 1].
    """
    metrics = {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    logger.info(
        f"Metrics — F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, NMI: {metrics['nmi']:.4f}, "
        f"ARI: {metrics['ari']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
    )
    return metrics


def compute_fold_summary(fold_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean ± std across fold metric dicts.

    Args:
        fold_metrics_list: List of dicts from compute_metrics(), one per fold.

    Returns:
        Dict with keys like 'precision_mean', 'precision_std', etc.
    """
    if not fold_metrics_list:
        raise ValueError("fold_metrics_list is empty — no folds to summarize")

    summary = {}
    for metric in METRIC_NAMES:
        values = np.array([fold[metric] for fold in fold_metrics_list])
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))

    return summary


def print_metrics_table(
    fold_metrics_list: List[Dict[str, float]],
    model_name: str,
) -> None:
    """Pretty-print a box table of mean +/- std metrics.

    Uses ASCII box-drawing characters for cross-platform compatibility
    (Windows cp1252 does not support Unicode box-drawing).

    Args:
        fold_metrics_list: List of per-fold metric dicts.
        model_name: Display name, e.g. "XGBoost (GS-BRCA)".
    """
    summary = compute_fold_summary(fold_metrics_list)

    col_metric_w = 12
    col_value_w = 20
    total_w = col_metric_w + 1 + col_value_w + 2

    display_names = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "nmi": "NMI",
        "ari": "ARI",
        "accuracy": "Accuracy",
    }

    print(f"+{'=' * total_w}+")
    print(f"| Model: {model_name:<{total_w - 9}}|")
    print(f"+{'=' * total_w}+")
    print(f"| {'Metric':<{col_metric_w}}| {'Mean +/- Std':<{col_value_w}}|")
    print(f"+{'-' * (col_metric_w + 1)}+{'-' * (col_value_w + 1)}+")

    for metric in METRIC_NAMES:
        mean_val = summary[f"{metric}_mean"]
        std_val = summary[f"{metric}_std"]
        label = display_names[metric]
        value_str = f"{mean_val:.4f} +/- {std_val:.4f}"
        print(f"| {label:<{col_metric_w}}| {value_str:<{col_value_w}}|")

    print(f"+{'=' * total_w}+")


def save_metrics(
    fold_metrics_list: List[Dict[str, float]],
    model_name: str,
    cancer_type: str,
    filepath: str,
) -> None:
    """Save per-fold metrics and summary to a CSV file.

    Creates parent directories if they don't exist.

    Args:
        fold_metrics_list: List of per-fold metric dicts.
        model_name: Model identifier, e.g. "XGBoost".
        cancer_type: Cancer type, e.g. "GS-BRCA".
        filepath: Output CSV path (relative or absolute).
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    rows = []
    for i, fold_metrics in enumerate(fold_metrics_list):
        row = {
            "model": model_name,
            "cancer_type": cancer_type,
            "fold": i,
            **{m: fold_metrics[m] for m in METRIC_NAMES},
        }
        rows.append(row)

    # Add summary row
    summary = compute_fold_summary(fold_metrics_list)
    summary_row = {
        "model": model_name,
        "cancer_type": cancer_type,
        "fold": "mean+/-std",
    }
    for metric in METRIC_NAMES:
        summary_row[metric] = f"{summary[f'{metric}_mean']:.4f}+/-{summary[f'{metric}_std']:.4f}"
    rows.append(summary_row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, float_format="%.4f")

    logger.info(f"Metrics saved to {filepath}")
    print(f"Metrics saved to {filepath}")
