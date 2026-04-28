"""
Convenience evaluation entrypoint.

This script mirrors the separation described in the commit strategy:

1. Load the saved per-fold prediction archives and print a consolidated
   metrics summary table.
2. Delegate to compute_auc.py to recompute macro OVR AUC outputs.

Use --auc-only to skip the summary step and run the AUC refresh only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RESULTS_DIR = os.path.join("results", "metrics")
PREDICTION_MODELS = ("xgb", "rf", "fusion", "pathway_fusion")
PREDICTION_CANCERS = ("brca", "coad")


def _prediction_path(model: str, cancer: str, fold: int) -> str:
    return os.path.join(RESULTS_DIR, f"{model}_{cancer}_fold{fold}_predictions.npz")


def _load_prediction_arrays(path: str):
    data = np.load(path)
    if "y_true" not in data or "y_pred" not in data:
        raise ValueError(f"Prediction archive missing required keys: {path}")
    return data["y_true"], data["y_pred"]


def _summarize_prediction_files() -> pd.DataFrame:
    rows: List[dict] = []
    for model in PREDICTION_MODELS:
        for cancer in PREDICTION_CANCERS:
            fold_paths = [
                _prediction_path(model, cancer, fold)
                for fold in range(5)
                if os.path.exists(_prediction_path(model, cancer, fold))
            ]
            if not fold_paths:
                continue

            fold_metrics = []
            for fold, path in enumerate(fold_paths):
                y_true, y_pred = _load_prediction_arrays(path)
                fold_metrics.append(
                    {
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "f1": float(
                            f1_score(y_true, y_pred, average="macro", zero_division=0)
                        ),
                        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
                        "ari": float(adjusted_rand_score(y_true, y_pred)),
                    }
                )

            df = pd.DataFrame(fold_metrics)
            rows.append(
                {
                    "model": model,
                    "cancer": cancer.upper()
                    .replace("BRCA", "GS-BRCA")
                    .replace("COAD", "GS-COAD"),
                    "accuracy_mean": float(df["accuracy"].mean()),
                    "accuracy_std": float(df["accuracy"].std()),
                    "f1_mean": float(df["f1"].mean()),
                    "f1_std": float(df["f1"].std()),
                    "nmi_mean": float(df["nmi"].mean()),
                    "nmi_std": float(df["nmi"].std()),
                    "ari_mean": float(df["ari"].mean()),
                    "ari_std": float(df["ari"].std()),
                }
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        print("\n=== Consolidated prediction summary ===")
        print(summary.to_string(index=False))
    else:
        print("No prediction archives found.")
    return summary


def _run_auc_script() -> None:
    script_path = os.path.join(os.path.dirname(__file__), "compute_auc.py")
    subprocess.run([sys.executable, script_path], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation summary and AUC refresh."
    )
    parser.add_argument(
        "--auc-only",
        action="store_true",
        help="Skip the summary table and run compute_auc.py only.",
    )
    args = parser.parse_args()

    if not args.auc_only:
        _summarize_prediction_files()

    _run_auc_script()


if __name__ == "__main__":
    main()
