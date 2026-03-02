"""
MLOmics Baseline Training Script
===================================
Trains early-fusion baselines (XGBoost and/or RandomForest) with 5-fold CV.

Usage:
    python scripts/train_baselines.py                    # Both models, full data
    python scripts/train_baselines.py --model xgb        # XGBoost only
    python scripts/train_baselines.py --model rf         # RandomForest only
    python scripts/train_baselines.py --model rf --toy   # RF on toy data

Artifacts produced:
    models/baseline_{xgb,rf}/{model}_{cancer}_fold{i}.pkl
    results/metrics/{model}_{cancer}_metrics.csv
    results/plots/confusion_matrix_{model}_{cancer}.png
    experiment_log.csv entries
"""

import argparse
import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    compute_fold_summary,
    compute_metrics,
    print_metrics_table,
    save_metrics,
)
from src.preprocessing import concatenate_modalities, load_cv_folds, prepare_fold_data
from src.utils import load_config, log_experiment, set_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """Compute per-sample weights for class imbalance (used by XGBoost)."""
    return compute_sample_weight("balanced", y_train)


def plot_best_fold_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cancer_type: str,
    model_name: str,
    save_dir: str,
    n_classes: int,
) -> str:
    """Save a confusion matrix heatmap (300 DPI) and return the file path."""
    os.makedirs(save_dir, exist_ok=True)
    cancer_short = cancer_type.split("-")[1]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Class {i}" for i in range(n_classes)],
        yticklabels=[f"Class {i}" for i in range(n_classes)],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix: {model_name.upper()} ({cancer_type}) - Best Fold",
        fontsize=13,
    )

    png_path = os.path.join(
        save_dir, f"confusion_matrix_{model_name.lower()}_{cancer_short}.png"
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Confusion matrix saved to {png_path}")
    print(f"Confusion matrix saved: {png_path}")
    return png_path


def _run_leakage_check(fold_metrics_list: list) -> None:
    """Print warnings if mean F1 is outside the expected range."""
    summary = compute_fold_summary(fold_metrics_list)
    if summary["f1_mean"] > 0.95:
        print("\n!!! WARNING: F1 > 0.95 -- SUSPECT DATA LEAKAGE !!!")
        print("Re-run label shuffle test before proceeding.")
    elif summary["f1_mean"] < 0.40:
        print("\n!!! WARNING: F1 < 0.40 -- Check data alignment / preprocessing !!!")


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost_cv(
    cancer_type: str,
    config: dict,
    use_toy: bool = False,
) -> None:
    """Train XGBoost early-fusion baseline with 5-fold CV."""
    cancer_short = cancer_type.split("-")[1].lower()
    tag = "toy" if use_toy else "full"
    print(f"\n{'='*60}")
    print(f"XGBoost 5-Fold CV: {cancer_type} ({tag} data)")
    print(f"{'='*60}")

    cv_data = load_cv_folds(cancer_type, config, use_toy=use_toy)
    folds = cv_data["folds"]
    n_folds = cv_data["n_folds"]
    xgb_config = config["baselines"]["xgboost"]

    fold_metrics_list = []
    best_fold_idx = -1
    best_f1 = -1.0
    best_fold_y_true = None
    best_fold_y_pred = None
    n_classes = None

    model_dir = os.path.join(config["paths"]["models"], "baseline_xgb")
    os.makedirs(model_dir, exist_ok=True)

    for fold_info in folds:
        fold_idx = fold_info["fold"]
        print(f"\n--- Fold {fold_idx}/{n_folds - 1} ---")

        result = prepare_fold_data(cancer_type, fold_info, config, use_toy=use_toy)

        X_train_concat, feature_names = concatenate_modalities(result["X_train"])
        X_val_concat, _ = concatenate_modalities(result["X_val"])
        y_train = result["y_train"]
        y_val = result["y_val"]

        if n_classes is None:
            n_classes = len(np.unique(np.concatenate([y_train, y_val])))

        print(f"  Train: {X_train_concat.shape}, Val: {X_val_concat.shape}")
        print(f"  Classes in train: {np.unique(y_train, return_counts=True)}")

        sample_weights = compute_class_weights(y_train)

        model = XGBClassifier(
            n_estimators=xgb_config["n_estimators"],
            max_depth=xgb_config["max_depth"],
            learning_rate=xgb_config["learning_rate"],
            random_state=xgb_config["random_state"],
            objective="multi:softmax",
            num_class=n_classes,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )

        model.fit(
            X_train_concat,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val_concat, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val_concat)
        metrics = compute_metrics(y_val, y_pred)
        fold_metrics_list.append(metrics)

        print(
            f"  Fold {fold_idx}: F1={metrics['f1']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_fold_idx = fold_idx
            best_fold_y_true = y_val.copy()
            best_fold_y_pred = y_pred.copy()

        model_path = os.path.join(model_dir, f"xgb_{cancer_short}_fold{fold_idx}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")

        log_experiment(
            config_path=config["paths"]["experiment_log"],
            experiment_name=f"xgb_baseline_{cancer_short}",
            cancer_type=cancer_type,
            model_type="XGBoost",
            features=f"early_fusion_{X_train_concat.shape[1]}",
            n_folds=n_folds,
            fold=fold_idx,
            precision=round(metrics["precision"], 4),
            recall=round(metrics["recall"], 4),
            f1=round(metrics["f1"], 4),
            nmi=round(metrics["nmi"], 4),
            ari=round(metrics["ari"], 4),
            notes=f"n_est={xgb_config['n_estimators']}, max_d={xgb_config['max_depth']}, lr={xgb_config['learning_rate']}",
            artifact_path=model_path,
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"XGBoost {cancer_type} -- Summary")
    print(f"{'='*60}")
    print_metrics_table(fold_metrics_list, f"XGBoost ({cancer_type})")

    metrics_dir = os.path.join(config["paths"]["results"], "metrics")
    metrics_path = os.path.join(metrics_dir, f"xgb_{cancer_short}_metrics.csv")
    save_metrics(fold_metrics_list, "XGBoost", cancer_type, metrics_path)

    print(f"\nBest fold: {best_fold_idx} (F1={best_f1:.4f})")
    plots_dir = os.path.join(config["paths"]["results"], "plots")
    plot_best_fold_confusion_matrix(
        best_fold_y_true, best_fold_y_pred, cancer_type, "xgb", plots_dir, n_classes,
    )

    _run_leakage_check(fold_metrics_list)


# ---------------------------------------------------------------------------
# RandomForest training
# ---------------------------------------------------------------------------

def train_random_forest_cv(
    cancer_type: str,
    config: dict,
    use_toy: bool = False,
) -> None:
    """Train RandomForest early-fusion baseline with 5-fold CV.

    Uses class_weight='balanced' for imbalance handling (built-in to RF).
    Same CV folds as XGBoost for fair comparison.
    """
    cancer_short = cancer_type.split("-")[1].lower()
    tag = "toy" if use_toy else "full"
    print(f"\n{'='*60}")
    print(f"RandomForest 5-Fold CV: {cancer_type} ({tag} data)")
    print(f"{'='*60}")

    cv_data = load_cv_folds(cancer_type, config, use_toy=use_toy)
    folds = cv_data["folds"]
    n_folds = cv_data["n_folds"]
    rf_config = config["baselines"]["random_forest"]

    fold_metrics_list = []
    best_fold_idx = -1
    best_f1 = -1.0
    best_fold_y_true = None
    best_fold_y_pred = None
    n_classes = None

    model_dir = os.path.join(config["paths"]["models"], "baseline_rf")
    os.makedirs(model_dir, exist_ok=True)

    for fold_info in folds:
        fold_idx = fold_info["fold"]
        print(f"\n--- Fold {fold_idx}/{n_folds - 1} ---")

        result = prepare_fold_data(cancer_type, fold_info, config, use_toy=use_toy)

        X_train_concat, feature_names = concatenate_modalities(result["X_train"])
        X_val_concat, _ = concatenate_modalities(result["X_val"])
        y_train = result["y_train"]
        y_val = result["y_val"]

        if n_classes is None:
            n_classes = len(np.unique(np.concatenate([y_train, y_val])))

        print(f"  Train: {X_train_concat.shape}, Val: {X_val_concat.shape}")
        print(f"  Classes in train: {np.unique(y_train, return_counts=True)}")

        # RF handles class imbalance via class_weight='balanced'
        model = RandomForestClassifier(
            n_estimators=rf_config["n_estimators"],
            max_depth=rf_config["max_depth"],
            random_state=rf_config["random_state"],
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(X_train_concat, y_train)

        y_pred = model.predict(X_val_concat)
        metrics = compute_metrics(y_val, y_pred)
        fold_metrics_list.append(metrics)

        print(
            f"  Fold {fold_idx}: F1={metrics['f1']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_fold_idx = fold_idx
            best_fold_y_true = y_val.copy()
            best_fold_y_pred = y_pred.copy()

        model_path = os.path.join(model_dir, f"rf_{cancer_short}_fold{fold_idx}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")

        log_experiment(
            config_path=config["paths"]["experiment_log"],
            experiment_name=f"rf_baseline_{cancer_short}",
            cancer_type=cancer_type,
            model_type="RandomForest",
            features=f"early_fusion_{X_train_concat.shape[1]}",
            n_folds=n_folds,
            fold=fold_idx,
            precision=round(metrics["precision"], 4),
            recall=round(metrics["recall"], 4),
            f1=round(metrics["f1"], 4),
            nmi=round(metrics["nmi"], 4),
            ari=round(metrics["ari"], 4),
            notes=f"n_est={rf_config['n_estimators']}, max_d={rf_config['max_depth']}, class_weight=balanced",
            artifact_path=model_path,
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"RandomForest {cancer_type} -- Summary")
    print(f"{'='*60}")
    print_metrics_table(fold_metrics_list, f"RandomForest ({cancer_type})")

    metrics_dir = os.path.join(config["paths"]["results"], "metrics")
    metrics_path = os.path.join(metrics_dir, f"rf_{cancer_short}_metrics.csv")
    save_metrics(fold_metrics_list, "RandomForest", cancer_type, metrics_path)

    print(f"\nBest fold: {best_fold_idx} (F1={best_f1:.4f})")
    plots_dir = os.path.join(config["paths"]["results"], "plots")
    plot_best_fold_confusion_matrix(
        best_fold_y_true, best_fold_y_pred, cancer_type, "rf", plots_dir, n_classes,
    )

    _run_leakage_check(fold_metrics_list)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train baseline models (XGBoost / RandomForest)")
    parser.add_argument(
        "--model",
        choices=["xgb", "rf", "all"],
        default="all",
        help="Which baseline to train (default: all)",
    )
    parser.add_argument("--toy", action="store_true", help="Use toy data for testing")
    args = parser.parse_args()

    set_seeds(42)
    config = load_config()

    run_xgb = args.model in ("xgb", "all")
    run_rf = args.model in ("rf", "all")

    if args.toy:
        cancer_types = ["GS-BRCA"]  # toy data only has BRCA
        print("Running on TOY data (BRCA only)")
    else:
        cancer_types = config["project"]["cancer_types"]

    if run_xgb:
        for ct in cancer_types:
            train_xgboost_cv(ct, config, use_toy=args.toy)
        print("\nXGBoost baseline training complete.")

    if run_rf:
        for ct in cancer_types:
            train_random_forest_cv(ct, config, use_toy=args.toy)
        print("\nRandomForest baseline training complete.")


if __name__ == "__main__":
    main()
