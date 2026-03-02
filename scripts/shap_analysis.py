"""
MLOmics SHAP Analysis Script
==============================
Computes SHAP values for the best tree-based model fold per cancer type
and generates thesis-quality plots + feature importance CSVs.

Usage:
    python scripts/shap_analysis.py                          # Both models, both cancers
    python scripts/shap_analysis.py --model rf               # RF only
    python scripts/shap_analysis.py --model xgb --cancer GS-BRCA
    python scripts/shap_analysis.py --model rf --toy         # RF on toy data

Artifacts produced (per model/cancer):
    results/shap/{model}_shap_summary_{cancer}.png / .pdf
    results/shap/{model}_shap_bar_{cancer}.png / .pdf
    results/shap/top_50_features_{model}_{cancer}.csv
    results/shap/{model}_shap_values_{cancer}.npz
"""

import argparse
import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import concatenate_modalities, load_cv_folds, prepare_fold_data
from src.utils import load_config, set_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    "xgb": {
        "display_name": "XGBoost",
        "log_model_type": "XGBoost",
        "model_subdir": "baseline_xgb",
        "shap_subdir": "xgb",
        "file_prefix": "xgb",
    },
    "rf": {
        "display_name": "RandomForest",
        "log_model_type": "RandomForest",
        "model_subdir": "baseline_rf",
        "shap_subdir": "rf",
        "file_prefix": "rf",
    },
}

MODALITY_COLORS = {
    "mrna": "#2196F3",
    "mirna": "#4CAF50",
    "methy": "#FF9800",
    "cnv": "#9C27B0",
}


def find_best_fold(cancer_type: str, model_type_label: str, config: dict) -> int:
    """Find the fold with highest F1 from the experiment log."""
    log_path = config["paths"]["experiment_log"]
    df = pd.read_csv(log_path)
    subset = df[
        (df["model_type"] == model_type_label) & (df["cancer_type"] == cancer_type)
    ]
    if subset.empty:
        raise ValueError(
            f"No {model_type_label} entries for {cancer_type} in {log_path}"
        )
    best_row = subset.loc[subset["f1"].idxmax()]
    fold = int(best_row["fold"])
    f1 = best_row["f1"]
    logger.info(
        f"Best {model_type_label} fold for {cancer_type}: fold {fold} (F1={f1:.4f})"
    )
    return fold


def save_figure(fig, base_path: str) -> None:
    """Save a matplotlib figure as 300-DPI PNG and vector PDF."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {base_path}.png and .pdf")


def run_shap_analysis(
    cancer_type: str,
    model_key: str,
    config: dict,
    use_toy: bool = False,
) -> None:
    """Run SHAP analysis for the best fold of a tree-based model."""
    mcfg = MODEL_CONFIG[model_key]
    prefix = mcfg["file_prefix"]
    display_name = mcfg["display_name"]

    cancer_short = cancer_type.split("-")[1]
    cancer_lower = cancer_short.lower()
    tag = "toy" if use_toy else "full"
    shap_dir = os.path.join(config["paths"]["results"], "shap", mcfg["shap_subdir"])
    os.makedirs(shap_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SHAP Analysis: {display_name} ({cancer_type}, {tag} data)")
    print(f"{'='*60}")

    # ---- 1. Identify best fold ----
    if use_toy:
        best_fold = 0
        print(f"  Toy mode: using fold 0")
    else:
        best_fold = find_best_fold(
            cancer_type, mcfg["log_model_type"], config
        )
    print(f"  Best fold: {best_fold}")

    # ---- 2. Load model ----
    model_dir = os.path.join(config["paths"]["models"], mcfg["model_subdir"])
    model_path = os.path.join(
        model_dir, f"{prefix}_{cancer_lower}_fold{best_fold}.pkl"
    )
    model = joblib.load(model_path)
    print(f"  Model loaded: {model_path}")

    # ---- 3. Reload fold data (same preprocessing as training) ----
    cv_data = load_cv_folds(cancer_type, config, use_toy=use_toy)
    fold_info = cv_data["folds"][best_fold]
    result = prepare_fold_data(cancer_type, fold_info, config, use_toy=use_toy)

    X_val_concat, feature_names = concatenate_modalities(result["X_val"])
    X_train_concat, _ = concatenate_modalities(result["X_train"])

    n_features = len(feature_names)
    print(f"  Val shape: {X_val_concat.shape}, Features: {n_features}")

    # ---- 4. TreeExplainer (exact for tree-based models) ----
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_val_concat)

    # Normalise to 3D: (n_samples, n_features, n_classes)
    # Older SHAP returns list of 2D arrays; newer returns 3D ndarray.
    if isinstance(shap_values_raw, list):
        shap_3d = np.stack(shap_values_raw, axis=-1)
        shap_list = shap_values_raw
    else:
        shap_3d = shap_values_raw
        n_cls = shap_3d.shape[2]
        shap_list = [shap_3d[:, :, c] for c in range(n_cls)]

    n_classes = shap_3d.shape[2]
    print(
        f"  SHAP computed: shape {shap_3d.shape} "
        f"({shap_3d.shape[0]} samples, {shap_3d.shape[1]} features, "
        f"{n_classes} classes)"
    )

    # ---- 5. Beeswarm summary plot ----
    print("  Generating beeswarm summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_list,
        X_val_concat,
        feature_names=feature_names,
        max_display=20,
        show=False,
        class_names=[f"Subtype {i}" for i in range(n_classes)],
    )
    plt.tight_layout()
    summary_base = os.path.join(shap_dir, f"{prefix}_shap_summary_{cancer_short}")
    plt.savefig(
        f"{summary_base}.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(f"{summary_base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  Summary plot saved: {summary_base}.png")

    # ---- 6. Mean |SHAP| per feature (averaged across classes + samples) ----
    mean_abs_shap = np.mean(np.abs(shap_3d), axis=(0, 2))

    # ---- 7. Top 50 features CSV ----
    top_indices = np.argsort(mean_abs_shap)[::-1][:50]
    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        fname = feature_names[idx]
        modality = fname.split("_")[0]
        rows.append({
            "rank": rank,
            "feature_name": fname,
            "modality": modality,
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
        })
    top_df = pd.DataFrame(rows)
    csv_path = os.path.join(shap_dir, f"top_50_features_{prefix}_{cancer_short}.csv")
    top_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Top 50 features saved: {csv_path}")
    print("  Top 5 features:")
    for _, row in top_df.head(5).iterrows():
        print(
            f"    #{row['rank']}: {row['feature_name']} ({row['modality']}) "
            f"SHAP={row['mean_abs_shap']:.6f}"
        )

    # ---- 8. Bar plot of top 20 features ----
    print("  Generating bar plot (top 20 features)...")
    top20 = top_df.head(20)
    colors = [MODALITY_COLORS.get(m, "#607D8B") for m in top20["modality"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, top20["mean_abs_shap"].values[::-1], color=colors[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["feature_name"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(
        f"Top 20 Features by SHAP Importance — {display_name} ({cancer_type})",
        fontsize=13,
    )

    legend_handles = [
        Patch(facecolor=c, label=k)
        for k, c in MODALITY_COLORS.items()
        if k in top20["modality"].values
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    bar_base = os.path.join(shap_dir, f"{prefix}_shap_bar_{cancer_short}")
    save_figure(fig, bar_base)
    print(f"  Bar plot saved: {bar_base}.png")

    # ---- 9. Save raw SHAP values for downstream use ----
    npz_path = os.path.join(shap_dir, f"{prefix}_shap_values_{cancer_short}.npz")
    np.savez_compressed(
        npz_path,
        shap_values=shap_3d,
        feature_names=np.array(feature_names),
        mean_abs_shap=mean_abs_shap,
    )
    print(f"  Raw SHAP values saved: {npz_path}")

    print(f"\n  SHAP analysis complete for {display_name} / {cancer_type}.")


def main():
    parser = argparse.ArgumentParser(
        description="SHAP analysis for tree-based baselines"
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "rf", "all"],
        default="all",
        help="Which model to analyse (default: all)",
    )
    parser.add_argument(
        "--cancer",
        choices=["GS-BRCA", "GS-COAD", "all"],
        default="all",
        help="Cancer type (default: all)",
    )
    parser.add_argument("--toy", action="store_true", help="Use toy data for testing")
    args = parser.parse_args()

    set_seeds(42)
    config = load_config()

    model_keys = (
        list(MODEL_CONFIG.keys()) if args.model == "all" else [args.model]
    )

    if args.cancer == "all":
        cancer_types = ["GS-BRCA"] if args.toy else config["project"]["cancer_types"]
    else:
        cancer_types = [args.cancer]

    for mk in model_keys:
        for ct in cancer_types:
            run_shap_analysis(ct, mk, config, use_toy=args.toy)

    print("\nAll SHAP analyses complete.")


if __name__ == "__main__":
    main()
