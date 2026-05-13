#!/usr/bin/env python
"""Generate per-fold macro-averaged OvR ROC curves for COAD models."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Load config
from src.utils import load_config

config = load_config()
results_dir = config["paths"]["results"]
metrics_dir = os.path.join(results_dir, "metrics")
plots_dir = os.path.join(results_dir, "plots")
cancer = "GS-COAD"
n_classes = 4

# Model info
models = ["XGBoost", "RandomForest", "IntermediateFusion", "PathwayAwareFusion"]
model_keys = ["xgb", "rf", "fusion", "pathway_fusion"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Load AUC summary for mean values
auc_summary_path = os.path.join(metrics_dir, "auc_summary.csv")
auc_data = {}
with open(auc_summary_path) as f:
    lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(",")
        model_name, cancer_name, auc_mean, auc_std = parts
        if cancer_name == cancer:
            auc_data[model_name] = (float(auc_mean), float(auc_std))

# Create figure with 5 subplots (one per fold) + one summary
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Per-Fold Macro-Averaged One-vs-Rest ROC Curves — GS-COAD",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
axes = axes.flatten()

folds = list(range(5))
all_fold_rocs = {m: [] for m in model_keys}

# ---- Per-fold ROC curves ----
for fold_idx, fold in enumerate(folds):
    ax = axes[fold_idx]

    # Plot random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.4, label="Random")

    for model_key, model_name, color in zip(model_keys, models, colors):
        pred_file = os.path.join(
            metrics_dir, f"{model_key}_coad_fold{fold}_predictions.npz"
        )

        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found, skipping")
            continue

        data = np.load(pred_file, allow_pickle=True)
        y_true = data["y_true"]

        # Get probability scores
        if "y_prob" in data.files:
            y_score = data["y_prob"]
        else:
            print(f"Warning: {pred_file} has no y_prob, skipping ROC")
            continue

        # One-vs-Rest: compute ROC per class and average
        fpr_list, tpr_list, roc_auc_list = [], [], []
        for class_idx in range(n_classes):
            y_bin = (y_true == class_idx).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue  # Skip if class not present
            fpr, tpr, _ = roc_curve(y_bin, y_score[:, class_idx])
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(auc(fpr, tpr))

        macro_auc = np.mean(roc_auc_list) if roc_auc_list else np.nan
        all_fold_rocs[model_key].append(macro_auc)

        # Interpolate all ROC curves to common FPR points and average
        mean_fpr = np.linspace(0, 1, 100)
        interp_tprs = []
        for fpr, tpr in zip(fpr_list, tpr_list):
            interp_tprs.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[0] = 0.0  # Start at origin
        mean_tpr[-1] = 1.0  # End at top-right

        # Plot the actual ROC curve
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            lw=2.5,
            label=f"{model_name} (AUC={macro_auc:.4f})",
        )

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"Fold {fold} - Macro OvR ROC", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

# ---- Summary plot (mean AUC per model across all folds) ----
ax_summary = axes[5]

# Create a simple text summary by printing AUC values
summary_text = "Mean AUC (±std across 5 folds):\n\n"
for model_key, model_name, color in zip(model_keys, models, colors):
    if model_name in auc_data:
        mean_auc, std_auc = auc_data[model_name]
        summary_text += f"{model_name}:\n{mean_auc:.4f} ± {std_auc:.4f}\n\n"

# Display as text in the subplot
ax_summary.text(
    0.5,
    0.5,
    summary_text,
    fontsize=11,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3, pad=1.0),
    family="monospace",
    fontweight="bold",
)
ax_summary.axis("off")

# Adjust layout and save
plt.tight_layout()
plt.savefig(
    os.path.join(plots_dir, "roc_coad_models_per_fold.png"),
    dpi=300,
    bbox_inches="tight",
)
print(f"✓ Saved: {os.path.join(plots_dir, 'roc_coad_models_per_fold.png')}")

# Print summary stats
print("\n=== Mean AUC Summary ===")
for model_key, model_name, color in zip(model_keys, models, colors):
    if model_name in auc_data:
        mean_auc, std_auc = auc_data[model_name]
        print(f"{model_name}: {mean_auc:.4f} ± {std_auc:.4f}")
