"""
Compute macro one-vs-rest AUC for all saved models and regenerate
the prediction archives used by the Streamlit demo.

This script restores the artifacts lost after the hard reset:

    results/metrics/{xgb,rf,fusion,pathway_fusion}_{cancer}_fold{i}_predictions.npz
    results/metrics/auc_scores.csv
    results/metrics/auc_summary.csv
    results/metrics/model_comparison.csv  (adds auc_mean / auc_std)
    results/plots/confusion_matrix_pathway_fusion_{BRCA,COAD}.png/pdf

It relies only on the committed fold splits and the saved checkpoints under
models/ so the regenerated artifacts remain deterministic.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import IntermediateFusionModel, PathwayAwareFusionModel
from src.preprocessing import concatenate_modalities, load_cv_folds, prepare_fold_data
from src.utils import load_config, set_seeds


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    checkpoint_dir: str
    checkpoint_prefix: str
    artifact_prefix: str
    kind: str


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec("XGBoost", os.path.join("models", "baseline_xgb"), "xgb", "xgb", "tree"),
    ModelSpec(
        "RandomForest", os.path.join("models", "baseline_rf"), "rf", "rf", "tree"
    ),
    ModelSpec(
        "IntermediateFusion",
        os.path.join("models", "intermediate_fusion"),
        "fusion",
        "fusion",
        "fusion",
    ),
    ModelSpec(
        "PathwayAwareFusion",
        os.path.join("models", "pathway_fusion"),
        "pathway_fusion",
        "pathway_fusion",
        "pathway",
    ),
)


def _cancer_suffix(cancer_type: str) -> str:
    return cancer_type.split("-")[1].lower()


def _load_model_checkpoint(spec: ModelSpec, cancer_type: str, fold_idx: int):
    suffix = _cancer_suffix(cancer_type)
    ext = "pkl" if spec.kind == "tree" else "pt"
    checkpoint_path = os.path.join(
        spec.checkpoint_dir, f"{spec.checkpoint_prefix}_{suffix}_fold{fold_idx}.{ext}"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    if spec.kind == "tree":
        return joblib.load(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if spec.kind == "pathway":
        mapping_path = os.path.join(
            "app", "model_artifacts", "pathway_gene_mapping.json"
        )
        with open(mapping_path, "r", encoding="utf-8") as handle:
            full_mapping = json.load(handle)
        mapping = full_mapping[cancer_type.split("-")[1]]
        model = PathwayAwareFusionModel(
            modality_dims=ckpt["modality_dims"],
            pathway_indices=mapping["pathways"],
            unmapped_indices=mapping["unmapped_indices"],
            latent_dim=ckpt["latent_dim"],
            hidden_dim=ckpt["classifier_hidden"],
            num_classes=ckpt["num_classes"],
            dropout=ckpt["dropout"],
            encoder_hidden=ckpt["encoder_hidden"],
        )
    else:
        model = IntermediateFusionModel(
            modality_dims=ckpt["modality_dims"],
            latent_dim=ckpt["latent_dim"],
            hidden_dim=ckpt["classifier_hidden"],
            num_classes=ckpt["num_classes"],
            dropout=ckpt["dropout"],
            encoder_hidden=ckpt["encoder_hidden"],
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _predict_tree_model(model, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(x_val)
    y_prob = model.predict_proba(x_val)
    return np.asarray(y_pred), np.asarray(y_prob)


def _predict_fusion_model(
    model, x_val_dict: Dict[str, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    x_val = {
        name: torch.tensor(frame.values.astype(np.float32), dtype=torch.float32)
        for name, frame in x_val_dict.items()
    }
    with torch.no_grad():
        logits = model(x_val)
        y_prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()
    return y_pred, y_prob


def _macro_ovr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < y_prob.shape[1]:
        return float("nan")
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(y_prob.shape[1])))
        return float(
            roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError:
        return float("nan")


def _save_predictions(
    out_path: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)


def _best_fold_from_metrics(metrics_path: str) -> int:
    df = pd.read_csv(metrics_path)
    df = df[pd.to_numeric(df["fold"], errors="coerce").notna()].copy()
    df["fold"] = df["fold"].astype(int)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    best_row = df.loc[df["f1"].idxmax()]
    return int(best_row["fold"])


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, cancer_type: str, out_stem: str
) -> None:
    n_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Class {i}" for i in labels],
        yticklabels=[f"Class {i}" for i in labels],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix: PathwayAwareFusion ({cancer_type}) - Best Fold",
        fontsize=13,
    )

    for ext in ("png", "pdf"):
        fig.savefig(
            f"{out_stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)


def _ensure_model_comparison_auc(
    summary_df: pd.DataFrame, model_comparison_path: str
) -> None:
    if not os.path.exists(model_comparison_path):
        return

    comp = pd.read_csv(model_comparison_path)
    if "auc_mean" not in comp.columns:
        comp["auc_mean"] = np.nan
    if "auc_std" not in comp.columns:
        comp["auc_std"] = np.nan

    for _, row in summary_df.iterrows():
        mask = (comp["Model"] == row["model"]) & (comp["Cancer"] == row["cancer"])
        comp.loc[mask, "auc_mean"] = row["auc_mean"]
        comp.loc[mask, "auc_std"] = row["auc_std"]

    comp.to_csv(model_comparison_path, index=False, float_format="%.4f")


def compute_for_fold(
    spec: ModelSpec,
    cancer_type: str,
    fold_info: Dict[str, List[str]],
    config: dict,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    result = prepare_fold_data(cancer_type, fold_info, config, use_toy=False)
    y_true = np.asarray(result["y_val"])

    if spec.kind == "tree":
        x_val, _ = concatenate_modalities(result["X_val"])
        model = _load_model_checkpoint(spec, cancer_type, fold_info["fold"])
        y_pred, y_prob = _predict_tree_model(model, x_val)
    else:
        model = _load_model_checkpoint(spec, cancer_type, fold_info["fold"])
        y_pred, y_prob = _predict_fusion_model(model, result["X_val"])

    auc_value = _macro_ovr_auc(y_true, y_prob)
    return auc_value, y_true, y_pred, y_prob


def main() -> None:
    set_seeds(42)
    config = load_config()

    results_dir = os.path.join(config["paths"]["results"], "metrics")
    plots_dir = os.path.join(config["paths"]["results"], "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    auc_rows: List[Dict[str, object]] = []

    for cancer_type in config["project"]["cancer_types"]:
        folds = load_cv_folds(cancer_type, config, use_toy=False)["folds"]
        suffix = _cancer_suffix(cancer_type)

        for spec in MODEL_SPECS:
            for fold_info in folds:
                fold_idx = int(fold_info["fold"])
                auc_value, y_true, y_pred, y_prob = compute_for_fold(
                    spec, cancer_type, fold_info, config
                )

                out_path = os.path.join(
                    results_dir,
                    f"{spec.artifact_prefix}_{suffix}_fold{fold_idx}_predictions.npz",
                )
                _save_predictions(out_path, y_true, y_pred, y_prob)

                auc_rows.append(
                    {
                        "model": spec.model_name,
                        "cancer": cancer_type,
                        "fold": fold_idx,
                        "auc": auc_value,
                    }
                )

    auc_df = pd.DataFrame(auc_rows)
    auc_df.sort_values(["cancer", "model", "fold"], inplace=True)
    auc_scores_path = os.path.join(results_dir, "auc_scores.csv")
    auc_df.to_csv(auc_scores_path, index=False, float_format="%.4f")

    summary_df = (
        auc_df.groupby(["model", "cancer"], as_index=False)["auc"]
        .agg(auc_mean="mean", auc_std="std")
        .reset_index(drop=True)
    )
    auc_summary_path = os.path.join(results_dir, "auc_summary.csv")
    summary_df.to_csv(auc_summary_path, index=False, float_format="%.4f")

    _ensure_model_comparison_auc(
        summary_df, os.path.join(results_dir, "model_comparison.csv")
    )

    for cancer_type in config["project"]["cancer_types"]:
        suffix = _cancer_suffix(cancer_type)
        metrics_path = os.path.join(results_dir, f"pathway_fusion_{suffix}_metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        best_fold = _best_fold_from_metrics(metrics_path)
        pred_path = os.path.join(
            results_dir, f"pathway_fusion_{suffix}_fold{best_fold}_predictions.npz"
        )
        if not os.path.exists(pred_path):
            continue

        pred = np.load(pred_path)
        out_stem = os.path.join(
            plots_dir, f"confusion_matrix_pathway_fusion_{cancer_type.split('-')[1]}"
        )
        _plot_confusion_matrix(pred["y_true"], pred["y_pred"], cancer_type, out_stem)

    print(f"AUC scores saved to {auc_scores_path}")
    print(f"AUC summary saved to {auc_summary_path}")


if __name__ == "__main__":
    main()
