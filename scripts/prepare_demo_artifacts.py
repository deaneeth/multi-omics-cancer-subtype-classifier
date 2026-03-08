"""
Prepare Demo Artifacts for Streamlit App
=========================================
Exports best-fold models, a plain StandardScaler, config.json,
and a sample input CSV for the Streamlit demo.

Usage:
    python scripts/prepare_demo_artifacts.py
"""

import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import concatenate_modalities, load_cv_folds, prepare_fold_data
from src.utils import load_config, set_seeds


def find_best_fold(metrics_path: str) -> int:
    """Return fold index with highest F1 from a metrics CSV."""
    df = pd.read_csv(metrics_path)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.dropna(subset=["f1"])
    best_idx = df["f1"].idxmax()
    best_fold = int(df.loc[best_idx, "fold"])
    best_f1 = df.loc[best_idx, "f1"]
    print(f"  Best fold: {best_fold} (F1={best_f1:.4f})")
    return best_fold


def update_model_comparison(config: dict):
    """Add PathwayAwareFusion rows to model_comparison.csv if missing."""
    comp_path = os.path.join(config["paths"]["results"], "metrics", "model_comparison.csv")
    comp_df = pd.read_csv(comp_path)

    if "PathwayAwareFusion" in comp_df["Model"].values:
        print("  model_comparison.csv already includes PathwayAwareFusion")
        return

    new_rows = []
    for cancer in config["project"]["cancer_types"]:
        c = cancer.split("-")[1].lower()
        pw_path = os.path.join(config["paths"]["results"], "metrics", f"pathway_fusion_{c}_metrics.csv")
        if not os.path.exists(pw_path):
            continue
        df = pd.read_csv(pw_path)
        for col in ["precision", "recall", "f1", "nmi", "ari"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["f1"])
        new_rows.append({
            "Model": "PathwayAwareFusion",
            "Cancer": cancer,
            "F1_mean": round(df["f1"].mean(), 4),
            "F1_std": round(df["f1"].std(), 4),
            "Precision_mean": round(df["precision"].mean(), 4),
            "Precision_std": round(df["precision"].std(), 4),
            "Recall_mean": round(df["recall"].mean(), 4),
            "Recall_std": round(df["recall"].std(), 4),
            "NMI_mean": round(df["nmi"].mean(), 4),
            "NMI_std": round(df["nmi"].std(), 4),
            "ARI_mean": round(df["ari"].mean(), 4),
            "ARI_std": round(df["ari"].std(), 4),
        })

    if new_rows:
        comp_df = pd.concat([comp_df, pd.DataFrame(new_rows)], ignore_index=True)
        comp_df.to_csv(comp_path, index=False)
        print(f"  Added {len(new_rows)} PathwayAwareFusion rows to model_comparison.csv")


def main():
    set_seeds(42)
    config = load_config()
    cancer_type = "GS-BRCA"
    artifact_dir = os.path.join("app", "model_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # --- 1. Find best folds ---
    print("=== Finding best folds ===")
    print("XGBoost BRCA:")
    xgb_best_fold = find_best_fold("results/metrics/xgb_brca_metrics.csv")
    print("IntermediateFusion BRCA:")
    fusion_best_fold = find_best_fold("results/metrics/fusion_brca_metrics.csv")

    # --- 2. Copy best XGBoost model ---
    print("\n=== Copying best XGBoost model ===")
    xgb_src = f"models/baseline_xgb/xgb_brca_fold{xgb_best_fold}.pkl"
    xgb_dst = os.path.join(artifact_dir, "xgb_best.pkl")
    shutil.copy2(xgb_src, xgb_dst)
    print(f"  {xgb_src} -> {xgb_dst}")

    # --- 3. Copy best Fusion model ---
    print("\n=== Copying best Fusion model ===")
    fusion_src = f"models/intermediate_fusion/fusion_brca_fold{fusion_best_fold}.pt"
    fusion_dst = os.path.join(artifact_dir, "fusion_best.pt")
    shutil.copy2(fusion_src, fusion_dst)
    print(f"  {fusion_src} -> {fusion_dst}")

    # --- 4. Load fold data and fit plain StandardScaler ---
    print("\n=== Fitting plain StandardScaler on concatenated training data ===")
    cv_data = load_cv_folds(cancer_type, config, use_toy=False)
    best_fold_info = cv_data["folds"][xgb_best_fold]

    result = prepare_fold_data(cancer_type, best_fold_info, config, use_toy=False)
    X_train_concat, feature_names = concatenate_modalities(result["X_train"])
    X_val_concat, _ = concatenate_modalities(result["X_val"])

    concat_scaler = StandardScaler()
    concat_scaler.fit(X_train_concat)
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")
    joblib.dump(concat_scaler, scaler_path)
    print(f"  Scaler fitted on {X_train_concat.shape} training matrix -> {scaler_path}")

    # --- 5. Record modality dims dynamically ---
    modality_order = ["mrna", "mirna", "methy", "cnv"]
    modality_dims = {}
    for mod in modality_order:
        if mod in result["X_train"]:
            modality_dims[mod] = result["X_train"][mod].shape[1]
    print(f"  Modality dims: {modality_dims}")

    # --- 6. Save config.json ---
    print("\n=== Saving config.json ===")
    y_all = np.concatenate([result["y_train"], result["y_val"]])
    class_labels = sorted(np.unique(y_all).tolist())
    class_names = {
        0: "Basal-like",
        1: "HER2-enriched",
        2: "Luminal A",
        3: "Luminal B",
        4: "Normal-like",
    }

    demo_config = {
        "cancer_type": cancer_type,
        "n_classes": len(class_labels),
        "class_labels": class_labels,
        "class_names": class_names,
        "feature_names": feature_names,
        "total_features": len(feature_names),
        "modality_dims": modality_dims,
        "modality_order": modality_order,
        "xgb_best_fold": xgb_best_fold,
        "fusion_best_fold": fusion_best_fold,
        "fusion_latent_dim": config["fusion"]["default_latent_dim"],
        "fusion_encoder_hidden": config["fusion"]["encoder_hidden"],
        "fusion_classifier_hidden": config["fusion"]["classifier_hidden"],
        "fusion_dropout": config["fusion"]["dropout"],
    }
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(demo_config, f, indent=2)
    print(f"  Saved: {config_path}")
    print(f"  {len(feature_names)} features, {len(class_labels)} classes")

    # --- 7. Create sample_input.csv ---
    # Format: features as rows, samples as columns (MLOmics native format)
    print("\n=== Creating sample_input.csv ===")
    n_samples = min(3, X_val_concat.shape[0])
    sample_data = X_val_concat[:n_samples]
    sample_labels = result["y_val"][:n_samples]

    # Build DataFrame: rows = features, columns = sample IDs
    val_ids = best_fold_info["val"][:n_samples]
    sample_df = pd.DataFrame(
        sample_data.T,
        index=feature_names,
        columns=val_ids,
    )
    sample_path = os.path.join("app", "sample_input.csv")
    sample_df.to_csv(sample_path)
    print(f"  Saved {n_samples} samples (features×samples format): {sample_path}")
    print(f"  True labels: {sample_labels.tolist()}")
    for i, lbl in enumerate(sample_labels):
        print(f"    Sample {val_ids[i]}: class {lbl} ({class_names.get(int(lbl), '?')})")

    # --- 8. Update model_comparison.csv ---
    print("\n=== Updating model_comparison.csv ===")
    update_model_comparison(config)

    print("\n=== All demo artifacts prepared ===")
    for f in os.listdir(artifact_dir):
        size = os.path.getsize(os.path.join(artifact_dir, f))
        print(f"  {f}: {size:,} bytes")


if __name__ == "__main__":
    main()
