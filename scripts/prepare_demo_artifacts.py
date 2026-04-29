"""
Prepare Demo Artifacts for Streamlit App
=========================================
Exports best-fold models, train-fitted per-modality preprocessing artifacts,
config_{cancer}.json, and sample input CSVs for the Streamlit demo.

Usage:
    python scripts/prepare_demo_artifacts.py                   # both cancers
    python scripts/prepare_demo_artifacts.py --cancer GS-BRCA  # BRCA only
    python scripts/prepare_demo_artifacts.py --cancer GS-COAD  # COAD only
    python scripts/prepare_demo_artifacts.py --cancer all      # both cancers
"""

import argparse
import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MODALITY_KEYS, get_common_samples, load_labels, load_modality
from src.preprocessing import (
    PerModalityImputer,
    PerModalityScaler,
    concatenate_modalities,
    load_cv_folds,
)
from src.utils import load_config, set_seeds


# Cancer-specific class label maps — update if label encoding changes
CLASS_NAMES = {
    "GS-BRCA": {
        0: "Basal-like",
        1: "HER2-enriched",
        2: "Luminal A",
        3: "Luminal B",
        4: "Normal-like",
    },
    "GS-COAD": {
        0: "CMS1 (MSI/Immune)",
        1: "CMS2 (Canonical)",
        2: "CMS3 (Metabolic)",
        3: "CMS4 (Mesenchymal)",
    },
}


def _make_columns_unique(columns) -> list:
    """Rename ALL occurrences of duplicate column names to 'name_0', 'name_1', ...

    Applied to raw modality DataFrames before fitting imputer/scaler so that
    duplicate NaN-named miRNA features get unique identifiers throughout the
    entire pipeline (config, sample CSV, and fitted artifact indices all agree).
    """
    from collections import Counter
    counts = Counter(columns)
    seen: dict = {}
    result = []
    for col in columns:
        if counts[col] > 1:
            idx = seen.get(col, 0)
            seen[col] = idx + 1
            result.append(f"{col}_{idx}")
        else:
            result.append(col)
    return result


def find_best_fold(metrics_path: str) -> int:
    """Return fold index with highest F1 from a metrics CSV."""
    df = pd.read_csv(metrics_path)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.dropna(subset=["f1"])
    # Exclude the mean+/-std summary row
    df = df[~df["fold"].astype(str).str.contains(r"\+/-", na=False)]
    best_idx = df["f1"].idxmax()
    best_fold = int(df.loc[best_idx, "fold"])
    best_f1 = df.loc[best_idx, "f1"]
    print(f"  Best fold: {best_fold} (F1={best_f1:.4f})")
    return best_fold


def update_model_comparison(config: dict):
    """Add PathwayAwareFusion rows to model_comparison.csv if missing."""
    comp_path = os.path.join(config["paths"]["results"], "metrics", "model_comparison.csv")
    if not os.path.exists(comp_path):
        print("  model_comparison.csv not found — skipping")
        return

    comp_df = pd.read_csv(comp_path)

    if "PathwayAwareFusion" in comp_df["Model"].values:
        print("  model_comparison.csv already includes PathwayAwareFusion")
        return

    new_rows = []
    for cancer in config["project"]["cancer_types"]:
        c = cancer.split("-")[1].lower()
        pw_path = os.path.join(
            config["paths"]["results"], "metrics", f"pathway_fusion_{c}_metrics.csv"
        )
        if not os.path.exists(pw_path):
            continue
        df = pd.read_csv(pw_path)
        for col in ["precision", "recall", "f1", "nmi", "ari"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["f1"])
        new_rows.append(
            {
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
            }
        )

    if new_rows:
        comp_df = pd.concat([comp_df, pd.DataFrame(new_rows)], ignore_index=True)
        comp_df.to_csv(comp_path, index=False)
        print(f"  Added {len(new_rows)} PathwayAwareFusion rows to model_comparison.csv")


def prepare_artifacts_for_cancer(cancer_type: str, config: dict, artifact_dir: str) -> dict:
    """Prepare all demo artifacts for a single cancer type.

    Returns a summary dict with fold numbers and dims for reference.
    """
    c = cancer_type.split("-")[1].lower()  # "brca" or "coad"
    print(f"\n{'='*60}")
    print(f"Preparing artifacts for {cancer_type} ({c.upper()})")
    print(f"{'='*60}")

    # --- 1. Find best folds ---
    print(f"\n=== Finding best folds for {cancer_type} ===")
    xgb_metrics_path = f"results/metrics/xgb_{c}_metrics.csv"
    fusion_metrics_path = f"results/metrics/fusion_{c}_metrics.csv"

    print(f"XGBoost {c.upper()}:")
    xgb_best_fold = find_best_fold(xgb_metrics_path)
    print(f"IntermediateFusion {c.upper()}:")
    fusion_best_fold = find_best_fold(fusion_metrics_path)

    # --- 2. Copy best XGBoost model ---
    print(f"\n=== Copying best XGBoost model ({c.upper()}) ===")
    xgb_src = f"models/baseline_xgb/xgb_{c}_fold{xgb_best_fold}.pkl"
    xgb_dst = os.path.join(artifact_dir, f"xgb_best_{c}.pkl")
    shutil.copy2(xgb_src, xgb_dst)
    print(f"  {xgb_src} -> {xgb_dst}")
    if cancer_type == "GS-BRCA":
        shutil.copy2(xgb_src, os.path.join(artifact_dir, "xgb_best.pkl"))

    # --- 3. Copy best Fusion model ---
    print(f"\n=== Copying best Fusion model ({c.upper()}) ===")
    fusion_src = f"models/intermediate_fusion/fusion_{c}_fold{fusion_best_fold}.pt"
    fusion_dst = os.path.join(artifact_dir, f"fusion_best_{c}.pt")
    shutil.copy2(fusion_src, fusion_dst)
    print(f"  {fusion_src} -> {fusion_dst}")

    # --- 4. Load fold raw data and fit train-only preprocessing artifacts ---
    print(f"\n=== Saving per-modality preprocessing artifacts ({c.upper()}) ===")
    cv_data = load_cv_folds(cancer_type, config, use_toy=False)
    best_fold_info = cv_data["folds"][xgb_best_fold]
    train_ids = [str(s) for s in best_fold_info["train"]]
    val_ids = [str(s) for s in best_fold_info["val"]]

    modality_order = [m for m in MODALITY_KEYS if m in config["modalities"]]
    raw_modalities = {
        mod: load_modality(cancer_type, mod, config, use_toy=False)
        for mod in modality_order
    }
    for mod in modality_order:
        raw_modalities[mod].columns = raw_modalities[mod].columns.astype(str)
        raw_modalities[mod].columns = _make_columns_unique(raw_modalities[mod].columns)

    common_samples = set(get_common_samples(cancer_type, config, use_toy=False))
    train_common = sorted(common_samples.intersection(train_ids))
    val_common = sorted(common_samples.intersection(val_ids))
    if not train_common or not val_common:
        raise ValueError(
            f"Fold {xgb_best_fold} for {cancer_type} has empty train/val after sample alignment."
        )

    train_dict = {mod: raw_modalities[mod].loc[train_common].copy() for mod in modality_order}
    val_dict = {mod: raw_modalities[mod].loc[val_common].copy() for mod in modality_order}

    imputer = PerModalityImputer()
    imputer.fit(train_dict, modality_order)
    train_dict = imputer.transform(train_dict, modality_order)
    val_dict = imputer.transform(val_dict, modality_order)
    for mod in modality_order:
        train_dict[mod].columns = train_dict[mod].columns.map(str)
        val_dict[mod].columns = val_dict[mod].columns.map(str)

    scaler = PerModalityScaler()
    scaler.fit(train_dict, modality_order)
    train_dict = scaler.transform(train_dict, modality_order)
    val_dict = scaler.transform(val_dict, modality_order)
    for mod in modality_order:
        train_dict[mod].columns = train_dict[mod].columns.map(str)
        val_dict[mod].columns = val_dict[mod].columns.map(str)

    _, feature_names = concatenate_modalities(train_dict, modality_order)

    imputer_path = os.path.join(artifact_dir, f"imputer_{c}.pkl")
    scaler_path = os.path.join(artifact_dir, f"per_modality_scaler_{c}.pkl")
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved train-fitted imputer -> {imputer_path}")
    print(f"  Saved train-fitted per-modality scaler -> {scaler_path}")

    # Backward-compatible BRCA aliases used by single-cancer demo flows
    if cancer_type == "GS-BRCA":
        joblib.dump(imputer, os.path.join(artifact_dir, "imputer.pkl"))
        joblib.dump(scaler, os.path.join(artifact_dir, "per_modality_scaler.pkl"))

    # --- 5. Record modality dims dynamically from train fold ---
    modality_dims = {
        mod: train_dict[mod].shape[1]
        for mod in modality_order
        if mod in train_dict
    }
    total_feats = len(feature_names)
    print(f"  Modality dims: {modality_dims}")
    print(f"  Total features: {total_feats}")

    # --- 6. Save config_{c}.json ---
    print(f"\n=== Saving config_{c}.json ===")
    labels = load_labels(cancer_type, config, use_toy=False)
    y_all = labels.loc[train_common + val_common].values
    class_labels = sorted(np.unique(y_all).tolist())
    class_names = CLASS_NAMES.get(cancer_type, {i: f"Class {i}" for i in class_labels})
    # JSON keys must be strings
    class_names_str = {str(k): v for k, v in class_names.items()}

    modality_names = list(modality_dims.keys())

    demo_config = {
        "cancer_type": cancer_type,
        "n_classes": len(class_labels),
        "class_labels": class_labels,
        "class_names": class_names_str,
        "feature_names": feature_names,
        "total_features": len(feature_names),
        "modality_names": modality_names,
        "modality_sizes": modality_dims,
        "modality_dims": modality_dims,
        "modality_order": modality_order,
        "xgb_best_fold": xgb_best_fold,
        "fusion_best_fold": fusion_best_fold,
        "fusion_latent_dim": config["fusion"]["default_latent_dim"],
        "fusion_encoder_hidden": config["fusion"]["encoder_hidden"],
        "fusion_classifier_hidden": config["fusion"]["classifier_hidden"],
        "fusion_dropout": config["fusion"]["dropout"],
    }
    config_path = os.path.join(artifact_dir, f"config_{c}.json")
    with open(config_path, "w") as f:
        json.dump(demo_config, f, indent=2)
    print(f"  Saved: {config_path}")
    print(f"  {len(feature_names)} features, {len(class_labels)} classes")

    if cancer_type == "GS-BRCA":
        with open(os.path.join(artifact_dir, "config.json"), "w") as f:
            json.dump(demo_config, f, indent=2)

    # --- 7. Create sample_input_{c}.csv ---
    # Format: one sample row, columns exactly aligned to config feature_names
    print(f"\n=== Creating sample_input_{c}.csv ===")
    X_val_concat, _ = concatenate_modalities(val_dict, modality_order)
    if X_val_concat.shape[0] == 0:
        raise ValueError(f"No validation samples available for {cancer_type} fold {xgb_best_fold}")

    sample_df = pd.DataFrame(X_val_concat[:1], columns=feature_names)
    sample_path = os.path.join("app", f"sample_input_{c}.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"  Saved sample input (1 x {sample_df.shape[1]:,}): {sample_path}")

    if cancer_type == "GS-BRCA":
        sample_df.to_csv(os.path.join("app", "sample_input.csv"), index=False)

    # --- 8. Copy best PathwayAwareFusion model ---
    print(f"\n=== Copying best PathwayAwareFusion model ({c.upper()}) ===")
    pw_metrics_path = f"results/metrics/pathway_fusion_{c}_metrics.csv"
    pw_best_fold = None
    if os.path.exists(pw_metrics_path):
        print(f"PathwayAwareFusion {c.upper()}:")
        pw_best_fold = find_best_fold(pw_metrics_path)
        pw_src = f"models/pathway_fusion/pathway_fusion_{c}_fold{pw_best_fold}.pt"
        pw_dst = os.path.join(artifact_dir, f"pathway_fusion_best_{c}.pt")
        shutil.copy2(pw_src, pw_dst)
        print(f"  {pw_src} -> {pw_dst}")
    else:
        print(f"  PathwayAwareFusion metrics not found: {pw_metrics_path} — skipping")

    return {
        "cancer_type": cancer_type,
        "c": c,
        "xgb_best_fold": xgb_best_fold,
        "fusion_best_fold": fusion_best_fold,
        "pw_best_fold": pw_best_fold,
        "modality_dims": modality_dims,
        "total_features": total_feats,
        "n_classes": len(class_labels),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare demo artifacts for the Streamlit app."
    )
    parser.add_argument(
        "--cancer",
        default="all",
        choices=["GS-BRCA", "GS-COAD", "all"],
        help="Cancer type to prepare artifacts for (default: all)",
    )
    args = parser.parse_args()

    set_seeds(42)
    config = load_config()
    artifact_dir = os.path.join("app", "model_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    if args.cancer == "all":
        cancer_types = config["project"]["cancer_types"]
    else:
        cancer_types = [args.cancer]

    summaries = []
    for cancer_type in cancer_types:
        summary = prepare_artifacts_for_cancer(cancer_type, config, artifact_dir)
        summaries.append(summary)

    # --- Copy shared pathway gene mapping (needed by PathwayAwareFusion) ---
    pw_mapping_src = os.path.join("data", "processed", "pathway_gene_mapping.json")
    pw_mapping_dst = os.path.join(artifact_dir, "pathway_gene_mapping.json")
    if os.path.exists(pw_mapping_src):
        shutil.copy2(pw_mapping_src, pw_mapping_dst)
        print(f"\nCopied pathway_gene_mapping.json -> {pw_mapping_dst}")
    else:
        print(f"\nWARNING: {pw_mapping_src} not found — PathwayAwareFusion demo will not work")

    # --- Update model_comparison.csv ---
    print("\n=== Updating model_comparison.csv ===")
    update_model_comparison(config)

    # --- Summary ---
    print("\n=== Artifact summary ===")
    for s in summaries:
        pw_fold_str = str(s["pw_best_fold"]) if s["pw_best_fold"] is not None else "N/A"
        print(
            f"  {s['cancer_type']}: {s['total_features']} features, "
            f"{s['n_classes']} classes, "
            f"XGB fold={s['xgb_best_fold']}, Fusion fold={s['fusion_best_fold']}, "
            f"PathwayFusion fold={pw_fold_str}"
        )
        print(f"    Modality dims: {s['modality_dims']}")

    print("\n=== All demo artifacts in app/model_artifacts/ ===")
    for fname in sorted(os.listdir(artifact_dir)):
        size = os.path.getsize(os.path.join(artifact_dir, fname))
        print(f"  {fname}: {size:,} bytes")


if __name__ == "__main__":
    main()
