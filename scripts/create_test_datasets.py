"""
Create Realistic Test Datasets for Streamlit Demo Evaluation
=============================================================
Extracts real validation-set TCGA samples (not synthetic) from the best CV
fold, preprocesses them with the saved train-fitted artifacts, and writes
upload-ready CSVs with a leading `sample_id` column.

The leading `sample_id` column is preserved as the DataFrame index by
normalize_uploaded_dataframe in streamlit_app.py, meaning:
  - Fusion model attribution appears automatically for the 3 pre-computed samples
  - All samples show XGBoost SHAP waterfall plots
  - True labels + per-model confidence are recorded in companion metadata CSVs

Output (written to app/test_datasets/):
    test_brca_batch.csv               20 BRCA samples, 4 per subtype
    test_brca_single_<subtype>.csv    1 highest-confidence BRCA sample per subtype
    test_brca_metadata.csv            True labels + expected predictions + descriptions
    (same four files for COAD)

Usage:
    python scripts/create_test_datasets.py
    python scripts/create_test_datasets.py --cancer GS-BRCA
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MODALITY_KEYS, load_labels, load_modality
from src.preprocessing import load_cv_folds
from src.utils import load_config, set_seeds


# Clinical subtype descriptions for context
SUBTYPE_DESCRIPTIONS = {
    "GS-BRCA": {
        "Basal-like": (
            "Triple-negative (ER-/PR-/HER2-) breast cancer. High proliferation, "
            "poor prognosis, BRCA1 mutations common. Responds to chemotherapy."
        ),
        "HER2-enriched": (
            "HER2 gene amplification drives growth. Targeted by trastuzumab "
            "(Herceptin). Historically poor prognosis, now improved with HER2 therapy."
        ),
        "Luminal A": (
            "ER+/PR+, HER2-, low Ki-67. Most common and best-prognosis subtype. "
            "Responds well to hormone therapy (tamoxifen/aromatase inhibitors)."
        ),
        "Luminal B": (
            "ER+ with high Ki-67 or HER2+. More aggressive than Luminal A. "
            "May benefit from chemotherapy in addition to hormone therapy."
        ),
        "Normal-like": (
            "Resembles normal breast tissue gene expression. Intermediate prognosis. "
            "Distinct from other subtypes; often classified with Luminal A clinically."
        ),
    },
    "GS-COAD": {
        "CMS1 (MSI/Immune)": (
            "Microsatellite-unstable (MSI-H), hypermutated, strong immune infiltration. "
            "Best prognosis. Responds to immune checkpoint inhibitors (pembrolizumab)."
        ),
        "CMS2 (Canonical)": (
            "WNT/MYC activation, chromosomal instability (CIN). Most common CMS subtype. "
            "Standard chemotherapy (FOLFOX/FOLFIRI) as primary treatment."
        ),
        "CMS3 (Metabolic)": (
            "KRAS mutations, metabolic dysregulation. Intermediate prognosis. "
            "Anti-EGFR therapy typically less effective due to KRAS mutations."
        ),
        "CMS4 (Mesenchymal)": (
            "Epithelial-to-mesenchymal transition, TGF-β activation. Worst prognosis, "
            "highest relapse rate. Potential targets: VEGF pathway, TGF-β inhibition."
        ),
    },
}


def _make_columns_unique(columns) -> list:
    """Rename duplicate column names to 'name_0', 'name_1', ... (matches prepare_demo_artifacts.py)."""
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


def _find_best_fold(metrics_path: str) -> int:
    df = pd.read_csv(metrics_path)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.dropna(subset=["f1"])
    df = df[~df["fold"].astype(str).str.contains(r"\+/-", na=False)]
    return int(df.loc[df["f1"].idxmax(), "fold"])


def create_test_datasets_for_cancer(
    cancer_type: str,
    config: dict,
    out_dir: Path,
    n_per_subtype: int = 4,
) -> None:
    c = cancer_type.split("-")[1].lower()
    modality_names = [m for m in MODALITY_KEYS if m in config["modalities"]]
    subtype_descriptions = SUBTYPE_DESCRIPTIONS[cancer_type]

    print(f"\n{'='*60}")
    print(f"Creating test datasets for {cancer_type}")
    print(f"{'='*60}")

    # --- Demo config (ground truth for feature names) ---
    with open(f"app/model_artifacts/config_{c}.json") as f:
        demo_cfg = json.load(f)
    feature_names = demo_cfg["feature_names"]
    class_names: dict = demo_cfg["class_names"]  # {"0": "Basal-like", ...}
    total_features = demo_cfg["total_features"]

    # --- Best fold val set ---
    best_fold_idx = _find_best_fold(f"results/metrics/xgb_{c}_metrics.csv")
    cv_data = load_cv_folds(cancer_type, config, use_toy=False)
    fold_info = cv_data["folds"][best_fold_idx]
    val_ids = [str(s) for s in fold_info["val"]]
    print(f"  Best fold: {best_fold_idx}  |  val samples: {len(val_ids)}")

    # --- Load and preprocess val raw data ---
    raw = {}
    for mod in modality_names:
        df = load_modality(cancer_type, mod, config, use_toy=False)
        df.columns = df.columns.astype(str)
        df.columns = _make_columns_unique(df.columns)
        raw[mod] = df

    common_all = set.intersection(*[set(raw[m].index) for m in modality_names])
    val_common = [s for s in val_ids if s in common_all]
    print(f"  Val samples with all modalities: {len(val_common)}")

    val_dict = {mod: raw[mod].loc[val_common].copy() for mod in modality_names}

    imputer = joblib.load(f"app/model_artifacts/imputer_{c}.pkl")
    scaler = joblib.load(f"app/model_artifacts/per_modality_scaler_{c}.pkl")
    val_imputed = imputer.transform(val_dict, modality_names)
    val_scaled = scaler.transform(val_imputed, modality_names)

    # Full feature matrix (n_val x total_features), index = TCGA IDs
    val_matrix = np.concatenate([val_scaled[m].values for m in modality_names], axis=1)
    val_df = pd.DataFrame(val_matrix, index=val_common, columns=feature_names)

    # --- True labels ---
    labels = load_labels(cancer_type, config, use_toy=False)
    val_labels = labels.loc[val_common].values  # integer labels

    # --- XGBoost predictions on val set ---
    import joblib as jl
    xgb_model = jl.load(f"app/model_artifacts/xgb_best_{c}.pkl")
    xgb_preds = xgb_model.predict(val_matrix)
    xgb_probs = xgb_model.predict_proba(val_matrix)
    xgb_confidence = xgb_probs[np.arange(len(xgb_preds)), xgb_preds]

    # --- Precomputed attribution IDs (these get the fusion attribution graph) ---
    attr_path = f"app/model_artifacts/fusion_attribution_results_{c}.json"
    precomputed_ids = set()
    if os.path.exists(attr_path):
        with open(attr_path) as f:
            attr_data = json.load(f)
        precomputed_ids = set(attr_data.get("sample_ids", []))
    print(f"  Precomputed attribution IDs: {precomputed_ids}")

    # --- Select samples: n_per_subtype per subtype, prefer high XGB confidence ---
    selected_indices = []   # indices into val_common
    selected_ids_set = set()

    # First pass: include precomputed attribution samples
    for i, sid in enumerate(val_common):
        if sid in precomputed_ids and sid not in selected_ids_set:
            selected_indices.append(i)
            selected_ids_set.add(sid)

    # Second pass: for each subtype, pick up to n_per_subtype highest-confidence samples
    class_idxs = sorted(set(val_labels))
    for cls in class_idxs:
        cls_mask = (val_labels == cls)
        cls_sample_indices = np.where(cls_mask)[0]
        cls_confs = xgb_confidence[cls_mask]
        # Sort by confidence descending
        sorted_by_conf = cls_sample_indices[np.argsort(cls_confs)[::-1]]
        added = sum(1 for idx in selected_indices if val_labels[idx] == cls)
        for idx in sorted_by_conf:
            if added >= n_per_subtype:
                break
            sid = val_common[idx]
            if sid not in selected_ids_set:
                selected_indices.append(idx)
                selected_ids_set.add(sid)
                added += 1

    selected_indices.sort()  # keep chronological order
    print(f"  Total selected samples: {len(selected_indices)}")

    # --- Build batch CSV (sample_id as first column) ---
    rows = []
    metadata_rows = []

    for idx in selected_indices:
        sid = val_common[idx]
        true_cls = int(val_labels[idx])
        true_name = class_names.get(str(true_cls), f"Class {true_cls}")
        xgb_cls = int(xgb_preds[idx])
        xgb_name = class_names.get(str(xgb_cls), f"Class {xgb_cls}")
        xgb_conf = float(xgb_confidence[idx])
        correct = true_cls == xgb_cls
        description = subtype_descriptions.get(true_name, "")

        row = {"sample_id": sid}
        row.update(dict(zip(feature_names, val_df.loc[sid].values)))
        rows.append(row)

        metadata_rows.append({
            "sample_id": sid,
            "true_class_idx": true_cls,
            "true_subtype": true_name,
            "xgb_predicted_subtype": xgb_name,
            "xgb_confidence": round(xgb_conf, 4),
            "xgb_correct": correct,
            "has_fusion_attribution": sid in precomputed_ids,
            "subtype_description": description,
        })
        status = "CORRECT" if correct else "WRONG"
        attr_marker = " [fusion attr]" if sid in precomputed_ids else ""
        print(f"    {sid}: true={true_name}, pred={xgb_name} ({xgb_conf:.1%}) {status}{attr_marker}")

    # Batch CSV
    batch_df = pd.DataFrame(rows)
    batch_path = out_dir / f"test_{c}_batch.csv"
    batch_df.to_csv(batch_path, index=False)
    print(f"\n  Saved: {batch_path.name}  shape={batch_df.shape}")

    # Per-subtype single-sample CSVs (highest-confidence correct prediction)
    print(f"\n  Per-subtype single-sample files:")
    for cls in class_idxs:
        cls_meta = [m for m in metadata_rows if m["true_class_idx"] == cls and m["xgb_correct"]]
        if not cls_meta:
            cls_meta = [m for m in metadata_rows if m["true_class_idx"] == cls]
        if not cls_meta:
            print(f"    Class {cls}: no samples selected — skipping")
            continue
        # Pick highest-confidence
        best = max(cls_meta, key=lambda x: x["xgb_confidence"])
        sid = best["sample_id"]
        subtype_name = best["true_subtype"].replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        single_row = {"sample_id": sid}
        single_row.update(dict(zip(feature_names, val_df.loc[sid].values)))
        single_df = pd.DataFrame([single_row])
        single_path = out_dir / f"test_{c}_single_{subtype_name}.csv"
        single_df.to_csv(single_path, index=False)
        print(f"    Saved: {single_path.name}  (true={best['true_subtype']}, conf={best['xgb_confidence']:.1%})")

    # Metadata companion file
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = out_dir / f"test_{c}_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\n  Saved metadata: {meta_path.name}")

    # Accuracy summary
    n_correct = sum(1 for m in metadata_rows if m["xgb_correct"])
    print(f"\n  XGBoost accuracy on these {len(metadata_rows)} selected samples: "
          f"{n_correct}/{len(metadata_rows)} = {n_correct/len(metadata_rows):.1%}")

    # Class breakdown
    for cls in class_idxs:
        cls_meta = [m for m in metadata_rows if m["true_class_idx"] == cls]
        n_cls_correct = sum(1 for m in cls_meta if m["xgb_correct"])
        cls_name = class_names.get(str(cls), f"Class {cls}")
        print(f"    {cls_name}: {n_cls_correct}/{len(cls_meta)} correct")


def main():
    parser = argparse.ArgumentParser(description="Create realistic test datasets.")
    parser.add_argument("--cancer", default="all", choices=["GS-BRCA", "GS-COAD", "all"])
    parser.add_argument("--n-per-subtype", type=int, default=4,
                        help="Max samples per subtype in batch file (default: 4)")
    args = parser.parse_args()

    set_seeds(42)
    config = load_config()

    out_dir = Path("app") / "test_datasets"
    out_dir.mkdir(exist_ok=True)

    cancer_types = (
        config["project"]["cancer_types"]
        if args.cancer == "all"
        else [args.cancer]
    )

    for cancer_type in cancer_types:
        create_test_datasets_for_cancer(cancer_type, config, out_dir, args.n_per_subtype)

    print(f"\n\nAll test datasets in {out_dir}/:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        suffix = " <-- UPLOAD TO APP" if f.suffix == ".csv" and "metadata" not in f.name else ""
        print(f"  {f.name:<55} {size_kb:>8.1f} KB{suffix}")

    print("""
=======================================================
UPLOAD GUIDE FOR STREAMLIT DEMO:
=======================================================

Single-sample files (test_{cancer}_single_*.csv):
  • 1 row = 1 real TCGA patient
  • XGBoost: shows SHAP waterfall plot
  • Fusion: attribution shown if sample_id is in precomputed list
  • Expected prediction is in the filename (e.g., Luminal_A)

Batch file (test_{cancer}_batch.csv):
  • 16-20 rows covering all subtypes
  • Shows one prediction card per patient
  • Select any row from dropdown for SHAP/attribution
  • Check test_{cancer}_metadata.csv for true labels

Metadata file (test_{cancer}_metadata.csv):
  • NOT for upload — reference only
  • Contains true labels, expected predictions, confidence,
    and clinical descriptions for each sample
=======================================================
""")


if __name__ == "__main__":
    main()
