"""
Create Two Realistic Demo Patient Datasets for Hospital Scenario
================================================================
Generates ONE individual patient CSV per cancer type, sampled from the real
per-subtype distributions in z-scored space (same methodology as
create_synthetic_patients.py).

Patient profiles:
  BRCA  — Sarah Mitchell, 47 F, Luminal A breast cancer
  COAD  — Robert Okonkwo, 62 M, CMS2 Canonical colon cancer

These are NOT real patients.  All omics values are statistically derived
from real TCGA training distributions.

Output:
    app/test_datasets/demo_patient_sarah_mitchell_BRCA.csv
    app/test_datasets/demo_patient_robert_okonkwo_COAD.csv

Usage:
    python scripts/create_demo_patients.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MODALITY_KEYS, load_labels, load_modality
from src.preprocessing import load_cv_folds
from src.utils import load_config, set_seeds

set_seeds(7)   # fixed seed so the profiles are reproducible

# ── Patient scenario definitions ──────────────────────────────────────────────

PATIENTS = {
    "GS-BRCA": {
        "target_class": 2,           # Luminal A  (class index in config)
        "patient_id":   "PT-2026-BRCA-0047",
        "name":         "Sarah Mitchell",
        "age":          47,
        "sex":          "Female",
        "clinical_notes": (
            "47-year-old schoolteacher. Presented after noticing a firm, non-tender "
            "lump in the upper-outer quadrant of the right breast during routine self-"
            "examination. Mammogram and ultrasound confirmed a 1.8 cm irregular mass; "
            "biopsy performed. Family history: maternal aunt diagnosed with breast "
            "cancer at age 54. No prior malignancy. ECOG performance status 0. "
            "Referred to the research lab for full multi-omics profiling to guide "
            "treatment selection."
        ),
        "filename": "demo_patient_sarah_mitchell_BRCA.csv",
    },
    "GS-COAD": {
        "target_class": 1,           # CMS2 Canonical  (class index in config)
        "patient_id":   "PT-2026-COAD-0062",
        "name":         "Robert Okonkwo",
        "age":          62,
        "sex":          "Male",
        "clinical_notes": (
            "62-year-old retired civil engineer. Presented with a 6-week history of "
            "intermittent rectal bleeding, altered bowel habits (alternating constipation "
            "and diarrhoea), and unintentional 4 kg weight loss over 3 months. "
            "Colonoscopy revealed a 3.2 cm semi-circumferential lesion at the "
            "recto-sigmoid junction; biopsy confirmed moderately differentiated "
            "adenocarcinoma. CT staging: T3 N1 M0 (Stage IIIB). No family history of "
            "colorectal cancer. Referred for multi-omics profiling to characterise "
            "molecular subtype and optimise adjuvant therapy planning."
        ),
        "filename": "demo_patient_robert_okonkwo_COAD.csv",
    },
}


def _make_columns_unique(columns) -> list:
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


def _find_best_fold(metrics_path: str) -> int:
    df = pd.read_csv(metrics_path)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.dropna(subset=["f1"])
    df = df[~df["fold"].astype(str).str.contains(r"\+/-", na=False)]
    return int(df.loc[df["f1"].idxmax(), "fold"])


def generate_patient(cancer_type: str, config: dict, out_dir: Path) -> Path:
    spec = PATIENTS[cancer_type]
    c = cancer_type.split("-")[1].lower()
    modality_names = [m for m in MODALITY_KEYS if m in config["modalities"]]

    print(f"\n{'='*65}")
    print(f"Generating patient: {spec['name']}  ({cancer_type})")
    print(f"  Target subtype class index: {spec['target_class']}")
    print(f"{'='*65}")

    # ── Load demo config ──────────────────────────────────────────────
    with open(f"app/model_artifacts/config_{c}.json") as f:
        demo_cfg = json.load(f)
    feature_names: list = demo_cfg["feature_names"]
    class_names: dict  = demo_cfg["class_names"]

    print(f"  Target subtype: {class_names[str(spec['target_class'])]}")

    # ── Load best-fold training data as reference distribution ────────
    best_fold = _find_best_fold(f"results/metrics/xgb_{c}_metrics.csv")
    cv_data   = load_cv_folds(cancer_type, config, use_toy=False)
    train_ids = [str(s) for s in cv_data["folds"][best_fold]["train"]]

    raw = {}
    for mod in modality_names:
        df = load_modality(cancer_type, mod, config, use_toy=False)
        df.columns = df.columns.astype(str)
        df.columns = _make_columns_unique(df.columns)
        raw[mod] = df

    common_all   = set.intersection(*[set(raw[m].index) for m in modality_names])
    train_common = [s for s in train_ids if s in common_all]

    train_dict    = {mod: raw[mod].loc[train_common].copy() for mod in modality_names}
    imputer       = joblib.load(f"app/model_artifacts/imputer_{c}.pkl")
    scaler        = joblib.load(f"app/model_artifacts/per_modality_scaler_{c}.pkl")
    train_imputed = imputer.transform(train_dict, modality_names)
    train_scaled  = scaler.transform(train_imputed, modality_names)

    train_matrix = np.concatenate(
        [train_scaled[m].values for m in modality_names], axis=1
    )
    train_labels = load_labels(
        cancer_type, config, use_toy=False
    ).loc[train_common].values

    print(f"  Reference: fold {best_fold} training set — "
          f"{train_matrix.shape[0]} patients × {train_matrix.shape[1]} features")

    # ── Compute per-class mean & std ──────────────────────────────────
    cls = spec["target_class"]
    mask   = train_labels == cls
    subset = train_matrix[mask]
    mu     = subset.mean(axis=0)
    sigma  = np.maximum(subset.std(axis=0), 0.01)
    print(f"  Reference class size: {mask.sum()} training patients")

    # ── Sample one patient from class distribution ────────────────────
    rng = np.random.default_rng(seed=7)
    patient_features = rng.normal(mu, sigma)
    patient_features = np.clip(patient_features, -4.5, 4.5)

    # ── Build upload-ready CSV ────────────────────────────────────────
    row = {"sample_id": spec["patient_id"]}
    row.update(dict(zip(feature_names, patient_features)))
    df_out = pd.DataFrame([row])

    out_path = out_dir / spec["filename"]
    df_out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({df_out.shape[0]} row × {df_out.shape[1]} cols)")

    # ── Print clinical card ───────────────────────────────────────────
    print(f"\n  -- Clinical card --")
    print(f"  Patient ID : {spec['patient_id']}")
    print(f"  Name       : {spec['name']}")
    print(f"  Age / Sex  : {spec['age']} / {spec['sex']}")
    print(f"  Cancer     : {cancer_type}")
    print(f"  Expected   : {class_names[str(cls)]}")
    print(f"  Notes      : {spec['clinical_notes'][:120]}...")

    return out_path


def main():
    config  = load_config()
    out_dir = Path("app/test_datasets/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for cancer_type in ["GS-BRCA", "GS-COAD"]:
        p = generate_patient(cancer_type, config, out_dir)
        paths.append(p)

    print(f"\n{'='*65}")
    print("DONE — upload these two files to the Streamlit demo:")
    for p in paths:
        print(f"  {p}")
    print()
    print("Patient 1 — Sarah Mitchell (BRCA)")
    print("  Expected subtype: Luminal A")
    print("  Clinical context: ER+/PR+ low-proliferation breast cancer")
    print("  Recommended: Hormone therapy (tamoxifen/aromatase inhibitor)")
    print()
    print("Patient 2 — Robert Okonkwo (COAD)")
    print("  Expected subtype: CMS2 Canonical")
    print("  Clinical context: Chromosomally unstable, WNT/MYC activated colon cancer")
    print("  Recommended: FOLFOX/CAPOX ± bevacizumab; high adjuvant chemo benefit")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
