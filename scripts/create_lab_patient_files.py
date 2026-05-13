"""
Generate Hospital-Style Modality Files for MLOmics Data Converter Testing
==========================================================================
Creates 4 separate per-modality CSV files mimicking real lab exports:
  - mRNA expression   (gene symbols, z-scored log2 TPM)
  - miRNA expression  (miRBase IDs with dash format e.g. hsa-miR-21-5p)
  - DNA methylation   (gene-linked probe names, z-scored M-values)
  - Copy number (CNV) (gene symbols, z-scored log2 CN ratios)

Patient: Amara Nwosu — 38F, triple-negative breast cancer (BRCA, Basal-like)

These files are meant to be uploaded into the MLOmics Data Converter tab:
  - mRNA / miRNA / Methylation / CNV uploaded separately
  - Converter maps to training features and produces one ready-to-predict CSV
  - That CSV is then uploaded in the Prediction tab

Usage:
    python scripts/create_lab_patient_files.py
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

set_seeds(13)   # different seed to get a distinct patient from the demo ones

# ── Patient definition ────────────────────────────────────────────────────────

PATIENT_ID   = "PT-BRCA-LAB-2026-038F"
PATIENT_NAME = "Amara Nwosu"
CANCER_TYPE  = "GS-BRCA"
TARGET_CLASS = 0          # Basal-like

CLINICAL_CONTEXT = """
Patient: Amara Nwosu, 38 years old, Female
Diagnosis: Invasive ductal carcinoma of the left breast
IHC results: ER negative, PR negative, HER2 negative (Triple-Negative)
Clinical stage: T2 N1 M0 (Stage IIB)
Ki-67 proliferation index: 74% (high)
BRCA1 germline mutation detected on genetic panel
Referred for full multi-omics profiling for subtype confirmation
and treatment planning (PARP inhibitor eligibility assessment)
"""

OUT_DIR = Path("app/test_datasets/demo")


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _mirna_to_lab_format(name: str) -> str:
    """Convert training miRNA name (dots) to lab miRBase format (dashes).

    hsa.let.7a.1  -> hsa-let-7a-1
    hsa.mir.21    -> hsa-miR-21        (capitalise 'R' for miR family)
    hsa.mir.21.5p -> hsa-miR-21-5p    (re-add strand suffix if present)
    """
    name = str(name).strip()
    parts = name.split(".")
    if len(parts) < 3:
        return name.replace(".", "-")

    org  = parts[0]      # hsa
    fam  = parts[1]      # let or mir
    rest = parts[2:]     # ['7a', '1'] or ['21'] etc.

    fam_dash = "miR" if fam == "mir" else fam
    return f"{org}-{fam_dash}-{'-'.join(rest)}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    c      = CANCER_TYPE.split("-")[1].lower()
    mod_names = [m for m in MODALITY_KEYS if m in config["modalities"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Generating lab files for: {PATIENT_NAME}")
    print(f"Cancer: {CANCER_TYPE}  |  Target class: {TARGET_CLASS} (Basal-like)")
    print("=" * 65)
    print(CLINICAL_CONTEXT)

    # ── Load demo config ──────────────────────────────────────────────
    with open(f"app/model_artifacts/config_{c}.json") as f:
        demo_cfg = json.load(f)
    feature_names  = demo_cfg["feature_names"]
    class_names    = demo_cfg["class_names"]
    modality_order = demo_cfg["modality_order"]
    modality_dims  = demo_cfg["modality_dims"]

    print(f"Target subtype: {class_names[str(TARGET_CLASS)]}")

    # ── Load training data reference ──────────────────────────────────
    best_fold    = _find_best_fold(f"results/metrics/xgb_{c}_metrics.csv")
    cv_data      = load_cv_folds(CANCER_TYPE, config, use_toy=False)
    train_ids    = [str(s) for s in cv_data["folds"][best_fold]["train"]]

    raw = {}
    for mod in mod_names:
        df = load_modality(CANCER_TYPE, mod, config, use_toy=False)
        df.columns = df.columns.astype(str)
        df.columns = _make_columns_unique(df.columns)
        raw[mod] = df

    common_all   = set.intersection(*[set(raw[m].index) for m in mod_names])
    train_common = [s for s in train_ids if s in common_all]

    imputer      = joblib.load(f"app/model_artifacts/imputer_{c}.pkl")
    scaler       = joblib.load(f"app/model_artifacts/per_modality_scaler_{c}.pkl")
    train_dict   = {mod: raw[mod].loc[train_common].copy() for mod in mod_names}
    t_imputed    = imputer.transform(train_dict, mod_names)
    t_scaled     = scaler.transform(t_imputed, mod_names)

    train_matrix = np.concatenate(
        [t_scaled[m].values for m in mod_names], axis=1
    )
    train_labels = load_labels(
        CANCER_TYPE, config, use_toy=False
    ).loc[train_common].values

    # ── Class statistics for Basal-like ──────────────────────────────
    mask   = train_labels == TARGET_CLASS
    subset = train_matrix[mask]
    mu     = subset.mean(axis=0)
    sigma  = np.maximum(subset.std(axis=0), 0.01)
    print(f"Reference: {mask.sum()} Basal-like training patients (fold {best_fold})")

    # ── Sample one patient ────────────────────────────────────────────
    rng         = np.random.default_rng(seed=13)
    patient_vec = rng.normal(mu, sigma)
    patient_vec = np.clip(patient_vec, -4.5, 4.5)

    # ── Split into per-modality arrays ────────────────────────────────
    start = 0
    modality_arrays = {}
    for mod in modality_order:
        dim = modality_dims[mod]
        modality_arrays[mod] = patient_vec[start:start + dim]
        start += dim

    # ── Build per-modality feature name lists (no prefix) ────────────
    start = 0
    modality_bare_names = {}
    for mod in modality_order:
        dim   = modality_dims[mod]
        block = feature_names[start:start + dim]
        modality_bare_names[mod] = [
            n.split("_", 1)[1] if "_" in n else n for n in block
        ]
        start += dim

    files_written = []

    # ─────────────────────────────────────────────────────────────────
    # 1. mRNA  —  gene symbol rows, patient column
    # ─────────────────────────────────────────────────────────────────
    mrna_df = pd.DataFrame(
        {PATIENT_ID: modality_arrays["mrna"]},
        index=modality_bare_names["mrna"],
    )
    mrna_df.index.name = "gene_symbol"
    mrna_path = OUT_DIR / f"{PATIENT_NAME.replace(' ', '_').lower()}_mrna.csv"
    mrna_df.to_csv(mrna_path)
    files_written.append(("mRNA", mrna_path, mrna_df.shape))
    print(f"\n[mRNA]  saved: {mrna_path.name}  "
          f"({mrna_df.shape[0]} genes x {mrna_df.shape[1]} sample)")
    print(f"  Sample values:  {modality_bare_names['mrna'][:3]}  "
          f"->  {modality_arrays['mrna'][:3].round(4).tolist()}")

    # ─────────────────────────────────────────────────────────────────
    # 2. miRNA  —  miRBase dash-format IDs, skip nan_ probes
    # ─────────────────────────────────────────────────────────────────
    mirna_bare   = modality_bare_names["mirna"]
    mirna_values = modality_arrays["mirna"]

    # Keep only real named probes (exclude nan_*)
    keep_mask  = [not n.startswith("nan") for n in mirna_bare]
    real_names = [n for n, k in zip(mirna_bare, keep_mask) if k]
    real_vals  = mirna_values[np.array(keep_mask)]

    # Convert to lab miRBase format (dots -> dashes)
    lab_names = [_mirna_to_lab_format(n) for n in real_names]

    mirna_df = pd.DataFrame(
        {PATIENT_ID: real_vals},
        index=lab_names,
    )
    mirna_df.index.name = "mirna_id"
    mirna_path = OUT_DIR / f"{PATIENT_NAME.replace(' ', '_').lower()}_mirna.csv"
    mirna_df.to_csv(mirna_path)
    files_written.append(("miRNA", mirna_path, mirna_df.shape))
    print(f"\n[miRNA] saved: {mirna_path.name}  "
          f"({mirna_df.shape[0]} miRNAs x {mirna_df.shape[1]} sample)")
    print(f"  Lab-format names (first 3): {lab_names[:3]}")
    print(f"  Values:                     {real_vals[:3].round(4).tolist()}")
    print(f"  Note: {sum(not k for k in keep_mask)} nan_ probes excluded "
          f"(will be zero-filled by converter)")

    # ─────────────────────────────────────────────────────────────────
    # 3. DNA Methylation  —  gene symbol rows
    # ─────────────────────────────────────────────────────────────────
    methy_df = pd.DataFrame(
        {PATIENT_ID: modality_arrays["methy"]},
        index=modality_bare_names["methy"],
    )
    methy_df.index.name = "gene_symbol"
    methy_path = OUT_DIR / f"{PATIENT_NAME.replace(' ', '_').lower()}_methylation.csv"
    methy_df.to_csv(methy_path)
    files_written.append(("Methylation", methy_path, methy_df.shape))
    print(f"\n[Methy] saved: {methy_path.name}  "
          f"({methy_df.shape[0]} probes x {methy_df.shape[1]} sample)")
    print(f"  Sample probes: {modality_bare_names['methy'][:3]}  "
          f"->  {modality_arrays['methy'][:3].round(4).tolist()}")

    # ─────────────────────────────────────────────────────────────────
    # 4. Copy Number Variation  —  gene symbol rows
    # ─────────────────────────────────────────────────────────────────
    cnv_df = pd.DataFrame(
        {PATIENT_ID: modality_arrays["cnv"]},
        index=modality_bare_names["cnv"],
    )
    cnv_df.index.name = "gene_symbol"
    cnv_path = OUT_DIR / f"{PATIENT_NAME.replace(' ', '_').lower()}_cnv.csv"
    cnv_df.to_csv(cnv_path)
    files_written.append(("CNV", cnv_path, cnv_df.shape))
    print(f"\n[CNV]   saved: {cnv_path.name}  "
          f"({cnv_df.shape[0]} genes x {cnv_df.shape[1]} sample)")
    print(f"  Sample genes:  {modality_bare_names['cnv'][:3]}  "
          f"->  {modality_arrays['cnv'][:3].round(4).tolist()}")

    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("LAB FILES READY")
    print(f"{'=' * 65}")
    print(f"Patient      : {PATIENT_NAME}  ({PATIENT_ID})")
    print(f"Cancer type  : {CANCER_TYPE}")
    print(f"Expected     : {class_names[str(TARGET_CLASS)]} "
          f"(ground truth — do NOT reveal until after prediction)")
    print()
    for label, path, shape in files_written:
        print(f"  {label:12s}  {path.name}  ({shape[0]} features)")
    print()
    print("Next steps:")
    print("  1. Open the MLOmics Streamlit app")
    print("  2. Go to the 'Data Converter' tab")
    print("  3. Select Cancer Type: GS-BRCA, Sample ID: " + PATIENT_ID)
    print("  4. Upload all 4 files in the correct slots")
    print("  5. Click 'Convert & Prepare File' and download the output CSV")
    print("  6. Go to the 'Prediction' tab and upload the downloaded CSV")
    print("  7. Compare the prediction to the ground truth above")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
