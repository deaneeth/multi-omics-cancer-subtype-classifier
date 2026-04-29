"""
Create Randomized Synthetic Patient Omics Profiles
====================================================
Generates realistic individual patient profiles by sampling each of the
~15,000 genomic features from the real per-subtype distribution:

    feature_value ~ N(class_mean_f, class_std_f)

This produces profiles with:
  - Correct subtype-level signal (mean matches real class centroid)
  - Realistic patient-to-patient variation (std matches real within-class spread)
  - All four modalities represented: mRNA (5,000), miRNA (366/200),
    Methylation (5,000), CNV (5,000)

Values are in the preprocessed (imputed + z-scored) space, exactly as the
Streamlit demo expects.  These are NOT real patients — they are
statistically grounded synthetic profiles for demo and educational use only.

Output (written to app/test_datasets/):
    synthetic_brca_all.csv           — 50 synthetic BRCA patients (10 per subtype)
    synthetic_brca_<Subtype>.csv     — 10 patients of one BRCA subtype
    synthetic_brca_metadata.csv      — per-row labels, descriptions, modality info
    (same three groups for COAD)

Usage:
    python scripts/create_synthetic_patients.py
    python scripts/create_synthetic_patients.py --cancer GS-BRCA --n-per-subtype 10
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


# ── Subtype metadata ──────────────────────────────────────────────────────────

SUBTYPE_INFO = {
    "GS-BRCA": {
        0: {
            "name": "Basal-like",
            "code": "BL",
            "description": (
                "Triple-negative (ER-/PR-/HER2-) breast cancer. High proliferation, "
                "genomic instability, BRCA1 mutations common. Responds to platinum-based "
                "chemotherapy and PARP inhibitors. ~15-20% of all breast cancers."
            ),
            "key_biology": "BRCA1 pathway, DNA repair deficiency, high Ki-67",
            "treatment": "Chemotherapy (carboplatin/cyclophosphamide), PARP inhibitors",
        },
        1: {
            "name": "HER2-enriched",
            "code": "HER2",
            "description": (
                "HER2 gene amplification (chromosome 17q12) drives uncontrolled growth. "
                "Historically poor prognosis, dramatically improved by trastuzumab (Herceptin) "
                "and pertuzumab. ~15% of breast cancers."
            ),
            "key_biology": "ERBB2 amplification, PIK3CA mutations, RTK/RAS signaling",
            "treatment": "Trastuzumab, pertuzumab, T-DM1 (ado-trastuzumab emtansine)",
        },
        2: {
            "name": "Luminal A",
            "code": "LumA",
            "description": (
                "ER+/PR+, HER2- with low proliferation (Ki-67 < 14%). Most common subtype "
                "(40-50% of breast cancers) and best prognosis. Driven by estrogen receptor "
                "signaling. Excellent response to hormone therapy."
            ),
            "key_biology": "ESR1/PGR expression, CDH1 mutations, GATA3 activation",
            "treatment": "Tamoxifen or aromatase inhibitors (anastrozole/letrozole), CDK4/6 inhibitors",
        },
        3: {
            "name": "Luminal B",
            "code": "LumB",
            "description": (
                "ER+ with high proliferation (Ki-67 >= 14%) or HER2+. More aggressive than "
                "Luminal A, greater genomic instability. ~15-20% of breast cancers. "
                "Often requires chemotherapy in addition to hormone therapy."
            ),
            "key_biology": "ESR1+ but high CDK4/6 activity, frequent PIK3CA/TP53 mutations",
            "treatment": "Hormone therapy + chemotherapy, CDK4/6 inhibitors (palbociclib)",
        },
        4: {
            "name": "Normal-like",
            "code": "NL",
            "description": (
                "Gene expression resembles normal breast tissue with high adipose content. "
                "Intermediate prognosis, ~5-10% of breast cancers. Often classified with "
                "Luminal A clinically. Genomically distinct but treatment is similar."
            ),
            "key_biology": "Adipocyte gene signature, low tumor cellularity, ER+ variable",
            "treatment": "Typically treated as Luminal A; hormone therapy if ER+",
        },
    },
    "GS-COAD": {
        0: {
            "name": "CMS1 (MSI/Immune)",
            "code": "CMS1",
            "description": (
                "Consensus Molecular Subtype 1: microsatellite-unstable (MSI-H), hypermutated "
                "(>12 mutations/Mb), strong immune cell infiltration. Best prognosis among CMS "
                "subtypes. BRAF V600E mutations frequent (~50%). ~14% of colon cancers."
            ),
            "key_biology": "MLH1 silencing (MSI), BRAF V600E, immune hot tumor microenvironment",
            "treatment": "Immune checkpoint inhibitors (pembrolizumab) — FDA approved for MSI-H",
        },
        1: {
            "name": "CMS2 (Canonical)",
            "code": "CMS2",
            "description": (
                "Most common CMS subtype (~37%). Chromosomal instability (CIN), WNT/MYC "
                "pathway activation, APC truncating mutations (>90%). Standard chemotherapy "
                "backbone. Anti-EGFR therapy effective if RAS/RAF wild-type."
            ),
            "key_biology": "APC truncation, WNT activation, chromosomal aneuploidy",
            "treatment": "FOLFOX/FOLFIRI + bevacizumab; anti-EGFR (cetuximab) if RAS WT",
        },
        2: {
            "name": "CMS3 (Metabolic)",
            "code": "CMS3",
            "description": (
                "Metabolic dysregulation, mixed microsatellite status, KRAS mutations (~68%). "
                "Intermediate prognosis. Anti-EGFR therapy typically ineffective due to KRAS. "
                "~13% of colon cancers. Overlaps genomically with CMS1 and CMS2."
            ),
            "key_biology": "KRAS activating mutations, IGF/PI3K pathway, lipid biosynthesis",
            "treatment": "FOLFOX/FOLFIRI + bevacizumab; KRAS inhibitors (sotorasib) for G12C",
        },
        3: {
            "name": "CMS4 (Mesenchymal)",
            "code": "CMS4",
            "description": (
                "Worst prognosis CMS subtype (~23%). Epithelial-to-mesenchymal transition (EMT), "
                "TGF-β pathway activation, high stromal infiltration, metastasis-prone. "
                "High relapse rate. Limited targeted therapy options currently."
            ),
            "key_biology": "TGF-β signaling, VEGF pathway, mesenchymal gene programme (ZEB1/SNAI1)",
            "treatment": "FOLFOX/FOLFIRI + bevacizumab; TGF-β inhibitors (investigational)",
        },
    },
}

MODALITY_DESCRIPTIONS = {
    "mrna":  "mRNA gene expression (RNA-seq TPM, log2-transformed)",
    "mirna": "miRNA expression (small non-coding RNAs regulating gene expression)",
    "methy": "DNA methylation (beta-values, CpG island promoter methylation)",
    "cnv":   "Copy Number Variation (gene-level somatic copy number alterations)",
}


def _make_columns_unique(columns) -> list:
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


def generate_synthetic_patients(
    cancer_type: str,
    config: dict,
    out_dir: Path,
    n_per_subtype: int,
    rng: np.random.Generator,
) -> None:
    c = cancer_type.split("-")[1].lower()
    modality_names = [m for m in MODALITY_KEYS if m in config["modalities"]]
    subtype_info = SUBTYPE_INFO[cancer_type]

    print(f"\n{'='*65}")
    print(f"Generating synthetic patients for {cancer_type}")
    print(f"{'='*65}")

    # ── Load demo config ──────────────────────────────────────────────
    with open(f"app/model_artifacts/config_{c}.json") as f:
        demo_cfg = json.load(f)
    feature_names: list = demo_cfg["feature_names"]
    class_names: dict = demo_cfg["class_names"]
    modality_dims: dict = demo_cfg["modality_dims"]
    total_features: int = demo_cfg["total_features"]

    # ── Load best-fold training data (reference distribution) ─────────
    best_fold_idx = _find_best_fold(f"results/metrics/xgb_{c}_metrics.csv")
    cv_data = load_cv_folds(cancer_type, config, use_toy=False)
    train_ids = [str(s) for s in cv_data["folds"][best_fold_idx]["train"]]
    print(f"  Reference: fold {best_fold_idx} training set")

    raw = {}
    for mod in modality_names:
        df = load_modality(cancer_type, mod, config, use_toy=False)
        df.columns = df.columns.astype(str)
        df.columns = _make_columns_unique(df.columns)
        raw[mod] = df

    common_all = set.intersection(*[set(raw[m].index) for m in modality_names])
    train_common = [s for s in train_ids if s in common_all]

    train_dict = {mod: raw[mod].loc[train_common].copy() for mod in modality_names}
    imputer = joblib.load(f"app/model_artifacts/imputer_{c}.pkl")
    scaler = joblib.load(f"app/model_artifacts/per_modality_scaler_{c}.pkl")
    train_imputed = imputer.transform(train_dict, modality_names)
    train_scaled = scaler.transform(train_imputed, modality_names)

    # Full feature matrix (train_samples × total_features)
    train_matrix = np.concatenate([train_scaled[m].values for m in modality_names], axis=1)
    train_labels = load_labels(cancer_type, config, use_toy=False).loc[train_common].values
    print(f"  Reference matrix: {train_matrix.shape[0]} patients x {train_matrix.shape[1]} features")

    # ── Build modality boundary map for metadata ──────────────────────
    modality_ranges = {}
    start = 0
    for mod in modality_names:
        dim = modality_dims[mod]
        modality_ranges[mod] = (start, start + dim)
        start += dim

    # ── Per-subtype statistics ────────────────────────────────────────
    class_stats = {}
    for cls in sorted(set(train_labels)):
        mask = train_labels == cls
        class_samples = train_matrix[mask]
        class_stats[cls] = {
            "mean": class_samples.mean(axis=0),
            "std": np.maximum(class_samples.std(axis=0), 0.01),  # floor to avoid zero-std features
            "n_real": int(mask.sum()),
        }
        info = subtype_info.get(cls, {})
        print(f"  Class {cls} ({class_names.get(str(cls),'?')}): "
              f"{mask.sum()} real training samples | "
              f"mean feature std = {class_samples.std(axis=0).mean():.3f}")

    # ── Generate synthetic patients ───────────────────────────────────
    all_rows = []
    all_metadata = []

    for cls in sorted(class_stats.keys()):
        mu = class_stats[cls]["mean"]
        sigma = class_stats[cls]["std"]
        n_real = class_stats[cls]["n_real"]
        info = subtype_info.get(cls, {})
        subtype_name = info.get("name", class_names.get(str(cls), f"Class{cls}"))
        subtype_code = info.get("code", f"C{cls}")

        print(f"\n  Generating {n_per_subtype} synthetic {subtype_name} patients...")

        for patient_num in range(1, n_per_subtype + 1):
            # Draw each feature independently from the class distribution.
            # clip to ±4 std from the global mean (realistic z-score range).
            patient_features = rng.normal(mu, sigma)
            patient_features = np.clip(patient_features, -4.5, 4.5)

            # Patient ID: SYN.<CANCER>.<SUBTYPE_CODE>.<NNN>
            cancer_tag = cancer_type.split("-")[1]  # BRCA or COAD
            patient_id = f"SYN.{cancer_tag}.{subtype_code}.{patient_num:03d}"

            row = {"sample_id": patient_id}
            row.update(dict(zip(feature_names, patient_features)))
            all_rows.append(row)

            # Modality-level summary stats for metadata
            mod_summaries = {}
            for mod in modality_names:
                s, e = modality_ranges[mod]
                vals = patient_features[s:e]
                mod_summaries[mod] = f"mean={vals.mean():.3f} std={vals.std():.3f}"

            all_metadata.append({
                "sample_id":            patient_id,
                "cancer_type":          cancer_type,
                "subtype_class":        cls,
                "subtype_name":         subtype_name,
                "n_real_training_ref":  n_real,
                "is_synthetic":         True,
                "total_features":       total_features,
                **{f"{mod}_summary": mod_summaries[mod] for mod in modality_names},
                "key_biology":          info.get("key_biology", ""),
                "treatment_context":    info.get("treatment", ""),
                "subtype_description":  info.get("description", ""),
            })

        print(f"    Patient IDs: {cancer_tag}.{subtype_code}.001 ... {cancer_tag}.{subtype_code}.{n_per_subtype:03d}")

    # ── Write per-subtype CSVs ────────────────────────────────────────
    print(f"\n  Writing per-subtype CSVs...")
    subtype_dfs = {}
    for cls in sorted(class_stats.keys()):
        info = subtype_info.get(cls, {})
        subtype_name = info.get("name", f"Class{cls}")
        subtype_code = info.get("code", f"C{cls}")
        safe_name = subtype_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        cls_rows = [r for r in all_rows if r["sample_id"].split(".")[2] == subtype_code]
        cls_df = pd.DataFrame(cls_rows)
        subtype_dfs[cls] = cls_df

        fpath = out_dir / f"synthetic_{c}_{safe_name}.csv"
        cls_df.to_csv(fpath, index=False)
        print(f"    {fpath.name}: {cls_df.shape[0]} rows x {cls_df.shape[1]} cols")

    # ── Write combined CSV (all subtypes) ────────────────────────────
    combined_df = pd.DataFrame(all_rows)
    combined_path = out_dir / f"synthetic_{c}_all.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\n  Combined: {combined_path.name}: {combined_df.shape[0]} rows x {combined_df.shape[1]} cols")

    # ── Write metadata CSV ────────────────────────────────────────────
    meta_df = pd.DataFrame(all_metadata)
    meta_path = out_dir / f"synthetic_{c}_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"  Metadata: {meta_path.name}: {meta_df.shape}")

    # ── Sanity checks ─────────────────────────────────────────────────
    print(f"\n  Sanity checks:")
    for cls in sorted(class_stats.keys()):
        info = subtype_info.get(cls, {})
        code = info.get("code", f"C{cls}")
        cls_df = subtype_dfs[cls]
        feat_vals = cls_df.iloc[:, 1:].values.astype(float)
        real_mu = class_stats[cls]["mean"]
        synth_mu = feat_vals.mean(axis=0)
        corr = np.corrcoef(real_mu, synth_mu)[0, 1]
        print(f"    {info.get('name','?')}: synthetic mean ~ real mean (correlation={corr:.4f}), "
              f"value range=[{feat_vals.min():.2f}, {feat_vals.max():.2f}]")

    assert combined_df.isna().sum().sum() == 0, "NaN in synthetic patients!"
    assert list(combined_df.columns[1:]) == feature_names, "Column mismatch!"
    print(f"\n  All assertions passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate randomized synthetic patient omics profiles."
    )
    parser.add_argument("--cancer", default="all", choices=["GS-BRCA", "GS-COAD", "all"])
    parser.add_argument(
        "--n-per-subtype", type=int, default=10,
        help="Synthetic patients to generate per subtype (default: 10)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    config = load_config()

    out_dir = Path("app") / "test_datasets"
    out_dir.mkdir(exist_ok=True)

    cancer_types = (
        config["project"]["cancer_types"]
        if args.cancer == "all"
        else [args.cancer]
    )

    for cancer_type in cancer_types:
        generate_synthetic_patients(cancer_type, config, out_dir, args.n_per_subtype, rng)

    # ── Final listing ──────────────────────────────────────────────────
    print(f"\n\nAll files in app/test_datasets/:")
    print(f"{'File':<55} {'Size':>9}  {'Type'}")
    print("-" * 78)
    for f in sorted(out_dir.iterdir()):
        kb = f.stat().st_size / 1024
        if "synthetic" in f.name and "metadata" not in f.name:
            ftype = "UPLOAD to app (synthetic patients)"
        elif "test_" in f.name and "metadata" not in f.name:
            ftype = "UPLOAD to app (real TCGA patients)"
        else:
            ftype = "Reference only (do not upload)"
        print(f"  {f.name:<53} {kb:>8.1f}KB  {ftype}")

    print(f"""
=================================================================
WHAT THESE FILES ARE
=================================================================

SYNTHETIC patients (synthetic_*.csv):
  - Generated by sampling each of ~15,000 features from
    N(class_mean, class_std) of real TCGA training data
  - 1 row = 1 synthetic patient, {args.n_per_subtype} per subtype
  - Sample IDs: SYN.BRCA.BL.001, SYN.BRCA.HER2.001, etc.
  - These have correct subtype-level signal so the model
    SHOULD predict the right subtype with good confidence

REAL TCGA patients (test_*_batch.csv, test_*_single_*.csv):
  - Actual de-identified TCGA validation samples
  - TCGA IDs: TCGA.XX.YYYY.01 format
  - 3 BRCA and 3 COAD samples have Integrated Gradients
    attribution graphs available in the Fusion model view

WHAT TO EXPECT:
  - High-confidence subtypes (Basal-like, HER2, CMS1, CMS2):
    >90% predicted correctly
  - Harder subtypes (Luminal B, CMS3): 60-75% accuracy
    (these overlap genomically with other subtypes)
  - CMS4 Mesenchymal: only 4 training samples total --
    model has very limited data to learn this subtype

MODALITY BREAKDOWN (feature prefix in column names):
  - mrna_*   : {5000} features | mRNA gene expression
  - mirna_*  : 366/200 features | microRNA expression
  - methy_*  : {5000} features | DNA methylation
  - cnv_*    : {5000} features | Copy Number Variation
=================================================================
""")


if __name__ == "__main__":
    main()
