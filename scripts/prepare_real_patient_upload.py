"""
Prepare Real Patient Omics Data for MLOmics Demo Upload
========================================================
Converts per-modality omics CSVs from a hospital/research pipeline into
a single upload-ready CSV for the MLOmics Streamlit demo.

IMPORTANT — READ BEFORE USING
-------------------------------
This project was trained on the MLOmics benchmark dataset, which provides
data in a pre-normalized z-scored space (NOT raw TPM/beta values).

For real patient data to be compatible, your lab must pre-normalize each
modality to match the MLOmics input convention:

  mRNA (gene expression)
    Expected: log2(TPM + 1) z-scored per-gene across your cohort.
    If you only have a single patient, use population-level z-scoring:
      z = (log2(TPM + 1) − gene_mean_TCGA) / gene_std_TCGA
    Raw units: TPM or FPKM from RNA-seq (STAR/HISAT2 + featureCounts)

  miRNA (microRNA expression)
    Expected: log2(RPM + 1) z-scored, same approach as mRNA.
    Feature name format must use dots: "hsa.let.7a.1", NOT "hsa-let-7a-1"
    Raw units: RPM-normalized counts from small RNA-seq

  DNA Methylation
    Expected: M-values (logit of beta) z-scored per probe.
    Convert: M = log2(beta / (1 - beta)), then z-score.
    Raw units: Beta values (0.0–1.0) from Illumina 450K / EPIC 850K array

  Copy Number Variation (CNV)
    Expected: z-scored log2 copy-number ratios per gene.
    Convert: log2(segment_CN / 2), averaged per gene, then z-score.
    Raw units: Segment-level log2 ratios from SNP array or WGS

If you cannot normalize to this space, the model predictions will be
unreliable.  This script handles everything AFTER normalization is done.

INPUT FORMAT PER MODALITY CSV
-------------------------------
  Each CSV should have:
    - First column = feature names (gene symbols / miRNA IDs / probe IDs)
    - Remaining columns = one column per patient sample
  Example (mRNA):
      gene,PATIENT_001,PATIENT_002
      ESR1,0.83,-0.45
      TP53,-0.12,1.20
      BRCA1,2.10,-0.88

  OR transposed (rows=patients, columns=features) — both are accepted.

USAGE
------
  # Single patient, all four modalities specified:
  python scripts/prepare_real_patient_upload.py \\
      --cancer GS-BRCA \\
      --sample-id PT-HOSP-2026-001 \\
      --mrna   /path/to/patient_mrna.csv \\
      --mirna  /path/to/patient_mirna.csv \\
      --methy  /path/to/patient_methy.csv \\
      --cnv    /path/to/patient_cnv.csv \\
      --output /path/to/patient_upload.csv

  # If a modality is missing (not run by the lab), omit its flag.
  # Missing features are filled with 0 (= training mean in z-scored space).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACT_DIR = os.path.join("app", "model_artifacts")

# miRNA name normalisation: hospital uses "hsa-miR-21-5p" / "hsa-let-7a-1"
# Training data uses "hsa.mir.21" / "hsa.let.7a.1"  (dots, lowercase, no strand suffix)
def _normalise_mirna_name(name: str) -> str:
    """Convert common miRNA naming variations to training-data dot format.

    hsa-miR-21-5p  → hsa.mir.21
    hsa-let-7a-1   → hsa.let.7a.1
    hsa.miR.21.5p  → hsa.mir.21
    """
    name = str(name).strip()
    # Replace dashes and underscores with dots
    name = name.replace("-", ".").replace("_", ".")
    # Lowercase everything
    name = name.lower()
    # Remove strand suffix: .5p / .3p / .5p. etc.
    for suffix in [".5p", ".3p", ".5p.", ".3p."]:
        if name.endswith(suffix.rstrip(".")):
            name = name[: -len(suffix.rstrip("."))]
    # Capitalise "miR" → "mir" already done by lower(); keep as-is
    return name


def _load_modality_file(path: str, sample_id: str) -> pd.Series:
    """Load a modality CSV and return a Series indexed by feature name.

    Accepts two orientations:
      - features-as-rows: index = feature names, columns = sample IDs
      - features-as-cols: index = sample IDs,   columns = feature names
    Detection priority: sample_id presence in columns or index first,
    then falls back to shape heuristic.
    """
    df = pd.read_csv(path, index_col=0)

    # Priority 1: exact sample_id match
    if sample_id in df.columns:
        # features are rows, samples are columns
        return df[sample_id].rename(sample_id)
    if sample_id in df.index:
        # samples are rows, features are columns
        return df.loc[sample_id]

    # Priority 2: shape heuristic (more rows than columns → features-as-rows)
    if df.shape[0] >= df.shape[1]:
        col = df.columns[0]
        print(f"    [info] sample_id '{sample_id}' not found — "
              f"using first column '{col}' (features-as-rows orientation)")
        return df[col].rename(sample_id)
    else:
        row = df.index[0]
        print(f"    [info] sample_id '{sample_id}' not found — "
              f"using first row '{row}' (features-as-cols orientation)")
        return df.loc[row].rename(sample_id)


def build_upload_csv(
    cancer_type: str,
    sample_id: str,
    modality_paths: dict,   # {"mrna": path, "mirna": path, ...}
    output_path: str,
) -> pd.DataFrame:
    """Main conversion function.

    Parameters
    ----------
    cancer_type : str
        "GS-BRCA" or "GS-COAD"
    sample_id : str
        Patient/sample identifier written into the output CSV.
    modality_paths : dict
        Keys are modality names ("mrna", "mirna", "methy", "cnv").
        Values are file paths (str).  Missing modalities are omitted.
    output_path : str
        Where to save the resulting upload-ready CSV.

    Returns
    -------
    pd.DataFrame
        The upload-ready DataFrame (also written to output_path).
    """
    c = cancer_type.split("-")[1].lower()

    # ── Load training config ──────────────────────────────────────────
    config_path = os.path.join(ARTIFACT_DIR, f"config_{c}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Run prepare_demo_artifacts.py first."
        )
    with open(config_path) as f:
        cfg = json.load(f)

    feature_names  = cfg["feature_names"]        # ["mrna_ESR1", "mirna_hsa.mir.21", ...]
    modality_order = cfg["modality_order"]        # ["mrna", "mirna", "methy", "cnv"]
    modality_dims  = cfg["modality_dims"]         # {"mrna": 5000, ...}

    # Build lookup: bare feature name (no prefix) → position in output vector
    # e.g. "ESR1" → index for "mrna_ESR1"
    bare_to_idx = {}
    for idx, fname in enumerate(feature_names):
        bare = fname.split("_", 1)[1] if "_" in fname else fname
        bare_to_idx[bare] = idx

    # ── Initialise output vector with 0 (= training mean in z-scored space) ──
    output_vector = np.zeros(len(feature_names), dtype=np.float64)

    filled_counts = {}

    for mod in modality_order:
        filled_counts[mod] = {"found": 0, "missing": 0, "total": modality_dims[mod]}

        if mod not in modality_paths or modality_paths[mod] is None:
            print(f"  [{mod}]  no file provided — all {modality_dims[mod]} features "
                  f"filled with 0 (training mean)")
            filled_counts[mod]["missing"] = modality_dims[mod]
            continue

        path = modality_paths[mod]
        print(f"  [{mod}]  loading {os.path.basename(path)} …")
        series = _load_modality_file(path, sample_id)

        # Normalise miRNA names in the patient file to match training format
        if mod == "mirna":
            series.index = [_normalise_mirna_name(n) for n in series.index]

        # For each training feature in this modality, look it up in patient data
        start = sum(modality_dims[m] for m in modality_order[:modality_order.index(mod)])
        for idx in range(start, start + modality_dims[mod]):
            bare = feature_names[idx].split("_", 1)[1]
            if bare in series.index:
                val = series[bare]
                output_vector[idx] = float(val) if pd.notna(val) else 0.0
                filled_counts[mod]["found"] += 1
            else:
                # Feature not in patient data — leave as 0
                filled_counts[mod]["missing"] += 1

        pct = 100 * filled_counts[mod]["found"] / modality_dims[mod]
        print(f"         matched {filled_counts[mod]['found']}/{modality_dims[mod]} "
              f"features ({pct:.1f}%)  |  "
              f"missing filled with 0: {filled_counts[mod]['missing']}")

        if pct < 50:
            print(f"  [WARNING] Less than 50% of {mod} features matched.")
            print(f"            Check feature name format in your file.")
            print(f"            Expected format: {list(series.index[:3])}")

    # ── Build output DataFrame ────────────────────────────────────────
    row = {"sample_id": sample_id}
    row.update(dict(zip(feature_names, output_vector)))
    df_out = pd.DataFrame([row])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_out.to_csv(output_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n  Output: {output_path}")
    print(f"  Shape:  {df_out.shape[0]} row × {df_out.shape[1]} cols")
    print(f"\n  Feature coverage summary:")
    for mod, counts in filled_counts.items():
        pct = 100 * counts["found"] / counts["total"]
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"    {mod:6s}  [{bar}] {pct:5.1f}%  "
              f"({counts['found']}/{counts['total']} matched)")

    print(f"\n  Ready to upload to Streamlit demo as cancer type: {cancer_type}")
    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-modality patient omics CSVs to MLOmics upload format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cancer",    required=True,
                        choices=["GS-BRCA", "GS-COAD"],
                        help="Cancer type to classify against")
    parser.add_argument("--sample-id", required=True,
                        help="Patient/sample identifier (e.g. PT-HOSP-2026-001)")
    parser.add_argument("--mrna",  default=None, help="Path to mRNA expression CSV")
    parser.add_argument("--mirna", default=None, help="Path to miRNA expression CSV")
    parser.add_argument("--methy", default=None, help="Path to DNA methylation CSV")
    parser.add_argument("--cnv",   default=None, help="Path to CNV CSV")
    parser.add_argument("--output", required=True,
                        help="Path for output upload-ready CSV")
    args = parser.parse_args()

    modality_paths = {
        "mrna":  args.mrna,
        "mirna": args.mirna,
        "methy": args.methy,
        "cnv":   args.cnv,
    }

    provided = [m for m, p in modality_paths.items() if p is not None]
    if not provided:
        parser.error("Provide at least one modality file (--mrna / --mirna / --methy / --cnv)")

    print(f"\nMLOmics Real Patient Upload Converter")
    print(f"  Cancer    : {args.cancer}")
    print(f"  Sample ID : {args.sample_id}")
    print(f"  Modalities provided: {provided}")
    print(f"  Modalities missing : {[m for m in ['mrna','mirna','methy','cnv'] if m not in provided]}")
    print()

    build_upload_csv(
        cancer_type    = args.cancer,
        sample_id      = args.sample_id,
        modality_paths = modality_paths,
        output_path    = args.output,
    )


if __name__ == "__main__":
    main()
