"""Verify label file positional alignment against mRNA feature files.

The MLOmics label CSVs have no sample-ID column. Labels are assigned to
sample IDs by assuming row i in the label file corresponds to column i of
the transposed mRNA CSV. This script verifies:

  1. Label count matches mRNA sample count for each cancer type.
  2. Class distribution matches documented values from PROJECT_SNAPSHOT.
  3. SHA-256 of each label file is recorded in data/label_file_checksums.json.

Run from the project root:
    python scripts/verify_label_alignment.py
    python scripts/verify_label_alignment.py --toy   # toy data only (always available)
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


DOCUMENTED_DISTRIBUTIONS = {
    "GS-BRCA": {0: 353, 1: 42, 2: 132, 3: 31, 4: 113},
    "GS-COAD": {0: 174, 1: 48, 2: 34, 3: 4},
}

DOCUMENTED_SAMPLE_COUNTS = {
    "GS-BRCA": 671,
    "GS-COAD": 260,
}


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_cancer(cancer_type: str, config: dict, use_toy: bool) -> tuple[bool, str | None]:
    cancer_short = cancer_type.split("-")[1]
    passed = True

    # ── Locate label file ──────────────────────────────────────────────────
    if use_toy:
        label_path = os.path.join(
            config["paths"]["toy_data"],
            f"{cancer_short}_label_num_toy.csv",
        )
        mrna_pattern = os.path.join(
            config["paths"]["toy_data"],
            f"{cancer_short}_mRNA_top_toy.csv",
        )
        mrna_path = mrna_pattern
    else:
        label_pattern = os.path.join(
            config["paths"]["raw_data"],
            cancer_type,
            "Top",
            f"*_{cancer_short}_label_num.csv",
        )
        matches = glob.glob(label_pattern)
        if not matches:
            print(f"  SKIP: No label file found at {label_pattern}")
            return True, None  # not a failure if raw data is absent
        label_path = matches[0]

        mrna_pattern = os.path.join(
            config["paths"]["raw_data"],
            cancer_type,
            "Top",
            f"*_{cancer_short}_mRNA_top.csv",
        )
        mrna_matches = glob.glob(mrna_pattern)
        if not mrna_matches:
            print(f"  SKIP: No mRNA file found for {cancer_type}")
            return True, None
        mrna_path = mrna_matches[0]

    if not os.path.exists(label_path):
        if use_toy:
            print(f"  SKIP: No toy label file for {cancer_type} at {label_path}")
            return True, None
        return True, None

    # ── Load files ─────────────────────────────────────────────────────────
    labels_df = pd.read_csv(label_path)
    mrna_df = pd.read_csv(mrna_path, index_col=0)
    # mRNA CSV: features as rows, samples as columns → transpose
    mrna_df = mrna_df.T
    n_samples = len(mrna_df)

    tag = "toy" if use_toy else "full"
    print(f"\n[{cancer_type} / {tag}]")

    # ── Check 1: row count ─────────────────────────────────────────────────
    if len(labels_df) != n_samples:
        print(
            f"  FAIL count mismatch: label rows={len(labels_df)}, "
            f"mRNA samples={n_samples}"
        )
        passed = False
    else:
        print(f"  PASS count: {len(labels_df)} labels == {n_samples} mRNA samples")

    # ── Check 2: class distribution (full data only) ───────────────────────
    if not use_toy and cancer_type in DOCUMENTED_DISTRIBUTIONS:
        observed = labels_df["Label"].value_counts().sort_index().to_dict()
        expected = DOCUMENTED_DISTRIBUTIONS[cancer_type]
        if observed == expected:
            print(f"  PASS distribution matches documented: {expected}")
        else:
            print(f"  FAIL distribution mismatch!")
            print(f"    Expected: {expected}")
            print(f"    Observed: {observed}")
            passed = False

        # Also check total
        if len(labels_df) != DOCUMENTED_SAMPLE_COUNTS.get(cancer_type, -1):
            print(
                f"  WARN total count {len(labels_df)} != "
                f"documented {DOCUMENTED_SAMPLE_COUNTS.get(cancer_type)}"
            )

    elif use_toy:
        dist = labels_df["Label"].value_counts().sort_index().to_dict()
        print(f"  INFO toy distribution: {dist} (full-data check skipped for toy)")

    return passed, str(Path(label_path).as_posix())


def main():
    parser = argparse.ArgumentParser(description="Verify label-to-sample alignment.")
    parser.add_argument("--toy", action="store_true", help="Verify toy data only")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    all_passed = True
    checksums = {}

    cancers = ["GS-BRCA"]  # Only BRCA has toy data; add GS-COAD when full data present
    if not args.toy:
        cancers = ["GS-BRCA", "GS-COAD"]

    for cancer in cancers:
        passed, label_path = verify_cancer(cancer, config, use_toy=args.toy)
        if not passed:
            all_passed = False
        if label_path:
            checksums[f"{cancer}_{'toy' if args.toy else 'full'}"] = {
                "path": Path(label_path).as_posix(),
                "sha256": sha256_file(label_path),
            }

    # ── Save checksums ─────────────────────────────────────────────────────
    checksum_path = "data/label_file_checksums.json"
    existing = {}
    if os.path.exists(checksum_path):
        with open(checksum_path) as f:
            existing = json.load(f)
    existing.update(checksums)
    with open(checksum_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nChecksums saved to {checksum_path}")
    for key, val in checksums.items():
        print(f"  {key}: {val['sha256'][:16]}... ({val['path']})")

    print()
    if all_passed:
        print("PASS: All label alignment checks passed.")
        sys.exit(0)
    else:
        print("FAIL: One or more label alignment checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
