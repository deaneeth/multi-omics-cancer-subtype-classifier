"""
Validate Demo Artifacts
========================
Checks that all required demo artifacts exist, have the expected structure,
and are internally consistent (feature counts, config keys, sample input columns).

Usage:
    python scripts/validate_artifacts.py          # both cancers
    python scripts/validate_artifacts.py --cancer GS-BRCA
    python scripts/validate_artifacts.py --cancer GS-COAD

Exit code 0 = all checks passed. Non-zero = one or more failures.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REQUIRED_CONFIG_KEYS = {
    "cancer_type",
    "n_classes",
    "class_labels",
    "class_names",
    "feature_names",
    "total_features",
    "modality_names",
    "modality_sizes",
    "modality_dims",
    "modality_order",
    "xgb_best_fold",
    "fusion_best_fold",
    "fusion_latent_dim",
    "fusion_encoder_hidden",
    "fusion_classifier_hidden",
    "fusion_dropout",
}

REQUIRED_FILES = {
    "brca": [
        "config_brca.json",
        "imputer_brca.pkl",
        "per_modality_scaler_brca.pkl",
        "xgb_best_brca.pkl",
        "fusion_best_brca.pt",
    ],
    "coad": [
        "config_coad.json",
        "imputer_coad.pkl",
        "per_modality_scaler_coad.pkl",
        "xgb_best_coad.pkl",
        "fusion_best_coad.pt",
    ],
}

CANCER_SAMPLE_INPUTS = {
    "brca": "app/sample_input_brca.csv",
    "coad": "app/sample_input_coad.csv",
}


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return passed


def validate_cancer(c: str, artifact_dir: str) -> list:
    """Run all checks for one cancer short code ('brca' or 'coad'). Returns list of failures."""
    failures = []
    print(f"\n{'='*55}")
    print(f"  {c.upper()} artifact validation")
    print(f"{'='*55}")

    # 1. Required files exist
    for fname in REQUIRED_FILES[c]:
        fpath = os.path.join(artifact_dir, fname)
        ok = check(f"exists: {fname}", os.path.exists(fpath))
        if not ok:
            failures.append(f"Missing file: {fpath}")

    # 2. Config JSON structure
    config_path = os.path.join(artifact_dir, f"config_{c}.json")
    if not os.path.exists(config_path):
        failures.append(f"Cannot validate config — {config_path} missing")
        return failures

    with open(config_path) as f:
        cfg = json.load(f)

    missing_keys = REQUIRED_CONFIG_KEYS - set(cfg.keys())
    ok = check("config keys complete", not missing_keys, f"missing: {missing_keys}" if missing_keys else "")
    if not ok:
        failures.append(f"config_{c}.json missing keys: {missing_keys}")

    # 3. Feature count consistency
    declared_total = cfg.get("total_features", -1)
    actual_feature_names = cfg.get("feature_names", [])
    ok = check(
        "feature_names length == total_features",
        len(actual_feature_names) == declared_total,
        f"{len(actual_feature_names)} vs declared {declared_total}",
    )
    if not ok:
        failures.append(f"config_{c}.json feature count mismatch")

    # 4. Modality dims sum to total features
    modality_dims = cfg.get("modality_dims", {})
    dim_sum = sum(modality_dims.values())
    ok = check(
        "sum(modality_dims) == total_features",
        dim_sum == declared_total,
        f"{dim_sum} vs {declared_total}",
    )
    if not ok:
        failures.append(f"config_{c}.json modality dim sum mismatch")

    # 5. Sample input CSV column alignment
    sample_path = CANCER_SAMPLE_INPUTS.get(c)
    if sample_path and os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        col_match = list(sample_df.columns) == actual_feature_names
        ok = check(
            f"sample_input_{c}.csv columns match config feature_names",
            col_match,
            f"{sample_df.shape[1]} cols, {len(actual_feature_names)} in config" if not col_match else "",
        )
        if not ok:
            failures.append(f"sample_input_{c}.csv column mismatch with config_{c}.json")
    else:
        print(f"  [SKIP] sample_input_{c}.csv not found — column alignment not checked")

    # 6. Imputer loadable and has fitted state
    imputer_path = os.path.join(artifact_dir, f"imputer_{c}.pkl")
    if os.path.exists(imputer_path):
        try:
            imp = joblib.load(imputer_path)
            # PerModalityImputer stores nan_counts (populated for all modalities)
            # and medians (only for modalities that had NaN — empty dict is valid).
            has_fitted = hasattr(imp, "nan_counts") and bool(imp.nan_counts)
            ok = check(f"imputer_{c}.pkl loadable and fitted", has_fitted)
            if not ok:
                failures.append(f"imputer_{c}.pkl appears unfitted")
        except Exception as e:
            check(f"imputer_{c}.pkl loadable", False, str(e))
            failures.append(f"imputer_{c}.pkl load error: {e}")

    # 7. Scaler loadable and has fitted state
    scaler_path = os.path.join(artifact_dir, f"per_modality_scaler_{c}.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            # PerModalityScaler stores self.scalers (not sklearn's scalers_ naming).
            has_scalers = hasattr(scaler, "scalers") and bool(scaler.scalers)
            ok = check(f"per_modality_scaler_{c}.pkl loadable with fitted scalers", has_scalers)
            if not ok:
                failures.append(f"per_modality_scaler_{c}.pkl appears unfitted")
        except Exception as e:
            check(f"per_modality_scaler_{c}.pkl loadable", False, str(e))
            failures.append(f"per_modality_scaler_{c}.pkl load error: {e}")

    # 8. XGBoost model loadable
    xgb_path = os.path.join(artifact_dir, f"xgb_best_{c}.pkl")
    if os.path.exists(xgb_path):
        try:
            xgb_model = joblib.load(xgb_path)
            ok = check(f"xgb_best_{c}.pkl loadable", True)
        except Exception as e:
            check(f"xgb_best_{c}.pkl loadable", False, str(e))
            failures.append(f"xgb_best_{c}.pkl load error: {e}")

    # 9. Fusion .pt header check (torch.load with weights_only=False on state dict)
    fusion_path = os.path.join(artifact_dir, f"fusion_best_{c}.pt")
    if os.path.exists(fusion_path):
        try:
            import torch
            ckpt = torch.load(fusion_path, map_location="cpu", weights_only=False)
            has_state = "model_state_dict" in ckpt
            ok = check(f"fusion_best_{c}.pt has model_state_dict", has_state)
            if not ok:
                failures.append(f"fusion_best_{c}.pt missing model_state_dict key")
        except Exception as e:
            check(f"fusion_best_{c}.pt loadable", False, str(e))
            failures.append(f"fusion_best_{c}.pt load error: {e}")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Validate demo artifacts.")
    parser.add_argument(
        "--cancer",
        default="all",
        choices=["GS-BRCA", "GS-COAD", "all"],
        help="Cancer type to validate (default: all)",
    )
    args = parser.parse_args()

    artifact_dir = os.path.join("app", "model_artifacts")
    cancers = (
        ["brca", "coad"]
        if args.cancer == "all"
        else [args.cancer.split("-")[1].lower()]
    )

    all_failures = []
    for c in cancers:
        failures = validate_cancer(c, artifact_dir)
        all_failures.extend(failures)

    print(f"\n{'='*55}")
    if all_failures:
        print(f"VALIDATION FAILED — {len(all_failures)} issue(s):")
        for f in all_failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED — all {len(cancers)} cancer(s) OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
