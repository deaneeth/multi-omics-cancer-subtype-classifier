"""
Audit Script: Patient-level split verification for cv_folds.json.
Verifies no patient ID appears in both train and val of any fold.
Run from project root: python audit_scripts/check_cv_folds_leakage.py
"""
import json, sys

with open("data/cv_folds.json") as f:
    all_folds = json.load(f)

all_ok = True
for cancer_type, data in all_folds.items():
    for fold in data["folds"]:
        fold_idx = fold["fold"]
        train_set = set(fold["train"])
        val_set   = set(fold["val"])
        overlap   = train_set & val_set
        if overlap:
            print(f"FAIL: {cancer_type} fold {fold_idx} — {len(overlap)} overlapping IDs: {list(overlap)[:3]}")
            all_ok = False
        else:
            print(f"OK: {cancer_type} fold {fold_idx} — train={len(train_set)}, val={len(val_set)}, overlap=0")

if all_ok:
    print("\nPASS: No train/val overlap in any fold for any cancer type.")
    sys.exit(0)
else:
    print("\nFAIL: Leakage detected — see above.")
    sys.exit(1)
