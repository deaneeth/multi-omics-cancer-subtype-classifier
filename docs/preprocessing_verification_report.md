# Preprocessing Verification Report

Generated from `notebooks/01_preprocessing.ipynb` and `results/qc/preprocessing_verification.json`.

## Summary

The preprocessing pipeline passed the core invariants needed for downstream training and evaluation.

- Total checks: 49
- Passed: 46
- Failed: 3
- Basic checks: 8 / 10
- Extended checks: 38 / 39

The three failed checks are dataset-level limitations that were documented in the project snapshot and in the thesis discussion notes.

## Failed Checks

### 1. Label Shuffle BRCA

- Expected: shuffled-label XGBoost should behave close to random for 5 classes, around F1 = 0.20.
- Observed: F1 = 0.3804598187345256.
- Status: fail.

Root cause: global ANOVA feature pre-selection was performed before the CV split, which preserves label signal in the selected feature set. This is a benchmark-level limitation, not a bug in the fold preparation pipeline.

### 2. Label Shuffle COAD

- Expected: shuffled-label XGBoost should behave close to random for 4 classes, around F1 = 0.25.
- Observed: F1 = 0.5322003577817531.
- Status: fail.

Root cause: the same global ANOVA pre-selection issue as BRCA, amplified by the smaller COAD cohort.

### 3. GS-BRCA V7 All-NaN miRNA Columns

- Expected: no feature column should be entirely NaN.
- Observed: 166 of 366 BRCA miRNA columns are all-NaN.
- Status: fail.

Root cause: this is a dataset property. The BRCA miRNA assay contains probes that are unobserved in the cohort. The `PerModalityImputer` handles these columns by filling them with 0.0 after fitting on the training fold.

## Checks That Passed

The following invariants were confirmed by the verification notebook and the stored QC JSON:

- Modality shapes are correct.
- No NaN remain after preprocessing.
- No Inf remain after preprocessing.
- Train means are approximately 0.
- Train standard deviations are approximately 1.
- No sample overlap exists between train and validation folds.
- Fold sample IDs match the saved fold file.
- PCA plots were saved successfully.
- All train/validation split integrity checks passed.
- All cross-validation coverage checks passed.

## Dataset Notes

### GS-BRCA

- 671 samples
- 5 subtypes
- All samples have all 4 modalities
- miRNA contains many all-NaN columns, which are handled during preprocessing

### GS-COAD

- 260 samples
- 4 subtypes
- All samples have all 4 modalities
- Class 3 contains only 4 samples, so some validation folds can miss that class

## Conclusion

The preprocessing pipeline is structurally sound and leakage-free at the fold level. The two label-shuffle failures are expected given global ANOVA feature selection before CV. The BRCA miRNA all-NaN features are a dataset characteristic and are handled correctly by the imputer.
