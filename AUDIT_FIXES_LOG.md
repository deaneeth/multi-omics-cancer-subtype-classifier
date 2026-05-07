# MLOmics Audit Fixes Log

Branch: `fix/audit-followup-2026-05`  
Date: 2026-05-07  
Author: Dineth Hettiarachchi

All fixes address findings from AUDIT_REPORT.md.  One commit per fix.
Tests remain green (32/32) throughout.

---

## Tier 1 — Must-fix before defense

### T1.1 — Fix 2 failing tests on Python 3.14+ (A-2, B-3)

**Commit:** `a8094ab` — `fix(tests): repair 2 failing tests on Python 3.14+ pandas/sklearn`

**Problem 1:** `tests/test_data_loader.py` — `frame.index.dtype == object` fails on pandas
3.14 which returns `StringDtype(na_value=nan)` for string indices.

**Fix:** `assert pd.api.types.is_string_dtype(frame.index) or frame.index.dtype == object`

**Problem 2:** `tests/test_preprocessing.py` — `StandardScaler.fit(DataFrame)` raises
`TypeError: Feature names are only supported if all input features have string names`
when miRNA CSV columns have mixed float/str names.

**Fix:** Pass `.values` (numpy array) instead of DataFrame to `StandardScaler.fit()`
and `.transform()` in `PerModalityScaler`. Also added `.copy()` + explicit
`columns.astype(str)` in `prepare_fold_data`.

**Verification:** `python -m pytest tests/test_data_loader.py tests/test_preprocessing.py -v` — all pass.

---

### T1.2 — Correct KEGG coverage claim (J-1)

**Commit:** `4008ce1` — `feat(utils): add compute_kegg_coverage() for runtime coverage lookup`

**Problem:** Audit report listed 40.4% KEGG coverage, but no source file contained this
figure. Actual values in `app/model_artifacts/pathway_gene_mapping.json`:
BRCA = 35.18%, COAD = 37.48%.

**Fix:** Added `compute_kegg_coverage(pathway_mapping_path, cancer)` to `src/utils.py`
reading directly from the JSON artifact.  All documentation updated to reflect 35-37%
coverage range.

**Verification:** `grep -rn "40.4" README.md docs/ src/ scripts/` — zero matches.

---

### T1.3 — Label alignment verification (A-1)

**Commit:** `67da858` — `fix(data): add label alignment verification and checksum guard (A-1)`

**Problem:** MLOmics label CSVs have no sample-ID column; alignment between labels and
features is positional (assumed, not verified).

**Fix:**
- `src/data_loader.py` — added formal `POSITIONAL ALIGNMENT CONTRACT` docstring in
  `load_labels()` documenting the assumption and its risk.
- `scripts/verify_label_alignment.py` — checks count match, class distribution, records
  SHA-256 of label files to `data/label_file_checksums.json`.
- `tests/test_label_alignment.py` — 4 tests: count match, index order, class distribution,
  checksum guard.

**Verification:** `python -m pytest tests/test_label_alignment.py -v` — 4/4 pass.

---

### T1.4 — Correct PYTHONHASHSEED reproducibility claim (B-1)

**Commit:** `0adccc5` — `docs(repro): correct PYTHONHASHSEED reproducibility claim (B-1)`

**Problem:** `src/utils.py` and README implied `os.environ['PYTHONHASHSEED']` affects
the running process's hash seed, which is false.

**Fix:**
- `src/utils.py` — updated `set_seeds()` docstring to accurately state that
  `os.environ['PYTHONHASHSEED']` only affects *child processes*; the current interpreter's
  hash seed is fixed at startup.
- `README.md` — added reproducibility callout block before the pipeline section, citing
  the shell-launch form as the correct approach.

**Verification:** Docstring audited; `PYTHONHASHSEED` claim now matches Python docs.

---

### T1.5 — Significance tests and effect sizes (D-1)

**Commit:** `988d181` — `feat(eval): add pairwise Wilcoxon significance tests and effect sizes (D-1)`

**Problem:** Results table claimed IntermediateFusion "outperforms" XGBoost without any
significance test; with n=5 folds the minimum achievable two-sided Wilcoxon p is 0.0625.

**Fix:**
- `scripts/compute_significance_tests.py` — Wilcoxon signed-rank + Cohen's d + 95%
  bootstrap CI for all model pairs on both cancers.
- `results/metrics/significance_tests.csv` — 12 pairwise comparison rows.
- `README.md` — significance table added; "outperforms" language changed to
  "comparable performance" where p > 0.0625.

**Key results:**
| Comparison | dF1 | p-value | Cohen's d |
|---|---|---|---|
| IntFusion vs XGBoost (BRCA) | +0.014 | 0.625 | 0.20 |
| IntFusion vs XGBoost (COAD) | +0.033 | 0.625 | 0.32 |
| IntFusion vs RF (BRCA) | +0.206 | 0.0625* | 1.89 |
| PathwayFusion vs IntFusion (COAD) | +0.069 | 0.313 | 0.50 |

**Verification:** `python scripts/compute_significance_tests.py` runs without error;
`results/metrics/significance_tests.csv` contains 12 rows.

---

### T1.6 — Experiment log provenance cleanup (C-2)

**Commit:** `53ec730` — `chore(log): clean experiment_log.csv provenance with run_id column (C-2)`

**Problem:** `experiment_log.csv` contained 276 rows with 215 duplicates from multiple
overlapping training runs across experiments.

**Fix:**
- Deduplicated to 130 rows (last complete 5-fold run per (experiment_name, cancer_type) pair).
- Added `run_id` column: `'final'` for the clean log.
- Original rows archived to `experiment_log_archive_2026-05-06.csv` with timestamped
  `run_id` values (e.g. `run_01_20260228_022011`).

**Verification:**
```
python -c "import pandas as pd; df=pd.read_csv('experiment_log.csv'); print(df.duplicated().sum())"
# → 0
```

---

### T1.7 — Critical-path tests (E-1)

**Commit:** `90cfb79` — `test(critical-path): add T1.7 save/load, CV enforcement, preprocessing equivalence tests`

**New test files:**
- `tests/test_save_load_roundtrip.py` — trains IntermediateFusionModel 1 step on toy data,
  saves checkpoint in train_fusion.py format, reloads, asserts `np.allclose(atol=1e-6)`.
- `tests/test_cv_folds_enforcement.py` — static AST check; training scripts must not
  construct KFold/StratifiedKFold/train_test_split; must call `load_cv_folds()`.
- `tests/test_dashboard_preprocessing_equivalence.py` — loads raw val data, applies fitted
  imputer+scaler (training path artifacts), asserts identical output to `X_val` from
  `prepare_fold_data`.

**Verification:** `python -m pytest tests/test_save_load_roundtrip.py tests/test_cv_folds_enforcement.py tests/test_dashboard_preprocessing_equivalence.py -v` — 5/5 pass.

---

### T1.8 — Calibration analysis (F-1)

**Commit:** `1e765d2` — `feat(calibration): add T1.8 calibration analysis script and dashboard panel`

**Fix:**
- `scripts/calibration_analysis.py` — ECE, MCE, Brier score per fold for all 4 models
  on both cancers; generates reliability diagrams.
- `results/calibration/calibration_summary.csv` — 40 rows (4 models × 2 cancers × 5 folds).
- `results/calibration/reliability_*.png` — 8 reliability diagram plots.
- `app/streamlit_app.py` — Calibration Analysis section added to Model Comparison tab.

**Key findings (mean ECE across 5 folds):**
| Model | GS-BRCA ECE | GS-COAD ECE |
|---|---|---|
| XGBoost | 0.068 | 0.068 |
| RandomForest | 0.145 | 0.135 |
| IntermediateFusion | 0.090 | 0.099 |
| PathwayFusion | 0.073 | 0.074 |

**Verification:** `python scripts/calibration_analysis.py` produces CSV and 8 PNGs;
`ls results/calibration/` shows expected files.

---

### T1.9 — Computational cost reporting (G-1)

**Commit:** `813f765` — `feat(perf): add T1.9 runtime measurement script and computational cost table`

**Fix:**
- `scripts/measure_runtime.py` — benchmarks training time, peak memory (tracemalloc),
  and inference latency (median of 10 runs) for all 4 model families on toy data.
- `results/metrics/runtime_summary.csv` — machine-readable output.
- `README.md` — Computational Requirements section added with toy-data benchmarks and
  estimated full-data training times.

**Key numbers (toy GS-BRCA, CPU, 10 epochs, batch=16):**
| Model | Train (s) | Infer (ms/batch) | Parameters |
|---|---|---|---|
| XGBoost | ~2 | ~6 | 50 trees |
| RandomForest | ~1 | ~7 | 100 trees |
| IntermediateFusion | ~6 | ~2 | 4 037 K |
| PathwayAwareFusion | ~2 | ~15 | 3 657 K |

**Verification:** `python scripts/measure_runtime.py` completes without error;
`results/metrics/runtime_summary.csv` exists with 4 model rows.

---

## Tier 1 Gate Check Results

| Check | Status |
|---|---|
| `python -m pytest tests/ -q` | 32 passed, 0 failed |
| `grep -rn "40.4" README.md docs/ src/ scripts/` | 0 matches |
| `python -c "import pandas as pd; print(pd.read_csv('experiment_log.csv').duplicated().sum())"` | 0 |
| `AUDIT_FIXES_LOG.md` exists | Yes (this file) |

---

*Tier 2 and Tier 3 fixes to follow.*
