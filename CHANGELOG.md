# Changelog

All notable changes to the MLOmics project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).
Tagged releases mark `dev → main` milestone merges.

## [Unreleased] — feature/wire-pathway-fusion-attributions

### Fixed
- **Demo (Data Converter):** `convert_patient_data()` was raising `TypeError: got an unexpected
  keyword argument 'data_format'` on Streamlit hot-reload. Root cause: Streamlit's hot-reload
  mechanism re-executes the main script but keeps `sys.modules['src.patient_converter']` cached
  from the previous run, so the top-level `from src.patient_converter import convert_patient_data`
  bound the old function object (without `data_format`). Fixed by moving the import inside the
  Convert button handler and using `importlib.reload(src.patient_converter)` before each call,
  ensuring the live module is always used. Constants `MODALITY_LABELS` and `MODALITY_NOTES`
  remain top-level (they are used at render time and never change at runtime).
- **Demo (PathwayAwareFusion IG):** `load_fusion_attributions()` now accepts a `model_type`
  parameter (`"pathway"` | `"intermediate"`). When PathwayAwareFusion is selected, the app
  correctly loads `pathway_fusion_attribution_results_{suffix}.json` instead of reusing the
  IntermediateFusion file. The misleading caption ("Attributions shown are from the
  Intermediate Fusion model …") has been removed.
- **Demo (Section headers):** Feature Importance heading dynamically shows "Integrated Gradients"
  for fusion models, "SHAP" for XGBoost. Colour caption updated per model type.

### Added
- **Demo (Prediction tab):** A "Pathway-Level Attention Weights" panel now renders below the
  IG bar chart when PathwayAwareFusion is selected. It shows a Plotly horizontal bar chart of
  the top-10 KEGG pathways by mean attention weight for the current cancer type (data from
  `results/enrichment/pathway_attention_scores.csv`). Note: weights are nearly uniform
  (0.003–0.004 range) — see Known Limitation #6 in `PROJECT_SNAPSHOT.md`.
- **Demo (Prediction tab):** Live Integrated Gradients computation via Captum as fallback when
  precomputed attributions are unavailable for uploaded samples. Matches XGBoost's always-available
  SHAP behaviour.
- **Demo (Data Converter tab):** Third tab "Lab Data Converter" for hospital/lab per-modality
  CSV upload. Walks users through 4 steps: patient info, modality file upload (any combination),
  conversion with auto miRNA name normalisation, and coverage report + download. Powered by
  `src/patient_converter.py`. Supports 4 input data formats: z-scored, log2-transformed,
  raw counts, and beta values (methylation). Non-z-scored data is automatically normalized
  using single-sample robust z-scoring (median + IQR).
- **Module:** `src/patient_converter.py` — in-memory conversion of per-modality lab CSVs to
  single prediction-ready row. Auto-detects CSV orientation, normalises miRNA names (dash-to-dot),
  reports per-modality feature coverage with status tiers. Supports normalization from raw counts
  (log2+zscore), log2-transformed (zscore), beta values (M-value+zscore), or pre-normalized data.
- **Script:** `scripts/create_lab_patient_files.py` (292 lines) — generates 4 per-modality lab-format CSVs
  for a synthetic Basal-like BRCA patient (Amara Nwosu). Used to test the Data Converter tab.
- **Script:** `scripts/create_demo_patients.py` (215 lines) — generates Sarah Mitchell (Luminal A BRCA)
  and Robert Okonkwo (CMS2 COAD) single-patient prediction-ready CSVs with clinical backstories.
- **Script:** `scripts/prepare_real_patient_upload.py` (294 lines) — standalone CLI converter (alternative to
  the Streamlit Data Converter tab) for batch processing.
- **Demo datasets:** `app/test_datasets/demo/demo_patient_sarah_mitchell_BRCA.csv` (Luminal A)
  and `demo_patient_robert_okonkwo_COAD.csv` (CMS2) — single-patient prediction-ready CSVs with
  clinical backstories.
- **Demo datasets:** `app/test_datasets/demo/convertion/` — 4 lab-format modality files for
  Amara Nwosu (mRNA, miRNA, methylation, CNV) to test Data Converter workflow.
- **Data cards:** `DATA_CARD_demo_patients.md` and `convertion/DATA_CARD.md` documenting
  generation methodology, expected results, and limitations.
- **Docs:** Comprehensive multi-session fact-check and update of all documentation against live filesystem and result CSVs:
  - `PROJECT_SNAPSHOT.md` — rev 14: accuracy std corrected to ddof=1 (8 values), Cohen's d exact values (0.197/1.888/0.504), pathway attention name fixed ("Taurine and hypotaurine metabolism"), script line counts corrected (3 scripts), src/ total ~2,235 lines, model parameter counts added (IntFusion 4,036,613; PathwayFusion 3,656,252), plain-English description updated to mention all 3 tabs, `importlib.reload` noted in Data Converter section, demo/ attribution corrected to 2 scripts, "35+" → "35 files"
  - `PROJECT_FILE_TREE.md` — major overhaul from 2026-05-03 state: 6→7 src modules with correct line counts, 15→22 scripts (7 missing added with correct line counts), 6→10 test files (4 missing added), 23→32 tests, docs 4→6 files, streamlit_app.py ~2,290→~3,875 lines with Data Converter tab described, model_artifacts 29→27 files (ghost entries removed), demo/ directory (8 files) added, data/label_file_checksums.json added, `abstract.md` ghost entry removed from root section
  - `PROJECT_STORY.md` — complete rewrite: all verified F1/AUC/accuracy values, all statistical test values (p/d), correct pathway attention names, correct SHAP top features, Data Converter 4-format description, 22 scripts/7 modules/32 tests
  - `PRESENTATION_NARRATION.md` — complete rewrite: all verified metrics, correct Cohen's d values, Data Converter tab described, 3-tab demo, verified KEGG p-values and gene lists
  - `README.md` — complete rewrite: verified parameter counts, measured latency values, correct dataset descriptions, all 3 tabs, complete script list
  - `CHANGELOG.md` — script entries corrected with exact line counts, `create_demo_patients.py` entry added
  - `AGENTS.md` — `patient_converter.py` added to module table, Data Converter `importlib.reload` note added, all 3 new scripts added to commands

---

## [v1.1-audit] - 2026-05-07

### Added
- **Audit**: Comprehensive data integrity checks, including `verify_label_alignment.py` and checksum guards.
- **Evaluation**: Calibration analysis (`calibration_analysis.py`) and significance tests (`compute_significance_tests.py`).
- **Performance**: End-to-end inference computational cost measurement (`measure_runtime.py`).
- **Tests**: Extensive test suite covering preprocessing equivalence, CV folds enforcement, and label alignment.
- **Docs**: New structured documentation including `PROJECT_FILE_TREE.md`, `DATA_CARD.md`, and an updated `PROJECT_STORY.md`.

### Changed

- **Code quality**: Deduplicated `compute_class_weights` and `dataframes_to_numpy` helpers into `src/utils.py`; both training scripts now import from there.
- **Data integrity**: Replaced stale `CONTEXT.md` docstring references with `config.yaml` in `src/data_loader.py` and `src/utils.py`.
- **XGBoost objective**: Corrected `multi:softmax` → `multi:softprob` in `scripts/train_baselines.py` (source fix; existing trained models are unchanged).
- **SHAP performance**: `compute_shap_explanation` now computes SHAP only for the selected sample row instead of all uploaded rows.
- **Preprocessing docs**: Added ANOVA pre-selection bias warning to `src/preprocessing.py` module docstring.
- **Encoding**: Re-encoded `requirements.txt` from UTF-16 LE to UTF-8 for cross-platform portability.
- **Tests**: Added `tests/test_cv_folds.py` (cross-fold leakage guard) and a train-only scaler test in `tests/test_preprocessing.py`.
- **README**: Added full conda setup, pipeline commands, and expected results table.

### Notable ablation finding

- GS-COAD: Removing the miRNA modality **improves** F1 by +0.133 (from 0.669 to 0.802), suggesting miRNA introduces noise for COAD subtype discrimination. Documented in `results/metrics/ablation_modality_removal.csv`.

---

## [v0.5-demo] - 2026-04-12

### Added

- Dual-cancer Streamlit demo flow for GS-BRCA and GS-COAD.
- Cancer-specific demo artifacts under `app/model_artifacts/`.
- Pathway-aware fusion demo support alongside XGBoost and intermediate fusion.
- Per-fold AUC outputs: `results/metrics/auc_scores.csv` and `auc_summary.csv`.
- ROC curve display in the demo and pathway-fusion confusion matrices.
- Preprocessing verification report in `docs/preprocessing_verification_report.md`.

### Changed

- Demo comparison table now includes AUC alongside F1, precision, recall, NMI, and ARI.
- Legacy un-suffixed demo artifacts were removed in favor of per-cancer files.
- Streamlit app layout was updated for the restored dual-cancer demo.

### Notes

- PathwayAwareFusion IG artifacts (`pathway_fusion_attribution_results_{suffix}.json`) are
  precomputed and wired in the app as of `feature/wire-pathway-fusion-attributions`.
- COAD fold 0 AUC may be NaN when the CMS4 class is absent from the validation split.

---

## [v0.4-analysis] - 2026-03-06

### Added

- `src/explainability.py` with SHAP, Deep Attribution (Integrated Gradients via Captum), and KEGG/GO pathway enrichment functions (P6-T1)
- `scripts/run_explainability.py` executing the full explainability pipeline and extracting top 50 features per model (P6-T2)
- `results/shap/` and `results/enrichment/` artifacts including baseline SHAP values, fusion Integrated Gradients, and pathway validation reports (P6-T2)
- `notebooks/04_explainability.ipynb` interactive notebook for visualizing feature attributions and biological enrichment (P6-T2)
- `src/models.py` updated to include `EarlyFusionMLP` class for architectural ablation studies (P7-T1)
- `scripts/run_ablations.py` to automate modality removal, early vs. intermediate fusion comparison, and missing modality simulation (P7-T1)
- `results/metrics/ablation_*.csv` and `results/plots/ablation_*` containing modality removal, fusion comparison, and missing-modality degradation curve results (P7-T1)
- `notebooks/04b_ablations.ipynb` for analyzing and visualizing ablation study robustness (P7-T1)

### Changed

- Evaluated up to 50% missing modality drop-out robustness on BRCA and COAD, demonstrating intermediate fusion resilience (P7-T1)

---

## [v0.3-fusion] - 2026-03-04

### Added

- `src/models.py` with `ModalityEncoder`, `FusionClassifier`, `IntermediateFusionModel`, and `MultiOmicsDataset` (P5-T1)
- `scripts/train_fusion.py` — 5-fold CV training loop with early stopping, LR scheduling, and dynamic class weighting (P5-T2)
- Intermediate fusion model trained on GS-BRCA (F1=0.81±0.05) and GS-COAD (F1=0.67±0.09) using same `cv_folds.json` as baselines (P5-T2)
- `models/intermediate_fusion/` — 10 saved `.pt` model checkpoints (5 per cancer type) (P5-T2)
- `results/metrics/fusion_brca_metrics.csv` and `fusion_coad_metrics.csv` with per-fold metrics (P5-T2)
- `results/metrics/model_comparison.csv` — XGBoost vs RandomForest vs IntermediateFusion comparison table (P5-T3)
- `results/plots/fusion_training_curves_*.png` and `.pdf` — training loss and validation F1 per fold (P5-T2)
- `results/plots/confusion_matrix_fusion_*.png` and `.pdf` for both cancer types (P5-T2)
- `results/plots/model_comparison_*.png` and `.pdf` — bar charts comparing all three models (P5-T3)
- `notebooks/03_latent_fusion.ipynb` — fusion model evaluation and cross-model comparison notebook (P5-T3)
- Phase 5 gate check passed: model trains, same CV folds, metrics in range, artifacts saved, comparison table generated

---

## [v0.2-baselines] - 2026-03-03

### Added

- `src/evaluation.py` with `compute_metrics()`, fold summary, and metrics export (P4-T1)
- XGBoost 5-fold CV pipeline for GS-BRCA and GS-COAD (P4-T2)
- RandomForest 5-fold CV pipeline for GS-BRCA and GS-COAD (P4-T3)
- SHAP analysis script, summary plots, and top 50 features per model (P4-T4)
- `results/metrics/baseline_comparison.csv` and individual model metric CSVs
- `results/plots/confusion_matrix_*` evaluating predictive performance
- `results/shap/xgb/` and `results/shap/rf/` SHAP artifact directories
- `notebooks/02_baselines.ipynb` evaluating baseline performance and biological relevance
- Phase 4 gate check passed: all baselines verified, no leakage detected

---

## [v0.1-preprocessing] - 2026-02-27

### Added

- `src/preprocessing.py` containing `PerModalityScaler` and `PerModalityImputer` (P3-T1)
- `src/preprocessing.py` containing `create_cv_folds()` and `load_cv_folds()` with StratifiedKFold implementation (P3-T2)
- `src/preprocessing.py` containing `prepare_fold_data()` to combine imputer, scaler, and concatenation (P3-T1)
- `notebooks/01_preprocessing.ipynb` — comprehensive preprocessing verification notebook running 31-point checks (P3-T3)
- `results/qc/preprocessing_verification.json` — machine-readable verification results (46/49 checks pass) (P3-T3)
- `results/plots/pca_preprocessed_BRCA.png` and `pca_preprocessed_COAD.png` (P3-T3)
- `data/cv_folds.json` and `data/toy/cv_folds.json` — persisted CV splits mapping patient IDs to 5 folds (P3-T2)

### Changed

- Label shuffle test implemented in notebook, confirming expected elevated F1 due to global ANOVA pre-selection (documented limitation) (P3-T3)

---

## [v0.0.1-data] - 2026-02-27

### Added

- `data/DATA_README.md` with download instructions, checksums, and verified dataset statistics (P2-T2)
- `notebooks/00_data_inspect.ipynb` — data inspection notebook with shape/NaN/class distribution checks (P2-T3)
- `results/qc/data_inspection_report.json` — machine-readable inspection output (P2-T3)
- `src/data_loader.py` with `load_modality()`, `load_labels()`, `get_common_samples()`, `create_sample_map()` (P2-T4)
- `data/sample_map.csv` — 931 samples (671 BRCA + 260 COAD), all with 4/4 modalities (P2-T4)
- `data/dropped_samples.csv` — 0 samples dropped (P2-T4)
- `data/checksums_brca.txt` and `data/checksums_coad.txt` for raw data integrity verification (P2-T2)
- `scripts/create_toy_dataset.py` — stratified 50-sample GS-BRCA subset for rapid prototyping (P2.5-T1)
- `data/toy/` — 5 toy files (4 modalities + labels) in original CSV format (P2.5-T1)
- `use_toy` flag on `load_modality()`, `load_labels()`, `get_common_samples()` to switch between full and toy data (P2.5-T1)

### Changed

- `config.yaml` miRNA `feature_count` set to `null` — varies by cancer (BRCA=366, COAD=200), read dynamically (P2-T3 finding)

---

## [v0.0-scaffold] - 2026-02-26

### Added

- Initial repo structure and folder layout per CONTEXT.md Section 5 (P1-T2)
- `.gitignore` for Python, data files, model artifacts, IDE configs (P1-T2)
- `config.yaml` — central configuration with 7 sections: project, paths, modalities, preprocessing, baselines, fusion, evaluation (P1-T4)
- `src/__init__.py` — package initializer (P1-T4)
- `src/utils.py` with `set_seeds(42)`, `load_config()`, `log_experiment()`, `get_device()` (P1-T4)
- `experiment_log.csv` with standardized headers (P1-T4)
- `requirements.txt` with pinned package versions (P1-T3)
- `CONTEXT.md` agent brain file and `.agents/rules/rule.md` (P1-T1)

---

Future releases will follow: v0.1-preprocessing → v0.2-baselines → v0.3-fusion → v0.4-analysis → v0.5-demo → v1.0-final.