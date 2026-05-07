# MLOmics — Complete Project File Tree

*Every file in this repository, listed with its full path and purpose. Generated 2026-05-03. Total: 441 files.*

---

## ROOT — Project Configuration & Documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview: setup, pipeline flow, expected results, tech stack, repo structure |
| `LICENSE` | MIT License |
| `config.yaml` | Single source of truth — 7 sections: project metadata, file paths, modality definitions, preprocessing parameters, baseline hyperparameters (XGBoost, RF), fusion hyperparameters, evaluation settings |
| `requirements.txt` | Pinned Python package versions (150 packages, UTF-8 encoded). Core: torch 2.7.1+cu118, scikit-learn 1.6.1, xgboost 2.1.4, shap 0.49.1, captum 0.8.0, gseapy 1.1.11, streamlit 1.50.0 |
| `experiment_log.csv` | 130 canonical rows (deduplicated from 276; archive at `experiment_log_archive_2026-05-06.csv`) tracking every training run: timestamp, experiment name, cancer type, model type, feature count, fold index, F1/precision/recall/NMI/ARI metrics, notes, artifact path |
| `.gitignore` | Git exclusion rules: ignores `data/raw/`, `data/preprocessed/`, `*.pkl`, `*.pt`, `__pycache__/`, `.ipynb_checkpoints/`, `content/` directory, legacy demo artifacts, `.env`, IDE configs |
| `AGENTS.md` | AI agent guidance: commands, import system, architecture, key invariants, data gotchas, hardware constraints, git workflow |
| `CHANGELOG.md` | Keep-a-changelog format: v0.0-scaffold through v0.5-demo and unreleased changes. Documents every feature addition, fix, and notable finding |
| `explain.md` | FAQ document: answers clinical context questions — why subtype classification matters, treatment implications, best demo files to use, project positioning |
| `abstract.md` | Thesis abstract: project summary, key results (BRCA F1=0.808, COAD F1=0.738, miRNA artifact +0.133, Cell cycle/p53 enrichment), keywords |

### Root hidden files

| File | Purpose |
|---|---|
| `.env` | Gitignored — environment variables (e.g. `GROQ_API_KEY=gsk_...` for AI summaries) |
| `.streamlit/config.toml` | Gitignored — Streamlit theme persistence (written by app at runtime) |

---

## `src/` — Core Library (6 modules, ~1,850 lines)

| File | Lines | Purpose |
|---|---|---|
| `src/__init__.py` | 2 | Package marker — makes `src/` an importable Python package |
| `src/utils.py` | 205 | `set_seeds(42)` — sets PYTHONHASHSEED, random, numpy, torch (CPU+CUDA), cuDNN flags for full reproducibility. `load_config()` — loads `config.yaml`. `log_experiment()` — appends to `experiment_log.csv`. `get_device()` — returns `cuda` or `cpu`. `compute_class_weights()` — inverse-frequency class weights. `dataframes_to_numpy()` — converts DataFrame dicts to float32 arrays |
| `src/data_loader.py` | 269 | `MODALITY_KEYS` — canonical order: `["mrna","mirna","methy","cnv"]`. `load_modality()` — loads CSV, transposes (features×samples → samples×features). `load_labels()` — globs `*_label_num.csv`, positionally aligns with mRNA columns. `get_common_samples()` — finds intersection of sample IDs across all 4 modalities. `create_sample_map()` — builds `sample_map.csv` and `dropped_samples.csv` |
| `src/preprocessing.py` | 449 | `PerModalityImputer` — median imputation per modality, fit on train only. `PerModalityScaler` — z-score StandardScaler per modality, fit on train only. `prepare_fold_data()` — full fold pipeline: load→split→assert no overlap→impute→assert zero NaN→scale→return. `concatenate_modalities()` — joins modality DataFrames with `modality_` prefixed feature names. `create_cv_folds()` — patient-level stratified 5-fold CV, verifies 1:1 patient-to-sample mapping, saves to `cv_folds.json`. `load_cv_folds()` — loads canonical fold assignments |
| `src/models.py` | 381 | `ModalityEncoder` — 2-layer dense: input_dim→256→64 (BatchNorm+ReLU+Dropout). `FusionClassifier` — MLP head: total_latent→128→num_classes. `IntermediateFusionModel` — 4 encoders → concat latents → classifier. `PathwayAttentionEncoder` — dual-path mRNA: Path A groups by KEGG pathway (mean-pool→attention→project), Path B handles unmapped genes. `PathwayAwareFusionModel` — replaces mRNA encoder with PathwayAttentionEncoder. `EarlyFusionMLP` — ablation baseline: concat→256→128→n_classes. `MultiOmicsDataset` — PyTorch Dataset wrapping `{modality: tensor}` dicts |
| `src/evaluation.py` | 170 | `compute_metrics()` — returns precision/recall/F1/NMI/ARI/accuracy (macro-averaged, zero_division=0). `compute_fold_summary()` — mean±std across folds. `print_metrics_table()` — ASCII box table. `save_metrics()` — per-fold + summary row to CSV |
| `src/explainability.py` | 376 | `compute_tree_shap()` — SHAP TreeExplainer for XGBoost/RF, subsamples to max_samples. `compute_deep_attribution()` — Captum Integrated Gradients via `_FusionModelWrapper` (converts tuple→dict). `extract_top_features()` — top-N by mean absolute attribution with modality labels. `run_kegg_enrichment()` / `run_go_enrichment()` — gseapy enrichr with fallback to file export. `check_known_pathways()` — validates 9 canonical cancer pathways against enrichment results |

---

## `scripts/` — Automation & Training (15 scripts, ~4,300 lines)

| Script | Lines | Purpose |
|---|---|---|
| `scripts/train_baselines.py` | 324 | CLI: `--model xgb/rf/all`, `--toy`. Trains XGBoost and/or RandomForest with 5-fold CV on concatenated early-fusion features. Outputs: `models/baseline_{xgb,rf}/*.pkl`, `results/metrics/{xgb,rf}_*.csv`, confusion matrices, experiment log entries |
| `scripts/train_fusion.py` | 373 | CLI: `--toy`, `--epochs`. Trains IntermediateFusionModel with 5-fold CV. Per-fold: `set_seeds(42)`, prepare fold data, train with early stopping (patience=10), ReduceLROnPlateau, class-weighted CrossEntropyLoss, gradient clipping. Saves `.pt` checkpoints, training curves, confusion matrices |
| `scripts/train_pathway_fusion.py` | 382 | Same structure as `train_fusion.py` but trains PathwayAwareFusionModel. Loads `data/processed/pathway_gene_mapping.json`. Saves attention scores per sample to `pathway_attention_scores.csv` |
| `scripts/run_explainability.py` | 324 | Full explainability pipeline: loads best-fold fusion model (from `experiment_log.csv`), runs Integrated Gradients, extracts top-50 features, runs KEGG+GO enrichment, validates against 9 canonical cancer pathways, generates plots with modality-coloured bars |
| `scripts/shap_analysis.py` | 254 | CLI: `--model xgb/rf`, `--cancer`, `--toy`. Computes SHAP values for best tree-model fold per cancer. Outputs: summary plots, bar plots, top-50 CSV, SHAP values NPZ |
| `scripts/run_ablations.py` | 579 | CLI: `--only a1/a2/a3`. Three ablation experiments: A1) modality removal — retrains IntermediateFusion with one modality omitted; A2) fusion comparison — XGBoost vs EarlyFusionMLP vs IntermediateFusion; A3) missing modality — zero-pads features at 10-50% rates. Outputs CSVs and plots |
| `scripts/run_evaluation.py` | 99 | Convenience entrypoint: loads per-fold NPZ predictions, prints consolidated metrics table, delegates to `compute_auc.py` for macro OVR AUC refresh |
| `scripts/compute_auc.py` | 263 | Re-inference from all 40 saved models. Loads each model, runs on its validation fold, computes macro-averaged One-vs-Rest AUC via `roc_auc_score`. Outputs: `auc_scores.csv` (40 rows) and `auc_summary.csv` (8 rows) |
| `scripts/prepare_demo_artifacts.py` | 324 | CLI: `--cancer GS-BRCA/GS-COAD/all`. Exports best-fold models, per-modality scalers, imputers, config JSONs, sample input CSVs, and preprocessing artifacts to `app/model_artifacts/`. Generates per-cancer files with `_brca`/`_coad` suffixes |
| `scripts/precompute_fusion_attribution.py` | 260 | Runs Integrated Gradients on demo sample CSV for both IntermediateFusion and PathwayAwareFusion models. Outputs `fusion_attributions_{c}.npz` and `fusion_attribution_results_{c}.json`. Includes migration helper for legacy file renaming |
| `scripts/precompute_latent_space.py` | 155 | Runs all training samples through `IntermediateFusionModel.get_latent()`, applies t-SNE dimensionality reduction, writes `latent_space_data_{c}.json` for demo visualization. Includes legacy combined-to-per-cancer file migration |
| `scripts/create_toy_dataset.py` | 79 | Generates stratified 50-sample BRCA subset from full data, saves to `data/toy/` in original features×samples CSV format (transposed back). Used by `--toy` flag throughout the pipeline |
| `scripts/create_test_datasets.py` | 292 | CLI: `--cancer`. Extracts real TCGA val-set patients from best CV fold, preprocesses with saved train-fitted artifacts, writes upload-ready CSVs with leading `sample_id` column to `app/test_datasets/test/`. Creates batch files (20 BRCA, 12 COAD), single-patient files per subtype, and metadata CSVs |
| `scripts/create_synthetic_patients.py` | 435 | CLI: `--cancer`, `--n-per-subtype`. Generates synthetic patient profiles by sampling each of ~15,000 features from `N(class_mean, class_std)` per subtype, clipped to [-4.5, 4.5]. Outputs to `app/test_datasets/synthetic/` |
| `scripts/validate_artifacts.py` | 199 | CLI: `--cancer`. Checks all demo artifacts exist, have expected structure, and are internally consistent (feature counts match config, NPZ arrays loadable, scaler dimensions correct). Exit code 0 = pass |

---

## `tests/` — Automated Test Suite (6 test files + conftest, 23 tests)

| File | Tests | Purpose |
|---|---|---|
| `tests/__init__.py` | — | Package marker |
| `tests/.gitkeep` | — | Keeps empty test directory in Git |
| `tests/conftest.py` | — | Session-scoped fixtures: `_seed_everything()` (set_seeds(42) autouse), `config` (loads config.yaml), `toy_brca_fold` (first toy BRCA fold) |
| `tests/test_models.py` | 5 | `test_modality_encoder_output_shape` — verifies (4,32)→(4,8). `test_intermediate_fusion_forward_and_latent_shapes` — logits (3,5), latent (3,24). `test_pathway_attention_encoder_produces_normalized_attention` — attention sums to 1. `test_pathway_aware_fusion_forward_shape` — logits (2,3), attention not None. `test_multiomics_dataset_returns_modality_dict_and_label` — correct dict keys and label type |
| `tests/test_preprocessing.py` | 5 | `test_per_modality_imputer_fills_nan_and_all_nan_features` — zero NaN after imputation, all-NaN features→0.0. `test_per_modality_scaler_centers_training_data` — train means≈0. `test_prepare_fold_data_toy_has_no_nan` — end-to-end toy fold integrity. `test_scaler_fitted_on_train_only_val_not_centered` — val means≠0 proves scaler fit on train only. `test_concatenate_modalities_returns_prefixed_feature_names` — names like `mrna_g1` |
| `tests/test_evaluation.py` | 2 | `test_compute_metrics_returns_expected_keys_and_ranges` — correct keys, values in [0,1]. `test_compute_fold_summary_generates_mean_and_std_for_each_metric` — mean/std keys present |
| `tests/test_data_loader.py` | 3 | `test_load_modality_toy_returns_samples_by_features` — DataFrame shape, string index. `test_load_labels_toy_aligns_to_mrna_index` — index equality. `test_common_samples_are_present_in_all_modalities` — intersection subset check |
| `tests/test_cv_folds.py` | 4 | `test_no_overlap_between_val_folds_brca` — no sample in multiple BRCA val folds. `test_no_overlap_between_val_folds_coad` — same for COAD. `test_no_train_val_overlap_within_fold_brca` — disjoint sets within each fold. `test_all_samples_covered_once_brca` — every sample appears exactly once in val |
| `tests/test_utils.py` | 4 | `test_set_seeds_produces_reproducible_random_streams` — identical after re-seed. `test_load_config_contains_core_sections` — project/paths/modalities/fusion present. `test_log_experiment_creates_csv_with_expected_columns` — correct headers. `test_get_device_returns_torch_device` — returns `cuda` or `cpu` |

---

## `content/` — Planning & Design Documents (14 files, gitignored)

| File | Purpose |
|---|---|
| `content/CONTEXT.md` | Master agent brain file — project identity, critical rules (patient-level CV, split before normalize, seeds=42, CSV transpose), repo structure, scope control kill-switch cuts, pull request checklist |
| `content/ARCHITECTURE.md` | System blueprint — pipeline stages, module breakdown with I/O contracts, data contracts between stages, runtime modes (training vs demo), non-functional requirements (reproducibility, evaluation integrity) |
| `content/PROJECT-CHARTER.md` | Scope document — in-scope/out-of-scope items, deliverables checklist (17 must-have + 7 nice-to-have), measurable success criteria, compute/time constraints, kill-switch cut priorities, decision log |
| `content/DATA-PROTOCOL.md` | Authoritative data handling rules — 21-point verification checklist, exact preprocessing sequence (14 steps), sample alignment rules, missingness policy (drop <50%), split rules, artifact requirements |
| `content/REQUIREMENTS-STACK.md` | Hardware/software spec — Python 3.9 (target; actual env is 3.11), PyTorch 2.x CUDA 11.8, memory estimates per task, 4 runtime modes, 8 common failure scenarios with fixes, conda+pip hybrid install |
| `content/End-to-End-Roadmap.md` | Phase 0–9 execution roadmap — each phase has agent prompts, verification checkboxes, expected outputs, fallback protocols, git actions, thesis artifact exports |
| `content/PHASE_7.5.md` | Pathway-aware attention layer spec — gene-to-pathway mapping, PathwayAwareFusionModel design, attention mechanism, expected outcomes, fallback conditions |
| `content/PHASE_8.md` | Streamlit demo spec — app structure, model artifact preparation, prediction + SHAP tab, model comparison tab, sample input, fallback protocols |
| `content/REPO-WORKFLOW.md` | Git workflow — folder structure, branch naming (`feature/`, `fix/`, `exp/`, `doc/`), commit message convention (`type(scope): summary`), PR checklist, merge-to-main process |
| `content/COMMIT-STRATEGY.md` | Step-by-step commit plan for separating uncommitted Phase 8 work into logical branches (`feat/demo-app-optimizations`, `chore/project-maintenance`) |
| `content/Tagging.md` | Git tagging conventions — milestone tags (`v0.1-preprocessing`, `v0.2-baselines`...) with semantic versioning |
| `content/report-content.md` | Thesis discussion notes — AUC vs F1 divergence observation for dissertation (RF: best AUC/worst F1 paradox) |
| `content/dev-to-main.md` | PR template rules for dev→main merges |
| `content/feat-to-dev.md` | PR template rules for feature→dev merges |

---

## `docs/` — Project Documentation (4 files)

| File | Purpose |
|---|---|
| `docs/PROJECT_SNAPSHOT.md` | Comprehensive project snapshot — 13 sections: identity, phase status, 5-model comparison table with all metrics, architectures, dataset facts, explainability findings, ablation results, Streamlit demo details, preprocessing verification, known limitations, complete filesystem map, genuine differentiators, open questions |
| `docs/PROJECT_STORY.md` | Narrative project guide for non-technical audiences — the cancer subtype problem, data explained simply, full model journey (XGBoost→RF→EarlyFusionMLP→IntermediateFusion→PathwayAwareFusion), explainability story, ablation discoveries (miRNA artifact), complete results table, demo walkthrough, engineering infrastructure, 8 lessons learned, honest limitations |
| `docs/PRESENTATION_NARRATION.md` | 15-minute presentation script — opening with patient analogy, data walkthrough, ANOVA discovery, model evolution, explainability, ablation findings, results table, demo screenshots, lessons learned, closing. Paced with slide hints |
| `docs/preprocessing_verification_report.md` | 49-check audit results — 46/49 pass, 3 failures documented with root-cause analysis (label shuffle BRCA/COAD, all-NaN miRNA columns). All critical invariants confirmed |

---

## `.github/` — GitHub Templates

| File | Purpose |
|---|---|
| `.github/copilot-instructions.md` | IDE copilot guidance — build/test commands, high-level architecture, key conventions (reproducibility, split discipline, leakage prevention, modality contract, artifact conventions) |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR template — summary, phase reference, changed files (grouped by new/modified/artifacts), key decisions, verification checklist (7 items), metrics table, next step |

---

## `app/` — Streamlit Demo Application

### Main Application

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | ~2,290-line single-page Streamlit app. Sidebar: cancer type selector (BRCA/COAD), 3-model radio (XGBoost/IntermediateFusion/PathwayAwareFusion), About/HowTo/KeyFindings expandables. Tab 1 (Prediction): CSV uploader, sample download, preprocessing, prediction cards with confidence tiers, SHAP waterfall (XGBoost) / IG bar chart (fusion), Groq AI research summary. Tab 2 (Model Comparison): trophy banner, metric cards, full table, ROC curves, per-fold AUC, radar chart, t-SNE latent space, training curves, confusion matrices (4-model selector), biological validation (KEGG/GO/pathway attention), ablation results (modality removal/fusion comparison/missing modality curve). Uses `@st.cache_resource` for model loading |
| `app/sample_input_brca.csv` | Upload-ready sample CSV for BRCA — single de-identified TCGA patient, 15,366 features |
| `app/sample_input_coad.csv` | Upload-ready sample CSV for COAD — single de-identified TCGA patient, 15,200 features |
| `app/sample_input.csv` | Gitignored — legacy un-suffixed alias, regenerated by `prepare_demo_artifacts.py` |

### `app/model_artifacts/` — Precomputed Demo Artifacts (29 files)

**Best-fold indices:** BRCA: xgb=fold3, fusion=fold3, pathway_fusion=fold3 (F1=0.869). COAD: xgb=fold0, fusion=fold3, pathway_fusion=fold1 (F1=0.913).

**Per-cancer configs:**

| File | Description |
|---|---|
| `app/model_artifacts/config_brca.json` | BRCA feature names (15,366), class labels (5 subtypes with names), modality dimensions, best-fold indices |
| `app/model_artifacts/config_coad.json` | COAD feature names (15,200), class labels (4 subtypes with names), modality dimensions, best-fold indices |

**Preprocessing artifacts:**

| File | Description |
|---|---|
| `app/model_artifacts/scaler_brca.pkl` | Concat StandardScaler fitted on BRCA best training fold |
| `app/model_artifacts/scaler_coad.pkl` | Concat StandardScaler fitted on COAD best training fold |
| `app/model_artifacts/imputer_brca.pkl` | PerModalityImputer fitted on BRCA best training fold |
| `app/model_artifacts/imputer_coad.pkl` | PerModalityImputer fitted on COAD best training fold |
| `app/model_artifacts/per_modality_scaler_brca.pkl` | PerModalityScaler fitted on BRCA best training fold |
| `app/model_artifacts/per_modality_scaler_coad.pkl` | PerModalityScaler fitted on COAD best training fold |

**Trained models (best fold per cancer):**

| File | Description |
|---|---|
| `app/model_artifacts/xgb_best_brca.pkl` | Best-fold XGBoost model for BRCA (fold 3) |
| `app/model_artifacts/xgb_best_coad.pkl` | Best-fold XGBoost model for COAD (fold 0) |
| `app/model_artifacts/fusion_best_brca.pt` | Best-fold IntermediateFusionModel for BRCA (fold 3) |
| `app/model_artifacts/fusion_best_coad.pt` | Best-fold IntermediateFusionModel for COAD (fold 3) |
| `app/model_artifacts/pathway_fusion_best_brca.pt` | Best-fold PathwayAwareFusionModel for BRCA (fold 3, F1=0.869) |
| `app/model_artifacts/pathway_fusion_best_coad.pt` | Best-fold PathwayAwareFusionModel for COAD (fold 1, F1=0.913) |

**Pathway mapping:**

| File | Description |
|---|---|
| `app/model_artifacts/pathway_gene_mapping.json` | KEGG pathway→mRNA feature indices for both cancers + unmapped gene indices. Used by PathwayAwareFusion in the demo |

**Precomputed visualizations & attributions:**

| File | Description |
|---|---|
| `app/model_artifacts/latent_space_data_brca.json` | Precomputed t-SNE embeddings of BRCA latent space (from IntermediateFusion.get_latent()) |
| `app/model_artifacts/latent_space_data_coad.json` | Precomputed t-SNE embeddings of COAD latent space |
| `app/model_artifacts/fusion_attribution_results_brca.json` | Precomputed Integrated Gradients per demo sample (IntermediateFusion, BRCA) |
| `app/model_artifacts/fusion_attribution_results_coad.json` | Same for COAD |
| `app/model_artifacts/fusion_attributions_brca.npz` | Raw IG attribution arrays (IntermediateFusion, BRCA) |
| `app/model_artifacts/fusion_attributions_coad.npz` | Same for COAD |
| `app/model_artifacts/pathway_fusion_attribution_results_brca.json` | Precomputed IG per demo sample (PathwayAwareFusion, BRCA) |
| `app/model_artifacts/pathway_fusion_attribution_results_coad.json` | Same for COAD |
| `app/model_artifacts/pathway_fusion_attributions_brca.npz` | Raw IG arrays (PathwayAwareFusion, BRCA) |
| `app/model_artifacts/pathway_fusion_attributions_coad.npz` | Same for COAD |

**Legacy files (exist on disk, explicitly gitignored):**

| File | Description |
|---|---|
| `app/model_artifacts/config.json` | Legacy un-suffixed config (pre-dates per-cancer split) |
| `app/model_artifacts/scaler.pkl` | Legacy un-suffixed scaler |
| `app/model_artifacts/xgb_best.pkl` | Legacy un-suffixed XGBoost model |
| `app/model_artifacts/imputer.pkl` | Legacy un-suffixed imputer |
| `app/model_artifacts/per_modality_scaler.pkl` | Legacy un-suffixed per-modality scaler |

### `app/test_datasets/` — Demo Test Data (27 files)

**Real TCGA patients (`test/` — 13 files):**

| File | Description |
|---|---|
| `app/test_datasets/test/Readme.md` | Documentation: file descriptions, format, usage notes |
| `app/test_datasets/test/test_brca_batch.csv` | 20 real TCGA BRCA val-set patients, 4 per subtype, sample_id column first |
| `app/test_datasets/test/test_brca_single_Basal-like.csv` | Single highest-confidence Basal-like patient |
| `app/test_datasets/test/test_brca_single_HER2-enriched.csv` | Single highest-confidence HER2-enriched patient |
| `app/test_datasets/test/test_brca_single_Luminal_A.csv` | Single highest-confidence Luminal A patient |
| `app/test_datasets/test/test_brca_single_Luminal_B.csv` | Single highest-confidence Luminal B patient |
| `app/test_datasets/test/test_brca_single_Normal-like.csv` | Single highest-confidence Normal-like patient |
| `app/test_datasets/test/test_brca_metadata.csv` | Reference: true labels, XGBoost predictions, confidence scores, clinical descriptions (do not upload to demo) |
| `app/test_datasets/test/test_coad_batch.csv` | 12 real TCGA COAD val-set patients |
| `app/test_datasets/test/test_coad_single_CMS1_MSI-Immune.csv` | Single highest-confidence CMS1 patient |
| `app/test_datasets/test/test_coad_single_CMS2_Canonical.csv` | Single highest-confidence CMS2 patient |
| `app/test_datasets/test/test_coad_single_CMS3_Metabolic.csv` | Single highest-confidence CMS3 patient |
| `app/test_datasets/test/test_coad_metadata.csv` | Reference: true labels + predictions for COAD |

**Synthetic patients (`synthetic/` — 14 files):**

| File | Description |
|---|---|
| `app/test_datasets/synthetic/Readme.md` | Documentation: generation method, file descriptions, why synthetic patients are valuable |
| `app/test_datasets/synthetic/synthetic_brca_all.csv` | 50 synthetic BRCA patients (10 per subtype), features sampled from N(class_mean, class_std) |
| `app/test_datasets/synthetic/synthetic_brca_Basal-like.csv` | 10 synthetic Basal-like patients |
| `app/test_datasets/synthetic/synthetic_brca_HER2-enriched.csv` | 10 synthetic HER2-enriched patients |
| `app/test_datasets/synthetic/synthetic_brca_Luminal_A.csv` | 10 synthetic Luminal A patients |
| `app/test_datasets/synthetic/synthetic_brca_Luminal_B.csv` | 10 synthetic Luminal B patients |
| `app/test_datasets/synthetic/synthetic_brca_Normal-like.csv` | 10 synthetic Normal-like patients |
| `app/test_datasets/synthetic/synthetic_brca_metadata.csv` | Reference: per-row labels, clinical descriptions, modality info |
| `app/test_datasets/synthetic/synthetic_coad_all.csv` | 40 synthetic COAD patients (10 per subtype) |
| `app/test_datasets/synthetic/synthetic_coad_CMS1_MSI-Immune.csv` | 10 synthetic CMS1 patients |
| `app/test_datasets/synthetic/synthetic_coad_CMS2_Canonical.csv` | 10 synthetic CMS2 patients |
| `app/test_datasets/synthetic/synthetic_coad_CMS3_Metabolic.csv` | 10 synthetic CMS3 patients |
| `app/test_datasets/synthetic/synthetic_coad_CMS4_Mesenchymal.csv` | 10 synthetic CMS4 patients |
| `app/test_datasets/synthetic/synthetic_coad_metadata.csv` | Reference: per-row labels + descriptions for COAD |

---

## `notebooks/` — Jupyter Notebooks (7 files)

| File | Purpose |
|---|---|
| `notebooks/00_data_inspect.ipynb` | Phase 2 — exploratory data analysis: file listing, shape checks, NaN/Inf counts, zero-variance features, class distribution plots, sample ID alignment across modalities, data inspection report generation |
| `notebooks/01_preprocessing.ipynb` | Phase 3 — preprocessing verification: 49-point check suite (46 pass, 3 fail), PCA plots, label shuffle tests, fold integrity checks, scaler verification (mean≈0, std≈1 on train only) |
| `notebooks/02_baselines.ipynb` | Phase 4 — baseline evaluation: XGBoost+RF per-fold metrics, confusion matrices, SHAP analysis, baseline comparison across cancers |
| `notebooks/03_latent_fusion.ipynb` | Phase 5 — fusion model evaluation: IntermediateFusion per-fold metrics, latent space analysis, model comparison table generation |
| `notebooks/04_explainability.ipynb` | Phase 6 — interactive explanations: SHAP waterfall/beeswarm for tree models, IG attribution for fusion, KEGG/GO enrichment visualization, pathway validation |
| `notebooks/04.5_pathway_fusion.ipynb` | Phase 7.5 — pathway fusion analysis: PathwayAwareFusion vs IntermediateFusion comparison, attention weight analysis, pathway ranking, training curves |
| `notebooks/04b_ablations.ipynb` | Phase 7 — ablation analysis: modality removal impact, early vs intermediate fusion comparison, missing modality degradation curves, miRNA artifact investigation |

---

## `models/` — Saved Model Checkpoints (40 files)

### Baseline XGBoost — `models/baseline_xgb/` (10 files)

| File | Fold |
|---|---|
| `models/baseline_xgb/xgb_brca_fold0.pkl` | GS-BRCA fold 0 |
| `models/baseline_xgb/xgb_brca_fold1.pkl` | GS-BRCA fold 1 |
| `models/baseline_xgb/xgb_brca_fold2.pkl` | GS-BRCA fold 2 |
| `models/baseline_xgb/xgb_brca_fold3.pkl` | GS-BRCA fold 3 |
| `models/baseline_xgb/xgb_brca_fold4.pkl` | GS-BRCA fold 4 |
| `models/baseline_xgb/xgb_coad_fold0.pkl` | GS-COAD fold 0 |
| `models/baseline_xgb/xgb_coad_fold1.pkl` | GS-COAD fold 1 |
| `models/baseline_xgb/xgb_coad_fold2.pkl` | GS-COAD fold 2 |
| `models/baseline_xgb/xgb_coad_fold3.pkl` | GS-COAD fold 3 |
| `models/baseline_xgb/xgb_coad_fold4.pkl` | GS-COAD fold 4 |

Each `.pkl` is a trained `XGBClassifier` (300 estimators, max_depth=6, lr=0.1, objective=multi:softprob) saved via joblib.

### Baseline Random Forest — `models/baseline_rf/` (10 files)

| File | Fold |
|---|---|
| `models/baseline_rf/rf_brca_fold0.pkl` | GS-BRCA fold 0 |
| `models/baseline_rf/rf_brca_fold1.pkl` | GS-BRCA fold 1 |
| `models/baseline_rf/rf_brca_fold2.pkl` | GS-BRCA fold 2 |
| `models/baseline_rf/rf_brca_fold3.pkl` | GS-BRCA fold 3 |
| `models/baseline_rf/rf_brca_fold4.pkl` | GS-BRCA fold 4 |
| `models/baseline_rf/rf_coad_fold0.pkl` | GS-COAD fold 0 |
| `models/baseline_rf/rf_coad_fold1.pkl` | GS-COAD fold 1 |
| `models/baseline_rf/rf_coad_fold2.pkl` | GS-COAD fold 2 |
| `models/baseline_rf/rf_coad_fold3.pkl` | GS-COAD fold 3 |
| `models/baseline_rf/rf_coad_fold4.pkl` | GS-COAD fold 4 |

Each `.pkl` is a trained `RandomForestClassifier` (300 trees, unlimited depth, class_weight=balanced) saved via joblib.

### Intermediate Fusion — `models/intermediate_fusion/` (10 files)

| File | Fold |
|---|---|
| `models/intermediate_fusion/fusion_brca_fold0.pt` | GS-BRCA fold 0 |
| `models/intermediate_fusion/fusion_brca_fold1.pt` | GS-BRCA fold 1 |
| `models/intermediate_fusion/fusion_brca_fold2.pt` | GS-BRCA fold 2 |
| `models/intermediate_fusion/fusion_brca_fold3.pt` | GS-BRCA fold 3 |
| `models/intermediate_fusion/fusion_brca_fold4.pt` | GS-BRCA fold 4 |
| `models/intermediate_fusion/fusion_coad_fold0.pt` | GS-COAD fold 0 |
| `models/intermediate_fusion/fusion_coad_fold1.pt` | GS-COAD fold 1 |
| `models/intermediate_fusion/fusion_coad_fold2.pt` | GS-COAD fold 2 |
| `models/intermediate_fusion/fusion_coad_fold3.pt` | GS-COAD fold 3 |
| `models/intermediate_fusion/fusion_coad_fold4.pt` | GS-COAD fold 4 |

Each `.pt` is a PyTorch checkpoint containing: `model_state_dict`, `modality_dims`, `latent_dim`, `num_classes`, `encoder_hidden`, `classifier_hidden`, `dropout`.

### Pathway-Aware Fusion — `models/pathway_fusion/` (10 files)

| File | Fold |
|---|---|
| `models/pathway_fusion/pathway_fusion_brca_fold0.pt` | GS-BRCA fold 0 |
| `models/pathway_fusion/pathway_fusion_brca_fold1.pt` | GS-BRCA fold 1 |
| `models/pathway_fusion/pathway_fusion_brca_fold2.pt` | GS-BRCA fold 2 |
| `models/pathway_fusion/pathway_fusion_brca_fold3.pt` | GS-BRCA fold 3 |
| `models/pathway_fusion/pathway_fusion_brca_fold4.pt` | GS-BRCA fold 4 |
| `models/pathway_fusion/pathway_fusion_coad_fold0.pt` | GS-COAD fold 0 |
| `models/pathway_fusion/pathway_fusion_coad_fold1.pt` | GS-COAD fold 1 |
| `models/pathway_fusion/pathway_fusion_coad_fold2.pt` | GS-COAD fold 2 |
| `models/pathway_fusion/pathway_fusion_coad_fold3.pt` | GS-COAD fold 3 |
| `models/pathway_fusion/pathway_fusion_coad_fold4.pt` | GS-COAD fold 4 |

Same checkpoint format as IntermediateFusion, with additional PathwayAttentionEncoder weights.

---

## `results/` — All Generated Outputs (164 files)

### `results/metrics/` — Performance Metrics (15 CSV + 40 NPZ = 55 files)

**Summary comparison CSVs:**

| File | Purpose |
|---|---|
| `results/metrics/model_comparison.csv` | 8 rows — XGBoost/RF/IntermediateFusion/PathwayAwareFusion × BRCA/COAD. Columns: F1/Precision/Recall/NMI/ARI mean±std + auc_mean/auc_std |
| `results/metrics/baseline_comparison.csv` | XGBoost vs RF comparison (earlier version of model_comparison) |
| `results/metrics/auc_scores.csv` | 40 rows — per-fold macro OVR AUC for all 4 models × 2 cancers × 5 folds. COAD fold 0 = NaN (CMS4 absent) |
| `results/metrics/auc_summary.csv` | 8 rows — mean±std AUC across folds for all model-cancer combos |

**Per-model metrics CSVs (per-fold values + mean±std summary row):**

| File | Model | Cancer |
|---|---|---|
| `results/metrics/xgb_brca_metrics.csv` | XGBoost | GS-BRCA |
| `results/metrics/xgb_coad_metrics.csv` | XGBoost | GS-COAD |
| `results/metrics/rf_brca_metrics.csv` | RandomForest | GS-BRCA |
| `results/metrics/rf_coad_metrics.csv` | RandomForest | GS-COAD |
| `results/metrics/fusion_brca_metrics.csv` | IntermediateFusion | GS-BRCA |
| `results/metrics/fusion_coad_metrics.csv` | IntermediateFusion | GS-COAD |
| `results/metrics/pathway_fusion_brca_metrics.csv` | PathwayAwareFusion | GS-BRCA |
| `results/metrics/pathway_fusion_coad_metrics.csv` | PathwayAwareFusion | GS-COAD |

**Ablation comparison CSVs:**

| File | Purpose |
|---|---|
| `results/metrics/ablation_modality_removal.csv` | F1 after removing each modality (mRNA/miRNA/methy/cnv) for BRCA and COAD |
| `results/metrics/ablation_fusion_comparison.csv` | XGBoost vs EarlyFusionMLP vs IntermediateFusion F1 comparison |
| `results/metrics/ablation_missing_modality.csv` | F1 at 0%/10%/20%/30%/50% missingness rates for BRCA and COAD |

**Per-fold prediction archives (y_true, y_pred, y_prob) — 40 files:**

| Pattern | Count |
|---|---|
| `results/metrics/xgb_brca_fold0_predictions.npz` through `xgb_brca_fold4_predictions.npz` | 5 |
| `results/metrics/xgb_coad_fold0_predictions.npz` through `xgb_coad_fold4_predictions.npz` | 5 |
| `results/metrics/rf_brca_fold0_predictions.npz` through `rf_brca_fold4_predictions.npz` | 5 |
| `results/metrics/rf_coad_fold0_predictions.npz` through `rf_coad_fold4_predictions.npz` | 5 |
| `results/metrics/fusion_brca_fold0_predictions.npz` through `fusion_brca_fold4_predictions.npz` | 5 |
| `results/metrics/fusion_coad_fold0_predictions.npz` through `fusion_coad_fold4_predictions.npz` | 5 |
| `results/metrics/pathway_fusion_brca_fold0_predictions.npz` through `pathway_fusion_brca_fold4_predictions.npz` | 5 |
| `results/metrics/pathway_fusion_coad_fold0_predictions.npz` through `pathway_fusion_coad_fold4_predictions.npz` | 5 |

### `results/plots/` — Visualization Plots (48 files)

**Ablation plots:**

| File(s) | Content |
|---|---|
| `ablation_modality_removal.{pdf,png}` | Bar chart: F1 after removing each modality |
| `ablation_fusion_comparison.{pdf,png}` | Bar chart: early vs intermediate fusion comparison |
| `missing_modality_curve.{pdf,png}` | Line chart: F1 vs missing modality rate |
| `nb_ablation_modality_removal.{pdf,png}` | Notebook-generated modality removal plot |
| `nb_ablation_fusion_comparison.{pdf,png}` | Notebook-generated fusion comparison plot |
| `nb_missing_modality_curve.{pdf,png}` | Notebook-generated missing modality curve |

**Baseline comparison:**

| File(s) | Content |
|---|---|
| `baseline_comparison_BRCA.{pdf,png}` | XGBoost vs RF bar chart, BRCA |
| `baseline_comparison_COAD.{pdf,png}` | XGBoost vs RF bar chart, COAD |

**Confusion matrices:**

| File(s) | Content |
|---|---|
| `confusion_matrix_xgb_BRCA.png`, `confusion_matrix_xgb_COAD.png` | XGBoost best-fold confusion matrices |
| `confusion_matrix_rf_BRCA.png`, `confusion_matrix_rf_COAD.png` | RF best-fold confusion matrices |
| `confusion_matrix_fusion_BRCA.{pdf,png}`, `confusion_matrix_fusion_COAD.{pdf,png}` | IntermediateFusion best-fold confusion matrices |
| `confusion_matrix_pathway_fusion_BRCA.{pdf,png}`, `confusion_matrix_pathway_fusion_COAD.{pdf,png}` | PathwayAwareFusion best-fold confusion matrices |

**Training curves:**

| File(s) | Content |
|---|---|
| `fusion_training_curves_BRCA.{pdf,png}` | IntermediateFusion train/val loss + val F1 per fold, BRCA |
| `fusion_training_curves_COAD.{pdf,png}` | Same for COAD |
| `pathway_fusion_training_curves_BRCA.{pdf,png}` | PathwayAwareFusion training curves, BRCA |
| `pathway_fusion_training_curves_COAD.{pdf,png}` | Same for COAD |

**Model comparison:**

| File(s) | Content |
|---|---|
| `model_comparison_BRCA.{pdf,png}` | All-models bar chart with error bars, BRCA |
| `model_comparison_COAD.{pdf,png}` | All-models bar chart with error bars, COAD |
| `pathway_vs_plain_comparison.{pdf,png}` | PathwayAwareFusion vs IntermediateFusion comparison |
| `pathway_attention_top10.{pdf,png}` | Top-10 pathway attention weights bar chart |

**Data visualization:**

| File(s) | Content |
|---|---|
| `class_distribution_BRCA.png` | BRCA subtype label distribution bar chart |
| `class_distribution_COAD.png` | COAD subtype label distribution bar chart |
| `pca_preprocessed_BRCA.png` | PCA plot of preprocessed BRCA data |
| `pca_preprocessed_COAD.png` | PCA plot of preprocessed COAD data |

### `results/qc/` — Quality Control (2 files)

| File | Purpose |
|---|---|
| `results/qc/data_inspection_report.json` | Phase 2 output — shapes, NaN counts, zero-variance features, class distributions for both cancers |
| `results/qc/preprocessing_verification.json` | Phase 3 output — machine-readable 46/49 pass results with per-check pass/fail status |

### `results/shap/` — SHAP & Attribution Artifacts (37 files)

**XGBoost SHAP (`shap/xgb/` — 14 files):**

| File(s) | Content |
|---|---|
| `top_50_features_xgb_BRCA.csv`, `top_50_features_xgb_COAD.csv` | Top-50 features by mean absolute SHAP value |
| `xgb_shap_bar_BRCA.{pdf,png}`, `xgb_shap_bar_COAD.{pdf,png}` | SHAP bar plots (feature importance) |
| `xgb_shap_summary_BRCA.{pdf,png}`, `xgb_shap_summary_COAD.{pdf,png}` | SHAP beeswarm summary plots |
| `xgb_shap_comparison.{pdf,png}` | Side-by-side BRCA vs COAD SHAP comparison |
| `xgb_shap_values_BRCA.npz`, `xgb_shap_values_COAD.npz` | Raw SHAP value arrays |

**Random Forest SHAP (`shap/rf/` — 12 files):**

| File(s) | Content |
|---|---|
| `top_50_features_rf_BRCA.csv`, `top_50_features_rf_COAD.csv` | Top-50 features by mean absolute SHAP value |
| `rf_shap_bar_BRCA.{pdf,png}`, `rf_shap_bar_COAD.{pdf,png}` | SHAP bar plots |
| `rf_shap_summary_BRCA.{pdf,png}`, `rf_shap_summary_COAD.{pdf,png}` | SHAP beeswarm summary plots |
| `rf_shap_values_BRCA.npz`, `rf_shap_values_COAD.npz` | Raw SHAP value arrays |

**Fusion Model Attribution (`shap/fusion/` — 11 files):**

| File(s) | Content |
|---|---|
| `top_50_features_fusion_BRCA.csv`, `top_50_features_fusion_COAD.csv` | Top-50 by IG attribution (all features) |
| `top_50_mrna_fusion_BRCA.csv`, `top_50_mrna_fusion_COAD.csv` | Top-50 mRNA-only by IG attribution |
| `fusion_attribution_BRCA.{pdf,png}` | IG bar chart with modality colors, BRCA |
| `fusion_attribution_COAD.{pdf,png}` | IG bar chart with modality colors, COAD |
| `fusion_ig_comparison.{pdf,png}` | Side-by-side BRCA vs COAD IG comparison |
| `modality_share_pie.png` | Pie chart: modality share of top-50 features |

### `results/enrichment/` — Pathway Enrichment (22 files)

**KEGG enrichment:**

| File | Content |
|---|---|
| `kegg_fusion_BRCA.csv` | 4 significant KEGG terms for IntermediateFusion BRCA (Cell cycle, Oocyte meiosis, p53 signaling, HTLV-1 infection) |
| `kegg_fusion_COAD.csv` | Empty — 0 significant terms |
| `kegg_xgb_BRCA.csv` | Empty — 0 significant terms |
| `kegg_xgb_COAD.csv` | Empty — 0 significant terms |

**GO Biological Process enrichment:**

| File | Content |
|---|---|
| `go_fusion_BRCA.csv` | 57 significant GO BP terms for IntermediateFusion BRCA |
| `go_fusion_COAD.csv` | Empty — 0 significant terms |
| `go_xgb_BRCA.csv` | Empty — 0 significant terms |
| `go_xgb_COAD.csv` | Empty — 0 significant terms |

**Pathway validation:**

| File | Content |
|---|---|
| `pathway_validation_BRCA.json` | Pass/fail for 9 canonical cancer pathways, BRCA |
| `pathway_validation_COAD.json` | Pass/fail for 9 canonical cancer pathways, COAD |
| `pathway_validation_table_BRCA.csv` | Formatted pathway validation table, BRCA |
| `pathway_validation_table_COAD.csv` | Formatted pathway validation table, COAD |

**Pathway attention:**

| File | Content |
|---|---|
| `pathway_attention_scores.csv` | Mean attention weight per KEGG pathway from PathwayAwareFusion (nearly uniform, ~0.003-0.004) |

**Visualization plots:**

| File(s) | Content |
|---|---|
| `kegg_barplot_fusion_BRCA.{pdf,png}` | KEGG enrichment bar chart |
| `kegg_enrichment_BRCA_notebook.{pdf,png}` | Notebook-generated KEGG enrichment plot |
| `enrichment_barplot_BRCA.{pdf,png}` | Combined enrichment bar chart |
| `go_enrichment_BRCA_notebook.{pdf,png}` | Notebook-generated GO enrichment plot |
| `.gitkeep` | Keeps empty directory tracked in Git |

---

## `data/` — Data Files

### Raw Data — `data/raw/` (10 files, gitignored)

**GS-BRCA (5 files):**

| File | Description |
|---|---|
| `data/raw/GS-BRCA/Top/BRCA_mRNA_top.csv` | 5,000 mRNA features × 671 samples (~61 MB). Rows=genes, cols=samples — transposed by loader |
| `data/raw/GS-BRCA/Top/BRCA_miRNA_top.csv` | 366 miRNA features × 671 samples (~2 MB). 166/366 columns all-NaN |
| `data/raw/GS-BRCA/Top/BRCA_Methy_top.csv` | 5,000 DNA methylation features × 671 samples (~62 MB) |
| `data/raw/GS-BRCA/Top/BRCA_CNV_top.csv` | 5,000 CNV features × 671 samples (~61 MB) |
| `data/raw/GS-BRCA/Top/1_BRCA_label_num.csv` | Subtype labels — 671 rows. Numeric prefix `1_` from Figshare. Positionally aligned with mRNA columns (no sample-ID column) |

**GS-COAD (5 files):**

| File | Description |
|---|---|
| `data/raw/GS-COAD/Top/COAD_mRNA_top.csv` | 5,000 mRNA features × 260 samples (~24 MB) |
| `data/raw/GS-COAD/Top/COAD_miRNA_top.csv` | 200 miRNA features × 260 samples (~1 MB) |
| `data/raw/GS-COAD/Top/COAD_Methy_top.csv` | 5,000 DNA methylation features × 260 samples (~24 MB) |
| `data/raw/GS-COAD/Top/COAD_CNV_top.csv` | 5,000 CNV features × 260 samples (~24 MB) |
| `data/raw/GS-COAD/Top/1_COAD_label_num.csv` | Subtype labels — 260 rows. Positionally aligned with mRNA columns |

### Processed Data — `data/processed/` (committed)

| File | Description |
|---|---|
| `data/processed/pathway_gene_mapping.json` | KEGG pathway→mRNA feature index mapping for both cancers. Used by PathwayAttentionEncoder. Contains `pathway_indices` dict and `unmapped_indices` list |

### Toy Data — `data/toy/` (6 files, committed)

| File | Description |
|---|---|
| `data/toy/BRCA_mRNA_top_toy.csv` | 5,000 mRNA features × 50 stratified BRCA samples |
| `data/toy/BRCA_miRNA_top_toy.csv` | 366 miRNA features × 50 samples |
| `data/toy/BRCA_Methy_top_toy.csv` | 5,000 methylation features × 50 samples |
| `data/toy/BRCA_CNV_top_toy.csv` | 5,000 CNV features × 50 samples |
| `data/toy/BRCA_label_num_toy.csv` | Subtype labels for 50 samples (all 5 subtypes represented) |
| `data/toy/cv_folds.json` | 5-fold CV splits for toy data (used by tests and `--toy` flag) |
| `data/toy/.gitkeep` | Keeps directory tracked |

### Preprocessed Data — `data/preprocessed/` (gitignored)

| File | Description |
|---|---|
| `data/preprocessed/.gitkeep` | Directory marker — preprocessing happens in-memory per fold, so no processed CSVs exist on disk |

### Core Data Artifacts (committed)

| File | Description |
|---|---|
| `data/cv_folds.json` | Canonical 5-fold CV splits for both cancers. Structure: `{cancer_type: {n_folds, seed, n_samples, class_distribution, folds: [{fold, train: [ids], val: [ids]}]}}`. BRCA=671 samples, COAD=260 samples |
| `data/sample_map.csv` | 931 rows — sample ID, has_mrna/mirna/methy/cnv boolean flags, modality_count, cancer_type. All samples have 4/4 modalities |
| `data/dropped_samples.csv` | 0 rows — all 931 samples kept (all had ≥50% modalities) |
| `data/checksums_brca.txt` | SHA256 checksums for 5 raw BRCA files |
| `data/checksums_coad.txt` | SHA256 checksums for 5 raw COAD files |
| `data/DATA_README.md` | Download instructions, citation, file placement, dataset statistics tables, CSV format notes, known limitations (ANOVA pre-selection), checksum verification |

---

## `.agents/` — Agent Skills & Rules (gitignored, 45 files)

*These are AI coding agent configuration files. Not project deliverables. Listed for completeness.*

**Rule files (5):**

| File | Purpose |
|---|---|
| `.agents/rules/CLAUDE.md` | Claude-specific agent configuration |
| `.agents/rules/context.md` | Agent context/grounding rules |
| `.agents/rules/dev-to-main.md` | PR rules for dev→main merges |
| `.agents/rules/feat-to-dev.md` | PR rules for feature→dev merges |
| `.agents/rules/repo-workflow.md` | Repository workflow rules for agent |

**Skill bundles (40 files across 9 skill directories):**
- `agent-browser/` — Browser automation skill
- `code-quality/` — Code quality reference
- `data-quality-frameworks/` — Data quality validation patterns
- `documentation/` — Documentation writing guides
- `frontend-design/` — Frontend design patterns + LICENSE
- `performance-profiling/` — Performance measurement scripts
- `scikit-learn/` — sklearn references (6 files) + example scripts (2 files)
- `security-review/` — Security review references (16 files) + language guides (2 files) + LICENSE
- `web-design-guidelines/` — Web design compliance rules

---

## `.claude/` — Claude IDE Config (gitignored)

| File | Purpose |
|---|---|
| `.claude/CLAUDE.md` | Claude IDE project context |
| `.claude/settings.local.json` | Local IDE settings |

---

## Metadata

```
Generated: 2026-05-03
Project: MLOmics v1.0-final
Author: Dineth Hettiarachchi
Total files: 441
Git branch: dev (HEAD = bac93b6)
Git tags: v0.0-scaffold, v0.0.1-data, v0.1-preprocessing, v0.2-baselines, v0.3-fusion, v0.4-analysis, v1.0-final

File type breakdown:
  Python (.py):       22  (6 src + 15 scripts + 1 agent skill)
  Jupyter (.ipynb):    7
  Markdown (.md):      26  (root docs + content/ + docs/ + .github/ + .agents/ + app readmes + data)
  YAML (.yaml):        1   (config.yaml)
  JSON (.json):       18  (configs, attributions, enrichment results, QC reports, settings)
  CSV (.csv):         51  (15 metrics + 22 enrichment + 1 sample_map + 1 dropped + 2 checksums + 10 test dataset metadata/samples)
  Pickle (.pkl):      50  (10 xgb + 10 rf + 29 app artifacts)
  PyTorch (.pt):      30  (10 fusion + 10 pathway_fusion + 10 app best-fold)
  NumPy (.npz):       53  (40 fold predictions + 4 SHAP values + 4 fusion attributions + 5 pathway attributions)
  PDF (.pdf):         24  (plots + shap + enrichment)
  PNG (.png):         27  (plots + shap + enrichment)
  TXT (.txt):          3  (checksums + requirements)
  Other:              129  (.gitignore, .env, .streamlit, .gitkeep, .claude, .agents skill files, test dataset CSVs, LICENSE)
```
