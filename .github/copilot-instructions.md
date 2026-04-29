# Copilot instructions for MLOmics

## Build, test, and lint commands

Run commands from the repository root (`load_config()` expects `config.yaml` in CWD).

```bash
# install deps
pip install -r requirements.txt

# tests (full suite)
python -m pytest tests -v

# tests (single file)
python -m pytest tests/test_models.py -v

# tests (single test)
python -m pytest tests/test_models.py::test_intermediate_fusion_forward_and_latent_shapes -v
```

There is no dedicated lint target/config in this repo (no `ruff`, `flake8`, or `pylint` config files).

## High-level architecture

The repository is an end-to-end multi-omics pipeline with this flow:

`data loading -> fold preprocessing -> model training -> metrics/artifacts -> explainability -> Streamlit demo`

1. **Configuration-first runtime**
   - `config.yaml` is the single source of truth for paths, modalities, model hyperparameters, folds path, and seed.
   - Scripts call `src.utils.load_config()` and `src.utils.set_seeds(42)` at startup.

2. **Core library (`src/`)**
   - `src/data_loader.py`: loads modality/label CSVs, resolves raw vs toy paths, and transposes modality files to **samples x features**.
   - `src/preprocessing.py`: fold-level leakage-safe preprocessing (`prepare_fold_data`) with train-only imputation/scaling; CV fold creation/loading.
   - `src/models.py`: model definitions (`IntermediateFusionModel`, `PathwayAwareFusionModel`, tree-ready dataset wrappers).
   - `src/evaluation.py`: shared metrics and metrics CSV writing.
   - `src/explainability.py`: SHAP/IG and enrichment utilities used by explainability scripts.

3. **Orchestration scripts (`scripts/`)**
   - `train_baselines.py`: XGBoost/RandomForest 5-fold training and artifact export.
   - `train_fusion.py`: intermediate fusion training.
   - `train_pathway_fusion.py`: pathway-aware fusion training (KEGG-guided mRNA encoder).
   - `run_explainability.py`, `shap_analysis.py`, `run_ablations.py`, `compute_auc.py`, `run_evaluation.py`: downstream evaluation/analysis and summary artifacts.
   - `prepare_demo_artifacts.py`: packages best-fold model artifacts and config for the app.

4. **Demo app (`app/streamlit_app.py`)**
   - Loads per-cancer artifacts from `app/model_artifacts/` (`config_*`, `scaler_*`, `xgb_best_*`, `fusion_best_*`, `pathway_fusion_best_*`, `pathway_gene_mapping.json`).
   - Expects precomputed outputs in `results/` for model comparison/enrichment/attribution views.

## Key conventions in this codebase

1. **Reproducibility and split discipline**
   - `set_seeds(42)` is expected immediately after imports in scripts/tests.
   - Cross-model comparisons rely on shared fixed fold files (`data/cv_folds.json`; toy uses `data/toy/cv_folds.json`).
   - Training scripts load folds; they do not regenerate splits during normal runs.

2. **Leakage prevention lives in preprocessing helpers**
   - Use `prepare_fold_data()` instead of ad-hoc preprocessing in scripts.
   - Imputation and scaling are always fit on train fold only, then applied to val fold.
   - Train/val overlap assertions are built into fold prep.

3. **Canonical modality contract**
   - Modality keys/order are `mrna`, `mirna`, `methy`, `cnv` (`MODALITY_KEYS`).
   - Modality CSVs are read as features x samples and transposed in loader.
   - Early-fusion concatenation uses modality-prefixed feature names (`mrna_*`, `mirna_*`, ...).

4. **Artifact and logging conventions**
   - Every training run appends to `experiment_log.csv` via `log_experiment()`.
   - Saved model naming is fold-specific and cancer-specific (for example `fusion_brca_fold0.pt`, `xgb_coad_fold3.pkl`).
   - Demo artifact names must match the app loader expectations (`*_brca` / `*_coad` suffixes).
