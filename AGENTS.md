# AGENTS.md

Guidance for OpenCode when working in this repository.

## Project Overview

MLOmics is a multi-omics cancer subtype classifier (BSc final-year research project). It trains XGBoost/RandomForest baselines and an intermediate-fusion neural network (per-modality encoders → latent concatenation → MLP) on four omics modalities (mRNA, miRNA, DNA methylation, CNV) to classify cancer subtypes for GS-BRCA (5 subtypes) and GS-COAD (4 subtypes).

**Stack:** Python 3.11, PyTorch 2.7 (CUDA 11.8), scikit-learn, XGBoost, SHAP, Captum, gseapy, Streamlit.

## Environment Setup

```bash
conda create -n mlomics python=3.11 -y
conda activate mlomics
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Common Commands

All commands must be run from the **project root** — `load_config()` reads `config.yaml` from CWD.

```bash
conda activate mlomics

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Quick smoke-test (toy data, no GPU required)
python scripts/train_baselines.py --toy
python scripts/train_fusion.py --toy --epochs 5
python -m pytest tests/ -v

# Train baselines (XGBoost + RandomForest, 5-fold CV)
python scripts/train_baselines.py
python scripts/train_baselines.py --model xgb
python scripts/train_baselines.py --model rf --toy

# Train fusion models
python scripts/train_fusion.py
python scripts/train_fusion.py --toy --epochs 10
python scripts/train_pathway_fusion.py

# Explainability & ablations
python scripts/run_explainability.py
python scripts/run_ablations.py
python scripts/shap_analysis.py

# Demo preparation & launch
python scripts/prepare_demo_artifacts.py          # Export best-fold models → app/model_artifacts/
python scripts/precompute_fusion_attribution.py   # Run once after training
python scripts/precompute_latent_space.py         # Run once after training
streamlit run app/streamlit_app.py                # Launch demo (port 8501)

# Evaluation & utilities
python scripts/run_evaluation.py
python scripts/compute_auc.py                     # Requires trained models
python scripts/create_toy_dataset.py

# Testing
python -m pytest tests/ -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_models.py::test_intermediate_fusion_output_shape -v

# Notebooks — clear output before committing
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

## Import System — No Package Install

This project is **not installed as a package** (no `setup.py`, no `pyproject.toml`). Every script manually adds the project root to `sys.path`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

Tests work because `pytest` from the project root adds CWD to `sys.path`, making `import src.utils` resolvable. When writing a new script, copy the `sys.path.insert` pattern from any existing script.

## Architecture

```
DATA → PREPROCESS → TRAIN → EVALUATE → EXPLAIN → DEMO
```

All hyperparameters, paths, and seeds live in `config.yaml`. Load with `from src.utils import load_config; config = load_config()`.

### `src/` Modules

| Module | Responsibility |
|---|---|
| `utils.py` | `set_seeds()`, `load_config()`, `log_experiment()`, `get_device()` |
| `data_loader.py` | Load MLOmics CSVs (features as rows, samples as columns — **transpose required**); align sample IDs across modalities |
| `preprocessing.py` | z-score normalization, median NaN imputation, `prepare_fold_data()`, `concatenate_modalities()` |
| `models.py` | `ModalityEncoder` (dense 256→64), `FusionClassifier` (128→num_classes), `IntermediateFusionModel`, `PathwayAwareFusionModel`, `EarlyFusionMLP`, `MultiOmicsDataset` |
| `evaluation.py` | `compute_metrics()`, `compute_fold_summary()`, `save_metrics()` |
| `explainability.py` | SHAP TreeExplainer / DeepSHAP / Integrated Gradients; gseapy KEGG enrichment |

### Demo App (`app/streamlit_app.py`)

Launched from project root (adds root to `sys.path`). Loads artifacts from `app/model_artifacts/` at startup (cached with `@st.cache_resource`). All artifacts produced by `scripts/prepare_demo_artifacts.py`.

Production logic belongs in `src/` or `scripts/`; notebooks are thin orchestrators for exploration.

## Key Invariants (DO NOT BREAK)

1. **`set_seeds(42)` first call** after imports in every script/notebook. The fusion training scripts re-call it before each fold for per-fold reproducibility (`scripts/train_fusion.py:245`).

2. **CV folds are fixed** — always loaded from `data/cv_folds.json` via `load_cv_folds()`. Never regenerate. All models must use identical splits.

3. **Scalers/Imputers fit on train folds only** — `prepare_fold_data()` handles this. Never fit on the full dataset.

4. **Patient-level splits** — all samples from one patient go to the same fold.

5. **Every training run appends to `experiment_log.csv`** via `log_experiment()`.

6. **`matplotlib.use("Agg")` is mandatory** in every script that uses matplotlib (no display on training nodes).

## Data Gotchas

- **Label CSV positionally aligned** (`src/data_loader.py:119-132`): The label file has no sample-ID column. Rows are assumed to match the column order of the mRNA CSV. If files are ever regenerated independently, this silently produces wrong class assignments.

- **ANOVA pre-selection bias**: Input CSVs contain only top-k features pre-selected by ANOVA F-test across ALL samples — label information leaked before CV splitting. A label-shuffle sanity test gives artificially elevated F1 (~0.4–0.5 vs expected ~0.2 for 5-class random). See `docs/preprocessing_verification_report.md`.

- **Modality key naming**: The config key for DNA methylation is `methy` (not `methylation`). Canonical keys: `["mrna", "mirna", "methy", "cnv"]`. Raw CSV filenames use `Methy`: e.g., `GS-BRCA_Methy_top.csv`.

- **Toy data = BRCA only**: The `--toy` flag restricts cancer types to `["GS-BRCA"]`. Toy data for COAD does not exist.

- **CSV transpose**: Raw CSVs store features as rows and samples as columns. `load_modality()` transposes on load to (n_samples, n_features).

- **miRNA mixed-type columns**: `prepare_fold_data()` force-casts all DataFrame columns to `str` to handle miRNA files with mixed float/string feature names.

## Hardware Constraints

Target: i7 11th Gen, 16GB RAM, 4GB VRAM. ~15,200 features per sample after concatenation. Batch size 32 for deep models; `num_workers=0` in DataLoader (avoids multiprocessing issues on Windows).

## Data Layout

- `data/raw/` — gitignored; download instructions in `data/DATA_README.md`
- `data/preprocessed/` — gitignored; regenerated by `notebooks/01_preprocessing.ipynb`
- `data/toy/` — small committed subset for testing
- `data/cv_folds.json` — canonical fold indices (committed; reused by all models)
- Trained models: `models/baseline_xgb/`, `models/baseline_rf/`, `models/intermediate_fusion/`, named `{model}_{cancer}_fold{i}.{pkl|pt}`

## Git Workflow

Branches: `feature/`, `fix/`, `exp/`, `doc/`. Commit format: `type(scope): summary` where type ∈ `feat`, `fix`, `data`, `exp`, `docs`, `chore`. Merge to `main` via squash merge.
