# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

MLOmics is a multi-omics cancer subtype classifier — a BSc final-year research project. It trains XGBoost/RandomForest baselines and an intermediate-fusion neural network (per-modality encoders → latent concatenation → MLP) on four omics modalities (mRNA, miRNA, DNA methylation, CNV) to classify cancer subtypes for GS-BRCA (5 subtypes) and GS-COAD (4 subtypes).

**Stack:** Python 3.9, PyTorch 2.x (CUDA 11.8), scikit-learn, XGBoost, SHAP, Captum, gseapy, Streamlit.

## Common Commands

# create and activate conda environment (named `mlomics`), then install dependencies:

```bash
conda create -n mlomics python=3.11 -y
conda activate mlomics
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

```bash
# Environment
conda activate mlomics
python -c "import torch; print(torch.cuda.is_available())"  # Verify GPU

# Train baselines (XGBoost + RandomForest, 5-fold CV)
python scripts/train_baselines.py                   # Both models, full data
python scripts/train_baselines.py --model xgb       # XGBoost only
python scripts/train_baselines.py --model rf --toy  # RandomForest on toy data (fast)

# Train fusion models
python scripts/train_fusion.py                      # Intermediate fusion, both cancers
python scripts/train_fusion.py --toy --epochs 10    # Quick test on toy data
python scripts/train_pathway_fusion.py              # Pathway-aware fusion variant

# Explainability & ablations
python scripts/run_ablations.py
python scripts/run_explainability.py
python scripts/shap_analysis.py

# Demo preparation & launch
python scripts/prepare_demo_artifacts.py            # Export best-fold models → app/model_artifacts/
python scripts/precompute_fusion_attribution.py     # Run once after training
python scripts/precompute_latent_space.py           # Run once after training
streamlit run app/streamlit_app.py                  # Launch demo (port 8501)

# Evaluation & utilities
python scripts/run_evaluation.py
python scripts/compute_auc.py                       # Requires trained models
python scripts/create_toy_dataset.py

# Testing
python -m pytest tests/ -v                          # Full suite
python -m pytest tests/test_models.py -v            # Single module
python -m pytest tests/test_models.py::test_intermediate_fusion_output_shape -v  # Single test

# Notebooks — clear output before committing
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

All scripts must be run from the **project root** — `load_config()` defaults to `config.yaml` in the current working directory.

## Architecture

### Pipeline Stages (in order)

```
DATA → PREPROCESS → TRAIN → EVALUATE → EXPLAIN → DEMO
```

All hyperparameters, paths, and seeds live in `config.yaml`. Load with:

```python
from src.utils import load_config, set_seeds
config = load_config()   # reads config.yaml from CWD
set_seeds(42)
```

### `src/` Modules

| Module | Responsibility |
|---|---|
| `data_loader.py` | Load MLOmics CSVs (features as rows, samples as columns — **transpose required**); align sample IDs across modalities |
| `preprocessing.py` | z-score normalization, sample filtering (drop if >50% modalities missing), `load_cv_folds()`, `prepare_fold_data()`, `concatenate_modalities()` |
| `models.py` | All model classes (see below) |
| `evaluation.py` | 5-fold CV loop, `compute_metrics()`, `compute_fold_summary()`, `save_metrics()` |
| `explainability.py` | SHAP TreeExplainer / DeepSHAP / Integrated Gradients; gseapy KEGG enrichment |
| `utils.py` | `set_seeds()`, `load_config()`, `log_experiment()`, `get_device()` |

### Model Classes in `src/models.py`

- **`ModalityEncoder`** — 2-layer dense encoder: `input_dim → 256 → latent_dim` (default 64) with BatchNorm + ReLU + Dropout
- **`FusionClassifier`** — MLP head: `total_latent → 128 → num_classes`
- **`IntermediateFusionModel`** — One `ModalityEncoder` per modality → concatenate latents → `FusionClassifier`. `get_latent()` extracts embeddings for visualization.
- **`PathwayAttentionEncoder`** — Dual-path mRNA encoder: Path A groups genes by KEGG pathway → mean-pool → learnable attention weights → project; Path B handles unmapped genes via standard encoder. `get_attention_weights()` returns last-pass attention.
- **`PathwayAwareFusionModel`** — Replaces mRNA `ModalityEncoder` with `PathwayAttentionEncoder`; other modalities use standard encoders.
- **`EarlyFusionMLP`** — Ablation baseline: concatenated features → single MLP.
- **`MultiOmicsDataset`** — PyTorch Dataset wrapping `{modality: ndarray}` dicts.

### Demo App (`app/streamlit_app.py`)

Launched from the project root (adds project root to `sys.path`). Loads artifacts from `app/model_artifacts/` at startup (cached with `@st.cache_resource`). Requires cancer-specific artifacts (`config_{brca,coad}.json`, `scaler_{brca,coad}.pkl`, `xgb_best_{brca,coad}.pkl`, `fusion_best_{brca,coad}.pt`, `pathway_fusion_best_{brca,coad}.pt`, `pathway_gene_mapping.json`) — all produced by `scripts/prepare_demo_artifacts.py`. Supports GS-BRCA (5 subtypes) and GS-COAD (4 subtypes) with 3 model options: XGBoost, Intermediate Fusion, Pathway-Aware Fusion.

### Notebooks (Execution Order)

| Notebook | Purpose |
|---|---|
| `00_data_inspect.ipynb` | EDA, shape checks |
| `01_preprocessing.ipynb` | Run preprocessing pipeline, save processed matrices |
| `02_baselines.ipynb` | Train XGBoost + RF with 5-fold CV |
| `03_latent_fusion.ipynb` | Train intermediate fusion model |
| `04_explainability.ipynb` | SHAP + DeepSHAP + KEGG enrichment |
| `04b_ablations.ipynb` | Modality removal, missing-modality curves |
| `04.5_pathway_fusion.ipynb` | Pathway-aware fusion experiment |

Production logic belongs in `src/` or `scripts/`; notebooks are thin orchestrators for exploration and visualization.

## Key Invariants

1. **`set_seeds(42)` must be the first call** after imports in every script/notebook.
2. **CV folds are fixed** — always loaded from `data/cv_folds.json`; never re-generated to ensure all models are compared on identical splits.
3. **Scalers are fit only on train folds** — `prepare_fold_data()` handles this; never fit on the full dataset.
4. **Patient-level splits** — all samples from one patient go to the same fold; no sample overlap between train/val.
5. **Every training run appends to `experiment_log.csv`** — use `log_experiment()` from `src.utils`.

## Data Layout

- `data/raw/` — gitignored; download instructions in `data/DATA_README.md`. CSV naming: `{CANCER}_{modality}_top.csv` (e.g., `GS-BRCA_mRNA_top.csv`). Label files match `*_label_num.csv`.
- `data/preprocessed/` — gitignored; regenerated by `notebooks/01_preprocessing.ipynb`.
- `data/toy/` — small committed subset for rapid testing; toy CSV naming: `{CANCER}_{modality}_top_toy.csv`.
- `data/cv_folds.json` — canonical fold indices (committed; reused by all models).
- Trained models save to `models/baseline_xgb/`, `models/baseline_rf/`, `models/intermediate_fusion/`, named `{model}_{cancer}_fold{i}.{pkl|pt}`.

## Git Workflow

Branches: `feature/`, `fix/`, `exp/`, `doc/` prefixes. Commit format: `type(scope): summary` where type is `feat`, `fix`, `data`, `exp`, `docs`, or `chore` and scope hints are `data`, `model`, `eval`, `app`, `nb`. Merge to `main` via squash merge. Milestone tags: `v0.1-baselines`, `v0.2-fusion`, `v0.3-explainability`, `v0.4-demo`, `v1.0-final`.

Hardware target: i7 11th Gen, 16GB RAM, 4GB VRAM. Top features only (~15,200 features per sample); batch size 32–64 for deep models.
