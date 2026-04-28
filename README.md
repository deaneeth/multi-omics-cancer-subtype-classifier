# MLOmics: Latent-Fusion Multi-Omics Classifier for Cancer Subtype Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **⚠️ Academic research prototype. Not for clinical use.**

## Overview

MLOmics is an end-to-end, reproducible multi-omics pipeline for cancer subtype classification on the [MLOmics benchmark dataset](https://figshare.com/articles/dataset/MLOmics_Cancer_Multi-Omics_Database_for_Machine_Learning/28729127).

It evaluates two cancer cohorts with four omics modalities:

- GS-BRCA: 5 subtypes
- GS-COAD: 4 subtypes
- Modalities: mRNA, miRNA, DNA methylation, CNV

The project includes early-fusion baselines, intermediate fusion, and a pathway-aware fusion variant that injects KEGG pathway structure into the mRNA encoder. The Streamlit demo supports both cancers and all three deployed model families.

## Models

| Model | Approach |
|---|---|
| XGBoost | Early fusion baseline on concatenated features |
| Random Forest | Early fusion baseline on concatenated features |
| EarlyFusionMLP | Ablation baseline on concatenated features |
| IntermediateFusion | Per-modality encoders → latent concat → MLP classifier |
| PathwayAwareFusion | KEGG-guided mRNA encoder + standard modality encoders |

## Key Outputs

- `results/metrics/model_comparison.csv` now includes F1, precision, recall, NMI, ARI, accuracy, and AUC.
- `results/metrics/auc_scores.csv` and `results/metrics/auc_summary.csv` store per-fold and mean AUC values.
- `app/model_artifacts/` contains the per-cancer demo artifacts used by the Streamlit app.
- `docs/preprocessing_verification_report.md` records the verified preprocessing checks and documented limitations.

## Project Structure

```
├── app/            # Streamlit demo application and demo artifacts
├── data/           # Raw, toy, and derived data files
├── docs/           # Verification and reporting documents
├── models/         # Saved model checkpoints per fold
├── notebooks/      # Exploration, preprocessing, and analysis notebooks
├── results/        # Metrics, plots, SHAP, enrichment, and QC outputs
├── scripts/        # Training, evaluation, attribution, and export scripts
├── src/            # Core data, preprocessing, model, and evaluation code
├── tests/          # Unit tests
├── config.yaml     # Project configuration and hyperparameters
└── experiment_log.csv  # Logged experiments
```

## Setup

The project is intended to run from the repository root inside the `mlomics` conda environment.

```bash
conda activate mlomics
python scripts/compute_auc.py
streamlit run app/streamlit_app.py
```

## Tech Stack

Python 3.9 · PyTorch 2.x · scikit-learn · XGBoost · SHAP · Captum · gseapy · Streamlit

## Author

**Dineth Hettiarachchi** — BSc Computer Science Final Year Project

---

*Academic research prototype. Not for clinical use.*
