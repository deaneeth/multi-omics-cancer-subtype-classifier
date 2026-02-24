# MLOmics: Latent-Fusion Multi-Omics Classifier for Cancer Subtype Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **⚠️ Academic research prototype. Not for clinical use.**

## Overview

End-to-end reproducible multi-omics machine learning pipeline for cancer subtype classification using the [MLOmics benchmark dataset](https://figshare.com/articles/dataset/MLOmics_Cancer_Multi-Omics_Database_for_Machine_Learning/28729127).

**Cancer Types:** GS-BRCA (breast, 5 subtypes) · GS-COAD (colon, 4 subtypes)

**Modalities:** mRNA (5000) · miRNA (200) · DNA Methylation (5000) · CNV (5000)

## Models

| Model | Approach |
|---|---|
| XGBoost | Early fusion baseline |
| Random Forest | Early fusion baseline |
| Intermediate Fusion | Per-modality encoders → latent concat → MLP classifier |

## Project Structure

```
├── data/           # Raw & processed data (gitignored)
├── src/            # Source modules (data_loader, preprocessing, models, evaluation, explainability, utils)
├── notebooks/      # Exploration & visualization notebooks
├── models/         # Saved model artifacts per fold
├── results/        # Metrics, plots, SHAP values, enrichment outputs
├── app/            # Streamlit demo application
├── scripts/        # Training & evaluation CLI scripts
├── tests/          # Test suite
├── docs/           # Final report, slides, demo video
├── config.yaml     # All hyperparameters, paths, seeds
└── experiment_log.csv  # Experiment tracking
```

## Setup

Setup instructions coming soon.

## Tech Stack

Python 3.9 · PyTorch ≥2.0 · scikit-learn ≥1.2 · XGBoost ≥1.7 · SHAP ≥0.42 · Captum ≥0.6 · gseapy ≥1.0 · Streamlit ≥1.28

## Author

**Dineth** — BSc Computer Science Final Year Project

---

*Academic research prototype. Not for clinical use.*
