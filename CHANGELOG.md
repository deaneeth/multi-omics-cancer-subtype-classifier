# Changelog

All notable changes to the MLOmics project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).
Tagged releases mark `dev → main` milestone merges.

## [Unreleased]
_Changes on `dev` not yet merged to `main`._

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