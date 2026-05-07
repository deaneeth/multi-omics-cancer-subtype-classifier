"""
MLOmics Preprocessing Pipeline
===============================
Leakage-free imputation and normalization for multi-omics data.

CRITICAL RULES:
  - fit() on TRAIN fold ONLY
  - transform() on both train and val
  - Never fit on combined or future data

KNOWN LIMITATION — ANOVA pre-selection bias:
  The input CSVs contain only the top-k features selected by ANOVA F-test
  across ALL samples before CV splitting. This constitutes a minor form of
  global pre-selection leakage: the feature set itself is informed by labels
  from the full dataset, so a label-shuffle sanity test will show artificially
  elevated F1 (≈0.4–0.5 vs. expected ≈0.2 for 5-class random). This is a
  documented limitation of the MLOmics benchmark dataset format — per-fold
  feature selection is not feasible in the current pipeline. See
  docs/preprocessing_verification_report.md for full analysis.
"""

import json
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data_loader import MODALITY_KEYS, get_common_samples, load_labels, load_modality

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PerModalityImputer
# ---------------------------------------------------------------------------
class PerModalityImputer:
    """Median imputation per modality, fitted on training data only.

    BRCA miRNA has ~45% NaN, including features that are entirely NaN
    in small folds. This class computes per-feature medians from training
    data and fills NaN accordingly. Features with no observed values in
    the training fold are filled with 0.0 (no signal available).
    """

    def __init__(self):
        self.medians: Dict[str, pd.Series] = {}
        self.nan_counts: Dict[str, int] = {}

    def fit(self, data_dict: Dict[str, pd.DataFrame], modality_names: List[str]) -> "PerModalityImputer":
        """Compute per-feature medians on training data only."""
        for name in modality_names:
            df = data_dict[name]
            nan_count = int(df.isna().sum().sum())
            self.nan_counts[name] = nan_count

            if nan_count > 0:
                # Median per feature, ignoring NaN. All-NaN features → NaN median
                feature_medians = df.median(axis=0)
                # Replace NaN medians (all-NaN features) with 0.0
                all_nan_count = int(feature_medians.isna().sum())
                feature_medians = feature_medians.fillna(0.0)
                self.medians[name] = feature_medians
                logger.info(
                    f"Fitted imputer for {name}: {nan_count} NaN values "
                    f"({nan_count / df.size * 100:.1f}% of {df.shape}), "
                    f"{all_nan_count} all-NaN features filled with 0"
                )
            else:
                logger.info(f"No NaN in {name} — skipping imputer")
        return self

    def transform(self, data_dict: Dict[str, pd.DataFrame], modality_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Impute NaN using train-fitted medians. Asserts zero NaN after."""
        result = {}
        for name in modality_names:
            df = data_dict[name]
            if name in self.medians:
                filled = df.fillna(self.medians[name])
                remaining_nan = int(filled.isna().sum().sum())
                assert remaining_nan == 0, (
                    f"NaN remain in {name} after imputation: {remaining_nan}"
                )
                result[name] = filled
                logger.info(f"Imputed {name}: {int(df.isna().sum().sum())} NaN filled")
            else:
                result[name] = df.copy()
        return result

    def fit_transform(self, data_dict: Dict[str, pd.DataFrame], modality_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Fit on data then transform. USE ONLY ON TRAINING DATA."""
        self.fit(data_dict, modality_names)
        return self.transform(data_dict, modality_names)

    def save(self, filepath: str) -> None:
        """Save fitted imputers to disk."""
        joblib.dump({"medians": self.medians, "nan_counts": self.nan_counts}, filepath)
        logger.info(f"Imputer saved to {filepath}")

    def load(self, filepath: str) -> "PerModalityImputer":
        """Load fitted imputers from disk."""
        data = joblib.load(filepath)
        self.medians = data["medians"]
        self.nan_counts = data["nan_counts"]
        logger.info(f"Imputer loaded from {filepath}")
        return self


# ---------------------------------------------------------------------------
# PerModalityScaler
# ---------------------------------------------------------------------------
class PerModalityScaler:
    """Z-score normalization per modality, fitted on training data only.

    Each modality is scaled independently so large-feature-count modalities
    (mRNA: 5000) don't dominate small ones (miRNA: 200-366) after
    concatenation in early fusion.
    """

    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}

    def fit(self, data_dict: Dict[str, pd.DataFrame], modality_names: List[str]) -> "PerModalityScaler":
        """Fit StandardScalers on training data only."""
        for name in modality_names:
            scaler = StandardScaler()
            # Pass .values (numpy array) to avoid sklearn's feature-name
            # validation, which rejects DataFrames with non-string column names
            # (e.g. miRNA CSV rows that parse as float NaN).
            scaler.fit(data_dict[name].values)
            self.scalers[name] = scaler
            logger.info(f"Fitted scaler for {name}: {data_dict[name].shape}")
        return self

    def transform(self, data_dict: Dict[str, pd.DataFrame], modality_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Transform data using train-fitted scalers. Returns DataFrames."""
        result = {}
        for name in modality_names:
            df = data_dict[name]
            scaled = self.scalers[name].transform(df.values)
            result[name] = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        return result

    def fit_transform(self, data_dict: Dict[str, pd.DataFrame], modality_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Fit on data then transform. USE ONLY ON TRAINING DATA."""
        self.fit(data_dict, modality_names)
        return self.transform(data_dict, modality_names)

    def save(self, filepath: str) -> None:
        """Save fitted scalers to disk."""
        joblib.dump(self.scalers, filepath)
        logger.info(f"Scaler saved to {filepath}")

    def load(self, filepath: str) -> "PerModalityScaler":
        """Load fitted scalers from disk."""
        self.scalers = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")
        return self


# ---------------------------------------------------------------------------
# prepare_fold_data
# ---------------------------------------------------------------------------
def prepare_fold_data(
    cancer_type: str,
    fold_info: Dict[str, List[str]],
    config: dict,
    use_toy: bool = False,
) -> Dict[str, Any]:
    """Prepare data for a single CV fold with leakage prevention.

    Pipeline: load → split → assert no overlap → impute (train-fit) →
              assert zero NaN → scale (train-fit) → return

    Args:
        cancer_type: e.g. "GS-BRCA".
        fold_info: Dict with 'train' and 'val' keys, each a list of sample IDs.
        config: Project config dict.
        use_toy: If True, load from data/toy/.

    Returns:
        Dict with keys:
            X_train: {modality: pd.DataFrame}
            X_val:   {modality: pd.DataFrame}
            y_train: np.ndarray
            y_val:   np.ndarray
            scaler:  fitted PerModalityScaler
            imputer: fitted PerModalityImputer
    """
    train_ids = [str(s) for s in fold_info["train"]]
    val_ids = [str(s) for s in fold_info["val"]]

    # LEAKAGE CHECK: no overlap between train and val
    overlap = set(train_ids) & set(val_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {len(overlap)} shared IDs: {list(overlap)[:5]}"

    # 1. Load all modalities
    modality_data = {}
    for mod_key in MODALITY_KEYS:
        modality_data[mod_key] = load_modality(cancer_type, mod_key, config, use_toy=use_toy)

    # 2. Load labels
    labels = load_labels(cancer_type, config, use_toy=use_toy)

    # 3. Split into train/val per modality
    train_dict = {}
    val_dict = {}
    for mod_key in MODALITY_KEYS:
        df = modality_data[mod_key]
        # Ensure column names are all strings (miRNA has numeric-looking names
        # that sklearn StandardScaler rejects when mixed with string names).
        # Use .copy() so sklearn doesn't see a view with stale column metadata.
        df.columns = df.columns.astype(str)
        train_dict[mod_key] = df.loc[df.index.isin(train_ids)].copy()
        val_dict[mod_key] = df.loc[df.index.isin(val_ids)].copy()
        train_dict[mod_key].columns = train_dict[mod_key].columns.astype(str)
        val_dict[mod_key].columns = val_dict[mod_key].columns.astype(str)

    # 4. Impute NaN — fit on train ONLY, transform both
    imputer = PerModalityImputer()
    train_dict = imputer.fit_transform(train_dict, MODALITY_KEYS)
    val_dict = imputer.transform(val_dict, MODALITY_KEYS)

    # 5. Scale — fit on train ONLY, transform both
    scaler = PerModalityScaler()
    train_dict = scaler.fit_transform(train_dict, MODALITY_KEYS)
    val_dict = scaler.transform(val_dict, MODALITY_KEYS)

    # 6. Split labels
    y_train = labels.loc[labels.index.isin(train_ids)].values
    y_val = labels.loc[labels.index.isin(val_ids)].values

    logger.info(
        f"Fold prepared for {cancer_type}: "
        f"train={len(train_ids)}, val={len(val_ids)}"
    )

    return {
        "X_train": train_dict,
        "X_val": val_dict,
        "y_train": y_train,
        "y_val": y_val,
        "scaler": scaler,
        "imputer": imputer,
    }


# ---------------------------------------------------------------------------
# concatenate_modalities
# ---------------------------------------------------------------------------
def concatenate_modalities(
    data_dict: Dict[str, pd.DataFrame],
    modality_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Concatenate modality DataFrames into a single array for early fusion.

    Concatenates in a fixed order, skipping modalities not present in
    data_dict (future-proofs for Phase 7 ablation A1: modality removal).
    Feature names are prefixed with the modality key.

    Args:
        data_dict: {modality_key: DataFrame(samples x features)}.
        modality_order: Order to concatenate. Defaults to MODALITY_KEYS.
            Only modalities present in data_dict are included.

    Returns:
        (array, feature_names) where array has shape (n_samples, total_features)
        and feature_names is a list like ['mrna_BRCA1', 'mirna_hsa-miR-21', ...].
    """
    if modality_order is None:
        modality_order = MODALITY_KEYS

    frames = []
    feature_names = []
    for mod_key in modality_order:
        if mod_key not in data_dict:
            logger.info(f"Skipping {mod_key} — not in data_dict")
            continue
        df = data_dict[mod_key]
        frames.append(df.values)
        feature_names.extend([f"{mod_key}_{col}" for col in df.columns])

    if not frames:
        raise ValueError("No modalities found in data_dict")

    concatenated = np.concatenate(frames, axis=1)
    logger.info(
        f"Concatenated {len(frames)} modalities: "
        f"{concatenated.shape[0]} samples x {concatenated.shape[1]} features"
    )
    return concatenated, feature_names


# ---------------------------------------------------------------------------
# create_cv_folds
# ---------------------------------------------------------------------------
def create_cv_folds(
    cancer_type: str,
    config: dict,
    n_folds: int = 5,
    seed: int = 42,
    use_toy: bool = False,
) -> List[Dict[str, Any]]:
    """Create patient-level stratified k-fold CV and save to cv_folds.json.

    Verifies that each TCGA sample ID maps to a unique patient
    (no patient split across folds). Saves all cancer types into
    a single JSON file as the single source of truth for all splits.

    Args:
        cancer_type: e.g. "GS-BRCA".
        config: Project config dict.
        n_folds: Number of folds (default 5).
        seed: Random seed (default 42).
        use_toy: If True, load toy data and save to data/toy/cv_folds.json.

    Returns:
        List of fold dicts: [{"fold": 0, "train": [...], "val": [...]}, ...]
    """
    # 1. Load labels and common samples
    common_samples = get_common_samples(cancer_type, config, use_toy=use_toy)
    labels = load_labels(cancer_type, config, use_toy=use_toy)
    labels = labels.loc[labels.index.isin(common_samples)]

    sample_ids = list(labels.index)
    y = labels.values
    n_samples = len(sample_ids)

    # 2. Patient-level verification
    # TCGA IDs: "TCGA.XX.XXXX.01" -> patient = "TCGA.XX.XXXX"
    patient_ids = [".".join(s.split(".")[:-1]) for s in sample_ids]
    n_unique_patients = len(set(patient_ids))
    duplicates = [
        pid for pid, count in Counter(patient_ids).items() if count > 1
    ]

    if duplicates:
        logger.warning(
            f"{cancer_type}: {len(duplicates)} patients with multiple samples. "
            f"Would need StratifiedGroupKFold. Examples: {duplicates[:3]}"
        )
        raise ValueError(
            f"Multi-sample patients detected in {cancer_type}. "
            f"StratifiedGroupKFold not yet implemented. "
            f"Duplicates: {duplicates[:5]}"
        )

    logger.info(
        f"{cancer_type}: {n_samples} samples, {n_unique_patients} unique patients "
        f"(1:1 mapping confirmed)"
    )

    # 3. Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sample_ids, y)):
        train_samples = [sample_ids[i] for i in train_idx]
        val_samples = [sample_ids[i] for i in val_idx]

        # Overlap assertion
        overlap = set(train_samples) & set(val_samples)
        assert len(overlap) == 0, f"LEAK in fold {fold_idx}: {len(overlap)} shared"

        folds.append({
            "fold": fold_idx,
            "train": train_samples,
            "val": val_samples,
        })

    # 4. Compute class distribution
    class_dist = {str(k): int(v) for k, v in Counter(y).items()}

    # 5. Log per-fold summary
    for f in folds:
        train_labels = labels.loc[f["train"]].values
        val_labels = labels.loc[f["val"]].values
        train_dist = dict(Counter(train_labels))
        val_dist = dict(Counter(val_labels))
        logger.info(
            f"  Fold {f['fold']}: train={len(f['train'])}, val={len(f['val'])} "
            f"| train_dist={train_dist} | val_dist={val_dist}"
        )

    # 6. Save to JSON
    cv_dir = os.path.join("data", "toy") if use_toy else "data"
    cv_path = os.path.join(cv_dir, "cv_folds.json")

    # Load existing file to preserve other cancer types
    existing = {}
    if os.path.exists(cv_path):
        with open(cv_path, "r") as f:
            existing = json.load(f)

    existing[cancer_type] = {
        "n_folds": n_folds,
        "seed": seed,
        "n_samples": n_samples,
        "class_distribution": class_dist,
        "folds": folds,
    }

    with open(cv_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"CV folds saved to {cv_path}")

    return folds


# ---------------------------------------------------------------------------
# load_cv_folds
# ---------------------------------------------------------------------------
def load_cv_folds(
    cancer_type: str,
    config: dict,
    use_toy: bool = False,
) -> Dict[str, Any]:
    """Load saved CV folds for a specific cancer type.

    Args:
        cancer_type: e.g. "GS-BRCA".
        config: Project config dict (reserved for future path overrides).
        use_toy: If True, load from data/toy/cv_folds.json.

    Returns:
        Dict with keys: n_folds, seed, n_samples, class_distribution, folds.
    """
    cv_dir = os.path.join("data", "toy") if use_toy else "data"
    cv_path = os.path.join(cv_dir, "cv_folds.json")

    if not os.path.exists(cv_path):
        raise FileNotFoundError(
            f"CV folds file not found: {cv_path}. "
            f"Run create_cv_folds() first."
        )

    with open(cv_path, "r") as f:
        all_folds = json.load(f)

    if cancer_type not in all_folds:
        available = list(all_folds.keys())
        raise KeyError(
            f"Cancer type '{cancer_type}' not in {cv_path}. "
            f"Available: {available}"
        )

    logger.info(
        f"Loaded CV folds for {cancer_type} from {cv_path}: "
        f"{all_folds[cancer_type]['n_folds']} folds, "
        f"{all_folds[cancer_type]['n_samples']} samples"
    )
    return all_folds[cancer_type]
