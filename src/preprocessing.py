"""
MLOmics Preprocessing Pipeline
===============================
Leakage-free imputation and normalization for multi-omics data.

CRITICAL RULES:
  - fit() on TRAIN fold ONLY
  - transform() on both train and val
  - Never fit on combined or future data
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
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
            scaler.fit(data_dict[name])
            self.scalers[name] = scaler
            logger.info(f"Fitted scaler for {name}: {data_dict[name].shape}")
        return self

    def transform(self, data_dict: Dict[str, pd.DataFrame], modality_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Transform data using train-fitted scalers. Returns DataFrames."""
        result = {}
        for name in modality_names:
            df = data_dict[name]
            scaled = self.scalers[name].transform(df)
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
        # Ensure column names are all strings (miRNA has mixed float/str names)
        df.columns = df.columns.astype(str)
        train_dict[mod_key] = df.loc[df.index.isin(train_ids)]
        val_dict[mod_key] = df.loc[df.index.isin(val_ids)]

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
