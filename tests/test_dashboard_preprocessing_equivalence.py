"""T1.7 — Dashboard preprocessing equivalence test.

Verifies that the inference preprocessing path (PerModalityImputer.transform +
PerModalityScaler.transform with fitted artifacts) produces the same output as
the training-time transform applied to the validation fold, ensuring the demo
app cannot silently diverge from the training pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    MODALITY_KEYS,
    PerModalityImputer,
    PerModalityScaler,
    load_cv_folds,
    prepare_fold_data,
)
from src.utils import load_config, set_seeds


@pytest.fixture(scope="module")
def fold_artifacts():
    """Return training-path preprocessed val data and the fitted artifacts."""
    set_seeds(42)
    config = load_config()
    fold_info = load_cv_folds("GS-BRCA", config, use_toy=True)["folds"][0]
    fold = prepare_fold_data("GS-BRCA", fold_info, config, use_toy=True)
    return fold


def test_inference_path_matches_training_path(fold_artifacts):
    """Applying saved artifacts to raw val data reproduces X_val from prepare_fold_data."""
    X_val_training = fold_artifacts["X_val"]   # produced by training path
    imputer = fold_artifacts["imputer"]
    scaler = fold_artifacts["scaler"]

    # Simulate raw val input: undo the imputer/scaler transforms by starting
    # from the scaled val data produced during training and verify that
    # re-applying the same fitted transform gives identical results.
    # To do this cleanly we use the pre-imputation state: reconstruct raw val
    # dict by reading the original toy data for the val sample IDs.
    from src.data_loader import load_modality
    config = load_config()

    # Read raw val DataFrames (same source as prepare_fold_data uses)
    # then run only transform (not fit_transform) — this is the inference path.
    raw_val_dict = {}
    for mod_key in MODALITY_KEYS:
        df = load_modality("GS-BRCA", mod_key, config, use_toy=True)
        df.columns = df.columns.astype(str)
        # Use same sample IDs that ended up in X_val
        val_sample_ids = list(X_val_training[mod_key].index)
        raw_val_dict[mod_key] = df.loc[df.index.isin(val_sample_ids)].copy()
        raw_val_dict[mod_key].columns = raw_val_dict[mod_key].columns.astype(str)

    # Inference path: impute then scale using fitted artifacts
    imputed = imputer.transform(raw_val_dict, MODALITY_KEYS)
    scaled = scaler.transform(imputed, MODALITY_KEYS)

    for mod in MODALITY_KEYS:
        expected = X_val_training[mod].values
        actual = scaled[mod].values
        assert actual.shape == expected.shape, (
            f"Shape mismatch for {mod}: got {actual.shape}, expected {expected.shape}"
        )
        assert np.allclose(actual, expected, atol=1e-6), (
            f"Preprocessing mismatch for {mod}: max diff = {np.abs(actual - expected).max():.2e}"
        )


def test_scaler_transform_is_deterministic(fold_artifacts):
    """Re-applying the fitted scaler twice gives identical results."""
    X_val = fold_artifacts["X_val"]
    scaler = fold_artifacts["scaler"]

    result1 = scaler.transform(X_val, MODALITY_KEYS)
    result2 = scaler.transform(X_val, MODALITY_KEYS)

    for mod in MODALITY_KEYS:
        assert np.allclose(result1[mod].values, result2[mod].values, atol=1e-9), (
            f"Scaler.transform is not deterministic for modality {mod}"
        )
