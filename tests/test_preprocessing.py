"""Tests for preprocessing utilities."""

import numpy as np
import pandas as pd

from src.preprocessing import (
    MODALITY_KEYS,
    PerModalityImputer,
    PerModalityScaler,
    concatenate_modalities,
    prepare_fold_data,
)


def test_per_modality_imputer_fills_nan_and_all_nan_features() -> None:
    train = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0],
            "f2": [np.nan, np.nan, np.nan],
        },
        index=["a", "b", "c"],
    )
    val = pd.DataFrame(
        {
            "f1": [np.nan, 2.0],
            "f2": [np.nan, np.nan],
        },
        index=["x", "y"],
    )
    train_dict = {name: train.copy() for name in MODALITY_KEYS}
    val_dict = {name: val.copy() for name in MODALITY_KEYS}

    imputer = PerModalityImputer().fit(train_dict, MODALITY_KEYS)
    out = imputer.transform(val_dict, MODALITY_KEYS)

    for modality in MODALITY_KEYS:
        assert int(out[modality].isna().sum().sum()) == 0
        assert (out[modality]["f2"] == 0.0).all()


def test_per_modality_scaler_centers_training_data() -> None:
    train = pd.DataFrame(
        np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float64),
        columns=["f1", "f2"],
        index=["s1", "s2", "s3"],
    )
    train_dict = {name: train.copy() for name in MODALITY_KEYS}

    scaler = PerModalityScaler()
    scaled = scaler.fit_transform(train_dict, MODALITY_KEYS)

    for modality in MODALITY_KEYS:
        means = scaled[modality].mean(axis=0).values
        assert np.allclose(means, np.zeros_like(means), atol=1e-7)


def test_prepare_fold_data_toy_has_no_nan(config: dict, toy_brca_fold: dict) -> None:
    prepared = prepare_fold_data("GS-BRCA", toy_brca_fold, config, use_toy=True)

    assert set(prepared.keys()) >= {"X_train", "X_val", "y_train", "y_val"}
    assert len(prepared["y_train"]) == len(toy_brca_fold["train"])
    assert len(prepared["y_val"]) == len(toy_brca_fold["val"])

    for modality in MODALITY_KEYS:
        assert prepared["X_train"][modality].shape[0] == len(toy_brca_fold["train"])
        assert prepared["X_val"][modality].shape[0] == len(toy_brca_fold["val"])
        assert int(prepared["X_train"][modality].isna().sum().sum()) == 0
        assert int(prepared["X_val"][modality].isna().sum().sum()) == 0


def test_scaler_fitted_on_train_only_val_not_centered() -> None:
    """Val data mean must not be zero — the scaler was fit on train, not val."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        rng.normal(loc=0, scale=1, size=(20, 3)).astype(np.float64),
        columns=["f1", "f2", "f3"],
        index=[f"train_{i}" for i in range(20)],
    )
    # Val drawn from a shifted distribution so its mean ≠ 0 after train-only scaling
    val = pd.DataFrame(
        rng.normal(loc=5, scale=1, size=(10, 3)).astype(np.float64),
        columns=["f1", "f2", "f3"],
        index=[f"val_{i}" for i in range(10)],
    )
    train_dict = {name: train.copy() for name in MODALITY_KEYS}
    val_dict = {name: val.copy() for name in MODALITY_KEYS}

    scaler = PerModalityScaler()
    scaler.fit(train_dict, MODALITY_KEYS)
    scaled_val = scaler.transform(val_dict, MODALITY_KEYS)

    for modality in MODALITY_KEYS:
        val_means = scaled_val[modality].mean(axis=0).values
        # If scaler were fit on val, means would be ~0; fit on train, they should be ~5
        assert not np.allclose(val_means, np.zeros_like(val_means), atol=1.0), (
            f"{modality}: val means near zero implies scaler was fit on val data"
        )


def test_concatenate_modalities_returns_prefixed_feature_names() -> None:
    x_dict = {
        "mrna": pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["g1", "g2"]),
        "mirna": pd.DataFrame([[5.0], [6.0]], columns=["m1"]),
    }

    concat, names = concatenate_modalities(x_dict, modality_order=["mrna", "mirna"])

    assert concat.shape == (2, 3)
    assert names == ["mrna_g1", "mrna_g2", "mirna_m1"]
