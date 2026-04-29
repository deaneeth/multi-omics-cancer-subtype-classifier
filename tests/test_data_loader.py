"""Tests for data loading helpers."""

import pandas as pd

from src.data_loader import (
    MODALITY_KEYS,
    get_common_samples,
    load_labels,
    load_modality,
)


def test_load_modality_toy_returns_samples_by_features(config: dict) -> None:
    frame = load_modality("GS-BRCA", "mrna", config, use_toy=True)

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0
    assert frame.index.dtype == object


def test_load_labels_toy_aligns_to_mrna_index(config: dict) -> None:
    labels = load_labels("GS-BRCA", config, use_toy=True)
    mrna = load_modality("GS-BRCA", "mrna", config, use_toy=True)

    assert len(labels) == len(mrna)
    assert labels.index.equals(mrna.index)
    assert labels.nunique() >= 2


def test_common_samples_are_present_in_all_modalities(config: dict) -> None:
    common_samples = get_common_samples("GS-BRCA", config, use_toy=True)

    assert len(common_samples) > 0
    for modality in MODALITY_KEYS:
        frame = load_modality("GS-BRCA", modality, config, use_toy=True)
        assert set(common_samples).issubset(set(frame.index))
