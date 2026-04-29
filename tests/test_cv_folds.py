"""Tests for CV fold integrity — no cross-fold leakage."""

import json
import os

import pytest

from src.preprocessing import load_cv_folds


def _load_raw_folds() -> dict:
    """Load the canonical cv_folds.json directly."""
    path = os.path.join("data", "cv_folds.json")
    with open(path) as f:
        return json.load(f)


def test_no_overlap_between_val_folds_brca() -> None:
    """No sample ID should appear in more than one validation fold for BRCA."""
    folds_data = _load_raw_folds()
    brca_folds = folds_data["GS-BRCA"]["folds"]
    seen: set = set()
    for fold in brca_folds:
        val_ids = set(str(s) for s in fold["val"])
        overlap = seen & val_ids
        assert not overlap, (
            f"Sample(s) {overlap} appear in multiple GS-BRCA validation folds"
        )
        seen.update(val_ids)


def test_no_overlap_between_val_folds_coad() -> None:
    """No sample ID should appear in more than one validation fold for COAD."""
    folds_data = _load_raw_folds()
    coad_folds = folds_data["GS-COAD"]["folds"]
    seen: set = set()
    for fold in coad_folds:
        val_ids = set(str(s) for s in fold["val"])
        overlap = seen & val_ids
        assert not overlap, (
            f"Sample(s) {overlap} appear in multiple GS-COAD validation folds"
        )
        seen.update(val_ids)


def test_no_train_val_overlap_within_fold_brca(config: dict) -> None:
    """Train and val sets must be disjoint within every BRCA fold."""
    cv_data = load_cv_folds("GS-BRCA", config, use_toy=True)
    for i, fold in enumerate(cv_data["folds"]):
        train_ids = set(str(s) for s in fold["train"])
        val_ids = set(str(s) for s in fold["val"])
        overlap = train_ids & val_ids
        assert not overlap, (
            f"GS-BRCA fold {i}: {len(overlap)} sample(s) in both train and val"
        )


def test_all_samples_covered_once_brca() -> None:
    """Every BRCA sample appears exactly once across all val folds."""
    folds_data = _load_raw_folds()
    brca_folds = folds_data["GS-BRCA"]["folds"]
    all_val: list = []
    for fold in brca_folds:
        all_val.extend(str(s) for s in fold["val"])
    assert len(all_val) == len(set(all_val)), "Some BRCA samples appear in multiple val folds"
