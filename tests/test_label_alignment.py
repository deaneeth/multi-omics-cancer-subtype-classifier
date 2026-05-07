"""Tests for label-to-sample positional alignment.

These tests guard against silent label scrambling if source files are
ever regenerated. The SHA-256 check ensures the label file has not changed
since the checksums were recorded by scripts/verify_label_alignment.py.
"""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_labels, load_modality


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_toy_label_count_matches_mrna_sample_count(config: dict) -> None:
    """Label row count must equal mRNA sample count for BRCA toy set."""
    labels = load_labels("GS-BRCA", config, use_toy=True)
    mrna = load_modality("GS-BRCA", "mrna", config, use_toy=True)
    assert len(labels) == len(mrna), (
        f"Label count {len(labels)} != mRNA sample count {len(mrna)}. "
        "Positional alignment is broken."
    )


def test_toy_label_index_equals_mrna_index(config: dict) -> None:
    """Label sample IDs must be identical to (and in the same order as) mRNA IDs."""
    labels = load_labels("GS-BRCA", config, use_toy=True)
    mrna = load_modality("GS-BRCA", "mrna", config, use_toy=True)
    assert labels.index.tolist() == mrna.index.tolist(), (
        "Label sample-ID order does not match mRNA column order. "
        "Positional alignment assumption is violated."
    )


def test_toy_label_class_distribution_matches_recorded(config: dict) -> None:
    """Class distribution of toy BRCA labels must match the committed record."""
    expected = {0: 26, 1: 3, 2: 10, 3: 2, 4: 9}
    labels = load_labels("GS-BRCA", config, use_toy=True)
    observed = labels.value_counts().sort_index().to_dict()
    assert observed == expected, (
        f"BRCA toy label distribution changed.\n"
        f"Expected: {expected}\nObserved: {observed}\n"
        "If this is intentional, update the expected dict and re-run "
        "scripts/verify_label_alignment.py to refresh checksums."
    )


def test_label_file_checksum_matches_recorded() -> None:
    """SHA-256 of the toy label file must match the value in label_file_checksums.json.

    If this fails, the label file has been modified since checksums were recorded.
    Re-run scripts/verify_label_alignment.py --toy to refresh.
    """
    checksum_path = "data/label_file_checksums.json"
    if not os.path.exists(checksum_path):
        pytest.skip("data/label_file_checksums.json not found; run verify_label_alignment.py first")

    with open(checksum_path) as f:
        checksums = json.load(f)

    key = "GS-BRCA_toy"
    if key not in checksums:
        pytest.skip(f"No checksum record for {key}; run scripts/verify_label_alignment.py --toy")

    record = checksums[key]
    label_path = Path(record["path"].replace("\\", "/"))  # normalize Windows separators before Path()
    if not label_path.exists():
        pytest.skip(f"Label file not found at recorded path {label_path}")

    actual_sha = _sha256(str(label_path))
    assert actual_sha == record["sha256"], (
        f"Label file checksum mismatch for {label_path}!\n"
        f"Recorded: {record['sha256']}\n"
        f"Actual:   {actual_sha}\n"
        "The label file has been modified. Re-audit label alignment before proceeding."
    )
