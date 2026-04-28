"""Tests for utility helpers."""

import random

import numpy as np
import pandas as pd
import torch

from src.utils import (
    EXPERIMENT_LOG_COLUMNS,
    get_device,
    load_config,
    log_experiment,
    set_seeds,
)


def test_set_seeds_produces_reproducible_random_streams() -> None:
    set_seeds(42)
    first = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )

    set_seeds(42)
    second = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )

    assert first == second


def test_load_config_contains_core_sections() -> None:
    config = load_config()

    assert "project" in config
    assert "paths" in config
    assert "modalities" in config
    assert "fusion" in config


def test_log_experiment_creates_csv_with_expected_columns(tmp_path) -> None:
    out_path = tmp_path / "experiment_log.csv"

    log_experiment(
        config_path=str(out_path),
        experiment_name="unit_test_run",
        cancer_type="GS-BRCA",
        model_type="XGBoost",
        f1=0.8,
        notes="test",
    )

    frame = pd.read_csv(out_path)
    assert list(frame.columns) == EXPERIMENT_LOG_COLUMNS
    assert frame.loc[0, "experiment_name"] == "unit_test_run"


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert str(device) in {"cpu", "cuda"}
