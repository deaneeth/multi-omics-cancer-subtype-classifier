"""Tests for evaluation metrics."""

import numpy as np

from src.evaluation import METRIC_NAMES, compute_fold_summary, compute_metrics


def test_compute_metrics_returns_expected_keys_and_ranges() -> None:
    y_true = np.array([0, 1, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 0, 0, 2, 1])

    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == set(METRIC_NAMES)
    for value in metrics.values():
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0


def test_compute_fold_summary_generates_mean_and_std_for_each_metric() -> None:
    fold_metrics = [
        {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "nmi": 0.6,
            "ari": 0.5,
            "accuracy": 0.7,
        },
        {
            "precision": 0.6,
            "recall": 0.65,
            "f1": 0.62,
            "nmi": 0.55,
            "ari": 0.45,
            "accuracy": 0.65,
        },
    ]

    summary = compute_fold_summary(fold_metrics)

    for metric_name in METRIC_NAMES:
        assert f"{metric_name}_mean" in summary
        assert f"{metric_name}_std" in summary

    assert summary["f1_mean"] > 0
    assert summary["f1_std"] >= 0
