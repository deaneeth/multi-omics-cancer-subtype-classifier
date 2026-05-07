"""T1.8 — Model calibration analysis.

Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Brier Score for all models on both cancer cohorts using the per-fold
prediction archives.  Saves:

    results/calibration/calibration_summary.csv  — per-fold ECE/MCE/Brier
    results/calibration/reliability_<model>_<cancer>.png — reliability diagrams

Usage:
    python scripts/calibration_analysis.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

# ── Config ─────────────────────────────────────────────────────────────────
METRICS_DIR = Path("results/metrics")
CALIB_DIR = Path("results/calibration")
N_BINS = 10

MODEL_PREFIX_MAP = {
    "XGBoost": "xgb",
    "RandomForest": "rf",
    "IntermediateFusion": "fusion",
    "PathwayFusion": "pathway_fusion",
}
CANCERS = {"brca": "GS-BRCA", "coad": "GS-COAD"}
N_CLASSES = {"brca": 5, "coad": 4}


# ── ECE / MCE / Brier helpers ──────────────────────────────────────────────

def ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> tuple[float, float]:
    """Compute ECE and MCE using equal-width confidence bins.

    Uses the maximum-class-probability (confidence) approach standard in
    multi-class calibration literature.

    Args:
        y_true: Integer class labels, shape (N,).
        y_prob: Softmax probabilities, shape (N, C).
        n_bins: Number of equal-width bins in [0, 1].

    Returns:
        (ECE, MCE) — both in [0, 1].
    """
    confidences = y_prob.max(axis=1)        # max prob per sample
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_acc = 0.0
    mce = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        gap = abs(acc - conf)
        ece_acc += (mask.sum() / n) * gap
        if gap > mce:
            mce = gap

    return float(ece_acc), float(mce)


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """Mean Brier score across all classes (one-vs-rest average)."""
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return float(np.mean((y_prob - y_bin) ** 2))


# ── Per-fold loader ─────────────────────────────────────────────────────────

def load_fold_predictions(prefix: str, cancer_short: str, fold: int):
    path = METRICS_DIR / f"{prefix}_{cancer_short}_fold{fold}_predictions.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["y_true"], d["y_prob"]


# ── Reliability diagram ─────────────────────────────────────────────────────

def plot_reliability_diagram(
    model_name: str,
    cancer_name: str,
    fold_data: list[tuple[np.ndarray, np.ndarray]],
    mean_ece: float,
    out_path: Path,
) -> None:
    """Plot a reliability diagram averaged across folds."""
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    all_accs = []
    for y_true, y_prob in fold_data:
        confidences = y_prob.max(axis=1)
        correct = (y_prob.argmax(axis=1) == y_true).astype(float)
        accs = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            accs.append(correct[mask].mean() if mask.sum() > 0 else np.nan)
        all_accs.append(accs)

    mean_accs = np.nanmean(all_accs, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1)
    ax.bar(
        bin_centers,
        mean_accs,
        width=1.0 / N_BINS,
        alpha=0.7,
        color="steelblue",
        label=f"Model (ECE={mean_ece:.3f})",
        align="center",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (max class probability)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram\n{model_name} — {cancer_name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, prefix in MODEL_PREFIX_MAP.items():
        for cancer_short, cancer_display in CANCERS.items():
            n_cls = N_CLASSES[cancer_short]
            fold_data = []

            for fold in range(5):
                result = load_fold_predictions(prefix, cancer_short, fold)
                if result is None:
                    continue
                y_true, y_prob = result

                # Ensure y_prob has the right shape
                if y_prob.ndim == 1 or y_prob.shape[1] != n_cls:
                    print(f"  Skipping {model_name}/{cancer_short}/fold{fold}: unexpected y_prob shape {y_prob.shape}")
                    continue

                ece, mce = ece_mce(y_true, y_prob)
                brier = brier_score_multiclass(y_true, y_prob, n_cls)

                rows.append({
                    "model": model_name,
                    "cancer": cancer_display,
                    "fold": fold,
                    "ece": round(ece, 4),
                    "mce": round(mce, 4),
                    "brier": round(brier, 4),
                    "n_samples": len(y_true),
                })
                fold_data.append((y_true, y_prob))

            if len(fold_data) == 0:
                continue

            mean_ece = np.mean([r["ece"] for r in rows if r["model"] == model_name and r["cancer"] == cancer_display])
            plot_path = CALIB_DIR / f"reliability_{prefix}_{cancer_short}.png"
            plot_reliability_diagram(model_name, cancer_display, fold_data, mean_ece, plot_path)
            print(f"  {model_name:25s} {cancer_display}: mean ECE={mean_ece:.4f}  -> {plot_path.name}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No prediction files found. Run training scripts first.")
        return

    csv_path = CALIB_DIR / "calibration_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved calibration summary to {csv_path}")

    # Print a clean mean±std table
    print("\n## Calibration summary (mean +/- std across 5 folds)\n")
    print(f"{'Model':<25} {'Cancer':<10} {'ECE':>8} {'MCE':>8} {'Brier':>8}")
    print("-" * 65)
    for (model, cancer), g in df.groupby(["model", "cancer"]):
        ece_str = f"{g['ece'].mean():.3f}+-{g['ece'].std():.3f}"
        mce_str = f"{g['mce'].mean():.3f}+-{g['mce'].std():.3f}"
        brier_str = f"{g['brier'].mean():.3f}+-{g['brier'].std():.3f}"
        print(f"{model:<25} {cancer:<10} {ece_str:>8} {mce_str:>8} {brier_str:>8}")


if __name__ == "__main__":
    main()
