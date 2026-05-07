"""Pairwise statistical significance tests for model comparison.

With n=5 folds the minimum achievable two-sided Wilcoxon p-value is
0.0625. Tests are reported alongside Cohen's d on fold-wise differences
and 95% bootstrap confidence intervals to characterise practical
significance independent of formal significance thresholds.

Output: results/metrics/significance_tests.csv  (machine-readable)
        Printed markdown table                   (human-readable)

Usage:
    python scripts/compute_significance_tests.py
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ── Config ─────────────────────────────────────────────────────────────────
METRICS_DIR = "results/metrics"
OUTPUT_PATH = os.path.join(METRICS_DIR, "significance_tests.csv")
N_BOOTSTRAP = 10_000
RNG_SEED = 42

MODEL_FILE_MAP = {
    "XGBoost":            "xgb_{cancer}_metrics.csv",
    "RandomForest":       "rf_{cancer}_metrics.csv",
    "IntermediateFusion": "fusion_{cancer}_metrics.csv",
    "PathwayFusion":      "pathway_fusion_{cancer}_metrics.csv",
}
CANCERS = ["brca", "coad"]
CANCER_DISPLAY = {"brca": "GS-BRCA", "coad": "GS-COAD"}

# Headline pairs to highlight in the printed summary
HEADLINE_PAIRS = [
    ("IntermediateFusion", "XGBoost"),
    ("IntermediateFusion", "PathwayFusion"),
    ("IntermediateFusion", "RandomForest"),
]


def load_fold_f1(model: str, cancer: str) -> np.ndarray | None:
    fname = MODEL_FILE_MAP[model].format(cancer=cancer)
    fpath = os.path.join(METRICS_DIR, fname)
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    fold_rows = df[df["fold"].astype(str).str.isdigit()]
    return fold_rows["f1"].astype(float).values


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    pooled_std = np.std(diff, ddof=1)
    if pooled_std == 0:
        return 0.0
    return float(np.mean(diff) / pooled_std)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = a - b
    bs = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    mean_diff = float(np.mean(diff))
    ci_lo, ci_hi = bootstrap_ci(a, b, N_BOOTSTRAP, RNG_SEED)
    d = cohens_d(a, b)

    # Wilcoxon requires non-zero differences; if all diffs are 0, p=1
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        p_val = 1.0
    else:
        try:
            _, p_val = wilcoxon(a, b)
        except Exception:
            p_val = float("nan")

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "wilcoxon_p": round(p_val, 4),
        "cohens_d": round(d, 3),
        "n_folds": len(a),
    }


def main() -> None:
    os.makedirs(METRICS_DIR, exist_ok=True)
    rows = []

    for cancer in CANCERS:
        # Load all fold vectors
        fold_f1 = {}
        for model in MODEL_FILE_MAP:
            v = load_fold_f1(model, cancer)
            if v is not None and len(v) == 5:
                fold_f1[model] = v

        for model_a, model_b in itertools.combinations(sorted(fold_f1.keys()), 2):
            stats = run_test(fold_f1[model_a], fold_f1[model_b])
            rows.append({
                "cancer": CANCER_DISPLAY[cancer],
                "model_a": model_a,
                "model_b": model_b,
                **stats,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}\n")

    # ── Print markdown table ────────────────────────────────────────────────
    print("## Pairwise significance tests (Wilcoxon signed-rank, n=5 folds)\n")
    print(
        "| Cancer | Model A | Model B | Mean dF1 | 95% CI | p-value | Cohen's d |"
    )
    print("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        sig = "*" if r["wilcoxon_p"] <= 0.0625 else ""
        print(
            f"| {r['cancer']} | {r['model_a']} | {r['model_b']} "
            f"| {r['mean_diff']:+.4f} | {ci} "
            f"| {r['wilcoxon_p']:.4f}{sig} | {r['cohens_d']:.3f} |"
        )

    print(
        "\n* = p <= 0.0625 (minimum achievable two-sided p with n=5 folds)."
        "\nInterpret Cohen's d: |d|<0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large."
        "\nAll reported metrics are fold-wise F1 macro-average."
    )

    # ── Print headline comparisons ──────────────────────────────────────────
    print("\n### Headline model comparisons\n")
    for cancer in CANCERS:
        print(f"**{CANCER_DISPLAY[cancer]}**")
        for ma, mb in HEADLINE_PAIRS:
            row = df[(df["cancer"] == CANCER_DISPLAY[cancer]) &
                     (df["model_a"] == min(ma, mb)) &
                     (df["model_b"] == max(ma, mb))]
            if row.empty:
                row = df[(df["cancer"] == CANCER_DISPLAY[cancer]) &
                         (df["model_a"].isin([ma, mb])) &
                         (df["model_b"].isin([ma, mb]))]
            if row.empty:
                print(f"  {ma} vs {mb}: no data")
                continue
            r = row.iloc[0]
            direction = "A > B" if r["mean_diff"] > 0 else "A < B"
            print(
                f"  {r['model_a']} vs {r['model_b']}: "
                f"dF1={r['mean_diff']:+.4f}, "
                f"p={r['wilcoxon_p']:.4f}, d={r['cohens_d']:.3f} ({direction})"
            )
        print()


if __name__ == "__main__":
    main()
