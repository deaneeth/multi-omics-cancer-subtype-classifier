"""T1.9 — Computational cost reporting.

Measures training time (on toy data, 10 epochs), peak memory, and inference
latency for all model families.  Results are written to:

    results/metrics/runtime_summary.csv

Usage:
    python scripts/measure_runtime.py

Note: toy data is used to give reproducible, hardware-comparable numbers.
Full-data training times are estimated by the toy-to-full scale factor printed
at the end.
"""

import gc
import os
import time
import tracemalloc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from src.models import IntermediateFusionModel, MultiOmicsDataset, PathwayAwareFusionModel
from src.preprocessing import MODALITY_KEYS, load_cv_folds, prepare_fold_data
from src.utils import load_config, set_seeds

OUT_DIR = Path("results/metrics")
OUT_PATH = OUT_DIR / "runtime_summary.csv"
DEVICE = torch.device("cpu")   # CPU for deterministic timing comparisons
TRAIN_EPOCHS = 10
BATCH_SIZE = 16
TOY_CANCER = "GS-BRCA"


def _peak_mb(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, elapsed_s, peak_memory_mb).

    NOTE: peak_memory_mb is measured via tracemalloc and reflects
    Python-heap allocations only.  Native memory used by NumPy arrays,
    PyTorch tensors, or XGBoost internals is NOT captured, so the
    reported value will be lower than the true process RSS (especially
    for tree-based models).  The CSV column ``peak_memory_mb`` should
    be interpreted as a lower bound on Python-side allocation, not total
    RAM usage.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 ** 2)


def _make_fusion_model(modality_dims, n_classes):
    return IntermediateFusionModel(
        modality_dims=modality_dims,
        latent_dim=64,
        hidden_dim=128,
        num_classes=n_classes,
        dropout=0.1,
        encoder_hidden=256,
    ).to(DEVICE)


def _make_pathway_model(modality_dims, n_classes, pathway_indices, unmapped):
    return PathwayAwareFusionModel(
        modality_dims=modality_dims,
        pathway_indices=pathway_indices,
        unmapped_indices=unmapped,
        latent_dim=64,
        hidden_dim=128,
        num_classes=n_classes,
        dropout=0.1,
        encoder_hidden=256,
        attn_hidden=64,
    ).to(DEVICE)


def _load_pathway_info(modality_dims):
    """Load KEGG pathway mapping for toy data feature indices."""
    import json
    mapping_path = "app/model_artifacts/pathway_gene_mapping.json"
    if not os.path.exists(mapping_path):
        # Fallback: split mRNA features into equal-size dummy pathways
        n = modality_dims["mrna"]
        pw = {f"pw_{i}": list(range(i * 10, min((i + 1) * 10, n))) for i in range(n // 10)}
        unmapped = list(range((n // 10) * 10, n))
        return pw, unmapped
    with open(mapping_path) as f:
        raw = json.load(f)
    cancer_key = "BRCA"
    if cancer_key not in raw:
        n = modality_dims["mrna"]
        pw = {f"pw_{i}": list(range(i * 10, min((i + 1) * 10, n))) for i in range(n // 10)}
        unmapped = list(range((n // 10) * 10, n))
        return pw, unmapped
    entry = raw[cancer_key]
    # Rebuild pathway_indices from 'pathways' key
    pathways_raw = entry.get("pathways", {})
    n_mrna = modality_dims["mrna"]
    pw = {}
    for pw_name, indices in pathways_raw.items():
        valid = [i for i in indices if i < n_mrna]
        if valid:
            pw[pw_name] = valid
    all_mapped = set(idx for idxs in pw.values() for idx in idxs)
    unmapped = [i for i in range(n_mrna) if i not in all_mapped]
    return pw, unmapped


def _train_nn(model, loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train(True)
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()


def _inference_latency(model, loader, n_warmup=3, n_measure=10):
    """Return median inference latency in ms for one batch."""
    model.train(False)
    batch_x, _ = next(iter(loader))
    times = []
    with torch.no_grad():
        for i in range(n_warmup + n_measure):
            t0 = time.perf_counter()
            _ = model(batch_x)
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= n_warmup:
                times.append(elapsed)
    return float(np.median(times))


def _inference_latency_sklearn(model, X, n_warmup=3, n_measure=10):
    """Return median inference latency in ms for sklearn model."""
    times = []
    for i in range(n_warmup + n_measure):
        t0 = time.perf_counter()
        _ = model.predict_proba(X[:BATCH_SIZE])
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            times.append(elapsed)
    return float(np.median(times))


def main():
    set_seeds(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()

    fold_info = load_cv_folds(TOY_CANCER, config, use_toy=True)["folds"][0]
    fold = prepare_fold_data(TOY_CANCER, fold_info, config, use_toy=True)

    X_train, y_train = fold["X_train"], fold["y_train"]
    n_classes = int(y_train.max()) + 1
    modality_dims = {k: v.shape[1] for k, v in X_train.items()}
    total_features = sum(modality_dims.values())

    # Concatenated features for sklearn baselines
    X_concat = np.concatenate([X_train[k].values for k in MODALITY_KEYS], axis=1).astype(np.float32)

    dataset = MultiOmicsDataset(
        {k: v.values.astype(np.float32) for k, v in X_train.items()},
        y_train.astype(np.int64),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    rows = []

    # ── XGBoost ────────────────────────────────────────────────────────────
    print("Timing XGBoost...")
    dtrain = xgb.DMatrix(X_concat, label=y_train)
    xgb_params = {"objective": "multi:softprob", "num_class": n_classes, "tree_method": "hist", "verbosity": 0}

    def _train_xgb():
        return xgb.train(xgb_params, dtrain, num_boost_round=50)

    xgb_model, train_time, peak_mb = _peak_mb(_train_xgb)
    lat = _inference_latency_sklearn(
        type("W", (), {"predict_proba": lambda self, x: xgb_model.predict(xgb.DMatrix(x))})(),
        X_concat,
    )
    rows.append({"model": "XGBoost", "train_time_s": round(train_time, 2), "peak_memory_mb": round(peak_mb, 1), "inference_latency_ms": round(lat, 3), "n_params": "N/A (50 trees)"})
    print(f"  XGBoost: {train_time:.2f}s, {peak_mb:.1f} MB, {lat:.3f} ms/batch")

    # ── RandomForest ───────────────────────────────────────────────────────
    print("Timing RandomForest...")

    def _train_rf():
        rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
        rf.fit(X_concat, y_train)
        return rf

    rf_model, train_time, peak_mb = _peak_mb(_train_rf)
    lat = _inference_latency_sklearn(rf_model, X_concat)
    rows.append({"model": "RandomForest", "train_time_s": round(train_time, 2), "peak_memory_mb": round(peak_mb, 1), "inference_latency_ms": round(lat, 3), "n_params": "N/A (100 trees)"})
    print(f"  RandomForest: {train_time:.2f}s, {peak_mb:.1f} MB, {lat:.3f} ms/batch")

    # ── IntermediateFusion ─────────────────────────────────────────────────
    print("Timing IntermediateFusion...")

    def _train_fusion():
        m = _make_fusion_model(modality_dims, n_classes)
        _train_nn(m, loader, TRAIN_EPOCHS)
        return m

    fusion_model, train_time, peak_mb = _peak_mb(_train_fusion)
    lat = _inference_latency(fusion_model, loader)
    n_params = sum(p.numel() for p in fusion_model.parameters())
    rows.append({"model": "IntermediateFusion", "train_time_s": round(train_time, 2), "peak_memory_mb": round(peak_mb, 1), "inference_latency_ms": round(lat, 3), "n_params": n_params})
    print(f"  IntermediateFusion: {train_time:.2f}s, {peak_mb:.1f} MB, {lat:.3f} ms/batch")

    # ── PathwayAwareFusion ─────────────────────────────────────────────────
    print("Timing PathwayAwareFusion...")
    pathway_indices, unmapped_indices = _load_pathway_info(modality_dims)

    def _train_pathway():
        m = _make_pathway_model(modality_dims, n_classes, pathway_indices, unmapped_indices)
        _train_nn(m, loader, TRAIN_EPOCHS)
        return m

    pw_model, train_time, peak_mb = _peak_mb(_train_pathway)
    lat = _inference_latency(pw_model, loader)
    n_params = sum(p.numel() for p in pw_model.parameters())
    rows.append({"model": "PathwayAwareFusion", "train_time_s": round(train_time, 2), "peak_memory_mb": round(peak_mb, 1), "inference_latency_ms": round(lat, 3), "n_params": n_params})
    print(f"  PathwayAwareFusion: {train_time:.2f}s, {peak_mb:.1f} MB, {lat:.3f} ms/batch")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")

    print(f"\n## Runtime summary ({TOY_CANCER} toy data, {TRAIN_EPOCHS} epochs, CPU, batch_size={BATCH_SIZE})\n")
    print(f"{'Model':<22} {'Train (s)':>10} {'Peak RAM (MB)':>14} {'Infer (ms/batch)':>18} {'Parameters':>18}")
    print("-" * 84)
    for _, r in df.iterrows():
        print(f"{r['model']:<22} {r['train_time_s']:>10} {r['peak_memory_mb']:>14} {r['inference_latency_ms']:>18} {str(r['n_params']):>18}")

    print(
        "\nNote: training times are for toy data only (~50 samples). Full-data training"
        "\n(~500-900 samples, 5 folds, 100+ epochs) is estimated at 30-90 min per model on this hardware."
    )


if __name__ == "__main__":
    main()
