"""
MLOmics Phase 7: Ablation Studies
===================================
Runs three ablation experiments to support thesis claims:

  A1: Modality Removal   — Remove each modality, measure F1 degradation
  A2: Fusion Comparison   — Early fusion MLP vs Intermediate fusion (+ XGBoost)
  A3: Missing Modality    — Zero-pad modalities at varying rates

Usage:
    python scripts/run_ablations.py              # All ablations, full data
    python scripts/run_ablations.py --only a1    # Run only A1
    python scripts/run_ablations.py --only a2    # Run only A2
    python scripts/run_ablations.py --only a3    # Run only A3

Artifacts:
    results/metrics/ablation_modality_removal.csv
    results/metrics/ablation_fusion_comparison.csv
    results/metrics/ablation_missing_modality.csv
    results/plots/ablation_modality_removal.{png,pdf}
    results/plots/ablation_fusion_comparison.{png,pdf}
    results/plots/missing_modality_curve.{png,pdf}
"""

import argparse
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import compute_fold_summary, compute_metrics
from src.models import EarlyFusionMLP, IntermediateFusionModel, MultiOmicsDataset
from src.preprocessing import concatenate_modalities, load_cv_folds, prepare_fold_data
from src.utils import get_device, load_config, log_experiment, set_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALL_MODALITIES = ["mrna", "mirna", "methy", "cnv"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def to_device(x, device):
    """Move batch input to device (handles dict or tensor)."""
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def compute_class_weights(y_train, device):
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts.astype(np.float64)
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def dataframes_to_numpy(data_dict):
    return {name: df.values.astype(np.float32) for name, df in data_dict.items()}


def train_one_epoch(model, loader, criterion, optimizer, device, max_norm=1.0):
    model.train()
    total_loss, n = 0.0, 0
    for x_batch, y_batch in loader:
        x_batch = to_device(x_batch, device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for x_batch, y_batch in loader:
        x_batch = to_device(x_batch, device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item()
        n += 1
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    return (
        total_loss / max(n, 1),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )


def train_single_fold(model, train_loader, val_loader, y_train, config, device):
    """Train one fold with early stopping. Returns final metrics dict."""
    fusion_cfg = config["fusion"]
    max_epochs = fusion_cfg["epochs"]
    patience = fusion_cfg["patience"]
    lr = fusion_cfg["learning_rate"]
    wd = fusion_cfg["weight_decay"]

    class_weights = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break
        if np.isnan(train_loss) or np.isinf(train_loss):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    _, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    return compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# A1: Modality Removal
# ---------------------------------------------------------------------------

def run_modality_removal(cancer_type, config, device):
    """Remove each modality one at a time and measure F1 degradation."""
    cancer_short = cancer_type.split("-")[1]
    fusion_cfg = config["fusion"]
    cv_data = load_cv_folds(cancer_type, config)
    folds = cv_data["folds"]
    n_folds = cv_data["n_folds"]

    rows = []

    for remove_mod in ALL_MODALITIES:
        keep_mods = [m for m in ALL_MODALITIES if m != remove_mod]
        logger.info(f"A1 {cancer_short}: removing {remove_mod}, keeping {keep_mods}")

        fold_metrics = []
        for fold_info in folds:
            fold_idx = fold_info["fold"]
            set_seeds(42)

            result = prepare_fold_data(cancer_type, fold_info, config)
            X_train_all = dataframes_to_numpy(result["X_train"])
            X_val_all = dataframes_to_numpy(result["X_val"])
            y_train = result["y_train"]
            y_val = result["y_val"]

            # Remove the target modality
            X_train = {k: v for k, v in X_train_all.items() if k != remove_mod}
            X_val = {k: v for k, v in X_val_all.items() if k != remove_mod}

            modality_dims = {name: arr.shape[1] for name, arr in X_train.items()}
            n_classes = len(np.unique(np.concatenate([y_train, y_val])))

            train_ds = MultiOmicsDataset(X_train, y_train)
            val_ds = MultiOmicsDataset(X_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=fusion_cfg["batch_size"],
                                      shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=fusion_cfg["batch_size"],
                                    shuffle=False, num_workers=0)

            model = IntermediateFusionModel(
                modality_dims=modality_dims,
                latent_dim=fusion_cfg["default_latent_dim"],
                hidden_dim=fusion_cfg["classifier_hidden"],
                num_classes=n_classes,
                dropout=fusion_cfg["dropout"],
                encoder_hidden=fusion_cfg["encoder_hidden"],
            ).to(device)

            metrics = train_single_fold(
                model, train_loader, val_loader, y_train, config, device,
            )
            fold_metrics.append(metrics)
            logger.info(
                f"  Fold {fold_idx}: F1={metrics['f1']:.4f} (without {remove_mod})"
            )

            log_experiment(
                config_path=config["paths"]["experiment_log"],
                experiment_name=f"ablation_remove_{remove_mod}",
                cancer_type=cancer_type,
                model_type="IntermediateFusion_ablation",
                features=f"without_{remove_mod}",
                n_folds=n_folds, fold=fold_idx,
                precision=round(metrics["precision"], 4),
                recall=round(metrics["recall"], 4),
                f1=round(metrics["f1"], 4),
                nmi=round(metrics["nmi"], 4),
                ari=round(metrics["ari"], 4),
                notes=f"A1: removed {remove_mod}",
                artifact_path="",
            )

        summary = compute_fold_summary(fold_metrics)
        rows.append({
            "cancer": cancer_short,
            "removed_modality": remove_mod,
            "f1_mean": round(summary["f1_mean"], 4),
            "f1_std": round(summary["f1_std"], 4),
            "precision_mean": round(summary["precision_mean"], 4),
            "recall_mean": round(summary["recall_mean"], 4),
        })
        logger.info(
            f"  {cancer_short} without {remove_mod}: "
            f"F1={summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}"
        )

    return rows


# ---------------------------------------------------------------------------
# A2: Early vs Intermediate Fusion Comparison
# ---------------------------------------------------------------------------

def run_fusion_comparison(cancer_type, config, device):
    """Train EarlyFusionMLP and compile 3-way comparison table."""
    cancer_short = cancer_type.split("-")[1].lower()
    fusion_cfg = config["fusion"]
    cv_data = load_cv_folds(cancer_type, config)
    folds = cv_data["folds"]
    n_folds = cv_data["n_folds"]

    fold_metrics_early = []

    for fold_info in folds:
        fold_idx = fold_info["fold"]
        set_seeds(42)

        result = prepare_fold_data(cancer_type, fold_info, config)
        X_train_np = dataframes_to_numpy(result["X_train"])
        X_val_np = dataframes_to_numpy(result["X_val"])
        y_train = result["y_train"]
        y_val = result["y_val"]

        # Concatenate all modalities — compute total_features DYNAMICALLY
        X_train_concat, _ = concatenate_modalities(result["X_train"])
        X_val_concat, _ = concatenate_modalities(result["X_val"])
        total_features = X_train_concat.shape[1]
        n_classes = len(np.unique(np.concatenate([y_train, y_val])))

        logger.info(
            f"A2 {cancer_short} fold {fold_idx}: "
            f"total_features={total_features}, n_classes={n_classes}"
        )

        train_ds = TensorDataset(
            torch.tensor(X_train_concat, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_concat, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=fusion_cfg["batch_size"],
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=fusion_cfg["batch_size"],
                                shuffle=False, num_workers=0)

        model = EarlyFusionMLP(
            input_dim=total_features,
            num_classes=n_classes,
            dropout=fusion_cfg["dropout"],
        ).to(device)

        metrics = train_single_fold(
            model, train_loader, val_loader, y_train, config, device,
        )
        fold_metrics_early.append(metrics)
        logger.info(f"  Fold {fold_idx}: F1={metrics['f1']:.4f} (early fusion)")

        log_experiment(
            config_path=config["paths"]["experiment_log"],
            experiment_name="ablation_early_fusion",
            cancer_type=cancer_type,
            model_type="EarlyFusionMLP",
            features=f"early_concat_{total_features}",
            n_folds=n_folds, fold=fold_idx,
            precision=round(metrics["precision"], 4),
            recall=round(metrics["recall"], 4),
            f1=round(metrics["f1"], 4),
            nmi=round(metrics["nmi"], 4),
            ari=round(metrics["ari"], 4),
            notes=f"A2: EarlyFusionMLP, total_features={total_features}",
            artifact_path="",
        )

    early_summary = compute_fold_summary(fold_metrics_early)

    # Load existing XGBoost and IntermediateFusion results
    metrics_dir = os.path.join(config["paths"]["results"], "metrics")
    xgb_path = os.path.join(metrics_dir, f"xgb_{cancer_short}_metrics.csv")
    fusion_path = os.path.join(metrics_dir, f"fusion_{cancer_short}_metrics.csv")

    rows = []

    # Row 1: XGBoost (early fusion, tree-based)
    if os.path.exists(xgb_path):
        xgb_df = pd.read_csv(xgb_path)
        # Filter out summary rows (fold column contains 'mean' or non-integer)
        xgb_df = xgb_df[pd.to_numeric(xgb_df["fold"], errors="coerce").notna()]
        for col in ["f1", "precision", "recall", "nmi", "ari"]:
            xgb_df[col] = pd.to_numeric(xgb_df[col], errors="coerce")
        rows.append({
            "cancer": cancer_short.upper(),
            "model": "XGBoost (early concat, tree)",
            "fusion_type": "early",
            "f1_mean": round(xgb_df["f1"].mean(), 4),
            "f1_std": round(xgb_df["f1"].std(), 4),
            "precision_mean": round(xgb_df["precision"].mean(), 4),
            "recall_mean": round(xgb_df["recall"].mean(), 4),
        })

    # Row 2: EarlyFusionMLP (early fusion, deep)
    rows.append({
        "cancer": cancer_short.upper(),
        "model": "EarlyFusionMLP (early concat, deep)",
        "fusion_type": "early",
        "f1_mean": round(early_summary["f1_mean"], 4),
        "f1_std": round(early_summary["f1_std"], 4),
        "precision_mean": round(early_summary["precision_mean"], 4),
        "recall_mean": round(early_summary["recall_mean"], 4),
    })

    # Row 3: IntermediateFusion (per-modality encoders)
    if os.path.exists(fusion_path):
        fusion_df = pd.read_csv(fusion_path)
        fusion_df = fusion_df[pd.to_numeric(fusion_df["fold"], errors="coerce").notna()]
        for col in ["f1", "precision", "recall", "nmi", "ari"]:
            fusion_df[col] = pd.to_numeric(fusion_df[col], errors="coerce")
        rows.append({
            "cancer": cancer_short.upper(),
            "model": "IntermediateFusion (per-encoder, deep)",
            "fusion_type": "intermediate",
            "f1_mean": round(fusion_df["f1"].mean(), 4),
            "f1_std": round(fusion_df["f1"].std(), 4),
            "precision_mean": round(fusion_df["precision"].mean(), 4),
            "recall_mean": round(fusion_df["recall"].mean(), 4),
        })

    return rows


# ---------------------------------------------------------------------------
# A3: Missing Modality Simulation
# ---------------------------------------------------------------------------

def run_missing_modality(cancer_type, config, device):
    """Simulate missing modalities by zero-padding at various rates."""
    cancer_short = cancer_type.split("-")[1]
    fusion_cfg = config["fusion"]
    cv_data = load_cv_folds(cancer_type, config)
    folds = cv_data["folds"]
    n_folds = cv_data["n_folds"]
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.5]

    rows = []

    for rate in missing_rates:
        logger.info(f"A3 {cancer_short}: missing_rate={rate:.0%}")
        fold_metrics = []

        for fold_info in folds:
            fold_idx = fold_info["fold"]
            set_seeds(42)

            result = prepare_fold_data(cancer_type, fold_info, config)
            X_train_np = dataframes_to_numpy(result["X_train"])
            X_val_np = dataframes_to_numpy(result["X_val"])
            y_train = result["y_train"]
            y_val = result["y_val"]

            # Zero-pad random modalities for `rate` fraction of TRAINING samples
            if rate > 0:
                rng = np.random.RandomState(42)
                n_train = len(y_train)
                for mod_name in ALL_MODALITIES:
                    if mod_name not in X_train_np:
                        continue
                    mask = rng.random(n_train) < rate
                    X_train_np[mod_name][mask] = 0.0

            modality_dims = {name: arr.shape[1] for name, arr in X_train_np.items()}
            n_classes = len(np.unique(np.concatenate([y_train, y_val])))

            train_ds = MultiOmicsDataset(X_train_np, y_train)
            val_ds = MultiOmicsDataset(X_val_np, y_val)
            train_loader = DataLoader(train_ds, batch_size=fusion_cfg["batch_size"],
                                      shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=fusion_cfg["batch_size"],
                                    shuffle=False, num_workers=0)

            model = IntermediateFusionModel(
                modality_dims=modality_dims,
                latent_dim=fusion_cfg["default_latent_dim"],
                hidden_dim=fusion_cfg["classifier_hidden"],
                num_classes=n_classes,
                dropout=fusion_cfg["dropout"],
                encoder_hidden=fusion_cfg["encoder_hidden"],
            ).to(device)

            metrics = train_single_fold(
                model, train_loader, val_loader, y_train, config, device,
            )
            fold_metrics.append(metrics)

            if rate > 0:
                log_experiment(
                    config_path=config["paths"]["experiment_log"],
                    experiment_name=f"ablation_missing_{int(rate*100)}pct",
                    cancer_type=cancer_type,
                    model_type="IntermediateFusion_ablation",
                    features=f"missing_{int(rate*100)}pct",
                    n_folds=n_folds, fold=fold_idx,
                    precision=round(metrics["precision"], 4),
                    recall=round(metrics["recall"], 4),
                    f1=round(metrics["f1"], 4),
                    nmi=round(metrics["nmi"], 4),
                    ari=round(metrics["ari"], 4),
                    notes=f"A3: {int(rate*100)}% modalities zeroed",
                    artifact_path="",
                )

        summary = compute_fold_summary(fold_metrics)
        rows.append({
            "cancer": cancer_short,
            "missing_rate": rate,
            "f1_mean": round(summary["f1_mean"], 4),
            "f1_std": round(summary["f1_std"], 4),
            "precision_mean": round(summary["precision_mean"], 4),
            "recall_mean": round(summary["recall_mean"], 4),
        })
        logger.info(
            f"  {cancer_short} missing={rate:.0%}: "
            f"F1={summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}"
        )

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_modality_removal(df, save_dir):
    """Grouped bar chart of F1 with each modality removed."""
    cancers = df["cancer"].unique()
    fig, axes = plt.subplots(1, len(cancers), figsize=(7 * len(cancers), 5))
    if len(cancers) == 1:
        axes = [axes]

    mod_colors = {
        "mrna": "#2196F3", "mirna": "#4CAF50",
        "methy": "#FF9800", "cnv": "#9C27B0",
    }

    for ax, cancer in zip(axes, cancers):
        sub = df[df["cancer"] == cancer]
        x = range(len(sub))
        bars = ax.bar(
            x, sub["f1_mean"], yerr=sub["f1_std"], capsize=4,
            color=[mod_colors.get(m, "#999") for m in sub["removed_modality"]],
            edgecolor="white", linewidth=1.2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"w/o {m}" for m in sub["removed_modality"]], fontsize=10,
        )
        ax.set_ylabel("F1 Score (mean +/- std)", fontsize=11)
        ax.set_title(
            f"Modality Removal -- {cancer}", fontsize=13, fontweight="bold",
        )
        ax.set_ylim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(save_dir, f"ablation_modality_removal.{ext}"),
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    logger.info("Saved: ablation_modality_removal.png/pdf")


def plot_fusion_comparison(df, save_dir):
    """Grouped bar chart comparing fusion strategies."""
    cancers = df["cancer"].unique()
    fig, axes = plt.subplots(1, len(cancers), figsize=(7 * len(cancers), 5))
    if len(cancers) == 1:
        axes = [axes]

    model_colors = {
        "XGBoost (early concat, tree)": "#E57373",
        "EarlyFusionMLP (early concat, deep)": "#64B5F6",
        "IntermediateFusion (per-encoder, deep)": "#81C784",
    }

    for ax, cancer in zip(axes, cancers):
        sub = df[df["cancer"] == cancer]
        x = range(len(sub))
        bars = ax.bar(
            x, sub["f1_mean"], yerr=sub["f1_std"], capsize=4,
            color=[model_colors.get(m, "#999") for m in sub["model"]],
            edgecolor="white", linewidth=1.2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["XGBoost\n(tree)", "Early MLP\n(deep)", "Intermediate\nFusion (deep)"],
            fontsize=9,
        )
        ax.set_ylabel("F1 Score (mean +/- std)", fontsize=11)
        ax.set_title(
            f"Fusion Strategy Comparison -- {cancer}", fontsize=13, fontweight="bold",
        )
        ax.set_ylim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(save_dir, f"ablation_fusion_comparison.{ext}"),
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    logger.info("Saved: ablation_fusion_comparison.png/pdf")


def plot_missing_modality_curve(df, save_dir):
    """Line plot of F1 vs missing modality rate."""
    cancers = df["cancer"].unique()
    fig, ax = plt.subplots(figsize=(8, 5))
    cancer_colors = {"BRCA": "#1976D2", "COAD": "#D32F2F"}

    for cancer in cancers:
        sub = df[df["cancer"] == cancer].sort_values("missing_rate")
        ax.errorbar(
            sub["missing_rate"] * 100, sub["f1_mean"], yerr=sub["f1_std"],
            marker="o", capsize=4, linewidth=2, markersize=6,
            color=cancer_colors.get(cancer, "#999"), label=cancer,
        )

    ax.set_xlabel("Missing Modality Rate (%)", fontsize=12)
    ax.set_ylabel("F1 Score (mean +/- std)", fontsize=12)
    ax.set_title(
        "Missing Modality Robustness", fontsize=14, fontweight="bold",
    )
    ax.set_xticks([0, 10, 20, 30, 50])
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(save_dir, f"missing_modality_curve.{ext}"),
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    logger.info("Saved: missing_modality_curve.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--only", choices=["a1", "a2", "a3"], default=None,
        help="Run only a specific ablation",
    )
    args = parser.parse_args()

    set_seeds(42)
    config = load_config()
    device = get_device()
    cancer_types = config["project"]["cancer_types"]

    metrics_dir = os.path.join(config["paths"]["results"], "metrics")
    plots_dir = os.path.join(config["paths"]["results"], "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    run_a1 = args.only is None or args.only == "a1"
    run_a2 = args.only is None or args.only == "a2"
    run_a3 = args.only is None or args.only == "a3"

    t0 = time.time()

    # --- A1: Modality Removal ---
    if run_a1:
        print("\n" + "=" * 60)
        print("  A1: MODALITY REMOVAL ABLATION")
        print("=" * 60)
        a1_rows = []
        for ct in cancer_types:
            a1_rows.extend(run_modality_removal(ct, config, device))
        a1_df = pd.DataFrame(a1_rows)
        a1_path = os.path.join(metrics_dir, "ablation_modality_removal.csv")
        a1_df.to_csv(a1_path, index=False, float_format="%.4f")
        logger.info(f"Saved: {a1_path}")
        plot_modality_removal(a1_df, plots_dir)

    # --- A2: Early vs Intermediate Fusion ---
    if run_a2:
        print("\n" + "=" * 60)
        print("  A2: FUSION STRATEGY COMPARISON")
        print("=" * 60)
        a2_rows = []
        for ct in cancer_types:
            a2_rows.extend(run_fusion_comparison(ct, config, device))
        a2_df = pd.DataFrame(a2_rows)
        a2_path = os.path.join(metrics_dir, "ablation_fusion_comparison.csv")
        a2_df.to_csv(a2_path, index=False, float_format="%.4f")
        logger.info(f"Saved: {a2_path}")
        plot_fusion_comparison(a2_df, plots_dir)

    # --- A3: Missing Modality ---
    if run_a3:
        print("\n" + "=" * 60)
        print("  A3: MISSING MODALITY SIMULATION")
        print("=" * 60)
        a3_rows = []
        for ct in cancer_types:
            a3_rows.extend(run_missing_modality(ct, config, device))
        a3_df = pd.DataFrame(a3_rows)
        a3_path = os.path.join(metrics_dir, "ablation_missing_modality.csv")
        a3_df.to_csv(a3_path, index=False, float_format="%.4f")
        logger.info(f"Saved: {a3_path}")
        plot_missing_modality_curve(a3_df, plots_dir)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ABLATION STUDIES COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
