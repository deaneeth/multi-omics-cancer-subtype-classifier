"""
Pathway-Aware Fusion Training Script
=====================================
Trains PathwayAwareFusionModel (KEGG pathway attention for mRNA +
standard encoders for mirna/methy/cnv) with patient-level stratified 5-fold CV.

Usage:
    python scripts/train_pathway_fusion.py
    python scripts/train_pathway_fusion.py --toy --epochs 5

Artifacts:
    models/pathway_fusion/pathway_fusion_{cancer}_fold{i}.pt
    results/metrics/pathway_fusion_{cancer}_metrics.csv
    results/enrichment/pathway_attention_scores.csv
    results/plots/pathway_fusion_training_curves_{cancer}.png/pdf
    experiment_log.csv entries
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    compute_fold_summary,
    compute_metrics,
    print_metrics_table,
    save_metrics,
)
from src.models import MultiOmicsDataset, PathwayAwareFusionModel
from src.preprocessing import load_cv_folds, prepare_fold_data
from src.utils import (
    compute_class_weights,
    dataframes_to_numpy,
    get_device,
    load_config,
    log_experiment,
    set_seeds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pathway_mapping(cancer_type: str) -> tuple:
    """Load KEGG pathway mapping for a specific cancer type.

    Returns:
        (pathway_indices, unmapped_indices, n_pathways, coverage_pct)
    """
    mapping_path = os.path.join("data", "processed", "pathway_gene_mapping.json")
    with open(mapping_path, "r") as f:
        full_mapping = json.load(f)

    cancer_short = cancer_type.split("-")[1]
    mapping = full_mapping[cancer_short]

    pathway_indices = mapping["pathways"]
    unmapped_indices = mapping["unmapped_indices"]
    coverage = mapping["coverage_pct"]
    n_mapped = mapping["n_mapped_features"]
    n_unmapped = mapping["n_unmapped_features"]

    print(f"  Pathway mapping loaded: {len(pathway_indices)} pathways, "
          f"{n_mapped} mapped + {n_unmapped} unmapped features ({coverage}% coverage)")

    return pathway_indices, unmapped_indices


def plot_training_curves(all_curves: list, cancer_type: str, save_dir: str):
    n_folds = len(all_curves)
    fig, axes = plt.subplots(2, n_folds, figsize=(5 * n_folds, 8), squeeze=False)

    for i, curves in enumerate(all_curves):
        epochs = range(1, len(curves["train_loss"]) + 1)

        ax_loss = axes[0, i]
        ax_loss.plot(epochs, curves["train_loss"], label="Train Loss", color="#1976D2")
        ax_loss.plot(epochs, curves["val_loss"], label="Val Loss", color="#D32F2F")
        ax_loss.set_title(f"Fold {i} — Loss", fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(alpha=0.3)

        ax_f1 = axes[1, i]
        ax_f1.plot(epochs, curves["val_f1"], label="Val F1", color="#388E3C")
        ax_f1.set_title(f"Fold {i} — Val F1", fontsize=11, fontweight="bold")
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("F1")
        ax_f1.legend(fontsize=8)
        ax_f1.grid(alpha=0.3)

    fig.suptitle(
        f"Pathway Fusion Training Curves — {cancer_type}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    cancer_short = cancer_type.split("-")[1]
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(save_dir, f"pathway_fusion_training_curves_{cancer_short}.{ext}"),
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    print(f"  Training curves saved: pathway_fusion_training_curves_{cancer_short}.png/pdf")


# ---------------------------------------------------------------------------
# Train / Eval steps
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, max_norm=1.0):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x_batch, y_batch in loader:
        x_batch = {k: v.to(device) for k, v in x_batch.items()}
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    for x_batch, y_batch in loader:
        x_batch = {k: v.to(device) for k, v in x_batch.items()}
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    return (total_loss / max(n_batches, 1),
            np.concatenate(all_labels), np.concatenate(all_preds))


@torch.no_grad()
def extract_attention_scores(model, loader, device):
    """Run full val set through model, collect per-sample attention weights."""
    model.eval()
    all_weights = []
    for x_batch, _ in loader:
        x_batch = {k: v.to(device) for k, v in x_batch.items()}
        _ = model(x_batch)
        weights = model.get_pathway_attention()  # (batch, n_pathways)
        all_weights.append(weights.cpu().numpy())
    return np.concatenate(all_weights, axis=0)  # (n_samples, n_pathways)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_pathway_fusion(cancer_type: str, config: dict,
                         use_toy: bool = False, epoch_override: int = None):
    cancer_short = cancer_type.split("-")[1].lower()
    tag = "toy" if use_toy else "full"
    fusion_cfg = config["fusion"]

    max_epochs = epoch_override if epoch_override else fusion_cfg["epochs"]
    patience = fusion_cfg["patience"]
    batch_size = fusion_cfg["batch_size"]
    lr = fusion_cfg["learning_rate"]
    wd = fusion_cfg["weight_decay"]
    latent_dim = fusion_cfg["default_latent_dim"]
    encoder_hidden = fusion_cfg["encoder_hidden"]
    classifier_hidden = fusion_cfg["classifier_hidden"]
    dropout = fusion_cfg["dropout"]

    device = get_device()

    print(f"\n{'='*60}")
    print(f"Pathway-Aware Fusion 5-Fold CV: {cancer_type} ({tag} data)")
    print(f"  latent_dim={latent_dim}, lr={lr}, batch_size={batch_size}")
    print(f"  max_epochs={max_epochs}, patience={patience}")
    print(f"{'='*60}")

    # Load pathway mapping for this cancer type
    pathway_indices, unmapped_indices = load_pathway_mapping(cancer_type)

    cv_data = load_cv_folds(cancer_type, config, use_toy=use_toy)
    folds = cv_data["folds"]
    n_folds = cv_data["n_folds"]

    model_dir = os.path.join(config["paths"]["models"], "pathway_fusion")
    metrics_dir = os.path.join(config["paths"]["results"], "metrics")
    plots_dir = os.path.join(config["paths"]["results"], "plots")
    enrich_dir = os.path.join(config["paths"]["results"], "enrichment")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(enrich_dir, exist_ok=True)

    fold_metrics_list = []
    all_training_curves = []
    all_attention_weights = []  # collect per-fold attention from val set
    n_classes = None

    for fold_info in folds:
        fold_idx = fold_info["fold"]
        set_seeds(42)
        print(f"\n--- Fold {fold_idx}/{n_folds - 1} ---")

        result = prepare_fold_data(cancer_type, fold_info, config, use_toy=use_toy)
        y_train = result["y_train"]
        y_val = result["y_val"]
        X_train_np = dataframes_to_numpy(result["X_train"])
        X_val_np = dataframes_to_numpy(result["X_val"])
        modality_dims = {name: arr.shape[1] for name, arr in X_train_np.items()}

        if n_classes is None:
            n_classes = len(np.unique(np.concatenate([y_train, y_val])))

        print(f"  Modality dims: {modality_dims}")
        print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Classes: {n_classes}")

        train_ds = MultiOmicsDataset(X_train_np, y_train)
        val_ds = MultiOmicsDataset(X_val_np, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=0)

        model = PathwayAwareFusionModel(
            modality_dims=modality_dims,
            pathway_indices=pathway_indices,
            unmapped_indices=unmapped_indices,
            latent_dim=latent_dim,
            hidden_dim=classifier_hidden,
            num_classes=n_classes,
            dropout=dropout,
            encoder_hidden=encoder_hidden,
            attn_hidden=64,
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")

        class_weights = compute_class_weights(y_train, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None
        curves = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
            )
            val_loss, val_y_true, val_y_pred = evaluate(
                model, val_loader, criterion, device,
            )
            val_metrics = compute_metrics(val_y_true, val_y_pred)
            scheduler.step(val_loss)

            curves["train_loss"].append(train_loss)
            curves["val_loss"].append(val_loss)
            curves["val_f1"].append(val_metrics["f1"])

            if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:3d}/{max_epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"val_f1={val_metrics['f1']:.4f}, lr={current_lr:.1e}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

            if np.isnan(train_loss) or np.isinf(train_loss):
                print(f"  ERROR: NaN/Inf loss at epoch {epoch}. Stopping fold.")
                break

        # Load best weights and compute final metrics
        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)

        _, final_y_true, final_y_pred = evaluate(model, val_loader, criterion, device)
        final_metrics = compute_metrics(final_y_true, final_y_pred)
        fold_metrics_list.append(final_metrics)
        all_training_curves.append(curves)

        print(
            f"  Fold {fold_idx} final: F1={final_metrics['f1']:.4f}, "
            f"Prec={final_metrics['precision']:.4f}, "
            f"Rec={final_metrics['recall']:.4f}"
        )

        # Extract attention weights from validation set
        attn_weights = extract_attention_scores(model, val_loader, device)
        all_attention_weights.append(attn_weights)

        # Save model
        model_path = os.path.join(
            model_dir, f"pathway_fusion_{cancer_short}_fold{fold_idx}.pt",
        )
        torch.save({
            "model_state_dict": best_state,
            "modality_dims": modality_dims,
            "latent_dim": latent_dim,
            "num_classes": n_classes,
            "encoder_hidden": encoder_hidden,
            "classifier_hidden": classifier_hidden,
            "dropout": dropout,
            "pathway_names": model.get_pathway_names(),
        }, model_path)
        print(f"  Model saved: {model_path}")

        # Log experiment
        total_features = sum(modality_dims.values())
        log_experiment(
            config_path=config["paths"]["experiment_log"],
            experiment_name=f"pathway_fusion_{cancer_short}",
            cancer_type=cancer_type,
            model_type="PathwayAwareFusion",
            features=f"pathway_fusion_{total_features}",
            n_folds=n_folds,
            fold=fold_idx,
            precision=round(final_metrics["precision"], 4),
            recall=round(final_metrics["recall"], 4),
            f1=round(final_metrics["f1"], 4),
            nmi=round(final_metrics["nmi"], 4),
            ari=round(final_metrics["ari"], 4),
            notes=f"pathway_attn, latent={latent_dim}, lr={lr}, epochs_ran={len(curves['train_loss'])}",
            artifact_path=model_path,
        )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Pathway Fusion {cancer_type} — Summary")
    print(f"{'='*60}")
    print_metrics_table(fold_metrics_list, f"PathwayAwareFusion ({cancer_type})")

    metrics_path = os.path.join(metrics_dir, f"pathway_fusion_{cancer_short}_metrics.csv")
    save_metrics(fold_metrics_list, "PathwayAwareFusion", cancer_type, metrics_path)

    plot_training_curves(all_training_curves, cancer_type, plots_dir)

    # Aggregate attention scores across folds → mean per pathway
    pathway_names = model.get_pathway_names()
    all_attn = np.concatenate(all_attention_weights, axis=0)  # (total_val, n_pathways)
    mean_attn = all_attn.mean(axis=0)  # (n_pathways,)

    top_k = min(20, len(pathway_names))
    top_indices = np.argsort(mean_attn)[::-1][:top_k]

    attn_rows = []
    for rank, idx in enumerate(top_indices, 1):
        attn_rows.append({
            "rank": rank,
            "cancer_type": cancer_type,
            "pathway": pathway_names[idx],
            "mean_attention_weight": round(float(mean_attn[idx]), 6),
            "std_attention_weight": round(float(all_attn[:, idx].std()), 6),
        })

    print(f"\nTop {top_k} pathways by attention weight:")
    for row in attn_rows[:10]:
        print(f"  {row['rank']:2d}. {row['pathway']}: {row['mean_attention_weight']:.6f}")

    summary = compute_fold_summary(fold_metrics_list)
    if summary["f1_mean"] > 0.95:
        print("\n!!! WARNING: F1 > 0.95 -- SUSPECT DATA LEAKAGE !!!")
    elif summary["f1_mean"] < 0.40:
        print("\n!!! WARNING: F1 < 0.40 -- Check data / hyperparameters !!!")

    return attn_rows, fold_metrics_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train pathway-aware fusion model with 5-fold CV",
    )
    parser.add_argument("--toy", action="store_true", help="Use toy data (BRCA only)")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max epochs (default: from config.yaml)",
    )
    args = parser.parse_args()

    set_seeds(42)
    config = load_config()

    if args.toy:
        cancer_types = ["GS-BRCA"]
        epoch_override = args.epochs if args.epochs else 5
        print(f"Running on TOY data (BRCA only, {epoch_override} epochs)")
    else:
        cancer_types = config["project"]["cancer_types"]
        epoch_override = args.epochs

    all_attn_rows = []
    for ct in cancer_types:
        attn_rows, _ = train_pathway_fusion(
            ct, config, use_toy=args.toy, epoch_override=epoch_override,
        )
        all_attn_rows.extend(attn_rows)

    # Save consolidated attention scores
    enrich_dir = os.path.join(config["paths"]["results"], "enrichment")
    os.makedirs(enrich_dir, exist_ok=True)
    attn_path = os.path.join(enrich_dir, "pathway_attention_scores.csv")
    pd.DataFrame(all_attn_rows).to_csv(attn_path, index=False)
    print(f"\nAttention scores saved: {attn_path}")
    print("Pathway-aware fusion training complete.")


if __name__ == "__main__":
    main()
