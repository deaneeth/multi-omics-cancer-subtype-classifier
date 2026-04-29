"""
MLOmics Intermediate Fusion Training Script
=============================================
Trains the intermediate fusion model (per-modality encoders → latent concat → MLP)
with patient-level stratified 5-fold CV.

Usage:
    python scripts/train_fusion.py                   # Full data, both cancers
    python scripts/train_fusion.py --toy             # Toy data (BRCA only), 5 epochs
    python scripts/train_fusion.py --toy --epochs 10 # Override epoch count

Artifacts produced:
    models/intermediate_fusion/fusion_{cancer}_fold{i}.pt
    results/metrics/fusion_{cancer}_metrics.csv
    results/metrics/fusion_{cancer}_fold{i}_predictions.npz
    results/plots/fusion_training_curves_{cancer}.png
    results/plots/confusion_matrix_fusion_{cancer}.png
    experiment_log.csv entries
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    compute_fold_summary,
    compute_metrics,
    print_metrics_table,
    save_metrics,
)
from src.models import IntermediateFusionModel, MultiOmicsDataset
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

def plot_training_curves(all_curves: list, cancer_type: str, save_dir: str):
    """Plot train/val loss and val F1 for all folds."""
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
        f"Intermediate Fusion Training Curves — {cancer_type}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    cancer_short = cancer_type.split("-")[1]
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(save_dir, f"fusion_training_curves_{cancer_short}.{ext}"),
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    print(f"Training curves saved: fusion_training_curves_{cancer_short}.png/pdf")


def plot_best_fold_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cancer_type: str,
    save_dir: str,
    n_classes: int,
) -> str:
    """Save confusion matrix heatmap for the best fold."""
    os.makedirs(save_dir, exist_ok=True)
    cancer_short = cancer_type.split("-")[1]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"Class {i}" for i in range(n_classes)],
        yticklabels=[f"Class {i}" for i in range(n_classes)],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix: IntermediateFusion ({cancer_type}) — Best Fold",
        fontsize=13,
    )

    png_path = os.path.join(save_dir, f"confusion_matrix_fusion_{cancer_short}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Confusion matrix saved: {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Train / Eval steps
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, max_norm=1.0):
    """Single training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

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
    """Evaluate model on a DataLoader. Returns loss, y_true, y_pred."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in loader:
        x_batch = {k: v.to(device) for k, v in x_batch.items()}
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return avg_loss, y_true, y_pred


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_fusion_model(cancer_type: str, config: dict,
                       use_toy: bool = False, epoch_override: int = None):
    """Train intermediate fusion model with 5-fold CV."""
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
    print(f"Intermediate Fusion 5-Fold CV: {cancer_type} ({tag} data)")
    print(f"  latent_dim={latent_dim}, lr={lr}, batch_size={batch_size}")
    print(f"  max_epochs={max_epochs}, patience={patience}")
    print(f"{'='*60}")

    cv_data = load_cv_folds(cancer_type, config, use_toy=use_toy)
    folds = cv_data["folds"]
    n_folds = cv_data["n_folds"]

    model_dir = os.path.join(config["paths"]["models"], "intermediate_fusion")
    metrics_dir = os.path.join(config["paths"]["results"], "metrics")
    plots_dir = os.path.join(config["paths"]["results"], "plots")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    fold_metrics_list = []
    all_training_curves = []
    best_fold_idx = -1
    best_f1 = -1.0
    best_fold_y_true = None
    best_fold_y_pred = None
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
        print(f"  Train: {len(y_train)} samples, Val: {len(y_val)} samples")
        print(f"  Classes: {n_classes}, distribution: {np.bincount(y_train)}")

        train_ds = MultiOmicsDataset(X_train_np, y_train)
        val_ds = MultiOmicsDataset(X_val_np, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0)

        model = IntermediateFusionModel(
            modality_dims=modality_dims,
            latent_dim=latent_dim,
            hidden_dim=classifier_hidden,
            num_classes=n_classes,
            dropout=dropout,
            encoder_hidden=encoder_hidden,
        ).to(device)

        class_weights = compute_class_weights(y_train, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

        # Training loop with early stopping
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

            # NaN/Inf guard
            if np.isnan(train_loss) or np.isinf(train_loss):
                print(f"  ERROR: NaN/Inf loss at epoch {epoch}. Stopping fold.")
                break

        # Load best model and compute final metrics
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

        # Track best fold
        if final_metrics["f1"] > best_f1:
            best_f1 = final_metrics["f1"]
            best_fold_idx = fold_idx
            best_fold_y_true = final_y_true.copy()
            best_fold_y_pred = final_y_pred.copy()

        # Save model weights
        model_path = os.path.join(
            model_dir, f"fusion_{cancer_short}_fold{fold_idx}.pt",
        )
        torch.save({
            "model_state_dict": best_state,
            "modality_dims": modality_dims,
            "latent_dim": latent_dim,
            "num_classes": n_classes,
            "encoder_hidden": encoder_hidden,
            "classifier_hidden": classifier_hidden,
            "dropout": dropout,
        }, model_path)
        print(f"  Model saved: {model_path}")

        # Save predictions for later confusion matrix
        pred_path = os.path.join(
            metrics_dir, f"fusion_{cancer_short}_fold{fold_idx}_predictions.npz",
        )
        np.savez(pred_path, y_true=final_y_true, y_pred=final_y_pred)

        # Log experiment
        total_features = sum(modality_dims.values())
        log_experiment(
            config_path=config["paths"]["experiment_log"],
            experiment_name=f"fusion_intermediate_{cancer_short}",
            cancer_type=cancer_type,
            model_type="IntermediateFusion",
            features=f"intermediate_fusion_{total_features}",
            n_folds=n_folds,
            fold=fold_idx,
            precision=round(final_metrics["precision"], 4),
            recall=round(final_metrics["recall"], 4),
            f1=round(final_metrics["f1"], 4),
            nmi=round(final_metrics["nmi"], 4),
            ari=round(final_metrics["ari"], 4),
            notes=f"latent={latent_dim}, lr={lr}, bs={batch_size}, epochs_ran={len(curves['train_loss'])}",
            artifact_path=model_path,
        )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Intermediate Fusion {cancer_type} — Summary")
    print(f"{'='*60}")
    print_metrics_table(fold_metrics_list, f"IntermediateFusion ({cancer_type})")

    metrics_path = os.path.join(metrics_dir, f"fusion_{cancer_short}_metrics.csv")
    save_metrics(fold_metrics_list, "IntermediateFusion", cancer_type, metrics_path)

    print(f"\nBest fold: {best_fold_idx} (F1={best_f1:.4f})")

    plot_training_curves(all_training_curves, cancer_type, plots_dir)
    plot_best_fold_confusion_matrix(
        best_fold_y_true, best_fold_y_pred, cancer_type, plots_dir, n_classes,
    )

    # Leakage check
    summary = compute_fold_summary(fold_metrics_list)
    if summary["f1_mean"] > 0.95:
        print("\n!!! WARNING: F1 > 0.95 -- SUSPECT DATA LEAKAGE !!!")
    elif summary["f1_mean"] < 0.40:
        print("\n!!! WARNING: F1 < 0.40 -- Check data / hyperparameters !!!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train intermediate fusion model with 5-fold CV",
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

    for ct in cancer_types:
        train_fusion_model(ct, config, use_toy=args.toy, epoch_override=epoch_override)

    print("\nIntermediate fusion training complete.")


if __name__ == "__main__":
    main()
