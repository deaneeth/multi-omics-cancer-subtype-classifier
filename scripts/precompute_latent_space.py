"""
Precompute latent space t-SNE/UMAP embeddings for Streamlit demo.

Run ONCE (per cancer type) before deploying the app:

    python scripts/precompute_latent_space.py                  # both cancers
    python scripts/precompute_latent_space.py --cancer GS-BRCA
    python scripts/precompute_latent_space.py --cancer GS-COAD
    python scripts/precompute_latent_space.py --cancer all

Reads:  app/model_artifacts/config_{c}.json  (for class names + best fold)
        models/intermediate_fusion/fusion_{c}_fold{n}.pt

Writes: app/model_artifacts/latent_space_data_{c}.json

Migration: splits legacy combined latent_space_data.json into per-cancer files.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import IntermediateFusionModel
from src.preprocessing import load_cv_folds, prepare_fold_data
from src.utils import load_config, set_seeds

set_seeds(42)

APP_DIR = "app"
ARTIFACT_DIR = os.path.join(APP_DIR, "model_artifacts")


def _migrate_legacy_file():
    """Split legacy combined latent_space_data.json into per-cancer files."""
    legacy_path = os.path.join(ARTIFACT_DIR, "latent_space_data.json")
    if not os.path.exists(legacy_path):
        return

    with open(legacy_path) as f:
        combined = json.load(f)

    for cancer_type, entry in combined.items():
        c = cancer_type.split("-")[1].lower()
        new_path = os.path.join(ARTIFACT_DIR, f"latent_space_data_{c}.json")
        if not os.path.exists(new_path):
            with open(new_path, "w") as fp:
                json.dump({cancer_type: entry}, fp)
            print(f"  Migrated {cancer_type} -> {new_path}")


def compute_for_cancer(cancer_type: str, config: dict):
    """Compute and save latent space embeddings for one cancer type."""
    c = cancer_type.split("-")[1].lower()
    print(f"\n{'='*50}\nProcessing {cancer_type}\n{'='*50}")

    # Load class names from the cancer-specific config artifact
    class_names = {}
    cfg_path = os.path.join(ARTIFACT_DIR, f"config_{c}.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            class_names = json.load(f).get("class_names", {})

    folds = load_cv_folds(cancer_type, config)["folds"]

    # Find best fold: prefer config artifact, fall back to metrics CSV
    best_fold_idx = 0
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            best_fold_idx = json.load(f).get("fusion_best_fold", 0)
    else:
        metrics_path = f"results/metrics/fusion_{c}_metrics.csv"
        if os.path.exists(metrics_path):
            mdf = pd.read_csv(metrics_path)
            f1_cols = [
                col for col in mdf.columns
                if "f1" in col.lower()
                and "mean" not in col.lower()
                and "std" not in col.lower()
            ]
            if f1_cols:
                best_fold_idx = int(mdf[f1_cols[0]].idxmax())

    print(f"Using fold {best_fold_idx}")

    fold_info = folds[best_fold_idx]
    fold_data = prepare_fold_data(cancer_type, fold_info, config, use_toy=False)

    # Load model from trained models directory
    model_path = f"models/intermediate_fusion/fusion_{c}_fold{best_fold_idx}.pt"
    print(f"Loading {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    fusion_model = IntermediateFusionModel(
        modality_dims=ckpt["modality_dims"],
        latent_dim=ckpt["latent_dim"],
        num_classes=ckpt["num_classes"],
        encoder_hidden=ckpt["encoder_hidden"],
        hidden_dim=ckpt["classifier_hidden"],
        dropout=ckpt["dropout"],
    )
    fusion_model.load_state_dict(ckpt["model_state_dict"])
    fusion_model.train(mode=False)  # inference mode (equivalent to .eval())

    # Get latent vectors from validation fold
    x_val = {
        k: torch.FloatTensor(v.values if hasattr(v, "values") else v)
        for k, v in fold_data["X_val"].items()
    }
    with torch.no_grad():
        latent = fusion_model.get_latent(x_val).numpy()
        preds = fusion_model(x_val).argmax(dim=1).numpy()

    y_true = np.array(fold_data["y_val"])
    n = latent.shape[0]
    print(f"Latent shape: {latent.shape}, samples: {n}")

    # t-SNE
    from sklearn.manifold import TSNE
    perp = min(30, n // 4, n - 1)
    print(f"Running t-SNE (perplexity={perp})...")
    tsne_2d = TSNE(
        n_components=2, perplexity=perp, random_state=42, n_iter=1000, init="pca"
    ).fit_transform(latent)

    # UMAP (optional)
    umap_2d = None
    try:
        from umap import UMAP
        nn = min(15, n - 1)
        print(f"Running UMAP (n_neighbors={nn})...")
        umap_2d = UMAP(n_components=2, random_state=42, n_neighbors=nn).fit_transform(latent)
    except ImportError:
        print("umap-learn not installed — t-SNE only.")

    entry = {
        "cancer_type": cancer_type,
        "fold": int(best_fold_idx),
        "n_samples": int(n),
        "latent_dim": int(latent.shape[1]),
        "class_names": class_names,
        "tsne": {"x": tsne_2d[:, 0].tolist(), "y": tsne_2d[:, 1].tolist()},
        "true_labels": y_true.tolist(),
        "predicted_labels": preds.tolist(),
        "sample_ids": fold_info["val"][:n],
    }
    if umap_2d is not None:
        entry["umap"] = {"x": umap_2d[:, 0].tolist(), "y": umap_2d[:, 1].tolist()}

    correct = int(np.sum(preds == y_true))
    print(f"Accuracy: {correct}/{n} ({correct/n:.1%})")

    # Save per-cancer file
    out_path = os.path.join(ARTIFACT_DIR, f"latent_space_data_{c}.json")
    with open(out_path, "w") as f:
        json.dump({cancer_type: entry}, f)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute latent space t-SNE/UMAP embeddings for the demo."
    )
    parser.add_argument(
        "--cancer",
        default="all",
        choices=["GS-BRCA", "GS-COAD", "all"],
        help="Cancer type to precompute for (default: all)",
    )
    args = parser.parse_args()

    config = load_config()

    # Migrate legacy combined file once
    _migrate_legacy_file()

    if args.cancer == "all":
        cancer_types = config["project"]["cancer_types"]
    else:
        cancer_types = [args.cancer]

    for cancer_type in cancer_types:
        compute_for_cancer(cancer_type, config)

    print("\n=== Latent space precomputation complete ===")


if __name__ == "__main__":
    main()
