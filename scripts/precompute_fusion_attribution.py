"""
Precompute Integrated Gradients attribution for Streamlit demo samples.
Run this ONCE (per cancer type / model) before deploying the app:

    python scripts/precompute_fusion_attribution.py                           # intermediate, both cancers
    python scripts/precompute_fusion_attribution.py --cancer GS-BRCA
    python scripts/precompute_fusion_attribution.py --cancer GS-COAD
    python scripts/precompute_fusion_attribution.py --model pathway           # PathwayAwareFusion, both cancers
    python scripts/precompute_fusion_attribution.py --model pathway --cancer GS-BRCA

Reads:  app/model_artifacts/config_{c}.json
        app/model_artifacts/scaler_{c}.pkl
        app/model_artifacts/fusion_best_{c}.pt          (intermediate)
        app/model_artifacts/pathway_fusion_best_{c}.pt  (pathway)
        app/sample_input_{c}.csv

Writes: app/model_artifacts/{prefix}_attributions_{c}.npz
        app/model_artifacts/{prefix}_attribution_results_{c}.json

Migration: renames legacy BRCA files (no cancer suffix) to *_brca.* on first run.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import IntermediateFusionModel, PathwayAwareFusionModel
from src.utils import load_config, set_seeds

set_seeds(42)

APP_DIR = "app"
ARTIFACT_DIR = os.path.join(APP_DIR, "model_artifacts")

# Legacy (no-suffix) filenames -> new BRCA-suffixed names
_LEGACY_RENAMES = {
    "fusion_attributions.npz": "fusion_attributions_brca.npz",
    "fusion_attribution_results.json": "fusion_attribution_results_brca.json",
}


def _migrate_legacy_files():
    """Rename old no-suffix BRCA attribution files to *_brca.* once."""
    for old_name, new_name in _LEGACY_RENAMES.items():
        old_path = os.path.join(ARTIFACT_DIR, old_name)
        new_path = os.path.join(ARTIFACT_DIR, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"  Migrated: {old_name} -> {new_name}")


class FusionWrapper(torch.nn.Module):
    """Captum-compatible wrapper: accepts tuple inputs, returns logits."""

    def __init__(self, inner_model, modality_order):
        super().__init__()
        self.inner_model = inner_model
        self.modality_order = modality_order

    def forward(self, *inputs):
        d = {name: inputs[i] for i, name in enumerate(self.modality_order)}
        return self.inner_model(d)


def _load_model(model_type: str, cancer_type: str, modality_dims_override=None):
    """Load a fusion model checkpoint and return the model instance.

    Parameters
    ----------
    model_type : str
        Either ``"intermediate"`` or ``"pathway"``.
    cancer_type : str
        ``"GS-BRCA"`` or ``"GS-COAD"``.
    modality_dims_override : dict or None
        If given, passed to the model constructor (used when config modality
        dims differ from the checkpoint).

    Returns
    -------
    torch.nn.Module
        The loaded model in eval mode.
    """
    c = cancer_type.split("-")[1].lower()

    if model_type == "pathway":
        model_path = os.path.join(ARTIFACT_DIR, f"pathway_fusion_best_{c}.pt")
    else:
        model_path = os.path.join(ARTIFACT_DIR, f"fusion_best_{c}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint missing: {model_path}\n"
            f"Run prepare_demo_artifacts.py first."
        )

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    if model_type == "pathway":
        cancer_short = cancer_type.split("-")[1]  # "BRCA" or "COAD"
        mapping_path = os.path.join(ARTIFACT_DIR, "pathway_gene_mapping.json")
        with open(mapping_path) as _f:
            full_mapping = json.load(_f)
        mapping = full_mapping[cancer_short]
        pathway_indices = mapping["pathways"]
        unmapped_indices = mapping["unmapped_indices"]

        model = PathwayAwareFusionModel(
            modality_dims=ckpt["modality_dims"],
            pathway_indices=pathway_indices,
            unmapped_indices=unmapped_indices,
            latent_dim=ckpt["latent_dim"],
            hidden_dim=ckpt["classifier_hidden"],
            num_classes=ckpt["num_classes"],
            dropout=ckpt["dropout"],
            encoder_hidden=ckpt["encoder_hidden"],
        )
    else:
        model = IntermediateFusionModel(
            modality_dims=ckpt["modality_dims"],
            latent_dim=ckpt["latent_dim"],
            num_classes=ckpt["num_classes"],
            encoder_hidden=ckpt["encoder_hidden"],
            hidden_dim=ckpt["classifier_hidden"],
            dropout=ckpt["dropout"],
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.train(mode=False)
    return model


def compute_for_cancer(cancer_type: str, model_type: str = "intermediate"):
    """Compute and save IG attributions for all demo samples of one cancer."""
    from captum.attr import IntegratedGradients

    c = cancer_type.split("-")[1].lower()
    prefix = "pathway_fusion" if model_type == "pathway" else "fusion"
    model_label = "PathwayAwareFusion" if model_type == "pathway" else "IntermediateFusion"

    print(f"\n{'='*60}")
    print(f"Computing IG attributions for {cancer_type} ({model_label})")
    print(f"{'='*60}")

    # --- Load artifacts ---
    config_path = os.path.join(ARTIFACT_DIR, f"config_{c}.json")
    scaler_path = os.path.join(ARTIFACT_DIR, f"scaler_{c}.pkl")
    sample_path = os.path.join(APP_DIR, f"sample_input_{c}.csv")

    for p in [config_path, scaler_path, sample_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required artifact missing: {p}\n"
                f"Run prepare_demo_artifacts.py --cancer {cancer_type} first."
            )

    with open(config_path) as f:
        app_config = json.load(f)

    feature_names = app_config["feature_names"]
    class_names = app_config.get("class_names", {})
    modality_order = app_config["modality_order"]
    modality_dims = app_config["modality_dims"]

    scaler = joblib.load(scaler_path)

    # Load the requested model
    fusion_model = _load_model(model_type, cancer_type)

    # --- Load and preprocess sample data ---
    sample_df = pd.read_csv(sample_path, index_col=0)
    # Ensure samples-as-rows orientation
    if sample_df.shape[0] > sample_df.shape[1]:
        sample_df = sample_df.T

    sample_ids = list(sample_df.index)
    data_scaled = scaler.transform(sample_df.values.astype(np.float64))

    # Split into per-modality tensors
    start = 0
    x_dict = {}
    for mod in modality_order:
        dim = modality_dims[mod]
        x_dict[mod] = torch.FloatTensor(data_scaled[:, start : start + dim])
        start += dim

    wrapper = FusionWrapper(fusion_model, modality_order)
    ig = IntegratedGradients(wrapper)

    all_attributions = []
    all_predictions = []

    for si in range(len(sample_ids)):
        print(f"  Computing IG for sample {si + 1}/{len(sample_ids)}: {sample_ids[si]}")

        inputs = tuple(x_dict[mod][si : si + 1] for mod in modality_order)

        with torch.no_grad():
            logits = wrapper(*inputs)
            pred_class = logits.argmax(dim=1).item()
            probs = torch.softmax(logits, dim=1).numpy()[0]

        all_predictions.append(
            {
                "sample_id": sample_ids[si],
                "predicted_class": pred_class,
                "confidence": float(probs[pred_class]),
                "probabilities": probs.tolist(),
            }
        )

        baselines = tuple(torch.zeros_like(inp) for inp in inputs)
        attributions = ig.attribute(
            inputs, baselines=baselines, target=pred_class, n_steps=50
        )
        attr_concat = (
            torch.cat([a.detach() for a in attributions], dim=1).numpy()[0]
        )
        all_attributions.append(attr_concat)

    all_attributions = np.array(all_attributions)

    # --- Build top features per sample ---
    results = {
        "cancer_type": cancer_type,
        "model_type": model_label,
        "sample_ids": sample_ids,
        "predictions": all_predictions,
        "top_features_per_sample": [],
    }

    for si in range(len(sample_ids)):
        attr = all_attributions[si]
        abs_attr = np.abs(attr)
        top_indices = np.argsort(abs_attr)[::-1][:20]

        top_features = []
        for idx in top_indices:
            feat_name = (
                feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            )
            pos = 0
            modality = "unknown"
            for mod in modality_order:
                dim = modality_dims[mod]
                if idx < pos + dim:
                    modality = mod
                    break
                pos += dim

            top_features.append(
                {
                    "feature_name": feat_name,
                    "modality": modality,
                    "attribution": float(attr[idx]),
                    "abs_attribution": float(abs_attr[idx]),
                }
            )

        results["top_features_per_sample"].append(top_features)

    # --- Save ---
    npz_path = os.path.join(ARTIFACT_DIR, f"{prefix}_attributions_{c}.npz")
    json_path = os.path.join(ARTIFACT_DIR, f"{prefix}_attribution_results_{c}.json")

    np.savez(npz_path, attributions=all_attributions, sample_ids=sample_ids)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: {npz_path} {all_attributions.shape}")
    print(f"  Saved: {json_path} (top 20 per sample)")
    for i, sid in enumerate(sample_ids):
        pred = all_predictions[i]
        cn = class_names.get(
            str(pred["predicted_class"]), f"Class {pred['predicted_class']}"
        )
        top3 = ", ".join(
            f["feature_name"] for f in results["top_features_per_sample"][i][:3]
        )
        print(f"  {sid}: {cn} ({pred['confidence']:.1%}) — top: {top3}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute IG attributions for Streamlit demo samples."
    )
    parser.add_argument(
        "--cancer",
        default="all",
        choices=["GS-BRCA", "GS-COAD", "all"],
        help="Cancer type to precompute for (default: all)",
    )
    parser.add_argument(
        "--model",
        default="intermediate",
        choices=["intermediate", "pathway"],
        help="Which fusion model to use: intermediate or pathway (default: intermediate)",
    )
    args = parser.parse_args()

    config = load_config()

    # Rename legacy no-suffix BRCA files once
    _migrate_legacy_files()

    if args.cancer == "all":
        cancer_types = config["project"]["cancer_types"]
    else:
        cancer_types = [args.cancer]

    for cancer_type in cancer_types:
        compute_for_cancer(cancer_type, model_type=args.model)

    print("\n=== Attribution precomputation complete ===")


if __name__ == "__main__":
    main()
