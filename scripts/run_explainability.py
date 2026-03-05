"""
MLOmics Full Explainability Pipeline
======================================
Computes fusion model attribution (IG), extracts top features,
runs KEGG/GO enrichment, and validates against known cancer pathways.

Usage:
    python scripts/run_explainability.py            # Full run
    python scripts/run_explainability.py --toy       # Toy data test
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, set_seeds
from src.preprocessing import prepare_fold_data
from src.models import IntermediateFusionModel
from src.explainability import (
    compute_deep_attribution,
    extract_top_features,
    run_kegg_enrichment,
    run_go_enrichment,
    check_known_pathways,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODALITIES = ["mrna", "mirna", "methy", "cnv"]

MODALITY_COLORS = {
    "mrna": "#2196F3",
    "mirna": "#4CAF50",
    "methy": "#FF9800",
    "cnv": "#9C27B0",
}


def find_best_fold(cancer_type, model_type_label, config):
    """Find the fold with highest F1 from the experiment log."""
    log_path = config["paths"].get("experiment_log", "experiment_log.csv")
    df = pd.read_csv(log_path)
    sub = df[(df["cancer_type"] == cancer_type) & (df["model_type"] == model_type_label)]
    # Exclude toy runs for fusion model
    if model_type_label == "IntermediateFusion":
        sub = sub[~sub["notes"].str.contains("epochs_ran=5", na=False)]
    if len(sub) == 0:
        return 0
    return int(sub.loc[sub["f1"].idxmax(), "fold"])


def save_figure(fig, base_path):
    """Save as 300-DPI PNG + vector PDF."""
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", facecolor="white")
    logger.info(f"Saved: {base_path}.png/pdf")


def load_fusion_model(cancer_short, fold_idx, config):
    """Load a fusion model checkpoint by cancer type and fold."""
    ckpt_path = os.path.join(
        config["paths"]["models"], "intermediate_fusion",
        f"fusion_{cancer_short}_fold{fold_idx}.pt",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IntermediateFusionModel(
        modality_dims=ckpt["modality_dims"],
        latent_dim=ckpt["latent_dim"],
        hidden_dim=ckpt["classifier_hidden"],
        num_classes=ckpt["num_classes"],
        dropout=ckpt["dropout"],
        encoder_hidden=ckpt["encoder_hidden"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded fusion model: {ckpt_path}")
    return model


def plot_attribution_bar(top_df, cancer_short, model_label, save_dir):
    """Bar chart of top features colored by modality."""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [MODALITY_COLORS.get(m, "#999") for m in top_df["modality"]]
    bars = ax.barh(
        range(len(top_df) - 1, -1, -1), top_df["importance"].values,
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(len(top_df) - 1, -1, -1))
    ax.set_yticklabels(top_df["feature_name"].values, fontsize=8)
    ax.set_xlabel("Mean |Attribution|", fontsize=11)
    ax.set_title(
        f"Top {len(top_df)} Features — {model_label} ({cancer_short})",
        fontsize=13, fontweight="bold",
    )

    from matplotlib.patches import Patch
    present = top_df["modality"].unique()
    legend_items = [
        Patch(color=MODALITY_COLORS[m], label=m)
        for m in MODALITIES if m in present
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    base = os.path.join(save_dir, f"fusion_attribution_{cancer_short}")
    save_figure(fig, base)
    plt.close(fig)


def plot_enrichment_bar(enrichment_df, title, save_path_base, max_terms=15):
    """Horizontal bar chart of enrichment results."""
    if enrichment_df is None or len(enrichment_df) == 0:
        logger.warning(f"No enrichment results for: {title}")
        return
    df = enrichment_df.head(max_terms).copy()
    df["-log10(padj)"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-30))

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
    ax.barh(
        range(len(df) - 1, -1, -1),
        df["-log10(padj)"].values,
        color="#1976D2", alpha=0.85, edgecolor="white",
    )
    ax.set_yticks(range(len(df) - 1, -1, -1))
    # Truncate long term names
    labels = [t[:60] + "..." if len(t) > 60 else t for t in df["Term"].values]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("-log₁₀(Adjusted P-value)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_figure(fig, save_path_base)
    plt.close(fig)


def run_pipeline_for_cancer(cancer_type, config, use_toy=False):
    """Run the full explainability pipeline for one cancer type."""
    cancer_short = cancer_type.split("-")[1]
    cancer_lower = cancer_short.lower()

    shap_dir = os.path.join(config["paths"]["results"], "shap")
    enrichment_dir = os.path.join(config["paths"]["results"], "enrichment")
    os.makedirs(enrichment_dir, exist_ok=True)
    fusion_shap_dir = os.path.join(shap_dir, "fusion")
    os.makedirs(fusion_shap_dir, exist_ok=True)

    # =============================================
    # 1. Fusion model IG attribution
    # =============================================
    logger.info(f"\n{'='*60}")
    logger.info(f"FUSION IG ATTRIBUTION — {cancer_type}")
    logger.info(f"{'='*60}")

    best_fold = find_best_fold(cancer_type, "IntermediateFusion", config)
    logger.info(f"Best fusion fold: {best_fold}")

    # Load fold data
    with open(config["paths"]["cv_folds"], "r") as f:
        all_folds = json.load(f)
    fold_info = all_folds[cancer_type]["folds"][best_fold]
    fold_data = prepare_fold_data(cancer_type, fold_info, config, use_toy=use_toy)

    # Load model
    fusion_model = load_fusion_model(cancer_lower, best_fold, config)

    # Run IG
    X_val_dict = {m: fold_data["X_val"][m].values for m in MODALITIES}
    ig_samples = 20 if use_toy else 100
    ig_steps = 10 if use_toy else 50
    attribs = compute_deep_attribution(
        fusion_model, X_val_dict,
        max_samples=ig_samples, n_steps=ig_steps, device="cpu",
    )

    # Feature names from fold data
    feat_names = {m: list(fold_data["X_val"][m].columns) for m in MODALITIES}

    # Top 50 features across ALL modalities
    top_all = extract_top_features(attribs, feat_names, n=50)
    top_all_path = os.path.join(fusion_shap_dir, f"top_50_features_fusion_{cancer_short}.csv")
    top_all.to_csv(top_all_path, index=False, float_format="%.6f")
    logger.info(f"Saved top 50 overall: {top_all_path}")

    # Top 50 mRNA features only (for KEGG/GO enrichment)
    all_features_df = extract_top_features(attribs, feat_names, n=len(attribs["mrna"]))
    mrna_top = all_features_df[all_features_df["modality"] == "mrna"].head(50).reset_index(drop=True)
    mrna_top["rank"] = range(1, len(mrna_top) + 1)
    mrna_path = os.path.join(fusion_shap_dir, f"top_50_mrna_fusion_{cancer_short}.csv")
    mrna_top.to_csv(mrna_path, index=False, float_format="%.6f")
    logger.info(f"Saved top 50 mRNA: {mrna_path}")

    # Plot attribution bar chart
    plot_attribution_bar(top_all, cancer_short, "IntermediateFusion", fusion_shap_dir)

    # =============================================
    # 2. Extract XGBoost top mRNA genes
    # =============================================
    logger.info(f"\nXGBoost mRNA gene extraction — {cancer_type}")
    xgb_top_path = os.path.join(shap_dir, "xgb", f"top_50_features_xgb_{cancer_short}.csv")
    xgb_top_df = pd.read_csv(xgb_top_path)
    # Feature names have modality prefix: mrna_GENENAME
    xgb_mrna = xgb_top_df[xgb_top_df["modality"] == "mrna"].copy()
    xgb_gene_list = [f.replace("mrna_", "") for f in xgb_mrna["feature_name"].tolist()]
    logger.info(f"XGBoost mRNA genes: {len(xgb_gene_list)}")

    # Fusion mRNA genes (no prefix in IG output)
    fusion_gene_list = mrna_top["feature_name"].tolist()
    logger.info(f"Fusion mRNA genes: {len(fusion_gene_list)}")

    # =============================================
    # 3. KEGG Enrichment
    # =============================================
    logger.info(f"\nKEGG ENRICHMENT — {cancer_type}")

    kegg_xgb = run_kegg_enrichment(
        xgb_gene_list, pval_cutoff=0.05,
        save_fallback_dir=enrichment_dir,
    )
    if kegg_xgb is not None:
        kegg_xgb_path = os.path.join(enrichment_dir, f"kegg_xgb_{cancer_short}.csv")
        kegg_xgb.to_csv(kegg_xgb_path, index=False)
        logger.info(f"Saved: {kegg_xgb_path} ({len(kegg_xgb)} terms)")
        plot_enrichment_bar(
            kegg_xgb,
            f"KEGG Enrichment — XGBoost Top mRNA ({cancer_short})",
            os.path.join(enrichment_dir, f"kegg_barplot_xgb_{cancer_short}"),
        )

    kegg_fus = run_kegg_enrichment(
        fusion_gene_list, pval_cutoff=0.05,
        save_fallback_dir=enrichment_dir,
    )
    if kegg_fus is not None:
        kegg_fus_path = os.path.join(enrichment_dir, f"kegg_fusion_{cancer_short}.csv")
        kegg_fus.to_csv(kegg_fus_path, index=False)
        logger.info(f"Saved: {kegg_fus_path} ({len(kegg_fus)} terms)")
        plot_enrichment_bar(
            kegg_fus,
            f"KEGG Enrichment — Fusion Top mRNA ({cancer_short})",
            os.path.join(enrichment_dir, f"kegg_barplot_fusion_{cancer_short}"),
        )

    # =============================================
    # 4. GO Enrichment
    # =============================================
    logger.info(f"\nGO ENRICHMENT — {cancer_type}")

    go_xgb = run_go_enrichment(
        xgb_gene_list, pval_cutoff=0.05,
        save_fallback_dir=enrichment_dir,
    )
    if go_xgb is not None:
        go_xgb_path = os.path.join(enrichment_dir, f"go_xgb_{cancer_short}.csv")
        go_xgb.to_csv(go_xgb_path, index=False)
        logger.info(f"Saved: {go_xgb_path} ({len(go_xgb)} terms)")

    go_fus = run_go_enrichment(
        fusion_gene_list, pval_cutoff=0.05,
        save_fallback_dir=enrichment_dir,
    )
    if go_fus is not None:
        go_fus_path = os.path.join(enrichment_dir, f"go_fusion_{cancer_short}.csv")
        go_fus.to_csv(go_fus_path, index=False)
        logger.info(f"Saved: {go_fus_path} ({len(go_fus)} terms)")

    # Combined enrichment barplot
    best_kegg = kegg_xgb if (kegg_xgb is not None and len(kegg_xgb) > 0) else kegg_fus
    if best_kegg is not None and len(best_kegg) > 0:
        plot_enrichment_bar(
            best_kegg,
            f"KEGG Pathways -- {cancer_short}",
            os.path.join(enrichment_dir, f"enrichment_barplot_{cancer_short}"),
        )

    # =============================================
    # 5. Pathway Validation
    # =============================================
    logger.info(f"\nPATHWAY VALIDATION — {cancer_type}")

    validation = {}
    for label, enr_df in [("xgb_kegg", kegg_xgb), ("fusion_kegg", kegg_fus),
                           ("xgb_go", go_xgb), ("fusion_go", go_fus)]:
        report = check_known_pathways(enr_df)
        validation[label] = report
        found = [k for k, v in report.items() if v["found"]]
        logger.info(f"  {label}: {len(found)} known pathways found")

    # Save validation JSON
    validation_path = os.path.join(
        enrichment_dir, f"pathway_validation_{cancer_short}.json"
    )
    # Convert for JSON serialization
    validation_serializable = {}
    for key, rep in validation.items():
        validation_serializable[key] = {
            pw: {
                "found": v["found"],
                "p_value": v["p_value"],
                "matched_term": v["matched_term"],
            }
            for pw, v in rep.items()
        }
    with open(validation_path, "w") as f:
        json.dump(validation_serializable, f, indent=2)
    logger.info(f"Saved: {validation_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Pathway Validation Summary -- {cancer_type}")
    print(f"{'='*60}")
    from src.explainability import DEFAULT_CANCER_PATHWAYS
    for pw in DEFAULT_CANCER_PATHWAYS:
        xk = validation["xgb_kegg"][pw]
        fk = validation["fusion_kegg"][pw]
        xk_mark = f"YES (p={xk['p_value']:.2e})" if xk["found"] else "NO"
        fk_mark = f"YES (p={fk['p_value']:.2e})" if fk["found"] else "NO"
        print(f"  {pw:30s}  XGB: {xk_mark:25s}  Fusion: {fk_mark}")

    return {
        "cancer_type": cancer_type,
        "attribs": attribs,
        "top_all": top_all,
        "mrna_top": mrna_top,
        "kegg_xgb": kegg_xgb,
        "kegg_fus": kegg_fus,
        "go_xgb": go_xgb,
        "go_fus": go_fus,
        "validation": validation,
    }


def main():
    parser = argparse.ArgumentParser(description="MLOmics Explainability Pipeline")
    parser.add_argument("--toy", action="store_true", help="Use toy data for testing")
    args = parser.parse_args()

    set_seeds(42)
    config = load_config()
    cancer_types = config["project"]["cancer_types"]

    results = {}
    for ct in cancer_types:
        results[ct] = run_pipeline_for_cancer(ct, config, use_toy=args.toy)

    print("\n" + "=" * 60)
    print("  EXPLAINABILITY PIPELINE COMPLETE")
    print("=" * 60)
    for ct in cancer_types:
        r = results[ct]
        print(f"\n  {ct}:")
        stats = r["top_all"]["modality"].value_counts()
        for mod, count in stats.items():
            print(f"    Top-50 features from {mod}: {count}")
        if r["kegg_xgb"] is not None:
            print(f"    KEGG (XGBoost): {len(r['kegg_xgb'])} pathways")
        if r["kegg_fus"] is not None:
            print(f"    KEGG (Fusion):  {len(r['kegg_fus'])} pathways")


if __name__ == "__main__":
    main()
