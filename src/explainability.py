"""
MLOmics Explainability Module
==============================
SHAP values for tree models, Integrated Gradients for the fusion model,
and KEGG/GO pathway enrichment via gseapy.

Functions:
    compute_tree_shap        — TreeExplainer for XGBoost/RF
    compute_deep_attribution — Integrated Gradients via Captum
    extract_top_features     — Top-N features by mean |attribution|
    run_kegg_enrichment      — KEGG 2021 pathway enrichment
    run_go_enrichment        — GO Biological Process enrichment
    check_known_pathways     — Sanity-check for expected cancer pathways
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Tree-model SHAP
# ---------------------------------------------------------------------------

def compute_tree_shap(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 200,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for a tree-based model (XGBoost / RandomForest).

    Args:
        model: Fitted sklearn/xgboost tree model.
        X: Feature matrix (n_samples, n_features).
        feature_names: Column names (optional).
        max_samples: Subsample size for speed.

    Returns:
        shap_values: np.ndarray — raw SHAP output.
        importance_df: DataFrame with columns [feature, importance]
                       sorted descending by mean |SHAP|.
    """
    import shap

    if X.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # Handle multi-class SHAP output across different SHAP versions:
    # - SHAP >=0.45: ndarray (n_samples, n_features, n_classes)
    # - SHAP <0.45:  list of ndarrays, one per class
    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
        mean_abs = np.mean(np.abs(stacked), axis=(0, 1))
    elif shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) — average over samples and classes
        mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_sub.shape[1])]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(
        f"Tree SHAP computed on {X_sub.shape[0]} samples, "
        f"{X_sub.shape[1]} features"
    )
    return shap_values, importance_df


# ---------------------------------------------------------------------------
# 2. Deep-model attribution (Integrated Gradients)
# ---------------------------------------------------------------------------

class _FusionModelWrapper(nn.Module):
    """Wrapper so Captum can pass a tuple of tensors instead of a dict.

    Captum's IntegratedGradients requires tensor/tuple inputs.
    This wrapper converts positional tensors → dict → model.forward().
    """

    def __init__(self, fusion_model, modality_order: List[str]):
        super().__init__()
        self.fusion_model = fusion_model
        self.modality_order = modality_order

    def forward(self, *args):
        x_dict = {
            name: tensor
            for name, tensor in zip(self.modality_order, args)
        }
        return self.fusion_model(x_dict)


def compute_deep_attribution(
    model: nn.Module,
    X_dict: Dict[str, np.ndarray],
    method: str = "integrated_gradients",
    max_samples: int = 100,
    n_steps: int = 50,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Compute per-feature attribution for the intermediate fusion model.

    Uses Captum's IntegratedGradients via a wrapper that converts
    tuple inputs to the dict interface expected by IntermediateFusionModel.

    Args:
        model: Trained IntermediateFusionModel.
        X_dict: {modality_name: np.ndarray (n_samples, n_features)}.
        method: 'integrated_gradients' (recommended).
        max_samples: Cap to prevent OOM on 4 GB VRAM.
        n_steps: IG approximation steps.
        device: 'cpu' or 'cuda'.

    Returns:
        {modality_name: np.ndarray of shape (n_features,)} —
        mean absolute attribution per feature.
    """
    from captum.attr import IntegratedGradients

    modality_order = model.modality_order
    n_samples = next(iter(X_dict.values())).shape[0]

    # Subsample to prevent OOM
    if n_samples > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_samples, max_samples, replace=False)
        X_dict = {k: v[idx] for k, v in X_dict.items()}
        n_samples = max_samples
        logger.info(f"Subsampled to {max_samples} for attribution")

    # Prepare wrapper and tensors
    dev = torch.device(device)
    model = model.to(dev).eval()
    wrapper = _FusionModelWrapper(model, modality_order).to(dev)

    inputs = tuple(
        torch.tensor(X_dict[name], dtype=torch.float32, device=dev).requires_grad_(True)
        for name in modality_order
    )
    baselines = tuple(torch.zeros_like(t) for t in inputs)

    # Get predicted classes as attribution targets
    with torch.no_grad():
        logits = wrapper(*inputs)
        targets = logits.argmax(dim=1)

    ig = IntegratedGradients(wrapper)

    logger.info(
        f"Running Integrated Gradients ({n_samples} samples, "
        f"{n_steps} steps, device={device})..."
    )
    attributions = ig.attribute(
        inputs, baselines=baselines, target=targets,
        n_steps=n_steps, return_convergence_delta=False,
    )

    # Convert to per-feature mean |attribution|
    result = {}
    for name, attr_tensor in zip(modality_order, attributions):
        attr_np = attr_tensor.detach().cpu().numpy()
        result[name] = np.mean(np.abs(attr_np), axis=0)
        logger.info(
            f"  {name}: {result[name].shape[0]} features, "
            f"max_attr={result[name].max():.6f}"
        )

    return result


# ---------------------------------------------------------------------------
# 3. Top feature extraction
# ---------------------------------------------------------------------------

def extract_top_features(
    attributions: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    n: int = 50,
) -> pd.DataFrame:
    """Extract the top-N features ranked by mean absolute attribution.

    Args:
        attributions: {modality: np.ndarray of importance per feature}.
        feature_names_dict: {modality: list of feature name strings}.
        n: Number of top features to return.

    Returns:
        DataFrame with columns [rank, feature_name, modality, importance].
    """
    rows = []
    for mod, imp in attributions.items():
        names = feature_names_dict.get(mod, [f"{mod}_f{i}" for i in range(len(imp))])
        for name, val in zip(names, imp):
            rows.append({"feature_name": name, "modality": mod, "importance": float(val)})

    df = pd.DataFrame(rows)
    df = df.sort_values("importance", ascending=False).head(n).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# 4 & 5. Pathway enrichment (KEGG / GO)
# ---------------------------------------------------------------------------

def _run_enrichment(
    gene_list: List[str],
    gene_sets: str,
    label: str,
    pval_cutoff: float = 0.05,
    save_fallback_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Internal: run gseapy.enrichr with fallback to file export."""
    import gseapy

    if not gene_list:
        logger.warning(f"{label}: empty gene list — skipping")
        return None

    try:
        enr = gseapy.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism="Human",
            no_plot=True,
        )
        results = enr.results
        results = results[results["Adjusted P-value"] < pval_cutoff]
        results = results.sort_values("Adjusted P-value").reset_index(drop=True)
        logger.info(
            f"{label}: {len(results)} significant terms "
            f"(p < {pval_cutoff})"
        )
        return results

    except Exception as e:
        logger.warning(f"{label} enrichment failed: {e}")
        if save_fallback_dir:
            os.makedirs(save_fallback_dir, exist_ok=True)
            fallback_path = os.path.join(
                save_fallback_dir, "top_genes_for_manual_enrichr.txt"
            )
            with open(fallback_path, "w") as f:
                f.write("\n".join(gene_list))
            logger.info(
                f"Gene list saved to {fallback_path}. "
                f"Upload to https://maayanlab.cloud/Enrichr/ for manual enrichment."
            )
            print(
                f"⚠ {label} enrichment failed. Gene list saved to:\n"
                f"  {fallback_path}\n"
                f"  Upload to https://maayanlab.cloud/Enrichr/"
            )
        return None


def run_kegg_enrichment(
    gene_list: List[str],
    pval_cutoff: float = 0.05,
    save_fallback_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Run KEGG 2021 Human pathway enrichment on a list of gene symbols.

    Only mRNA gene symbols are valid inputs. miRNA/Methy/CNV probe IDs
    are not recognized by KEGG and should be filtered beforehand.

    Args:
        gene_list: List of gene symbol strings (e.g. ['TP53', 'ESR1']).
        pval_cutoff: Adjusted P-value threshold.
        save_fallback_dir: Dir to save a gene-list file if enrichr fails.

    Returns:
        DataFrame of significant KEGG pathways, or None on failure.
    """
    return _run_enrichment(
        gene_list, "KEGG_2021_Human", "KEGG",
        pval_cutoff, save_fallback_dir,
    )


def run_go_enrichment(
    gene_list: List[str],
    pval_cutoff: float = 0.05,
    save_fallback_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Run GO Biological Process 2021 enrichment on a list of gene symbols.

    Args:
        gene_list: Gene symbol strings.
        pval_cutoff: Adjusted P-value threshold.
        save_fallback_dir: Fallback file dir if enrichr fails.

    Returns:
        DataFrame of significant GO BP terms, or None on failure.
    """
    return _run_enrichment(
        gene_list, "GO_Biological_Process_2021", "GO_BP",
        pval_cutoff, save_fallback_dir,
    )


# ---------------------------------------------------------------------------
# 6. Known pathway sanity check
# ---------------------------------------------------------------------------

DEFAULT_CANCER_PATHWAYS = [
    "PI3K-Akt signaling",
    "p53 signaling",
    "MAPK signaling",
    "Cell cycle",
    "Apoptosis",
    "Wnt signaling",
    "Breast cancer",
    "Colorectal cancer",
    "Pathways in cancer",
]


def check_known_pathways(
    enrichment_results: Optional[pd.DataFrame],
    expected_pathways: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Check whether known cancer pathways appear in enrichment results.

    Args:
        enrichment_results: Output of run_kegg_enrichment or run_go_enrichment.
        expected_pathways: Pathway name substrings to search for.

    Returns:
        {pathway_name: {'found': bool, 'p_value': float or None,
                        'matched_term': str or None}}
    """
    if expected_pathways is None:
        expected_pathways = DEFAULT_CANCER_PATHWAYS

    report = {}
    for pathway in expected_pathways:
        entry = {"found": False, "p_value": None, "matched_term": None}
        if enrichment_results is not None and len(enrichment_results) > 0:
            term_col = "Term"
            mask = enrichment_results[term_col].str.contains(
                pathway, case=False, na=False,
            )
            if mask.any():
                hit = enrichment_results.loc[mask].iloc[0]
                entry["found"] = True
                entry["p_value"] = float(hit["Adjusted P-value"])
                entry["matched_term"] = hit[term_col]
        report[pathway] = entry

    found_count = sum(1 for v in report.values() if v["found"])
    logger.info(
        f"Pathway check: {found_count}/{len(expected_pathways)} "
        f"expected pathways found"
    )
    return report
