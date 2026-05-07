"""
MLOmics Data Loader
===================
Functions for loading modality CSVs, labels, and building the sample map.
All paths derived from config.yaml via load_config().
"""

import glob
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from src.utils import load_config

logger = logging.getLogger(__name__)

# Canonical modality keys (must match config.yaml 'modalities' section)
MODALITY_KEYS = ["mrna", "mirna", "methy", "cnv"]


def _resolve_modality_path(
    cancer_type: str,
    modality: str,
    config: dict,
    use_toy: bool,
) -> str:
    """Build the filepath for a modality CSV (raw or toy)."""
    cancer_short = cancer_type.split("-")[1]
    pattern = config["modalities"][modality]["file_pattern"].format(cancer=cancer_short)

    if use_toy:
        # Toy files: data/toy/{cancer}_mRNA_top_toy.csv
        base, ext = os.path.splitext(pattern)
        toy_name = f"{base}_toy{ext}"
        return os.path.join(config["paths"]["toy_data"], toy_name)

    return os.path.join(config["paths"]["raw_data"], cancer_type, "Top", pattern)


def load_modality(
    cancer_type: str,
    modality: str,
    config: dict,
    use_toy: bool = False,
) -> pd.DataFrame:
    """Load a single modality CSV, transpose it, and return samples x features.

    Args:
        cancer_type: e.g. "GS-BRCA" or "GS-COAD".
        modality: Config key — one of "mrna", "mirna", "methy", "cnv".
        config: Project config dict from load_config().
        use_toy: If True, load from data/toy/ instead of data/raw/.

    Returns:
        DataFrame with shape (n_samples, n_features).
        Index = sample IDs (str), columns = feature names (str).
    """
    filepath = _resolve_modality_path(cancer_type, modality, config, use_toy)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modality file not found: {filepath}")

    # Load (rows=features, cols=samples) then transpose
    df = pd.read_csv(filepath, index_col=0).T
    df.index = df.index.astype(str)

    tag = "toy" if use_toy else "full"
    logger.info(
        f"Loaded {cancer_type}/{modality} ({tag}): "
        f"{df.shape[0]} samples x {df.shape[1]} features"
    )
    return df


def load_labels(cancer_type: str, config: dict, use_toy: bool = False) -> pd.Series:
    """Load the label file for a cancer type.

    For raw data, uses glob to match *_label_num.csv (numeric prefix varies).
    For toy data, loads the fixed-name file from data/toy/.

    POSITIONAL ALIGNMENT CONTRACT:
        Label assignment is positional — the i-th row of the label CSV is
        assigned to the i-th sample ID from the mRNA feature file. The MLOmics
        benchmark guarantees this ordering. There is NO sample-ID column in the
        label file to verify row-by-row alignment.

        To confirm alignment integrity, run:
            python scripts/verify_label_alignment.py [--toy]

        Expected SHA-256 of the label files is recorded in
        ``data/label_file_checksums.json``. If label files are ever regenerated,
        re-run the verification script to refresh checksums and audit alignment.

    Args:
        cancer_type: e.g. "GS-BRCA".
        config: Project config dict.
        use_toy: If True, load from data/toy/.

    Returns:
        Series with sample IDs as index and integer labels as values.
    """
    cancer_short = cancer_type.split("-")[1]

    if use_toy:
        label_path = os.path.join(
            config["paths"]["toy_data"],
            f"{cancer_short}_label_num_toy.csv",
        )
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Toy label file not found: {label_path}")
    else:
        label_pattern = os.path.join(
            config["paths"]["raw_data"],
            cancer_type,
            "Top",
            f"*_{cancer_short}_label_num.csv",
        )
        label_files = glob.glob(label_pattern)
        if len(label_files) != 1:
            raise FileNotFoundError(
                f"Expected 1 label file matching '{label_pattern}', "
                f"found {len(label_files)}"
            )
        label_path = label_files[0]

    labels_df = pd.read_csv(label_path)

    # ASSUMPTION: label CSV rows are ordered identically to the columns of the
    # first-modality CSV (mRNA). The MLOmics dataset guarantees this, but there
    # is no sample-ID column in the label file to verify row-by-row alignment.
    # If the source files are ever regenerated independently, this positional
    # assignment can silently produce wrong class assignments.
    first_mod = MODALITY_KEYS[0]
    ref_df = load_modality(cancer_type, first_mod, config, use_toy=use_toy)
    sample_ids = list(ref_df.index)

    if len(labels_df) != len(sample_ids):
        raise ValueError(
            f"Label count ({len(labels_df)}) does not match "
            f"sample count from {first_mod} ({len(sample_ids)}). "
            f"Ensure the label file rows are in the same order as {first_mod} columns."
        )

    labels = pd.Series(
        labels_df["Label"].values,
        index=sample_ids,
        name="label",
    )
    labels.index.name = "sample_id"

    tag = "toy" if use_toy else "full"
    logger.info(
        f"Loaded labels for {cancer_type} ({tag}): {len(labels)} samples, "
        f"{labels.nunique()} subtypes"
    )
    return labels


def get_common_samples(
    cancer_type: str, config: dict, use_toy: bool = False,
) -> List[str]:
    """Find sample IDs present in ALL modalities for a cancer type.

    Args:
        cancer_type: e.g. "GS-BRCA".
        config: Project config dict.
        use_toy: If True, load from data/toy/.

    Returns:
        Sorted list of sample IDs common to all 4 modalities.
    """
    sample_sets = {}
    for mod_key in MODALITY_KEYS:
        df = load_modality(cancer_type, mod_key, config, use_toy=use_toy)
        sample_sets[mod_key] = set(df.index)

    common = sorted(set.intersection(*sample_sets.values()))
    tag = "toy" if use_toy else "full"
    logger.info(
        f"{cancer_type} ({tag}): {len(common)} common samples across all modalities"
    )
    return common


def create_sample_map(config: Optional[dict] = None) -> pd.DataFrame:
    """Build and save the sample map linking patient IDs across modalities.

    For each cancer type, records which modalities each sample has.
    Drops samples with <50% modalities (drop_threshold from config.yaml)
    and logs dropped samples to data/dropped_samples.csv.

    Args:
        config: Project config dict. If None, loads from config.yaml.

    Returns:
        DataFrame of the sample map (also saved to data/sample_map.csv).
    """
    if config is None:
        config = load_config()

    drop_threshold = config["preprocessing"]["drop_threshold"]
    n_modalities = len(MODALITY_KEYS)
    all_rows = []
    all_dropped = []

    for cancer_type in config["project"]["cancer_types"]:
        cancer_short = cancer_type.split("-")[1]

        # Load sample IDs per modality
        mod_samples: Dict[str, set] = {}
        for mod_key in MODALITY_KEYS:
            df = load_modality(cancer_type, mod_key, config)
            mod_samples[mod_key] = set(df.index)

        # Load labels to check label availability
        labels = load_labels(cancer_type, config)
        label_ids = set(labels.index)

        # Union of all sample IDs across modalities
        all_ids = sorted(set.union(*mod_samples.values()))

        for sid in all_ids:
            has = {f"has_{k}": sid in mod_samples[k] for k in MODALITY_KEYS}
            has["has_label"] = sid in label_ids
            modality_count = sum(v for k, v in has.items() if k != "has_label")
            available = [k.replace("has_", "") for k, v in has.items()
                         if v and k != "has_label"]

            row = {
                "sample_id": sid,
                **has,
                "modality_count": modality_count,
                "cancer_type": cancer_type,
            }

            # Check drop threshold
            if modality_count / n_modalities < drop_threshold:
                all_dropped.append({
                    "sample_id": sid,
                    "cancer_type": cancer_type,
                    "modality_count": modality_count,
                    "available_modalities": ",".join(available),
                    "reason": f"<{drop_threshold*100:.0f}% modalities ({modality_count}/{n_modalities})",
                })
            else:
                all_rows.append(row)

    sample_map = pd.DataFrame(all_rows)
    sample_map.to_csv(config["paths"]["sample_map"], index=False)

    # Save dropped samples (always write headers even if empty)
    dropped_path = os.path.join(os.path.dirname(config["paths"]["sample_map"]), "dropped_samples.csv")
    dropped_cols = ["sample_id", "cancer_type", "modality_count", "available_modalities", "reason"]
    dropped_df = pd.DataFrame(all_dropped, columns=dropped_cols)
    dropped_df.to_csv(dropped_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"SAMPLE MAP SUMMARY")
    print(f"{'='*60}")
    for cancer_type in config["project"]["cancer_types"]:
        ct_map = sample_map[sample_map["cancer_type"] == cancer_type]
        ct_dropped = dropped_df[dropped_df["cancer_type"] == cancer_type] if len(dropped_df) > 0 else pd.DataFrame()
        all_mods = ct_map["modality_count"] == n_modalities
        print(f"\n  {cancer_type}:")
        print(f"    Total kept:     {len(ct_map)}")
        print(f"    All modalities: {all_mods.sum()}")
        print(f"    Dropped:        {len(ct_dropped)}")

    total_dropped = len(dropped_df)
    print(f"\n  Total samples in map: {len(sample_map)}")
    print(f"  Total dropped:        {total_dropped}")
    if total_dropped > 0:
        print(f"  Dropped samples saved to: {dropped_path}")
    print(f"  Sample map saved to: {config['paths']['sample_map']}")

    logger.info(f"Sample map saved: {len(sample_map)} samples, {total_dropped} dropped")
    return sample_map
