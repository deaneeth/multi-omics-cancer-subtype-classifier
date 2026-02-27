"""
Generate a 50-sample toy subset of GS-BRCA for rapid pipeline prototyping.

Saves files in the SAME CSV format as the originals (features x samples)
so the existing data_loader works on both via the use_toy flag.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import (
    MODALITY_KEYS,
    get_common_samples,
    load_labels,
    load_modality,
)
from src.utils import load_config, set_seeds

TOY_SIZE = 50


def create_toy_dataset():
    set_seeds(42)
    config = load_config()
    cancer_type = "GS-BRCA"

    # 1. Get common samples and labels
    common = get_common_samples(cancer_type, config)
    labels = load_labels(cancer_type, config)
    labels = labels.loc[common]

    # 2. Stratified sampling to preserve class proportions
    _, toy_ids, _, toy_labels = train_test_split(
        common,
        labels.values,
        test_size=TOY_SIZE,
        stratify=labels.values,
        random_state=42,
    )
    toy_ids = sorted(toy_ids)

    # 3. Ensure output directory exists
    toy_dir = config["paths"]["toy_data"]
    os.makedirs(toy_dir, exist_ok=True)

    # 4. Save each modality subset (transposed back to features x samples)
    cancer_short = cancer_type.split("-")[1]
    for mod_key in MODALITY_KEYS:
        df_full = load_modality(cancer_type, mod_key, config)
        df_toy = df_full.loc[toy_ids]

        # Transpose back: samples x features -> features x samples (original format)
        df_save = df_toy.T
        pattern = config["modalities"][mod_key]["file_pattern"].format(cancer=cancer_short)
        base, ext = os.path.splitext(pattern)
        out_path = os.path.join(toy_dir, f"{base}_toy{ext}")
        df_save.to_csv(out_path)
        print(f"  Saved {mod_key}: {df_toy.shape} -> {out_path}")

    # 5. Save labels
    toy_label_series = labels.loc[toy_ids]
    label_df = pd.DataFrame({"Label": toy_label_series.values})
    label_path = os.path.join(toy_dir, f"{cancer_short}_label_num_toy.csv")
    label_df.to_csv(label_path, index=False)
    print(f"  Saved labels: {len(label_df)} -> {label_path}")

    # 6. Print class distribution
    dist = toy_label_series.value_counts().sort_index()
    print(f"\nToy dataset: {TOY_SIZE} samples from {cancer_type}")
    print(f"Class distribution:")
    for cls, cnt in dist.items():
        print(f"  Class {cls}: {cnt} ({cnt/TOY_SIZE*100:.1f}%)")

    # 7. Verify roundtrip: load back with use_toy=True
    print("\n--- Verification ---")
    for mod_key in MODALITY_KEYS:
        df_check = load_modality(cancer_type, mod_key, config, use_toy=True)
        assert df_check.shape[0] == TOY_SIZE, f"{mod_key} has {df_check.shape[0]} rows"
        print(f"  {mod_key}: {df_check.shape}")

    labels_check = load_labels(cancer_type, config, use_toy=True)
    assert len(labels_check) == TOY_SIZE
    assert labels_check.nunique() == labels.nunique(), "Missing classes in toy labels"
    print(f"  labels: {labels_check.shape}, {labels_check.nunique()} classes")

    common_check = get_common_samples(cancer_type, config, use_toy=True)
    assert len(common_check) == TOY_SIZE
    print(f"  common samples: {len(common_check)}")

    print("\nAll verifications passed!")


if __name__ == "__main__":
    create_toy_dataset()
