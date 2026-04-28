"""Tests for model components and output shapes."""

import numpy as np
import torch

from src.models import (
    IntermediateFusionModel,
    ModalityEncoder,
    MultiOmicsDataset,
    PathwayAttentionEncoder,
    PathwayAwareFusionModel,
)


def test_modality_encoder_output_shape() -> None:
    encoder = ModalityEncoder(input_dim=32, hidden_dim=16, latent_dim=8, dropout=0.0)
    inputs = torch.randn(4, 32)
    outputs = encoder(inputs)

    assert outputs.shape == (4, 8)


def test_intermediate_fusion_forward_and_latent_shapes() -> None:
    modality_dims = {"mrna": 20, "mirna": 10, "methy": 12, "cnv": 8}
    model = IntermediateFusionModel(
        modality_dims=modality_dims,
        latent_dim=6,
        hidden_dim=10,
        num_classes=5,
        dropout=0.0,
        encoder_hidden=16,
    )

    batch = {name: torch.randn(3, dim) for name, dim in modality_dims.items()}

    logits = model(batch)
    latent = model.get_latent(batch)

    assert logits.shape == (3, 5)
    assert latent.shape == (3, 24)


def test_pathway_attention_encoder_produces_normalized_attention() -> None:
    pathway_indices = {"pw_a": [0, 1, 2], "pw_b": [3, 4]}
    unmapped_indices = [5, 6, 7]
    encoder = PathwayAttentionEncoder(
        pathway_indices=pathway_indices,
        unmapped_indices=unmapped_indices,
        latent_dim=8,
        attn_hidden=4,
        dropout=0.0,
    )

    x = torch.randn(5, 8)
    out = encoder(x)
    attn = encoder.get_attention_weights()

    assert out.shape == (5, 8)
    assert attn.shape == (5, 2)
    assert torch.allclose(attn.sum(dim=1), torch.ones(5), atol=1e-6)


def test_pathway_aware_fusion_forward_shape() -> None:
    modality_dims = {"mrna": 8, "mirna": 4, "methy": 6, "cnv": 5}
    pathway_indices = {"pw_a": [0, 1, 2], "pw_b": [3, 4]}
    unmapped_indices = [5, 6, 7]

    model = PathwayAwareFusionModel(
        modality_dims=modality_dims,
        pathway_indices=pathway_indices,
        unmapped_indices=unmapped_indices,
        latent_dim=4,
        hidden_dim=6,
        num_classes=3,
        dropout=0.0,
        encoder_hidden=12,
        attn_hidden=4,
    )

    batch = {name: torch.randn(2, dim) for name, dim in modality_dims.items()}

    logits = model(batch)
    attn = model.get_pathway_attention()

    assert logits.shape == (2, 3)
    assert attn is not None
    assert attn.shape[0] == 2


def test_multiomics_dataset_returns_modality_dict_and_label() -> None:
    arrays = {
        "mrna": np.random.randn(6, 5).astype(np.float32),
        "mirna": np.random.randn(6, 3).astype(np.float32),
    }
    labels = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)

    dataset = MultiOmicsDataset(arrays, labels)
    x, y = dataset[0]

    assert len(dataset) == 6
    assert set(x.keys()) == {"mrna", "mirna"}
    assert x["mrna"].shape[0] == 5
    assert int(y.item()) in {0, 1, 2}
