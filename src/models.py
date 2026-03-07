"""
MLOmics Fusion Models
=====================
Per-modality encoders → latent concatenation → MLP classifier.

Classes:
    ModalityEncoder            — Dense encoder for a single modality
    FusionClassifier           — MLP head over concatenated latent vectors
    IntermediateFusionModel    — Full pipeline (encoders + classifier)
    EarlyFusionMLP             — Ablation baseline: concatenated features → MLP
    PathwayAttentionEncoder    — Dual-path mRNA encoder with KEGG pathway attention
    PathwayAwareFusionModel    — Fusion model using pathway attention for mRNA
    MultiOmicsDataset          — PyTorch Dataset for dict-of-modalities input
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from collections import OrderedDict


class ModalityEncoder(nn.Module):
    """Encodes a single modality into a latent representation.

    Architecture: Linear → BatchNorm → ReLU → Dropout → Linear → BatchNorm → ReLU
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, input_dim) → (batch, latent_dim)."""
        return self.encoder(x)


class FusionClassifier(nn.Module):
    """MLP head that takes concatenated latent vectors and predicts class.

    Architecture: Linear → ReLU → Dropout → Linear (logits)
    """

    def __init__(self, total_latent_dim: int, hidden_dim: int = 128,
                 num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(total_latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, total_latent_dim) → (batch, num_classes) logits."""
        return self.classifier(x)


class IntermediateFusionModel(nn.Module):
    """Full intermediate fusion pipeline.

    1. Per-modality dense encoders produce latent vectors.
    2. Latent vectors are concatenated.
    3. An MLP classifier predicts class logits.

    Args:
        modality_dims: dict mapping modality name → input feature count,
                       e.g. {'mrna': 5000, 'mirna': 200, 'methy': 5000, 'cnv': 5000}.
                       Iteration order is preserved (Python 3.7+).
        latent_dim:    Latent dimension per encoder (default 64).
        hidden_dim:    Hidden dimension in encoder and classifier (default encoder=256, classifier=128).
        num_classes:   Number of output classes.
        dropout:       Dropout rate throughout the model.
        encoder_hidden: Hidden dimension for the per-modality encoder layers.
    """

    def __init__(self, modality_dims: dict, latent_dim: int = 64,
                 hidden_dim: int = 128, num_classes: int = 5,
                 dropout: float = 0.3, encoder_hidden: int = 256):
        super().__init__()
        self.modality_order = list(modality_dims.keys())

        # One encoder per modality (ModuleDict for parameter registration)
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(
                input_dim=dim,
                hidden_dim=encoder_hidden,
                latent_dim=latent_dim,
                dropout=dropout,
            )
            for name, dim in modality_dims.items()
        })

        total_latent = latent_dim * len(modality_dims)
        self.classifier = FusionClassifier(
            total_latent_dim=total_latent,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x_dict: dict) -> torch.Tensor:
        """Forward pass.

        Args:
            x_dict: {modality_name: tensor(batch, features)}

        Returns:
            Logits of shape (batch, num_classes).
        """
        latents = [self.encoders[name](x_dict[name]) for name in self.modality_order]
        fused = torch.cat(latents, dim=1)
        return self.classifier(fused)

    def get_latent(self, x_dict: dict) -> torch.Tensor:
        """Extract the concatenated latent representation (for visualization).

        Returns:
            (batch, latent_dim * n_modalities)
        """
        latents = [self.encoders[name](x_dict[name]) for name in self.modality_order]
        return torch.cat(latents, dim=1)


class EarlyFusionMLP(nn.Module):
    """Early fusion baseline: concatenated features → single MLP.

    Architecture: Linear(input_dim, 256) → ReLU → Dropout
                  → Linear(256, 128) → ReLU → Dropout
                  → Linear(128, num_classes)

    Args:
        input_dim:   Total concatenated feature count (compute dynamically from data).
        num_classes: Number of output classes.
        dropout:     Dropout rate.
    """

    def __init__(self, input_dim: int, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, input_dim) → (batch, num_classes) logits."""
        return self.mlp(x)


class PathwayAttentionEncoder(nn.Module):
    """Dual-path mRNA encoder with KEGG pathway attention.

    Path A (mapped genes): Raw mRNA features are grouped by KEGG pathway
    membership, mean-pooled per pathway, and passed through a learnable
    attention mechanism that produces interpretable per-pathway weights.

    Path B (unmapped genes): Features not in any KEGG pathway are processed
    by a standard dense encoder.

    Both paths are concatenated and projected to produce a single latent vector
    of dimension `latent_dim`, matching the output of ModalityEncoder.

    Args:
        pathway_indices: dict mapping pathway name → list of feature indices
                         in the raw mRNA feature vector.
        unmapped_indices: list of feature indices not covered by any pathway.
        latent_dim:      Output latent dimension (must match other encoders).
        attn_hidden:     Hidden dimension in the attention scoring network.
        dropout:         Dropout rate.
    """

    def __init__(self, pathway_indices: dict, unmapped_indices: list,
                 latent_dim: int = 64, attn_hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.latent_dim = latent_dim

        pw_names = sorted(pathway_indices.keys())
        self.n_pathways = len(pw_names)
        self.pathway_names = pw_names

        # Store pathway masks as a list of LongTensors for index_select
        self._pathway_idx = nn.ParameterList()  # not trained, just stored
        for name in pw_names:
            idx_tensor = torch.tensor(pathway_indices[name], dtype=torch.long)
            self.register_buffer(f"pw_{pw_names.index(name)}", idx_tensor)

        self.unmapped_idx = torch.tensor(unmapped_indices, dtype=torch.long)
        self.register_buffer("_unmapped_idx", self.unmapped_idx)
        self.n_unmapped = len(unmapped_indices)

        # Path A: Attention over pathways
        # Input: (batch, n_pathways) of mean-pooled pathway activations
        self.attn_net = nn.Sequential(
            nn.Linear(self.n_pathways, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, self.n_pathways),
        )
        self.pathway_proj = nn.Sequential(
            nn.Linear(self.n_pathways, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

        # Path B: Standard encoder for unmapped features
        unmapped_hidden = min(256, max(64, self.n_unmapped // 4))
        self.unmapped_encoder = nn.Sequential(
            nn.Linear(self.n_unmapped, unmapped_hidden),
            nn.BatchNorm1d(unmapped_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(unmapped_hidden, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

        # Combine both paths → single latent_dim output
        self.combiner = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self._last_attention_weights = None

    def _get_pathway_buffer(self, idx: int) -> torch.Tensor:
        return getattr(self, f"pw_{idx}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, n_mrna_features) → (batch, latent_dim)."""
        batch_size = x.size(0)

        # Path A: Mean-pool features per pathway → (batch, n_pathways)
        pathway_means = torch.zeros(batch_size, self.n_pathways, device=x.device)
        for i in range(self.n_pathways):
            idx = self._get_pathway_buffer(i)
            pw_features = x[:, idx]              # (batch, pw_size)
            pathway_means[:, i] = pw_features.mean(dim=1)

        # Attention: learnable scoring → softmax → weights
        attn_logits = self.attn_net(pathway_means)      # (batch, n_pathways)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (batch, n_pathways)
        self._last_attention_weights = attn_weights.detach()

        weighted_pathways = attn_weights * pathway_means  # (batch, n_pathways)
        latent_pathway = self.pathway_proj(weighted_pathways)  # (batch, latent_dim)

        # Path B: Unmapped features → standard encoder
        unmapped_feats = x[:, self._unmapped_idx]  # (batch, n_unmapped)
        latent_unmapped = self.unmapped_encoder(unmapped_feats)  # (batch, latent_dim)

        # Combine
        combined = torch.cat([latent_pathway, latent_unmapped], dim=1)
        return self.combiner(combined)  # (batch, latent_dim)

    def get_attention_weights(self) -> torch.Tensor:
        """Return the attention weights from the last forward pass.

        Returns:
            (batch, n_pathways) tensor of attention weights (sum to 1 per sample).
        """
        return self._last_attention_weights


class PathwayAwareFusionModel(nn.Module):
    """Fusion model with pathway attention for mRNA, standard encoders for others.

    The mRNA modality uses a PathwayAttentionEncoder that applies KEGG
    pathway-level attention to raw input features (not latent space).
    All other modalities use standard ModalityEncoders. Concatenated
    latents are passed through a FusionClassifier.

    Args:
        modality_dims: dict mapping modality name → input feature count.
        pathway_indices: dict mapping pathway name → list of mRNA feature indices.
        unmapped_indices: list of mRNA feature indices not in any pathway.
        latent_dim:    Latent dimension per encoder.
        hidden_dim:    Hidden dimension for the fusion classifier.
        num_classes:   Number of output classes.
        dropout:       Dropout rate.
        encoder_hidden: Hidden dimension for standard modality encoders.
        attn_hidden:   Hidden dimension for the pathway attention network.
    """

    def __init__(self, modality_dims: dict, pathway_indices: dict,
                 unmapped_indices: list, latent_dim: int = 64,
                 hidden_dim: int = 128, num_classes: int = 5,
                 dropout: float = 0.3, encoder_hidden: int = 256,
                 attn_hidden: int = 64):
        super().__init__()
        self.modality_order = list(modality_dims.keys())

        self.encoders = nn.ModuleDict()
        for name, dim in modality_dims.items():
            if name == "mrna":
                self.encoders[name] = PathwayAttentionEncoder(
                    pathway_indices=pathway_indices,
                    unmapped_indices=unmapped_indices,
                    latent_dim=latent_dim,
                    attn_hidden=attn_hidden,
                    dropout=dropout,
                )
            else:
                self.encoders[name] = ModalityEncoder(
                    input_dim=dim,
                    hidden_dim=encoder_hidden,
                    latent_dim=latent_dim,
                    dropout=dropout,
                )

        total_latent = latent_dim * len(modality_dims)
        self.classifier = FusionClassifier(
            total_latent_dim=total_latent,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x_dict: dict) -> torch.Tensor:
        """Forward pass.

        Args:
            x_dict: {modality_name: tensor(batch, features)}

        Returns:
            Logits of shape (batch, num_classes).
        """
        latents = [self.encoders[name](x_dict[name]) for name in self.modality_order]
        fused = torch.cat(latents, dim=1)
        return self.classifier(fused)

    def get_latent(self, x_dict: dict) -> torch.Tensor:
        """Extract concatenated latent representation."""
        latents = [self.encoders[name](x_dict[name]) for name in self.modality_order]
        return torch.cat(latents, dim=1)

    def get_pathway_attention(self) -> torch.Tensor:
        """Return mRNA pathway attention weights from last forward pass."""
        return self.encoders["mrna"].get_attention_weights()

    def get_pathway_names(self) -> list:
        """Return ordered list of pathway names."""
        return self.encoders["mrna"].pathway_names


class MultiOmicsDataset(Dataset):
    """PyTorch Dataset for multi-omics dict input.

    Args:
        data_dict: {modality_name: numpy array of shape (n_samples, n_features)}
        labels:    numpy array of shape (n_samples,) with integer class labels
    """

    def __init__(self, data_dict: dict, labels: np.ndarray):
        self.modalities = {}
        for name, arr in data_dict.items():
            self.modalities[name] = torch.tensor(arr, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self._n = len(labels)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        x = {name: tensor[idx] for name, tensor in self.modalities.items()}
        return x, self.labels[idx]
