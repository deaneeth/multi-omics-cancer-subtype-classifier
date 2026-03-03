"""
MLOmics Intermediate Fusion Model
===================================
Per-modality encoders → latent concatenation → MLP classifier.

Classes:
    ModalityEncoder       — Dense encoder for a single modality
    FusionClassifier      — MLP head over concatenated latent vectors
    IntermediateFusionModel — Full pipeline (encoders + classifier)
    MultiOmicsDataset     — PyTorch Dataset for dict-of-modalities input
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
