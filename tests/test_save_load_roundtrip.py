"""T1.7 — Model checkpoint save/load round-trip test.

Trains IntermediateFusionModel for 1 epoch on toy GS-BRCA data, saves the
checkpoint in the same format used by train_fusion.py, reloads it from disk,
and asserts that predictions on a fixed batch are identical within floating-
point tolerance.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import IntermediateFusionModel, MultiOmicsDataset
from src.preprocessing import load_cv_folds, prepare_fold_data
from src.utils import load_config, set_seeds


@pytest.fixture(scope="module")
def fold_data():
    set_seeds(42)
    config = load_config()
    fold_info = load_cv_folds("GS-BRCA", config, use_toy=True)["folds"][0]
    return prepare_fold_data("GS-BRCA", fold_info, config, use_toy=True)


def _make_model(modality_dims, num_classes):
    return IntermediateFusionModel(
        modality_dims=modality_dims,
        latent_dim=16,
        hidden_dim=32,
        num_classes=num_classes,
        dropout=0.0,
        encoder_hidden=32,
    )


def test_fusion_save_load_preserves_predictions(fold_data):
    set_seeds(42)
    X_train = fold_data["X_train"]
    y_train = fold_data["y_train"]

    modality_dims = {k: v.shape[1] for k, v in X_train.items()}
    num_classes = int(y_train.max()) + 1

    model = _make_model(modality_dims, num_classes)
    model.train(True)

    # One mini-training step so weights differ from init
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = MultiOmicsDataset(
        {k: v.values.astype(np.float32) for k, v in X_train.items()},
        y_train.astype(np.int64),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        break  # one step only

    # Capture predictions before saving (inference mode)
    model.train(False)
    with torch.no_grad():
        fixed_batch = next(iter(loader))
        batch_x_fixed = fixed_batch[0]
        preds_before = model(batch_x_fixed).numpy()

    # Save checkpoint (same format as train_fusion.py)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "fusion_brca_fold0.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "modality_dims": modality_dims,
                "latent_dim": 16,
                "num_classes": num_classes,
                "encoder_hidden": 32,
                "classifier_hidden": 32,
                "dropout": 0.0,
            },
            ckpt_path,
        )

        # Reload checkpoint
        ckpt = torch.load(ckpt_path, weights_only=True)
        model2 = _make_model(ckpt["modality_dims"], ckpt["num_classes"])
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.train(False)

        with torch.no_grad():
            preds_after = model2(batch_x_fixed).numpy()

    assert np.allclose(preds_before, preds_after, atol=1e-6), (
        f"Predictions differ after save/load: max diff = {np.abs(preds_before - preds_after).max():.2e}"
    )
