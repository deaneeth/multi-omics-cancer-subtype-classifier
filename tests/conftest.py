"""Pytest fixtures for project-level tests."""

import pytest

from src.preprocessing import load_cv_folds
from src.utils import load_config, set_seeds


@pytest.fixture(scope="session", autouse=True)
def _seed_everything() -> None:
    set_seeds(42)


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def toy_brca_fold(config: dict) -> dict:
    return load_cv_folds("GS-BRCA", config, use_toy=True)["folds"][0]
