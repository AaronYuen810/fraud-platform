"""Resolve the Feast repo path and `FeatureStore` instance."""

from __future__ import annotations

import os
from pathlib import Path

from feast import FeatureStore


def feast_repo_path() -> Path:
    """Directory containing `feature_store.yaml` (override with `FEAST_FEATURE_REPO_PATH`)."""
    override = os.environ.get("FEAST_FEATURE_REPO_PATH")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent.parent.parent / "feature_repo"


def get_feature_store() -> FeatureStore:
    return FeatureStore(repo_path=str(feast_repo_path()))
