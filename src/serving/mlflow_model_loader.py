"""Resolve a Model Registry URI into a loaded XGBoost model and metrics path for serving."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion


def parse_models_uri(model_uri: str) -> tuple[str, str | None, int | None]:
    """
    Parse ``models:/name@alias`` or ``models:/name/version`` into components.

    Returns (name, alias or None, version or None).
    """
    if not model_uri.startswith("models:/"):
        raise ValueError(
            f"Expected a models:/ URI, got: {model_uri!r}. "
            "Example: models:/fraud_xgb_classifier@champion"
        )
    rest = model_uri[len("models:/") :]
    if "@" in rest:
        name, alias = rest.split("@", 1)
        if not name or not alias:
            raise ValueError(f"Invalid models:/ URI: {model_uri!r}")
        return name, alias, None
    parts = rest.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Expected models:/<name>/<version> or models:/<name>@<alias>, got: {model_uri!r}"
        )
    name, ver = parts[0], parts[1]
    if not name:
        raise ValueError(f"Invalid models:/ URI: {model_uri!r}")
    try:
        version = int(ver)
    except ValueError as exc:
        raise ValueError(
            f"Model version must be an integer, got {ver!r} in {model_uri!r}"
        ) from exc
    return name, None, version


def get_model_version(client: MlflowClient, model_uri: str) -> ModelVersion:
    """Return the :class:`ModelVersion` for a ``models:/`` URI."""
    name, alias, version = parse_models_uri(model_uri)
    if alias is not None:
        return client.get_model_version_by_alias(name, alias)
    assert version is not None
    return client.get_model_version(name, str(version))


def load_bundle_from_registry(
    *,
    tracking_uri: str | None,
    model_uri: str,
    cache_dir: Path | None = None,
) -> tuple[object, Path, str]:
    """
    Load an XGBoost model and training metrics from a Model Registry URI.

    ``model_uri`` uses ``models:/RegisteredModelName@alias`` or ``models:/Name/version``.
    Metrics are read from the training run artifact ``training_metrics.json``.

    Returns ``(xgboost_model, metrics_path, model_id)`` where ``model_id`` is ``name:vN``.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri)
    mv = get_model_version(client, model_uri)
    model_id = f"{mv.name}:v{mv.version}"

    dst_root = cache_dir or Path(tempfile.mkdtemp(prefix="mlflow_serve_"))
    dst_root.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(
        mlflow.artifacts.download_artifacts(
            run_id=mv.run_id,
            artifact_path="training_metrics.json",
            dst_path=str(dst_root),
            tracking_uri=tracking_uri,
        )
    )

    xgb_model = mlflow.xgboost.load_model(model_uri)
    return xgb_model, metrics_path, model_id


def env_load_bundle() -> tuple[object, Path, str] | None:
    """
    If ``MLFLOW_MODEL_URI`` is set, load from the registry; otherwise return None.

    Uses ``MLFLOW_TRACKING_URI`` and optional ``MLFLOW_MODEL_CACHE_DIR``.
    """
    model_uri = os.getenv("MLFLOW_MODEL_URI")
    if not model_uri:
        return None
    cache_raw = os.getenv("MLFLOW_MODEL_CACHE_DIR")
    cache_dir = Path(cache_raw).expanduser() if cache_raw else None
    return load_bundle_from_registry(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        model_uri=model_uri,
        cache_dir=cache_dir,
    )
