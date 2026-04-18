"""Integration: Feast write → read → XGBoost predict (minimal end-to-end)."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from xgboost import XGBClassifier

from src.feature_store.reader import get_features
from src.feature_store.writer import write_features
from src.models.train_xgboost import FEATURE_COLUMNS
from src.serving.bento_service import PredictRequest, run_predict


def test_write_read_roundtrip_matches_values() -> None:
    tid = f"feast-it-{uuid.uuid4().hex}"
    feat = {name: float(i + 1) for i, name in enumerate(FEATURE_COLUMNS)}
    ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    write_features(tid, feat, ts)
    out = get_features(tid)
    for name in FEATURE_COLUMNS:
        assert out[name] == pytest.approx(feat[name])


def test_write_read_predict(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, len(FEATURE_COLUMNS)))
    y = rng.integers(0, 2, size=40)
    clf = XGBClassifier(
        n_estimators=4,
        max_depth=2,
        objective="binary:logistic",
        random_state=0,
    )
    clf.fit(X, y)
    model_path = tmp_path / "model.json"
    clf.save_model(str(model_path))
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"feature_columns": list(FEATURE_COLUMNS), "threshold": 0.42}),
        encoding="utf-8",
    )
    keys = ("MODEL_PATH", "METRICS_PATH", "MODEL_ID")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["METRICS_PATH"] = str(metrics_path)
    os.environ["MODEL_ID"] = "feast-integration-model"
    try:
        tid = f"predict-{uuid.uuid4().hex}"
        feat = {name: float(i) for i, name in enumerate(FEATURE_COLUMNS)}
        ts = datetime(2024, 7, 1, 9, 30, 0, tzinfo=timezone.utc)
        write_features(tid, feat, ts)
        served = get_features(tid)
        req = PredictRequest(**served)
        pred = run_predict(
            list(FEATURE_COLUMNS),
            0.42,
            "feast-integration-model",
            clf,
            req,
        )
        assert 0.0 <= pred.fraud_score <= 1.0
    finally:
        for key in keys:
            val = previous[key]
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
