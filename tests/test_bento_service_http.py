"""HTTP integration tests for the BentoML FraudScoringService (Starlette TestClient + ASGI)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from starlette.testclient import TestClient
from xgboost import XGBClassifier

from src.models.train_xgboost import FEATURE_COLUMNS


def _full_feature_json() -> dict[str, float]:
    return {name: float(i) for i, name in enumerate(FEATURE_COLUMNS)}


@pytest.fixture(scope="module")
def bento_test_client(tmp_path_factory):
    """One ASGI app per module — BentoML registers Prometheus metrics once per process."""
    tmp_path = tmp_path_factory.mktemp("bento_http")
    rng = np.random.default_rng(0)
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
    os.environ["MODEL_ID"] = "integration-test-model"

    from src.serving.bento_service import FraudScoringService

    app = FraudScoringService.to_asgi()
    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client
    finally:
        for key in keys:
            val = previous[key]
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def test_http_healthz(bento_test_client: TestClient):
    r = bento_test_client.get("/healthz")
    assert r.status_code == 200


def test_http_predict_returns_scoring_payload(bento_test_client: TestClient):
    body = _full_feature_json()
    r = bento_test_client.post("/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "fraud_score" in data
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["threshold"] == pytest.approx(0.42)
    assert isinstance(data["flagged"], bool)
    assert data["model_id"] == "integration-test-model"
    assert data["feature_order"] == list(FEATURE_COLUMNS)
    assert data["flagged"] == (data["fraud_score"] >= data["threshold"])


def test_http_predict_validation_error_on_incomplete_body(bento_test_client: TestClient):
    r = bento_test_client.post("/predict", json={"amount": 1.0})
    assert r.status_code == 400
    err = r.json()
    assert "error" in err


def _valid_raw_transaction_body(**overrides: object) -> dict:
    base = {
        "timestamp": "2024-06-01T12:00:00Z",
        "amount": 250.5,
        "sender_account": "A001",
        "beneficiary_account": "A002",
    }
    base.update(overrides)
    return base


def test_http_transactions_score_returns_scoring_payload(bento_test_client: TestClient):
    body = _valid_raw_transaction_body(transaction_id="corr-tx-001")
    r = bento_test_client.post("/v1/transactions:score", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "fraud_score" in data
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["threshold"] == pytest.approx(0.42)
    assert isinstance(data["flagged"], bool)
    assert data["model_id"] == "integration-test-model"
    assert data["feature_order"] == list(FEATURE_COLUMNS)
    assert data["transaction_id"] == "corr-tx-001"
    assert data["flagged"] == (data["fraud_score"] >= data["threshold"])

    r2 = bento_test_client.post("/v1/transactions:score", json=_valid_raw_transaction_body())
    assert r2.status_code == 200
    assert r2.json().get("transaction_id") is None


def test_http_transactions_score_validation_error_on_incomplete_body(bento_test_client: TestClient):
    r = bento_test_client.post("/v1/transactions:score", json={"amount": 1.0})
    assert r.status_code == 400
    err = r.json()
    assert "error" in err


def test_http_transactions_score_rejects_self_transfer(bento_test_client: TestClient):
    body = _valid_raw_transaction_body(sender_account="A001", beneficiary_account="A001")
    r = bento_test_client.post("/v1/transactions:score", json=body)
    assert r.status_code == 400
    err = r.json()
    assert "error" in err


def test_http_openapi_spec_available(bento_test_client: TestClient):
    r = bento_test_client.get("/docs.json")
    assert r.status_code == 200
