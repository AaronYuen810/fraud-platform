from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from src.models.train_xgboost import FEATURE_COLUMNS
from src.serving.bento_service import (
    PredictRequest,
    _load_training_metadata,
    run_predict,
)


def _full_feature_payload() -> dict[str, float]:
    return {name: float(i) for i, name in enumerate(FEATURE_COLUMNS)}


def test_load_training_metadata_success(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "feature_columns": ["a", "b"],
                "threshold": 0.42,
            }
        ),
        encoding="utf-8",
    )
    columns, threshold = _load_training_metadata(path)
    assert columns == ["a", "b"]
    assert threshold == pytest.approx(0.42)


def test_load_training_metadata_coerces_column_types_to_str(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({"feature_columns": [1, 2], "threshold": 0.5}),
        encoding="utf-8",
    )
    columns, _ = _load_training_metadata(path)
    assert columns == ["1", "2"]


@pytest.mark.parametrize(
    "payload, match",
    [
        (
            {"threshold": 0.5},
            "`feature_columns` is missing or invalid",
        ),
        (
            {"feature_columns": [], "threshold": 0.5},
            "`feature_columns` is missing or invalid",
        ),
        (
            {"feature_columns": ["x"]},
            "`threshold` is missing",
        ),
        (
            {"feature_columns": ["x"], "threshold": None},
            "`threshold` is missing",
        ),
    ],
)
def test_load_training_metadata_errors(tmp_path, payload, match):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _load_training_metadata(path)


def test_predict_returns_scored_response():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.73]])

    payload = _full_feature_payload()
    req = PredictRequest.model_validate(payload)
    out = run_predict(
        list(FEATURE_COLUMNS),
        0.5,
        "test-model",
        mock_model,
        req,
    )

    assert out.fraud_score == pytest.approx(0.73)
    assert out.threshold == pytest.approx(0.5)
    assert out.flagged is True
    assert out.model_id == "test-model"
    assert out.feature_order == list(FEATURE_COLUMNS)
    mock_model.predict_proba.assert_called_once()
    call_x = mock_model.predict_proba.call_args[0][0]
    expected_row = np.array([[float(payload[name]) for name in FEATURE_COLUMNS]], dtype=float)
    np.testing.assert_array_equal(call_x, expected_row)


def test_predict_flagged_false_below_threshold():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.5, 0.1]])

    req = PredictRequest.model_validate(_full_feature_payload())
    out = run_predict(list(FEATURE_COLUMNS), 0.9, "m", mock_model, req)
    assert out.fraud_score == pytest.approx(0.1)
    assert out.flagged is False


def test_predict_missing_features_raises():
    class PartialPayload:
        def model_dump(self):
            return {"amount": 1.0}

    with pytest.raises(ValueError, match="Missing required features"):
        run_predict(FEATURE_COLUMNS, 0.5, "m", MagicMock(), PartialPayload())


def test_predict_request_accepts_full_valid_payload():
    data = _full_feature_payload()
    req = PredictRequest.model_validate(data)
    assert req.model_dump() == data


def test_predict_request_forbids_extra_keys():
    data = _full_feature_payload()
    data["unknown_feature"] = 0.0
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(data)
