from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model
from xgboost import XGBClassifier

from src.feature_store.reader import wait_for_features
from src.models.train_xgboost import FEATURE_COLUMNS
from src.serving.contracts import RawTransactionRequest, ScoreResponse
from src.serving.mlflow_model_loader import env_load_bundle
from src.streaming import TransactionEvent, ensure_worker_started, enqueue_transaction


DEFAULT_MODEL_PATH = "models/xgboost_fraud_model.json"
DEFAULT_METRICS_PATH = "models/training_metrics.json"

PredictRequest = create_model(
    "PredictRequest",
    __base__=BaseModel,
    __config__=ConfigDict(extra="forbid"),
    **{
        name: (float, Field(..., description=f"Training feature `{name}`."))
        for name in FEATURE_COLUMNS
    },
)


class PredictResponse(BaseModel):
    fraud_score: float
    threshold: float
    flagged: bool
    model_id: str
    feature_order: list[str]


def _load_training_metadata(metrics_path: Path) -> tuple[list[str], float]:
    """Read feature order and threshold from training metadata JSON."""
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    feature_columns = payload.get("feature_columns", [])
    threshold = payload.get("threshold")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("`feature_columns` is missing or invalid in metrics file.")
    if threshold is None:
        raise ValueError("`threshold` is missing in metrics file.")

    return [str(column) for column in feature_columns], float(threshold)


def run_predict(
    feature_columns: list[str],
    threshold: float,
    model_id: str,
    model: Any,
    request: Any,
) -> PredictResponse:
    """Score one feature payload; shared by the Bento API and unit tests."""
    payload = request.model_dump()
    missing = [name for name in feature_columns if name not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    ordered = np.array(
        [[float(payload[name]) for name in feature_columns]],
        dtype=float,
    )
    fraud_score = float(model.predict_proba(ordered)[0, 1])
    flagged = bool(fraud_score >= threshold)

    return PredictResponse(
        fraud_score=fraud_score,
        threshold=threshold,
        flagged=flagged,
        model_id=model_id,
        feature_order=feature_columns,
    )


@bentoml.service(name="fraud_scoring_service")
class FraudScoringService:
    def __init__(self) -> None:
        """Initialize model paths, metadata, and the loaded XGBoost model."""
        bundle = env_load_bundle()
        if bundle is not None:
            self.model, self.metrics_path, self.model_id = bundle
        else:
            model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
            self.metrics_path = Path(os.getenv("METRICS_PATH", DEFAULT_METRICS_PATH))
            self.model_id = os.getenv("MODEL_ID", model_path.name)
            self.model = XGBClassifier()
            self.model.load_model(str(model_path))

        self.feature_columns, self.threshold = _load_training_metadata(self.metrics_path)
        ensure_worker_started()

    @bentoml.api(route="/predict", input_spec=PredictRequest)
    def predict(self, **kwargs: Any) -> PredictResponse:
        """Score one feature payload and return fraud decision fields."""
        request = PredictRequest(**kwargs)
        return run_predict(
            self.feature_columns,
            self.threshold,
            self.model_id,
            self.model,
            request,
        )

    @bentoml.api(route="/v1/transactions:score", input_spec=RawTransactionRequest)
    def score_transaction(self, **kwargs: Any) -> ScoreResponse:
        """Enqueue raw transaction for streaming feature computation; read from Feast, then score."""
        raw = RawTransactionRequest(**kwargs)
        transaction_id = raw.transaction_id or str(uuid.uuid4())
        enqueue_transaction(
            TransactionEvent(
                transaction_id=transaction_id,
                timestamp=raw.timestamp,
                amount=raw.amount,
                sender_account=raw.sender_account,
                beneficiary_account=raw.beneficiary_account,
            )
        )
        served_features = wait_for_features(transaction_id)
        predict_request = PredictRequest(**served_features)
        pred = run_predict(
            self.feature_columns,
            self.threshold,
            self.model_id,
            self.model,
            predict_request,
        )
        return ScoreResponse(
            fraud_score=pred.fraud_score,
            threshold=pred.threshold,
            flagged=pred.flagged,
            model_id=pred.model_id,
            feature_order=pred.feature_order,
            transaction_id=transaction_id,
        )

