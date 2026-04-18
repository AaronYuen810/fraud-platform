"""Versioned HTTP contracts for fraud scoring (raw transactions vs precomputed features)."""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RawTransactionRequest(BaseModel):
    """Payload for `POST /v1/transactions:score` — mirrors raw CSV columns used in feature building."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = Field(
        default=None,
        description="Optional idempotency / correlation id (string or UUID).",
    )
    timestamp: datetime = Field(..., description="Transaction time (ISO 8601).")
    amount: float = Field(..., description="Transaction amount.")
    sender_account: str = Field(..., min_length=1, description="Sender account id (e.g. A001).")
    beneficiary_account: str = Field(
        ...,
        min_length=1,
        description="Beneficiary account id (e.g. A002).",
    )

    @model_validator(mode="after")
    def reject_self_transfer(self) -> Self:
        if self.sender_account == self.beneficiary_account:
            raise ValueError("Input transaction is a self-transfer; scoring is not supported.")
        return self


class ScoreResponse(BaseModel):
    """Scoring result for a raw transaction; aligns with `PredictResponse` plus optional correlation id."""

    fraud_score: float
    threshold: float
    flagged: bool
    model_id: str
    feature_order: list[str]
    transaction_id: str | None = None
