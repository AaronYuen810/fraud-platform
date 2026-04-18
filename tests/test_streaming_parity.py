"""Parity: streaming FeatureState (compute → update) matches batch `build_features`."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import FeatureState, build_features
from src.feature_store.reader import get_features
from src.models.train_xgboost import FEATURE_COLUMNS
from src.streaming import (
    TransactionEvent,
    ensure_worker_started,
    enqueue_transaction,
    wait_for_queue_drain,
)
from src.streaming.worker import reset_streaming_pipeline_for_tests


@pytest.fixture(autouse=True)
def _reset_streaming_state_between_tests() -> None:
    reset_streaming_pipeline_for_tests()
    yield


def _synthetic_transactions() -> pd.DataFrame:
    """Small ordered dataset with no self-transfers; account ids use A{n} for `sender_idx`."""
    base = pd.Timestamp("2024-03-15 09:00:00")
    rows: list[dict[str, object]] = []
    # Strictly increasing timestamps; mixed senders/beneficiaries to exercise windows.
    spec: list[tuple[str, pd.Timedelta, float, str, str]] = [
        ("t1", pd.Timedelta(minutes=0), 120.5, "A001", "A002"),
        ("t2", pd.Timedelta(minutes=5), 80.0, "A001", "A003"),
        ("t3", pd.Timedelta(minutes=12), 200.0, "A002", "A001"),
        ("t4", pd.Timedelta(minutes=20), 50.0, "A001", "A002"),
        ("t5", pd.Timedelta(minutes=45), 99.9, "A003", "A002"),
        ("t6", pd.Timedelta(hours=2), 300.0, "A001", "A002"),
        ("t7", pd.Timedelta(hours=2, minutes=30), 10.0, "A002", "A003"),
        ("t8", pd.Timedelta(hours=5), 77.7, "A001", "A004"),
    ]
    for tid, delta, amount, sender, ben in spec:
        ts = base + delta
        rows.append(
            {
                "transaction_id": tid,
                "timestamp": ts,
                "amount": amount,
                "sender_account": sender,
                "beneficiary_account": ben,
                "is_fraud_label": 0,
                "is_fraud_ground_truth": 0,
            }
        )
    return pd.DataFrame(rows)


def test_streaming_feature_state_matches_build_features_batch() -> None:
    df = _synthetic_transactions()
    assert df["timestamp"].is_monotonic_increasing
    assert not (df["sender_account"] == df["beneficiary_account"]).any()

    batch = build_features(df)

    state = FeatureState()
    streamed: list[dict[str, float]] = []
    for row in df.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        amount = float(row.amount)
        sender = str(row.sender_account)
        beneficiary = str(row.beneficiary_account)
        feats = state.compute_row_features(ts, amount, sender, beneficiary)
        streamed.append(feats)
        state.update(ts, amount, sender, beneficiary)

    assert len(streamed) == len(batch)
    for i, feat_dict in enumerate(streamed):
        for col in FEATURE_COLUMNS:
            assert feat_dict[col] == pytest.approx(float(batch[col].iloc[i]))


def test_streaming_worker_queue_matches_batch_via_feast() -> None:
    """Worker thread + Feast push/read matches batch `build_features` (same event order)."""
    df = _synthetic_transactions()
    batch = build_features(df)

    ensure_worker_started()
    for row in df.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        enqueue_transaction(
            TransactionEvent(
                transaction_id=str(row.transaction_id),
                timestamp=ts.to_pydatetime(),
                amount=float(row.amount),
                sender_account=str(row.sender_account),
                beneficiary_account=str(row.beneficiary_account),
            )
        )
    wait_for_queue_drain()

    for i, row in enumerate(df.itertuples(index=False)):
        tid = str(row.transaction_id)
        served = get_features(tid)
        for col in FEATURE_COLUMNS:
            assert served[col] == pytest.approx(float(batch[col].iloc[i]))
