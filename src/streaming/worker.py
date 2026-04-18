"""Single-consumer worker: FeatureState compute → Feast write → state update."""

from __future__ import annotations

import threading

import pandas as pd

from src.feature_store.writer import write_features
from src.features.build_features import FeatureState
from src.streaming.ingestion import TransactionEvent

_feature_state = FeatureState()
_state_lock = threading.Lock()


def get_feature_state() -> FeatureState:
    """Return the process-global streaming FeatureState (single consumer)."""
    return _feature_state


def run_worker_loop() -> None:
    """Block forever: consume `TransactionEvent`s FIFO and process each one."""
    from src.streaming.ingestion import get_transaction_queue

    q = get_transaction_queue()
    while True:
        event: TransactionEvent = q.get()
        try:
            with _state_lock:
                _process_event(event)
        finally:
            q.task_done()


def _process_event(event: TransactionEvent) -> None:
    ts = pd.to_datetime(event.timestamp, utc=False)
    pd_ts = pd.Timestamp(ts)
    sender = str(event.sender_account)
    beneficiary = str(event.beneficiary_account)
    amount = float(event.amount)

    features = _feature_state.compute_row_features(pd_ts, amount, sender, beneficiary)
    write_features(
        str(event.transaction_id),
        features,
        pd_ts.to_pydatetime(),
    )
    _feature_state.update(pd_ts, amount, sender, beneficiary)


def reset_streaming_pipeline_for_tests() -> None:
    """Replace `FeatureState` with a fresh instance and drop queued events.

    Call this only when no concurrent enqueues are in flight (typical in tests).
    Waits for the worker to finish all pending work (`wait_for_queue_drain`), so
    the worker is not between ``queue.get()`` and state update; then drops any
    stragglers and resets in-memory state.
    """
    global _feature_state
    from src.streaming.ingestion import (
        drain_pending_transactions,
        ensure_worker_started,
        wait_for_queue_drain,
    )

    ensure_worker_started()
    wait_for_queue_drain()
    with _state_lock:
        drain_pending_transactions()
        _feature_state = FeatureState()
