"""In-memory streaming pipeline: one worker, FIFO queue, FeatureState parity with training."""

from src.streaming.ingestion import (
    TransactionEvent,
    drain_pending_transactions,
    enqueue_transaction,
    ensure_worker_started,
    get_transaction_queue,
    wait_for_queue_drain,
)
from src.streaming.worker import get_feature_state, reset_streaming_pipeline_for_tests

__all__ = [
    "TransactionEvent",
    "drain_pending_transactions",
    "enqueue_transaction",
    "ensure_worker_started",
    "get_feature_state",
    "get_transaction_queue",
    "reset_streaming_pipeline_for_tests",
    "wait_for_queue_drain",
]
