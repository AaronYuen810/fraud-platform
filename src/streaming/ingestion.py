"""In-memory transaction queue and enqueue helpers for the streaming pipeline.

Events are processed FIFO by a single daemon worker. For parity with batch
training (`build_features`), **enqueue order should match timestamp-sorted batch
order** (oldest first): the worker applies `compute_row_features` → persist →
`update` in the same sequence as `build_features` / `build_features_for_incoming`.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import datetime

_transaction_queue: queue.Queue["TransactionEvent"] = queue.Queue()
_worker_thread: threading.Thread | None = None
_worker_start_lock = threading.Lock()


@dataclass(frozen=True)
class TransactionEvent:
    """One raw transaction for streaming feature computation."""

    transaction_id: str
    timestamp: datetime
    amount: float
    sender_account: str
    beneficiary_account: str


def get_transaction_queue() -> queue.Queue["TransactionEvent"]:
    """Return the shared in-memory queue (for the worker and test helpers)."""
    return _transaction_queue


def enqueue_transaction(event: TransactionEvent) -> None:
    """Enqueue a transaction for FIFO processing by the single worker thread."""
    _transaction_queue.put(event)


def ensure_worker_started() -> None:
    """Start the daemon worker thread if it is not already running (idempotent)."""
    global _worker_thread
    with _worker_start_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        from src.streaming.worker import run_worker_loop

        _worker_thread = threading.Thread(target=run_worker_loop, daemon=True)
        _worker_thread.start()


def wait_for_queue_drain() -> None:
    """Block until all items currently in the queue have been processed.

    Pair with `enqueue_transaction`: each `put` is matched by the worker with
    `task_done`, so `join` completes when the backlog is cleared.
    """
    _transaction_queue.join()


def drain_pending_transactions() -> None:
    """Remove and discard all not-yet-processed queue items (for tests / reset).

    Each discarded item completes the queue bookkeeping (`task_done`) so `join`
    and internal counters stay consistent.
    """
    q = _transaction_queue
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
        else:
            q.task_done()
