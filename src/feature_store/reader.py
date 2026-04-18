"""Read materialized features from the Feast online store."""

from __future__ import annotations

import time

from src.feature_store.constants import FEATURE_VIEW_NAME
from src.feature_store.store import get_feature_store
from src.models.train_xgboost import FEATURE_COLUMNS


def get_features(transaction_id: str) -> dict[str, float]:
    """Return the latest feature dict for `transaction_id` (model columns only)."""
    fs = get_feature_store()
    refs = [f"{FEATURE_VIEW_NAME}:{name}" for name in FEATURE_COLUMNS]
    resp = fs.get_online_features(
        features=refs,
        entity_rows=[{"transaction_id": transaction_id}],
    )
    payload = resp.to_dict()
    out: dict[str, float] = {}
    for name in FEATURE_COLUMNS:
        col = payload.get(name)
        if col is None or len(col) == 0:
            raise KeyError(f"Missing feature '{name}' in Feast online response.")
        val = col[0]
        if val is None:
            raise KeyError(f"Null feature '{name}' for transaction_id={transaction_id!r}.")
        out[name] = float(val)
    return out


def wait_for_features(
    transaction_id: str,
    *,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.05,
) -> dict[str, float]:
    """Poll the online store until the feature row exists (streaming write latency)."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return get_features(transaction_id)
        except KeyError as exc:
            last_err = exc
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Timed out waiting for Feast features for transaction_id={transaction_id!r}. "
        f"Last error: {last_err!r}"
    )
