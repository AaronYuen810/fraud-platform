"""Push feature rows into the Feast online store."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from feast.data_source import PushMode

from src.feature_store.constants import PUSH_SOURCE_NAME
from src.feature_store.store import get_feature_store


def write_features(
    transaction_id: str,
    features: dict[str, float],
    timestamp: datetime,
) -> None:
    """Write one transaction's feature vector to the online store via Feast push."""
    fs = get_feature_store()
    ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
    row: dict = {
        "transaction_id": transaction_id,
        "event_timestamp": pd.Timestamp(ts),
        **features,
    }
    df = pd.DataFrame([row])
    fs.push(PUSH_SOURCE_NAME, df, to=PushMode.ONLINE)
