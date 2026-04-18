"""Feast entity + push feature view aligned with `FEATURE_COLUMNS` (training schema)."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, PushSource, ValueType

from feast.types import Float64, String, UnixTimestamp

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.train_xgboost import FEATURE_COLUMNS

transaction_entity = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    value_type=ValueType.STRING,
)

_schema_fields: list = [
    Field(name="transaction_id", dtype=String),
    Field(name="event_timestamp", dtype=UnixTimestamp),
]
_schema_fields.extend(Field(name=name, dtype=Float64) for name in FEATURE_COLUMNS)

transaction_scoring_batch = FileSource(
    name="transaction_features_batch",
    path="data/transaction_features.parquet",
    event_timestamp_column="event_timestamp",
)

transaction_scoring_push = PushSource(
    name="transaction_features_push",
    batch_source=transaction_scoring_batch,
)

transaction_scoring_view = FeatureView(
    name="transaction_scoring",
    entities=[transaction_entity],
    ttl=timedelta(days=0),
    schema=_schema_fields,
    source=transaction_scoring_push,
)
