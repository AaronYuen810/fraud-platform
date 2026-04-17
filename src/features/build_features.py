from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class HistoryWindow:
    timestamps: deque[pd.Timestamp]
    amounts: deque[float]


class FeatureState:
    def __init__(self) -> None:
        self.sender_events: dict[str, HistoryWindow] = defaultdict(
            lambda: HistoryWindow(deque(), deque())
        )
        self.pair_timestamps: dict[tuple[str, str], deque[pd.Timestamp]] = defaultdict(deque)
        self.beneficiary_timestamps: dict[str, deque[pd.Timestamp]] = defaultdict(deque)
        self.beneficiary_sender_events: dict[str, deque[tuple[pd.Timestamp, str]]] = defaultdict(deque)

    @staticmethod
    def _trim_timestamp_deque(
        dq: deque[pd.Timestamp], now: pd.Timestamp, window_seconds: int
    ) -> None:
        cutoff = now - pd.Timedelta(seconds=window_seconds)
        while dq and dq[0] < cutoff:
            dq.popleft()

    @staticmethod
    def _trim_sender_events(
        hist: HistoryWindow, now: pd.Timestamp, window_seconds: int
    ) -> None:
        cutoff = now - pd.Timedelta(seconds=window_seconds)
        while hist.timestamps and hist.timestamps[0] < cutoff:
            hist.timestamps.popleft()
            hist.amounts.popleft()

    @staticmethod
    def _trim_beneficiary_senders(
        dq: deque[tuple[pd.Timestamp, str]], now: pd.Timestamp, window_seconds: int
    ) -> None:
        cutoff = now - pd.Timedelta(seconds=window_seconds)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def compute_row_features(
        self, timestamp: pd.Timestamp, amount: float, sender: str, beneficiary: str
    ) -> dict[str, float]:
        sender_hist = self.sender_events[sender]
        self._trim_sender_events(sender_hist, timestamp, 24 * 3600)

        sender_txn_count_24h = len(sender_hist.timestamps)
        sender_txn_count_1h = sum(
            int(t >= timestamp - pd.Timedelta(hours=1)) for t in sender_hist.timestamps
        )
        sender_avg_amount_24h = float(np.mean(sender_hist.amounts)) if sender_hist.amounts else 0.0
        sender_std_amount_24h = float(np.std(sender_hist.amounts)) if sender_hist.amounts else 0.0

        pair_key = (sender, beneficiary)
        pair_dq = self.pair_timestamps[pair_key]
        self._trim_timestamp_deque(pair_dq, timestamp, 24 * 3600)
        pair_txn_count_24h = len(pair_dq)
        seconds_since_last_pair_txn = (
            (timestamp - pair_dq[-1]).total_seconds() if pair_dq else -1.0
        )

        ben_dq = self.beneficiary_timestamps[beneficiary]
        self._trim_timestamp_deque(ben_dq, timestamp, 3600)
        beneficiary_inbound_count_1h = len(ben_dq)

        ben_sender_dq = self.beneficiary_sender_events[beneficiary]
        self._trim_beneficiary_senders(ben_sender_dq, timestamp, 24 * 3600)
        beneficiary_unique_senders_24h = len({s for _, s in ben_sender_dq})

        return {
            "amount": float(amount),
            "hour_of_day": float(timestamp.hour),
            "day_of_week": float(timestamp.dayofweek),
            "sender_idx": float(int(sender[1:])),
            "beneficiary_idx": float(int(beneficiary[1:])),
            "sender_txn_count_1h": float(sender_txn_count_1h),
            "sender_txn_count_24h": float(sender_txn_count_24h),
            "sender_avg_amount_24h": sender_avg_amount_24h,
            "sender_std_amount_24h": sender_std_amount_24h,
            "pair_txn_count_24h": float(pair_txn_count_24h),
            "seconds_since_last_pair_txn": float(seconds_since_last_pair_txn),
            "beneficiary_inbound_count_1h": float(beneficiary_inbound_count_1h),
            "beneficiary_unique_senders_24h": float(beneficiary_unique_senders_24h),
        }

    def update(self, timestamp: pd.Timestamp, amount: float, sender: str, beneficiary: str) -> None:
        sender_hist = self.sender_events[sender]
        sender_hist.timestamps.append(timestamp)
        sender_hist.amounts.append(float(amount))
        self.pair_timestamps[(sender, beneficiary)].append(timestamp)
        self.beneficiary_timestamps[beneficiary].append(timestamp)
        self.beneficiary_sender_events[beneficiary].append((timestamp, sender))


def build_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "transaction_id",
        "timestamp",
        "amount",
        "sender_account",
        "beneficiary_account",
        "is_fraud_label",
        "is_fraud_ground_truth",
    }
    missing = required - set(transactions_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = transactions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if (df["sender_account"] == df["beneficiary_account"]).any():
        raise ValueError("Input transactions contain self-transfers.")

    state = FeatureState()
    feature_rows: list[dict[str, float]] = []

    for row in df.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        amount = float(row.amount)
        sender = str(row.sender_account)
        beneficiary = str(row.beneficiary_account)
        row_features = state.compute_row_features(timestamp, amount, sender, beneficiary)
        feature_rows.append(row_features)
        state.update(timestamp, amount, sender, beneficiary)

    features_df = pd.DataFrame(feature_rows)
    out = pd.concat(
        [
            df[
                [
                    "transaction_id",
                    "timestamp",
                    "sender_account",
                    "beneficiary_account",
                    "is_fraud_label",
                    "is_fraud_ground_truth",
                ]
            ].reset_index(drop=True),
            features_df.reset_index(drop=True),
        ],
        axis=1,
    )
    return out


def build_features_for_incoming(
    history_rows: Iterable[dict[str, object]], incoming_row: dict[str, object]
) -> dict[str, float]:
    state = FeatureState()
    history_df = pd.DataFrame(list(history_rows))
    if not history_df.empty:
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], utc=False)
        history_df = history_df.sort_values("timestamp")
        for row in history_df.itertuples(index=False):
            state.update(
                pd.Timestamp(row.timestamp),
                float(row.amount),
                str(row.sender_account),
                str(row.beneficiary_account),
            )

    ts = pd.to_datetime(incoming_row["timestamp"], utc=False)
    return state.compute_row_features(
        pd.Timestamp(ts),
        float(incoming_row["amount"]),
        str(incoming_row["sender_account"]),
        str(incoming_row["beneficiary_account"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe transaction features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/transactions.csv"),
        help="Path to raw transactions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/model_dataset.csv"),
        help="Path to output feature dataset CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    features = build_features(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Saved feature dataset to {args.output}")
    print(f"Rows: {len(features)}")


if __name__ == "__main__":
    main()
