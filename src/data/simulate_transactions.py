from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class SimulationConfig:
    """Configuration for synthetic transaction simulation."""

    seed: int = 42
    num_accounts: int = 30
    num_colluding_accounts: int = 5
    num_transactions: int = 6000
    fraud_ratio: float = 0.16
    label_capture_rate: float = 0.35
    start_timestamp: str = "2026-01-01T00:00:00"
    end_timestamp: str = "2026-03-31T23:59:59"
    structuring_threshold: float = 10_000
    output_transactions_path: str = "data/raw/transactions.csv"
    output_metadata_path: str = "data/raw/simulation_metadata.json"


def load_config(config_path: Path | None) -> SimulationConfig:
    """Load simulation configuration from YAML or return defaults.

    Args:
        config_path: Path to a YAML config file. If None, defaults are used.

    Returns:
        A populated `SimulationConfig`.
    """
    if config_path is None:
        return SimulationConfig()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SimulationConfig(**raw)


def _random_normal_amount(rng: np.random.Generator) -> float:
    """Sample a realistic non-fraud amount from a heavy-tailed distribution."""
    # Lognormal gives a realistic heavy-tailed amount distribution.
    amount = rng.lognormal(mean=6.2, sigma=0.9)
    return float(np.clip(amount, 5, 30_000))


def _fraud_amount(rng: np.random.Generator, threshold: float) -> float:
    """Sample a fraud-like amount (often near a threshold, sometimes above it)."""
    near = rng.uniform(threshold * 0.96, threshold * 0.999)
    if rng.random() < 0.35:
        near = rng.uniform(threshold * 1.2, threshold * 2.2)
    return float(round(near, 2))


def _uniform_timestamp(
    rng: np.random.Generator, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.Timestamp:
    """Draw a timestamp uniformly at random between `start_ts` and `end_ts`."""
    seconds = (end_ts - start_ts).total_seconds()
    return start_ts + pd.to_timedelta(rng.uniform(0, seconds), unit="s")


def simulate_transactions(config: SimulationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a synthetic transaction dataset and accompanying metadata.

    Args:
        config: Simulation configuration.

    Returns:
        (transactions_df, metadata) where `transactions_df` is time-ordered and
        `metadata` contains summary statistics and generator settings.

    Raises:
        ValueError: If the simulation produces invalid self-transfers.
    """
    rng = np.random.default_rng(config.seed)
    accounts = [f"A{i:03d}" for i in range(1, config.num_accounts + 1)]
    colluding_accounts = sorted(
        rng.choice(accounts, size=config.num_colluding_accounts, replace=False).tolist()
    )

    start_ts = pd.Timestamp(config.start_timestamp)
    end_ts = pd.Timestamp(config.end_timestamp)

    ring_pairs = [
        (colluding_accounts[i], colluding_accounts[(i + 1) % len(colluding_accounts)])
        for i in range(len(colluding_accounts))
    ]
    repeated_pairs = [ring_pairs[0], ring_pairs[2]]
    burst_centers = [
        _uniform_timestamp(rng, start_ts, end_ts)
        for _ in range(max(4, config.num_transactions // 1200))
    ]

    rows: list[dict[str, Any]] = []
    for i in range(config.num_transactions):
        tx_id = f"TX{i + 1:07d}"
        is_fraud = int(rng.random() < config.fraud_ratio)
        pattern = "normal"

        if is_fraud:
            p = rng.random()
            if p < 0.35:
                sender, beneficiary = ring_pairs[i % len(ring_pairs)]
                pattern = "circular"
                timestamp = _uniform_timestamp(rng, start_ts, end_ts)
            elif p < 0.60:
                sender, beneficiary = repeated_pairs[int(rng.integers(0, len(repeated_pairs)))]
                pattern = "repeated_pair"
                timestamp = _uniform_timestamp(rng, start_ts, end_ts)
            elif p < 0.82:
                sender = colluding_accounts[int(rng.integers(0, len(colluding_accounts)))]
                beneficiary_choices = [a for a in colluding_accounts if a != sender]
                beneficiary = beneficiary_choices[int(rng.integers(0, len(beneficiary_choices)))]
                pattern = "bursty"
                burst_center = burst_centers[int(rng.integers(0, len(burst_centers)))]
                jitter_seconds = rng.normal(loc=0.0, scale=25 * 60)
                timestamp = burst_center + pd.to_timedelta(jitter_seconds, unit="s")
                timestamp = min(max(timestamp, start_ts), end_ts)
            else:
                sender = colluding_accounts[int(rng.integers(0, len(colluding_accounts)))]
                beneficiary_choices = [a for a in colluding_accounts if a != sender]
                beneficiary = beneficiary_choices[int(rng.integers(0, len(beneficiary_choices)))]
                pattern = "structuring"
                timestamp = _uniform_timestamp(rng, start_ts, end_ts)

            amount = _fraud_amount(rng, config.structuring_threshold)
        else:
            sender = accounts[int(rng.integers(0, len(accounts)))]
            beneficiary = sender
            while beneficiary == sender:
                beneficiary = accounts[int(rng.integers(0, len(accounts)))]
            timestamp = _uniform_timestamp(rng, start_ts, end_ts)
            amount = _random_normal_amount(rng)

        is_fraud_label = int(is_fraud and rng.random() < config.label_capture_rate)
        rows.append(
            {
                "transaction_id": tx_id,
                "timestamp": timestamp,
                "amount": round(float(amount), 2),
                "sender_account": sender,
                "beneficiary_account": beneficiary,
                "is_fraud_ground_truth": is_fraud,
                "is_fraud_label": is_fraud_label,
                "fraud_pattern": pattern,
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if (df["sender_account"] == df["beneficiary_account"]).any():
        raise ValueError("Simulation created invalid self-transfers.")

    # Ensure we have at least one positive observed label.
    if df["is_fraud_label"].sum() == 0 and df["is_fraud_ground_truth"].sum() > 0:
        first_fraud_idx = int(df.index[df["is_fraud_ground_truth"] == 1][0])
        df.loc[first_fraud_idx, "is_fraud_label"] = 1

    metadata = {
        "seed": config.seed,
        "generated_at": datetime.utcnow().isoformat(),
        "num_accounts": config.num_accounts,
        "accounts": accounts,
        "colluding_accounts": colluding_accounts,
        "num_transactions": int(len(df)),
        "fraud_ratio_observed_ground_truth": float(df["is_fraud_ground_truth"].mean()),
        "fraud_ratio_observed_labels": float(df["is_fraud_label"].mean()),
        "label_capture_rate_empirical": float(
            df["is_fraud_label"].sum() / max(df["is_fraud_ground_truth"].sum(), 1)
        ),
    }
    return df, metadata


def save_outputs(df: pd.DataFrame, metadata: dict[str, Any], config: SimulationConfig) -> None:
    """Persist the generated transactions CSV and metadata JSON to disk."""
    tx_path = Path(config.output_transactions_path)
    md_path = Path(config.output_metadata_path)
    tx_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tx_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the transaction simulator."""
    parser = argparse.ArgumentParser(description="Simulate fraud transaction data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/simulation.yaml"),
        help="Path to simulation YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for generating synthetic transaction data."""
    args = parse_args()
    config = load_config(args.config)
    df, metadata = simulate_transactions(config)
    save_outputs(df, metadata, config)
    print(f"Saved transactions to {config.output_transactions_path}")
    print(f"Saved metadata to {config.output_metadata_path}")
    print(
        "Ground-truth fraud ratio:",
        round(float(metadata["fraud_ratio_observed_ground_truth"]), 4),
    )
    print("Observed label ratio:", round(float(metadata["fraud_ratio_observed_labels"]), 4))


if __name__ == "__main__":
    main()
