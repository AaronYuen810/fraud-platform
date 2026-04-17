from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.build_features import build_features_for_incoming


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a single incoming transaction.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/xgboost_fraud_model.json"),
        help="Path to trained XGBoost model.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("models/training_metrics.json"),
        help="Path to training metadata with feature columns and threshold.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("data/raw/transactions.csv"),
        help="Path to historical transactions used for feature context.",
    )
    parser.add_argument(
        "--incoming-json",
        type=str,
        required=True,
        help=(
            "Incoming transaction JSON with keys: transaction_id,timestamp,amount,"
            "sender_account,beneficiary_account."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.metrics_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_columns = meta["feature_columns"]
    threshold = float(meta["threshold"])

    model = XGBClassifier()
    model.load_model(str(args.model_path))

    history_df = pd.read_csv(args.history_path)
    incoming = json.loads(args.incoming_json)
    if incoming["sender_account"] == incoming["beneficiary_account"]:
        raise ValueError("Invalid incoming transaction: sender and beneficiary cannot match.")

    incoming_features = build_features_for_incoming(
        history_rows=history_df[
            ["timestamp", "amount", "sender_account", "beneficiary_account"]
        ].to_dict(orient="records"),
        incoming_row=incoming,
    )

    x = np.array([[incoming_features[c] for c in feature_columns]], dtype=float)
    score = float(model.predict_proba(x)[0, 1])
    flagged = bool(score >= threshold)

    output = {
        "transaction_id": incoming.get("transaction_id"),
        "fraud_score": score,
        "threshold": threshold,
        "flagged": flagged,
        "features": incoming_features,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
