from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "sender_idx",
    "beneficiary_idx",
    "sender_txn_count_1h",
    "sender_txn_count_24h",
    "sender_avg_amount_24h",
    "sender_std_amount_24h",
    "pair_txn_count_24h",
    "seconds_since_last_pair_txn",
    "beneficiary_inbound_count_1h",
    "beneficiary_unique_senders_24h",
]


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def select_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 37):
        preds = (probs >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    metrics = {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(np.mean(preds)),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline XGBoost fraud model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/model_dataset.csv"),
        help="Path to model dataset CSV.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/xgboost_fraud_model.json"),
        help="Output path for trained XGBoost model.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("models/training_metrics.json"),
        help="Output path for metrics and metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    train_df, val_df, test_df = time_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["is_fraud_label"].astype(int).to_numpy()
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["is_fraud_label"].astype(int).to_numpy()
    X_test = test_df[FEATURE_COLUMNS]
    y_test_label = test_df["is_fraud_label"].astype(int).to_numpy()
    y_test_truth = test_df["is_fraud_ground_truth"].astype(int).to_numpy()

    positives = max(int(y_train.sum()), 1)
    negatives = max(int(len(y_train) - y_train.sum()), 1)
    scale_pos_weight = negatives / positives

    model = XGBClassifier(
        n_estimators=240,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_probs)

    test_probs = model.predict_proba(X_test)[:, 1]
    metrics_against_label = compute_metrics(y_test_label, test_probs, threshold)
    metrics_against_truth = compute_metrics(y_test_truth, test_probs, threshold)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_output))

    payload = {
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "scale_pos_weight": float(scale_pos_weight),
        "metrics_against_observed_label": metrics_against_label,
        "metrics_against_ground_truth": metrics_against_truth,
    }
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved model to {args.model_output}")
    print(f"Saved metrics to {args.metrics_output}")
    print("Threshold:", round(threshold, 4))
    print("Test PR-AUC (label):", round(metrics_against_label["pr_auc"], 4))
    print("Test PR-AUC (truth):", round(metrics_against_truth["pr_auc"], 4))

    enable_mlflow = os.getenv("ENABLE_MLFLOW", "").strip().lower() in {"1", "true", "yes", "y"}
    if enable_mlflow:
        try:
            import mlflow
        except ImportError:
            print("ENABLE_MLFLOW is set, but mlflow is not installed. Run `uv sync`.")
        else:
            mlflow.set_experiment("fraud-xgb")
            with mlflow.start_run():
                model_params = model.get_params()
                mlflow.log_params(
                    {
                        k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                        for k, v in model_params.items()
                    }
                )
                mlflow.log_params(
                    {
                        "threshold": float(threshold),
                        "train_rows": int(len(train_df)),
                        "val_rows": int(len(val_df)),
                        "test_rows": int(len(test_df)),
                    }
                )
                mlflow.log_metrics(
                    {
                        **{f"label_{k}": float(v) for k, v in metrics_against_label.items()},
                        **{f"truth_{k}": float(v) for k, v in metrics_against_truth.items()},
                    }
                )
                mlflow.log_artifact(str(args.model_output))
                mlflow.log_artifact(str(args.metrics_output))


if __name__ == "__main__":
    main()
