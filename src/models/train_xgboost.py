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

try:
    import mlflow
    from mlflow.models import infer_signature
except Exception:  # pragma: no cover
    mlflow = None
    infer_signature = None  # type: ignore[misc, assignment]


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


def _parse_json_dict(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("--mlflow-tags must be a JSON object")
    return {str(k): str(v) for k, v in loaded.items()}


def _coerce_param_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _flatten_params(prefix: str, params: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}{k}": _coerce_param_value(v) for k, v in params.items()}


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
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking for this run.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (e.g. file:./mlruns or http://localhost:5000).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
        help="MLflow experiment name (will be created if missing).",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default="train_xgboost",
        help="MLflow run name.",
    )
    parser.add_argument(
        "--mlflow-tags",
        type=str,
        default=None,
        help='Optional JSON object of MLflow tags (e.g. {"stage":"dev"}).',
    )
    parser.add_argument(
        "--mlflow-log-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log the trained estimator with mlflow.xgboost.log_model (signature + input example).",
    )
    parser.add_argument(
        "--mlflow-registered-model-name",
        type=str,
        default=os.environ.get("MLFLOW_REGISTERED_MODEL_NAME"),
        help="If set, register the logged model under this Model Registry name (same as log_model registered_model_name).",
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

    mlflow_enabled = bool(args.mlflow)
    if mlflow_enabled and mlflow is None:
        raise RuntimeError(
            "MLflow tracking requested (--mlflow) but mlflow could not be imported."
        )
    if mlflow_enabled and infer_signature is None:
        raise RuntimeError(
            "MLflow tracking requested (--mlflow) but mlflow.models.infer_signature could not be imported."
        )
    if mlflow_enabled and args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if mlflow_enabled and args.mlflow_experiment:
        mlflow.set_experiment(args.mlflow_experiment)

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

    if mlflow_enabled:
        tags = _parse_json_dict(args.mlflow_tags)
        # When executed via MLflow Projects (`mlflow run`), an active run already exists.
        # Avoid creating a nested run (which shows up as a second entry in the UI).
        active = mlflow.active_run()
        if active is None:
            mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.set_tag("mlflow.runName", args.mlflow_run_name)
        if tags:
            mlflow.set_tags(tags)

        mlflow.log_params(_flatten_params("xgb.", model.get_params()))
        mlflow.log_param("data.input_path", str(args.input))
        mlflow.log_param("data.train_rows", int(len(train_df)))
        mlflow.log_param("data.val_rows", int(len(val_df)))
        mlflow.log_param("data.test_rows", int(len(test_df)))
        mlflow.log_param("derived.scale_pos_weight", float(scale_pos_weight))

        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_probs)
        mlflow.log_param("derived.threshold", float(threshold))

        test_probs = model.predict_proba(X_test)[:, 1]
        metrics_against_label = compute_metrics(y_test_label, test_probs, threshold)
        metrics_against_truth = compute_metrics(y_test_truth, test_probs, threshold)

        mlflow.log_metric("auprc", float(metrics_against_truth["pr_auc"]))
        mlflow.log_metric("auroc", float(metrics_against_truth["roc_auc"]))
        mlflow.log_metric("f1", float(metrics_against_truth["f1"]))
        mlflow.log_metric("precision", float(metrics_against_truth["precision"]))
        mlflow.log_metric("recall", float(metrics_against_truth["recall"]))
        mlflow.log_metric("positive_rate", float(metrics_against_truth["positive_rate"]))
    else:
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

    if mlflow_enabled:
        if args.mlflow_log_model:
            sample_X = X_train.head(200)
            signature = infer_signature(sample_X, model.predict_proba(sample_X))
            log_kwargs: dict[str, object] = {
                "signature": signature,
                "input_example": X_train.head(5),
            }
            reg_name = (args.mlflow_registered_model_name or "").strip()
            if reg_name:
                log_kwargs["registered_model_name"] = reg_name
            mlflow.xgboost.log_model(model, name="model", **log_kwargs)
        mlflow.log_artifact(str(args.metrics_output))
        if mlflow.active_run() is not None:
            mlflow.end_run()

    print(f"Saved model to {args.model_output}")
    print(f"Saved metrics to {args.metrics_output}")
    print("Threshold:", round(threshold, 4))
    print("Test PR-AUC (label):", round(metrics_against_label["pr_auc"], 4))
    print("Test PR-AUC (truth):", round(metrics_against_truth["pr_auc"], 4))


if __name__ == "__main__":
    main()
