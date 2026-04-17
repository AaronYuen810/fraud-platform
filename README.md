# Fraud Detection MVP

**fraud-platform** is a Python fraud-detection MVP: an end-to-end baseline from synthetic transactions through features, an XGBoost model, and single-transaction scoring—not a production fraud stack.

## Walkthrough

**What it is.** A small **baseline** you can run locally: mock data with partial fraud signal (including labels tied to colluding accounts), **leakage-safe** feature engineering, **XGBoost** training, and **single-transaction** scoring.

**Pipeline.**

1. **Simulate** — [`src/data/simulate_transactions.py`](src/data/simulate_transactions.py) using [`config/simulation.yaml`](config/simulation.yaml).
2. **Features** — [`src/features/build_features.py`](src/features/build_features.py) builds a modeling dataset with features available at scoring time.
3. **Train** — [`src/models/train_xgboost.py`](src/models/train_xgboost.py) fits XGBoost and writes artifacts under `models/` (for example `xgboost_fraud_model.json` and metrics).
4. **Score** — [`src/models/score_transaction.py`](src/models/score_transaction.py) scores one new transaction JSON using the trained model and transaction history.

**Stack and tooling.** Python **3.11–3.13** (`requires-python` is `>=3.11,<3.14` in [`pyproject.toml`](pyproject.toml)); install and run with **uv**. Core libraries: NumPy, pandas, scikit-learn, XGBoost, PyYAML. **MLflow** is optional for experiment tracking (see below). **Makefile** and **docker-compose** support a local MLflow tracking server.

**Summary.** Simulate data, build honest features, train XGBoost, score incoming transactions, and optionally log experiments with MLflow.

## Quickstart (uv)

```bash
uv sync
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run src/data/simulate_transactions.py --config config/simulation.yaml
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run src/features/build_features.py --input data/raw/transactions.csv --output data/processed/model_dataset.csv
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run src/models/train_xgboost.py --input data/processed/model_dataset.csv --model-output models/xgboost_fraud_model.json --metrics-output models/training_metrics.json
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run -m src.models.score_transaction \
  --model-path models/xgboost_fraud_model.json \
  --metrics-path models/training_metrics.json \
  --history-path data/raw/transactions.csv \
  --incoming-json '{"transaction_id":"TX_NEW_1","timestamp":"2026-04-01T11:30:00","amount":9950,"sender_account":"A003","beneficiary_account":"A011"}'
```

## Experiment tracking (MLflow)

Start a local MLflow Tracking Server (persists data under `./mlflow-data/`):

```bash
make mlflow-up
```

Then open `http://localhost:5000`.

For training with MLflow as the experiment tracker, you can use the Makefile target:

```bash
make training-mlflow
```

To log to the local tracking server started above:

```bash
make training-mlflow MLFLOW_TRACKING_URI=http://localhost:5000
```

To log runs to this server from Python, set:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Then pass `--mlflow` when training:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run src/models/train_xgboost.py \
  --mlflow \
  --mlflow-experiment fraud-xgb \
  --input data/processed/model_dataset.csv \
  --model-output models/xgboost_fraud_model.json \
  --metrics-output models/training_metrics.json
```

To stop the service:

```bash
make mlflow-down
```
