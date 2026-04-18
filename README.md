# Fraud Detection MVP

**fraud-platform** is a Python fraud-detection MVP: an end-to-end baseline from synthetic transactions through features, an XGBoost model, and single-transaction scoring—not a production fraud stack.

## Walkthrough

**What it is.** A small **baseline** you can run locally: mock data with partial fraud signal (including labels tied to colluding accounts), **leakage-safe** feature engineering, **XGBoost** training, and **single-transaction** scoring.

**Pipeline.**

1. **Simulate** — `[src/data/simulate_transactions.py](src/data/simulate_transactions.py)` using `[config/simulation.yaml](config/simulation.yaml)`.
2. **Features** — `[src/features/build_features.py](src/features/build_features.py)` builds a modeling dataset with features available at scoring time.
3. **Train** — `[src/models/train_xgboost.py](src/models/train_xgboost.py)` fits XGBoost and writes artifacts under `models/` (for example `xgboost_fraud_model.json` and metrics).
4. **Score** — `[src/models/score_transaction.py](src/models/score_transaction.py)` scores one new transaction JSON using the trained model and transaction history.

**Stack and tooling.** Python **3.11–3.13** (`requires-python` is `>=3.11,<3.14` in `[pyproject.toml](pyproject.toml)`); install and run with **uv**. Core libraries: NumPy, pandas, scikit-learn, XGBoost, PyYAML. **MLflow** is optional for experiment tracking (see below). **Makefile** and **docker-compose** support a local MLflow tracking server.

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

### Model Registry (training and promotion)

Training logs an MLflow **model** (XGBoost flavor with signature and input example) under the run artifact path `model/`, and still logs `training_metrics.json` at the run root for serving metadata.

- **Register on train:** pass a registry name so the model is registered as a new version in one step:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run src/models/train_xgboost.py \
  --mlflow \
  --mlflow-experiment fraud-xgb \
  --mlflow-registered-model-name fraud_xgb_classifier \
  --input data/processed/model_dataset.csv \
  --model-output models/xgboost_fraud_model.json \
  --metrics-output models/training_metrics.json
```

Or set `MLFLOW_REGISTERED_MODEL_NAME=fraud_xgb_classifier` in the environment.

Conventions (see [`src/mlops/registry_constants.py`](src/mlops/registry_constants.py)):

- Registered model name default: **`fraud_xgb_classifier`**
- Production alias: **`champion`** (assign after validation; not set automatically on train)

**Promote** a specific version to `champion` after optional metric gates (`auprc` and `f1` must exist on the training run when gates are used):

```bash
uv run python -m src.mlops.promote_model \
  --tracking-uri http://localhost:5000 \
  --model-name fraud_xgb_classifier \
  --version 1 \
  --alias champion \
  --min-auprc 0.0 \
  --min-f1 0.0
```

Or `make promote-model VERSION=1 MLFLOW_TRACKING_URI=http://localhost:5000` (add `MLFLOW_TRACKING_URI` in your shell if not using the Makefile default).

### MLflow tracking backend (team or production)

The bundled [`docker-compose.yml`](docker-compose.yml) uses **SQLite** and local disk under `./mlflow-data/`, which is fine for local development. Before sharing a tracking server across a team or running it as a long-lived production service, migrate to a **database-backed store** (for example Postgres or MySQL) and an **object store** for artifacts (for example S3 or GCS). That combination avoids SQLite locking issues, supports concurrent writers, and keeps large artifacts off the server disk. See the [MLflow tracking server storage documentation](https://www.mlflow.org/docs/latest/tracking.html#storage) for backend and artifact store configuration.

## Serve inference API (BentoML)

Run the BentoML service for online scoring:

```bash
uv sync
make serve-bento
```

The service loads either **local files** or a **Model Registry URI**:

**Local paths (default)**

- `models/xgboost_fraud_model.json` (`MODEL_PATH` to override)
- `models/training_metrics.json` (`METRICS_PATH` to override)
- Optional `MODEL_ID` for the value returned in API responses (defaults to the model filename)

**Registry-backed serving (production-style pin)**

Set `MLFLOW_MODEL_URI` to a pinned reference, for example `models:/fraud_xgb_classifier@champion` or `models:/fraud_xgb_classifier/3`. The process resolves the training run, downloads `training_metrics.json`, and loads the XGBoost model with MLflow. Set `MLFLOW_TRACKING_URI` to match your tracking server.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_MODEL_URI=models:/fraud_xgb_classifier@champion
# Optional: cache directory for downloaded artifacts (default is a temp directory)
# export MLFLOW_MODEL_CACHE_DIR=/var/cache/mlflow_models
make serve-bento
```

Responses use a **`model_id`** of the form `fraud_xgb_classifier:v3` (registered name and version). When using local files only, you can still set `MODEL_ID` yourself.

- Bento runtime state under `.bentoml/` in the project directory

Call the prediction endpoint:

```bash
curl -sS http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "features": {
        "amount": 9950,
        "hour_of_day": 11,
        "day_of_week": 2,
        "sender_idx": 3,
        "beneficiary_idx": 11,
        "sender_txn_count_1h": 2,
        "sender_txn_count_24h": 10,
        "sender_avg_amount_24h": 1320.5,
        "sender_std_amount_24h": 410.3,
        "pair_txn_count_24h": 1,
        "seconds_since_last_pair_txn": 7200,
        "beneficiary_inbound_count_1h": 3,
        "beneficiary_unique_senders_24h": 7
      }
    }
  }'
```

Response shape:

```json
{
  "fraud_score": 0.91,
  "threshold": 0.05,
  "flagged": true,
  "model_id": "xgboost_fraud_model.json",
  "feature_order": [
    "amount",
    "hour_of_day"
  ]
}
```

