# Fraud Detection MVP

Simple end-to-end fraud detection baseline with:
- Mock transaction simulation
- Partial fraud labels from colluding accounts
- Leakage-safe transaction feature engineering
- XGBoost training and single-transaction scoring

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

To log runs to this server from Python, set:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

To stop the service:

```bash
make mlflow-down
```
