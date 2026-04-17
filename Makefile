DYLD_LIBRARY_PATH := /opt/homebrew/opt/libomp/lib:$(DYLD_LIBRARY_PATH)

.PHONY: mlflow-up mlflow-logs mlflow-down sync simulate features training score e2e

mlflow-up:
	docker compose up -d mlflow

mlflow-logs:
	docker compose logs -f mlflow

mlflow-down:
	docker compose down

sync:
	uv sync

simulate:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) uv run src/data/simulate_transactions.py --config config/simulation.yaml

features:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) uv run src/features/build_features.py --input data/raw/transactions.csv --output data/processed/model_dataset.csv

training:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) uv run src/models/train_xgboost.py --input data/processed/model_dataset.csv --model-output models/xgboost_fraud_model.json --metrics-output models/training_metrics.json

score:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) uv run -m src.models.score_transaction --model-path models/xgboost_fraud_model.json --metrics-path models/training_metrics.json --history-path data/raw/transactions.csv --incoming-json '{"transaction_id":"TX_NEW_1","timestamp":"2026-04-01T11:30:00","amount":9950,"sender_account":"A003","beneficiary_account":"A011"}'

e2e: sync simulate features training score
