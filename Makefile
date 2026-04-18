DYLD_LIBRARY_PATH := /opt/homebrew/opt/libomp/lib:$(DYLD_LIBRARY_PATH)

.PHONY: sync mlflow-up simulate features training-mlflow promote-model score mlflow-down serve-bento

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

# MLflow-enabled training. Override MLFLOW_TRACKING_URI if desired.
# Example: make training-mlflow MLFLOW_TRACKING_URI=http://localhost:5000
training-mlflow:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) uv run src/models/train_xgboost.py \
	--mlflow \
	--mlflow-experiment fraud-xgb \
	--input data/processed/model_dataset.csv \
	--model-output models/xgboost_fraud_model.json \
	--metrics-output models/training_metrics.json

# Register under Model Registry when training (optional). Example:
#   make training-mlflow MLFLOW_TRACKING_URI=http://localhost:5000 MLFLOW_REGISTERED_MODEL_NAME=fraud_xgb_classifier
# Promote version N to alias champion after gates (requires VERSION=):
#   make promote-model VERSION=1 MLFLOW_TRACKING_URI=http://localhost:5000
promote-model:
	@test -n "$(VERSION)" || (echo "Set VERSION to a model version number, e.g. VERSION=1"; exit 1)
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) uv run python -m src.mlops.promote_model --version $(VERSION)

score:
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) uv run -m src.models.score_transaction --model-path models/xgboost_fraud_model.json --metrics-path models/training_metrics.json --history-path data/raw/transactions.csv --incoming-json '{"transaction_id":"TX_NEW_1","timestamp":"2026-04-01T11:30:00","amount":9950,"sender_account":"A003","beneficiary_account":"A011"}'

serve-bento:
	BENTOML_HOME=.bentoml DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) uv run bentoml serve src.serving.bento_service:FraudScoringService --port 3000

e2e: sync simulate features training score
