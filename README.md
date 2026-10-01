# Fraud Detection API (FastAPI + MLflow + Docker Compose)

Production-style inference service for a fraud-detection model. The model is trained
by a separate script, tracked and registered in the **MLflow Model Registry**, and loaded
by the **FastAPI** service at startup. Everything runs locally with one command.

## Features
- **REST API** (`/api/v1`): `GET /health`, `GET /ready`, `POST /predict`, auto-generated OpenAPI docs at `/docs`.
- **Pydantic validation** → `422` on invalid input, `400` on malformed JSON, `503` if the model is unavailable, `500` (sanitised) on unexpected errors. One consistent JSON error envelope.
- **Structured JSON logging** for every request (method, path, status, duration, request id), prediction and error.
- **MLflow integration**: training logs params + metrics + model, registers `FraudDetectionModel`; the API loads a pinned version (or `latest` / an alias) from the registry. Training and serving are fully decoupled.
- **Containerised**: multi-stage, non-root Dockerfile; Compose stack with PostgreSQL (MLflow backend store), MLflow server (artifact proxy + persistent volume), a one-shot trainer, and the API with health checks.
- **Resilient model loading**: if the registry/model is unavailable the API still starts, `/health` reports `model_loaded: false`, `/predict` returns `503`, and loading is retried lazily (with cooldown).

## Project layout
```
app/            FastAPI app (main.py, api/v1/endpoints.py, services/fraud_service.py, models/schemas.py, core/)
tests/          unit/ and integration/ tests
generate_data.py  synthetic transactions -> data/raw_transactions.csv
train_model.py    train RandomForest, log to MLflow, register model
Dockerfile, docker/mlflow.Dockerfile, docker-compose.yml, requirements*.txt, .env.example
```

## Quick start (Docker Compose)
```bash
cp .env.example .env          # optional – every value has a default
docker compose up --build
```
Startup order: `db` → `mlflow` → `trainer` (generates data, trains, registers model v1; skipped if already registered) → `fraud-api`.

| Service | URL |
|---|---|
| API | http://localhost:8000 (also http://localhost:8888) |
| Swagger / OpenAPI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

Stop with `docker compose down` (data persists in volumes); `docker compose down -v` wipes everything.

## Local development (without Docker for the API)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# MLflow server on SQLite (or only start the db/mlflow compose services: docker compose up db mlflow)
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --serve-artifacts

export MLFLOW_TRACKING_URI=http://localhost:5000
python generate_data.py            # -> data/raw_transactions.csv
python train_model.py              # logs run + registers FraudDetectionModel
uvicorn app.main:app --reload --port 8000
```
`train_model.py` also generates the data itself if the CSV is missing. Options: `--n-estimators`, `--max-depth`, `--skip-if-registered`.

## API
### `GET /api/v1/health`
```bash
curl http://localhost:8000/api/v1/health
```
```json
{"status":"ok","message":"Fraud Detection API is running","model_loaded":true,"model_name":"FraudDetectionModel","model_version":"1"}
```
`GET /api/v1/ready` returns `503` until the model is loaded.

### `POST /api/v1/predict`
| Field | Type | Constraint |
|---|---|---|
| `transaction_amount` | float | > 0 |
| `time_of_day_hour` | int | 0–23 |
| `num_transactions_1h` | int | ≥ 0 |
| `avg_transaction_7d` | float | > 0 |
| `location_risk_score` | float | 0–1 |

```bash
curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d '{
  "transaction_amount": 950.0, "time_of_day_hour": 3, "num_transactions_1h": 7,
  "avg_transaction_7d": 60.0, "location_risk_score": 0.9 }'
```
```json
{"is_fraud": true, "fraud_probability": 0.87}
```
Invalid input (`422`):
```json
{"error":{"code":"validation_error","message":"Request validation failed",
  "details":[{"loc":["body","transaction_amount"],"msg":"Input should be greater than 0","type":"greater_than"}]},
 "request_id":"7c9d1c70..."}
```
| Status | When |
|---|---|
| 200 | success |
| 400 | malformed JSON body |
| 422 | schema / range violations |
| 503 | model not loaded (`model_unavailable`) |
| 500 | prediction failure or unhandled error |

Every response carries an `X-Request-ID` header (client-supplied value is echoed).
The decision threshold is configurable via `FRAUD_THRESHOLD` (default 0.5).

## MLflow usage
Open http://localhost:5000:
- **Experiments → `fraud-detection`**: run `rf-fraud-baseline` with params (`n_estimators`, `max_depth`, …), metrics (`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`) and the model artifact.
- **Models → `FraudDetectionModel`**: registered versions (alias `champion` points at the latest trained one).

Serve a different version: set `MODEL_VERSION` (`2`, `latest`, or an alias such as `champion`) and restart the API – the model is immutable while a container is running.
Backend store = PostgreSQL volume `db_data`; artifacts = volume `mlflow_artifacts` (served to clients through MLflow's artifact proxy, so the API container needs no shared filesystem).

## Testing
```bash
pip install -r requirements-dev.txt
pytest                                   # unit + integration (in-process, fake model, no MLflow needed)
pytest --cov=app --cov-report=term-missing

# integration tests against the running docker-compose stack
API_BASE_URL=http://localhost:8000 pytest tests/integration
```

## Configuration (`.env.example`)
`MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_VERSION`, `FRAUD_THRESHOLD`, `LOG_LEVEL`, `WEB_CONCURRENCY`, `DATABASE_URL` / `POSTGRES_*` (MLflow backend), `API_PORT`, `MLFLOW_PORT`. No secrets are hard-coded; the Postgres credentials shown are local-dev defaults – change them for anything real.

## Architecture decisions
- **FastAPI** – native Pydantic validation, OpenAPI docs, high throughput; sync endpoints run in a threadpool so CPU-bound inference doesn't block the event loop.
- **Gunicorn + Uvicorn workers** – multiple processes for concurrency; each loads the model once at startup (never per request).
- **MLflow registry** – versioned, traceable models; the API only knows a model *name + version*, giving a clean separation between training and serving.
- **PostgreSQL + artifact proxy** – durable backend store and no shared volume between services.
- **Trainer service** – makes `docker compose up` fully self-contained and idempotent.
- **Multi-stage, non-root image** – only the virtualenv and app code reach the final image.
- **Graceful degradation** – model failures yield `503` instead of crashing the service.

## Future enhancements
API-key/JWT auth and rate limiting, Prometheus metrics + Grafana, OpenTelemetry tracing, CI/CD (lint, tests, image scan, push), model-drift monitoring, canary / A-B model rollout via registry aliases, batch prediction endpoint, Kubernetes manifests with HPA.
