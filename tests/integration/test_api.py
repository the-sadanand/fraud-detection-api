"""API tests.

By default they run in-process with a fake model. To run them against the real
docker-compose stack:   API_BASE_URL=http://localhost:8000 pytest tests/integration
"""
import os

import httpx
import pytest

from app.api.deps import get_fraud_service
from app.main import app
from app.services.fraud_service import FraudDetectionService

BASE_URL = os.getenv("API_BASE_URL")
LIVE = BASE_URL is not None

VALID = {
    "transaction_amount": 950.0,
    "time_of_day_hour": 3,
    "num_transactions_1h": 7,
    "avg_transaction_7d": 60.0,
    "location_risk_score": 0.95,
}


@pytest.fixture
def client(fake_model):
    if LIVE:
        with httpx.Client(base_url=BASE_URL, timeout=15) as c:
            yield c
        return
    from fastapi.testclient import TestClient

    svc = FraudDetectionService(model_name="M", model_version="1", tracking_uri="http://x")
    svc._model, svc.resolved_version = fake_model, "1"
    app.dependency_overrides[get_fraud_service] = lambda: svc
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_success(client):
    r = client.post("/api/v1/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"is_fraud", "fraud_probability"}
    assert isinstance(body["is_fraud"], bool)
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_high_risk_vs_low_risk_ordering(client):
    hi = client.post("/api/v1/predict", json=VALID).json()["fraud_probability"]
    lo = client.post("/api/v1/predict", json={
        **VALID, "transaction_amount": 20, "time_of_day_hour": 14,
        "num_transactions_1h": 0, "avg_transaction_7d": 60, "location_risk_score": 0.02,
    }).json()["fraud_probability"]
    assert hi > lo


@pytest.mark.parametrize("field,value", [
    ("transaction_amount", -5),
    ("transaction_amount", 0),
    ("time_of_day_hour", 24),
    ("time_of_day_hour", -1),
    ("num_transactions_1h", -1),
    ("avg_transaction_7d", 0),
    ("location_risk_score", 1.5),
    ("location_risk_score", -0.1),
    ("transaction_amount", "abc"),
])
def test_predict_invalid_values_return_422(client, field, value):
    r = client.post("/api/v1/predict", json={**VALID, field: value})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID.items() if k != "location_risk_score"}
    assert client.post("/api/v1/predict", json=payload).status_code == 422


def test_predict_malformed_json_returns_400(client):
    r = client.post("/api/v1/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_unknown_route_returns_404(client):
    assert client.get("/api/v1/nope").status_code == 404


def test_wrong_method_returns_405(client):
    assert client.get("/api/v1/predict").status_code == 405


def test_request_id_header_is_echoed(client):
    r = client.get("/api/v1/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


@pytest.mark.skipif(LIVE, reason="in-process only")
def test_model_unavailable_returns_503():
    from fastapi.testclient import TestClient

    svc = FraudDetectionService(model_name="M", model_version="1", tracking_uri="http://x")
    svc.load = lambda: False  # simulate registry/model failure
    app.dependency_overrides[get_fraud_service] = lambda: svc
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/api/v1/predict", json=VALID)
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_unavailable"
        assert c.get("/api/v1/health").json()["model_loaded"] is False
        assert c.get("/api/v1/ready").status_code == 503
    app.dependency_overrides.clear()


@pytest.mark.skipif(LIVE, reason="in-process only")
def test_unexpected_error_returns_500_json():
    from fastapi.testclient import TestClient

    class Exploding:
        def predict(self, _):
            raise RuntimeError("kaboom")

    app.dependency_overrides[get_fraud_service] = lambda: Exploding()
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/api/v1/predict", json=VALID)
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "internal_error"
        assert "kaboom" not in r.text  # no internal details leaked
    app.dependency_overrides.clear()
