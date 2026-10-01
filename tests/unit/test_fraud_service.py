from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import ModelNotLoadedError, PredictionError
from app.services.fraud_service import FraudDetectionService

SAMPLE = {
    "transaction_amount": 500.0,
    "time_of_day_hour": 3,
    "num_transactions_1h": 5,
    "avg_transaction_7d": 50.0,
    "location_risk_score": 0.9,
}


def make_service(model, **kw):
    svc = FraudDetectionService(model_name="M", model_version="1", tracking_uri="http://x", **kw)
    svc._model = model
    svc.resolved_version = "1"
    return svc


def test_predict_fraud(fake_model):
    is_fraud, p = make_service(fake_model).predict(SAMPLE)
    assert is_fraud is True
    assert p == pytest.approx(0.9)


def test_predict_legit(fake_model):
    is_fraud, p = make_service(fake_model).predict({**SAMPLE, "location_risk_score": 0.1})
    assert is_fraud is False
    assert 0 <= p <= 1


def test_threshold_is_configurable(fake_model):
    is_fraud, _ = make_service(fake_model, threshold=0.95).predict(SAMPLE)
    assert is_fraud is False


def test_missing_feature_raises_prediction_error(fake_model):
    data = {k: v for k, v in SAMPLE.items() if k != "transaction_amount"}
    with pytest.raises(PredictionError):
        make_service(fake_model).predict(data)


def test_model_failure_is_wrapped():
    bad = MagicMock()
    bad.predict_proba.side_effect = RuntimeError("boom")
    with pytest.raises(PredictionError):
        make_service(bad).predict(SAMPLE)


def test_load_success_pins_version(fake_model):
    svc = FraudDetectionService(model_name="M", model_version="3", tracking_uri="http://x")
    with patch("app.services.fraud_service.mlflow.sklearn.load_model", return_value=fake_model) as load, \
         patch("app.services.fraud_service.MlflowClient"):
        assert svc.load() is True
    load.assert_called_once_with("models:/M/3")
    assert svc.is_ready and svc.resolved_version == "3"


def test_load_latest_resolves_highest_version(fake_model):
    svc = FraudDetectionService(model_name="M", model_version="latest", tracking_uri="http://x")
    versions = [MagicMock(version="1"), MagicMock(version="10"), MagicMock(version="2")]
    with patch("app.services.fraud_service.MlflowClient") as client, \
         patch("app.services.fraud_service.mlflow.sklearn.load_model", return_value=fake_model) as load:
        client.return_value.search_model_versions.return_value = versions
        assert svc.load() is True
    load.assert_called_once_with("models:/M/10")


def test_load_failure_is_graceful_and_predict_returns_unavailable():
    svc = FraudDetectionService(model_name="M", model_version="1", tracking_uri="http://x", retry_seconds=60)
    with patch("app.services.fraud_service.MlflowClient"), \
         patch("app.services.fraud_service.mlflow.sklearn.load_model", side_effect=OSError("corrupt")):
        assert svc.load() is False
        assert not svc.is_ready and "corrupt" in svc.last_error
        with pytest.raises(ModelNotLoadedError):
            svc.predict(SAMPLE)


def test_lazy_load_recovers_when_model_appears(fake_model):
    svc = FraudDetectionService(model_name="M", model_version="1", tracking_uri="http://x", retry_seconds=0)
    with patch("app.services.fraud_service.MlflowClient"), \
         patch("app.services.fraud_service.mlflow.sklearn.load_model", return_value=fake_model):
        is_fraud, _ = svc.predict(SAMPLE)
    assert is_fraud is True and svc.is_ready
