"""Loads the registered model from MLflow and serves predictions."""
import logging
import threading
import time
from typing import Mapping

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from app.core.config import settings
from app.core.exceptions import ModelNotLoadedError, PredictionError

logger = logging.getLogger(__name__)

# Must match the column order used in train_model.py
FEATURE_COLUMNS = [
    "transaction_amount",
    "time_of_day_hour",
    "num_transactions_1h",
    "avg_transaction_7d",
    "location_risk_score",
]


class FraudDetectionService:
    """Model is loaded once (startup) and treated as immutable afterwards.

    Loading failures never crash the process: the service stays "not ready",
    /predict answers 503 and loading is retried lazily with a cooldown.
    """

    def __init__(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
        tracking_uri: str | None = None,
        threshold: float | None = None,
        retry_seconds: float | None = None,
    ):
        self.model_name = model_name or settings.model_name
        self.requested_version = str(model_version or settings.model_version)
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.threshold = settings.fraud_threshold if threshold is None else threshold
        self.retry_seconds = settings.model_load_retry_seconds if retry_seconds is None else retry_seconds

        self._model = None
        self.resolved_version: str | None = None
        self.last_error: str | None = None
        self._last_attempt = 0.0
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _resolve_version(self, client: MlflowClient) -> str:
        if self.requested_version.lower() == "latest":
            versions = client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                raise LookupError(f"No versions registered for model '{self.model_name}'")
            return str(max(int(v.version) for v in versions))
        return self.requested_version

    def load(self) -> bool:
        """Load the model from the MLflow Model Registry. Returns True on success."""
        with self._lock:
            if self._model is not None:
                return True
            self._last_attempt = time.monotonic()
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                client = MlflowClient(tracking_uri=self.tracking_uri)
                version = self._resolve_version(client)
                if version.isdigit():
                    uri = f"models:/{self.model_name}/{version}"
                else:  # registry alias
                    uri = f"models:/{self.model_name}@{version}"
                logger.info("Loading model", extra={"model_uri": uri})
                self._model = mlflow.sklearn.load_model(uri)
                self.resolved_version = version
                self.last_error = None
                logger.info("Model loaded", extra={"model_uri": uri})
                return True
            except Exception as exc:  # corrupted / missing model, registry unreachable...
                self._model = None
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Model loading failed", extra={"error": self.last_error})
                return False

    def _ensure_loaded(self) -> None:
        if self.is_ready:
            return
        cooled_down = time.monotonic() - self._last_attempt >= self.retry_seconds
        if self._last_attempt == 0.0 or cooled_down:
            self.load()
        if not self.is_ready:
            raise ModelNotLoadedError(
                f"Model '{self.model_name}' (version '{self.requested_version}') is not available"
            )

    def predict(self, data: Mapping[str, float]) -> tuple[bool, float]:
        self._ensure_loaded()
        try:
            row = [[float(data[col]) for col in FEATURE_COLUMNS]]
            df = pd.DataFrame(row, columns=FEATURE_COLUMNS)
            if hasattr(self._model, "predict_proba"):
                probability = float(self._model.predict_proba(df)[0][1])
            else:
                probability = float(self._model.predict(df)[0])
        except KeyError as exc:
            raise PredictionError(f"Missing feature: {exc}") from exc
        except Exception as exc:
            raise PredictionError(f"Inference failed: {exc}") from exc
        probability = min(max(probability, 0.0), 1.0)
        return probability >= self.threshold, probability
