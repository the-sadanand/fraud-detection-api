"""API v1 routes."""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_fraud_service
from app.models.schemas import FraudPredictionRequest, FraudPredictionResponse, HealthResponse
from app.services.fraud_service import FraudDetectionService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health_check(service: FraudDetectionService = Depends(get_fraud_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="Fraud Detection API is running",
        model_loaded=service.is_ready,
        model_name=service.model_name,
        model_version=service.resolved_version,
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe (model loaded)")
def readiness(service: FraudDetectionService = Depends(get_fraud_service)) -> HealthResponse:
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return health_check(service)


@router.post("/predict", response_model=FraudPredictionResponse, summary="Predict fraud for a transaction")
def predict_fraud(
    request: FraudPredictionRequest,
    service: FraudDetectionService = Depends(get_fraud_service),
) -> FraudPredictionResponse:
    # Sync endpoint -> runs in FastAPI's threadpool, so CPU-bound inference never blocks the event loop.
    start = time.perf_counter()
    is_fraud, probability = service.predict(request.model_dump())
    logger.info(
        "prediction",
        extra={
            "is_fraud": is_fraud,
            "fraud_probability": round(probability, 6),
            "model_name": service.model_name,
            "model_version": service.resolved_version,
            "inference_ms": round((time.perf_counter() - start) * 1000, 3),
        },
    )
    return FraudPredictionResponse(is_fraud=is_fraud, fraud_probability=probability)
