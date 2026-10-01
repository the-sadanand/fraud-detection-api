from fastapi import Request

from app.services.fraud_service import FraudDetectionService


def get_fraud_service(request: Request) -> FraudDetectionService:
    return request.app.state.fraud_service
