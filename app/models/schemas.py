"""Pydantic request/response schemas."""
from pydantic import BaseModel, ConfigDict, Field


class FraudPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_amount": 950.0,
                "time_of_day_hour": 3,
                "num_transactions_1h": 7,
                "avg_transaction_7d": 60.0,
                "location_risk_score": 0.9,
            }
        }
    )

    transaction_amount: float = Field(..., gt=0, allow_inf_nan=False, description="Amount of the transaction")
    time_of_day_hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    num_transactions_1h: int = Field(..., ge=0, description="Number of transactions in the last hour")
    avg_transaction_7d: float = Field(..., gt=0, allow_inf_nan=False, description="Average transaction amount in the last 7 days")
    location_risk_score: float = Field(..., ge=0, le=1, allow_inf_nan=False, description="Risk score associated with the transaction location")


class FraudPredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="True if the transaction is detected as fraud")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    message: str
    model_loaded: bool
    model_name: str
    model_version: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str
