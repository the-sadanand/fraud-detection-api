"""Domain exceptions mapped to HTTP responses in app.main."""


class FraudServiceError(Exception):
    status_code = 500
    code = "internal_error"


class ModelNotLoadedError(FraudServiceError):
    status_code = 503
    code = "model_unavailable"


class PredictionError(FraudServiceError):
    status_code = 500
    code = "prediction_failed"
