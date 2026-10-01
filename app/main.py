"""FastAPI application entry point."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import endpoints
from app.core.config import settings
from app.core.exceptions import FraudServiceError
from app.core.logging import request_id_ctx, setup_logging
from app.services.fraud_service import FraudDetectionService

setup_logging(settings.log_level)
logger = logging.getLogger("app")


def _error(status: int, code: str, message: str, details: list | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}, "request_id": request_id_ctx.get()}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup", extra={"mlflow_tracking_uri": settings.mlflow_tracking_uri})
    service = FraudDetectionService()
    if settings.load_model_on_startup:
        service.load()  # never raises; service stays "not ready" on failure
    app.state.fraud_service = service
    yield
    logger.info("Application shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Real-time fraud detection served from the MLflow Model Registry.",
        lifespan=lifespan,
    )
    app.include_router(endpoints.router, prefix="/api/v1", tags=["Fraud Prediction"])

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error", extra={"path": request.url.path, "method": request.method})
            response = _error(500, "internal_error", "Internal server error")
        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        response.headers["X-Request-ID"] = rid
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        request_id_ctx.reset(token)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errors = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        # Malformed JSON body -> 400; schema violations -> 422
        if any(e["type"] == "json_invalid" for e in errors):
            return _error(400, "bad_request", "Malformed JSON body", errors)
        logger.warning("Validation failed", extra={"path": request.url.path, "errors": errors})
        return _error(422, "validation_error", "Request validation failed", errors)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(FraudServiceError)
    async def service_handler(request: Request, exc: FraudServiceError):
        logger.error("Service error", extra={"path": request.url.path, "error": str(exc), "code": exc.code})
        return _error(exc.status_code, exc.code, str(exc))

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception", extra={"path": request.url.path})
        return _error(500, "internal_error", "Internal server error")

    return app


app = create_app()
