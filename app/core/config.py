"""Application settings, loaded from environment variables / .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),  # allow fields named model_*
    )

    app_name: str = "Fraud Detection API"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "FraudDetectionModel"
    model_version: str = "1"  # version number, "latest" or a registry alias

    fraud_threshold: float = 0.5
    load_model_on_startup: bool = True
    model_load_retry_seconds: float = 15.0  # cooldown between lazy reload attempts


settings = Settings()
