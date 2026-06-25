"""Runtime configuration for the inference service.

Values come from environment variables (prefix ``GALAXYSERVE_``) with sane
defaults, so the service is configurable in Docker/CI without code changes.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import DEFAULT_WEIGHTS_PATH, PRODUCTION_DATA_DIR, REFERENCE_DATA_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GALAXYSERVE_",
        env_file=".env",
        extra="ignore",
    )

    # --- Model source ---
    # When use_registry is True the service loads the model from the MLflow
    # registry (the configured stage); otherwise it loads weights_path directly.
    # On registry failure it falls back to weights_path if that file exists.
    use_registry: bool = Field(default=False)
    mlflow_tracking_uri: str = Field(default="", description="e.g. http://mlflow:5000")
    registry_model_name: str = Field(default="galaxy-morphology-resnet18")
    registry_stage: str = Field(default="Production")

    weights_path: str = Field(default=str(DEFAULT_WEIGHTS_PATH))
    device: str = Field(default="cpu", description="cpu or cuda")
    # If True, a missing weights file is a hard startup error (use in real deploys).
    require_weights: bool = Field(default=False)

    # --- Request limits (resilience) ---
    max_file_size_mb: float = Field(default=10.0, description="Max single image upload size.")
    max_batch_size: int = Field(default=32, description="Max images per /predict_batch call.")
    request_timeout_s: float = Field(default=30.0, description="Per-inference timeout.")

    # --- Monitoring / drift (Phase 3) ---
    monitoring_enabled: bool = Field(default=True, description="Log features+predictions per request.")
    prediction_log_path: str = Field(default=str(PRODUCTION_DATA_DIR / "prediction_log.jsonl"))
    reference_path: str = Field(default=str(REFERENCE_DATA_DIR / "reference.csv"))
    drift_min_samples: int = Field(default=30, description="Min logged rows before drift is assessed.")
    drift_check_interval_s: int = Field(
        default=0, description="Background drift-check period in seconds (0 disables the loop)."
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")

    # --- Metadata surfaced in /health and metrics (populated by registry in Phase 2) ---
    model_name: str = Field(default="galaxy-morphology-resnet18")
    model_version: str = Field(default="local")
    model_stage: str = Field(default="local-file")

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def allowed_content_types(self) -> set[str]:
        return {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so settings are parsed once per process."""
    return Settings()
