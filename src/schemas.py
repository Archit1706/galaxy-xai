"""Pydantic response/request models — the API contract.

Keeping these explicit (rather than returning bare dicts) gives us validated,
self-documenting responses in the OpenAPI schema and stable shapes for clients
and contract tests.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config import CLASS_NAMES


class Prediction(BaseModel):
    """A single image's classification result."""

    class_id: int = Field(..., description="Predicted class index.", examples=[1])
    label: str = Field(..., description="Predicted class name.", examples=[CLASS_NAMES[-1]])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted class.")
    probabilities: dict[str, float] = Field(
        ..., description="Per-class probabilities (sum to 1)."
    )


class PredictResponse(Prediction):
    """Single-image prediction response."""

    filename: str | None = Field(default=None, description="Original upload filename, if any.")
    inference_ms: float = Field(..., description="Server-side inference time in milliseconds.")
    model_version: str = Field(..., description="Version/identifier of the serving model.")


class BatchItem(BaseModel):
    """One element of a batch response (success or per-item error)."""

    index: int
    filename: str | None = None
    prediction: Prediction | None = None
    error: str | None = Field(
        default=None, description="Set when this item failed; prediction is then null."
    )


class BatchPredictResponse(BaseModel):
    """Batch prediction response. Partial failures are reported per item."""

    count: int = Field(..., description="Number of items submitted.")
    succeeded: int
    failed: int
    inference_ms: float
    model_version: str
    results: list[BatchItem]


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    weights_loaded: bool = Field(
        ..., description="False means random init — predictions are not meaningful."
    )
    model_name: str
    model_version: str
    model_stage: str
    device: str
    uptime_s: float
    version: str = Field(..., description="GalaxyServe package version.")


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by all handled failures."""

    error: str = Field(..., description="Machine-readable error code.")
    detail: str = Field(..., description="Human-readable explanation.")
    request_id: str | None = None
