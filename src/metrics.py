"""Prometheus metrics for the inference service.

Exposes service-health signals (QPS, latency, errors, in-flight) and
model-level signals (prediction class distribution, confidence) that Phase 3
will join with Evidently drift data in Grafana.

NOTE: metrics live in the default process registry. Run the service with a
single worker (or enable prometheus multiprocess mode) so /metrics is coherent.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

from src import __version__

# A dedicated registry keeps our metrics isolated and testable.
REGISTRY = CollectorRegistry()

# --- HTTP-level metrics --------------------------------------------------
REQUEST_COUNT = Counter(
    "galaxyserve_requests_total",
    "Total HTTP requests.",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "galaxyserve_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)
IN_PROGRESS = Gauge(
    "galaxyserve_requests_in_progress",
    "In-flight HTTP requests.",
    ["endpoint"],
    registry=REGISTRY,
)

# --- Model-level metrics -------------------------------------------------
PREDICTIONS = Counter(
    "galaxyserve_predictions_total",
    "Predictions emitted, by predicted class label.",
    ["label"],
    registry=REGISTRY,
)
PREDICTION_CONFIDENCE = Histogram(
    "galaxyserve_prediction_confidence",
    "Distribution of predicted-class confidence.",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)
INFERENCE_DURATION = Histogram(
    "galaxyserve_inference_duration_seconds",
    "Model forward-pass duration in seconds (excludes HTTP/IO overhead).",
    ["batch"],  # "single" or "batch"
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
ERRORS = Counter(
    "galaxyserve_errors_total",
    "Handled errors, by type.",
    ["type"],
    registry=REGISTRY,
)

# --- Build / model info --------------------------------------------------
MODEL_LOADED = Gauge(
    "galaxyserve_model_loaded",
    "1 if a model is loaded and ready to serve, else 0.",
    registry=REGISTRY,
)
WEIGHTS_LOADED = Gauge(
    "galaxyserve_weights_loaded",
    "1 if trained weights were loaded (not random init), else 0.",
    registry=REGISTRY,
)
BUILD_INFO = Gauge(
    "galaxyserve_build_info",
    "Build/model metadata (value is always 1; see labels).",
    ["version", "model_name", "model_version", "model_stage"],
    registry=REGISTRY,
)


def set_build_info(model_name: str, model_version: str, model_stage: str) -> None:
    BUILD_INFO.labels(
        version=__version__,
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage,
    ).set(1)


def record_prediction(label: str, confidence: float) -> None:
    PREDICTIONS.labels(label=label).inc()
    PREDICTION_CONFIDENCE.observe(confidence)


def metrics_response() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def _route_template(request: Request) -> str:
    """Use the matched route path (e.g. /predict) not the raw URL, to bound cardinality."""
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Time every request and record count/latency/in-flight by route."""
    endpoint = _route_template(request)
    method = request.method

    # Don't let the scrape endpoint pollute its own latency stats.
    if request.url.path == "/metrics":
        return await call_next(request)

    IN_PROGRESS.labels(endpoint=endpoint).inc()
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        elapsed = time.perf_counter() - start
        # Re-resolve the route post-dispatch (it's populated during routing).
        endpoint = _route_template(request)
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        IN_PROGRESS.labels(endpoint=endpoint).dec()
