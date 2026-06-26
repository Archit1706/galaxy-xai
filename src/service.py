"""GalaxyServe inference service (FastAPI).

Endpoints:
    GET  /              -> service banner + links
    GET  /health        -> liveness + model/weights status (200, or 503 if no model)
    GET  /ready         -> readiness (200 only when model is loaded)
    GET  /metrics       -> Prometheus exposition
    POST /predict       -> single image (multipart 'file') -> PredictResponse
    POST /predict_batch -> multiple images (multipart 'files') -> BatchPredictResponse

Resilience choices:
    * Model is loaded once at startup (lifespan) and warmed up to avoid cold-start latency.
    * CPU-bound inference is offloaded to a worker thread so the event loop stays responsive,
      and wrapped in a timeout.
    * Uploads are validated (content-type + size) before decode; decode failures and oversized
      payloads return structured 4xx errors rather than 500s.
    * Batch failures are reported per-item; one bad image doesn't fail the whole request.
    * Every response carries an X-Request-ID; all errors share one ErrorResponse envelope.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src import __version__, monitoring
from src import metrics as m
from src.model import load_image, predict_batch, predict_image
from src.schemas import (
    BatchItem,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    Prediction,
    PredictResponse,
)
from src.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("galaxyserve")


# --- Application state ----------------------------------------------------
class AppState:
    model = None
    weights_loaded: bool = False
    started_at: float = 0.0
    last_drift: dict | None = None
    drift_task: object | None = None


state = AppState()


def _record_monitoring(img: Image.Image, result: dict, request_id: str | None) -> None:
    """Log features + prediction for drift monitoring. Never breaks a prediction."""
    settings = get_settings()
    if not settings.monitoring_enabled:
        return
    try:
        features = monitoring.extract_features(img)
        monitoring.log_prediction(features, result, settings.prediction_log_path, request_id)
    except Exception as exc:  # noqa: BLE001 — monitoring is best-effort
        logger.warning("Monitoring log failed: %s", exc)


async def _drift_loop(interval_s: int) -> None:
    """Periodically recompute drift and update the Prometheus gauges."""
    settings = get_settings()
    while True:
        await asyncio.sleep(interval_s)
        try:
            summary = await run_in_threadpool(
                monitoring.run_drift_check,
                settings.reference_path,
                settings.prediction_log_path,
                settings.drift_min_samples,
            )
            m.update_drift_metrics(summary)
            state.last_drift = summary
            logger.info("Drift check: share=%.3f dataset_drift=%s",
                        summary["share_of_drifted_columns"], summary["dataset_drift"])
        except ValueError as exc:
            logger.info("Drift check skipped: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Drift check error: %s", exc)


async def _load_model_for_serving(settings) -> tuple[object, bool]:
    """Resolve the serving model: registry first (if configured), else weights file.

    Returns ``(model, weights_loaded)``. Updates settings' model_* metadata in
    place so /health and metrics reflect the actual source. Falls back to the
    local file if the registry is configured but unreachable/empty.
    """
    from src.model import load_model

    if settings.use_registry:
        try:
            from src.registry import load_model_from_registry

            model, meta = await run_in_threadpool(
                load_model_from_registry,
                settings.registry_model_name,
                settings.registry_stage,
                settings.mlflow_tracking_uri or None,
                settings.device,
            )
            settings.model_name = meta["model_name"]
            settings.model_version = meta["model_version"]
            settings.model_stage = meta["model_stage"]
            logger.info("Serving from registry: %s v%s (%s)", meta["model_name"],
                        meta["model_version"], meta["model_stage"])
            return model, True
        except Exception as exc:  # noqa: BLE001 — degrade gracefully to the file
            logger.error("Registry load failed (%s); falling back to weights file.", exc)

    weights_path = settings.weights_path
    exists = Path(weights_path).exists()
    if not exists and settings.require_weights:
        raise RuntimeError(
            f"Required weights not found at {weights_path} (GALAXYSERVE_REQUIRE_WEIGHTS=true)."
        )
    settings.model_stage = "local-file"
    model = await run_in_threadpool(load_model, weights_path if exists else None, settings.device)
    return model, exists


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.started_at = time.time()

    logger.info("Loading model (device=%s, use_registry=%s)...", settings.device, settings.use_registry)
    state.model, state.weights_loaded = await _load_model_for_serving(settings)

    # Warmup: a dummy forward pass so the first real request isn't slow.
    await run_in_threadpool(
        predict_image, state.model, Image.new("RGB", (64, 64)), settings.device
    )

    m.MODEL_LOADED.set(1)
    m.WEIGHTS_LOADED.set(1 if state.weights_loaded else 0)
    m.set_build_info(settings.model_name, settings.model_version, settings.model_stage)
    if not state.weights_loaded:
        logger.warning("Serving with RANDOM weights — predictions are not meaningful.")
    logger.info("Model ready.")

    if settings.monitoring_enabled and settings.drift_check_interval_s > 0:
        state.drift_task = asyncio.create_task(_drift_loop(settings.drift_check_interval_s))
        logger.info("Background drift checks every %ss.", settings.drift_check_interval_s)

    yield

    if state.drift_task is not None:
        state.drift_task.cancel()
    state.model = None
    m.MODEL_LOADED.set(0)
    logger.info("Service shutting down.")


app = FastAPI(
    title="GalaxyServe",
    version=__version__,
    description="Galaxy morphology classifier (Smooth vs Featured) — production inference API.",
    lifespan=lifespan,
)


# --- Middleware -----------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.middleware("http")(m.metrics_middleware)


# --- Error handling -------------------------------------------------------
class APIError(Exception):
    """Raised internally to produce a uniform structured error response."""

    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail


def _error_response(request: Request, status_code: int, error: str, detail: str) -> JSONResponse:
    m.ERRORS.labels(type=error).inc()
    payload = ErrorResponse(
        error=error, detail=detail, request_id=getattr(request.state, "request_id", None)
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return _error_response(request, exc.status_code, exc.error, exc.detail)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return _error_response(request, 422, "validation_error", str(exc.errors()))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return _error_response(request, 500, "internal_error", "An unexpected error occurred.")


# --- Upload validation + inference helpers --------------------------------
async def _read_validated_image(file: UploadFile) -> Image.Image:
    """Validate content-type + size, then decode. Raises APIError on bad input."""
    settings = get_settings()

    if file.content_type and file.content_type.lower() not in settings.allowed_content_types:
        raise APIError(
            415,
            "unsupported_media_type",
            f"Content-type '{file.content_type}' not supported. "
            f"Allowed: {sorted(settings.allowed_content_types)}.",
        )

    data = await file.read()
    if not data:
        raise APIError(422, "empty_file", f"Uploaded file '{file.filename}' is empty.")
    if len(data) > settings.max_file_size_bytes:
        raise APIError(
            413,
            "file_too_large",
            f"File '{file.filename}' exceeds limit of {settings.max_file_size_mb} MB.",
        )

    try:
        img = load_image(data)
        img.load()  # force decode now so a truncated image fails here, not mid-inference
        return img
    except (UnidentifiedImageError, OSError) as exc:
        raise APIError(
            422, "invalid_image", f"Could not decode '{file.filename}' as an image: {exc}"
        ) from exc


async def _infer_single(img: Image.Image) -> dict:
    settings = get_settings()
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(predict_image, state.model, img, settings.device),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise APIError(504, "inference_timeout", "Inference exceeded the time limit.") from exc
    m.INFERENCE_DURATION.labels(batch="single").observe(time.perf_counter() - t0)
    return result


# --- Endpoints ------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "GalaxyServe",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def health():
    settings = get_settings()
    loaded = state.model is not None
    body = HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        weights_loaded=state.weights_loaded,
        model_name=settings.model_name,
        model_version=settings.model_version,
        model_stage=settings.model_stage,
        device=settings.device,
        uptime_s=round(time.time() - state.started_at, 3) if state.started_at else 0.0,
        version=__version__,
    )
    return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())


@app.get("/ready", responses={200: {}, 503: {"model": ErrorResponse}})
async def ready(request: Request):
    if state.model is None:
        return _error_response(request, 503, "not_ready", "Model is not loaded yet.")
    return {"status": "ready"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return m.metrics_response()


@app.post("/drift/check", responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def drift_check(request: Request):
    """Recompute drift against the reference now, update gauges, return the summary."""
    settings = get_settings()
    try:
        summary = await run_in_threadpool(
            monitoring.run_drift_check,
            settings.reference_path,
            settings.prediction_log_path,
            settings.drift_min_samples,
        )
    except ValueError as exc:
        raise APIError(409, "drift_unavailable", str(exc)) from exc

    m.update_drift_metrics(summary)
    state.last_drift = summary
    return summary


@app.get("/drift/status")
async def drift_status():
    """Return the most recent drift summary (or a hint if none computed yet)."""
    if state.last_drift is None:
        return {"status": "no_drift_check_run", "hint": "POST /drift/check after sending traffic."}
    return state.last_drift


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(request: Request, file: UploadFile = File(...)):
    if state.model is None:
        raise APIError(503, "not_ready", "Model is not loaded yet.")

    settings = get_settings()
    img = await _read_validated_image(file)

    t0 = time.perf_counter()
    result = await _infer_single(img)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    m.record_prediction(result["label"], result["confidence"])
    _record_monitoring(img, result, getattr(request.state, "request_id", None))
    return PredictResponse(
        **result,
        filename=file.filename,
        inference_ms=elapsed_ms,
        model_version=settings.model_version,
    )


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
    responses={413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_batch_endpoint(request: Request, files: list[UploadFile] = File(...)):
    if state.model is None:
        raise APIError(503, "not_ready", "Model is not loaded yet.")

    settings = get_settings()
    if not files:
        raise APIError(422, "empty_batch", "No files provided.")
    if len(files) > settings.max_batch_size:
        raise APIError(
            413,
            "batch_too_large",
            f"Batch of {len(files)} exceeds max {settings.max_batch_size}.",
        )

    # Decode + validate each upload, tracking which succeeded so we can batch the good ones.
    decoded: list[tuple[int, str | None, Image.Image]] = []
    results: list[BatchItem | None] = [None] * len(files)
    for i, f in enumerate(files):
        try:
            img = await _read_validated_image(f)
            decoded.append((i, f.filename, img))
        except APIError as exc:
            results[i] = BatchItem(index=i, filename=f.filename, error=f"{exc.error}: {exc.detail}")

    t0 = time.perf_counter()
    if decoded:
        imgs = [img for _, _, img in decoded]
        try:
            preds = await asyncio.wait_for(
                run_in_threadpool(predict_batch, state.model, imgs, settings.device),
                timeout=settings.request_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise APIError(504, "inference_timeout", "Batch inference exceeded the time limit.") from exc

        m.INFERENCE_DURATION.labels(batch="batch").observe(time.perf_counter() - t0)
        for (idx, fname, img), pred in zip(decoded, preds, strict=True):
            m.record_prediction(pred["label"], pred["confidence"])
            _record_monitoring(img, pred, getattr(request.state, "request_id", None))
            results[idx] = BatchItem(index=idx, filename=fname, prediction=Prediction(**pred))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    final = [r for r in results if r is not None]
    succeeded = sum(1 for r in final if r.prediction is not None)
    return BatchPredictResponse(
        count=len(files),
        succeeded=succeeded,
        failed=len(files) - succeeded,
        inference_ms=elapsed_ms,
        model_version=settings.model_version,
        results=final,
    )
