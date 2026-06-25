"""Phase 1 API contract tests for the FastAPI inference service.

Uses TestClient as a context manager so startup/shutdown (model load + warmup)
runs. These assert the response *contract*, not model accuracy.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.config import CLASS_NAMES
from src.service import app


def _png_bytes(size: int = 64, color=None) -> bytes:
    if color is None:
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
    else:
        img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    # X-Request-ID is echoed on every response.
    assert "x-request-id" in {k.lower() for k in r.headers}


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "GalaxyServe"


def test_metrics_endpoint(client):
    # Generate at least one prediction so model metrics are present.
    client.post("/predict", files={"file": ("g.png", _png_bytes(), "image/png")})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "galaxyserve_requests_total" in r.text
    assert "galaxyserve_predictions_total" in r.text


def test_predict_happy_path(client):
    r = client.post("/predict", files={"file": ("galaxy.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["filename"] == "galaxy.png"
    assert body["inference_ms"] >= 0


def test_predict_no_file_is_422(client):
    # Missing required 'file' field -> FastAPI validation error.
    assert client.post("/predict").status_code == 422


def test_predict_empty_file_is_422(client):
    r = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 422
    assert r.json()["error"] == "empty_file"


def test_predict_unsupported_type_is_415(client):
    r = client.post("/predict", files={"file": ("note.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert r.json()["error"] == "unsupported_media_type"


def test_predict_corrupt_image_is_422(client):
    r = client.post("/predict", files={"file": ("bad.png", b"not-an-image", "image/png")})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_image"


def test_predict_batch_happy(client):
    files = [
        ("files", ("a.png", _png_bytes(color=(10, 10, 10)), "image/png")),
        ("files", ("b.png", _png_bytes(color=(200, 200, 200)), "image/png")),
        ("files", ("c.png", _png_bytes(), "image/png")),
    ]
    r = client.post("/predict_batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert body["succeeded"] == 3
    assert body["failed"] == 0
    assert len(body["results"]) == 3
    assert all(item["prediction"]["label"] in CLASS_NAMES for item in body["results"])


def test_predict_batch_partial_failure(client):
    files = [
        ("files", ("good.png", _png_bytes(), "image/png")),
        ("files", ("bad.png", b"garbage", "image/png")),
    ]
    r = client.post("/predict_batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["succeeded"] == 1
    assert body["failed"] == 1
    errored = [i for i in body["results"] if i["error"]]
    assert len(errored) == 1
    assert "invalid_image" in errored[0]["error"]


def test_error_envelope_has_request_id(client):
    r = client.post("/predict", files={"file": ("e.png", b"", "image/png")})
    body = r.json()
    assert set(body) >= {"error", "detail", "request_id"}
    assert body["request_id"]
