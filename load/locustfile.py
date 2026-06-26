"""Locust load test for GalaxyServe.

Generates throughput and latency (p50/p95/p99) numbers for the README metrics
table. Each simulated user POSTs a synthetic galaxy image to /predict, with a
smaller share of /health and /predict_batch calls.

Run (headless):
    locust -f load/locustfile.py --headless -u 20 -r 5 -t 30s \
        --host http://localhost:8000 --csv load/results

Or open the web UI:
    locust -f load/locustfile.py --host http://localhost:8000
"""

from __future__ import annotations

import io

import numpy as np
from locust import HttpUser, between, task
from PIL import Image


def _galaxy_png(seed: int = 0) -> bytes:
    """A synthetic 224x224 galaxy-ish image as PNG bytes."""
    rng = np.random.default_rng(seed)
    s = 224
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2)
    blob = np.exp(-(r**2) / (2 * (s / 5) ** 2))
    arr = np.clip((blob + rng.normal(0, 0.02, blob.shape)) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB").save(buf, format="PNG")
    return buf.getvalue()


class GalaxyUser(HttpUser):
    # Think time between requests; tune -u/-r/-t at the CLI for load level.
    wait_time = between(0.0, 0.05)

    def on_start(self) -> None:
        # Build a few distinct images once per user so payloads aren't identical.
        self._images = [_galaxy_png(seed) for seed in range(4)]
        self._i = 0

    def _next_image(self) -> bytes:
        img = self._images[self._i % len(self._images)]
        self._i += 1
        return img

    @task(10)
    def predict(self) -> None:
        files = {"file": ("galaxy.png", self._next_image(), "image/png")}
        with self.client.post("/predict", files=files, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")

    @task(2)
    def predict_batch(self) -> None:
        files = [("files", (f"g{i}.png", self._next_image(), "image/png")) for i in range(4)]
        with self.client.post("/predict_batch", files=files, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
