"""Send synthetic traffic to a running service to demonstrate drift.

Sends a batch of "clean" images (matching the reference distribution) and/or a
batch of "shifted" images (corrupted to mimic a different survey: brighter,
colour-shifted, blurred). After the shifted batch, run a drift check and watch
``galaxyserve_drift_score`` spike in Prometheus/Grafana.

Uses only the standard library so it can run anywhere.

Example:
    python -m src.simulate_traffic --clean 80 --shifted 80 --trigger
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import urllib.request
import uuid

import numpy as np
from PIL import Image, ImageFilter

from src.data import make_synthetic


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _shift(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Corrupt an image so its feature distribution drifts from the reference."""
    img = arr.astype(np.float32)
    img = img * 1.35 + 35  # brighter -> mean_brightness / channel means drift
    img[..., 0] = np.clip(img[..., 0] * 1.25, 0, 255)  # red boost -> colour drift
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=2.0))  # blur -> contrast drift
    out = np.asarray(pil, dtype=np.float32)
    out += rng.normal(0, 6, out.shape)
    return np.clip(out, 0, 255)


def _post_image(url: str, png: bytes, filename: str) -> dict:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.types_map.get(".png", "image/png")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + png + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{url}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _send_batch(url: str, n: int, shifted: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    images, _ = make_synthetic(n_per_class=max(1, (n + 1) // 2), seed=seed)
    sent = 0
    for i in range(min(n, len(images))):
        arr = _shift(images[i], rng) if shifted else images[i]
        try:
            _post_image(url, _png_bytes(arr), f"{'shifted' if shifted else 'clean'}_{i}.png")
            sent += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  request {i} failed: {exc}")
    print(f"  sent {sent}/{n} {'shifted' if shifted else 'clean'} images")


def _drift_check(url: str) -> None:
    req = urllib.request.Request(f"{url}/drift/check", data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            summary = json.loads(resp.read())
        print(
            "drift: share={share_of_drifted_columns:.3f} dataset_drift={dataset_drift} "
            "drifted={number_of_drifted_columns} pred_drift={prediction_drift_detected}".format(**summary)
        )
    except urllib.error.HTTPError as exc:
        print(f"drift check: {exc.code} {exc.read().decode()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Send clean/shifted traffic to demo drift.")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--clean", type=int, default=0, help="Number of clean (in-distribution) images.")
    p.add_argument("--shifted", type=int, default=0, help="Number of shifted (drifted) images.")
    p.add_argument("--trigger", action="store_true", help="Run /drift/check after sending.")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    if args.clean:
        print(f"Sending {args.clean} clean images...")
        _send_batch(args.url, args.clean, shifted=False, seed=args.seed)
    if args.shifted:
        print(f"Sending {args.shifted} shifted images...")
        _send_batch(args.url, args.shifted, shifted=True, seed=args.seed + 1)
    if args.trigger:
        _drift_check(args.url)


if __name__ == "__main__":
    main()
