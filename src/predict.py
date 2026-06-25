"""Command-line single-image prediction — the Phase 0 demoable.

Usage:
    python -m src.predict path/to/galaxy.jpg
    python -m src.predict path/to/galaxy.jpg --weights models/resnet18_galaxy_best.pth
    python -m src.predict --demo        # synthesize an image, prove the pipeline runs

Prints a JSON prediction: predicted class, confidence, and per-class probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import DEFAULT_WEIGHTS_PATH, IMAGE_SIZE
from src.model import load_model, predict_image


def _demo_image() -> Image.Image:
    """A deterministic synthetic 'galaxy-ish' image for smoke demos (no data needed)."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    cx = cy = IMAGE_SIZE / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    blob = np.exp(-(r**2) / (2 * (IMAGE_SIZE / 6) ** 2))  # central bright core
    noise = rng.normal(0, 0.03, size=blob.shape)
    arr = np.clip((blob + noise) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify a galaxy image (Smooth vs Featured).")
    parser.add_argument("image", nargs="?", help="Path to an image file.")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to model checkpoint (.pth). Falls back to random init if missing.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a synthesized image instead of a file (proves the pipeline runs).",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    args = parser.parse_args(argv)

    if not args.demo and not args.image:
        parser.error("provide an image path or use --demo")

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = load_model(args.weights, device=device)

    source: object = _demo_image() if args.demo else args.image
    if not args.demo and not Path(args.image).exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    result = predict_image(model, source, device=device)
    result["weights_loaded"] = Path(args.weights).exists()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
