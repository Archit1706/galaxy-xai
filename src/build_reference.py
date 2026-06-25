"""Build the drift reference dataset from the training distribution.

The reference captures the feature + prediction distribution the model was
trained on; live traffic is compared against it. Sources:

    synthetic  – fast, no download (default; good for the drift demo)
    galaxy10   – the real training set from HuggingFace (requires the `train` extra)
    dir        – a folder of images you provide

Example:
    python -m src.build_reference --source synthetic --n 300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image

from src.config import DEFAULT_WEIGHTS_PATH, IMAGE_SIZE
from src.model import load_model, predict_image
from src.monitoring import DEFAULT_REFERENCE_PATH, build_reference_dataframe, save_reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("galaxyserve.reference")


def _images_synthetic(n: int, seed: int) -> list[Image.Image]:
    from src.data import make_synthetic

    images, _ = make_synthetic(n_per_class=max(1, n // 2), seed=seed)
    return [Image.fromarray(images[i]) for i in range(len(images))]


def _images_galaxy10(n: int, seed: int) -> list[Image.Image]:
    from src.data import load_galaxy10_binary

    images, _ = load_galaxy10_binary(max_per_class=max(1, n // 2), seed=seed)
    return [Image.fromarray(images[i]) for i in range(min(n, len(images)))]


def _images_dir(path: str) -> list[Image.Image]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    files = [p for p in Path(path).rglob("*") if p.suffix.lower() in exts]
    return [Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)) for p in files]


def main() -> None:
    p = argparse.ArgumentParser(description="Build the drift reference dataset.")
    p.add_argument("--source", choices=["synthetic", "galaxy10", "dir"], default="synthetic")
    p.add_argument("--n", type=int, default=300, help="Number of reference images (synthetic/galaxy10).")
    p.add_argument("--image-dir", help="Directory of images when --source dir.")
    p.add_argument("--out", default=str(DEFAULT_REFERENCE_PATH))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH))
    p.add_argument("--use-registry", action="store_true")
    p.add_argument("--tracking-uri", default=None)
    args = p.parse_args()

    if args.use_registry:
        from src.registry import load_model_from_registry

        model, meta = load_model_from_registry(tracking_uri=args.tracking_uri, device=args.device)
        logger.info("Loaded model %s v%s from registry.", meta["model_name"], meta["model_version"])
    else:
        model = load_model(args.weights, device=args.device)

    if args.source == "synthetic":
        images = _images_synthetic(args.n, args.seed)
    elif args.source == "galaxy10":
        images = _images_galaxy10(args.n, args.seed)
    else:
        if not args.image_dir:
            p.error("--image-dir is required when --source dir")
        images = _images_dir(args.image_dir)

    logger.info("Scoring %d reference images...", len(images))
    predictions = [predict_image(model, img, device=args.device) for img in images]

    df = build_reference_dataframe(images, predictions)
    save_reference(df, args.out)
    print(f"reference rows={len(df)} -> {args.out}")


if __name__ == "__main__":
    main()
