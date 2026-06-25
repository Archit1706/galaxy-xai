"""Bootstrap the MLflow registry from an existing trained checkpoint.

Use this to seed the registry with the already-trained ResNet-18 (from the
research notebooks) without retraining: it logs a run, registers a model
version, and promotes it to a stage (default Production) so the service can load
the model from the registry immediately.

Example:
    python -m src.register_model --tracking-uri http://localhost:5000 --promote
"""

from __future__ import annotations

import argparse
import logging
import sys

# MLflow prints unicode (e.g. 🏃) to stdout; force UTF-8 so it doesn't crash on
# Windows' cp1252 console. No-op where stdout is already UTF-8 (Linux/Docker).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import torch

from src.config import DEFAULT_WEIGHTS_PATH, REGISTERED_MODEL_NAME
from src.evaluate import evaluate_model
from src.model import load_model
from src.registry import configure_mlflow, log_and_register, transition_stage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("galaxyserve.register")

# Published test metrics for the ResNet-18 champion (research notebooks, Galaxy10
# DECaLS held-out test set). Used when --evaluate is not requested.
PUBLISHED_METRICS = {
    "test_accuracy": 0.9610,
    "test_f1": 0.9610,
    "test_precision": 0.9612,
    "test_recall": 0.9610,
    "test_roc_auc": 0.9897,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Register an existing checkpoint in MLflow.")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--experiment", default=None)
    p.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    p.add_argument("--stage", default="Production")
    p.add_argument("--promote", action="store_true", help="Transition the version to --stage.")
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Recompute metrics on Galaxy10 test split (requires the `train` extra + download).",
    )
    args = p.parse_args()

    model = load_model(args.weights, device=args.device, strict=True)

    if args.evaluate:
        from src.data import build_loaders, load_galaxy10_binary

        images, labels = load_galaxy10_binary()
        loaders = build_loaders(images, labels)
        metrics = {f"test_{k}": v for k, v in evaluate_model(model, loaders["test"], args.device).items()}
    else:
        metrics = dict(PUBLISHED_METRICS)
        logger.info("Using published research metrics (pass --evaluate to recompute).")

    configure_mlflow(args.tracking_uri, args.experiment)
    run_id, version = log_and_register(
        model,
        params={"architecture": "resnet18", "source": "research-checkpoint", "weights": args.weights},
        metrics=metrics,
        run_name="bootstrap-research-checkpoint",
        model_name=args.model_name,
        register=True,
        tags={"source": "research-checkpoint", "bootstrap": "true"},
        device=args.device,
    )
    logger.info("Registered %s version %s (run %s)", args.model_name, version, run_id)

    if args.promote and version is not None:
        transition_stage(args.model_name, version, stage=args.stage, archive_existing=True)
        logger.info("Promoted version %s to %s", version, args.stage)

    print(f"registered version={version} stage={'/'.join([args.stage]) if args.promote else 'none'}")


if __name__ == "__main__":
    main()
