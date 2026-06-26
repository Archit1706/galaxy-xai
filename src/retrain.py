"""Retraining orchestrator: drift check -> train challenger -> evaluate -> promote.

This is what the scheduled `drift-retrain` workflow runs. It checks live drift,
and if drift crosses the threshold (or --force is given) it trains a challenger,
which is registered with its test metrics, then runs champion/challenger
promotion — so a new version only reaches Production if it beats the champion.

Example:
    python -m src.retrain --tracking-uri http://localhost:5000 --smoke --force
"""

from __future__ import annotations

import argparse
import logging
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.config import ACCURACY_FLOOR, REGISTERED_MODEL_NAME
from src.promote import evaluate_and_promote
from src.registry import configure_mlflow
from src.train import build_parser as train_build_parser
from src.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("galaxyserve.retrain")


def _check_drift(reference_path: str, log_path: str, min_samples: int) -> float | None:
    """Return the drift share, or None if it can't be assessed yet."""
    from src.monitoring import run_drift_check

    try:
        summary = run_drift_check(reference_path, log_path, min_samples=min_samples)
        logger.info(
            "Drift: share=%.3f dataset_drift=%s", summary["share_of_drifted_columns"],
            summary["dataset_drift"],
        )
        return summary["share_of_drifted_columns"]
    except ValueError as exc:
        logger.info("Drift not assessable: %s", exc)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Drift-triggered retrain + promote.")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    p.add_argument("--metric", default="test_accuracy")
    p.add_argument("--floor", type=float, default=ACCURACY_FLOOR)
    p.add_argument("--min-improvement", type=float, default=0.0)
    p.add_argument("--stage", default="Production")
    # Drift trigger
    p.add_argument("--drift-threshold", type=float, default=0.3)
    p.add_argument("--reference-path", default=None)
    p.add_argument("--log-path", default=None)
    p.add_argument("--drift-min-samples", type=int, default=30)
    p.add_argument("--force", action="store_true", help="Retrain regardless of drift.")
    # Training
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    # Default paths come from the monitoring module (src.config-backed) so this
    # batch job doesn't depend on the web service's settings layer (pydantic-settings).
    from src.monitoring import DEFAULT_LOG_PATH, DEFAULT_REFERENCE_PATH

    reference_path = args.reference_path or str(DEFAULT_REFERENCE_PATH)
    log_path = args.log_path or str(DEFAULT_LOG_PATH)

    configure_mlflow(args.tracking_uri)

    # 1. Drift gate ------------------------------------------------------
    if not args.force:
        drift = _check_drift(reference_path, log_path, args.drift_min_samples)
        if drift is None:
            logger.info("Skipping retrain: drift not assessable (use --force to override).")
            print("retrain skipped reason=drift_not_assessable")
            return
        if drift < args.drift_threshold:
            logger.info("Skipping retrain: drift %.3f below threshold %.3f.", drift, args.drift_threshold)
            print(f"retrain skipped reason=below_threshold drift={drift:.3f}")
            return
        logger.info("Drift %.3f >= threshold %.3f -> retraining.", drift, args.drift_threshold)

    # 2. Train challenger (registers a new version with test metrics) -----
    targv = ["--tracking-uri", args.tracking_uri or "", "--model-name", args.model_name,
             "--run-name", "challenger"]
    if args.smoke:
        targv.append("--smoke")
    if args.epochs is not None:
        targv += ["--epochs", str(args.epochs)]
    targs = train_build_parser().parse_args([a for a in targv if a != ""])
    if targs.smoke and args.epochs is None:
        targs.epochs = targs.epochs_smoke
    run_id, version, metrics = run_training(targs)
    logger.info("Challenger v%s trained: %s=%.4f", version, args.metric,
                metrics.get(args.metric.replace("test_", ""), float("nan")))

    # 3. Champion/challenger promotion -----------------------------------
    decision = evaluate_and_promote(
        model_name=args.model_name,
        challenger_version=version,
        metric=args.metric,
        floor=args.floor,
        min_improvement=args.min_improvement,
        stage=args.stage,
        tracking_uri=args.tracking_uri,
    )
    print(
        f"retrain done challenger=v{version} promoted={decision['promoted']} "
        f"target={decision['target_stage']} reason={decision['reason']}"
    )


if __name__ == "__main__":
    main()
