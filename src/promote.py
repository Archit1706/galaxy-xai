"""Champion / challenger promotion against the MLflow registry.

A challenger (a newly trained model version) is promoted to Production only if it
(1) clears the absolute accuracy floor and (2) beats the current champion (the
version in Production) by at least a margin. Otherwise it is parked in Staging so
it is still inspectable but never serves. This is the gate that decides what the
service actually loads.
"""

from __future__ import annotations

import argparse
import logging
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from mlflow.tracking import MlflowClient

from src.config import ACCURACY_FLOOR, REGISTERED_MODEL_NAME
from src.registry import configure_mlflow, transition_stage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("galaxyserve.promote")


def _version_metric(client: MlflowClient, model_name: str, version: str, metric: str) -> float | None:
    """Read a metric for a model version from its source run."""
    mv = client.get_model_version(model_name, version)
    if not mv.run_id:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get(metric)


def _latest_version(client: MlflowClient, model_name: str) -> str | None:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version)).version


def evaluate_and_promote(
    model_name: str = REGISTERED_MODEL_NAME,
    challenger_version: str | None = None,
    metric: str = "test_accuracy",
    floor: float = ACCURACY_FLOOR,
    min_improvement: float = 0.0,
    stage: str = "Production",
    tracking_uri: str | None = None,
) -> dict:
    """Decide whether to promote the challenger. Returns a decision dict."""
    if tracking_uri:
        configure_mlflow(tracking_uri)
    client = MlflowClient()

    if challenger_version is None:
        challenger_version = _latest_version(client, model_name)
    if challenger_version is None:
        raise RuntimeError(f"No versions found for model '{model_name}'.")

    challenger_metric = _version_metric(client, model_name, challenger_version, metric)
    if challenger_metric is None:
        raise RuntimeError(
            f"Challenger v{challenger_version} has no '{metric}' metric logged."
        )

    champion = None
    champion_metric = None
    current = client.get_latest_versions(model_name, stages=[stage])
    if current:
        champion = current[0]
        champion_metric = _version_metric(client, model_name, champion.version, metric)

    decision = {
        "model_name": model_name,
        "metric": metric,
        "floor": floor,
        "challenger_version": str(challenger_version),
        "challenger_metric": challenger_metric,
        "champion_version": str(champion.version) if champion else None,
        "champion_metric": champion_metric,
    }

    # Gate 1: absolute floor.
    if challenger_metric < floor:
        decision.update(promoted=False, target_stage="Staging",
                        reason=f"{metric}={challenger_metric:.4f} below floor {floor}")
    # Gate 2: must beat the champion (if one exists).
    elif champion is not None and champion.version != challenger_version and champion_metric is not None \
            and challenger_metric <= champion_metric + min_improvement:
        decision.update(promoted=False, target_stage="Staging",
                        reason=f"{metric}={challenger_metric:.4f} does not beat champion {champion_metric:.4f}")
    else:
        decision.update(promoted=True, target_stage=stage,
                        reason="clears floor and beats champion" if champion else "clears floor (no champion)")

    transition_stage(model_name, str(challenger_version), stage=decision["target_stage"],
                     archive_existing=decision["promoted"])
    logger.info("Decision: %s", decision["reason"])
    return decision


def main() -> None:
    p = argparse.ArgumentParser(description="Champion/challenger promotion in MLflow.")
    p.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    p.add_argument("--challenger-version", default=None, help="Defaults to the latest version.")
    p.add_argument("--metric", default="test_accuracy")
    p.add_argument("--floor", type=float, default=ACCURACY_FLOOR)
    p.add_argument("--min-improvement", type=float, default=0.0)
    p.add_argument("--stage", default="Production")
    p.add_argument("--tracking-uri", default=None)
    args = p.parse_args()

    decision = evaluate_and_promote(
        model_name=args.model_name,
        challenger_version=args.challenger_version,
        metric=args.metric,
        floor=args.floor,
        min_improvement=args.min_improvement,
        stage=args.stage,
        tracking_uri=args.tracking_uri,
    )
    print(
        f"promoted={decision['promoted']} target={decision['target_stage']} "
        f"challenger=v{decision['challenger_version']}({decision['challenger_metric']:.4f}) "
        f"champion={decision['champion_version']} reason={decision['reason']}"
    )
    # Non-zero exit if not promoted, so CI/automation can branch on it.
    sys.exit(0 if decision["promoted"] else 2)


if __name__ == "__main__":
    main()
