"""MLflow tracking + model-registry helpers.

This module is the "spine" of GalaxyServe: training logs runs and registers
model versions here, promotion transitions a version to a stage (Staging /
Production), and the service loads the current Production model from the
registry rather than a hardcoded path.

Pinned to MLflow 2.x so the Stages API is available (see pyproject `track`).
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.config import IMAGE_SIZE, MLFLOW_EXPERIMENT, REGISTERED_MODEL_NAME

logger = logging.getLogger(__name__)


def configure_mlflow(tracking_uri: str | None = None, experiment: str | None = None) -> None:
    """Point the MLflow client at a tracking server and select an experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment or MLFLOW_EXPERIMENT)
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())


def _example_signature(model: nn.Module, device: str | torch.device = "cpu"):
    """Build a model input/output signature + example from a dummy batch."""
    example = np.random.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(example).to(device)).cpu().numpy()
    return infer_signature(example, out), example


def log_and_register(
    model: nn.Module,
    params: dict,
    metrics: dict,
    *,
    run_name: str | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
    register: bool = True,
    artifacts: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[str, str | None]:
    """Log a run (params, metrics, model) and optionally register a new version.

    Returns ``(run_id, version)`` where ``version`` is None if not registered.
    """
    model.eval()
    signature, example = _example_signature(model, device)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        if artifacts:
            for local_path, artifact_path in artifacts.items():
                mlflow.log_artifact(local_path, artifact_path)

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=example,
            registered_model_name=model_name if register else None,
        )

    version = None
    if register:
        # The just-registered version is the newest for this model name.
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        version = max(versions, key=lambda v: int(v.version)).version
        logger.info("Registered %s version %s (run %s)", model_name, version, run_id)
    return run_id, version


def transition_stage(
    model_name: str,
    version: str,
    stage: str = "Production",
    archive_existing: bool = True,
) -> None:
    """Move a model version to a registry stage (Production/Staging/Archived)."""
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info("Transitioned %s v%s -> %s", model_name, version, stage)


def get_stage_version(model_name: str, stage: str = "Production") -> "mlflow.entities.model_registry.ModelVersion | None":
    """Return the ModelVersion currently in ``stage``, or None."""
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    return versions[0] if versions else None


def load_model_from_registry(
    model_name: str = REGISTERED_MODEL_NAME,
    stage: str = "Production",
    tracking_uri: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    """Load the model in ``stage`` from the registry.

    Returns ``(model, metadata)`` where metadata carries version/run/stage for
    surfacing in /health and metrics. Raises if no version is in that stage.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mv = get_stage_version(model_name, stage)
    if mv is None:
        raise RuntimeError(f"No version of '{model_name}' in stage '{stage}'.")

    model = mlflow.pytorch.load_model(f"models:/{model_name}/{stage}", map_location=str(device))
    model.to(device).eval()
    metadata = {
        "model_name": model_name,
        "model_version": str(mv.version),
        "model_stage": stage,
        "run_id": mv.run_id,
    }
    logger.info("Loaded %s v%s (%s) from registry", model_name, mv.version, stage)
    return model, metadata
