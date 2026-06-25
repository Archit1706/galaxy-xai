"""Phase 2 registry round-trip test.

Proves the spine end-to-end without any external services: log + register a
model in a temp sqlite-backed MLflow store, promote it to Production, load it
back from the registry, and confirm it predicts. (The model registry requires a
database backend — file store won't do — hence sqlite.)
"""

from __future__ import annotations

import mlflow
import numpy as np
import pytest
import torch

from src.model import build_model


@pytest.fixture()
def mlflow_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "mlflow.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    # Default artifact root for runs in this test.
    mlflow.set_registry_uri(uri)
    return uri


def test_register_promote_load_predict(mlflow_sqlite):
    from src.registry import (
        configure_mlflow,
        get_stage_version,
        load_model_from_registry,
        log_and_register,
        transition_stage,
    )

    name = "test-galaxy-model"
    configure_mlflow(mlflow_sqlite, experiment="test-exp")

    model = build_model(pretrained=False)
    run_id, version = log_and_register(
        model,
        params={"architecture": "resnet18", "smoke": True},
        metrics={"test_accuracy": 0.97},
        run_name="unit-test",
        model_name=name,
        register=True,
    )
    assert run_id
    assert version is not None

    # Promote and confirm it lands in Production.
    transition_stage(name, version, stage="Production")
    mv = get_stage_version(name, "Production")
    assert mv is not None
    assert str(mv.version) == str(version)

    # Load back from the registry and predict.
    loaded, meta = load_model_from_registry(name, "Production", mlflow_sqlite, device="cpu")
    assert str(meta["model_version"]) == str(version)
    assert meta["model_stage"] == "Production"

    x = torch.from_numpy(np.random.rand(2, 3, 224, 224).astype("float32"))
    with torch.no_grad():
        out = loaded(x)
    assert out.shape == (2, 2)
