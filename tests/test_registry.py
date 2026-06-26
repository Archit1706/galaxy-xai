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


def test_champion_challenger_promotion(mlflow_sqlite):
    """A challenger reaches Production only if it clears the floor and beats the champion."""
    from src.promote import evaluate_and_promote
    from src.registry import configure_mlflow, get_stage_version, log_and_register, transition_stage

    name = "cc-model"
    floor = 0.9
    configure_mlflow(mlflow_sqlite, experiment="cc-exp")

    def register_with_accuracy(acc: float) -> str:
        _, version = log_and_register(
            build_model(pretrained=False),
            params={"acc": acc},
            metrics={"test_accuracy": acc},
            model_name=name,
            register=True,
        )
        return version

    # Champion: v1 @ 0.92, promoted to Production.
    champ = register_with_accuracy(0.92)
    transition_stage(name, champ, stage="Production")

    # Challenger that beats the champion and clears the floor -> promoted.
    better = register_with_accuracy(0.96)
    d1 = evaluate_and_promote(name, better, floor=floor, tracking_uri=mlflow_sqlite)
    assert d1["promoted"] is True
    assert str(get_stage_version(name, "Production").version) == str(better)

    # Challenger below the new champion (0.96) -> not promoted, parked in Staging.
    worse = register_with_accuracy(0.93)
    d2 = evaluate_and_promote(name, worse, floor=floor, tracking_uri=mlflow_sqlite)
    assert d2["promoted"] is False
    assert str(get_stage_version(name, "Production").version) == str(better)

    # Challenger below the absolute floor -> not promoted.
    bad = register_with_accuracy(0.80)
    d3 = evaluate_and_promote(name, bad, floor=floor, tracking_uri=mlflow_sqlite)
    assert d3["promoted"] is False
    assert "floor" in d3["reason"]
