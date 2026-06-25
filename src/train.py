"""Train the ResNet-18 galaxy classifier and log it to MLflow.

Logs params + per-epoch metrics + the model artifact, registers a new model
version, and optionally promotes it to a registry stage.

Examples:
    # Fast smoke run on synthetic data (no download) — used by CI:
    python -m src.train --smoke --tracking-uri sqlite:///mlflow.db

    # Full training on Galaxy10 DECaLS:
    python -m src.train --epochs 15 --tracking-uri http://localhost:5000 --promote
"""

from __future__ import annotations

import argparse
import logging
import sys

# Force UTF-8 stdout so MLflow's unicode logging doesn't crash Windows' cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import REGISTERED_MODEL_NAME
from src.data import build_loaders, load_galaxy10_binary, make_synthetic
from src.evaluate import evaluate_model
from src.model import build_model, freeze_backbone
from src.registry import configure_mlflow, log_and_register, transition_stage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("galaxyserve.train")


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def val_loss(model, loader, criterion, device) -> float:
    model.eval()
    running = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        running += criterion(model(images), labels).item() * images.size(0)
    return running / len(loader.dataset)


def run_training(args) -> tuple[str, str | None, dict]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if args.smoke:
        logger.info("SMOKE mode: synthetic data, no download.")
        images, labels = make_synthetic(n_per_class=args.smoke_per_class, seed=args.seed)
        pretrained = False
    else:
        images, labels = load_galaxy10_binary(max_per_class=args.max_per_class, seed=args.seed)
        pretrained = not args.no_pretrained

    loaders = build_loaders(images, labels, batch_size=args.batch_size, seed=args.seed)

    model = build_model(pretrained=pretrained)
    freeze_backbone(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    params = {
        "architecture": "resnet18",
        "pretrained": pretrained,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "n_samples": len(labels),
        "smoke": args.smoke,
        "seed": args.seed,
    }

    configure_mlflow(args.tracking_uri, args.experiment)

    best_val = float("inf")
    best_state = None
    history: list[tuple[float, float]] = []
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        va = val_loss(model, loaders["val"], criterion, device)
        scheduler.step(va)
        history.append((tr, va))
        logger.info("Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f", epoch + 1, args.epochs, tr, va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, loaders["test"], device)
    logger.info("Test metrics: %s", test_metrics)

    # Log per-epoch loss curves as stepped metrics, then the run + model.
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        for step, (tr, va) in enumerate(history):
            mlflow.log_metric("train_loss", tr, step=step)
            mlflow.log_metric("val_loss", va, step=step)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    # log_and_register opens its own run for the model artifact + registration.
    model_run_id, version = log_and_register(
        model,
        params=params,
        metrics={f"test_{k}": v for k, v in test_metrics.items()},
        run_name=(args.run_name or "training") + "-model",
        model_name=args.model_name,
        register=not args.no_register,
        tags={"smoke": str(args.smoke), "stage_intent": args.stage if args.promote else "none"},
        device=device,
    )

    if args.promote and version is not None:
        transition_stage(args.model_name, version, stage=args.stage, archive_existing=True)

    return model_run_id, version, test_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and register the galaxy classifier.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-per-class", type=int, default=5000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet backbone download.")
    p.add_argument("--smoke", action="store_true", help="Fast synthetic-data run for CI.")
    p.add_argument("--smoke-per-class", type=int, default=48)
    p.add_argument("--epochs-smoke", type=int, default=2)
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--experiment", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--promote", action="store_true", help="Transition the new version to --stage.")
    p.add_argument("--stage", default="Staging", help="Stage to promote to if --promote.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.epochs = args.epochs_smoke
    run_id, version, metrics = run_training(args)
    print(f"run_id={run_id} version={version} test_accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
