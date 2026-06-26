"""Evaluation gate for CI.

Trains a quick model on the (separable) synthetic data, evaluates it on a
held-out split, and exits non-zero if accuracy is below the floor. Wired into
the CI pipeline so a code change that breaks the model — or a deliberate
accuracy regression — fails the build and blocks the merge.

In production the same gate runs against Galaxy10 with the 0.96 floor; in CI it
runs on synthetic data (no download/GPU) with a floor a healthy pipeline clears.

Example:
    python -m src.eval_gate --floor 0.9
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import build_loaders, make_synthetic
from src.evaluate import evaluate_model
from src.model import build_model
from src.train import train_one_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("galaxyserve.eval_gate")


def run_gate(floor: float, epochs: int, per_class: int, seed: int, device: str) -> tuple[bool, float]:
    torch.manual_seed(seed)
    images, labels = make_synthetic(n_per_class=per_class, seed=seed)
    loaders = build_loaders(images, labels, batch_size=32, seed=seed, augment=False)

    # From-scratch backbone: train all layers (no freezing) so it can learn.
    model = build_model(pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for epoch in range(epochs):
        loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        logger.info("epoch %d/%d train_loss=%.4f", epoch + 1, epochs, loss)

    metrics = evaluate_model(model, loaders["test"], device)
    accuracy = metrics["accuracy"]
    passed = accuracy >= floor
    logger.info("EVAL GATE: accuracy=%.4f floor=%.4f -> %s", accuracy, floor, "PASS" if passed else "FAIL")
    return passed, accuracy


def main() -> None:
    p = argparse.ArgumentParser(description="CI evaluation gate.")
    p.add_argument("--floor", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--per-class", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    passed, accuracy = run_gate(args.floor, args.epochs, args.per_class, args.seed, args.device)
    print(f"eval_gate accuracy={accuracy:.4f} floor={args.floor} passed={passed}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
