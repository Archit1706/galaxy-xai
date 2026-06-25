"""Model evaluation — returns a metrics dict used by training, the eval gate, and
champion/challenger promotion.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: str | torch.device = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader, returning (labels, preds, probs[:,1])."""
    device = torch.device(device)
    model.to(device).eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())
        all_labels.append(np.asarray(labels))
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: str | torch.device = "cpu"
) -> dict[str, float]:
    """Evaluate a classifier and return a flat metrics dict.

    Keys: accuracy, f1, precision, recall, roc_auc, n_samples. ROC-AUC is set to
    NaN if only one class is present in the eval set.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels, preds, probs = collect_predictions(model, loader, device)

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "roc_auc": roc_auc,
        "n_samples": int(len(labels)),
    }
    logger.info("Eval: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics
