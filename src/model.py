"""Model definition, loading, and single-image inference.

The architecture here is an exact match for the one trained in the research
notebooks (`research/CS517_Galaxy_XAI_Midterm.ipynb`): an ImageNet-pretrained
ResNet-18 with `layer3`, `layer4`, and the classification head fine-tuned, and
the final FC replaced by `Dropout(0.3) -> Linear(512, 2)`. Keeping this in sync
is what lets a checkpoint from training load cleanly for serving.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from src.config import (
    CLASS_NAMES,
    HEAD_DROPOUT,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


def build_model(pretrained: bool = False, dropout: float = HEAD_DROPOUT) -> nn.Module:
    """Construct the ResNet-18 classifier with the production head.

    Args:
        pretrained: If True, download ImageNet weights for the backbone (used at
            the start of *training*). For *inference* leave False — we load our
            own fine-tuned ``state_dict`` over a randomly-initialized backbone,
            avoiding an unnecessary ImageNet download.
        dropout: Dropout probability in the classification head.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the final FC for binary classification (matches training).
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze all layers except layer3, layer4, and fc — the fine-tuning recipe."""
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ("layer3", "layer4", "fc"))
    return model


def get_eval_transform() -> transforms.Compose:
    """Deterministic preprocessing for inference/eval (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model(
    weights_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Build the model and load a trained ``state_dict`` if available.

    If ``weights_path`` is None or missing, returns a randomly-initialized model
    (still functional for shape/smoke tests). Real predictions require weights.
    """
    device = torch.device(device)
    model = build_model(pretrained=False)

    if weights_path is not None and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device)
        # Tolerate checkpoints saved as {"state_dict": ...} or raw state_dict.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=strict)
        logger.info("Loaded model weights from %s", weights_path)
    else:
        logger.warning(
            "No weights found at %s — using random initialization. "
            "Predictions will be meaningless until a checkpoint is provided.",
            weights_path,
        )

    model.to(device)
    model.eval()
    return model


def load_image(source: str | Path | bytes | Image.Image) -> Image.Image:
    """Load an image from a path, raw bytes, or a PIL image; return RGB PIL."""
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    return img.convert("RGB")


@torch.no_grad()
def predict_image(
    model: nn.Module,
    source: str | Path | bytes | Image.Image,
    device: str | torch.device = "cpu",
    transform: transforms.Compose | None = None,
) -> dict:
    """Run inference on a single image and return a structured prediction.

    Returns a dict: ``{class_id, label, confidence, probabilities}``.
    """
    device = torch.device(device)
    transform = transform or get_eval_transform()

    img = load_image(source)
    tensor = transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    class_id = int(probs.argmax().item())

    return {
        "class_id": class_id,
        "label": CLASS_NAMES[class_id],
        "confidence": float(probs[class_id].item()),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs.tolist(), strict=True)},
    }


def _format_prediction(probs_row: torch.Tensor) -> dict:
    class_id = int(probs_row.argmax().item())
    return {
        "class_id": class_id,
        "label": CLASS_NAMES[class_id],
        "confidence": float(probs_row[class_id].item()),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs_row.tolist(), strict=True)},
    }


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    images: list[Image.Image],
    device: str | torch.device = "cpu",
    transform: transforms.Compose | None = None,
) -> list[dict]:
    """Run a single batched forward pass over already-decoded PIL images.

    Returns one prediction dict per input, in order. Raises ValueError on an
    empty list. Decoding/validation of raw uploads is the caller's job so that
    per-item failures can be reported without sinking the whole batch.
    """
    if not images:
        raise ValueError("predict_batch received an empty image list")

    device = torch.device(device)
    transform = transform or get_eval_transform()

    batch = torch.stack([transform(img.convert("RGB")) for img in images]).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    return [_format_prediction(probs[i]) for i in range(probs.shape[0])]
