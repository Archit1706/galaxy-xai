"""Phase 0 smoke tests: the model builds, loads, and predicts a valid schema.

These run WITHOUT a trained checkpoint (random init) so CI is green before real
weights exist. The eval-gate accuracy test (Phase 4) lives separately and only
runs when weights + data are present.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
from src.model import build_model, get_eval_transform, load_model, predict_image


@pytest.fixture(scope="module")
def model():
    # No weights path -> random init; still valid for shape/schema checks.
    return load_model(weights_path=None, device="cpu")


@pytest.fixture
def dummy_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_build_model_head_shape():
    """Final layer must output exactly NUM_CLASSES logits (binary)."""
    m = build_model(pretrained=False)
    # fc is Sequential(Dropout, Linear); the Linear is the last module.
    last_linear = m.fc[-1]
    assert isinstance(last_linear, torch.nn.Linear)
    assert last_linear.out_features == NUM_CLASSES == 2
    assert last_linear.in_features == 512  # ResNet-18 feature dim


def test_forward_pass_shape(model):
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, NUM_CLASSES)


def test_predict_image_schema(model, dummy_image):
    result = predict_image(model, dummy_image, device="cpu")

    assert set(result) == {"class_id", "label", "confidence", "probabilities"}
    assert result["class_id"] in (0, 1)
    assert result["label"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0

    probs = result["probabilities"]
    assert set(probs) == set(CLASS_NAMES)
    # Softmax probabilities must sum to 1.
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    # Reported confidence is the max class probability.
    assert result["confidence"] == pytest.approx(max(probs.values()), abs=1e-6)


def test_eval_transform_output():
    t = get_eval_transform()
    img = Image.new("RGB", (300, 200))  # arbitrary size -> resized to IMAGE_SIZE
    tensor = t(img)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_predict_from_bytes(model):
    import io

    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = predict_image(model, buf.getvalue(), device="cpu")
    assert result["label"] in CLASS_NAMES
