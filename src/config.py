"""Central configuration for GalaxyServe.

Single source of truth for the constants that must stay consistent across
training, serving, and monitoring (image size, normalization, class names,
artifact locations). These mirror the research notebooks so a checkpoint
trained there loads and predicts identically in production.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Repo layout ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
PRODUCTION_DATA_DIR = DATA_DIR / "production"

# Default checkpoint the service / registry bootstrap looks for.
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "resnet18_galaxy_best.pth"

# --- Task definition -----------------------------------------------------
# Binary morphology classification. Index order is the model's output order
# and MUST NOT change without retraining (it is baked into the checkpoint).
CLASS_NAMES: list[str] = ["Smooth", "Featured"]
NUM_CLASSES: int = len(CLASS_NAMES)

# --- Preprocessing (must match training) ---------------------------------
IMAGE_SIZE: int = 224
# ImageNet statistics — the model was fine-tuned from ImageNet-pretrained ResNet-18.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Dropout in the replaced classification head (matches notebook).
HEAD_DROPOUT: float = 0.3

# --- Datasets (HuggingFace sources) --------------------------------------
TRAIN_DATASET = "matthieulel/galaxy10_decals"  # reference / training distribution
DRIFT_DATASET = "mwalmsley/gz_evo"  # cross-survey shift used to demo drift

# --- MLflow / registry (used from Phase 2 onward) ------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "galaxyserve")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "galaxy-morphology-resnet18")

# Eval gate: deploys/promotions below this test accuracy are blocked (Phase 4).
ACCURACY_FLOOR: float = float(os.environ.get("ACCURACY_FLOOR", "0.96"))
