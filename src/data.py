"""Dataset loading and splitting for training / evaluation.

Mirrors the research notebook: Galaxy10 DECaLS is downloaded from HuggingFace
and mapped to a balanced binary Smooth/Featured problem. A synthetic generator
provides a fast, dependency-light path for smoke training in CI.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TRAIN_DATASET

logger = logging.getLogger(__name__)

# Galaxy10 DECaLS class -> binary label (matches the notebook).
SMOOTH_CLASSES = {2, 3, 4}  # round / in-between / cigar smooth  -> 0
FEATURED_CLASSES = {5, 6, 7}  # barred / tight / loose spiral     -> 1


class GalaxyDataset(Dataset):
    """Wraps uint8 HWC image arrays + integer labels with a transform."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def get_train_transform() -> transforms.Compose:
    """Augmented preprocessing for training (galaxies have no preferred orientation)."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_galaxy10_binary(
    max_per_class: int = 5000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Download Galaxy10 DECaLS and build a balanced binary dataset.

    Returns ``(images_uint8[N,224,224,3], labels[N])``. Requires the `datasets`
    package (install the `train` extra).
    """
    from datasets import load_dataset  # local import: heavy, only needed for real training
    from tqdm.auto import tqdm

    logger.info("Loading %s from HuggingFace...", TRAIN_DATASET)
    ds = load_dataset(TRAIN_DATASET, split="train")

    images, labels = [], []
    for ex in tqdm(ds, desc="Processing Galaxy10"):
        lbl = int(ex["label"])
        if lbl in SMOOTH_CLASSES:
            binary = 0
        elif lbl in FEATURED_CLASSES:
            binary = 1
        else:
            continue
        img = ex["image"]
        if not isinstance(img, Image.Image):
            continue
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        images.append(np.asarray(img, dtype=np.uint8))
        labels.append(binary)

    images_np = np.stack(images)
    labels_np = np.asarray(labels)

    rng = np.random.default_rng(seed)
    idx_s = np.where(labels_np == 0)[0]
    idx_f = np.where(labels_np == 1)[0]
    n = min(len(idx_s), len(idx_f), max_per_class)
    rng.shuffle(idx_s)
    rng.shuffle(idx_f)
    keep = np.concatenate([idx_s[:n], idx_f[:n]])
    rng.shuffle(keep)

    logger.info("Balanced Galaxy10 subset: %d images (%d per class)", len(keep), n)
    return images_np[keep], labels_np[keep]


def make_synthetic(n_per_class: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two visually-separable synthetic classes for fast smoke training/tests.

    Class 0 (Smooth): centered Gaussian blob. Class 1 (Featured): blob + spiral
    streaks. Not science — just enough structure for the pipeline to learn on.
    """
    rng = np.random.default_rng(seed)
    s = IMAGE_SIZE
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2)
    blob = np.exp(-(r**2) / (2 * (s / 5) ** 2))

    imgs, lbls = [], []
    for cls in (0, 1):
        for _ in range(n_per_class):
            base = blob.copy()
            if cls == 1:
                theta = np.arctan2(yy - s / 2, xx - s / 2)
                base = base + 0.4 * np.cos(4 * theta + r / 12) * (r < s / 3)
            noise = rng.normal(0, 0.03, base.shape)
            arr = np.clip((base + noise) * 255, 0, 255).astype(np.uint8)
            imgs.append(np.stack([arr] * 3, axis=-1))
            lbls.append(cls)

    images_np = np.stack(imgs)
    labels_np = np.asarray(lbls)
    perm = rng.permutation(len(labels_np))
    return images_np[perm], labels_np[perm]


def make_splits(
    n: int, seed: int = 42, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) permutation splits."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(train_frac * n)
    n_va = int(val_frac * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def build_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders with the standard transforms."""
    tr, va, te = make_splits(len(labels), seed=seed)
    train_t, eval_t = get_train_transform(), get_eval_transform()
    g = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(
            GalaxyDataset(images[tr], labels[tr], train_t),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g,
        ),
        "val": DataLoader(
            GalaxyDataset(images[va], labels[va], eval_t),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        ),
        "test": DataLoader(
            GalaxyDataset(images[te], labels[te], eval_t),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        ),
    }
