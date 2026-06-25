"""Monitoring + data/prediction drift.

The service logs a small numeric feature vector for every image it scores, plus
the prediction. A reference dataset built from the training distribution is the
baseline; Evidently compares the live log against it (PSI/KS) to flag data drift
(input features) and prediction drift (confidence / predicted class).

Image drift is detected on interpretable summary features (brightness, contrast,
per-channel colour statistics) rather than raw pixels — this is what makes a
survey shift (e.g. Galaxy10 -> Galaxy Zoo Evo) show up as tabular drift.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import PRODUCTION_DATA_DIR, REFERENCE_DATA_DIR

logger = logging.getLogger(__name__)

# Interpretable image features used for drift detection.
FEATURE_COLUMNS = [
    "mean_brightness",
    "contrast",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
]
# Prediction columns — drift here is "prediction drift".
PREDICTION_COLUMNS = ["confidence", "class_id"]
ALL_COLUMNS = FEATURE_COLUMNS + PREDICTION_COLUMNS

DEFAULT_LOG_PATH = PRODUCTION_DATA_DIR / "prediction_log.jsonl"
DEFAULT_REFERENCE_PATH = REFERENCE_DATA_DIR / "reference.csv"

_log_lock = threading.Lock()


# --- Feature extraction ---------------------------------------------------
def extract_features(image: Image.Image) -> dict[str, float]:
    """Compute normalized (0..1) summary features from a PIL image."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return {
        "mean_brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "std_r": float(r.std()),
        "std_g": float(g.std()),
        "std_b": float(b.std()),
    }


# --- Prediction logging ---------------------------------------------------
def log_prediction(
    features: dict[str, float],
    prediction: dict,
    log_path: str | Path = DEFAULT_LOG_PATH,
    request_id: str | None = None,
) -> None:
    """Append one row (features + prediction) to the JSONL prediction log."""
    row = {
        "timestamp": time.time(),
        "request_id": request_id,
        **features,
        "confidence": float(prediction["confidence"]),
        "class_id": int(prediction["class_id"]),
        "label": prediction["label"],
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row)
    with _log_lock:  # serialize concurrent appends from worker threads
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_log(log_path: str | Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    """Load the prediction log as a DataFrame (empty if missing)."""
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame(columns=ALL_COLUMNS)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


# --- Reference dataset ----------------------------------------------------
def build_reference_dataframe(
    images: list[Image.Image], predictions: list[dict]
) -> pd.DataFrame:
    """Build a reference DataFrame (features + predictions) from images."""
    rows = []
    for img, pred in zip(images, predictions):
        rows.append(
            {
                **extract_features(img),
                "confidence": float(pred["confidence"]),
                "class_id": int(pred["class_id"]),
                "label": pred["label"],
            }
        )
    return pd.DataFrame(rows)


def save_reference(df: pd.DataFrame, path: str | Path = DEFAULT_REFERENCE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved reference dataset (%d rows) to %s", len(df), path)


def load_reference(path: str | Path = DEFAULT_REFERENCE_PATH) -> pd.DataFrame | None:
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


# --- Drift computation ----------------------------------------------------
def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    html_report_path: str | Path | None = None,
) -> dict:
    """Run Evidently DataDriftPreset and return a flat summary dict.

    Summary keys: dataset_drift, number_of_drifted_columns,
    share_of_drifted_columns, prediction_drift_detected, per_column (dict),
    reference_rows, current_rows, timestamp.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    columns = columns or [c for c in ALL_COLUMNS if c in reference.columns and c in current.columns]
    ref = reference[columns].astype(float)
    cur = current[columns].astype(float)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()

    dataset_drift = False
    n_drifted = 0
    share = 0.0
    per_column: dict[str, dict] = {}
    for metric in result["metrics"]:
        res = metric.get("result", {})
        if "dataset_drift" in res:
            dataset_drift = bool(res["dataset_drift"])
            n_drifted = int(res.get("number_of_drifted_columns", 0))
            share = float(res.get("share_of_drifted_columns", 0.0))
        if "drift_by_columns" in res:
            for col, info in res["drift_by_columns"].items():
                per_column[col] = {
                    "stattest": info.get("stattest_name"),
                    "drift_score": info.get("drift_score"),
                    "drift_detected": bool(info.get("drift_detected")),
                }

    prediction_drift_detected = any(
        per_column.get(c, {}).get("drift_detected") for c in PREDICTION_COLUMNS
    )

    if html_report_path:
        Path(html_report_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_report_path))

    return {
        "dataset_drift": dataset_drift,
        "number_of_drifted_columns": n_drifted,
        "share_of_drifted_columns": share,
        "prediction_drift_detected": prediction_drift_detected,
        "per_column": per_column,
        "reference_rows": int(len(ref)),
        "current_rows": int(len(cur)),
        "timestamp": time.time(),
    }


def run_drift_check(
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    log_path: str | Path = DEFAULT_LOG_PATH,
    min_samples: int = 30,
    html_report_path: str | Path | None = None,
) -> dict:
    """Load reference + live log, compute drift, and return the summary.

    Raises ValueError if the reference is missing or the live log has fewer than
    ``min_samples`` rows (drift on tiny samples is noise).
    """
    reference = load_reference(reference_path)
    if reference is None or reference.empty:
        raise ValueError(f"No reference dataset at {reference_path}. Build one first.")
    current = load_log(log_path)
    if len(current) < min_samples:
        raise ValueError(
            f"Only {len(current)} logged predictions (<{min_samples}); not enough to assess drift."
        )
    return compute_drift(reference, current, html_report_path=html_report_path)
