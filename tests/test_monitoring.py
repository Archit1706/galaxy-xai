"""Phase 3 monitoring/drift tests.

Verifies feature extraction, the prediction logger round-trip, and that
Evidently flags drift on a shifted distribution but not on an in-distribution
one. No running service required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import monitoring


def _img(color, size=64):
    return Image.fromarray(np.full((size, size, 3), color, dtype=np.uint8))


def test_extract_features_keys_and_range():
    feats = monitoring.extract_features(_img((128, 64, 200)))
    assert set(feats) == set(monitoring.FEATURE_COLUMNS)
    assert all(0.0 <= v <= 1.0 for v in feats.values())
    # A bright image has higher mean_brightness than a dark one.
    assert monitoring.extract_features(_img((240, 240, 240)))["mean_brightness"] > \
        monitoring.extract_features(_img((10, 10, 10)))["mean_brightness"]


def test_log_prediction_roundtrip(tmp_path):
    log = tmp_path / "log.jsonl"
    pred = {"class_id": 1, "label": "Featured", "confidence": 0.9}
    for _ in range(3):
        monitoring.log_prediction(monitoring.extract_features(_img((100, 100, 100))), pred, log, "rid")
    df = monitoring.load_log(log)
    assert len(df) == 3
    assert {"confidence", "class_id", "label", *monitoring.FEATURE_COLUMNS} <= set(df.columns)


def _make_df(rng, n, shift=0.0, conf_shift=0.0):
    """Build a reference/current-shaped DataFrame; `shift` offsets all feature means
    (as a real survey shift would), `conf_shift` lowers prediction confidence."""
    data = {c: np.clip(rng.normal(0.4 + shift, 0.05, n), 0, 1) for c in monitoring.FEATURE_COLUMNS}
    data["confidence"] = np.clip(rng.normal(0.85 - conf_shift, 0.03, n), 0, 1)
    data["class_id"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_drift_detected_on_shift():
    rng = np.random.default_rng(0)
    ref = _make_df(rng, 300, shift=0.0, conf_shift=0.0)
    cur = _make_df(rng, 300, shift=0.25, conf_shift=0.25)  # whole feature vector + confidence shift
    summary = monitoring.compute_drift(ref, cur)
    assert summary["dataset_drift"] is True
    assert summary["share_of_drifted_columns"] > 0.5
    assert summary["per_column"]["mean_brightness"]["drift_detected"] is True
    assert summary["prediction_drift_detected"] is True


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(1)
    ref = _make_df(rng, 300)
    cur = _make_df(rng, 300)
    summary = monitoring.compute_drift(ref, cur)
    # Same distribution: K-S may flag the odd column by chance, but not a majority.
    assert summary["dataset_drift"] is False
    assert summary["share_of_drifted_columns"] < 0.5


def test_run_drift_check_requires_min_samples(tmp_path):
    ref = _make_df(np.random.default_rng(2), 50, 0.4, 0.85)
    ref_path = tmp_path / "reference.csv"
    ref.to_csv(ref_path, index=False)
    log = tmp_path / "log.jsonl"
    monitoring.log_prediction(monitoring.extract_features(_img((100, 100, 100))),
                              {"class_id": 0, "label": "Smooth", "confidence": 0.8}, log)
    with pytest.raises(ValueError, match="not enough|fewer"):
        monitoring.run_drift_check(ref_path, log, min_samples=30)
