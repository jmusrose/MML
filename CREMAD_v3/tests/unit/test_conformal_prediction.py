import math
import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_conformal_threshold_uses_finite_sample_quantile():
    from utils.conformal import conformal_threshold

    scores = np.array([0.1, 0.2, 0.4, 0.8])

    assert conformal_threshold(scores, alpha=0.4) == 0.4


def test_conformal_threshold_returns_infinity_when_sample_is_too_small():
    from utils.conformal import conformal_threshold

    scores = np.array([0.1, 0.2])

    assert math.isinf(conformal_threshold(scores, alpha=0.1))


def test_conformal_metrics_are_reported_for_each_cremad_modality():
    from utils.conformal import calibrate_conformal, evaluate_conformal

    calibration_logits = {
        "fused": np.array([[3.0, 0.0], [0.0, 3.0], [1.0, 0.0], [0.0, 1.0]]),
        "audio": np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]]),
        "video": np.array([[1.5, 0.0], [0.0, 1.5], [1.0, 0.0], [0.0, 1.0]]),
    }
    calibration_labels = np.array([0, 1, 0, 1])

    result = calibrate_conformal(calibration_logits, calibration_labels, alpha=0.4)

    assert result["alpha"] == 0.4
    assert result["n_calibration"] == 4
    assert set(result["thresholds"]) == {"fused", "audio", "video"}

    test_logits = {
        "fused": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
        "audio": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
        "video": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
    }
    test_labels = np.array([0, 1, 1])

    metrics = evaluate_conformal(
        test_logits,
        test_labels,
        {"fused": 0.2, "audio": 0.2, "video": 0.2},
    )

    assert metrics["fused"]["coverage"] == 2 / 3
    assert metrics["fused"]["avg_set_size"] == 2 / 3
    assert metrics["fused"]["empty_rate"] == 1 / 3
    assert metrics["fused"]["singleton_rate"] == 2 / 3
    assert metrics["audio"] == metrics["fused"]
    assert metrics["video"] == metrics["fused"]
