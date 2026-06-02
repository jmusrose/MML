import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_conformal_threshold_uses_finite_sample_quantile():
    from utils.conformal import conformal_threshold

    scores = np.array([0.1, 0.2, 0.4, 0.8])

    assert conformal_threshold(scores, alpha=0.4) == 0.4


def test_conformal_threshold_returns_infinity_when_sample_is_too_small():
    from utils.conformal import conformal_threshold

    scores = np.array([0.1, 0.2])

    assert math.isinf(conformal_threshold(scores, alpha=0.1))


def test_conformal_metrics_are_reported_for_each_modality():
    from utils.conformal import calibrate_conformal, evaluate_conformal

    calibration_logits = {
        "fusion": np.array([[3.0, 0.0], [0.0, 3.0], [1.0, 0.0], [0.0, 1.0]]),
        "rgb": np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]]),
        "depth": np.array([[1.5, 0.0], [0.0, 1.5], [1.0, 0.0], [0.0, 1.0]]),
    }
    calibration_labels = np.array([0, 1, 0, 1])

    result = calibrate_conformal(calibration_logits, calibration_labels, alpha=0.4)

    assert result["alpha"] == 0.4
    assert result["n_calibration"] == 4
    assert set(result["thresholds"]) == {"fusion", "rgb", "depth"}

    test_logits = {
        "fusion": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
        "rgb": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
        "depth": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
    }
    test_labels = np.array([0, 1, 1])

    metrics = evaluate_conformal(test_logits, test_labels, {"fusion": 0.2, "rgb": 0.2, "depth": 0.2})

    assert metrics["fusion"]["coverage"] == 2 / 3
    assert metrics["fusion"]["avg_set_size"] == 2 / 3
    assert metrics["fusion"]["empty_rate"] == 1 / 3
    assert metrics["fusion"]["singleton_rate"] == 2 / 3
    assert metrics["rgb"] == metrics["fusion"]
    assert metrics["depth"] == metrics["fusion"]
