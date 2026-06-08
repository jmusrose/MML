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


def test_conformal_metrics_are_reported_for_each_mvsa_modality():
    from utils.conformal import calibrate_conformal, evaluate_conformal

    calibration_logits = {
        "fused": np.array([[3.0, 0.0], [0.0, 3.0], [1.0, 0.0], [0.0, 1.0]]),
        "text": np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]]),
        "image": np.array([[1.5, 0.0], [0.0, 1.5], [1.0, 0.0], [0.0, 1.0]]),
    }
    calibration_labels = np.array([0, 1, 0, 1])

    result = calibrate_conformal(calibration_logits, calibration_labels, alpha=0.4)

    assert result["alpha"] == 0.4
    assert result["n_calibration"] == 4
    assert set(result["thresholds"]) == {"fused", "text", "image"}

    test_logits = {
        "fused": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
        "text": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
        "image": np.array([[4.0, 0.0], [1.0, 0.0], [0.0, 4.0]]),
    }
    test_labels = np.array([0, 1, 1])

    metrics = evaluate_conformal(
        test_logits,
        test_labels,
        {"fused": 0.2, "text": 0.2, "image": 0.2},
    )

    assert metrics["fused"]["coverage"] == 2 / 3
    assert metrics["fused"]["avg_set_size"] == 2 / 3
    assert metrics["fused"]["empty_rate"] == 1 / 3
    assert metrics["fused"]["singleton_rate"] == 2 / 3
    assert metrics["text"] == metrics["fused"]
    assert metrics["image"] == metrics["fused"]


def test_conformal_uncertainty_weights_use_prediction_set_sizes():
    from utils.conformal import evaluate_conformal

    logits = {
        "fused": np.array([[4.0, 0.0], [0.0, 4.0]]),
        "text": np.array([[4.0, 0.0], [0.0, 4.0]]),
        "image": np.array([[4.0, 0.0], [0.0, 4.0]]),
    }
    labels = np.array([0, 1])
    tau = 2.0

    metrics = evaluate_conformal(
        logits,
        labels,
        {"fused": float("inf"), "text": 0.0, "image": 0.2},
        tau=tau,
    )

    expected_reliability = {
        "fused": math.exp(-tau * 1.0),
        "text": math.exp(-tau * 0.0),
        "image": math.exp(-tau * 0.5),
    }
    denom = sum(expected_reliability.values())

    assert metrics["fused"]["avg_uncertainty"] == 1.0
    assert metrics["text"]["avg_uncertainty"] == 0.0
    assert metrics["image"]["avg_uncertainty"] == 0.5
    for modality in expected_reliability:
        assert metrics[modality]["avg_reliability"] == expected_reliability[modality]
        assert metrics[modality]["avg_weight"] == expected_reliability[modality] / denom


def test_conformal_uncertainty_from_logits_reports_average_set_size_ratio():
    from utils.conformal import conformal_uncertainty_from_logits

    logits = {
        "text": np.array([[4.0, 0.0], [0.0, 4.0]]),
        "image": np.array([[4.0, 0.0], [0.0, 4.0]]),
    }

    uncertainties = conformal_uncertainty_from_logits(
        logits,
        {"text": float("inf"), "image": 0.2},
    )

    assert uncertainties["text"] == 1.0
    assert uncertainties["image"] == 0.5
