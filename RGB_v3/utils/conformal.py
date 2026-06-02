import math

import numpy as np


def _as_1d_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    return labels


def softmax_logits(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits.shape}")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def conformal_threshold(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1D array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    k = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if k > scores.size:
        return float("inf")
    return float(scores[k - 1])


def calibrate_conformal(logits_by_modality, labels, alpha):
    labels = _as_1d_labels(labels)
    thresholds = {}

    for modality, logits in logits_by_modality.items():
        probs = softmax_logits(logits)
        if probs.shape[0] != labels.size:
            raise ValueError(
                f"{modality} has {probs.shape[0]} rows but labels has {labels.size}"
            )
        scores = 1.0 - probs[np.arange(labels.size), labels]
        thresholds[modality] = conformal_threshold(scores, alpha)

    return {
        "alpha": float(alpha),
        "n_calibration": int(labels.size),
        "thresholds": thresholds,
    }


def evaluate_conformal(logits_by_modality, labels, thresholds):
    labels = _as_1d_labels(labels)
    metrics = {}

    for modality, logits in logits_by_modality.items():
        if modality not in thresholds:
            raise KeyError(f"missing conformal threshold for modality {modality}")

        probs = softmax_logits(logits)
        if probs.shape[0] != labels.size:
            raise ValueError(
                f"{modality} has {probs.shape[0]} rows but labels has {labels.size}"
            )

        threshold = thresholds[modality]
        if math.isinf(threshold):
            prediction_sets = np.ones_like(probs, dtype=bool)
        else:
            prediction_sets = probs >= (1.0 - threshold)

        set_sizes = prediction_sets.sum(axis=1)
        covered = prediction_sets[np.arange(labels.size), labels]
        metrics[modality] = {
            "coverage": float(np.mean(covered)),
            "avg_set_size": float(np.mean(set_sizes)),
            "empty_rate": float(np.mean(set_sizes == 0)),
            "singleton_rate": float(np.mean(set_sizes == 1)),
        }

    return metrics
