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


def evaluate_conformal(logits_by_modality, labels, thresholds, tau=1.0):
    labels = _as_1d_labels(labels)
    tau = float(tau)
    if tau < 0:
        raise ValueError("tau must be non-negative")

    metrics = {}
    per_modality = {}
    reliabilities = []
    modality_names = []

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
        uncertainty = set_sizes.astype(np.float64) / float(probs.shape[1])
        reliability = np.exp(-tau * uncertainty)
        per_modality[modality] = {
            "covered": covered,
            "set_sizes": set_sizes,
            "uncertainty": uncertainty,
            "reliability": reliability,
        }
        reliabilities.append(reliability)
        modality_names.append(modality)

    reliability_matrix = np.vstack(reliabilities)
    weight_matrix = reliability_matrix / np.sum(reliability_matrix, axis=0, keepdims=True)

    for idx, modality in enumerate(modality_names):
        values = per_modality[modality]
        metrics[modality] = {
            "coverage": float(np.mean(values["covered"])),
            "avg_set_size": float(np.mean(values["set_sizes"])),
            "empty_rate": float(np.mean(values["set_sizes"] == 0)),
            "singleton_rate": float(np.mean(values["set_sizes"] == 1)),
            "avg_uncertainty": float(np.mean(values["uncertainty"])),
            "avg_reliability": float(np.mean(values["reliability"])),
            "avg_weight": float(np.mean(weight_matrix[idx])),
        }

    return metrics
