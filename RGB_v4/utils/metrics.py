"""Simplified metric utilities for the DML RGB-Depth baseline.

This is a slimmed-down version of ``CPSC_RGB/utils/metrics.py`` that keeps only
the metric primitives that operate on already-collected ``(softmax, logit,
label, correct)`` arrays. All helpers that depended on temperature scaling or
on the CRL pipeline (e.g. ``get_metric_values``, ``calc_metrics``,
``calc_metrics_for_CPM``) have been removed because the baseline does not yet
include those components.

Public functions
----------------
- ``calc_aurc_eaurc(softmax, correct)``
- ``calc_fpr_aupr(softmax, correct)``
- ``calc_ece(softmax, label, bins=15)``
- ``calc_nll_brier(softmax, logit, label, label_onehot)``
"""

import numpy as np
import torch
from sklearn import metrics


# ---------------------------------------------------------------------------
# AURC / E-AURC
# ---------------------------------------------------------------------------
def calc_aurc_eaurc(softmax, correct):
    """Compute AURC and E-AURC from softmax confidences and per-sample correctness."""
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)
    sort_values = sorted(
        zip(softmax_max[:], correctness[:]), key=lambda x: x[0], reverse=True
    )
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc


# ---------------------------------------------------------------------------
# FPR @ TPR=0.95 / AUPR-Error
# ---------------------------------------------------------------------------
def calc_fpr_aupr(softmax, correct):
    """Compute AUPR-Error and FPR at TPR=0.95 from softmax confidences."""
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]

    aupr_err = metrics.average_precision_score(correctness, softmax_max)

    return aupr_err, fpr_in_tpr_95


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------
def calc_ece(softmax, label, bins=15):
    """Compute Expected Calibration Error with equal-width confidence bins."""
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


# ---------------------------------------------------------------------------
# NLL / Brier Score
# ---------------------------------------------------------------------------
def calc_nll_brier(softmax, logit, label, label_onehot):
    """Compute negative log-likelihood and Brier score."""
    brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label.long())

    return nll.item(), brier_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def calc_nll(log_softmax, label):
    """Average negative log-likelihood given log-softmax outputs and integer labels."""
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum() / len(out)


def coverage_risk(confidence, correctness):
    """Build the risk-coverage curve from a confidence-sorted correctness sequence."""
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list


def aurc_eaurc(risk_list):
    """Compute AURC and Excess-AURC from a risk-coverage risk sequence."""
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    return aurc, eaurc
