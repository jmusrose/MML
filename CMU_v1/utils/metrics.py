"""CMU-MOSI/MOSEI standard 5-metric evaluation suite.

This module provides ``compute_cmu_metrics`` which derives the canonical
five evaluation metrics used by the CMU multimodal-sentiment community from
a single scalar regression prediction:

- ``mae``  : mean absolute error
- ``corr`` : Pearson correlation between ``pred`` and ``label``
- ``acc7`` : 7-class accuracy on ``np.clip(., -3, 3)`` of BOTH pred and label,
             then ``np.round`` (matching MMIM / Self-MM convention)
- ``acc2`` : binary (sign) accuracy on the ``label != 0`` subset
- ``f1``   : weighted F1 on the same ``label != 0`` subset

Fallback behaviour (Requirement 13.5):
    * ``corr = 0.0`` when ``pred.size < 2`` or either ``pred`` / ``label``
      has zero variance, or when ``pearsonr`` returns a non-finite value.
    * ``acc2 = 0.0`` and ``f1 = 0.0`` when the ``label != 0`` mask is empty.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _multiclass_acc(preds: np.ndarray, truths: np.ndarray) -> float:
    """CMU community standard ``multiclass_acc`` (e.g. MMIM, Self-MM, MISA).

    Both ``preds`` and ``truths`` are rounded to the nearest integer; the
    accuracy is the fraction of positions where the rounded values match.
    Note: this function does **not** clip — clipping is done explicitly by
    the caller so the contract is symmetric on pred and label.
    """
    return float(np.sum(np.round(preds) == np.round(truths)) / float(len(truths)))


def compute_cmu_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    """Compute the standard 5-metric evaluation suite for CMU-MOSI/MOSEI.

    Args:
        pred:  1-D array-like of scalar predictions, typically produced by
               ``both_output.squeeze(-1)`` on the fused regression branch.
        label: 1-D array-like of ground-truth scalar labels in ``[-3, 3]``.

    Returns:
        Dict with float values for keys ``mae``, ``corr``, ``acc7``,
        ``acc2``, ``f1``.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)

    # Mean absolute error
    mae = float(np.mean(np.abs(pred - label)))

    # Pearson correlation with safe fallback
    if pred.size < 2 or pred.std() == 0.0 or label.std() == 0.0:
        corr = 0.0
    else:
        from scipy.stats import pearsonr
        corr_val, _ = pearsonr(pred, label)
        corr = float(corr_val) if np.isfinite(corr_val) else 0.0

    # 7-class accuracy: clip BOTH pred and label to [-3, 3], then round.
    # This is exactly the MMIM / Self-MM / MISA convention:
    #     test_preds_a7  = np.clip(test_preds,  -3., 3.)
    #     test_truth_a7  = np.clip(test_truth,  -3., 3.)
    #     mult_a7        = multiclass_acc(test_preds_a7, test_truth_a7)
    pred_a7 = np.clip(pred, -3.0, 3.0)
    label_a7 = np.clip(label, -3.0, 3.0)
    acc7 = _multiclass_acc(pred_a7, label_a7)

    # Acc-2 / F1 on the non-zero-label subset
    mask = label != 0.0
    if mask.sum() == 0:
        acc2 = 0.0
        f1 = 0.0
    else:
        pred_pos = (pred[mask] > 0.0)
        label_pos = (label[mask] > 0.0)
        acc2 = float(accuracy_score(label_pos, pred_pos))
        f1 = float(f1_score(label_pos, pred_pos, average='weighted', zero_division=0))

    return {"mae": mae, "corr": corr, "acc7": acc7, "acc2": acc2, "f1": f1}
