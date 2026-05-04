"""Probability calibration metrics for classifier evaluation.

Pure-numpy implementations of standard calibration metrics. All functions
operate on caller-supplied arrays and never fabricate data.

References
----------
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    "On Calibration of Modern Neural Networks." ICML 2017.
Brier, G. W. (1950).
    "Verification of Forecasts Expressed in Terms of Probability."
    Monthly Weather Review, 78(1), 1-3.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_curve",
    "brier_score_multiclass",
]


def _validate_probs_labels(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate predicted probabilities and integer labels.

    Args:
        probs: (N,) for binary or (N, K) for K-way multiclass.
        labels: (N,) integer class labels.

    Returns:
        Validated (probs, labels) cast to numpy arrays.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)

    if probs.ndim not in (1, 2):
        raise ValueError(f"probs must be 1D (binary) or 2D (multiclass); got shape {probs.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D; got shape {labels.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs and labels length mismatch: {probs.shape[0]} vs {labels.shape[0]}"
        )
    if probs.shape[0] == 0:
        raise ValueError("probs/labels are empty")
    if not np.issubdtype(labels.dtype, np.integer):
        if np.any(labels != labels.astype(int)):
            raise ValueError("labels must be integer-valued")
        labels = labels.astype(int)
    if np.any(probs < 0.0) or np.any(probs > 1.0 + 1e-6):
        raise ValueError("probs must lie in [0, 1]")
    if probs.ndim == 2:
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            raise ValueError("multiclass probs rows must sum to 1")
    return probs, labels


def _confidences_and_correct(
    probs: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce probs/labels to (confidence, correct) per sample.

    For binary 1D probs, the confidence is max(p, 1 - p) and the predicted
    class is round(p). For 2D multiclass, the confidence is the top-1
    probability and the predicted class is argmax.
    """
    if probs.ndim == 1:
        pred = (probs >= 0.5).astype(int)
        conf = np.where(pred == 1, probs, 1.0 - probs)
    else:
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
    correct = (pred == labels).astype(np.float64)
    return conf, correct


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error (ECE) with equal-width confidence bins.

    ECE partitions [0, 1] into ``n_bins`` equal-width intervals and computes
    a sample-weighted average of |accuracy - confidence| within each bin.

    Args:
        probs: Predicted probabilities. Either (N,) for binary classification
            or (N, K) for multiclass classification.
        labels: Integer class labels of shape (N,).
        n_bins: Number of equal-width confidence bins. Defaults to 15.

    Returns:
        ECE in [0, 1]. Lower is better; 0 means perfect calibration.

    References:
        Guo et al. (2017), "On Calibration of Modern Neural Networks."
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    probs, labels = _validate_probs_labels(probs, labels)
    conf, correct = _confidences_and_correct(probs, labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = conf.shape[0]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Right-inclusive on the last bin so confidence == 1 is captured
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def maximum_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Maximum Calibration Error (MCE) over equal-width confidence bins.

    Returns the worst-case |accuracy - confidence| across populated bins.
    Useful for surfacing single-region miscalibration that ECE may average out.

    Args:
        probs: Predicted probabilities; (N,) binary or (N, K) multiclass.
        labels: Integer class labels of shape (N,).
        n_bins: Number of equal-width confidence bins. Defaults to 15.

    Returns:
        MCE in [0, 1].

    References:
        Guo et al. (2017), "On Calibration of Modern Neural Networks."
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    probs, labels = _validate_probs_labels(probs, labels)
    conf, correct = _confidences_and_correct(probs, labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        if gap > mce:
            mce = gap
    return float(mce)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability diagram data: per-bin (center, accuracy, confidence).

    Args:
        probs: Predicted probabilities; (N,) binary or (N, K) multiclass.
        labels: Integer class labels of shape (N,).
        n_bins: Number of equal-width confidence bins. Defaults to 15.

    Returns:
        Tuple ``(bin_centers, accuracies, confidences)``, each of shape
        (n_bins,). Empty bins receive NaN entries.

    References:
        Guo et al. (2017), "On Calibration of Modern Neural Networks."
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    probs, labels = _validate_probs_labels(probs, labels)
    conf, correct = _confidences_and_correct(probs, labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    accs = np.full(n_bins, np.nan, dtype=np.float64)
    confs = np.full(n_bins, np.nan, dtype=np.float64)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        accs[i] = correct[mask].mean()
        confs[i] = conf[mask].mean()
    return centers, accs, confs


def brier_score_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error vs one-hot labels).

    For binary 1D probs, computes the standard binary Brier score
    ``mean((p - y) ** 2)``. For multiclass (N, K), computes
    ``mean(sum_k (p_k - y_k) ** 2)`` where y is one-hot.

    Args:
        probs: Predicted probabilities; (N,) binary or (N, K) multiclass.
        labels: Integer class labels of shape (N,).

    Returns:
        Brier score >= 0; lower is better.

    References:
        Brier (1950), "Verification of Forecasts Expressed in Terms of Probability."
    """
    probs, labels = _validate_probs_labels(probs, labels)
    if probs.ndim == 1:
        return float(np.mean((probs - labels.astype(np.float64)) ** 2))
    n, k = probs.shape
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError(f"labels must lie in [0, {k - 1}]; got [{labels.min()}, {labels.max()}]")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
