"""Probabilistic forecasting metrics for trajectory and quantile predictions.

Pure-numpy implementations of pinball loss, CRPS, and quantile MACE.

References
----------
Koenker, R. (2005). "Quantile Regression." Cambridge University Press.
Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules,
    Prediction, and Estimation." JASA, 102(477), 359-378.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "pinball_loss",
    "crps_ensemble",
    "mean_absolute_calibration_error",
]


def pinball_loss(y_true: np.ndarray, y_pred_quantile: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss for a single quantile level tau.

    Defined as ``mean( max(tau * (y - q), (tau - 1) * (y - q)) )``.

    Symmetry property: at ``tau = 0.5`` the pinball loss equals
    ``0.5 * MAE(y, q)``.

    Args:
        y_true: Observed values, shape (N,).
        y_pred_quantile: Predicted quantile of level tau, shape (N,).
        tau: Quantile level in (0, 1).

    Returns:
        Mean pinball loss (>= 0).

    References:
        Koenker (2005), "Quantile Regression."
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1); got {tau}")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred_quantile, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred_quantile shape mismatch: {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def crps_ensemble(y_true: np.ndarray, ensemble_preds: np.ndarray) -> float:
    """Continuous Ranked Probability Score from an ensemble (empirical CDF).

    Uses the energy-form estimator
    ``CRPS = mean_i |X_i - y| - 0.5 * mean_{i,j} |X_i - X_j|``.

    Args:
        y_true: Observed values, shape (N,).
        ensemble_preds: Ensemble samples, shape (N, M) with M samples per
            observation.

    Returns:
        Average CRPS across the N observations (>= 0; lower is better).

    References:
        Gneiting & Raftery (2007), JASA.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    ens = np.asarray(ensemble_preds, dtype=np.float64)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if ens.ndim != 2 or ens.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"ensemble_preds must be (N, M); got {ens.shape} for N={y_true.shape[0]}"
        )
    if ens.shape[1] < 1:
        raise ValueError("ensemble must have at least one sample")

    n, m = ens.shape
    # Term 1: mean |X_i - y| per observation.
    term1 = np.mean(np.abs(ens - y_true[:, None]), axis=1)
    # Term 2: 0.5 * E|X - X'|, computed without forming the (M, M) matrix
    # by sorting and using the closed-form sum_{i<j} (x_(j) - x_(i)).
    sorted_ens = np.sort(ens, axis=1)
    weights = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)
    term2 = (sorted_ens * weights[None, :]).sum(axis=1) / (m * m)
    crps_per_obs = term1 - term2
    return float(np.mean(crps_per_obs))


def mean_absolute_calibration_error(
    pred_quantiles: np.ndarray,
    y_true: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    """Mean absolute calibration error for predicted quantiles.

    For each level ``q`` measures the empirical coverage
    ``mean(y_true <= pred_quantiles[:, q])`` and reports the mean absolute
    deviation from the nominal levels.

    Args:
        pred_quantiles: Predicted quantiles, shape (N, Q).
        y_true: Observed values, shape (N,).
        quantile_levels: Nominal levels, shape (Q,) with values in (0, 1)
            in increasing order.

    Returns:
        Mean absolute calibration error in [0, 1]; 0 = perfect calibration.
    """
    pred_quantiles = np.asarray(pred_quantiles, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    quantile_levels = np.asarray(quantile_levels, dtype=np.float64)
    if pred_quantiles.ndim != 2:
        raise ValueError("pred_quantiles must be 2D (N, Q)")
    if y_true.ndim != 1 or y_true.shape[0] != pred_quantiles.shape[0]:
        raise ValueError("y_true must be 1D with length matching pred_quantiles rows")
    if quantile_levels.ndim != 1 or quantile_levels.shape[0] != pred_quantiles.shape[1]:
        raise ValueError("quantile_levels length must match pred_quantiles columns")
    if np.any(quantile_levels <= 0) or np.any(quantile_levels >= 1):
        raise ValueError("quantile_levels must lie strictly in (0, 1)")
    if np.any(np.diff(quantile_levels) <= 0):
        raise ValueError("quantile_levels must be strictly increasing")

    coverage = np.mean(
        (y_true[:, None] <= pred_quantiles).astype(np.float64), axis=0
    )
    return float(np.mean(np.abs(coverage - quantile_levels)))
