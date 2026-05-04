"""Calibration: empirical coverage and proper interval scores.

Point forecasts + MC-dropout widths without empirical coverage are clinically
unsafe. This replaces README's unqualified "MAE at 3/6/12 months" with a table
that includes coverage and interval score at every horizon.
"""
from __future__ import annotations

import numpy as np


def empirical_coverage(
    y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, nominal: float = 0.9
) -> dict:
    y_true = np.asarray(y_true)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    covered = float(((y_true >= lo) & (y_true <= hi)).mean())
    return {
        "nominal": nominal,
        "empirical": covered,
        "miscoverage": covered - nominal,
        "n": int(y_true.size),
    }


def interval_score(
    y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.1
) -> float:
    """Gneiting-Raftery proper interval score (lower is better)."""
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    under = (2.0 / alpha) * np.maximum(lo - y_true, 0.0)
    over = (2.0 / alpha) * np.maximum(y_true - hi, 0.0)
    return float((width + under + over).mean())


def conformal_quantile(residuals_cal: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal quantile. Defect #126 fix extended for trajectory use."""
    n = len(residuals_cal)
    if n == 0:
        raise ValueError("Empty calibration residuals.")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(np.abs(residuals_cal), level, method="higher"))


def conformal_trajectory_intervals(
    pred: np.ndarray, residuals_cal: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    q = conformal_quantile(residuals_cal, alpha=alpha)
    return pred - q, pred + q


def reliability_table(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Reliability diagram rows for binary calibration audits."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        n = int(mask.sum())
        rows.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n": n,
            "mean_pred": float(probs[mask].mean()) if n else float("nan"),
            "empirical_rate": float(y_true[mask].mean()) if n else float("nan"),
        })
    return rows
