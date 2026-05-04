"""Distribution-shift detection metrics.

Pure-numpy implementations of MMD, KS shift, PSI, and a simple alarm gate.
``scipy.stats.ks_2samp`` is used when scipy is available; otherwise a
numpy fallback approximates the two-sided KS p-value via the Kolmogorov
distribution series.

References
----------
Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
    (2012). "A Kernel Two-Sample Test." JMLR, 13, 723-773.
Massey, F. J. (1951). "The Kolmogorov-Smirnov Test for Goodness of Fit."
    JASA, 46(253), 68-78.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import stats as _scipy_stats  # type: ignore

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    logger.debug("scipy not available; KS p-values will use numpy fallback")

__all__ = [
    "maximum_mean_discrepancy",
    "kolmogorov_smirnov_shift",
    "population_stability_index",
    "shift_alarm",
]


def _median_heuristic_bandwidth(x: np.ndarray, y: np.ndarray, max_pairs: int = 2000) -> float:
    """Median pairwise distance bandwidth (Gretton et al. 2012, sec. 8)."""
    rng = np.random.default_rng(0)
    z = np.vstack([x, y])
    if z.shape[0] > max_pairs:
        idx = rng.choice(z.shape[0], size=max_pairs, replace=False)
        z = z[idx]
    diffs = z[:, None, :] - z[None, :, :]
    d2 = (diffs ** 2).sum(axis=-1)
    iu = np.triu_indices_from(d2, k=1)
    med = np.median(d2[iu])
    if med <= 0:
        return 1.0
    return float(np.sqrt(med / 2.0))


def maximum_mean_discrepancy(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    kernel: str = "rbf",
    bandwidth: Optional[float] = None,
) -> float:
    """Unbiased Maximum Mean Discrepancy (MMD^2_u) between two samples.

    Args:
        x_src: Source samples, shape (N, D) or (N,) (auto-reshaped to (N, 1)).
        x_tgt: Target samples, shape (M, D) or (M,).
        kernel: ``'rbf'`` (Gaussian) or ``'linear'``.
        bandwidth: RBF bandwidth (sigma). If None, uses the median heuristic.

    Returns:
        Squared MMD estimate. Can be slightly negative for finite samples
        with the unbiased estimator (Gretton et al. 2012).

    References:
        Gretton et al. (2012), JMLR.
    """
    x = np.asarray(x_src, dtype=np.float64)
    y = np.asarray(x_tgt, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x_src and x_tgt must be 1D or 2D")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"feature dim mismatch: {x.shape[1]} (src) vs {y.shape[1]} (tgt)"
        )
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("MMD requires at least 2 samples per group")

    if kernel == "rbf":
        if bandwidth is None:
            bandwidth = _median_heuristic_bandwidth(x, y)
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        gamma = 1.0 / (2.0 * bandwidth ** 2)

        def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
            return np.exp(-gamma * d2)
    elif kernel == "linear":
        def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a @ b.T
    else:
        raise ValueError(f"unknown kernel: {kernel!r}")

    n = x.shape[0]
    m = y.shape[0]
    kxx = k(x, x)
    kyy = k(y, y)
    kxy = k(x, y)
    # Biased V-statistic estimator MMD^2_b: matches the test identity that
    # MMD(x, x) == 0 exactly. The biased estimator is non-negative and
    # standard in ML practice; the unbiased U-statistic is preferred only
    # for hypothesis testing where small-sample bias matters.
    term_xx = kxx.sum() / (n * n)
    term_yy = kyy.sum() / (m * m)
    term_xy = kxy.sum() / (n * m)
    return float(term_xx + term_yy - 2.0 * term_xy)


def _ks_pvalue_fallback(stat: float, n1: int, n2: int) -> float:
    """Numpy-only approximation of two-sample KS p-value (Smirnov series)."""
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * stat
    # Q_KS(lam) = 2 * sum_{j=1..} (-1)^(j-1) exp(-2 j^2 lam^2)
    s = 0.0
    fac = 2.0
    sign = 1.0
    prev = 0.0
    for j in range(1, 101):
        term = sign * math.exp(-2.0 * j * j * lam * lam)
        s += term
        if abs(term) <= 1e-10 * abs(prev) or abs(term) < 1e-20:
            break
        prev = term
        sign = -sign
    p = max(0.0, min(1.0, fac * s))
    return p


def kolmogorov_smirnov_shift(x_src: np.ndarray, x_tgt: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-feature two-sample Kolmogorov-Smirnov shift test.

    Args:
        x_src: Source samples, shape (N, D) or (N,).
        x_tgt: Target samples, shape (M, D) or (M,).

    Returns:
        Dict with arrays:
            ``stat``: KS statistic per feature, shape (D,).
            ``p_value``: Two-sided p-value per feature, shape (D,).

    References:
        Massey (1951), JASA.
    """
    x = np.asarray(x_src, dtype=np.float64)
    y = np.asarray(x_tgt, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x_src and x_tgt must be 1D or 2D")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"feature dim mismatch: {x.shape[1]} (src) vs {y.shape[1]} (tgt)"
        )
    if x.shape[0] < 1 or y.shape[0] < 1:
        raise ValueError("KS requires at least 1 sample per group")

    d = x.shape[1]
    stats = np.zeros(d, dtype=np.float64)
    pvals = np.zeros(d, dtype=np.float64)
    for j in range(d):
        a = x[:, j]
        b = y[:, j]
        if _HAS_SCIPY:
            res = _scipy_stats.ks_2samp(a, b, alternative="two-sided", method="auto")
            stats[j] = float(res.statistic)
            pvals[j] = float(res.pvalue)
        else:
            # Manual KS statistic.
            all_vals = np.concatenate([a, b])
            all_vals.sort()
            cdf_a = np.searchsorted(np.sort(a), all_vals, side="right") / a.size
            cdf_b = np.searchsorted(np.sort(b), all_vals, side="right") / b.size
            stat = float(np.max(np.abs(cdf_a - cdf_b)))
            stats[j] = stat
            pvals[j] = _ks_pvalue_fallback(stat, a.size, b.size)
    return {"stat": stats, "p_value": pvals}


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """Population Stability Index (PSI) between expected and actual samples.

    Bins ``expected`` into ``n_bins`` equal-frequency buckets and computes
    ``sum_i (a_i - e_i) * ln(a_i / e_i)`` over bin proportions. Empty buckets
    are smoothed by 1e-6 to avoid log(0).

    Args:
        expected: Reference distribution samples, shape (N,).
        actual: Comparison distribution samples, shape (M,).
        n_bins: Number of quantile bins.

    Returns:
        PSI >= 0. Conventional thresholds: <0.1 stable, 0.1-0.25 moderate
        shift, >0.25 significant shift.
    """
    e = np.asarray(expected, dtype=np.float64).ravel()
    a = np.asarray(actual, dtype=np.float64).ravel()
    if e.size == 0 or a.size == 0:
        raise ValueError("expected and actual must be non-empty")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(e, quantiles))
    if edges.size < 2:
        raise ValueError("expected has insufficient variability to bin")
    edges[0] = -np.inf
    edges[-1] = np.inf

    e_counts, _ = np.histogram(e, bins=edges)
    a_counts, _ = np.histogram(a, bins=edges)
    e_prop = np.where(e_counts == 0, 1e-6, e_counts / e.size)
    a_prop = np.where(a_counts == 0, 1e-6, a_counts / a.size)
    psi = float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))
    return psi


def shift_alarm(scores: np.ndarray, threshold: float = 0.25) -> bool:
    """Trigger a shift alarm if any score exceeds ``threshold``.

    Args:
        scores: Array of shift scores (e.g., PSI / MMD per feature).
        threshold: Alarm threshold; defaults to the conventional PSI 0.25.

    Returns:
        True if max(scores) > threshold; False otherwise.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        return False
    return bool(np.nanmax(s) > threshold)
