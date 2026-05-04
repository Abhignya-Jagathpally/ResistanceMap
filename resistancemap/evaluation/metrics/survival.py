"""Survival-analysis metrics for time-to-event evaluation.

Pure-numpy implementations. ``scipy`` is optional and only used when present
to accelerate Kaplan-Meier interpolation; otherwise a numpy fallback is used.

References
----------
Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L., & Rosati, R. A. (1982).
    "Evaluating the yield of medical tests." JAMA, 247(18), 2543-2546.
Graf, E., Schmoor, C., Sauerbrei, W., & Schumacher, M. (1999).
    "Assessment and comparison of prognostic classification schemes for
    survival data." Statistics in Medicine, 18(17-18), 2529-2545.
Uno, H., Cai, T., Tian, L., & Wei, L. J. (2007).
    "Evaluating prediction rules for t-year survivors with censored
    regression models." JASA, 102(478), 527-537.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # scipy is optional; metrics work without it.
    import scipy  # noqa: F401

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised in environments without scipy
    _HAS_SCIPY = False
    logger.debug("scipy not available; survival metrics will use numpy fallbacks")

__all__ = [
    "integrated_brier_score",
    "time_dependent_auc",
    "concordance_index",
]


def _validate_survival_inputs(
    times: np.ndarray, events: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate paired (times, events) survival arrays."""
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events)
    if times.ndim != 1 or events.ndim != 1:
        raise ValueError("times and events must be 1D arrays")
    if times.shape[0] != events.shape[0]:
        raise ValueError(
            f"times/events length mismatch: {times.shape[0]} vs {events.shape[0]}"
        )
    if times.shape[0] == 0:
        raise ValueError("times/events are empty")
    if np.any(times < 0):
        raise ValueError("times must be non-negative")
    events_int = events.astype(int)
    if np.any((events_int != 0) & (events_int != 1)):
        raise ValueError("events must be 0 (censored) or 1 (event)")
    return times, events_int


def _kaplan_meier_censoring(
    times: np.ndarray, events: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute KM estimator of the *censoring* survival function G(t).

    Returns sorted unique event/censoring times and G(t) at each, where the
    "events" for the censoring KM are the censored observations.
    """
    censor_events = 1 - events  # flip: censoring is the event of interest
    order = np.argsort(times)
    t_sorted = times[order]
    c_sorted = censor_events[order]
    unique_t, idx = np.unique(t_sorted, return_index=True)
    g = np.ones(unique_t.shape[0], dtype=np.float64)
    n_at_risk = t_sorted.shape[0]
    surv = 1.0
    for i, ut in enumerate(unique_t):
        end = idx[i + 1] if i + 1 < len(idx) else len(t_sorted)
        d = c_sorted[idx[i]:end].sum()
        if n_at_risk > 0 and d > 0:
            surv *= 1.0 - d / n_at_risk
        g[i] = surv
        n_at_risk -= (end - idx[i])
    return unique_t, g


def _step_eval(grid_t: np.ndarray, grid_v: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Evaluate a right-continuous step function at ``query`` points.

    The step function takes value ``grid_v[i]`` on the half-open interval
    ``[grid_t[i], grid_t[i+1])`` (and value 1 before ``grid_t[0]``).
    """
    out = np.ones_like(query, dtype=np.float64)
    idx = np.searchsorted(grid_t, query, side="right") - 1
    valid = idx >= 0
    out[valid] = grid_v[idx[valid]]
    return out


def integrated_brier_score(
    times: np.ndarray,
    events: np.ndarray,
    surv_pred: np.ndarray,
    eval_times: np.ndarray,
) -> float:
    """Integrated Brier Score with IPCW (Graf et al. 1999).

    Computes the time-integrated, censoring-weighted Brier score over
    ``eval_times`` using the Kaplan-Meier estimate of the censoring
    distribution G.

    Args:
        times: Observed times of shape (N,).
        events: Event indicators 0/1 of shape (N,).
        surv_pred: Predicted survival probabilities at each eval time;
            shape (N, T) where T == len(eval_times).
        eval_times: Strictly increasing 1D array of evaluation time points.

    Returns:
        Integrated Brier score (>= 0; lower is better).

    References:
        Graf et al. (1999), Statistics in Medicine.
    """
    times, events = _validate_survival_inputs(times, events)
    surv_pred = np.asarray(surv_pred, dtype=np.float64)
    eval_times = np.asarray(eval_times, dtype=np.float64)
    if eval_times.ndim != 1 or eval_times.shape[0] < 1:
        raise ValueError("eval_times must be a non-empty 1D array")
    if eval_times.shape[0] >= 2 and np.any(np.diff(eval_times) <= 0):
        raise ValueError("eval_times must be strictly increasing")
    if surv_pred.ndim != 2 or surv_pred.shape != (times.shape[0], eval_times.shape[0]):
        raise ValueError(
            f"surv_pred must have shape ({times.shape[0]}, {eval_times.shape[0]}); "
            f"got {surv_pred.shape}"
        )
    if np.any(surv_pred < 0) or np.any(surv_pred > 1.0 + 1e-6):
        raise ValueError("surv_pred values must lie in [0, 1]")

    g_t, g_v = _kaplan_meier_censoring(times, events)
    # G evaluated just before t_i (use t-epsilon for events occurring exactly at t_i).
    bs_per_t = np.zeros(eval_times.shape[0], dtype=np.float64)
    n = times.shape[0]
    for k, t in enumerate(eval_times):
        g_at_t = _step_eval(g_t, g_v, np.array([t]))[0]
        # G(t_i-) for IPCW: use the value just before each subject's event time.
        g_at_ti = _step_eval(g_t, g_v, np.maximum(times - 1e-12, 0.0))
        # Subjects with event before or at t and event=1 (Category 1)
        mask_event = (times <= t) & (events == 1)
        # Subjects still at risk at t (Category 2)
        mask_risk = times > t
        contrib = np.zeros(n, dtype=np.float64)
        # Category 1
        with np.errstate(divide="ignore", invalid="ignore"):
            denom_e = np.where(g_at_ti > 0, g_at_ti, np.nan)
            contrib_e = (surv_pred[:, k] ** 2) / denom_e
        contrib[mask_event] = np.where(
            np.isnan(contrib_e[mask_event]), 0.0, contrib_e[mask_event]
        )
        # Category 2: Controls must also be weighted by IPCW at eval time
        if g_at_t > 0:
            contrib[mask_risk] = ((1.0 - surv_pred[mask_risk, k]) ** 2) / g_at_t
        bs_per_t[k] = contrib.sum() / n

    # Single-horizon special case: integration is undefined; return the
    # mean Brier score at that horizon.
    if eval_times.shape[0] == 1:
        return float(bs_per_t[0])

    # Trapezoidal time integration normalized by the eval window.
    duration = eval_times[-1] - eval_times[0]
    if duration <= 0:
        raise ValueError("eval_times must span a positive interval")
    # np.trapezoid is the numpy >= 2.0 name; np.trapz is the legacy alias.
    trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(trapz(bs_per_t, eval_times) / duration)


def time_dependent_auc(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Cumulative/dynamic time-dependent AUC (Uno et al. 2007).

    For each ``t`` in ``eval_times`` defines cases as subjects with an
    observed event by ``t`` and controls as subjects still event-free at
    ``t``, then computes an IPCW-weighted AUC.

    Args:
        times: Observed times of shape (N,).
        events: Event indicators 0/1 of shape (N,).
        risk_scores: Marker values of shape (N,); higher = higher risk.
        eval_times: 1D array of evaluation time points.

    Returns:
        AUC values, shape (len(eval_times),). Time points with no valid
        case/control comparisons receive NaN.

    References:
        Uno et al. (2007), JASA.
    """
    times, events = _validate_survival_inputs(times, events)
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    eval_times = np.asarray(eval_times, dtype=np.float64)
    if risk_scores.ndim != 1 or risk_scores.shape[0] != times.shape[0]:
        raise ValueError("risk_scores must be 1D and match times length")
    if eval_times.ndim != 1 or eval_times.shape[0] == 0:
        raise ValueError("eval_times must be a non-empty 1D array")

    g_t, g_v = _kaplan_meier_censoring(times, events)
    g_at_ti = _step_eval(g_t, g_v, np.maximum(times - 1e-12, 0.0))
    aucs = np.full(eval_times.shape[0], np.nan, dtype=np.float64)
    n = times.shape[0]

    for k, t in enumerate(eval_times):
        case_mask = (times <= t) & (events == 1)
        control_mask = times > t
        if not case_mask.any() or not control_mask.any():
            continue
        # IPCW weights for cases (1 / G(T_i-))
        with np.errstate(divide="ignore", invalid="ignore"):
            w_case = np.where(g_at_ti > 0, 1.0 / g_at_ti, 0.0)
        w_case = w_case * case_mask
        w_control = control_mask.astype(np.float64)

        num = 0.0
        denom = 0.0
        # Pairwise loop is O(N^2) but stays in numpy via broadcasting.
        case_idx = np.where(case_mask)[0]
        ctrl_idx = np.where(control_mask)[0]
        rs_case = risk_scores[case_idx]
        rs_ctrl = risk_scores[ctrl_idx]
        wc = w_case[case_idx]
        wt = w_control[ctrl_idx]
        # Indicator: case score > control score (with 0.5 credit for ties).
        diff = rs_case[:, None] - rs_ctrl[None, :]
        ind = (diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)
        weight_mat = wc[:, None] * wt[None, :]
        num = float((ind * weight_mat).sum())
        denom = float(weight_mat.sum())
        if denom > 0:
            aucs[k] = num / denom
        del diff, ind, weight_mat  # explicit cleanup for large N
        _ = n  # silence linter; n unused after broadcasting refactor
    return aucs


def concordance_index(
    times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray
) -> float:
    """Harrell's concordance index (C-index) with tie handling.

    Counts admissible pairs ``(i, j)`` where ``i`` had an event and
    ``T_i < T_j``. A pair is concordant if the higher risk_score belongs
    to ``i``; ties in risk score contribute 0.5.

    Args:
        times: Observed times of shape (N,).
        events: Event indicators 0/1 of shape (N,).
        risk_scores: Marker values of shape (N,); higher = higher risk.

    Returns:
        C-index in [0, 1]; 0.5 = chance, 1.0 = perfect concordance.

    References:
        Harrell et al. (1982), JAMA.
    """
    times, events = _validate_survival_inputs(times, events)
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    if risk_scores.ndim != 1 or risk_scores.shape[0] != times.shape[0]:
        raise ValueError("risk_scores must be 1D and match times length")

    n = times.shape[0]
    num = 0.0
    denom = 0.0
    # Vectorized over j for each event subject i.
    event_idx = np.where(events == 1)[0]
    for i in event_idx:
        ti = times[i]
        ri = risk_scores[i]
        # Admissible: j has T_j > T_i (j may be censored or event)
        adm = times > ti
        if not adm.any():
            continue
        rj = risk_scores[adm]
        denom += adm.sum()
        num += float((ri > rj).sum()) + 0.5 * float((ri == rj).sum())
    if denom == 0:
        raise ValueError("no admissible pairs; cannot compute C-index")
    _ = n
    return float(num / denom)