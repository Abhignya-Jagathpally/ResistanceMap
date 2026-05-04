"""Simple survival baseline.

Pure-numpy Kaplan-Meier estimator stratified by a single covariate. If
scipy is available, additionally fits a one-covariate Cox proportional-
hazards model via Newton-Raphson on the partial likelihood for the
``predict`` head; otherwise the predict head returns 1 - S(t_max) per
patient based on the KM curve of the patient's stratum.

Operates on REAL survival data; raises ``ValueError`` for missing or
malformed inputs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from resistancemap.evaluation.baselines.base import Baseline

logger = logging.getLogger(__name__)

try:  # scipy optional
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_minimize = None  # type: ignore[assignment]
    _HAS_SCIPY = False


class SimpleSurvivalBaseline(Baseline):
    """Stratified Kaplan-Meier (+ optional Cox).

    Inputs to ``fit``:

    - ``X_train``: 2D array. Column 0 is interpreted as the stratum
      indicator (e.g. cytogenetic risk band). Additional columns are used
      only if scipy is available, for the optional Cox head.
    - ``y_train``: 2D array of shape ``(n, 2)`` where column 0 is event
      time and column 1 is the event indicator (1 = event, 0 = censored).
    """

    def __init__(self) -> None:
        super().__init__()
        self.km_curves_: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        self.cox_beta_: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "simple_survival"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        meta: Optional[dict[str, Any]] = None,
    ) -> "SimpleSurvivalBaseline":
        X = self._require_nonempty(X_train, "X_train")
        y = self._require_nonempty(y_train, "y_train")
        if X.ndim != 2 or X.shape[1] < 1:
            raise ValueError(
                "SimpleSurvivalBaseline expects X with shape (n, >=1); "
                "column 0 must be the stratum indicator."
            )
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError(
                "SimpleSurvivalBaseline expects y with shape (n, 2): "
                "[time, event]."
            )

        strata = X[:, 0]
        times = y[:, 0].astype(np.float64)
        events = y[:, 1].astype(np.int32)

        for s in np.unique(strata):
            mask = strata == s
            t_s = times[mask]
            e_s = events[mask]
            curve = _kaplan_meier(t_s, e_s)
            self.km_curves_[s] = curve

        # Optional Cox if scipy is available and there are extra covariates.
        if _HAS_SCIPY and X.shape[1] > 1:
            try:
                self.cox_beta_ = _fit_cox(X[:, 1:].astype(np.float64), times, events)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Cox fit failed (%s); falling back to KM only.", exc)
                self.cox_beta_ = None

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return per-patient hazard score.

        With Cox: linear predictor X[:, 1:] @ beta. Without Cox: 1 - S(t_max)
        from the patient's stratum (so higher = worse prognosis).
        """
        self._require_fitted()
        X = self._require_nonempty(X, "X")
        if X.ndim != 2:
            raise ValueError("predict expects 2D X")
        if self.cox_beta_ is not None and X.shape[1] > 1:
            return X[:, 1:].astype(np.float64) @ self.cox_beta_
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            s = X[i, 0]
            curve = self.km_curves_.get(s)
            if curve is None:
                # Honest fallback: average across strata.
                tail_vals = [c[1][-1] for c in self.km_curves_.values() if len(c[1])]
                s_tail = float(np.mean(tail_vals)) if tail_vals else 1.0
            else:
                _, s_curve = curve
                s_tail = float(s_curve[-1]) if len(s_curve) else 1.0
            out[i] = 1.0 - s_tail
        return out

    def predict_survival(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        self._require_fitted()
        X = self._require_nonempty(X, "X")
        times = self._require_nonempty(times, "times").astype(np.float64).ravel()
        out = np.empty((X.shape[0], times.shape[0]), dtype=np.float64)
        for i in range(X.shape[0]):
            s = X[i, 0]
            curve = self.km_curves_.get(s)
            if curve is None:
                out[i] = 1.0
                continue
            t_grid, s_grid = curve
            # Step function: at time t, S(t) = last s_grid where t_grid <= t.
            for j, t in enumerate(times):
                idx = np.searchsorted(t_grid, t, side="right") - 1
                if idx < 0:
                    out[i, j] = 1.0
                else:
                    out[i, j] = s_grid[idx]
        return out


def _kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy KM estimator. Returns (event_times, S_at_event_times)."""
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]
    unique_times = np.unique(t_sorted[e_sorted == 1])
    if unique_times.size == 0:
        return np.array([0.0]), np.array([1.0])
    s = 1.0
    s_curve = []
    for t in unique_times:
        n_at_risk = int((t_sorted >= t).sum())
        d = int(((t_sorted == t) & (e_sorted == 1)).sum())
        if n_at_risk == 0:
            break
        s *= 1.0 - d / n_at_risk
        s_curve.append(s)
    return unique_times, np.array(s_curve)


def _fit_cox(
    X: np.ndarray, times: np.ndarray, events: np.ndarray
) -> np.ndarray:
    """Tiny Cox PH fit via scipy minimize on negative partial log-likelihood.

    No tie correction beyond Breslow. Intended only as a baseline.
    """
    n, d = X.shape

    def neg_pll(beta: np.ndarray) -> float:
        eta = X @ beta
        order = np.argsort(-times)  # descending so risk set is a prefix
        eta_o = eta[order]
        e_o = events[order]
        # cumulative log-sum-exp from the start over the descending-time order
        max_eta = np.maximum.accumulate(eta_o)
        # log-sum-exp trick
        cum_exp = np.cumsum(np.exp(eta_o - max_eta))
        log_risk_set = np.log(cum_exp) + max_eta
        ll = float((eta_o[e_o == 1] - log_risk_set[e_o == 1]).sum())
        return -ll

    res = _scipy_minimize(neg_pll, np.zeros(d), method="L-BFGS-B")  # type: ignore[misc]
    return res.x
