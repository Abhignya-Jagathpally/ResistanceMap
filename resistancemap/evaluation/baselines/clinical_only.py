"""Clinical-only baseline.

Penalized logistic regression on clinical features only:
- Age
- ISS stage
- Cytogenetic risk score (e.g. del(17p), t(4;14), t(14;16), 1q+, etc.)

Uses scipy.optimize.minimize when scipy is importable; otherwise falls back
to a numpy gradient descent on log-loss with L2 regularization. Both paths
operate on REAL features supplied by the caller; the baseline raises if
none are provided.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from resistancemap.evaluation.baselines.base import Baseline

logger = logging.getLogger(__name__)

try:  # scipy is OPTIONAL
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - exercised only when scipy missing
    _scipy_minimize = None  # type: ignore[assignment]
    _HAS_SCIPY = False


class ClinicalOnlyBaseline(Baseline):
    """L2-penalized logistic regression on clinical features."""

    def __init__(
        self,
        l2: float = 1.0,
        lr: float = 0.05,
        n_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.l2 = float(l2)
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "clinical_only"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        meta: Optional[dict[str, Any]] = None,
    ) -> "ClinicalOnlyBaseline":
        if X_train is None or (hasattr(X_train, "size") and X_train.size == 0):
            raise ValueError(
                "ClinicalOnlyBaseline requires real clinical features; "
                "none provided"
            )
        X = self._require_nonempty(X_train, "X_train")
        y = self._require_nonempty(y_train, "y_train").astype(np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_train rows ({X.shape[0]}) != y_train rows ({y.shape[0]})"
            )

        # Normalize features (zero-mean unit-variance) for stable optimization.
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_[self.feature_std_ == 0] = 1.0
        Xn = (X - self.feature_mean_) / self.feature_std_

        n, d = Xn.shape

        def neg_log_lik(params: np.ndarray) -> float:
            w = params[:d]
            b = params[d]
            z = Xn @ w + b
            # Stable log(1 + exp(z))
            log1p_exp = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
            ll = (y * z - log1p_exp).sum()
            reg = 0.5 * self.l2 * np.sum(w * w)
            return float(-ll + reg)

        def grad(params: np.ndarray) -> np.ndarray:
            w = params[:d]
            b = params[d]
            z = Xn @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            gw = Xn.T @ err + self.l2 * w
            gb = err.sum()
            return np.concatenate([gw, [gb]])

        x0 = np.zeros(d + 1, dtype=np.float64)

        if _HAS_SCIPY:
            res = _scipy_minimize(  # type: ignore[misc]
                neg_log_lik,
                x0,
                jac=grad,
                method="L-BFGS-B",
                options={"maxiter": self.n_iter, "ftol": self.tol},
            )
            params = res.x
        else:
            params = x0.copy()
            prev_loss = neg_log_lik(params)
            for it in range(self.n_iter):
                g = grad(params)
                params = params - self.lr * g
                loss = neg_log_lik(params)
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

        self.w_ = params[:d]
        self.b_ = float(params[d])
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = self._require_nonempty(X, "X")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xn = (X - self.feature_mean_) / self.feature_std_
        z = Xn @ self.w_ + self.b_
        return 1.0 / (1.0 + np.exp(-z))
