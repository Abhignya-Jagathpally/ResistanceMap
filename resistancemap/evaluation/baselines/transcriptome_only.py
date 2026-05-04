"""Transcriptome-only baseline.

Reduces a high-dimensional expression matrix via PCA (numpy SVD) and fits
a simple linear classifier/regressor on the resulting components. Operates
exclusively on REAL transcriptome features supplied by the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from resistancemap.evaluation.baselines.base import Baseline

logger = logging.getLogger(__name__)


class TranscriptomeOnlyBaseline(Baseline):
    """PCA + linear-head baseline on transcriptome data."""

    def __init__(self, n_components: int = 32, l2: float = 1.0) -> None:
        super().__init__()
        self.n_components = int(n_components)
        self.l2 = float(l2)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self._is_classification: bool = False

    @property
    def name(self) -> str:
        return "transcriptome_only"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        meta: Optional[dict[str, Any]] = None,
    ) -> "TranscriptomeOnlyBaseline":
        X = self._require_nonempty(X_train, "X_train")
        y = self._require_nonempty(y_train, "y_train").astype(np.float64).ravel()
        if X.ndim != 2:
            raise ValueError(
                f"TranscriptomeOnlyBaseline expects 2D input, got shape {X.shape}"
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_train rows ({X.shape[0]}) != y_train rows ({y.shape[0]})"
            )

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Truncated SVD on the centered matrix.
        k = min(self.n_components, *Xc.shape)
        # Use np.linalg.svd with full_matrices=False then truncate.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:k]  # (k, d)
        Z = U[:, :k] * S[:k]  # (n, k)

        # Decide head: classification if y is binary, else ridge regression.
        unique = np.unique(y)
        self._is_classification = unique.size <= 2 and set(
            unique.astype(int)
        ).issubset({0, 1})

        Z_aug = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
        if self._is_classification:
            # IRLS for ridge logistic.
            d = Z_aug.shape[1]
            w = np.zeros(d, dtype=np.float64)
            for _ in range(50):
                z_score = Z_aug @ w
                p = 1.0 / (1.0 + np.exp(-z_score))
                W = p * (1.0 - p)
                # Hessian: Z^T diag(W) Z + l2 I (no penalty on bias)
                H = (Z_aug.T * W) @ Z_aug
                pen = self.l2 * np.eye(d)
                pen[-1, -1] = 0.0
                H += pen
                grad = Z_aug.T @ (p - y) + self.l2 * np.concatenate(
                    [w[:-1], [0.0]]
                )
                try:
                    delta = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H, grad, rcond=None)[0]
                w = w - delta
                if np.linalg.norm(delta) < 1e-6:
                    break
            self.w_ = w[:-1]
            self.b_ = float(w[-1])
        else:
            # Ridge regression closed form.
            d = Z_aug.shape[1]
            pen = self.l2 * np.eye(d)
            pen[-1, -1] = 0.0
            try:
                w = np.linalg.solve(Z_aug.T @ Z_aug + pen, Z_aug.T @ y)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(Z_aug.T @ Z_aug + pen, Z_aug.T @ y, rcond=None)[0]
            self.w_ = w[:-1]
            self.b_ = float(w[-1])

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = self._require_nonempty(X, "X")
        if X.ndim != 2:
            raise ValueError(
                f"TranscriptomeOnlyBaseline expects 2D input, got shape {X.shape}"
            )
        Xc = X - self.mean_
        Z = Xc @ self.components_.T  # (n, k)
        out = Z @ self.w_ + self.b_
        if self._is_classification:
            return 1.0 / (1.0 + np.exp(-out))
        return out
