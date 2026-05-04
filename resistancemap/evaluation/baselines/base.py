"""Baseline ABC.

All baselines must operate on REAL data supplied by the caller. None of
them are permitted to fabricate inputs, draw from random distributions to
fill missing arrays, or return zero/constant predictions as a placeholder.
If real data is unavailable the baseline must raise ``ValueError`` with a
message that names the missing input. This is a hard project rule for
ResistanceMap.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Baseline(ABC):
    """Abstract base class for evaluation baselines.

    Subclasses must implement :meth:`fit` and :meth:`predict`. Survival
    baselines may additionally implement :meth:`predict_survival`.

    Subclasses must validate inputs and raise ``ValueError`` for missing
    or empty data; they must NOT silently fall back to zeros or random
    outputs.
    """

    def __init__(self) -> None:
        self._fitted: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable string identifier used by BaselineAdversaryAgent."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        meta: Optional[dict[str, Any]] = None,
    ) -> "Baseline":
        """Fit the baseline on real training data.

        Args:
            X_train: feature matrix from real cohort data (e.g. MMRF).
            y_train: target labels (binary or continuous).
            meta: optional dict for per-patient metadata (IDs, censoring,
                visit times). Must NOT contain fabricated entries.

        Returns:
            self (fitted)
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for real input X. Must not return placeholders."""

    def predict_survival(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Optional: per-patient survival probability at the given times."""
        raise NotImplementedError(
            f"{self.name} does not implement predict_survival"
        )

    # ------------------------------------------------------------------
    # Shared validation helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _require_nonempty(X: Any, name: str) -> np.ndarray:
        if X is None:
            raise ValueError(
                f"{name} is None; baselines require REAL data and refuse "
                "to fabricate inputs."
            )
        arr = np.asarray(X)
        if arr.size == 0:
            raise ValueError(
                f"{name} is empty; baselines require REAL data and refuse "
                "to fabricate inputs."
            )
        return arr

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} has not been fit yet; call fit(X_train, y_train) "
                "with real data first."
            )
