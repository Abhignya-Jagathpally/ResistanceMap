"""Last-observation carry-forward baseline.

For each patient, predict the most recent labeled observation. Requires a
genuinely longitudinal input: either a 3D array of shape
``(n_patients, n_visits, n_features)`` plus a per-visit label tensor of
shape ``(n_patients, n_visits)``, OR a list-of-records meta dict that maps
patient IDs to lists of (time, label) tuples.

Refuses to operate on cross-sectional data; raises ``ValueError`` because
"carry forward" is undefined when there is nothing to carry.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from resistancemap.evaluation.baselines.base import Baseline

logger = logging.getLogger(__name__)


class LastObservationBaseline(Baseline):
    """Per-patient last-observation carry-forward."""

    def __init__(self) -> None:
        super().__init__()
        self.last_label_by_patient_: dict[Any, float] = {}
        self._global_mean: Optional[float] = None

    @property
    def name(self) -> str:
        return "last_observation"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        meta: Optional[dict[str, Any]] = None,
    ) -> "LastObservationBaseline":
        self._require_nonempty(X_train, "X_train")
        y = self._require_nonempty(y_train, "y_train").astype(np.float64)

        # Path A: per-patient longitudinal arrays.
        if y.ndim == 2:
            # (n_patients, n_visits)
            patient_ids = (
                meta.get("patient_ids")
                if meta and "patient_ids" in meta
                else list(range(y.shape[0]))
            )
            visit_mask = (
                meta.get("visit_mask")
                if meta and "visit_mask" in meta
                else np.ones_like(y, dtype=bool)
            )
            visit_mask = np.asarray(visit_mask, dtype=bool)
            for i, pid in enumerate(patient_ids):
                valid = np.where(visit_mask[i])[0]
                if valid.size == 0:
                    continue
                last_idx = valid[-1]
                self.last_label_by_patient_[pid] = float(y[i, last_idx])
            if not self.last_label_by_patient_:
                raise ValueError(
                    "LastObservationBaseline received no valid (patient, "
                    "visit) labeled observations; cannot carry forward."
                )
            self._global_mean = float(
                np.mean(list(self.last_label_by_patient_.values()))
            )
            self._fitted = True
            return self

        # Path B: meta-driven longitudinal records.
        if meta and "longitudinal_records" in meta:
            records = meta["longitudinal_records"]
            if not isinstance(records, dict) or not records:
                raise ValueError(
                    "LastObservationBaseline requires a non-empty "
                    "meta['longitudinal_records'] dict."
                )
            for pid, history in records.items():
                if not history:
                    continue
                # Sort by time, take latest.
                latest = sorted(history, key=lambda r: r[0])[-1]
                self.last_label_by_patient_[pid] = float(latest[1])
            if not self.last_label_by_patient_:
                raise ValueError(
                    "LastObservationBaseline found no labeled observations "
                    "in meta['longitudinal_records']."
                )
            self._global_mean = float(
                np.mean(list(self.last_label_by_patient_.values()))
            )
            self._fitted = True
            return self

        raise ValueError(
            "LastObservationBaseline requires longitudinal input. Pass "
            "either a 2D y_train of shape (n_patients, n_visits) "
            "(optionally with meta['patient_ids']/meta['visit_mask']) or a "
            "meta['longitudinal_records'] dict mapping patient_id -> "
            "list[(time, label)]."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        # X is interpreted as a list of patient_ids if it's 1D and matches
        # known patients; otherwise we expect meta-style routing via the
        # caller. We do not invent identities.
        X = self._require_nonempty(X, "X")
        flat = np.asarray(X).ravel()
        out = np.empty(flat.shape[0], dtype=np.float64)
        for i, pid in enumerate(flat.tolist()):
            if pid in self.last_label_by_patient_:
                out[i] = self.last_label_by_patient_[pid]
            else:
                # Honest fallback: report the cohort mean and warn.
                logger.warning(
                    "LastObservationBaseline.predict: no history for "
                    "patient %r; falling back to cohort mean.",
                    pid,
                )
                out[i] = float(self._global_mean)
        return out
