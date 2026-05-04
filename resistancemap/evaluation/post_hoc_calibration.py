"""Post-hoc calibration methods for ResistanceMap probability outputs.

Implements temperature scaling and isotonic regression calibration,
with comprehensive diagnostics for calibration quality assessment.

References
----------
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    "On Calibration of Modern Neural Networks." ICML 2017.
Niculescu-Mizil, A., & Caruana, R. (2005).
    "Predicting good probabilities with supervised learning."
    ICML 2005.
"""

from __future__ import annotations

import logging
from typing import Literal, Tuple

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)

__all__ = [
    "TemperatureScaling",
    "IsotonicCalibrator",
    "CalibrationDiagnostics",
    "calibrate_model_outputs",
]

# Try to import sklearn for isotonic regression; graceful fallback if unavailable
try:
    from sklearn.isotonic import IsotonicRegression as SklearnIsotonicRegression

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False
    logger.debug("sklearn not available; IsotonicCalibrator will not be available")


def _validate_logits_labels(
    logits: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate logits (K-way or binary) and integer class labels.

    Args:
        logits: Either (N,) for binary classification or (N, K) for K-way.
        labels: (N,) integer class labels.

    Returns:
        Validated (logits, labels) as float64 and int arrays.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    if logits.ndim not in (1, 2):
        raise ValueError(f"logits must be 1D or 2D; got shape {logits.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D; got shape {labels.shape}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits/labels length mismatch: {logits.shape[0]} vs {labels.shape[0]}"
        )
    if logits.shape[0] == 0:
        raise ValueError("logits/labels are empty")

    return logits, labels


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: (N,) binary or (N, K) multiclass logits.

    Returns:
        Probabilities of same shape, rows summing to 1.
    """
    if logits.ndim == 1:
        # Binary: apply sigmoid with per-sample max normalization
        x = logits - np.max(logits)
        ex = np.exp(x)
        return ex / (1.0 + ex)
    else:
        # Multiclass: apply softmax with per-sample max normalization
        x = logits - np.max(logits, axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=1, keepdims=True)


def _nll_loss(
    probs: np.ndarray, labels: np.ndarray, reduction: str = "mean"
) -> float:
    """Compute negative log-likelihood loss.

    Args:
        probs: Probabilities, (N,) binary or (N, K) multiclass, rows sum to 1.
        labels: Integer class labels, (N,).
        reduction: "mean" or "sum".

    Returns:
        Scalar NLL.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    # Clamp to avoid log(0)
    probs = np.clip(probs, 1e-15, 1.0)

    if probs.ndim == 1:
        # Binary: select p or 1-p depending on label
        nll = -np.where(labels == 1, np.log(probs), np.log(1.0 - probs))
    else:
        # Multiclass: -log(p[y])
        nll = -np.log(probs[np.arange(len(labels)), labels])

    return float(nll.mean() if reduction == "mean" else nll.sum())


class TemperatureScaling:
    """Single-parameter temperature scaling calibration (Guo et al. 2017).

    Temperature scaling applies a learned scalar T to logits before softmax:
        p_calibrated = softmax(logits / T)

    The parameter T is fit by minimizing negative log-likelihood on a
    validation set. T > 1 increases entropy (under-confident logits),
    T < 1 decreases entropy (over-confident logits), T = 1 leaves logits
    unchanged.

    Attributes:
        temperature: Learned temperature parameter. None until fit() is called.
    """

    def __init__(self) -> None:
        """Initialize the temperature scaler."""
        self.temperature: float | None = None

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        init_temp: float = 1.0,
    ) -> TemperatureScaling:
        """Fit temperature to validation logits and labels.

        Minimizes NLL loss via scipy.optimize.minimize.

        Args:
            logits: (N,) binary or (N, K) multiclass logits.
            labels: (N,) integer class labels.
            init_temp: Initial temperature guess. Defaults to 1.0.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If inputs are invalid.
        """
        logits, labels = _validate_logits_labels(logits, labels)

        def nll(t: float) -> float:
            if t <= 0:
                return 1e10
            probs = _softmax(logits / t)
            return _nll_loss(probs, labels, reduction="mean")

        result = optimize.minimize_scalar(
            nll, bounds=(0.01, 10.0), method="bounded"
        )
        self.temperature = float(result.x)
        logger.info(
            "TemperatureScaling fitted: T = %.4f, NLL = %.4f",
            self.temperature,
            float(result.fun),
        )
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply fitted temperature to logits and return probabilities.

        Args:
            logits: (N,) or (N, K) logits.

        Returns:
            Calibrated probabilities of the same shape as logits.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.temperature is None:
            raise RuntimeError("fit() must be called before calibrate()")
        logits = np.asarray(logits, dtype=np.float64)
        return _softmax(logits / self.temperature)


class IsotonicCalibrator:
    """Non-parametric isotonic regression calibration (Niculescu-Mizil & Caruana 2005).

    Fits a separate isotonic regression (monotone non-decreasing function) for
    each class. This is more flexible than temperature scaling but requires
    more data.

    For binary classification, fits a single isotonic function mapping
    the predicted probability to the true label probability.

    For multiclass, fits one isotonic regressor per class (one-vs-rest).

    Attributes:
        is_fitted: Whether fit() has been called.
        _isotonic_regressors: Dict mapping class index to isotonic regressor.
        _n_classes: Number of classes (inferred from data).
        _mode: Either "binary" or "multiclass".
    """

    def __init__(self) -> None:
        """Initialize the isotonic calibrator."""
        if not _HAS_SKLEARN:
            raise ImportError(
                "IsotonicCalibrator requires sklearn; install with "
                "pip install scikit-learn"
            )
        self.is_fitted = False
        self._isotonic_regressors: dict[int, SklearnIsotonicRegression] = {}
        self._n_classes = 0
        self._mode = ""

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> IsotonicCalibrator:
        """Fit isotonic regressors to probabilities and labels.

        Args:
            probs: (N,) binary probabilities or (N, K) multiclass probabilities.
            labels: (N,) integer class labels.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If inputs are invalid.
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)

        if probs.ndim not in (1, 2):
            raise ValueError(f"probs must be 1D or 2D; got {probs.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D; got {labels.shape}")
        if probs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"probs/labels length mismatch: {probs.shape[0]} vs {labels.shape[0]}"
            )

        if probs.ndim == 1:
            self._mode = "binary"
            self._n_classes = 2
            iso = SklearnIsotonicRegression(out_of_bounds="clip")
            iso.fit(probs, labels)
            self._isotonic_regressors[1] = iso
        else:
            self._mode = "multiclass"
            self._n_classes = probs.shape[1]
            for k in range(self._n_classes):
                labels_k = (labels == k).astype(int)
                iso = SklearnIsotonicRegression(out_of_bounds="clip")
                iso.fit(probs[:, k], labels_k)
                self._isotonic_regressors[k] = iso

        self.is_fitted = True
        logger.info(
            "IsotonicCalibrator fitted: mode=%s, n_classes=%d",
            self._mode,
            self._n_classes,
        )
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply fitted isotonic regressors to probabilities.

        Args:
            probs: (N,) binary or (N, K) multiclass probabilities.

        Returns:
            Calibrated probabilities of the same shape.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before calibrate()")

        probs = np.asarray(probs, dtype=np.float64)

        if self._mode == "binary":
            probs_cal = self._isotonic_regressors[1].predict(probs)
            # Clamp to valid probability range [0, 1]
            probs_cal = np.clip(probs_cal, 0.0, 1.0)
            return probs_cal
        else:
            probs_cal = np.zeros_like(probs)
            for k in range(self._n_classes):
                probs_cal[:, k] = self._isotonic_regressors[k].predict(
                    probs[:, k]
                )
            # Clamp individual calibrated probabilities before renormalization
            probs_cal = np.clip(probs_cal, 0.0, 1.0)
            # Renormalize to sum to 1
            row_sums = probs_cal.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            probs_cal /= row_sums
            return probs_cal


class CalibrationDiagnostics:
    """Comprehensive calibration diagnostics and visualization data.

    Computes expected calibration error (ECE), adaptive calibration error (ACE),
    reliability diagrams, and per-class metrics before and after calibration.

    All metrics are imported from resistancemap.evaluation.metrics.calibration
    if available; otherwise, this class provides fallback implementations.
    """

    def __init__(self) -> None:
        """Initialize diagnostics."""
        self.before: dict = {}
        self.after: dict = {}

    def compute(
        self,
        probs_before: np.ndarray,
        probs_after: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
    ) -> dict:
        """Compute comprehensive calibration diagnostics.

        Args:
            probs_before: (N,) or (N, K) probabilities before calibration.
            probs_after: Same shape as probs_before, after calibration.
            labels: (N,) integer class labels.
            n_bins: Number of bins for reliability diagrams. Defaults to 15.

        Returns:
            Dictionary with keys:

            - ``"before"`` (dict): ECE, MCE, reliability curves, per-class metrics.
            - ``"after"`` (dict): Same as before.
            - ``"improvement"`` (dict): Change in metrics (after - before).

        """
        # Try to import existing calibration metrics
        try:
            from resistancemap.evaluation.metrics.calibration import (
                expected_calibration_error,
                maximum_calibration_error,
                reliability_curve,
            )
        except ImportError:
            logger.warning(
                "Could not import calibration metrics; using fallback implementations"
            )
            expected_calibration_error = self._fallback_ece
            maximum_calibration_error = self._fallback_mce
            reliability_curve = self._fallback_reliability_curve

        probs_before = np.asarray(probs_before, dtype=np.float64)
        probs_after = np.asarray(probs_after, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)

        self.before = {
            "ece": float(expected_calibration_error(probs_before, labels, n_bins)),
            "mce": float(maximum_calibration_error(probs_before, labels, n_bins)),
            "reliability_curve": reliability_curve(probs_before, labels, n_bins),
        }

        self.after = {
            "ece": float(expected_calibration_error(probs_after, labels, n_bins)),
            "mce": float(maximum_calibration_error(probs_after, labels, n_bins)),
            "reliability_curve": reliability_curve(probs_after, labels, n_bins),
        }

        return {
            "before": self.before,
            "after": self.after,
            "improvement": {
                "ece_change": self.before["ece"] - self.after["ece"],
                "mce_change": self.before["mce"] - self.after["mce"],
            },
        }

    @staticmethod
    def _fallback_ece(
        probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
    ) -> float:
        """Fallback ECE implementation (same as calibration.py)."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)

        if probs.ndim == 1:
            pred = (probs >= 0.5).astype(int)
            conf = np.where(pred == 1, probs, 1.0 - probs)
        else:
            pred = probs.argmax(axis=1)
            conf = probs.max(axis=1)
        correct = (pred == labels).astype(np.float64)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        n = conf.shape[0]
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
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

    @staticmethod
    def _fallback_mce(
        probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
    ) -> float:
        """Fallback MCE implementation."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)

        if probs.ndim == 1:
            pred = (probs >= 0.5).astype(int)
            conf = np.where(pred == 1, probs, 1.0 - probs)
        else:
            pred = probs.argmax(axis=1)
            conf = probs.max(axis=1)
        correct = (pred == labels).astype(np.float64)

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

    @staticmethod
    def _fallback_reliability_curve(
        probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback reliability curve implementation."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)

        if probs.ndim == 1:
            pred = (probs >= 0.5).astype(int)
            conf = np.where(pred == 1, probs, 1.0 - probs)
        else:
            pred = probs.argmax(axis=1)
            conf = probs.max(axis=1)
        correct = (pred == labels).astype(np.float64)

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


def calibrate_model_outputs(
    logits: np.ndarray,
    labels: np.ndarray,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    method: Literal["temperature", "isotonic"] = "temperature",
) -> Tuple[np.ndarray, dict]:
    """Convenience function to calibrate logits using a validation set.

    Splits the workflow into fit (on validation set) and calibrate (on test set).

    Args:
        logits: (N,) or (N, K) logits to be calibrated (test set).
        labels: (N,) labels for test set (used only in diagnostics).
        val_logits: Logits for validation/tuning set.
        val_labels: Labels for validation set.
        method: Calibration method, "temperature" or "isotonic".
            Defaults to "temperature".

    Returns:
        Tuple of:

        - Calibrated probabilities (same shape as logits).
        - Diagnostics dict from CalibrationDiagnostics.compute().

    Raises:
        ValueError: If method is invalid or inputs are malformed.
    """
    if method not in ("temperature", "isotonic"):
        raise ValueError(
            f"method must be 'temperature' or 'isotonic'; got {method!r}"
        )

    logits, labels = _validate_logits_labels(logits, labels)
    val_logits, val_labels = _validate_logits_labels(val_logits, val_labels)

    if method == "temperature":
        calibrator = TemperatureScaling()
        val_probs = _softmax(val_logits)
        calibrator.fit(val_logits, val_labels)
        test_probs = calibrator.calibrate(logits)
    else:  # isotonic
        val_probs = _softmax(val_logits)
        calibrator = IsotonicCalibrator()
        calibrator.fit(val_probs, val_labels)
        test_probs = calibrator.calibrate(_softmax(logits))

    # Compute diagnostics
    test_probs_before = _softmax(logits)
    diagnostics = CalibrationDiagnostics().compute(
        test_probs_before, test_probs, labels
    )

    return test_probs, diagnostics