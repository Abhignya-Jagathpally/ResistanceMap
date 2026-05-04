"""Benchmarking Against Clinical Risk Scores and SOTA Comparison Framework.

Gap 3.6: Clinical Risk Score Comparison
- Discrimination: Harrell's C-index, time-dependent AUC (2/3/5-year), integrated Brier score
- Calibration: Slope (target=1), intercept (target=0), Hosmer-Lemeshow test
- Reclassification: NRI >0.05 clinically meaningful, IDI >0.01 threshold
- Decision Curve Analysis (DCA): Net benefit across 50%, 60%, 70% 5-year OS thresholds

Gap 3.7: SOTA Comparison Framework (11 methods)
- Methods: DrugCell, DeepCDR, GraphDRP, PathDSP, PaccMann, MOLI, scDrug, TCRP, PRECISE, Velodrome + ResistanceMap
- 4 dimensions: (1) Regression, (2) Classification, (3) Multi-omics contribution (Shapley), (4) Computational efficiency
- 6-phase protocol: baseline → integration → statistical comparison → profiling → generalization → reproducibility

References:
    Harrell, F. E., et al. (1982). "Evaluating the yield of medical tests." JAMA, 247(18).
    Uno, H., et al. (2007). "Evaluating prediction rules for t-year survivors." JASA, 102(478).
    Vickers, A. J., & Elkin, E. B. (2006). "Decision curve analysis." Medical Decision Making, 26(6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

__all__ = [
    "HarrellsCIndex",
    "TimeDependentAUC",
    "IntegratedBrierScore",
    "CalibrationAssessor",
    "NetReclassificationImprovement",
    "DecisionCurveAnalysis",
    "SOTABenchmarkRunner",
    "ComputationalProfiler",
    "BenchmarkOrchestrator",
]


def _validate_survival_inputs(
    times: np.ndarray, events: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate paired (times, events) survival arrays."""
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int8)
    if times.ndim != 1 or events.ndim != 1:
        raise ValueError("times and events must be 1D arrays")
    if times.shape[0] != events.shape[0]:
        raise ValueError(f"times/events length mismatch: {times.shape[0]} vs {events.shape[0]}")
    if times.shape[0] == 0:
        raise ValueError("times/events are empty")
    if np.any(times < 0):
        raise ValueError("times must be non-negative")
    if np.any((events != 0) & (events != 1)):
        raise ValueError("events must be 0 (censored) or 1 (event)")
    return times, events


def _kaplan_meier_censoring(
    times: np.ndarray, events: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute KM estimator of censoring survival G(t).

    Returns sorted unique times and G(t) at each, treating censored observations
    as the "event" of interest for this reverse calculation.
    """
    censor_events = 1 - events
    order = np.argsort(times)
    t_sorted = times[order]
    c_sorted = censor_events[order]
    unique_t, idx = np.unique(t_sorted, return_index=True)

    g = np.ones(unique_t.shape[0], dtype=np.float64)
    n_at_risk = t_sorted.shape[0]
    surv = 1.0
    for i, ut in enumerate(unique_t):
        end = idx[i + 1] if i + 1 < len(idx) else len(t_sorted)
        d = c_sorted[idx[i] : end].sum()
        if n_at_risk > 0 and d > 0:
            surv *= 1.0 - d / n_at_risk
        g[i] = surv
        n_at_risk -= end - idx[i]
    return unique_t, g


class HarrellsCIndex:
    """Harrell's concordance index with censoring-adjusted handling.

    Estimates the probability that, for a randomly chosen pair of observations
    (one with event, one without or censored later), the risk scores correctly
    order the two. Ranges from 0 to 1; 0.5 is random chance, 1.0 is perfect.

    References:
        Harrell, F. E., et al. (1982). "Evaluating the yield of medical tests." JAMA.
    """

    @staticmethod
    def compute(
        times: np.ndarray, events: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Compute C-index for censored survival data.

        Args:
            times: Observed times (shape N).
            events: Event indicators 0/1 (shape N). 1 = event, 0 = censored.
            predictions: Predicted risk scores (shape N). Higher = higher risk.

        Returns:
            C-index in [0, 1].
        """
        times, events = _validate_survival_inputs(times, events)
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.shape[0] != times.shape[0]:
            raise ValueError(f"predictions length mismatch: {predictions.shape[0]} vs {times.shape[0]}")

        n = times.shape[0]
        concordant = 0.0
        total = 0.0

        for i in range(n):
            if events[i] == 0:
                continue
            for j in range(n):
                if i == j or times[j] < times[i]:
                    continue
                if times[j] == times[i] and events[j] == 1:
                    continue
                total += 1.0
                if predictions[i] > predictions[j]:
                    concordant += 1.0

        if total == 0.0:
            logger.warning("No valid pairs for C-index; returning 0.5 (random chance)")
            return 0.5
        return concordant / total


class TimeDependentAUC:
    """Time-dependent AUC (ROC at a fixed time horizon).

    Evaluates discrimination at specific follow-up times (e.g., 2, 3, 5 years).
    Uses Kaplan-Meier weighting to account for censoring.

    References:
        Uno, H., et al. (2007). "Evaluating prediction rules for t-year survivors." JASA.
    """

    @staticmethod
    def compute(
        times: np.ndarray, events: np.ndarray, predictions: np.ndarray, horizon: float
    ) -> float:
        """Compute time-dependent AUC at a fixed horizon.

        Args:
            times: Observed times (shape N).
            events: Event indicators 0/1 (shape N).
            predictions: Predicted risk scores (shape N).
            horizon: Time horizon for evaluation.

        Returns:
            AUC in [0, 1].
        """
        times, events = _validate_survival_inputs(times, events)
        predictions = np.asarray(predictions, dtype=np.float64)

        if horizon <= 0:
            raise ValueError(f"horizon must be positive; got {horizon}")

        t_km, g = _kaplan_meier_censoring(times, events)
        g_interp = np.interp(times, t_km, g, left=1.0, right=g[-1])
        g_interp = np.clip(g_interp, 1e-8, 1.0)

        at_risk = times >= horizon
        status = (times <= horizon) & (events == 1)

        n_case = status.sum()
        n_ctrl = at_risk.sum() - n_case

        if n_case == 0 or n_ctrl == 0:
            logger.warning(
                f"No events or controls at horizon {horizon}; returning 0.5"
            )
            return 0.5

        concordant = 0.0
        total_weight = 0.0

        for i in range(len(times)):
            if not status[i]:
                continue
            weight_i = 1.0 / g_interp[i]
            for j in range(len(times)):
                if i == j or not at_risk[j]:
                    continue
                total_weight += weight_i
                if predictions[i] > predictions[j]:
                    concordant += weight_i

        if total_weight == 0:
            return 0.5
        return concordant / total_weight


class IntegratedBrierScore:
    """Integrated Brier Score over a time interval.

    Measures average squared error of survival probability predictions
    integrated over time, accounting for censoring via inverse probability weighting.

    References:
        Graf, E., et al. (1999). "Assessment and comparison of prognostic
        classification schemes for survival data." Statistics in Medicine.
    """

    @staticmethod
    def compute(
        times: np.ndarray,
        events: np.ndarray,
        survival_probs: np.ndarray,
        times_grid: Optional[np.ndarray] = None,
    ) -> float:
        """Compute integrated Brier score.

        Args:
            times: Observed times (shape N).
            events: Event indicators 0/1 (shape N).
            survival_probs: Predicted survival probabilities (shape N, T) where
                T is number of time points. Or (N,) if single time point.
            times_grid: Time grid at which survival_probs are evaluated.
                If None, uses evenly spaced grid from 0 to max(times).

        Returns:
            Integrated Brier score in [0, 1].
        """
        times, events = _validate_survival_inputs(times, events)
        survival_probs = np.asarray(survival_probs, dtype=np.float64)

        if survival_probs.shape[0] != times.shape[0]:
            raise ValueError(
                f"survival_probs N mismatch: {survival_probs.shape[0]} vs {times.shape[0]}"
            )

        if survival_probs.ndim == 1:
            survival_probs = survival_probs.reshape(-1, 1)
            if times_grid is None:
                times_grid = np.array([times.max()])
        else:
            if times_grid is None:
                times_grid = np.linspace(0, times.max(), survival_probs.shape[1])

        t_km, g = _kaplan_meier_censoring(times, events)
        g_interp = np.interp(times_grid, t_km, g, left=1.0, right=g[-1])
        g_interp = np.clip(g_interp, 1e-8, 1.0)

        brier = np.zeros(len(times_grid))
        for t_idx, t_grid in enumerate(times_grid):
            case_mask = (times <= t_grid) & (events == 1)
            ctrl_mask = times > t_grid

            if case_mask.sum() > 0:
                brier[t_idx] += (case_mask.sum() / len(times)) * (
                    (1.0 - survival_probs[case_mask, t_idx]) ** 2 / g_interp[t_idx]
                ).mean()

            if ctrl_mask.sum() > 0:
                brier[t_idx] += (ctrl_mask.sum() / len(times)) * (
                    survival_probs[ctrl_mask, t_idx] ** 2 / g_interp[t_idx]
                ).mean()

        try:
            ibs = np.trapezoid(brier, times_grid) / (times_grid[-1] - times_grid[0])
        except AttributeError:
            ibs = np.trapz(brier, times_grid) / (times_grid[-1] - times_grid[0])
        return float(np.clip(np.abs(ibs), 0.0, 1.0))


class CalibrationAssessor:
    """Calibration metrics: slope, intercept, Hosmer-Lemeshow test.

    Assesses whether predicted probabilities match observed event rates.
    Targets: slope = 1, intercept = 0.

    References:
        Harrell, F. E., et al. (1996). "Evaluating the yield of medical tests." JAMA.
    """

    @staticmethod
    def compute_slope_intercept(
        observed_events: np.ndarray, predicted_probs: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Compute calibration slope and intercept via logistic regression.

        Args:
            observed_events: Binary event labels (0/1).
            predicted_probs: Predicted probabilities [0, 1].

        Returns:
            (slope, intercept, slope_se, intercept_se)
        """
        observed_events = np.asarray(observed_events, dtype=np.int8)
        predicted_probs = np.asarray(predicted_probs, dtype=np.float64)

        if observed_events.shape[0] != predicted_probs.shape[0]:
            raise ValueError("observed_events/predicted_probs length mismatch")
        if observed_events.shape[0] == 0:
            raise ValueError("observed_events/predicted_probs are empty")

        eps = 1e-7
        predicted_probs = np.clip(predicted_probs, eps, 1.0 - eps)
        logit_pred = np.log(predicted_probs / (1.0 - predicted_probs))

        n = len(observed_events)
        x = np.c_[np.ones(n), logit_pred]
        w = predicted_probs * (1.0 - predicted_probs)
        w = np.clip(w, eps, None)

        try:
            xtx_inv = np.linalg.inv(x.T @ np.diag(w) @ x)
            beta = xtx_inv @ x.T @ np.diag(w) @ observed_events
            se = np.sqrt(np.diag(xtx_inv))
            intercept, slope = beta[0], beta[1]
            intercept_se, slope_se = se[0], se[1]
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in calibration regression; returning NaN")
            return np.nan, np.nan, np.nan, np.nan

        return float(slope), float(intercept), float(slope_se), float(intercept_se)

    @staticmethod
    def hosmer_lemeshow(
        observed_events: np.ndarray, predicted_probs: np.ndarray, n_bins: int = 10
    ) -> Tuple[float, float]:
        """Hosmer-Lemeshow goodness-of-fit test.

        Args:
            observed_events: Binary event labels (0/1).
            predicted_probs: Predicted probabilities [0, 1].
            n_bins: Number of quantile-based bins.

        Returns:
            (chi2_statistic, p_value)
        """
        observed_events = np.asarray(observed_events, dtype=np.int8)
        predicted_probs = np.asarray(predicted_probs, dtype=np.float64)

        if observed_events.shape[0] != predicted_probs.shape[0]:
            raise ValueError("observed_events/predicted_probs length mismatch")

        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(predicted_probs, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        chi2 = 0.0
        for i in range(len(bin_edges) - 1):
            mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            o = observed_events[mask].sum()
            e = predicted_probs[mask].sum()
            if e > 0:
                chi2 += (o - e) ** 2 / e
            if (1 - observed_events[mask]).sum() > 0:
                chi2 += ((mask.sum() - o) - (mask.sum() - e)) ** 2 / (mask.sum() - e)

        p_val = 1.0 - sp_stats.chi2.cdf(chi2, n_bins - 2)
        return float(chi2), float(p_val)


class NetReclassificationImprovement:
    """NRI and IDI: quantify improved risk classification.

    NRI measures the proportion of subjects whose risk classification improves.
    IDI measures integrated improvement in discrimination slopes.

    References:
        Pencina, M. J., et al. (2008). "Evaluating the added predictive ability." Epidemiology.
    """

    @staticmethod
    def compute_nri(
        observed_events: np.ndarray,
        risk_old: np.ndarray,
        risk_new: np.ndarray,
        cutoff: float = 0.2,
    ) -> Tuple[float, float]:
        """Compute Net Reclassification Improvement.

        Args:
            observed_events: Binary event labels (0/1).
            risk_old: Old risk estimates [0, 1].
            risk_new: New risk estimates [0, 1].
            cutoff: Risk classification threshold.

        Returns:
            (NRI, NRI_p_value)
        """
        observed_events = np.asarray(observed_events, dtype=np.int8)
        risk_old = np.asarray(risk_old, dtype=np.float64)
        risk_new = np.asarray(risk_new, dtype=np.float64)

        if not (observed_events.shape[0] == risk_old.shape[0] == risk_new.shape[0]):
            raise ValueError("Shape mismatch among observed_events, risk_old, risk_new")

        event_mask = observed_events == 1
        nonevent_mask = observed_events == 0

        n_event = event_mask.sum()
        n_nonevent = nonevent_mask.sum()

        if n_event == 0 or n_nonevent == 0:
            logger.warning("No events or non-events; returning NRI=0")
            return 0.0, 1.0

        old_class = (risk_old >= cutoff).astype(int)
        new_class = (risk_new >= cutoff).astype(int)

        event_up = ((old_class[event_mask] == 0) & (new_class[event_mask] == 1)).sum()
        event_down = ((old_class[event_mask] == 1) & (new_class[event_mask] == 0)).sum()
        nonevent_up = ((old_class[nonevent_mask] == 0) & (new_class[nonevent_mask] == 1)).sum()
        nonevent_down = ((old_class[nonevent_mask] == 1) & (new_class[nonevent_mask] == 0)).sum()

        nri_event = (event_up - event_down) / n_event if n_event > 0 else 0.0
        nri_nonevent = (nonevent_down - nonevent_up) / n_nonevent if n_nonevent > 0 else 0.0
        nri = nri_event + nri_nonevent

        se_nri = np.sqrt(
            (event_up + event_down) / (n_event ** 2)
            + (nonevent_up + nonevent_down) / (n_nonevent ** 2)
        )
        z = nri / (se_nri + 1e-8)
        p_val = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(z)))

        return float(nri), float(p_val)

    @staticmethod
    def compute_idi(
        observed_events: np.ndarray, risk_old: np.ndarray, risk_new: np.ndarray
    ) -> float:
        """Compute Integrated Discrimination Improvement.

        Args:
            observed_events: Binary event labels (0/1).
            risk_old: Old risk estimates.
            risk_new: New risk estimates.

        Returns:
            IDI value.
        """
        observed_events = np.asarray(observed_events, dtype=np.int8)
        risk_old = np.asarray(risk_old, dtype=np.float64)
        risk_new = np.asarray(risk_new, dtype=np.float64)

        event_mask = observed_events == 1
        nonevent_mask = observed_events == 0

        idi = (risk_new[event_mask].mean() - risk_old[event_mask].mean()) - (
            risk_new[nonevent_mask].mean() - risk_old[nonevent_mask].mean()
        )

        return float(idi)


class DecisionCurveAnalysis:
    """Decision Curve Analysis: evaluates clinical utility across threshold range.

    Computes net benefit at different probability thresholds for binary decisions.
    Allows comparison of multiple strategies (e.g., AI model vs clinical scores).

    References:
        Vickers, A. J., & Elkin, E. B. (2006). "Decision curve analysis." Med Decis Making.
    """

    @staticmethod
    def compute_net_benefit(
        observed_events: np.ndarray,
        predictions: np.ndarray,
        threshold: float,
        harm_benefit_ratio: float = 1.0,
    ) -> float:
        """Compute net benefit at a decision threshold.

        Net benefit = (TP/N) - (FP/N) * (threshold / (1 - threshold)) * harm_benefit_ratio

        Args:
            observed_events: Binary event labels (0/1).
            predictions: Predicted probabilities [0, 1].
            threshold: Decision threshold.
            harm_benefit_ratio: Weight for benefit vs harm (default 1.0 is symmetric).

        Returns:
            Net benefit value.
        """
        observed_events = np.asarray(observed_events, dtype=np.int8)
        predictions = np.asarray(predictions, dtype=np.float64)

        n = len(observed_events)
        decision = predictions >= threshold

        tp = ((decision == 1) & (observed_events == 1)).sum()
        fp = ((decision == 1) & (observed_events == 0)).sum()

        if threshold >= 1.0:
            return 0.0
        nb = (tp / n) - (fp / n) * (threshold / (1.0 - threshold)) * harm_benefit_ratio
        return float(nb)

    @staticmethod
    def dca_multiple_models(
        observed_events: np.ndarray,
        model_predictions: Dict[str, np.ndarray],
        thresholds: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute DCA curves for multiple models.

        Args:
            observed_events: Binary event labels (0/1).
            model_predictions: Dict of model_name -> predictions array.
            thresholds: Array of decision thresholds to evaluate.

        Returns:
            Dict of model_name -> net benefit array.
        """
        results = {}
        for model_name, preds in model_predictions.items():
            nb_curve = np.array([
                DecisionCurveAnalysis.compute_net_benefit(observed_events, preds, t)
                for t in thresholds
            ])
            results[model_name] = nb_curve
        return results


@dataclass
class BenchmarkResult:
    """Result container for SOTA benchmark comparison.

    Attributes:
        method_name: Name of the method evaluated.
        regression_metrics: Dict of Pearson r, Spearman rho, RMSE, MAE, R2.
        classification_metrics: Dict of AUC-ROC, balanced accuracy, F1, MCC.
        shapley_importance: Dict of modality -> mean Shapley value.
        flops: Floating point operations count.
        inference_time_sec: Inference latency in seconds.
        memory_mb: Peak memory usage in MB.
        cross_dataset_auc: AUC on held-out dataset.
        reproducibility_score: Fraction of reproducibility checklist passed [0, 1].
    """

    method_name: str
    regression_metrics: Dict[str, float] = field(default_factory=dict)
    classification_metrics: Dict[str, float] = field(default_factory=dict)
    shapley_importance: Dict[str, float] = field(default_factory=dict)
    flops: Optional[int] = None
    inference_time_sec: Optional[float] = None
    memory_mb: Optional[float] = None
    cross_dataset_auc: Optional[float] = None
    reproducibility_score: Optional[float] = None


class ComputationalProfiler:
    """Profile computational efficiency: FLOPs, inference time, memory.

    Wraps model inference and measures execution metrics.
    """

    @staticmethod
    def profile_inference(
        model_fn: Callable, inputs: Dict[str, np.ndarray], n_runs: int = 5
    ) -> Dict[str, float]:
        """Profile model inference.

        Args:
            model_fn: Callable that takes inputs dict and returns predictions.
            inputs: Input dict with data arrays.
            n_runs: Number of runs to average over.

        Returns:
            Dict with 'inference_time_sec', 'memory_mb'.
        """
        import tracemalloc
        import time

        times = []
        peak_mem = 0.0

        for _ in range(n_runs):
            tracemalloc.start()
            t0 = time.perf_counter()
            _ = model_fn(inputs)
            t1 = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            times.append(t1 - t0)
            peak_mem = max(peak_mem, peak / (1024 ** 2))

        return {
            "inference_time_sec": float(np.mean(times)),
            "memory_mb": float(peak_mem),
        }

    @staticmethod
    def estimate_flops(
        model_name: str, input_size: int, hidden_size: int, output_size: int
    ) -> int:
        """Estimate FLOPs for a neural model.

        Simplified: assumes fully connected layers.
        FLOPs ≈ 2 * (input * hidden + hidden * output)

        Args:
            model_name: For logging.
            input_size: Input dimension.
            hidden_size: Hidden layer dimension.
            output_size: Output dimension.

        Returns:
            Estimated FLOPs.
        """
        flops = 2 * (input_size * hidden_size + hidden_size * output_size)
        logger.info(f"{model_name}: ~{flops} FLOPs")
        return flops


class SOTABenchmarkRunner:
    """Orchestrate 11-method SOTA comparison across 4 dimensions.

    Implements 6-phase protocol:
    1. Baseline: evaluate each method on test set
    2. Integration: aggregate multi-omics contributions
    3. Statistical comparison: DeLong/Fisher tests
    4. Computational profiling: FLOPs, latency, memory
    5. Cross-dataset generalization: held-out data
    6. Reproducibility checklist: code, data, hyperparams
    """

    METHODS = [
        "DrugCell",
        "DeepCDR",
        "GraphDRP",
        "PathDSP",
        "PaccMann",
        "MOLI",
        "scDrug",
        "TCRP",
        "PRECISE",
        "Velodrome",
        "ResistanceMap",
    ]

    def __init__(self):
        """Initialize SOTA benchmark runner."""
        self.results: Dict[str, BenchmarkResult] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def phase1_baseline(
        self,
        y_true: np.ndarray,
        y_pred_dict: Dict[str, np.ndarray],
        task: str = "classification",
    ) -> None:
        """Phase 1: Evaluate each method on test set.

        Args:
            y_true: Ground truth labels or values.
            y_pred_dict: Dict of method_name -> predictions.
            task: "regression" or "classification".
        """
        for method_name, y_pred in y_pred_dict.items():
            if task == "regression":
                metrics = self._compute_regression_metrics(y_true, y_pred)
            else:
                metrics = self._compute_classification_metrics(y_true, y_pred)

            result = BenchmarkResult(method_name=method_name)
            if task == "regression":
                result.regression_metrics = metrics
            else:
                result.classification_metrics = metrics
            self.results[method_name] = result
            self.logger.info(f"Phase 1 baseline: {method_name} -> {metrics}")

    def phase2_integration(
        self,
        shapley_scores: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        """Phase 2: Aggregate multi-omics Shapley contributions.

        Args:
            shapley_scores: Dict of method_name -> Dict of modality -> Shapley array.
        """
        for method_name, modality_scores in shapley_scores.items():
            if method_name not in self.results:
                self.results[method_name] = BenchmarkResult(method_name=method_name)

            importance = {
                modality: float(scores.mean())
                for modality, scores in modality_scores.items()
            }
            self.results[method_name].shapley_importance = importance
            self.logger.info(f"Phase 2 integration: {method_name} -> {importance}")

    def phase3_statistical_comparison(
        self, y_true: np.ndarray, y_pred_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        """Phase 3: DeLong/Fisher statistical comparison vs reference (ResistanceMap).

        Args:
            y_true: Ground truth.
            y_pred_dict: Dict of method_name -> predictions.

        Returns:
            Dict of method_name -> (test_statistic, p_value).
        """
        if "ResistanceMap" not in y_pred_dict:
            self.logger.warning("ResistanceMap not in predictions; skipping statistical test")
            return {}

        ref_auc = self._compute_auc(y_true, y_pred_dict["ResistanceMap"])
        results_stat = {}

        for method_name, y_pred in y_pred_dict.items():
            if method_name == "ResistanceMap":
                continue
            auc = self._compute_auc(y_true, y_pred)
            z_stat, p_val = self._delong_test(y_true, y_pred_dict["ResistanceMap"], y_pred)
            results_stat[method_name] = (z_stat, p_val)
            self.logger.info(
                f"Phase 3 comparison: {method_name} vs ResistanceMap -> z={z_stat:.4f}, p={p_val:.4e}"
            )

        return results_stat

    def phase4_computational_profiling(
        self,
        model_inferences: Dict[str, Callable],
        inputs: Dict[str, np.ndarray],
    ) -> None:
        """Phase 4: Profile computational efficiency.

        Args:
            model_inferences: Dict of method_name -> inference callable.
            inputs: Input data dict.
        """
        for method_name, infer_fn in model_inferences.items():
            if method_name not in self.results:
                self.results[method_name] = BenchmarkResult(method_name=method_name)

            prof = ComputationalProfiler.profile_inference(infer_fn, inputs, n_runs=5)
            self.results[method_name].inference_time_sec = prof["inference_time_sec"]
            self.results[method_name].memory_mb = prof["memory_mb"]
            self.logger.info(
                f"Phase 4 profiling: {method_name} -> "
                f"t={prof['inference_time_sec']:.4f}s, mem={prof['memory_mb']:.1f}MB"
            )

    def phase5_cross_dataset_generalization(
        self, y_true: np.ndarray, y_pred_dict: Dict[str, np.ndarray]
    ) -> None:
        """Phase 5: Evaluate on held-out external dataset.

        Args:
            y_true: External dataset ground truth.
            y_pred_dict: Dict of method_name -> external predictions.
        """
        for method_name, y_pred in y_pred_dict.items():
            if method_name not in self.results:
                self.results[method_name] = BenchmarkResult(method_name=method_name)

            auc = self._compute_auc(y_true, y_pred)
            self.results[method_name].cross_dataset_auc = auc
            self.logger.info(f"Phase 5 generalization: {method_name} -> AUC={auc:.4f}")

    def phase6_reproducibility(
        self, reproducibility_checklist: Dict[str, Dict[str, bool]]
    ) -> None:
        """Phase 6: Assess reproducibility.

        Args:
            reproducibility_checklist: Dict of method_name -> Dict of check_name -> bool.
        """
        for method_name, checks in reproducibility_checklist.items():
            if method_name not in self.results:
                self.results[method_name] = BenchmarkResult(method_name=method_name)

            score = sum(checks.values()) / len(checks) if checks else 0.0
            self.results[method_name].reproducibility_score = float(score)
            self.logger.info(f"Phase 6 reproducibility: {method_name} -> {score:.2%}")

    @staticmethod
    def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics: Pearson r, Spearman rho, RMSE, MAE, R2."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
        spearman_rho = float(sp_stats.spearmanr(y_true, y_pred)[0])
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        return {
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    @staticmethod
    def _compute_classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics: AUC-ROC, balanced accuracy, F1, MCC."""
        y_true = np.asarray(y_true, dtype=np.int8)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        auc_roc = float(SOTABenchmarkRunner._compute_auc(y_true, y_pred))

        y_pred_binary = (y_pred >= 0.5).astype(int)

        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        tn = ((y_pred_binary == 0) & (y_true == 0)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        bal_acc = float((sens + spec) / 2.0)

        f1 = float(2 * tp / (2 * tp + fp + fn + 1e-8))

        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = float((tp * tn - fp * fn) / (mcc_denom + 1e-8))

        return {
            "auc_roc": auc_roc,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "mcc": mcc,
        }

    @staticmethod
    def _compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC-ROC via Mann-Whitney U statistic."""
        y_true = np.asarray(y_true, dtype=np.int8)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        pos_pred = y_pred[y_true == 1]
        neg_pred = y_pred[y_true == 0]

        n_pos = len(pos_pred)
        n_neg = len(neg_pred)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        concordant = 0.0
        for pos in pos_pred:
            concordant += np.sum(pos > neg_pred) + 0.5 * np.sum(pos == neg_pred)

        auc = concordant / (n_pos * n_neg)
        return float(np.clip(auc, 0.0, 1.0))

    @staticmethod
    def _delong_test(
        y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray
    ) -> Tuple[float, float]:
        """DeLong's test for comparing two AUC values.

        Args:
            y_true: Binary labels.
            y_pred1: Predictions from model 1.
            y_pred2: Predictions from model 2.

        Returns:
            (z_statistic, p_value)
        """
        y_true = np.asarray(y_true, dtype=np.int8)
        y_pred1 = np.asarray(y_pred1, dtype=np.float64)
        y_pred2 = np.asarray(y_pred2, dtype=np.float64)

        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()

        if n_pos == 0 or n_neg == 0:
            return 0.0, 1.0

        def _mann_whitney_auc(y_true, y_pred):
            pos = y_pred[y_true == 1]
            neg = y_pred[y_true == 0]
            n_pos_local = len(pos)
            n_neg_local = len(neg)
            z_mat = np.zeros((n_pos_local, n_neg_local))
            for i in range(n_pos_local):
                z_mat[i, :] = (pos[i] > neg).astype(float) + 0.5 * (pos[i] == neg).astype(float)
            return z_mat

        z1 = _mann_whitney_auc(y_true, y_pred1)
        z2 = _mann_whitney_auc(y_true, y_pred2)

        s1 = z1.sum() / (n_pos * n_neg)
        s2 = z2.sum() / (n_pos * n_neg)

        var1 = (z1 ** 2).mean() - s1 ** 2
        var2 = (z2 ** 2).mean() - s2 ** 2

        se = np.sqrt((var1 / n_pos + var2 / n_neg))
        z_stat = (s1 - s2) / (se + 1e-8)
        p_val = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(z_stat)))

        return float(z_stat), float(p_val)


class BenchmarkOrchestrator:
    """Top-level orchestrator: Gap 3.6 + Gap 3.7 benchmarking pipeline.

    Coordinates clinical risk score comparison and SOTA benchmarking,
    producing a unified report with all metrics, statistical tests, and recommendations.
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.clinical_results: Dict[str, Dict] = {}
        self.sota_results: Dict[str, BenchmarkResult] = {}

    def benchmark_clinical_scores(
        self,
        times: np.ndarray,
        events: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        risk_thresholds: Optional[List[float]] = None,
    ) -> Dict[str, Dict]:
        """Gap 3.6: Benchmark clinical risk scores.

        Args:
            times: Observed survival times.
            events: Event indicators (0/1).
            predictions_dict: Dict of score_name -> predictions array.
            risk_thresholds: Risk thresholds for DCA (default [0.5, 0.6, 0.7]).

        Returns:
            Dict of score_name -> comprehensive metrics.
        """
        if risk_thresholds is None:
            risk_thresholds = [0.5, 0.6, 0.7]

        results = {}

        for score_name, preds in predictions_dict.items():
            metrics = {}

            c_index = HarrellsCIndex.compute(times, events, preds)
            metrics["c_index"] = c_index
            metrics["c_index_pass"] = c_index >= 0.55

            for horizon in [2.0, 3.0, 5.0]:
                auc = TimeDependentAUC.compute(times, events, preds, horizon)
                metrics[f"auc_{horizon}yr"] = auc

            ibs = IntegratedBrierScore.compute(times, events, preds)
            metrics["integrated_brier"] = ibs

            slope, intercept, slope_se, intercept_se = CalibrationAssessor.compute_slope_intercept(
                events, preds
            )
            metrics["calibration_slope"] = slope
            metrics["calibration_intercept"] = intercept
            metrics["slope_pass"] = 0.85 <= slope <= 1.15
            metrics["intercept_pass"] = abs(intercept) <= 0.1

            chi2, p_hl = CalibrationAssessor.hosmer_lemeshow(events, preds)
            metrics["hl_chi2"] = chi2
            metrics["hl_pvalue"] = p_hl
            metrics["hl_pass"] = p_hl > 0.05

            nri, nri_p = NetReclassificationImprovement.compute_nri(events, preds, preds, 0.5)
            metrics["nri"] = nri
            metrics["nri_pass"] = abs(nri) > 0.05
            idi = NetReclassificationImprovement.compute_idi(events, preds, preds)
            metrics["idi"] = idi
            metrics["idi_pass"] = abs(idi) > 0.01

            dca_data = {score_name: preds}
            nb_dict = DecisionCurveAnalysis.dca_multiple_models(
                events, dca_data, np.array(risk_thresholds)
            )
            metrics["dca_net_benefits"] = {
                f"{t:.0%}": float(nb_dict[score_name][i])
                for i, t in enumerate(risk_thresholds)
            }

            results[score_name] = metrics
            self.clinical_results[score_name] = metrics
            self.logger.info(f"Clinical score {score_name}: {metrics}")

        return results

    def benchmark_sota(
        self,
        y_true: np.ndarray,
        y_pred_dict: Dict[str, np.ndarray],
        task: str = "classification",
    ) -> Dict[str, BenchmarkResult]:
        """Gap 3.7: Run SOTA benchmark.

        Args:
            y_true: Ground truth labels/values.
            y_pred_dict: Dict of method_name -> predictions.
            task: "regression" or "classification".

        Returns:
            Dict of method_name -> BenchmarkResult.
        """
        runner = SOTABenchmarkRunner()
        runner.phase1_baseline(y_true, y_pred_dict, task=task)
        self.sota_results = runner.results
        self.logger.info(f"SOTA benchmark completed: {len(runner.results)} methods")
        return runner.results

    def generate_report(self) -> Dict:
        """Generate consolidated benchmarking report.

        Returns:
            Dict with clinical results, SOTA results, and summary statistics.
        """
        report = {
            "clinical_scores": self.clinical_results,
            "sota_methods": {
                name: {
                    "regression_metrics": result.regression_metrics,
                    "classification_metrics": result.classification_metrics,
                    "shapley_importance": result.shapley_importance,
                    "flops": result.flops,
                    "inference_time_sec": result.inference_time_sec,
                    "memory_mb": result.memory_mb,
                    "cross_dataset_auc": result.cross_dataset_auc,
                    "reproducibility_score": result.reproducibility_score,
                }
                for name, result in self.sota_results.items()
            },
            "summary": {
                "num_clinical_scores": len(self.clinical_results),
                "num_sota_methods": len(self.sota_results),
            },
        }
        return report
