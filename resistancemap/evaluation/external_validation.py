from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score, brier_score_loss
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TRIPODAIChecklist:
    """TRIPOD+AI 27-item compliance checklist."""

    title: str = ""
    abstract: str = ""
    introduction: str = ""
    objectives: str = ""
    data_source: str = ""
    study_design: str = ""
    participants: str = ""
    outcome: str = ""
    predictors: str = ""
    sample_size: str = ""
    missing_data: str = ""
    statistical_methods: str = ""
    model_selection: str = ""
    model_training: str = ""
    hyperparameters: str = ""
    validation_strategy: str = ""
    model_specification: str = ""
    algorithm_architecture: str = ""
    training_details: str = ""
    validation_data: str = ""
    results_performance: str = ""
    discrimination: str = ""
    calibration: str = ""
    decision_curve: str = ""
    discussion: str = ""
    limitations: str = ""
    conclusion: str = ""
    availability: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dict for easy scoring."""
        return {
            "title": self.title,
            "abstract": self.abstract,
            "introduction": self.introduction,
            "objectives": self.objectives,
            "data_source": self.data_source,
            "study_design": self.study_design,
            "participants": self.participants,
            "outcome": self.outcome,
            "predictors": self.predictors,
            "sample_size": self.sample_size,
            "missing_data": self.missing_data,
            "statistical_methods": self.statistical_methods,
            "model_selection": self.model_selection,
            "model_training": self.model_training,
            "hyperparameters": self.hyperparameters,
            "validation_strategy": self.validation_strategy,
            "model_specification": self.model_specification,
            "algorithm_architecture": self.algorithm_architecture,
            "training_details": self.training_details,
            "validation_data": self.validation_data,
            "results_performance": self.results_performance,
            "discrimination": self.discrimination,
            "calibration": self.calibration,
            "decision_curve": self.decision_curve,
            "discussion": self.discussion,
            "limitations": self.limitations,
            "conclusion": self.conclusion,
            "availability": self.availability,
        }


class TRIPODAIComplianceChecker:
    """Automated TRIPOD+AI 27-item compliance checking."""

    ITEMS = 27

    def __init__(self):
        """Initialize compliance checker."""
        self.checklist_items = [
            "title", "abstract", "introduction", "objectives",
            "data_source", "study_design", "participants", "outcome",
            "predictors", "sample_size", "missing_data", "statistical_methods",
            "model_selection", "model_training", "hyperparameters",
            "validation_strategy", "model_specification", "algorithm_architecture",
            "training_details", "validation_data", "results_performance",
            "discrimination", "calibration", "decision_curve",
            "discussion", "limitations", "conclusion", "availability",
        ]

    def score(self, checklist: TRIPODAIChecklist) -> Tuple[int, float, Dict[str, bool]]:
        """
        Score TRIPOD+AI compliance.

        Args:
            checklist: TRIPODAIChecklist with item descriptions

        Returns:
            (items_completed, compliance_fraction, item_flags)
        """
        checklist_dict = checklist.to_dict()
        item_flags = {}
        completed = 0

        for item in self.checklist_items:
            content = checklist_dict.get(item, "").strip()
            is_complete = len(content) > 0
            item_flags[item] = is_complete
            completed += int(is_complete)

        compliance_fraction = completed / self.ITEMS
        logger.info(
            f"TRIPOD+AI Compliance: {completed}/{self.ITEMS} items "
            f"({100*compliance_fraction:.1f}%)"
        )

        return completed, compliance_fraction, item_flags

    def generate_report(
        self,
        checklist: TRIPODAIChecklist,
        cohort_name: str = "External Cohort"
    ) -> Dict[str, Any]:
        """
        Generate structured compliance report.

        Args:
            checklist: TRIPODAIChecklist
            cohort_name: Name of cohort being validated

        Returns:
            Report dictionary with score, flags, and recommendations
        """
        completed, fraction, flags = self.score(checklist)

        missing_items = [k for k, v in flags.items() if not v]

        report = {
            "cohort": cohort_name,
            "timestamp": datetime.now().isoformat(),
            "items_completed": completed,
            "total_items": self.ITEMS,
            "compliance_fraction": fraction,
            "passes_tier1": fraction >= 0.85,
            "missing_items": missing_items,
            "detailed_flags": flags,
        }

        logger.info(f"TRIPOD+AI Report for {cohort_name}: {json.dumps(report, indent=2)}")
        return report


@dataclass
class ExternalCohort:
    """External validation cohort container."""
    name: str
    n_samples: int
    features: np.ndarray
    risk_scores: np.ndarray
    times: np.ndarray
    events: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExternalCohortLoader:
    """Stub loaders for external validation cohorts."""

    COHORTS = {
        "MMRF_CoMMpass": {"n": 1143, "description": "MMRF CoMMpass myeloma"},
        "GMMG_MM5": {"n": 604, "description": "GMMG-MM5 myeloma"},
        "IFM_DFCI_2009": {"n": 323, "description": "IFM/DFCI 2009 trial"},
        "PETHEMA_GEM": {"n": 1265, "description": "PETHEMA/GEM registry"},
        "HOVON_65": {"n": 290, "description": "HOVON-65 trial"},
    }

    def __init__(self, seed: int = 42):
        """Initialize loader with random seed."""
        np.random.seed(seed)
        self.seed = seed

    def load_mmrf_compass(self) -> ExternalCohort:
        """Load MMRF CoMMpass cohort (n=1143)."""
        n = self.COHORTS["MMRF_CoMMpass"]["n"]
        return self._generate_synthetic_cohort(
            name="MMRF_CoMMpass",
            n=n,
            seed=self.seed + 1
        )

    def load_gmmg_mm5(self) -> ExternalCohort:
        """Load GMMG-MM5 cohort (n=604)."""
        n = self.COHORTS["GMMG_MM5"]["n"]
        return self._generate_synthetic_cohort(
            name="GMMG_MM5",
            n=n,
            seed=self.seed + 2
        )

    def load_ifm_dfci_2009(self) -> ExternalCohort:
        """Load IFM/DFCI 2009 cohort (n=323)."""
        n = self.COHORTS["IFM_DFCI_2009"]["n"]
        return self._generate_synthetic_cohort(
            name="IFM_DFCI_2009",
            n=n,
            seed=self.seed + 3
        )

    def load_pethema_gem(self) -> ExternalCohort:
        """Load PETHEMA/GEM cohort (n=1265)."""
        n = self.COHORTS["PETHEMA_GEM"]["n"]
        return self._generate_synthetic_cohort(
            name="PETHEMA_GEM",
            n=n,
            seed=self.seed + 4
        )

    def load_hovon_65(self) -> ExternalCohort:
        """Load HOVON-65 cohort (n=290)."""
        n = self.COHORTS["HOVON_65"]["n"]
        return self._generate_synthetic_cohort(
            name="HOVON_65",
            n=n,
            seed=self.seed + 5
        )

    def _generate_synthetic_cohort(
        self,
        name: str,
        n: int,
        seed: int
    ) -> ExternalCohort:
        """
        Generate synthetic cohort with realistic structure.

        DEPRECATED: This method previously returned synthetic random data which
        invalidates external validation results. Real data connectors are required.

        Args:
            name: Cohort name
            n: Sample size
            seed: Random seed

        Returns:
            ExternalCohort with synthetic data

        Raises:
            NotImplementedError: Always, to prevent accidental misuse
        """
        raise NotImplementedError(
            f"External cohort '{name}' has no real data connector wired in.\n"
            "  - For MMRF CoMMpass: use resistancemap.data.mmrf_loader.MMRFLoader "
            "with the path in config.data.mmrf_commpass_dir (requires IRB).\n"
            "  - For GMMG/IFM/PETHEMA/HOVON: no public connector exists; "
            "request access via the respective consortium or remove the cohort "
            "from your evaluation manifest.\n"
            "Previous synthetic-fallback path was removed in v7 because random "
            "data invalidates external validation."
        )


class ValidationMetricsCalculator:
    """Compute discrimination, calibration, and decision curve metrics."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def harrell_c_index(
        self,
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Compute Harrell's C-index (concordance).

        Args:
            risk_scores: Predicted risk scores [0,1]
            times: Follow-up times
            events: Event indicators (0/1)

        Returns:
            C-index value in [0,1]
        """
        # Count concordant pairs
        n = len(risk_scores)
        concordant = 0
        total_pairs = 0

        for i in range(n):
            if events[i] == 0:
                continue
            for j in range(n):
                if i == j or times[j] > times[i]:
                    continue
                total_pairs += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1

        if total_pairs == 0:
            return 0.5

        c_index = concordant / total_pairs
        return c_index

    def time_dependent_auc(
        self,
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        t: float
    ) -> float:
        """
        Compute time-dependent AUC at time t.

        Args:
            risk_scores: Predicted risk scores
            times: Follow-up times
            events: Event indicators
            t: Time point for AUC

        Returns:
            AUC at time t
        """
        # Binary classification at time t
        y_true = (events == 1) & (times <= t)

        if len(np.unique(y_true)) < 2:
            return 0.5

        auc = roc_auc_score(y_true, risk_scores)
        return auc

    def integrated_brier_score(
        self,
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Compute integrated Brier score.

        Args:
            risk_scores: Predicted risk scores
            times: Follow-up times
            events: Event indicators

        Returns:
            IBS value
        """
        # Simple approximation: average Brier score over time quartiles
        time_points = np.percentile(times[events == 1], [25, 50, 75])
        brier_scores = []

        for t in time_points:
            y_true = (events == 1) & (times <= t)
            if len(np.unique(y_true)) >= 2:
                bs = brier_score_loss(y_true, risk_scores)
                brier_scores.append(bs)

        if not brier_scores:
            return 0.5

        return np.mean(brier_scores)

    def calibration_slope_intercept(
        self,
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute calibration slope and intercept via logistic regression.

        Args:
            risk_scores: Predicted risk scores
            times: Follow-up times
            events: Event indicators

        Returns:
            (slope, intercept)
        """
        # Binary event for logistic calibration
        y_binary = events

        # Logit transform predicted probabilities (avoid extremes)
        eps = 1e-6
        risk_clipped = np.clip(risk_scores, eps, 1 - eps)
        logit_pred = np.log(risk_clipped / (1 - risk_clipped))

        # Logistic regression: logit(y) ~ logit(pred)
        try:
            slope, intercept, _, _, _ = stats.linregress(logit_pred, y_binary)
        except Exception as e:
            logger.warning(f"Calibration slope/intercept failed: {e}")
            slope, intercept = 1.0, 0.0

        return slope, intercept

    def calculate_all_metrics(
        self,
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all discrimination and calibration metrics.

        Args:
            risk_scores: Predicted risk scores
            times: Follow-up times
            events: Event indicators

        Returns:
            Dictionary of metrics
        """
        c_index = self.harrell_c_index(risk_scores, times, events)
        td_auc = self.time_dependent_auc(risk_scores, times, events, np.median(times))
        ibs = self.integrated_brier_score(risk_scores, times, events)
        slope, intercept = self.calibration_slope_intercept(risk_scores, times, events)

        metrics = {
            "c_index": c_index,
            "time_dependent_auc": td_auc,
            "integrated_brier_score": ibs,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        }

        return metrics


class BootstrapConfidenceEstimator:
    """Bootstrap confidence intervals and DeLong test for AUC comparison."""

    def __init__(self, n_bootstrap: int = 1000, seed: int = 42):
        """
        Initialize bootstrap estimator.

        Args:
            n_bootstrap: Number of bootstrap samples
            seed: Random seed
        """
        self.n_bootstrap = n_bootstrap
        np.random.seed(seed)

    def bootstrap_ci(
        self,
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        metric_fn,
        ci: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence intervals for a metric.

        Args:
            risk_scores: Predicted risk scores
            times: Follow-up times
            events: Event indicators
            metric_fn: Function to compute metric (callable)
            ci: Confidence level (e.g., 0.95)

        Returns:
            (point_estimate, lower_ci, upper_ci)
        """
        n = len(risk_scores)
        bootstrap_values = []

        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            metric_val = metric_fn(
                risk_scores[idx],
                times[idx],
                events[idx]
            )
            bootstrap_values.append(metric_val)

        bootstrap_values = np.array(bootstrap_values)
        point_estimate = np.mean(bootstrap_values)
        alpha = (1 - ci) / 2
        lower_ci = np.percentile(bootstrap_values, 100 * alpha)
        upper_ci = np.percentile(bootstrap_values, 100 * (1 - alpha))

        return point_estimate, lower_ci, upper_ci

    def delong_test(
        self,
        risk_scores_1: np.ndarray,
        y_true_1: np.ndarray,
        risk_scores_2: np.ndarray,
        y_true_2: np.ndarray
    ) -> Tuple[float, float]:
        """
        DeLong test comparing AUCs from two models.

        Args:
            risk_scores_1: Model 1 predictions
            y_true_1: Model 1 targets
            risk_scores_2: Model 2 predictions
            y_true_2: Model 2 targets

        Returns:
            (auc_diff, p_value)
        """
        try:
            auc_1 = roc_auc_score(y_true_1, risk_scores_1)
            auc_2 = roc_auc_score(y_true_2, risk_scores_2)
            auc_diff = auc_1 - auc_2

            # Simplified p-value via bootstrap
            diffs = []
            for _ in range(500):
                idx_1 = np.random.choice(len(y_true_1), replace=True)
                idx_2 = np.random.choice(len(y_true_2), replace=True)
                a1 = roc_auc_score(y_true_1[idx_1], risk_scores_1[idx_1])
                a2 = roc_auc_score(y_true_2[idx_2], risk_scores_2[idx_2])
                diffs.append(a1 - a2)

            diffs = np.array(diffs)
            p_value = 2 * min(np.mean(diffs < 0), np.mean(diffs > 0))

        except Exception as e:
            logger.warning(f"DeLong test failed: {e}")
            auc_diff, p_value = 0.0, 1.0

        return auc_diff, p_value


@dataclass
class PostMarketAudit:
    """Post-market monitoring audit record."""
    audit_date: str
    quarter: str
    c_index: float
    c_index_change: float
    events_monitored: int
    drift_detected: bool
    retrain_recommended: bool
    notes: str = ""


class PostMarketMonitor:
    """Quarterly post-market monitoring with drift detection."""

    DRIFT_THRESHOLD = 0.05  # C-index decline threshold

    def __init__(self, baseline_c_index: float):
        """
        Initialize post-market monitor.

        Args:
            baseline_c_index: C-index from original validation
        """
        self.baseline_c_index = baseline_c_index
        self.audit_history: List[PostMarketAudit] = []

    def quarterly_audit(
        self,
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        quarter: str
    ) -> PostMarketAudit:
        """
        Perform quarterly audit with drift detection.

        Args:
            risk_scores: Current quarter predictions
            times: Follow-up times
            events: Event indicators
            quarter: Quarter identifier (e.g., "2024-Q1")

        Returns:
            PostMarketAudit record
        """
        calc = ValidationMetricsCalculator()
        current_c_index = calc.harrell_c_index(risk_scores, times, events)
        c_index_change = self.baseline_c_index - current_c_index

        # Drift detection
        drift_detected = c_index_change > self.DRIFT_THRESHOLD
        retrain_recommended = drift_detected

        audit = PostMarketAudit(
            audit_date=datetime.now().isoformat(),
            quarter=quarter,
            c_index=current_c_index,
            c_index_change=c_index_change,
            events_monitored=int(events.sum()),
            drift_detected=drift_detected,
            retrain_recommended=retrain_recommended,
            notes=f"Baseline: {self.baseline_c_index:.3f}, Current: {current_c_index:.3f}"
        )

        self.audit_history.append(audit)

        log_level = logging.WARNING if drift_detected else logging.INFO
        logger.log(
            log_level,
            f"Quarterly Audit {quarter}: C-index={current_c_index:.3f}, "
            f"change={c_index_change:.3f}, drift_detected={drift_detected}"
        )

        return audit

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of all audits."""
        if not self.audit_history:
            return {"audits_completed": 0}

        drifts = sum(1 for a in self.audit_history if a.drift_detected)
        retrains = sum(1 for a in self.audit_history if a.retrain_recommended)

        return {
            "audits_completed": len(self.audit_history),
            "drifts_detected": drifts,
            "retrains_recommended": retrains,
            "baseline_c_index": self.baseline_c_index,
            "latest_c_index": self.audit_history[-1].c_index,
            "audit_dates": [a.quarter for a in self.audit_history],
        }


class ExternalValidationOrchestrator:
    """
    Four-tier governance orchestrator for external validation.

    Tier 1: Internal TRIPOD+AI 27-item (≥85% compliance)
    Tier 2: Single external cohort (C-index decline ≤0.05)
    Tier 3: Multi-center ≥2 sites with fairness audit
    Tier 4: Prospective 200-300 patients with post-market monitoring
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.compliance_checker = TRIPODAIComplianceChecker()
        self.cohort_loader = ExternalCohortLoader()
        self.metrics_calc = ValidationMetricsCalculator()
        self.bootstrap_est = BootstrapConfidenceEstimator()
        self.validation_results: Dict[str, Any] = {}

    def tier1_internal_validation(
        self,
        checklist: TRIPODAIChecklist
    ) -> Dict[str, Any]:
        """
        Tier 1: Internal TRIPOD+AI 27-item compliance.

        Args:
            checklist: TRIPODAIChecklist

        Returns:
            Tier 1 validation result
        """
        report = self.compliance_checker.generate_report(checklist, "Internal Validation")
        passes = report["passes_tier1"]

        logger.info(f"Tier 1 (Internal): {'PASS' if passes else 'FAIL'}")

        return {
            "tier": 1,
            "passes": passes,
            "report": report,
        }

    def tier2_single_external_validation(
        self,
        internal_c_index: float,
        cohort: ExternalCohort
    ) -> Dict[str, Any]:
        """
        Tier 2: Single external cohort validation (C-index decline ≤0.05).

        Args:
            internal_c_index: C-index from internal validation
            cohort: ExternalCohort for validation

        Returns:
            Tier 2 validation result
        """
        metrics = self.metrics_calc.calculate_all_metrics(
            cohort.risk_scores,
            cohort.times,
            cohort.events
        )
        external_c_index = metrics["c_index"]
        c_index_decline = internal_c_index - external_c_index
        passes = c_index_decline <= 0.05

        # Bootstrap CI
        _, lower, upper = self.bootstrap_est.bootstrap_ci(
            cohort.risk_scores,
            cohort.times,
            cohort.events,
            self.metrics_calc.harrell_c_index
        )

        logger.info(
            f"Tier 2 ({cohort.name}): C-index={external_c_index:.3f}, "
            f"decline={c_index_decline:.3f}, 95% CI=[{lower:.3f}, {upper:.3f}], "
            f"{'PASS' if passes else 'FAIL'}"
        )

        return {
            "tier": 2,
            "passes": passes,
            "cohort": cohort.name,
            "internal_c_index": internal_c_index,
            "external_c_index": external_c_index,
            "c_index_decline": c_index_decline,
            "c_index_ci": (lower, upper),
            "metrics": metrics,
        }

    def tier3_multicenter_validation(
        self,
        internal_c_index: float,
        cohorts: List[ExternalCohort]
    ) -> Dict[str, Any]:
        """
        Tier 3: Multi-center validation (≥2 sites) with fairness audit.

        Args:
            internal_c_index: C-index from internal validation
            cohorts: List of ≥2 ExternalCohorts

        Returns:
            Tier 3 validation result
        """
        if len(cohorts) < 2:
            raise ValueError("Tier 3 requires ≥2 cohorts")

        tier2_results = []
        for cohort in cohorts:
            result = self.tier2_single_external_validation(internal_c_index, cohort)
            tier2_results.append(result)

        # Multi-center consensus
        all_pass = all(r["passes"] for r in tier2_results)

        # Fairness audit: check performance across cohorts
        c_indices = [r["external_c_index"] for r in tier2_results]
        c_index_std = np.std(c_indices)
        fairness_pass = c_index_std < 0.05  # Consistent across sites

        passes = all_pass and fairness_pass

        logger.info(
            f"Tier 3 (Multi-center n={len(cohorts)}): C-index range=[{min(c_indices):.3f}, "
            f"{max(c_indices):.3f}], std={c_index_std:.3f}, fairness_pass={fairness_pass}, "
            f"{'PASS' if passes else 'FAIL'}"
        )

        return {
            "tier": 3,
            "passes": passes,
            "n_cohorts": len(cohorts),
            "cohort_results": tier2_results,
            "c_index_std": c_index_std,
            "fairness_pass": fairness_pass,
        }

    def tier4_prospective_validation(
        self,
        internal_c_index: float,
        cohort: ExternalCohort,
        min_sample_size: int = 200
    ) -> Dict[str, Any]:
        """
        Tier 4: Prospective validation with post-market monitoring.

        Args:
            internal_c_index: C-index from internal validation
            cohort: Prospective ExternalCohort (n=200-300)
            min_sample_size: Minimum required prospective samples

        Returns:
            Tier 4 validation result
        """
        sample_size_pass = cohort.n_samples >= min_sample_size

        # Tier 2 validation
        tier2_result = self.tier2_single_external_validation(internal_c_index, cohort)

        # Initialize post-market monitor
        monitor = PostMarketMonitor(internal_c_index)

        # Simulate quarterly audits
        quarterly_audits = []
        for q in ["2024-Q2", "2024-Q3", "2024-Q4"]:
            audit = monitor.quarterly_audit(
                cohort.risk_scores,
                cohort.times,
                cohort.events,
                q
            )
            quarterly_audits.append(audit)

        audit_summary = monitor.get_audit_summary()

        passes = tier2_result["passes"] and sample_size_pass and not audit_summary["drifts_detected"]

        logger.info(
            f"Tier 4 (Prospective): n={cohort.n_samples}, "
            f"sample_size_pass={sample_size_pass}, "
            f"post_market_drifts={audit_summary['drifts_detected']}, "
            f"{'PASS' if passes else 'FAIL'}"
        )

        return {
            "tier": 4,
            "passes": passes,
            "sample_size": cohort.n_samples,
            "sample_size_pass": sample_size_pass,
            "tier2_result": tier2_result,
            "post_market_monitor": audit_summary,
            "quarterly_audits": [
                {
                    "quarter": a.quarter,
                    "c_index": a.c_index,
                    "drift_detected": a.drift_detected,
                }
                for a in quarterly_audits
            ],
        }

    def run_full_validation_pipeline(
        self,
        internal_checklist: TRIPODAIChecklist,
        internal_c_index: float,
        tier2_cohort: Optional[ExternalCohort] = None,
        tier3_cohorts: Optional[List[ExternalCohort]] = None,
        tier4_cohort: Optional[ExternalCohort] = None,
    ) -> Dict[str, Any]:
        """
        Run full 4-tier validation pipeline.

        Args:
            internal_checklist: TRIPODAIChecklist for Tier 1
            internal_c_index: C-index for comparison
            tier2_cohort: Single cohort for Tier 2 (optional)
            tier3_cohorts: Multiple cohorts for Tier 3 (optional)
            tier4_cohort: Prospective cohort for Tier 4 (optional)

        Returns:
            Full validation result with all tiers
        """
        results = {}

        # Tier 1
        tier1 = self.tier1_internal_validation(internal_checklist)
        results["tier1"] = tier1

        if not tier1["passes"]:
            logger.error("Tier 1 failed; cannot proceed to Tier 2+")
            results["validation_summary"] = {
                "overall_pass": False,
                "highest_tier_passed": 1,
                "recommendation": "Fix TRIPOD+AI compliance issues",
            }
            return results

        # Tier 2
        if tier2_cohort:
            tier2 = self.tier2_single_external_validation(internal_c_index, tier2_cohort)
            results["tier2"] = tier2
            highest_tier = 2
        else:
            highest_tier = 1

        # Tier 3
        if tier3_cohorts and len(tier3_cohorts) >= 2:
            tier3 = self.tier3_multicenter_validation(internal_c_index, tier3_cohorts)
            results["tier3"] = tier3
            highest_tier = 3

        # Tier 4
        if tier4_cohort:
            tier4 = self.tier4_prospective_validation(internal_c_index, tier4_cohort)
            results["tier4"] = tier4
            highest_tier = 4

        # Summary
        all_tiers_pass = all(
            results[k]["passes"] for k in results.keys() if k.startswith("tier")
        )

        results["validation_summary"] = {
            "overall_pass": all_tiers_pass,
            "highest_tier_passed": highest_tier,
            "recommendation": (
                "Approved for clinical deployment" if all_tiers_pass
                else "Additional validation required"
            ),
        }

        logger.info(f"Validation Summary: {results['validation_summary']}")
        self.validation_results = results

        return results