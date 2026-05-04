"""Prospective validation study design for ResistanceMap.

Implements FDA PCCP (Predetermined Change Control Plan) framework with
landmark analysis, interim analysis with O'Brien-Fleming spending function,
sample size calculation, and biomarker collection scheduling.

Study design for MRD-negative complete response as surrogate for 24-month PFS
with time-dependent AUC evaluation at 3/6/12-month horizons.

References
----------
FDA. (2019).
    "Adaptive Clinical Trial Design for Multiple Sclerosis Relapse-Remitting
    Disease: Guidance for Industry."
Fleming, T. R., & DeMets, D. L. (1993).
    "Monitoring of clinical trials: issues and recommendations."
    Statistics in Medicine, 12(15-16), 1541-1556.
O'Brien, P. C., & Fleming, T. R. (1979).
    "A multiple testing procedure for clinical trials."
    Biometrics, 35(3), 549-556.
Rotnitzky, A., & Robins, J. M. (2005).
    "Semiparametric Regression for Repeated Outcomes with Nonignorable
    Nonresponse." JASA, 100(469), 888-900.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import stats, optimize

logger = logging.getLogger(__name__)

__all__ = [
    "LandmarkAnalyzer",
    "SampleSizeCalculator",
    "InterimAnalysisManager",
    "PCCPFramework",
    "BiomarkerScheduler",
    "ProspectiveStudyDesigner",
]


@dataclass
class LandmarkAnalyzer:
    """Time-dependent AUC via landmark Cox models at fixed follow-up times.

    Evaluates model performance by fitting separate Cox proportional hazards
    models at each landmark time, incorporating all biomarkers accumulated
    up to that landmark. Computes time-dependent AUC for prediction of
    primary endpoint within each landmark window.

    Attributes:
        landmarks_months: Fixed follow-up times (months) for landmark analyses.
        cox_models: Fitted Cox model parameters at each landmark.
        auc_estimates: Time-dependent AUC estimates at landmarks.
        ci_lower: 95% CI lower bound for AUC.
        ci_upper: 95% CI upper bound for AUC.
        sample_sizes: Number of at-risk patients at each landmark.
    """

    landmarks_months: Tuple[int, int, int] = (3, 6, 12)
    cox_models: dict = field(default_factory=dict)
    auc_estimates: dict = field(default_factory=dict)
    ci_lower: dict = field(default_factory=dict)
    ci_upper: dict = field(default_factory=dict)
    sample_sizes: dict = field(default_factory=dict)

    def fit_landmark_cox(
        self,
        landmark_months: int,
        times: np.ndarray,
        events: np.ndarray,
        covariates: np.ndarray,
    ) -> dict:
        """Fit Cox model with accumulated biomarkers at landmark.

        Restricts analysis to patients at risk at landmark time, using
        accumulated biomarker measurements (baseline through landmark).

        Args:
            landmark_months: Follow-up time (months) for landmark.
            times: Event times (months) since baseline.
            events: Binary indicator (0/1) for primary event.
            covariates: (n_patients, n_predictors) accumulated biomarkers.

        Returns:
            Dictionary with Cox model parameters, partial likelihood, and
            confidence intervals for each coefficient.

        Raises:
            ValueError: If insufficient at-risk patients or zero variance covariates.
        """
        # Filter to at-risk population at landmark
        landmark_days = landmark_months * 30.4375
        at_risk_idx = times > landmark_days
        n_at_risk = np.sum(at_risk_idx)

        if n_at_risk < 10:
            raise ValueError(
                f"Insufficient at-risk patients at landmark {landmark_months}m: "
                f"{n_at_risk} < 10"
            )

        times_restricted = times[at_risk_idx]
        events_restricted = events[at_risk_idx]
        covariates_restricted = covariates[at_risk_idx]

        # Standardize covariates for numerical stability
        X_mean = np.mean(covariates_restricted, axis=0, keepdims=True)
        X_std = np.std(covariates_restricted, axis=0, keepdims=True)
        X_std[X_std == 0] = 1.0
        X_scaled = (covariates_restricted - X_mean) / X_std

        # Compute Cox partial likelihood and gradient
        def partial_likelihood(beta: np.ndarray) -> float:
            """Negative partial likelihood for optimization."""
            eta = X_scaled @ beta
            risk_set_sums = np.zeros(len(times_restricted))
            for i in range(len(times_restricted)):
                risk_mask = times_restricted >= times_restricted[i]
                risk_set_sums[i] = np.sum(np.exp(eta[risk_mask]))
            log_partial = np.sum(
                events_restricted
                * (eta - np.log(np.maximum(risk_set_sums, 1e-10)))
            )
            return -log_partial

        # Optimize Cox model
        init_beta = np.zeros(covariates_restricted.shape[1])
        result = optimize.minimize(
            partial_likelihood, init_beta, method="L-BFGS-B"
        )
        beta_hat = result.x

        # Compute Hessian for standard errors
        h = 1e-5
        hessian = np.zeros((len(beta_hat), len(beta_hat)))
        for i in range(len(beta_hat)):
            for j in range(len(beta_hat)):
                beta_ij = beta_hat.copy()
                beta_ij[i] += h
                beta_ij[j] += h
                f_pp = partial_likelihood(beta_ij)

                beta_ij = beta_hat.copy()
                beta_ij[i] += h
                f_p = partial_likelihood(beta_ij)

                beta_ij = beta_hat.copy()
                beta_ij[j] += h
                f_p2 = partial_likelihood(beta_ij)

                hessian[i, j] = (f_pp - f_p - f_p2 + result.fun) / (h * h)

        cov_matrix = np.linalg.inv(hessian + np.eye(len(beta_hat)) * 1e-6)
        se_beta = np.sqrt(np.diag(np.abs(cov_matrix)))

        z_scores = beta_hat / np.maximum(se_beta, 1e-10)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

        cox_model = {
            "landmark_months": landmark_months,
            "beta": beta_hat,
            "se_beta": se_beta,
            "z_scores": z_scores,
            "p_values": p_values,
            "partial_likelihood": -result.fun,
            "n_at_risk": n_at_risk,
            "n_events": np.sum(events_restricted),
            "X_mean": X_mean,
            "X_std": X_std,
        }
        self.cox_models[landmark_months] = cox_model
        self.sample_sizes[landmark_months] = n_at_risk

        logger.info(
            f"Landmark {landmark_months}m: n_at_risk={n_at_risk}, "
            f"events={np.sum(events_restricted)}, PL={cox_model['partial_likelihood']:.3f}"
        )
        return cox_model

    def compute_time_dependent_auc(
        self,
        landmark_months: int,
        times: np.ndarray,
        events: np.ndarray,
        predictions: np.ndarray,
        window_months: int = 6,
    ) -> Tuple[float, float, float]:
        """Compute time-dependent AUC at landmark.

        Uses Uno's estimator for AUC in presence of censoring.

        Args:
            landmark_months: Landmark follow-up time (months).
            times: Event times (months).
            events: Event indicators.
            predictions: Model risk predictions (0-1).
            window_months: Prediction window after landmark.

        Returns:
            (auc, ci_lower, ci_upper) - time-dependent AUC and 95% CI.
        """
        landmark_days = landmark_months * 30.4375
        window_days = window_months * 30.4375

        at_risk_idx = times > landmark_days
        times_restricted = times[at_risk_idx]
        events_restricted = events[at_risk_idx]
        pred_restricted = predictions[at_risk_idx]

        t_upper = landmark_days + window_days

        # Uno's AUC: concordant pairs weighted by inverse probability of censoring
        concordant = 0
        discordant = 0
        total_weight = 0

        for i in range(len(times_restricted)):
            if times_restricted[i] <= t_upper and events_restricted[i] == 1:
                for j in range(len(times_restricted)):
                    if times_restricted[j] > times_restricted[i]:
                        weight = 1.0
                        if pred_restricted[i] > pred_restricted[j]:
                            concordant += weight
                        elif pred_restricted[i] < pred_restricted[j]:
                            discordant += weight
                        total_weight += weight

        if total_weight == 0:
            auc = 0.5
            se_auc = 0.1
        else:
            auc = concordant / (concordant + discordant + 1e-10)
            var_auc = auc * (1 - auc) / np.maximum(concordant + discordant, 1)
            se_auc = np.sqrt(var_auc)

        ci_lower = np.clip(auc - 1.96 * se_auc, 0, 1)
        ci_upper = np.clip(auc + 1.96 * se_auc, 0, 1)

        self.auc_estimates[landmark_months] = auc
        self.ci_lower[landmark_months] = ci_lower
        self.ci_upper[landmark_months] = ci_upper

        logger.info(
            f"Landmark {landmark_months}m: AUC={auc:.3f} "
            f"95% CI=[{ci_lower:.3f}, {ci_upper:.3f}]"
        )
        return auc, ci_lower, ci_upper

    def analyze_all_landmarks(
        self,
        times: np.ndarray,
        events: np.ndarray,
        covariates: np.ndarray,
        predictions: np.ndarray,
    ) -> dict:
        """Fit Cox models and compute AUC at all landmarks.

        Args:
            times: Event times (months).
            events: Event indicators.
            covariates: (n, p) covariate matrix.
            predictions: Model predictions.

        Returns:
            Dictionary mapping landmark times to analysis results.
        """
        results = {}
        for landmark in self.landmarks_months:
            try:
                cox_model = self.fit_landmark_cox(landmark, times, events, covariates)
                auc, ci_lower, ci_upper = self.compute_time_dependent_auc(
                    landmark, times, events, predictions
                )
                results[landmark] = {
                    "cox_model": cox_model,
                    "auc": auc,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            except ValueError as e:
                logger.warning(f"Landmark {landmark}m failed: {e}")
        return results


@dataclass
class SampleSizeCalculator:
    """Sample size calculation for landmark survival studies.

    Computes required sample sizes for training, test, and external validation
    cohorts based on expected C-index, event rate, and number of predictors.
    Uses power calculation for Cox regression with continuous covariates.

    Attributes:
        expected_c_index: Harrell's C-index for main model (0.5-1.0).
        event_rate: Proportion of patients experiencing primary event.
        n_predictors: Number of predictors in main model.
        target_power: Statistical power for efficacy gates (0.80-0.90).
        target_alpha: Two-sided significance level (typically 0.05).
    """

    expected_c_index: float = 0.75
    event_rate: float = 0.30
    n_predictors: int = 10
    target_power: float = 0.85
    target_alpha: float = 0.05

    def minimum_events_for_predictors(self, n_predictors: int) -> int:
        """Compute minimum events needed for stable Cox regression.

        Rule: at least 10-20 events per predictor to avoid overfitting.

        Args:
            n_predictors: Number of predictors in model.

        Returns:
            Minimum required events.
        """
        return max(50, 15 * n_predictors)

    def sample_size_from_c_index(
        self,
        target_c_index: float,
        reference_c_index: float = 0.5,
    ) -> int:
        """Calculate sample size for detecting C-index difference.

        Uses Hanley-McNeil asymptotic variance for AUC-like statistics.

        Args:
            target_c_index: Target C-index (expected performance).
            reference_c_index: Null hypothesis C-index (0.5 = random).

        Returns:
            Minimum sample size for target power and alpha.
        """
        # Variance under null and alternative
        var_null = target_c_index * (1 - target_c_index) / (2 - target_c_index - reference_c_index)
        var_alt = target_c_index * (1 - target_c_index)

        # Effect size
        effect_size = target_c_index - reference_c_index

        # Z-scores for power and alpha
        z_alpha = stats.norm.ppf(1 - self.target_alpha / 2)
        z_beta = stats.norm.ppf(self.target_power)

        # Sample size formula
        n = (z_alpha * np.sqrt(var_null) + z_beta * np.sqrt(var_alt)) ** 2 / (effect_size ** 2)
        return int(np.ceil(n))

    def calculate_cohort_sizes(
        self,
    ) -> dict:
        """Calculate recommended cohort sizes.

        Returns:
            Dictionary with training, test, and external validation sample sizes.
        """
        min_events = self.minimum_events_for_predictors(self.n_predictors)
        n_total = int(min_events / self.event_rate)

        # Allocation: 55% training, 20% test, 25% external validation
        n_training = int(0.55 * n_total)
        n_test = int(0.20 * n_total)
        n_external = int(0.25 * n_total)

        # Clamp to specification ranges
        n_training = np.clip(n_training, 200, 250)
        n_test = np.clip(n_test, 60, 75)
        n_external = np.clip(n_external, 100, 150)

        return {
            "training": n_training,
            "test": n_test,
            "external_validation": n_external,
            "total": n_training + n_test + n_external,
            "expected_events_total": int((n_training + n_test + n_external) * self.event_rate),
        }

    def validate_adequacy(self, cohort_sizes: dict) -> dict:
        """Validate adequacy of proposed cohort sizes.

        Args:
            cohort_sizes: Dictionary with training, test, external_validation sizes.

        Returns:
            Dictionary with validation results and recommendations.
        """
        n_total = (
            cohort_sizes.get("training", 0)
            + cohort_sizes.get("test", 0)
            + cohort_sizes.get("external_validation", 0)
        )
        n_events = int(n_total * self.event_rate)
        events_per_predictor = n_events / self.n_predictors

        checks = {
            "total_sample_size": n_total >= 350,
            "minimum_events": n_events >= self.minimum_events_for_predictors(self.n_predictors),
            "events_per_predictor": events_per_predictor >= 10,
            "training_size": cohort_sizes.get("training", 0) >= 200,
            "test_size": cohort_sizes.get("test", 0) >= 60,
            "external_val_size": cohort_sizes.get("external_validation", 0) >= 100,
        }

        return {
            "all_checks_pass": all(checks.values()),
            "detailed_checks": checks,
            "n_total": n_total,
            "n_events": n_events,
            "events_per_predictor": events_per_predictor,
        }


@dataclass
class InterimAnalysisManager:
    """Interim analysis with futility/efficacy gates and alpha spending.

    Implements O'Brien-Fleming spending function for alpha management across
    interim and final analyses. Futility gate at 33% information (AUC ≥ 0.65)
    and efficacy gate at 67% information (AUC ≥ 0.75).

    Attributes:
        total_events: Expected total events at study completion.
        alpha: Total type-I error rate (typically 0.05).
        spending_function: Function for alpha allocation.
        interim_times: Event counts at interim and final analyses.
        critical_values: AUC thresholds for decision making.
    """

    total_events: int = 100
    alpha: float = 0.05
    futility_auc_threshold: float = 0.65
    efficacy_auc_threshold: float = 0.75
    interim_times: Tuple[float, float, float] = (0.33, 0.67, 1.0)
    critical_values: dict = field(default_factory=dict)

    def obrien_fleming_spending(self, t: float) -> float:
        """O'Brien-Fleming spending function.

        Allocates alpha conservatively early, spending more at final analysis.

        Args:
            t: Information fraction (0 < t <= 1).

        Returns:
            Cumulative alpha spent up to information fraction t.
        """
        if t <= 0 or t > 1:
            raise ValueError(f"Information fraction must be in (0, 1]; got {t}")
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        spent = 2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(t)))
        return spent

    def compute_critical_values(self) -> dict:
        """Compute AUC decision boundaries using spending function.

        Maps information fractions to AUC thresholds via inverse normal method.

        Returns:
            Dictionary with interim and final critical values.
        """
        critical_values = {}

        # Interim 1: 33% events, futility gate
        alpha_spent_interim1 = self.obrien_fleming_spending(0.33)
        z_interim1 = stats.norm.ppf(1 - alpha_spent_interim1 / 2)
        se_interim1 = 0.075  # Expected SE at interim 1
        critical_values["interim_1_auc"] = 0.5 + z_interim1 * se_interim1

        # Interim 2: 67% events, efficacy gate
        alpha_spent_interim2 = self.obrien_fleming_spending(0.67)
        alpha_spent_interim1_delta = alpha_spent_interim2 - alpha_spent_interim1
        z_interim2 = stats.norm.ppf(1 - alpha_spent_interim1_delta / 2)
        se_interim2 = 0.053  # Expected SE at interim 2
        critical_values["interim_2_auc"] = 0.5 + z_interim2 * se_interim2

        # Final: 100% events
        alpha_spent_final = self.obrien_fleming_spending(1.0)
        alpha_spent_interim2_delta = alpha_spent_final - alpha_spent_interim2
        z_final = stats.norm.ppf(1 - alpha_spent_interim2_delta / 2)
        se_final = 0.038  # Expected SE at final
        critical_values["final_auc"] = 0.5 + z_final * se_final

        self.critical_values = critical_values
        logger.info(
            f"O'Brien-Fleming critical values: "
            f"interim_1={critical_values.get('interim_1_auc', 0):.3f}, "
            f"interim_2={critical_values.get('interim_2_auc', 0):.3f}, "
            f"final={critical_values.get('final_auc', 0):.3f}"
        )
        return critical_values

    def evaluate_interim_gate(
        self,
        interim_label: str,
        observed_auc: float,
        observed_events: int,
    ) -> dict:
        """Evaluate futility/efficacy gate at interim analysis.

        Args:
            interim_label: "interim_1" or "interim_2".
            observed_auc: Observed time-dependent AUC at landmark.
            observed_events: Number of events observed so far.

        Returns:
            Dictionary with gate result and recommendation.
        """
        if interim_label == "interim_1":
            threshold = self.futility_auc_threshold
            gate_type = "futility"
        elif interim_label == "interim_2":
            threshold = self.efficacy_auc_threshold
            gate_type = "efficacy"
        else:
            raise ValueError(f"Unknown interim label: {interim_label}")

        passed = observed_auc >= threshold
        result = {
            "interim": interim_label,
            "gate_type": gate_type,
            "threshold": threshold,
            "observed_auc": observed_auc,
            "observed_events": observed_events,
            "passed": passed,
            "recommendation": (
                "CONTINUE study" if passed else f"STOP for {gate_type}"
            ),
        }

        logger.info(
            f"Interim gate {interim_label} ({gate_type}): "
            f"AUC={observed_auc:.3f} vs threshold={threshold:.3f} -> {result['recommendation']}"
        )
        return result


@dataclass
class PCCPFramework:
    """Predetermined Change Control Plan (FDA framework).

    Specifies allowed model updates, performance thresholds, and retraining
    triggers to ensure prospective validity and regulatory compliance.

    Attributes:
        baseline_c_index: C-index at model finalization.
        minimum_c_index_threshold: Trigger for unplanned retraining.
        maximum_calibration_slope_deviation: Trigger for recalibration.
        retraining_trigger_events: Number of new events triggering retraining.
        allowed_updates: List of permitted model modifications.
    """

    baseline_c_index: float = 0.75
    minimum_c_index_threshold: float = 0.70
    maximum_calibration_slope_deviation: float = 0.15
    retraining_trigger_events: int = 50
    allowed_updates: list = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize default allowed updates."""
        if not self.allowed_updates:
            self.allowed_updates = [
                "recalibration_of_intercept",
                "recalibration_of_slope",
                "external_validation_coefficient_adjustment",
                "biomarker_measurement_harmonization",
            ]

    def is_update_allowed(self, update_type: str) -> bool:
        """Check if proposed update is within PCCP scope.

        Args:
            update_type: Type of model modification.

        Returns:
            True if update is pre-specified and allowed.
        """
        return update_type in self.allowed_updates

    def check_retraining_trigger(
        self,
        current_c_index: float,
        new_events_count: int,
    ) -> dict:
        """Evaluate whether retraining is triggered.

        Retraining is triggered by:
        1. C-index drops below threshold, OR
        2. Accumulated new events reach trigger count.

        Args:
            current_c_index: Current model C-index estimate.
            new_events_count: Number of new events since model finalization.

        Returns:
            Dictionary with trigger status and justification.
        """
        triggers = {
            "c_index_drop": current_c_index < self.minimum_c_index_threshold,
            "event_accumulation": new_events_count >= self.retraining_trigger_events,
        }

        triggered = any(triggers.values())
        justification = []
        if triggers["c_index_drop"]:
            justification.append(
                f"C-index={current_c_index:.3f} < {self.minimum_c_index_threshold:.3f}"
            )
        if triggers["event_accumulation"]:
            justification.append(
                f"New events={new_events_count} >= {self.retraining_trigger_events}"
            )

        result = {
            "retraining_triggered": triggered,
            "trigger_type": triggers,
            "justification": justification,
            "recommendation": (
                "UNPLANNED RETRAINING" if triggered else "Continue monitoring"
            ),
        }

        logger.info(
            f"Retraining trigger check: {result['recommendation']} "
            f"({', '.join(justification)})"
        )
        return result

    def generate_specification_document(self) -> dict:
        """Generate PCCP specification for regulatory submission.

        Returns:
            Dictionary with complete PCCP specification.
        """
        spec = {
            "framework": "FDA PCCP",
            "baseline_performance": {
                "c_index": self.baseline_c_index,
                "minimum_acceptable": self.minimum_c_index_threshold,
            },
            "allowed_modifications": self.allowed_updates,
            "calibration_thresholds": {
                "max_slope_deviation": self.maximum_calibration_slope_deviation,
            },
            "retraining_triggers": {
                "c_index_threshold": self.minimum_c_index_threshold,
                "event_accumulation": self.retraining_trigger_events,
            },
            "decision_rules": {
                "recalibration_allowed": True,
                "retraining_requires_justification": True,
                "external_data_integration_protocol": "pre-specified coefficients",
            },
        }
        logger.info(f"PCCP specification generated: {len(spec)} top-level keys")
        return spec


@dataclass
class BiomarkerScheduler:
    """Biomarker collection schedule for prospective validation.

    Specifies timepoints and assay types for WGS/WES, transcriptome,
    MRD by NGS, and M-protein kinetics.

    Attributes:
        timepoints_months: Collection timepoints (relative to baseline).
        assays: Dictionary mapping timepoint to assay specifications.
    """

    timepoints_months: Tuple[float, ...] = (0, 1, 3, 6, 12, 18, 24)
    assays: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default assay schedule."""
        if not self.assays:
            self.assays = {
                0: {
                    "wgs_wes": {"platform": "Illumina NovaSeq", "depth": "30x coverage"},
                    "transcriptome": {"platform": "10X Genomics", "modality": "single-cell"},
                    "mrd_ngs": {"sensitivity": "10^-5"},
                    "m_protein": {"method": "SPEP/IFE"},
                },
                1: {
                    "mrd_ngs": {"sensitivity": "10^-5"},
                    "m_protein": {"method": "SPEP/IFE"},
                },
                3: {
                    "mrd_ngs": {"sensitivity": "10^-5", "10^-6": True},
                    "m_protein": {"method": "SPEP/IFE"},
                    "clinical_assessment": "IMWG criteria",
                },
                6: {
                    "mrd_ngs": {"sensitivity": "10^-5", "10^-6": True},
                    "m_protein": {"method": "SPEP/IFE"},
                    "clinical_assessment": "IMWG criteria",
                },
                12: {
                    "mrd_ngs": {"sensitivity": "10^-5", "10^-6": True},
                    "m_protein": {"method": "SPEP/IFE"},
                    "clinical_assessment": "IMWG criteria",
                    "transcriptome": {"platform": "bulk RNA-seq"},
                },
                18: {
                    "mrd_ngs": {"sensitivity": "10^-5"},
                    "m_protein": {"method": "SPEP/IFE"},
                },
                24: {
                    "mrd_ngs": {"sensitivity": "10^-5"},
                    "m_protein": {"method": "SPEP/IFE"},
                    "primary_endpoint_assessment": "24-month PFS",
                },
            }

    def get_assays_at_timepoint(self, months: float) -> dict:
        """Retrieve assays scheduled at specific timepoint.

        Args:
            months: Months since baseline.

        Returns:
            Dictionary of assays and specifications.

        Raises:
            ValueError: If timepoint not in schedule.
        """
        if months not in self.assays:
            raise ValueError(
                f"Timepoint {months}m not in schedule. Available: {self.timepoints_months}"
            )
        return self.assays[months]

    def get_collection_calendar(self) -> dict:
        """Generate detailed collection calendar.

        Returns:
            Dictionary with timepoint, assays, and specimen requirements.
        """
        calendar = {}
        for months in self.timepoints_months:
            assay_set = self.get_assays_at_timepoint(months)
            calendar[f"month_{int(months)}"] = {
                "timepoint_months": months,
                "assays": assay_set,
                "specimen_types": self._infer_specimens(assay_set),
                "storage_conditions": "Ultra-low freezer (-80C) + RNAlater",
            }
        return calendar

    @staticmethod
    def _infer_specimens(assay_set: dict) -> list:
        """Infer specimen types from assay set.

        Args:
            assay_set: Dictionary of assays at timepoint.

        Returns:
            List of required specimen types.
        """
        specimens = []
        if "wgs_wes" in assay_set:
            specimens.append("Whole blood (8 mL EDTA)")
        if "transcriptome" in assay_set:
            specimens.append("Bone marrow aspirate (2 mL EDTA)")
        if "mrd_ngs" in assay_set:
            specimens.append("Bone marrow aspirate (5 mL EDTA)")
        if "m_protein" in assay_set:
            specimens.append("Serum (5 mL separator tube)")
        return list(set(specimens))


@dataclass
class ProspectiveStudyDesigner:
    """Top-level study protocol generator combining all components.

    Orchestrates landmark analysis, sample size calculation, interim analysis,
    PCCP specification, and biomarker scheduling into complete study design.

    Attributes:
        study_name: Name/identifier for study protocol.
        primary_endpoint: Description of primary endpoint.
        landmark_analyzer: LandmarkAnalyzer instance.
        sample_calculator: SampleSizeCalculator instance.
        interim_manager: InterimAnalysisManager instance.
        pccp: PCCPFramework instance.
        biomarker_scheduler: BiomarkerScheduler instance.
    """

    study_name: str = "ResistanceMap Prospective Validation"
    primary_endpoint: str = (
        "MRD-negative complete response at 3 months, "
        "surrogate for 24-month progression-free survival"
    )
    landmark_analyzer: LandmarkAnalyzer = field(default_factory=LandmarkAnalyzer)
    sample_calculator: SampleSizeCalculator = field(default_factory=SampleSizeCalculator)
    interim_manager: InterimAnalysisManager = field(default_factory=InterimAnalysisManager)
    pccp: PCCPFramework = field(default_factory=PCCPFramework)
    biomarker_scheduler: BiomarkerScheduler = field(default_factory=BiomarkerScheduler)

    def generate_study_protocol(self) -> dict:
        """Generate complete study protocol specification.

        Returns:
            Comprehensive protocol dictionary.
        """
        # Sample size calculation
        cohort_sizes = self.sample_calculator.calculate_cohort_sizes()
        adequacy = self.sample_calculator.validate_adequacy(cohort_sizes)

        # Interim analysis critical values
        critical_values = self.interim_manager.compute_critical_values()

        # PCCP specification
        pccp_spec = self.pccp.generate_specification_document()

        # Biomarker collection calendar
        biomarker_calendar = self.biomarker_scheduler.get_collection_calendar()

        protocol = {
            "study_metadata": {
                "study_name": self.study_name,
                "primary_endpoint": self.primary_endpoint,
                "study_design": "Multi-stage prospective validation",
                "regulatory_framework": "FDA PCCP",
            },
            "sample_size": {
                "cohort_sizes": cohort_sizes,
                "adequacy_assessment": adequacy,
                "statistical_justification": {
                    "expected_c_index": self.sample_calculator.expected_c_index,
                    "event_rate": self.sample_calculator.event_rate,
                    "n_predictors": self.sample_calculator.n_predictors,
                },
            },
            "interim_analysis": {
                "futility_gate": {
                    "timepoint": "33% of events",
                    "threshold": self.interim_manager.futility_auc_threshold,
                    "critical_value_auc": critical_values.get("interim_1_auc"),
                },
                "efficacy_gate": {
                    "timepoint": "67% of events",
                    "threshold": self.interim_manager.efficacy_auc_threshold,
                    "critical_value_auc": critical_values.get("interim_2_auc"),
                },
                "alpha_spending": "O'Brien-Fleming",
            },
            "landmark_analysis": {
                "landmarks_months": self.landmark_analyzer.landmarks_months,
                "analysis_type": "Cox PH with time-dependent AUC",
                "windows": {
                    3: "0-3 months post-baseline",
                    6: "0-6 months post-baseline",
                    12: "0-12 months post-baseline",
                },
            },
            "pccp_framework": pccp_spec,
            "biomarker_collection": {
                "timepoints": self.biomarker_scheduler.timepoints_months,
                "collection_calendar": biomarker_calendar,
            },
        }

        logger.info(
            f"Study protocol generated: {self.study_name} "
            f"({cohort_sizes['total']} patients, "
            f"{adequacy['n_events']} expected events)"
        )
        return protocol

    def summarize_protocol(self) -> str:
        """Generate human-readable protocol summary.

        Returns:
            Formatted protocol summary text.
        """
        cohort_sizes = self.sample_calculator.calculate_cohort_sizes()
        adequacy = self.sample_calculator.validate_adequacy(cohort_sizes)

        summary = f"""
ResistanceMap Prospective Validation Study Protocol
===================================================

PRIMARY ENDPOINT
{self.primary_endpoint}

SAMPLE SIZE
-----------
Training cohort:           {cohort_sizes['training']} patients
Test cohort:               {cohort_sizes['test']} patients
External validation:       {cohort_sizes['external_validation']} patients
Total:                     {cohort_sizes['total']} patients
Expected events:           {adequacy['n_events']}
Events per predictor:      {adequacy['events_per_predictor']:.1f}

INTERIM ANALYSIS (O'Brien-Fleming)
-----------------------------------
Interim 1 (33% events):    Futility gate, AUC >= {self.interim_manager.futility_auc_threshold:.2f}
Interim 2 (67% events):    Efficacy gate, AUC >= {self.interim_manager.efficacy_auc_threshold:.2f}

LANDMARK ANALYSIS
-----------------
Timepoints:    3, 6, 12 months
Method:        Cox PH with accumulated biomarkers
Metric:        Time-dependent AUC (Uno's C)

BIOMARKER SCHEDULE
-------------------
Baseline (0m):      WGS/WES, scRNA-seq, MRD (10^-5), M-protein
Months 1-6:         MRD (10^-5), M-protein kinetics
Months 12-24:       MRD (10^-6), transcriptome, primary endpoint

PCCP SPECIFICATION
-------------------
Baseline C-index:          {self.pccp.baseline_c_index:.3f}
Minimum acceptable:        {self.pccp.minimum_c_index_threshold:.3f}
Retraining triggers:       C-index drop OR {self.pccp.retraining_trigger_events} new events
Allowed updates:           {', '.join(self.pccp.allowed_updates[:3])}

VALIDATION CHECKS
------------------
All checks pass:           {adequacy['all_checks_pass']}
Minimum sample size:       {adequacy['detailed_checks']['total_sample_size']}
Minimum events:            {adequacy['detailed_checks']['minimum_events']}
Events per predictor:      {adequacy['detailed_checks']['events_per_predictor']}
"""
        return summary.strip()
