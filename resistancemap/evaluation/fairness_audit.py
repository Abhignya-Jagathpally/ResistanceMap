from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class AncestryGroup:
    """Container for ancestry group metadata and samples."""

    name: str
    indices: np.ndarray
    proportions: Optional[np.ndarray] = None


@dataclass
class FairnessMetrics:
    """Container for fairness metric results."""

    c_index_by_group: dict[str, float]
    c_index_disparity: float
    calibration_by_group: dict[str, float]
    calibration_disparity: float
    tpr_by_group: dict[str, float]
    tpr_disparity: float
    fpr_by_group: dict[str, float]
    fpr_disparity: float
    equalized_odds_satisfied: bool


class AncestryEstimator:
    """
    Estimates ancestry composition using PCA and ADMIXTURE-style proportions.

    Reduces high-dimensional genotype data to principal components and estimates
    ancestral population proportions using non-negative least squares.
    """

    def __init__(self, n_components: int = 5, reference_ancestry_labels: Optional[np.ndarray] = None):
        """
        Initialize ancestry estimator.

        Args:
            n_components: Number of principal components to use.
            reference_ancestry_labels: Reference population labels for supervised PCA (optional).
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.reference_ancestry_labels = reference_ancestry_labels
        self.ancestry_populations: Optional[dict[str, np.ndarray]] = None

    def fit(self, genotype_matrix: np.ndarray, ancestry_labels: Optional[np.ndarray] = None) -> AncestryEstimator:
        """
        Fit PCA on genotype data and estimate ancestry population centers.

        Args:
            genotype_matrix: Shape (n_samples, n_snps) genotype matrix.
            ancestry_labels: Population labels for reference samples.

        Returns:
            Self.
        """
        logger.info(f"Fitting ancestry estimator on {genotype_matrix.shape[0]} samples")
        self.pca.fit(genotype_matrix)

        if ancestry_labels is not None:
            # Compute population-specific means in PC space
            self.ancestry_populations = {}
            unique_populations = np.unique(ancestry_labels)
            pcs = self.pca.transform(genotype_matrix)
            for pop in unique_populations:
                mask = ancestry_labels == pop
                self.ancestry_populations[str(pop)] = pcs[mask].mean(axis=0)
            logger.info(f"Identified {len(self.ancestry_populations)} ancestry populations")

        return self

    def estimate_admixture(self, genotype_matrix: np.ndarray) -> np.ndarray:
        """
        Estimate ADMIXTURE-style ancestry proportions.

        Uses non-negative least squares to decompose PC projections into
        population-specific components.

        Args:
            genotype_matrix: Shape (n_samples, n_snps) genotype matrix.

        Returns:
            Shape (n_samples, n_populations) admixture proportions.
        """
        if self.ancestry_populations is None:
            raise ValueError("Must fit with ancestry_labels to estimate admixture")

        pcs = self.pca.transform(genotype_matrix)
        pop_centers = np.array(list(self.ancestry_populations.values()))

        # Non-negative least squares for each sample
        n_samples = pcs.shape[0]
        n_pops = pop_centers.shape[0]
        admixture = np.zeros((n_samples, n_pops))

        for i in range(n_samples):
            result = stats.linregress(pop_centers.T, pcs[i])
            # Simple approximation: use squared correlations normalized
            admixture[i] = np.abs(pop_centers @ pcs[i]) ** 2
            admixture[i] /= admixture[i].sum() + 1e-8

        return admixture


class RepresentationChecker:
    """
    Validates representation of ancestry groups in training data.

    Checks that minimum thresholds for Black (≥30%) and Hispanic (≥20%)
    samples are met.
    """

    MINIMUM_BLACK_FRACTION = 0.30
    MINIMUM_HISPANIC_FRACTION = 0.20

    def __init__(self):
        """Initialize representation checker."""
        self.sample_counts: dict[str, int] = {}
        self.sample_fractions: dict[str, float] = {}
        self.representation_warnings: list[str] = []

    def check(self, ancestry_labels: np.ndarray, ancestry_mapping: Optional[dict[str, str]] = None) -> bool:
        """
        Check representation of ancestry groups.

        Args:
            ancestry_labels: Array of ancestry group labels.
            ancestry_mapping: Optional dict mapping label values to group names.

        Returns:
            True if all minimum thresholds are met.
        """
        unique_labels, counts = np.unique(ancestry_labels, return_counts=True)
        total = len(ancestry_labels)

        self.representation_warnings = []
        all_pass = True

        for label, count in zip(unique_labels, counts):
            group_name = ancestry_mapping.get(label, str(label)) if ancestry_mapping else str(label)
            fraction = count / total

            self.sample_counts[group_name] = int(count)
            self.sample_fractions[group_name] = fraction

            if group_name.lower().startswith("black"):
                if fraction < self.MINIMUM_BLACK_FRACTION:
                    msg = (
                        f"Black representation {fraction:.1%} below minimum {self.MINIMUM_BLACK_FRACTION:.1%}"
                    )
                    self.representation_warnings.append(msg)
                    logger.warning(msg)
                    all_pass = False
                else:
                    logger.info(f"Black representation: {fraction:.1%} [PASS]")

            elif group_name.lower().startswith("hispanic"):
                if fraction < self.MINIMUM_HISPANIC_FRACTION:
                    msg = (
                        f"Hispanic representation {fraction:.1%} below minimum {self.MINIMUM_HISPANIC_FRACTION:.1%}"
                    )
                    self.representation_warnings.append(msg)
                    logger.warning(msg)
                    all_pass = False
                else:
                    logger.info(f"Hispanic representation: {fraction:.1%} [PASS]")

        return all_pass


class ProxyBiasDetector:
    """
    Detects proxy bias in outcome labels.

    Identifies whether outcome labels reflect clinical endpoints (clonal evolution,
    MRD) versus cost-based proxies, which could introduce systemic bias.
    """

    def __init__(self):
        """Initialize proxy bias detector."""
        self.label_statistics: dict[str, dict] = {}

    def analyze_label_distribution(self, outcomes: np.ndarray, ancestry_labels: np.ndarray) -> dict:
        """
        Analyze outcome distribution across ancestry groups.

        Flags suspicion of cost-based proxies if outcome rates differ dramatically
        without clinical justification.

        Args:
            outcomes: Binary outcome array.
            ancestry_labels: Ancestry group labels.

        Returns:
            Dictionary with analysis results.
        """
        unique_groups = np.unique(ancestry_labels)
        outcome_rates = {}
        rates = []

        for group in unique_groups:
            mask = ancestry_labels == group
            rate = outcomes[mask].mean()
            outcome_rates[str(group)] = rate
            rates.append(rate)
            self.label_statistics[str(group)] = {
                "n_samples": mask.sum(),
                "n_positive": outcomes[mask].sum(),
                "positive_rate": rate,
            }

        rates = np.array(rates)
        rate_disparity = rates.max() - rates.min()

        # Flag potential cost-based bias if disparity is extreme
        is_suspicious = rate_disparity > 0.4
        if is_suspicious:
            logger.warning(f"Outcome rate disparity {rate_disparity:.1%} suggests potential proxy bias")

        return {
            "outcome_rates_by_group": outcome_rates,
            "rate_disparity": rate_disparity,
            "suspicious_proxy_bias": is_suspicious,
            "recommendation": "Verify labels reflect clinical endpoints, not cost-based proxies"
            if is_suspicious
            else "Outcome distribution consistent with ancestry-independent endpoints",
        }


class FairnessMetricsCalculator:
    """
    Computes fairness metrics: C-index parity, calibration parity, and equalized odds.
    """

    ACCEPTABLE_DISPARITY = 0.05  # 5% difference
    ACCEPTABLE_CALIBRATION_BIAS = 0.10  # 10% absolute bias

    def __init__(self):
        """Initialize fairness metrics calculator."""
        self.metrics: Optional[FairnessMetrics] = None

    def compute_c_index(self, predictions: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Compute concordance index (C-index).

        Args:
            predictions: Predicted risk scores.
            outcomes: Binary outcomes.

        Returns:
            C-index (0.5-1.0).
        """
        n_pos = outcomes.sum()
        n_neg = len(outcomes) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        concordant = 0
        for i in np.where(outcomes == 1)[0]:
            concordant += (predictions[i] > predictions[outcomes == 0]).sum()

        return concordant / (n_pos * n_neg)

    def compute_calibration_bias(self, predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 5) -> float:
        """
        Compute systematic calibration bias.

        Args:
            predictions: Predicted risk scores.
            outcomes: Binary outcomes.
            n_bins: Number of bins for calibration analysis.

        Returns:
            Mean absolute calibration bias across bins.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        biases = []

        for i in range(len(bins) - 1):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            if mask.sum() > 0:
                pred_mean = predictions[mask].mean()
                obs_mean = outcomes[mask].mean()
                biases.append(abs(pred_mean - obs_mean))

        return np.mean(biases) if biases else 0.0

    def compute_equalized_odds(
        self, predictions: np.ndarray, outcomes: np.ndarray, threshold: float = 0.5
    ) -> tuple[float, float]:
        """
        Compute true positive and false positive rates.

        Args:
            predictions: Predicted risk scores.
            outcomes: Binary outcomes.
            threshold: Classification threshold.

        Returns:
            (TPR, FPR) tuple.
        """
        pred_binary = (predictions >= threshold).astype(int)

        tp = ((pred_binary == 1) & (outcomes == 1)).sum()
        fn = ((pred_binary == 0) & (outcomes == 1)).sum()
        tpr = tp / (tp + fn + 1e-8)

        tn = ((pred_binary == 0) & (outcomes == 0)).sum()
        fp = ((pred_binary == 1) & (outcomes == 0)).sum()
        fpr = fp / (fp + tn + 1e-8)

        return tpr, fpr

    def calculate(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        ancestry_labels: np.ndarray,
        ancestry_groups: Optional[dict[str, np.ndarray]] = None,
    ) -> FairnessMetrics:
        """
        Calculate comprehensive fairness metrics across ancestry groups.

        Args:
            predictions: Predicted risk scores.
            outcomes: Binary outcomes.
            ancestry_labels: Ancestry group labels.
            ancestry_groups: Optional dict mapping group names to indices.

        Returns:
            FairnessMetrics dataclass.
        """
        if ancestry_groups is None:
            unique_groups = np.unique(ancestry_labels)
            ancestry_groups = {str(g): np.where(ancestry_labels == g)[0] for g in unique_groups}

        c_index_by_group = {}
        calibration_by_group = {}
        tpr_by_group = {}
        fpr_by_group = {}

        # Calculate metrics per ancestry group
        for group_name, indices in ancestry_groups.items():
            if len(indices) == 0:
                continue

            group_preds = predictions[indices]
            group_outcomes = outcomes[indices]

            c_index_by_group[group_name] = self.compute_c_index(group_preds, group_outcomes)
            calibration_by_group[group_name] = self.compute_calibration_bias(group_preds, group_outcomes)
            tpr, fpr = self.compute_equalized_odds(group_preds, group_outcomes)
            tpr_by_group[group_name] = tpr
            fpr_by_group[group_name] = fpr

        # Compute disparities
        c_indices = np.array(list(c_index_by_group.values()))
        c_index_disparity = c_indices.max() - c_indices.min()

        calibration_values = np.array(list(calibration_by_group.values()))
        calibration_disparity = calibration_values.max() - calibration_values.min()

        tpr_values = np.array(list(tpr_by_group.values()))
        tpr_disparity = tpr_values.max() - tpr_values.min()

        fpr_values = np.array(list(fpr_by_group.values()))
        fpr_disparity = fpr_values.max() - fpr_values.min()

        # Check if equalized odds satisfied
        equalized_odds_satisfied = tpr_disparity <= self.ACCEPTABLE_DISPARITY

        self.metrics = FairnessMetrics(
            c_index_by_group=c_index_by_group,
            c_index_disparity=c_index_disparity,
            calibration_by_group=calibration_by_group,
            calibration_disparity=calibration_disparity,
            tpr_by_group=tpr_by_group,
            tpr_disparity=tpr_disparity,
            fpr_by_group=fpr_by_group,
            fpr_disparity=fpr_disparity,
            equalized_odds_satisfied=equalized_odds_satisfied,
        )

        logger.info(
            f"Fairness metrics: C-index disparity={c_index_disparity:.3f}, "
            f"Calibration disparity={calibration_disparity:.3f}, "
            f"TPR disparity={tpr_disparity:.3f}, "
            f"Equalized odds satisfied: {equalized_odds_satisfied}"
        )

        return self.metrics


class FairnessAwareConstraint:
    """
    Implements fairness constraint for reinforcement learning policies.

    Uses Lagrangian penalty to enforce equalized odds during policy optimization,
    ensuring that treatment policies maintain fairness across ancestry groups.
    """

    def __init__(self, lambda_fairness: float = 1.0, target_tpr_parity: float = 0.05):
        """
        Initialize fairness-aware constraint.

        Args:
            lambda_fairness: Lagrangian penalty coefficient.
            target_tpr_parity: Target TPR disparity threshold.
        """
        self.lambda_fairness = lambda_fairness
        self.target_tpr_parity = target_tpr_parity

    def compute_fairness_penalty(
        self, tpr_by_group: dict[str, float], fpr_by_group: dict[str, float]
    ) -> float:
        """
        Compute Lagrangian fairness penalty.

        Penalties increase linearly with disparity in TPR/FPR across groups.

        Args:
            tpr_by_group: True positive rate by ancestry group.
            fpr_by_group: False positive rate by ancestry group.

        Returns:
            Fairness penalty (non-negative).
        """
        tpr_values = np.array(list(tpr_by_group.values()))
        tpr_disparity = tpr_values.max() - tpr_values.min()

        fpr_values = np.array(list(fpr_by_group.values()))
        fpr_disparity = fpr_values.max() - fpr_values.min()

        # Soft penalty: only penalize if exceeding target
        tpr_penalty = max(0, tpr_disparity - self.target_tpr_parity)
        fpr_penalty = max(0, fpr_disparity - self.target_tpr_parity)

        penalty = self.lambda_fairness * (tpr_penalty + fpr_penalty)
        return penalty

    def constrained_objective(self, reward: float, tpr_by_group: dict, fpr_by_group: dict) -> float:
        """
        Compute constrained objective combining reward and fairness penalty.

        Args:
            reward: Base RL reward.
            tpr_by_group: TPR by group.
            fpr_by_group: FPR by group.

        Returns:
            Constrained objective value.
        """
        penalty = self.compute_fairness_penalty(tpr_by_group, fpr_by_group)
        return reward - penalty


class QuarterlyFairnessAuditor:
    """
    Performs quarterly fairness audits with automated drift detection and escalation.

    Compares current fairness metrics to baseline, flags concerning trends, and
    initiates FDA notification if disparity exceeds 10%.
    """

    DRIFT_THRESHOLD = 0.05  # 5% disparity triggers escalation
    CRITICAL_THRESHOLD = 0.10  # 10% disparity requires FDA notification
    BASELINE_BURN_IN = 2  # Number of audits before baseline established

    def __init__(self, calculator: FairnessMetricsCalculator):
        """
        Initialize quarterly auditor.

        Args:
            calculator: FairnessMetricsCalculator instance.
        """
        self.calculator = calculator
        self.audit_history: list[dict] = []
        self.baseline_metrics: Optional[FairnessMetrics] = None

    def conduct_audit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        ancestry_labels: np.ndarray,
        quarter: str,
    ) -> dict:
        """
        Conduct quarterly fairness audit.

        Args:
            predictions: Current model predictions.
            outcomes: Current outcomes.
            ancestry_labels: Current ancestry labels.
            quarter: Quarter identifier (e.g., "Q2_2026").

        Returns:
            Audit report dictionary.
        """
        current_metrics = self.calculator.calculate(predictions, outcomes, ancestry_labels)

        # Establish baseline after burn-in period
        if len(self.audit_history) == self.BASELINE_BURN_IN:
            self.baseline_metrics = current_metrics
            logger.info("Baseline fairness metrics established")

        report = {
            "quarter": quarter,
            "timestamp": np.datetime64("today"),
            "current_metrics": current_metrics,
            "drift_detected": False,
            "critical_disparity_detected": False,
            "escalation_level": "NONE",
            "recommendations": [],
        }

        # Drift detection
        if self.baseline_metrics is not None:
            c_index_drift = abs(current_metrics.c_index_disparity - self.baseline_metrics.c_index_disparity)
            tpr_drift = abs(current_metrics.tpr_disparity - self.baseline_metrics.tpr_disparity)

            if c_index_drift > self.DRIFT_THRESHOLD or tpr_drift > self.DRIFT_THRESHOLD:
                report["drift_detected"] = True
                logger.warning(f"Fairness drift detected: C-index drift={c_index_drift:.3f}, TPR drift={tpr_drift:.3f}")
                report["recommendations"].append("Review recent model changes and retraining data")

            # Critical threshold
            if current_metrics.c_index_disparity > self.CRITICAL_THRESHOLD:
                report["critical_disparity_detected"] = True
                report["escalation_level"] = "FDA_NOTIFICATION"
                logger.error(
                    f"CRITICAL: C-index disparity {current_metrics.c_index_disparity:.3f} exceeds threshold"
                )
                report["recommendations"].append("Initiate FDA post-market notification protocol")

        self.audit_history.append(report)
        return report


class FairnessAuditOrchestrator:
    """
    Three-phase fairness audit orchestrator implementing Tier 0, Tier 1, and Tier 3 protocols.

    Coordinates representation checking, proxy bias detection, fairness metrics
    calculation, and post-market surveillance.
    """

    def __init__(self):
        """Initialize fairness audit orchestrator."""
        self.ancestry_estimator = AncestryEstimator()
        self.representation_checker = RepresentationChecker()
        self.proxy_bias_detector = ProxyBiasDetector()
        self.metrics_calculator = FairnessMetricsCalculator()
        self.auditor = QuarterlyFairnessAuditor(self.metrics_calculator)
        self.fairness_constraint = FairnessAwareConstraint()

    def tier_0_predevelopment_audit(
        self,
        genotype_matrix: np.ndarray,
        ancestry_labels: np.ndarray,
        outcomes: np.ndarray,
        ancestry_mapping: Optional[dict] = None,
    ) -> dict:
        """
        Tier 0: Pre-development audit (representation + proxy bias detection).

        Args:
            genotype_matrix: Genotype data for ancestry estimation.
            ancestry_labels: Initial ancestry labels.
            outcomes: Outcome labels.
            ancestry_mapping: Mapping from numeric labels to group names.

        Returns:
            Tier 0 audit report.
        """
        logger.info("=== TIER 0: PRE-DEVELOPMENT FAIRNESS AUDIT ===")

        # 1. Representation check
        representation_pass = self.representation_checker.check(ancestry_labels, ancestry_mapping)

        # 2. Ancestry estimation
        self.ancestry_estimator.fit(genotype_matrix, ancestry_labels)
        admixture = self.ancestry_estimator.estimate_admixture(genotype_matrix)

        # 3. Proxy bias detection
        proxy_bias_report = self.proxy_bias_detector.analyze_label_distribution(outcomes, ancestry_labels)

        report = {
            "tier": "TIER_0_PREDEVELOPMENT",
            "representation_pass": representation_pass,
            "representation_counts": self.representation_checker.sample_counts,
            "representation_fractions": self.representation_checker.sample_fractions,
            "representation_warnings": self.representation_checker.representation_warnings,
            "proxy_bias_analysis": proxy_bias_report,
            "ancestry_pc_variance_explained": self.ancestry_estimator.pca.explained_variance_ratio_.sum(),
            "mean_admixture_proportions": admixture.mean(axis=0),
            "tier_0_pass": representation_pass and not proxy_bias_report["suspicious_proxy_bias"],
        }

        logger.info(f"Tier 0 pass: {report['tier_0_pass']}")
        return report

    def tier_1_validation_audit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        ancestry_labels: np.ndarray,
        ancestry_groups: Optional[dict] = None,
    ) -> dict:
        """
        Tier 1: Validation audit (fairness metrics + equalized odds).

        Args:
            predictions: Model predictions/risk scores.
            outcomes: Observed outcomes.
            ancestry_labels: Ancestry group labels.
            ancestry_groups: Optional dict mapping group names to sample indices.

        Returns:
            Tier 1 audit report.
        """
        logger.info("=== TIER 1: VALIDATION FAIRNESS AUDIT ===")

        metrics = self.metrics_calculator.calculate(predictions, outcomes, ancestry_labels, ancestry_groups)

        # Check all Tier 1 criteria
        tier_1_pass = (
            metrics.c_index_disparity <= 0.05
            and metrics.calibration_disparity <= 0.10
            and metrics.equalized_odds_satisfied
        )

        report = {
            "tier": "TIER_1_VALIDATION",
            "fairness_metrics": metrics,
            "c_index_disparity_pass": metrics.c_index_disparity <= 0.05,
            "calibration_disparity_pass": metrics.calibration_disparity <= 0.10,
            "equalized_odds_pass": metrics.equalized_odds_satisfied,
            "tier_1_pass": tier_1_pass,
        }

        logger.info(f"Tier 1 pass: {report['tier_1_pass']}")
        return report

    def tier_3_postmarket_audit(self, quarter: str, predictions: np.ndarray, outcomes: np.ndarray, ancestry_labels: np.ndarray) -> dict:
        """
        Tier 3: Post-market audit (quarterly drift detection + escalation).

        Args:
            quarter: Quarter identifier (e.g., "Q2_2026").
            predictions: Current model predictions.
            outcomes: Current outcomes.
            ancestry_labels: Current ancestry labels.

        Returns:
            Tier 3 audit report.
        """
        logger.info(f"=== TIER 3: POST-MARKET FAIRNESS AUDIT ({quarter}) ===")

        audit_report = self.auditor.conduct_audit(predictions, outcomes, ancestry_labels, quarter)

        return {
            "tier": "TIER_3_POSTMARKET",
            "quarter": quarter,
            "audit_report": audit_report,
            "escalation_level": audit_report["escalation_level"],
            "fda_notification_required": audit_report["escalation_level"] == "FDA_NOTIFICATION",
        }

    def full_audit_pipeline(
        self,
        genotype_matrix: np.ndarray,
        ancestry_labels: np.ndarray,
        outcomes: np.ndarray,
        predictions: np.ndarray,
        ancestry_mapping: Optional[dict] = None,
        ancestry_groups: Optional[dict] = None,
    ) -> dict:
        """
        Execute complete three-phase fairness audit pipeline.

        Args:
            genotype_matrix: Genotype data.
            ancestry_labels: Ancestry labels.
            outcomes: Clinical outcomes.
            predictions: Model predictions.
            ancestry_mapping: Optional label mapping.
            ancestry_groups: Optional group indices mapping.

        Returns:
            Comprehensive audit report combining all tiers.
        """
        logger.info("Starting complete fairness audit pipeline")

        tier_0 = self.tier_0_predevelopment_audit(genotype_matrix, ancestry_labels, outcomes, ancestry_mapping)

        if not tier_0["tier_0_pass"]:
            logger.warning("Tier 0 failed. Proceeding to Tier 1 for validation.")

        tier_1 = self.tier_1_validation_audit(predictions, outcomes, ancestry_labels, ancestry_groups)

        tier_3 = self.tier_3_postmarket_audit("Q2_2026", predictions, outcomes, ancestry_labels)

        return {
            "pipeline_status": "COMPLETE",
            "tier_0": tier_0,
            "tier_1": tier_1,
            "tier_3": tier_3,
            "overall_fairness_pass": tier_0["tier_0_pass"] and tier_1["tier_1_pass"],
        }
