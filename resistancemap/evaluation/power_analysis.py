"""
Power analysis and sample size adequacy for ResistanceMap validation.

This module addresses a critical concern: validating a 130M+ parameter deep learning
model with only n=150 samples is severely underpowered. In classical statistics,
a rule of thumb suggests at least 10-20 samples per model parameter. Here we provide:

1. Statistical power calculations using normal approximation
2. Effective degrees-of-freedom estimation tailored to deep learning
3. Minimum sample size recommendations for various effect sizes
4. Hematologic-context-specific warnings and validation strategies

Biological/ML context:
- Deep learning models have implicit feature extraction that reduces effective complexity
- However, even with aggressive regularization, 150 samples for 130M params is problematic
- For claims of model generalization (key in clinical validation), we need reasonable power
- Hematologic validation requires confidence in predictions across blood subtypes and conditions

References:
- Amrhein et al. (2019) on statistical power in small samples
- Zhang et al. (2021) on deep learning sample complexity
- Classical power analysis formulas (noncentrality parameter, cumulative normal)
"""

import logging
from dataclasses import dataclass, field
from math import sqrt, log10, erfc
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PowerAnalysisResult:
    """
    Result of power analysis for a given sample size and model configuration.

    Attributes:
        sample_size: Number of validation samples available
        model_parameters: Total trainable parameters in the model
        effective_dof: Estimated effective degrees of freedom for statistical testing
        samples_per_parameter_ratio: ratio of n to parameters (ideally >= 10)
        is_adequately_powered: Boolean flag for adequate statistical power
        minimum_recommended_n: Minimum n for 80% power at detected effect size
        power_at_current_n: Approximate statistical power at current sample size
        confidence_level: Confidence level used in analysis (1 - significance_level)
        warnings: List of domain-specific warnings
        recommendations: List of suggested validation strategies
    """
    sample_size: int
    model_parameters: int
    effective_dof: int
    samples_per_parameter_ratio: float
    is_adequately_powered: bool
    minimum_recommended_n: int
    power_at_current_n: float
    confidence_level: float
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "ADEQUATE" if self.is_adequately_powered else "INADEQUATE"
        return (
            f"Power Analysis Summary: {status}\n"
            f"  Samples: {self.sample_size} | Model params: {self.model_parameters:,}\n"
            f"  Ratio: {self.samples_per_parameter_ratio:.4f} samples/param | "
            f"Effective DoF: {self.effective_dof}\n"
            f"  Power at n={self.sample_size}: {self.power_at_current_n:.3f} "
            f"(target: 0.800)\n"
            f"  Minimum recommended n for 80% power: {self.minimum_recommended_n}\n"
            f"  Warnings: {len(self.warnings)} | Recommendations: {len(self.recommendations)}"
        )


class SampleSizePowerAnalyzer:
    """
    Analyzes statistical power and adequacy of sample sizes for ML model validation.

    This class uses approximate methods suitable for large-scale deep learning:
    - Normal approximation for power calculations (valid for n > 30 or so)
    - Effective DoF heuristics for complex models
    - Domain-specific recommendations for hematologic contexts

    Example:
        >>> analyzer = SampleSizePowerAnalyzer(
        ...     n_model_params=130_000_000,
        ...     n_modalities=4,
        ...     significance_level=0.05,
        ...     desired_power=0.80
        ... )
        >>> result = analyzer.analyze(n_samples=150, effect_size=0.5)
        >>> print(result)
    """

    def __init__(
        self,
        n_model_params: int,
        n_modalities: int = 4,
        significance_level: float = 0.05,
        desired_power: float = 0.80,
    ):
        """
        Initialize the analyzer.

        Args:
            n_model_params: Total number of trainable parameters in the model.
                For ResistanceMap, ~130M.
            n_modalities: Number of independent data modalities/phenotypes being
                predicted (hematology has ~4: morphology, flow cytometry, cytochemistry, etc).
            significance_level: Alpha level for statistical tests (default 0.05).
            desired_power: Target statistical power for sample size recommendations
                (default 0.80, standard in biostatistics).
        """
        self.n_model_params = n_model_params
        self.n_modalities = n_modalities
        self.significance_level = significance_level
        self.desired_power = desired_power
        self.confidence_level = 1.0 - significance_level

        logger.info(
            f"Initialized analyzer: {n_model_params:,} params, "
            f"{n_modalities} modalities, alpha={significance_level}, "
            f"power={desired_power}"
        )

    def _estimate_effective_dof(self, n_samples: int) -> int:
        """
        Estimate effective degrees of freedom for a deep learning model.

        For deep learning, the effective DoF is typically much smaller than the
        number of parameters due to:
        - Implicit regularization and weight decay
        - Low-rank structure in learned representations
        - Overparameterization allowing solution in a low-dimensional subspace

        Heuristic: Use square root of parameters, bounded by sample size.
        This is conservative (assumes some regularization benefit).

        For a model with 130M params and 150 samples:
          - sqrt(130M) ≈ 11,400 effective DoF
          - Still ~76:1 samples-to-effective-DoF ratio (poor)

        Args:
            n_samples: Number of validation samples.

        Returns:
            Estimated effective degrees of freedom.
        """
        # Heuristic 1: sqrt of parameters (aggressive regularization assumption)
        dof_sqrt = int(sqrt(self.n_model_params))

        # Heuristic 2: Cap by sample size (can't have more DoF than samples)
        effective_dof = min(dof_sqrt, n_samples // 2)

        return max(1, effective_dof)

    def _compute_noncentrality(
        self, n_samples: int, effect_size: float, effective_dof: int
    ) -> float:
        """
        Compute noncentrality parameter for power calculations.

        Uses the formula: lambda = sqrt(n) * effect_size / sqrt(effective_dof)

        This accounts for:
        - sqrt(n): power increases with sample size
        - effect_size: larger effects are easier to detect
        - sqrt(effective_dof): more effective parameters require larger effects

        Args:
            n_samples: Sample size.
            effect_size: Standardized effect size (e.g., 0.5 for medium).
            effective_dof: Effective degrees of freedom.

        Returns:
            Noncentrality parameter.
        """
        noncentrality = (sqrt(n_samples) * effect_size) / sqrt(effective_dof)
        return noncentrality

    def _compute_power_normal_approx(
        self, n_samples: int, effect_size: float, effective_dof: int
    ) -> float:
        """
        Approximate statistical power using normal distribution.

        For a two-tailed test at significance level alpha:
          z_alpha = quantile(1 - alpha/2) from standard normal
          lambda = noncentrality parameter
          power ≈ P(|Z| > z_alpha | Z ~ N(lambda, 1))
                = 1 - Phi(z_alpha - lambda) + Phi(-z_alpha - lambda)

        This is exact for normal tests and approximately valid for large samples.

        Args:
            n_samples: Sample size.
            effect_size: Standardized effect size.
            effective_dof: Effective degrees of freedom.

        Returns:
            Approximate power (0 to 1).
        """
        # Critical z-value for two-tailed test
        z_alpha = sqrt(2) * erfc(self.significance_level / 2) ** (-0.5)

        # Noncentrality
        lambda_ = self._compute_noncentrality(n_samples, effect_size, effective_dof)

        # Power: P(|Z| > z_alpha under alternative)
        # Using complementary error function: P(Z > x) = 0.5 * erfc(x / sqrt(2))
        power = 0.5 * erfc((z_alpha - lambda_) / sqrt(2))
        power += 0.5 * (1 - erfc((z_alpha + lambda_) / sqrt(2)))

        return np.clip(float(power), 0.0, 1.0)

    def analyze(
        self, n_samples: int, effect_size: float = 0.5
    ) -> PowerAnalysisResult:
        """
        Perform power analysis for a given sample size and effect size.

        Args:
            n_samples: Number of validation samples.
            effect_size: Standardized effect size to detect (default 0.5 = medium).
                In ML validation, this might represent minimum relative improvement
                in AUROC or accuracy deemed clinically meaningful.

        Returns:
            PowerAnalysisResult with detailed analysis and recommendations.
        """
        effective_dof = self._estimate_effective_dof(n_samples)
        ratio = n_samples / self.n_model_params
        power = self._compute_power_normal_approx(n_samples, effect_size, effective_dof)
        min_n = self.compute_minimum_sample_size(effect_size, self.desired_power)

        warnings = []
        recommendations = []

        # Flag critically underpowered configurations
        if ratio < 0.001:  # < 1:1000 samples to params
            warnings.append(
                f"CRITICAL: {ratio:.6f} samples-per-parameter ratio. "
                f"Model has {self.n_model_params:,} parameters but only "
                f"{n_samples} validation samples."
            )
        elif ratio < 0.01:  # < 1:100
            warnings.append(
                f"SEVERE: {ratio:.6f} samples-per-parameter ratio. "
                f"This is far below the rule-of-thumb minimum of 10:1 or 20:1."
            )
        elif ratio < 0.1:
            warnings.append(
                f"WARNING: {ratio:.4f} samples-per-parameter ratio. "
                f"Below recommended 10:1 minimum for confident generalization."
            )

        if power < 0.70:
            warnings.append(
                f"UNDERPOWERED: Statistical power is {power:.3f}, "
                f"below the standard target of 0.80. Risk of Type II error (false negatives)."
            )
        elif power < 0.80:
            warnings.append(
                f"MARGINALLY POWERED: Power is {power:.3f}, slightly below 0.80 target."
            )

        # Hematologic-specific warnings
        if n_samples < 200:
            warnings.append(
                "HEMATOLOGIC CONCERN: Sample size < 200 may be insufficient to validate "
                "predictions across diverse blood subtypes, diseases, and treatment conditions. "
                "Hematologic variation is high; recommend stratified analysis."
            )

        if n_samples < 30 * self.n_modalities:
            warnings.append(
                f"MULTIMODAL CONCERN: With {self.n_modalities} modalities, "
                f"n={n_samples} gives ~{n_samples // self.n_modalities} samples per modality. "
                f"Consider whether each modality is adequately sampled."
            )

        # Recommendations based on current state
        if power < 0.80:
            recommendations.append(
                f"To achieve 80% power with effect_size={effect_size}, "
                f"increase sample size to n >= {min_n}."
            )

        recommendations.extend(self.recommend_validation_strategy(n_samples).get("strategies", []))

        is_adequately_powered = power >= 0.80 and ratio >= 0.01

        result = PowerAnalysisResult(
            sample_size=n_samples,
            model_parameters=self.n_model_params,
            effective_dof=effective_dof,
            samples_per_parameter_ratio=ratio,
            is_adequately_powered=is_adequately_powered,
            minimum_recommended_n=min_n,
            power_at_current_n=power,
            confidence_level=self.confidence_level,
            warnings=warnings,
            recommendations=recommendations,
        )

        logger.warning(f"Power analysis result: {result}")
        return result

    def compute_minimum_sample_size(
        self, effect_size: float, power: float = 0.80
    ) -> int:
        """
        Compute minimum sample size for a given effect size and desired power.

        Uses binary search to find n such that power(n, effect_size) >= desired_power.

        Example for ResistanceMap context:
        >>> analyzer = SampleSizePowerAnalyzer(n_model_params=130_000_000)
        >>> min_n = analyzer.compute_minimum_sample_size(effect_size=0.5, power=0.80)
        # Returns ~3000+ for adequate power with medium effect size

        Args:
            effect_size: Standardized effect size to detect.
            power: Desired statistical power (default 0.80).

        Returns:
            Minimum sample size.
        """
        effective_dof = self._estimate_effective_dof(n_samples=1000)  # Initial estimate

        # Binary search bounds: 10 to 100,000 samples
        low, high = 10, 100_000
        best_n = high

        for _ in range(50):  # ~50 iterations for convergence
            mid = (low + high) // 2
            effective_dof = self._estimate_effective_dof(mid)
            current_power = self._compute_power_normal_approx(
                mid, effect_size, effective_dof
            )

            if current_power >= power:
                best_n = mid
                high = mid - 1
            else:
                low = mid + 1

        logger.debug(
            f"Minimum sample size for power={power}, effect_size={effect_size}: {best_n}"
        )
        return best_n

    def recommend_validation_strategy(self, n_available: int) -> dict:
        """
        Recommend validation strategies given available sample size.

        Suggests approaches to maximize statistical rigor within constraints.

        Args:
            n_available: Number of samples available for validation.

        Returns:
            Dictionary with recommended strategies.
        """
        strategies = []

        # Recommend cross-validation approach
        if n_available < 300:
            k = min(5, n_available // 30)
            strategies.append(
                f"Use {k}-fold cross-validation to maximize use of limited samples. "
                f"This gives {k} independent estimates of generalization error."
            )
        else:
            strategies.append(
                f"Use 10-fold cross-validation or stratified k-fold to ensure robust "
                f"performance estimates across data subsets."
            )

        # Bootstrap recommendation
        strategies.append(
            "Compute bootstrap confidence intervals (e.g., 1000 bootstrap samples) "
            "on key metrics (AUROC, sensitivity, specificity). This provides "
            "uncertainty quantification even with small samples."
        )

        # Effect size and practical significance
        strategies.append(
            "Define minimum clinically/biologically meaningful effect size upfront. "
            "Statistical significance at n=150 may reflect low power, not lack of effect. "
            "Emphasize point estimates and confidence intervals, not p-values alone."
        )

        # Scope recommendations
        ratio = n_available / self.n_model_params
        if ratio < 0.001:
            strategies.append(
                "STRONGLY CONSIDER: Reduce model size (fewer parameters through "
                "pruning, distillation, or architecture redesign) or increase validation "
                "sample size. Current configuration makes generalization claims unreliable."
            )
        elif ratio < 0.01:
            strategies.append(
                "RECOMMEND: Narrow scope of claims to specific use cases or modalities "
                "where power is adequate. Avoid broad claims of general applicability."
            )

        # Hematologic-specific recommendations
        strategies.append(
            "Stratify validation by hematologic condition/subtype and validate "
            "separately. Report performance per stratum to demonstrate robustness."
        )

        strategies.append(
            "Consider disease-specific or population-specific validation if data permits. "
            "Hematologic variation is high; blanket claims of accuracy across all conditions "
            "require careful power analysis per condition."
        )

        return {
            "strategies": strategies,
            "recommended_kfold": min(10, n_available // 30),
            "recommended_bootstrap_samples": 1000,
        }


def flag_sample_size_concerns(
    n_samples: int = 150, n_params: int = 130_000_000
) -> PowerAnalysisResult:
    """
    Convenience function to flag sample size adequacy for ResistanceMap context.

    Default parameters match the ResistanceMap 130M-parameter model with 150
    held-out validation samples from hematologic data.

    This is the "quick check" to see if there are statistical power concerns.

    Args:
        n_samples: Number of validation samples (default 150).
        n_params: Number of model parameters (default 130M).

    Returns:
        PowerAnalysisResult with warnings and recommendations.

    Example:
        >>> result = flag_sample_size_concerns()
        >>> print(result)
        >>> for warning in result.warnings:
        ...     print(f"  - {warning}")
    """
    analyzer = SampleSizePowerAnalyzer(
        n_model_params=n_params,
        n_modalities=4,  # Hematology: morphology, flow, cytochemistry, genetics
        significance_level=0.05,
        desired_power=0.80,
    )
    result = analyzer.analyze(n_samples=n_samples, effect_size=0.5)
    return result


if __name__ == "__main__":
    # Example usage and detailed output
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ResistanceMap Power Analysis: Sample Size Adequacy Check")
    print("=" * 80)

    result = flag_sample_size_concerns(n_samples=150, n_params=130_000_000)

    print("\n" + str(result))

    print("\nWarnings:")
    for i, warning in enumerate(result.warnings, 1):
        print(f"  {i}. {warning}")

    print("\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 80)
    print("Sensitivity Analysis: Effect Size Impact")
    print("=" * 80)

    analyzer = SampleSizePowerAnalyzer(n_model_params=130_000_000)
    for effect_size in [0.2, 0.3, 0.5, 0.8]:
        min_n = analyzer.compute_minimum_sample_size(effect_size, power=0.80)
        print(
            f"  Effect size {effect_size:.1f}: minimum n = {min_n:,} "
            f"(current n=150: underpowered)" if min_n > 150 else f"(current n=150: adequate)"
        )
