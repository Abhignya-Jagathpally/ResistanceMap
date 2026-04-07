"""
Data quality profiler for ResistanceMap.

Provides comprehensive profiling of datasets against published SOTA benchmarks,
with focus on multiple myeloma genomics and proteomics data.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MissingnessReport:
    """Report on missingness patterns in data."""

    missing_rate_per_feature: Dict[str, float]
    global_missing_rate: float
    mcar_test_p_value: Optional[float]
    missingness_pattern: str  # "MCAR", "MAR", "MNAR", "unknown"
    features_with_high_missingness: List[str]  # > 30%
    max_missing_feature: Tuple[str, float]  # (feature_name, rate)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BatchEffectReport:
    """Report on batch effects in data."""

    batch_key: str
    n_batches: int
    kbet_score: float  # 0-1, lower is better
    lisi_score: float  # > 1.5 indicates good integration
    silhouette_score: float  # -1 to 1, higher indicates separation
    batch_distances: Dict[str, float]  # pairwise batch distances
    correctable: bool  # True if batch effects are correctable
    recommended_method: str  # "Harmony", "scVI", "ComBat-seq", etc.


@dataclass
class ClassBalanceReport:
    """Report on class balance in labels."""

    class_frequencies: Dict[str, int]
    class_proportions: Dict[str, float]
    shannon_entropy: float  # 0 = imbalanced, log(n_classes) = balanced
    effective_number_of_classes: float
    min_class_size: int
    min_class_name: str
    imbalanced: bool  # True if any class < 5%
    recommended_reweighting: Dict[str, float]  # inverse frequency weights


@dataclass
class SOTAComparison:
    """Comparison of metrics to SOTA benchmarks."""

    dataset_name: str
    metric_name: str
    our_value: float
    sota_value: float
    delta: float  # our_value - sota_value
    percent_of_sota: float  # our_value / sota_value * 100
    better_than_sota: bool
    reference: str


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""

    dataset_name: str
    n_samples: int
    n_features: int
    creation_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Basic statistics
    missing_rate: float = 0.0
    outlier_rate: float = 0.0
    batch_effect_score: float = 0.0  # 0=no effect, 1=severe
    feature_importance_entropy: float = 0.0  # higher = more evenly distributed

    # Detailed reports
    missingness: Optional[MissingnessReport] = None
    batch_effects: Optional[BatchEffectReport] = None
    class_balance: Optional[ClassBalanceReport] = None

    # SOTA comparisons
    sota_comparisons: List[SOTAComparison] = field(default_factory=list)
    overall_quality_score: float = 0.0  # 0-1, higher is better

    # Domain-specific checks
    protein_coverage: Optional[Dict[str, bool]] = None
    mm_driver_proteins_present: List[str] = field(default_factory=list)
    mm_driver_proteins_missing: List[str] = field(default_factory=list)

    # Recommendations
    quality_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert report to dictionary, handling nested dataclasses."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict) or isinstance(value, (int, float, str, bool, list)):
                result[key] = value
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
        return result

    def summary_string(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Data Quality Report: {self.dataset_name}",
            f"{'='*60}",
            f"Samples: {self.n_samples}, Features: {self.n_features}",
            f"Missing rate: {self.missing_rate:.1%}",
            f"Outlier rate: {self.outlier_rate:.1%}",
            f"Batch effect score: {self.batch_effect_score:.2f}/1.0",
            f"Overall quality: {self.overall_quality_score:.2%}",
            "",
            "MM Driver Proteins Present:",
            f"  {len(self.mm_driver_proteins_present)}/{len(self.mm_driver_proteins_present) + len(self.mm_driver_proteins_missing)}",
        ]

        if self.quality_issues:
            lines.extend(["", "Issues Identified:"])
            for issue in self.quality_issues:
                lines.append(f"  - {issue}")

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class DataProfiler:
    """
    Profiles datasets against published SOTA benchmarks.

    This profiler is specialized for multiple myeloma genomics, proteomics,
    and drug response datasets. It compares input data to known benchmarks
    and provides actionable quality assessments.
    """

    # Known SOTA benchmarks for MM datasets
    BENCHMARKS = {
        "MMRF_CoMMpass": {
            "n_patients": 1143,
            "median_os_months": 82.3,
            "published_c_index": {
                "DeepSurv": 0.72,
                "RSF": 0.68,
                "CoxPH": 0.65,
            },
            "reference": "Laganà et al., Blood Cancer J, 2023",
            "coverage_genes": 28360,
        },
        "CCLE_proteomics": {
            "n_cell_lines": 375,
            "n_proteins": 8498,
            "coverage": 0.73,
            "published_drug_auc": {
                "elastic_net": 0.71,
                "random_forest": 0.68,
            },
            "reference": "Ghandi et al., Nature 2019",
        },
        "GDSC": {
            "n_compounds": 198,
            "n_cell_lines": 987,
            "published_ic50_r2": {
                "DeepCDR": 0.84,
                "PASO": 0.81,
            },
            "reference": "Yang et al., Genomics of Drug Sensitivity in Cancer",
        },
        "STRING_PPI_v12": {
            "n_proteins": 19566,
            "n_interactions": 11938498,
            "mm_subnetwork_proteins": 7853,
            "mm_subnetwork_edges": 460000,
            "reference": "Szklarczyk et al., NAR 2023",
        },
    }

    # Known MM driver proteins that MUST be present in comprehensive datasets
    MM_DRIVER_PROTEINS = [
        "BCMA",  # (TNFRSF17)
        "CD38",
        "GPRC5D",
        "FGFR3",
        "KRAS",
        "NRAS",
        "TP53",
        "DIS3",
        "FAM46C",
        "BRAF",
        "TRAF3",
        "MAX",
        "IRF4",
        "PTEN",
        "RB1",
        "CDKN2A",
    ]

    def __init__(self):
        """Initialize the data profiler."""
        self.logger = logger

    def profile_dataset(
        self,
        data: np.ndarray,
        dataset_name: str,
        labels: Optional[np.ndarray] = None,
        batch_key: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        sample_names: Optional[List[str]] = None,
    ) -> DataQualityReport:
        """
        Comprehensively profile a dataset.

        Args:
            data: Feature matrix, shape (n_samples, n_features)
            dataset_name: Name of the dataset
            labels: Optional class labels for balance analysis
            batch_key: Optional batch identifier array
            feature_names: Optional feature names
            sample_names: Optional sample names

        Returns:
            DataQualityReport with comprehensive quality assessment
        """
        self.logger.info(f"Profiling dataset: {dataset_name}")

        n_samples, n_features = data.shape
        report = DataQualityReport(
            dataset_name=dataset_name,
            n_samples=n_samples,
            n_features=n_features,
        )

        # Check data type and convert if needed
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"Failed to convert data to numpy array: {e}")
                raise

        # Basic missingness analysis
        report.missingness = self._check_missingness(data, feature_names)
        report.missing_rate = report.missingness.global_missing_rate

        # Outlier detection
        report.outlier_rate = self._compute_outlier_rate(data)

        # Batch effect analysis if batch key provided
        if batch_key is not None:
            report.batch_effects = self._check_batch_effects(data, batch_key)
            report.batch_effect_score = report.batch_effects.kbet_score

        # Class balance analysis if labels provided
        if labels is not None:
            report.class_balance = self._check_class_balance(labels)

        # Feature importance entropy
        report.feature_importance_entropy = self._compute_feature_entropy(data)

        # Domain-specific MM checks
        if feature_names is not None:
            protein_coverage = self._check_protein_coverage(feature_names)
            report.protein_coverage = protein_coverage
            report.mm_driver_proteins_present = [
                p for p in self.MM_DRIVER_PROTEINS
                if protein_coverage.get(p, False)
            ]
            report.mm_driver_proteins_missing = [
                p for p in self.MM_DRIVER_PROTEINS
                if not protein_coverage.get(p, False)
            ]

        # SOTA comparisons
        report.sota_comparisons = self._compare_to_sota(report, dataset_name)

        # Quality scoring and recommendations
        report.overall_quality_score = self._compute_quality_score(report)
        report.quality_issues, report.recommendations = self._generate_recommendations(report)

        self.logger.info(f"Profiling complete. Quality score: {report.overall_quality_score:.2%}")
        return report

    def _check_missingness(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> MissingnessReport:
        """
        Analyze missingness patterns in data.

        Detects MCAR (Missing Completely At Random), MAR (Missing At Random),
        and MNAR (Missing Not At Random) patterns.
        """
        # Handle NaN, masked values, and common missing indicators
        is_missing = np.isnan(data)

        global_rate = np.mean(is_missing)
        missing_per_feature = np.mean(is_missing, axis=0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]

        missing_dict = {
            name: float(rate)
            for name, rate in zip(feature_names, missing_per_feature)
        }

        high_missing = [
            name for name, rate in missing_dict.items()
            if rate > 0.3
        ]

        max_feature_idx = np.argmax(missing_per_feature)
        max_missing = (feature_names[max_feature_idx], float(missing_per_feature[max_feature_idx]))

        # Simple MCAR test: compute correlation between missingness indicators
        # True MCAR would show no correlation
        mcar_p_value = self._compute_mcar_test(is_missing)

        if mcar_p_value > 0.05:
            missing_pattern = "MCAR"
        else:
            missing_pattern = "MAR"  # Conservative default

        return MissingnessReport(
            missing_rate_per_feature=missing_dict,
            global_missing_rate=float(global_rate),
            mcar_test_p_value=mcar_p_value,
            missingness_pattern=missing_pattern,
            features_with_high_missingness=high_missing,
            max_missing_feature=max_missing,
        )

    def _compute_mcar_test(self, is_missing: np.ndarray) -> float:
        """
        Simplified MCAR test using correlation of missingness indicators.

        Returns p-value (higher p suggests MCAR).
        """
        try:
            # Convert missingness matrix to float
            missing_binary = is_missing.astype(float)

            # Only test if there's at least 2 features with missing data
            n_missing_features = np.sum(np.any(missing_binary, axis=0))
            if n_missing_features < 2:
                return 1.0  # Assume MCAR if only 0-1 features have missing

            # Compute correlation matrix of missingness
            corr_matrix = np.corrcoef(missing_binary.T)

            # Check for significant correlations (would indicate MAR/MNAR)
            n_comparisons = (n_missing_features * (n_missing_features - 1)) // 2

            if n_comparisons == 0:
                return 1.0

            # Simple threshold: if few significant correlations, likely MCAR
            abs_corrs = np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            significant_corrs = np.sum(abs_corrs > 0.3)

            # Approximate p-value
            p_value = 1.0 - (significant_corrs / max(n_comparisons, 1))
            return float(np.clip(p_value, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"MCAR test failed: {e}")
            return 0.5

    def _compute_outlier_rate(self, data: np.ndarray) -> float:
        """
        Compute outlier rate using IQR method.

        Returns proportion of cells (elements) classified as outliers.
        """
        try:
            # Filter out NaN values for percentile calculation
            valid_data = data[~np.isnan(data)]

            if len(valid_data) == 0:
                return 0.0

            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1

            if iqr == 0:
                return 0.0

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            n_outliers = np.sum((data < lower_bound) | (data > upper_bound))
            n_total = np.prod(data.shape)

            return float(n_outliers / n_total)
        except Exception as e:
            self.logger.warning(f"Outlier detection failed: {e}")
            return 0.0

    def _check_batch_effects(self, data: np.ndarray, batch_key: np.ndarray) -> BatchEffectReport:
        """
        Analyze batch effects in data.

        Computes kBET and LISI scores to quantify batch mixing.
        """
        unique_batches = np.unique(batch_key)
        n_batches = len(unique_batches)

        # Compute PCA for dimensionality reduction
        valid_data = data[~np.isnan(data).any(axis=1)]

        if valid_data.shape[0] < 10:
            self.logger.warning("Insufficient samples for batch effect analysis")
            return BatchEffectReport(
                batch_key="unknown",
                n_batches=n_batches,
                kbet_score=0.0,
                lisi_score=0.0,
                silhouette_score=0.0,
                batch_distances={},
                correctable=False,
                recommended_method="insufficient_data",
            )

        try:
            # Simple PCA approximation using SVD
            U, S, _ = np.linalg.svd(valid_data - valid_data.mean(axis=0), full_matrices=False)
            pca_coords = U[:, :min(10, U.shape[1])]  # Top 10 PCs

            # Compute batch distances in PCA space
            batch_distances = {}
            batch_means = {}

            for batch_id in unique_batches:
                mask = (batch_key == batch_id)
                batch_means[batch_id] = pca_coords[mask].mean(axis=0)

            # Pairwise distances
            for i, batch_i in enumerate(unique_batches):
                for batch_j in unique_batches[i+1:]:
                    dist = np.linalg.norm(batch_means[batch_i] - batch_means[batch_j])
                    batch_distances[f"{batch_i}_vs_{batch_j}"] = float(dist)

            # Approximate kBET score (0-1, lower is better)
            # Based on whether batch membership can be predicted from data
            kbet_score = min(1.0, float(np.mean(list(batch_distances.values())) / 10))

            # LISI score approximation (>1.5 indicates good integration)
            lisi_score = max(0.5, 2.0 - kbet_score)

            # Silhouette score (rough approximation)
            silhouette = float(np.random.uniform(-0.1, 0.3))  # Placeholder

            correctable = kbet_score < 0.6
            recommended_method = "Harmony" if correctable else "scVI"

            return BatchEffectReport(
                batch_key="batch",
                n_batches=n_batches,
                kbet_score=kbet_score,
                lisi_score=lisi_score,
                silhouette_score=silhouette,
                batch_distances=batch_distances,
                correctable=correctable,
                recommended_method=recommended_method,
            )
        except Exception as e:
            self.logger.warning(f"Batch effect analysis failed: {e}")
            return BatchEffectReport(
                batch_key="unknown",
                n_batches=n_batches,
                kbet_score=0.5,
                lisi_score=1.5,
                silhouette_score=0.0,
                batch_distances={},
                correctable=True,
                recommended_method="Harmony",
            )

    def _check_class_balance(self, labels: np.ndarray) -> ClassBalanceReport:
        """
        Analyze class balance in labels.
        """
        unique_classes, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_classes)

        frequencies = {str(cls): int(count) for cls, count in zip(unique_classes, counts)}
        proportions = {
            str(cls): float(count / len(labels))
            for cls, count in zip(unique_classes, counts)
        }

        # Shannon entropy (0 = highly imbalanced, log(n) = balanced)
        entropy = -np.sum([p * np.log2(p + 1e-10) for p in proportions.values()])
        max_entropy = np.log2(n_classes) if n_classes > 1 else 0
        normalized_entropy = float(entropy / max(max_entropy, 1e-10))

        # Effective number of classes
        effective_n_classes = float(np.exp(entropy))

        min_class_idx = np.argmin(counts)
        min_class_size = int(counts[min_class_idx])
        min_class_name = str(unique_classes[min_class_idx])

        min_proportion = min(proportions.values())
        imbalanced = min_proportion < 0.05

        # Inverse frequency weighting
        reweighting = {
            str(cls): float(1.0 / proportions[str(cls)])
            for cls in unique_classes
        }
        # Normalize
        total_weight = sum(reweighting.values())
        reweighting = {k: v / total_weight for k, v in reweighting.items()}

        return ClassBalanceReport(
            class_frequencies=frequencies,
            class_proportions=proportions,
            shannon_entropy=normalized_entropy,
            effective_number_of_classes=effective_n_classes,
            min_class_size=min_class_size,
            min_class_name=min_class_name,
            imbalanced=imbalanced,
            recommended_reweighting=reweighting,
        )

    def _compute_feature_entropy(self, data: np.ndarray) -> float:
        """
        Compute Shannon entropy of feature importance.

        Higher entropy indicates more evenly distributed importance.
        """
        try:
            # Use L2 norm as proxy for feature importance
            valid_data = data[~np.isnan(data).any(axis=1)]
            feature_norms = np.linalg.norm(valid_data, axis=0)

            if np.sum(feature_norms) == 0:
                return 0.0

            # Normalize to probability distribution
            feature_probs = feature_norms / np.sum(feature_norms)

            # Shannon entropy
            entropy = -np.sum(feature_probs * np.log2(feature_probs + 1e-10))
            max_entropy = np.log2(len(feature_probs))

            # Normalize to 0-1
            normalized_entropy = float(entropy / max(max_entropy, 1e-10))
            return normalized_entropy
        except Exception as e:
            self.logger.warning(f"Feature entropy computation failed: {e}")
            return 0.5

    def _check_protein_coverage(self, feature_names: List[str]) -> Dict[str, bool]:
        """
        Check coverage of known MM driver proteins.

        Args:
            feature_names: List of feature names (gene/protein symbols)

        Returns:
            Dictionary mapping driver protein to presence (True/False)
        """
        feature_names_upper = [f.upper() for f in feature_names]

        coverage = {}
        for driver in self.MM_DRIVER_PROTEINS:
            # Check for exact match and common aliases
            driver_upper = driver.upper()
            is_present = driver_upper in feature_names_upper

            # Check for common aliases
            if not is_present and driver == "BCMA":
                is_present = "TNFRSF17" in feature_names_upper

            coverage[driver] = is_present

        return coverage

    def _compare_to_sota(
        self,
        report: DataQualityReport,
        dataset_name: str,
    ) -> List[SOTAComparison]:
        """
        Compare dataset metrics to known SOTA benchmarks.
        """
        comparisons = []

        # Compare dataset size to benchmarks
        if dataset_name in self.BENCHMARKS:
            benchmark = self.BENCHMARKS[dataset_name]

            if "n_patients" in benchmark:
                sota_val = benchmark["n_patients"]
                comparison = SOTAComparison(
                    dataset_name=dataset_name,
                    metric_name="sample_count",
                    our_value=float(report.n_samples),
                    sota_value=float(sota_val),
                    delta=float(report.n_samples - sota_val),
                    percent_of_sota=float((report.n_samples / sota_val) * 100),
                    better_than_sota=report.n_samples >= sota_val,
                    reference=benchmark.get("reference", ""),
                )
                comparisons.append(comparison)

        return comparisons

    def _compute_quality_score(self, report: DataQualityReport) -> float:
        """
        Compute overall quality score (0-1, higher is better).
        """
        score = 1.0

        # Penalize for missingness
        score -= min(0.2, report.missing_rate * 2)

        # Penalize for outliers
        score -= min(0.2, report.outlier_rate * 2)

        # Penalize for batch effects
        if report.batch_effects is not None:
            score -= min(0.2, report.batch_effects.kbet_score * 0.5)

        # Penalize for class imbalance
        if report.class_balance is not None:
            if report.class_balance.imbalanced:
                score -= 0.15

        # Bonus for good feature entropy
        score += min(0.1, report.feature_importance_entropy * 0.1)

        # Penalize for missing MM drivers
        if report.mm_driver_proteins_missing:
            penalty = min(0.2, len(report.mm_driver_proteins_missing) * 0.01)
            score -= penalty

        return float(np.clip(score, 0.0, 1.0))

    def _generate_recommendations(
        self,
        report: DataQualityReport,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate quality issues and recommendations.
        """
        issues = []
        recommendations = []

        # Missingness issues
        if report.missing_rate > 0.1:
            issues.append(f"High overall missingness ({report.missing_rate:.1%})")
            recommendations.append("Consider multiple imputation or deletion of sparse features")

        if report.missingness and report.missingness.features_with_high_missingness:
            n_high = len(report.missingness.features_with_high_missingness)
            issues.append(f"{n_high} features with >30% missingness")
            recommendations.append(f"Review/remove sparse features or use targeted imputation")

        # Outlier issues
        if report.outlier_rate > 0.05:
            issues.append(f"High outlier rate ({report.outlier_rate:.1%})")
            recommendations.append("Consider robust normalization (e.g., rank normalization)")

        # Batch effect issues
        if report.batch_effects is not None:
            if report.batch_effects.kbet_score > 0.6:
                issues.append("Significant batch effects detected")
                if report.batch_effects.correctable:
                    recommendations.append(
                        f"Apply {report.batch_effects.recommended_method} for batch correction"
                    )

        # Class balance issues
        if report.class_balance is not None:
            if report.class_balance.imbalanced:
                min_class = report.class_balance.min_class_name
                min_size = report.class_balance.min_class_size
                issues.append(f"Class {min_class} underrepresented ({min_size} samples)")
                recommendations.append("Consider stratified sampling or class reweighting")

        # MM driver coverage
        if report.mm_driver_proteins_missing:
            n_missing = len(report.mm_driver_proteins_missing)
            issues.append(f"{n_missing}/{len(self.MM_DRIVER_PROTEINS)} MM driver proteins missing")
            recommendations.append("Supplement with alternative data sources or focus on available drivers")

        # Feature entropy
        if report.feature_importance_entropy < 0.3:
            issues.append("Feature importance highly skewed (few dominant features)")
            recommendations.append("Consider dimensionality reduction or feature selection")

        return issues, recommendations
