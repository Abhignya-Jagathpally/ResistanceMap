"""Epistemic boundaries and formal claim structure for ResistanceMap.

Defines a hierarchy of evidence levels, formal claim structures, and validation
requirements to ensure that all published statements about model capabilities
are accurately categorized and properly caveated.

This module enforces a contract: claims about ResistanceMap must be explicitly
registered, evidence-graded, and validated against publication requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = [
    "EvidenceLevel",
    "EpistemicClaim",
    "CLAIM_REGISTRY",
    "ClaimValidator",
    "publication_checklist",
]


class EvidenceLevel(Enum):
    """Hierarchy of evidence for model claims.

    Members:
        OBSERVED: Direct empirical measurement in the training/test data.
            No extrapolation or interpretation required.
        INTERPOLATED: Interpolation within the data envelope (e.g., predictions
            for patients similar to training cohort). Requires calibration.
        EXTRAPOLATED: Extension beyond the training population (e.g., different
            cancer type, treatment context, cell-line to patient). Requires
            explicit disclaimers and reduced confidence.
        HYPOTHESIZED: Mechanistic claim not directly measured. Requires supporting
            evidence from literature and validation against orthogonal measurements.
        UNSUPPORTED: No credible evidence; cannot be published without
            independent validation.
    """

    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    HYPOTHESIZED = "hypothesized"
    UNSUPPORTED = "unsupported"


@dataclass
class EpistemicClaim:
    """Formal structure for a claim about ResistanceMap capabilities.

    Every claim must include:
    - the statement itself,
    - its evidence level,
    - explicit conditions under which it applies,
    - mandatory disclaimers,
    - validation requirements before publication.

    Attributes:
        statement: Concise claim (e.g., "Model predicts intrinsic vs acquired resistance").
        evidence_level: EvidenceLevel enum value indicating the strength of support.
        conditions: List of conditions under which the claim holds
            (e.g., "training cohort only", "screened compounds only").
        disclaimers: Mandatory caveats to appear in publication
            (e.g., "not validated on patient data").
        validation_requirements: List of checks that must pass before publication
            (e.g., "subgroup analysis by age", "external validation on held-out patients").
        references: Optional list of supporting citations or URLs.
    """

    statement: str
    evidence_level: EvidenceLevel
    conditions: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)
    validation_requirements: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate evidence level."""
        if isinstance(self.evidence_level, str):
            self.evidence_level = EvidenceLevel(self.evidence_level)


# Global registry of all claims ResistanceMap makes
CLAIM_REGISTRY: dict[str, EpistemicClaim] = {
    "state_classification": EpistemicClaim(
        statement=(
            "Model classifies cells into distinct resistance states "
            "(sensitive, intrinsic resistance, acquired resistance) "
            "based on multi-omic features."
        ),
        evidence_level=EvidenceLevel.OBSERVED,
        conditions=[
            "Applies to cell-line training cohort only",
            "Input features must include all modalities used during training",
            "Requires proteomics, epigenomics, PPI, and drug sensitivity data "
            "for complete coverage",
        ],
        disclaimers=[
            "Model is trained on cell-line data; patient applicability requires "
            "independent validation.",
            "State boundaries are defined computationally; biological interpretation "
            "must be validated experimentally.",
            "Out-of-distribution samples (e.g., rare mutations) may be misclassified.",
        ],
        validation_requirements=[
            "Confusion matrix and per-class metrics (precision, recall, F1) "
            "on held-out test set",
            "Analysis of failure modes: which state transitions are confused?",
            "Sensitivity analysis: which features most influence classification?",
        ],
        references=[
            "See calibration metrics and ablation studies in evaluation/metrics/",
        ],
    ),
    "temporal_trajectory": EpistemicClaim(
        statement=(
            "Model predicts temporal dynamics of resistance acquisition "
            "using an ODE/SDE system fitted to time-series measurements."
        ),
        evidence_level=EvidenceLevel.INTERPOLATED,
        conditions=[
            "Requires longitudinal measurements (at least 3+ time points) "
            "for reliable calibration",
            "Time span must overlap with training data range; extrapolation "
            "beyond training times is unreliable",
            "Assumes constant kinetic rates (no time-varying parameters)",
        ],
        disclaimers=[
            "ODE/SDE model is a simplification of complex biological dynamics; "
            "mechanistic interpretation is speculative.",
            "Model is fitted to aggregate data; individual trajectories are "
            "expected to deviate significantly.",
            "No experimental validation of predicted rates; calibration against "
            "held-out time points only.",
            "Uncertainty quantification reflects epistemic uncertainty in fitted "
            "parameters, not aleatoric noise in biology.",
        ],
        validation_requirements=[
            "Cross-validation on held-out time windows",
            "Comparison against null models (constant rate, Brownian motion)",
            "Sensitivity to initial conditions and parameter perturbations",
            "Independent measurement of kinetic parameters in controlled system "
            "(e.g., competition assays)",
        ],
        references=[
            "See dynamics/ module; SDE calibration uses bridge sampling",
        ],
    ),
    "pathway_attribution": EpistemicClaim(
        statement=(
            "Model identifies pathway activities and regulatory mechanisms "
            "associated with each resistance state."
        ),
        evidence_level=EvidenceLevel.INTERPOLATED,
        conditions=[
            "Identifies correlations between features and states, not causation",
            "PPI module only: requires annotated pathway database and "
            "interaction network",
        ],
        disclaimers=[
            "Pathway attribution is correlational; not causal without "
            "independent perturbation experiments (CRISPR, RNAi).",
            "Pathway scores are derived from linear/additive models; "
            "epistasis and non-linearities are not captured.",
            "Network representation is incomplete; missing edges in PPI "
            "may hide alternative pathways.",
        ],
        validation_requirements=[
            "CRISPR or RNAi knock-in/knockdown of predicted effectors "
            "in candidate pathways",
            "Comparison against permuted pathway annotations (null model)",
            "Cross-validation of pathway signatures across independent cohorts",
        ],
        references=[
            "See evaluation/tier_b/pathway_ppi.py for annotation details",
        ],
    ),
    "drug_specific_predictions": EpistemicClaim(
        statement=(
            "Model predicts sensitivity or resistance to specific drugs "
            "based on cell state and molecular features."
        ),
        evidence_level=EvidenceLevel.EXTRAPOLATED,
        conditions=[
            "Applies only to compounds present in GDSC or CTRPv2 training screens",
            "Requires drug fingerprints (Morgan fingerprints or SMILES)",
            "Model cannot extrapolate to untested compounds or structural scaffolds",
        ],
        disclaimers=[
            "Predictions are based on historical cell-line screens (in vitro); "
            "clinical efficacy is unknown and may differ substantially.",
            "Limited to screened compound set; novel compounds or combinations "
            "are not supported.",
            "Drug response varies with culture conditions, passage number, and "
            "media composition; model does not account for these variations.",
            "No patient validation; applicability to patient-derived samples "
            "is speculative.",
        ],
        validation_requirements=[
            "Held-out test set evaluation on screened compounds",
            "Sensitivity analysis: uncertainty increase with structural distance "
            "from training compounds?",
            "Validation on patient-derived samples (if available)",
            "Comparison against baseline models (e.g., drug feature similarity alone)",
        ],
        references=[
            "Training data: GDSC (https://www.cancerrxgene.org/) and "
            "CTRPv2 (https://portals.broadinstitute.org/ctrp.v2.1/)",
        ],
    ),
    "uncertainty_quantification": EpistemicClaim(
        statement=(
            "Model provides epistemic uncertainty (from model parameters) "
            "and aleatoric uncertainty (from stochastic dynamics)."
        ),
        evidence_level=EvidenceLevel.HYPOTHESIZED,
        conditions=[
            "Epistemic uncertainty: derived from Dirichlet posterior over state "
            "mixture parameters",
            "Aleatoric uncertainty: from SDE noise term; reflects inherent "
            "biological variability",
        ],
        disclaimers=[
            "Uncertainty estimates are model-based; they do not account for "
            "unknown unknowns (unmodeled sources of variation).",
            "Calibration of uncertainty is limited to training data envelope; "
            "extrapolation calibration is untested.",
            "SDE noise term is estimated from data, not from first principles; "
            "magnitude may not reflect true biological noise.",
        ],
        validation_requirements=[
            "Calibration of predictive intervals: do 95% intervals contain "
            "95% of held-out observations?",
            "Comparison against ensemble and bootstrap uncertainty estimates",
            "Sensitivity: how do uncertainty estimates change with hyperparameter "
            "perturbation?",
        ],
        references=[
            "Lakshminarayanan et al. (2017) NIPS: Simple and Scalable Predictive "
            "Uncertainty Estimation using Deep Ensembles",
        ],
    ),
    "competing_risks": EpistemicClaim(
        statement=(
            "Model decomposes resistance development into intrinsic (at baseline) "
            "and acquired (progressive) mechanisms via competing-risks mixture model."
        ),
        evidence_level=EvidenceLevel.HYPOTHESIZED,
        conditions=[
            "Applies to longitudinal cohorts with multiple time points",
            "Requires censoring information (unobserved progression times)",
        ],
        disclaimers=[
            "Competing-risks model assumes two distinct pathways; intermediate "
            "or hybrid mechanisms are not explicitly modeled.",
            "Cause-specific hazards are estimated from observed event types; "
            "unobserved mechanisms cannot be distinguished.",
            "Model is fit to aggregate data; individual patient trajectories "
            "may not align with mixture components.",
            "No clinical validation; competing-risks decomposition must be "
            "validated against CRISPR/functional studies.",
        ],
        validation_requirements=[
            "Fine-Gray regression: test for proportional cause-specific hazards",
            "Sensitivity to unmeasured confounding (e.g., via E-value)",
            "Validation in independent cohort with cause-specific event labels",
            "Experimental validation of predicted intrinsic vs acquired mechanisms",
        ],
        references=[
            "Tsiatis (1975) PNAS: A Nonidentifiability Aspect of the Problem "
            "of Competing Risks; Gaynor et al. (1993) Statistics in Medicine",
        ],
    ),
}


class ClaimValidator:
    """Validator that checks claims against conditions and generates disclaimers.

    Used to ensure that any publication or model deployment includes all
    required disclaimers and satisfies validation requirements.
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        pass

    def validate_claim(
        self,
        claim_key: str,
        context: dict | None = None,
    ) -> dict:
        """Validate a registered claim in a given context.

        Args:
            claim_key: Key in CLAIM_REGISTRY (e.g., "state_classification").
            context: Optional dict with keys describing the usage context
                (e.g., {"data_type": "patient", "time_points": 5}).

        Returns:
            Dictionary with keys:

            - ``"claim_key"`` (str): The registered claim.
            - ``"statement"`` (str): Full claim statement.
            - ``"evidence_level"`` (str): Evidence level.
            - ``"applicable"`` (bool): True if all conditions can be met.
            - ``"conditions"`` (list): All applicable conditions.
            - ``"mandatory_disclaimers"`` (list): All disclaimers that must appear.
            - ``"validation_checklist"`` (list): Pre-publication validation steps.
            - ``"warnings"`` (list): Context-specific warnings.
        """
        if claim_key not in CLAIM_REGISTRY:
            return {
                "claim_key": claim_key,
                "applicable": False,
                "warnings": [f"Claim {claim_key!r} is not registered in CLAIM_REGISTRY"],
            }

        claim = CLAIM_REGISTRY[claim_key]
        warnings = []

        # Check context-specific warnings
        if context is None:
            context = {}

        if context.get("data_type") == "patient" and claim.evidence_level in (
            EvidenceLevel.OBSERVED,
            EvidenceLevel.INTERPOLATED,
        ):
            if claim_key in ("state_classification", "pathway_attribution"):
                warnings.append(
                    "Claim was developed on cell-line data; patient validation "
                    "is required before confident claims."
                )

        if (
            context.get("time_points", 1) < 3
            and claim_key == "temporal_trajectory"
        ):
            warnings.append(
                "Temporal trajectory claims require at least 3 time points; "
                "current data may be insufficient."
            )

        applicable = len(warnings) == 0 or claim.evidence_level in (
            EvidenceLevel.EXTRAPOLATED,
            EvidenceLevel.HYPOTHESIZED,
        )

        return {
            "claim_key": claim_key,
            "statement": claim.statement,
            "evidence_level": claim.evidence_level.value,
            "applicable": applicable,
            "conditions": claim.conditions,
            "mandatory_disclaimers": claim.disclaimers,
            "validation_checklist": claim.validation_requirements,
            "warnings": warnings,
        }

    def generate_disclaimers(
        self,
        claim_keys: list[str] | None = None,
    ) -> list[str]:
        """Generate all mandatory disclaimers for a set of claims.

        Args:
            claim_keys: List of claim keys. If None, uses all registered claims.

        Returns:
            List of unique disclaimers (deduplicated).
        """
        if claim_keys is None:
            claim_keys = list(CLAIM_REGISTRY.keys())

        disclaimers = []
        for key in claim_keys:
            if key in CLAIM_REGISTRY:
                disclaimers.extend(CLAIM_REGISTRY[key].disclaimers)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for d in disclaimers:
            if d not in seen:
                seen.add(d)
                unique.append(d)

        return unique


def publication_checklist() -> list[str]:
    """Return mandatory pre-publication checks for ResistanceMap.

    This list enforces minimum standards for any peer-reviewed publication
    or clinical deployment claiming ResistanceMap capabilities.

    Returns:
        List of checks, each described as an actionable requirement.
    """
    return [
        "Sample size and statistical power: Verify n_eff >= 100 after "
        "accounting for clustering (ICC). Run power_analysis module.",
        "Subgroup analysis: Report metrics separately for age, cancer type, "
        "treatment context. Use claim_adequacy_checklist.",
        "Calibration assessment: Compute ECE, MCE, and reliability diagrams "
        "on held-out test set. Apply post_hoc_calibration if needed.",
        "Feature importance and ablation: Report which features most influence "
        "predictions. Remove redundant features.",
        "External validation: Evaluate on an independent cohort not used in "
        "training or hyperparameter tuning.",
        "Competing risks validation: If claiming intrinsic vs acquired "
        "decomposition, validate against functional experiments or independent "
        "temporal cohort.",
        "Uncertainty calibration: Verify that 95% predictive intervals contain "
        "~95% of held-out observations. Check via CalibrationDiagnostics.",
        "Reproducibility: Provide random seeds, hyperparameter values, and "
        "training/test split details. Code must be reproducible.",
        "Claim registration: Every claim in the paper must be registered in "
        "CLAIM_REGISTRY with evidence level and validation requirements.",
        "Disclaimer inclusion: All mandatory disclaimers from EpistemicClaim "
        "must appear in the methods or limitations section.",
        "Pathway causation: If claiming mechanistic pathways, include disclaimer "
        "that these are correlational without CRISPR/RNAi validation.",
        "Drug applicability: If making drug-specific predictions, restrict to "
        "compounds in GDSC/CTRPv2 and include disclaimer about in vitro limitations.",
        "Temporal extrapolation: If predicting beyond training time range, "
        "quantify uncertainty increase and include extrapolation warning.",
        "Data leakage: Run leakage detection (resistancemap/evaluation/leakage.py) "
        "to confirm patient/cell-line overlap is acceptable.",
        "No fabrication: No synthetic or augmented data in training without explicit "
        "notation. All claims must be backed by real observations.",
    ]
