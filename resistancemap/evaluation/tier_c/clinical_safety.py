"""Tier C: Clinical Translation Safety agent.

Assesses whether ResistanceMap outputs can support actionable therapy
decisions in multiple myeloma WITHOUT overclaiming. The agent draws an
explicit line between **analytical validity** (does the model do what it
claims on benchmark data) and **clinical validity** (does it improve
patient outcomes), and refuses to let analytical-validity claims be
laundered into clinical-validity claims.

Framing follows the CLIA / FDA evidence ladder for laboratory-developed
tests:

  analytical validity  ->  clinical validity  ->  clinical utility

Each drug in ``config.data.target_drugs`` is mapped to a safety-critical
consideration grounded in its known toxicity profile and the consequences
of a wrong selection signal.
"""

from __future__ import annotations

import logging
from typing import Any

from resistancemap.evaluation.base import EvalAgent, EvalFinding, Verdict

logger = logging.getLogger(__name__)


# Drug -> safety-critical consideration. These are public-knowledge cautions
# every MM clinician would already cite; the agent's job is to make sure the
# model output never silently ignores them.
DRUG_SAFETY_NOTES: dict[str, str] = {
    "Bortezomib": (
        "peripheral neuropathy is dose-cumulative; a false 'sensitive' call "
        "may push intensification that the patient cannot tolerate."
    ),
    "Lenalidomide": (
        "thromboembolism risk and second-primary malignancy in long-term "
        "maintenance; a false 'resistant' call risks premature discontinuation "
        "of an otherwise-effective backbone."
    ),
    "Dexamethasone": (
        "hyperglycemia, infection, mood/psychiatric effects; the model must "
        "not be used to titrate steroid pulses without endocrine review."
    ),
    "Carfilzomib": (
        "cardiopulmonary toxicity (CHF, hypertension); a false 'sensitive' "
        "call in a patient with cardiac comorbidity is potentially fatal."
    ),
    "Pomalidomide": (
        "hematologic toxicity and rare hepatotoxicity; the model should not "
        "drive switch decisions without recent CBC and LFT review."
    ),
    "Daratumumab": (
        "infusion reactions and interference with blood-bank typing; the "
        "model output must not be used as a standalone switch trigger and "
        "should require transfusion-medicine awareness."
    ),
}


# Failure modes the agent enumerates *unconditionally*. These are not bugs
# to find; they are pre-registered modes of harm the model design must
# defend against.
THERAPY_FAILURE_MODES: list[str] = [
    "False negative for aggressive disease -> missed intensification window.",
    "Overconfident calls on rare phenotypes -> unwarranted therapy switch.",
    "Calibration drift between training cohort (CCLE/MMRF) and target "
    "patient population -> systematic bias against under-represented "
    "racial/ethnic groups.",
    "Subgroup-blind aggregate metrics hide harm to a small but clinically "
    "important subgroup (e.g. del(17p), t(14;16), high LDH).",
    "Time-axis miscalibration (ODE arbitrary units) -> wrong urgency.",
    "Reliance on a single biomarker the model has memorized rather than a "
    "biological mechanism -> brittle to assay changes.",
    "Lack of an abstention pathway -> model produces a recommendation even "
    "when out-of-distribution detection should trigger 'defer to clinician'.",
]


class ClinicalTranslationSafetyAgent(EvalAgent):
    """Tier C agent: actionability + safety review.

    Expected ``intake`` keys (all optional):

    - ``per_subgroup_metrics``: dict of subgroup -> {metric: value}.
    - ``ood_detector_present``: bool.
    - ``abstention_rate``: float (per-prediction abstention fraction).
    - ``analytical_validity_metrics``: dict (in-distribution test metrics).
    - ``clinical_validity_evidence``: dict (prospective / external cohort
      results, if any).
    - ``clinical_utility_evidence``: dict (decision-impact studies, if any).

    The agent never invents missing categories; absent evidence is reported
    as absent and downgrades the verdict accordingly.
    """

    tier = "C"

    def __init__(self) -> None:
        super().__init__(
            name="clinical_translation_safety_agent",
            dependencies=[],
        )

    async def assess(
        self, intake: dict[str, Any], config: Any
    ) -> EvalFinding:  # type: ignore[override]
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = list(THERAPY_FAILURE_MODES)
        required_changes: list[str] = []

        target_drugs = list(config.data.target_drugs)
        # Map every configured drug to its safety note; warn if a drug has no
        # entry (forces an explicit safety review when new drugs are added).
        drug_safety_map: dict[str, str] = {}
        for drug in target_drugs:
            if drug in DRUG_SAFETY_NOTES:
                drug_safety_map[drug] = DRUG_SAFETY_NOTES[drug]
            else:
                drug_safety_map[drug] = (
                    "NO PRE-REGISTERED SAFETY NOTE; add one before this drug "
                    "can drive any therapy recommendation."
                )
                required_changes.append(
                    f"add a pre-registered safety note for '{drug}' to "
                    "ClinicalTranslationSafetyAgent.DRUG_SAFETY_NOTES"
                )
        criteria_results["drug_safety_map"] = drug_safety_map
        evidence["drug_safety_map"] = drug_safety_map

        # ------------------------------------------------------------------
        # CLIA-adjacent ladder: analytical / clinical validity / utility.
        # ------------------------------------------------------------------
        analytical = intake.get("analytical_validity_metrics") or {}
        clinical_v = intake.get("clinical_validity_evidence") or {}
        clinical_u = intake.get("clinical_utility_evidence") or {}

        ladder = {
            "analytical_validity": {
                "present": bool(analytical),
                "evidence": analytical,
                "definition": (
                    "does the model accurately and reproducibly measure what "
                    "it claims to measure on the target assay/data type"
                ),
            },
            "clinical_validity": {
                "present": bool(clinical_v),
                "evidence": clinical_v,
                "definition": (
                    "does the model output correlate with the clinical "
                    "outcome (response, PFS, OS) in an INDEPENDENT cohort"
                ),
            },
            "clinical_utility": {
                "present": bool(clinical_u),
                "evidence": clinical_u,
                "definition": (
                    "does using the model improve patient outcomes vs. "
                    "current standard of care, demonstrated prospectively"
                ),
            },
        }
        criteria_results["clia_ladder"] = ladder
        evidence["clia_ladder"] = ladder

        if not analytical:
            failure_modes.append(
                "No analytical validity metrics provided; the model has not "
                "even cleared the first rung of the CLIA-adjacent ladder."
            )
            required_changes.append(
                "report analytical validity metrics on a held-out test set "
                "(intake['analytical_validity_metrics'])"
            )
        if not clinical_v:
            required_changes.append(
                "demonstrate clinical validity in an INDEPENDENT cohort "
                "(intake['clinical_validity_evidence']) before any clinical "
                "claim is made"
            )
        if not clinical_u:
            required_changes.append(
                "demonstrate clinical utility in a prospective decision-"
                "impact study (intake['clinical_utility_evidence']) before "
                "any clinical-utility claim is made"
            )

        # ------------------------------------------------------------------
        # Subgroup audit.
        # ------------------------------------------------------------------
        subgroup_metrics = intake.get("per_subgroup_metrics") or {}
        criteria_results["subgroup_audit_present"] = bool(subgroup_metrics)
        if not subgroup_metrics:
            required_changes.append(
                "report per-subgroup metrics for at least: cytogenetic risk "
                "(standard / high), age stratum, sex, race/ethnicity, prior "
                "lines of therapy, ISS stage"
            )
        else:
            evidence["per_subgroup_metrics"] = subgroup_metrics
            # Flag any subgroup whose primary metric is materially worse
            # than the cohort median.
            try:
                values_by_metric: dict[str, list[float]] = {}
                for sub, metrics in subgroup_metrics.items():
                    if not isinstance(metrics, dict):
                        continue
                    for k, v in metrics.items():
                        try:
                            values_by_metric.setdefault(k, []).append(float(v))
                        except (TypeError, ValueError):
                            continue
                worst_offenders: list[str] = []
                for metric_name, vals in values_by_metric.items():
                    if len(vals) < 2:
                        continue
                    median = float(sorted(vals)[len(vals) // 2])
                    for sub, metrics in subgroup_metrics.items():
                        v = metrics.get(metric_name) if isinstance(metrics, dict) else None
                        try:
                            v = float(v)
                        except (TypeError, ValueError):
                            continue
                        if median - v > 0.10:
                            worst_offenders.append(
                                f"{sub}: {metric_name}={v:.3f} (median {median:.3f})"
                            )
                if worst_offenders:
                    failure_modes.append(
                        "Subgroups underperform the cohort median by >0.10: "
                        + "; ".join(worst_offenders)
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Subgroup audit failed: %s", exc)

        # ------------------------------------------------------------------
        # Out-of-distribution / abstention pathway.
        # ------------------------------------------------------------------
        ood_present = bool(intake.get("ood_detector_present", False))
        abstention_rate = intake.get("abstention_rate")
        criteria_results["ood_detector_present"] = ood_present
        criteria_results["abstention_rate"] = abstention_rate
        if not ood_present:
            failure_modes.append(
                "No OOD detector present: the model cannot recognize when a "
                "patient lies outside its training distribution and will "
                "still emit a confident recommendation."
            )
            required_changes.append(
                "add an OOD detector and route OOD patients to a 'defer to "
                "clinician' abstention path"
            )
        if abstention_rate is None:
            required_changes.append(
                "report abstention_rate so reviewers can see how often the "
                "model defers vs. acts"
            )

        criteria_results["failure_modes_enumerated"] = len(failure_modes)
        criteria_results["pre_registered_failure_modes"] = THERAPY_FAILURE_MODES

        # ------------------------------------------------------------------
        # Verdict.
        # ------------------------------------------------------------------
        if not analytical:
            verdict = Verdict.FAIL
        elif not clinical_v or not ood_present:
            verdict = Verdict.CONDITIONAL
        else:
            verdict = Verdict.CONDITIONAL  # never PASS without utility data

        # Score: fraction of CLIA rungs cleared (max 1.0 only with utility).
        rungs_cleared = sum(
            [bool(analytical), bool(clinical_v), bool(clinical_u)]
        )
        score = rungs_cleared / 3.0

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )
