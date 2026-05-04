"""Tier D: Principal Integrator (chair) agent.

Aggregates findings from Tier A (data/identifiability), Tier B (model
mechanism), and Tier C (forecasting / baselines / clinical safety) into a
single ``ChairReport``. Resolves disagreements via ``debate_resolution()``,
which only awards weight to:

  1. Peer-reviewed-style evidence (citations, public benchmarks).
  2. Pre-registered experiments (specs declared *before* the run).

Anonymous opinions, post-hoc rationalizations, and ungrounded claims are
discarded.

Hard rule (documented in ``__init__``):

    If any Tier A finding has verdict==FAIL, the overall TRL grade is
    bounded above at 2.

The reasoning: Tier A audits data integrity, identifiability, and the
ability to even *learn* the system. NASA's Technology Readiness Level
ladder treats TRL 1 as "basic principles observed" and TRL 2 as "technology
concept formulated". A model whose data foundations are broken cannot have
formulated a *valid* technology concept; it can at best be "principles
observed". Bounding above at TRL 2 is generous -- a strict reading would
cap at TRL 1 -- but we leave room for the case where the data failure is
isolated to one cohort while the conceptual framework is still defensible.
TRL 3 ("experimental proof of concept") and above all require working
analytical results that a Tier A FAIL invalidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from resistancemap.evaluation.base import EvalAgent, EvalFinding, Verdict

logger = logging.getLogger(__name__)


# Severity ordering for required-changes prioritization. Higher = more urgent.
SEVERITY_ORDER: dict[str, int] = {
    "block_clinical": 100,
    "block_publication": 90,
    "block_merge": 80,
    "high": 70,
    "medium": 50,
    "low": 30,
    "info": 10,
}


@dataclass
class PrioritizedChange:
    """One actionable change item, ordered by severity."""

    description: str
    severity: str
    severity_score: int
    source_agent: str
    source_tier: str


@dataclass
class ChairReport:
    """Final integrated assessment.

    Attributes:
        overall_grade: TRL on the 1-9 scale.
        blocking_issues: Issues that prevent any further progress.
        required_changes_prioritized: Sorted list (highest severity first).
        re_run_conditions: Conditions under which the chair will re-evaluate.
        rationale: Short prose explanation of the grade.
        per_tier_summary: Aggregated counts per tier.
        debate_log: Conflicts encountered and how they were resolved.
    """

    overall_grade: int
    blocking_issues: list[str] = field(default_factory=list)
    required_changes_prioritized: list[PrioritizedChange] = field(default_factory=list)
    re_run_conditions: list[str] = field(default_factory=list)
    rationale: str = ""
    per_tier_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    debate_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_grade": self.overall_grade,
            "blocking_issues": self.blocking_issues,
            "required_changes_prioritized": [
                {
                    "description": c.description,
                    "severity": c.severity,
                    "severity_score": c.severity_score,
                    "source_agent": c.source_agent,
                    "source_tier": c.source_tier,
                }
                for c in self.required_changes_prioritized
            ],
            "re_run_conditions": self.re_run_conditions,
            "rationale": self.rationale,
            "per_tier_summary": self.per_tier_summary,
            "debate_log": self.debate_log,
        }


class PrincipalIntegratorAgent(EvalAgent):
    """Tier D: chair that integrates all lower-tier findings.

    Expected ``intake`` keys:

    - ``findings``: list[EvalFinding] from Tier A/B/C agents.
    """

    tier = "D"

    def __init__(self) -> None:
        super().__init__(name="principal_integrator_agent", dependencies=[])

    async def assess(
        self, intake: dict[str, Any], config: Any
    ) -> EvalFinding:  # type: ignore[override]
        findings: list[EvalFinding] = list(intake.get("findings") or [])
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        if not findings:
            logger.warning(
                "PrincipalIntegratorAgent: no findings provided; cannot grade."
            )
            report = ChairReport(
                overall_grade=1,
                blocking_issues=["No Tier A/B/C findings supplied to chair."],
                rationale=(
                    "TRL 1 is the floor; with no evidence at all the system "
                    "is at 'basic principles observed' by default."
                ),
                re_run_conditions=[
                    "supply intake['findings'] with at least one Tier A finding"
                ],
            )
            return self._wrap(report, criteria_results, evidence)

        # ------------------------------------------------------------------
        # Per-tier verdict tally.
        # ------------------------------------------------------------------
        per_tier: dict[str, dict[str, int]] = {}
        for f in findings:
            tier_summary = per_tier.setdefault(
                f.tier, {"PASS": 0, "CONDITIONAL_PASS": 0, "FAIL": 0, "SKIP": 0}
            )
            verdict_key = (
                f.verdict.value if hasattr(f.verdict, "value") else str(f.verdict)
            )
            verdict_key = verdict_key.upper()
            if verdict_key not in tier_summary:
                tier_summary[verdict_key] = 0
            tier_summary[verdict_key] += 1
        criteria_results["per_tier_summary"] = per_tier
        evidence["per_tier_summary"] = per_tier

        # ------------------------------------------------------------------
        # Debate / conflict resolution.
        # ------------------------------------------------------------------
        debate_log = self.debate_resolution(findings)
        evidence["debate_log"] = debate_log
        criteria_results["debate_resolution_count"] = len(debate_log)

        # ------------------------------------------------------------------
        # TRL grading. Start optimistic and walk down.
        # ------------------------------------------------------------------
        tier_a_fail = any(
            f.tier == "A" and _is_fail(f.verdict) for f in findings
        )
        tier_b_fail = any(
            f.tier == "B" and _is_fail(f.verdict) for f in findings
        )
        tier_c_fail = any(
            f.tier == "C" and _is_fail(f.verdict) for f in findings
        )
        any_clinical_validity = any(
            f.tier == "C"
            and "clinical_validity" in (f.criteria_results or {}).get(
                "clia_ladder", {}
            )
            and (f.criteria_results or {})
            .get("clia_ladder", {})
            .get("clinical_validity", {})
            .get("present", False)
            for f in findings
        )
        any_clinical_utility = any(
            f.tier == "C"
            and "clinical_utility" in (f.criteria_results or {}).get(
                "clia_ladder", {}
            )
            and (f.criteria_results or {})
            .get("clia_ladder", {})
            .get("clinical_utility", {})
            .get("present", False)
            for f in findings
        )

        # Optimistic ceiling on TRL.
        grade = 9
        rationale_parts: list[str] = []

        if not any_clinical_utility:
            grade = min(grade, 6)
            rationale_parts.append(
                "no prospective clinical-utility evidence -> TRL <= 6"
            )
        if not any_clinical_validity:
            grade = min(grade, 4)
            rationale_parts.append(
                "no independent-cohort clinical-validity evidence -> TRL <= 4"
            )
        if tier_c_fail:
            grade = min(grade, 3)
            rationale_parts.append(
                "Tier C failure (forecasting/baseline/safety) -> TRL <= 3"
            )
        if tier_b_fail:
            grade = min(grade, 3)
            rationale_parts.append(
                "Tier B failure (model mechanism) -> TRL <= 3"
            )
        if tier_a_fail:
            grade = min(grade, 2)
            rationale_parts.append(
                "Tier A failure (data integrity / identifiability) -> "
                "TRL bounded above at 2 by hard rule"
            )

        grade = max(grade, 1)
        criteria_results["overall_trl_grade"] = grade

        # ------------------------------------------------------------------
        # Aggregate required changes with severity scoring + sort.
        # ------------------------------------------------------------------
        prioritized: list[PrioritizedChange] = []
        for f in findings:
            for change in f.required_changes or []:
                severity = _classify_severity(change, f.tier, f.verdict)
                prioritized.append(
                    PrioritizedChange(
                        description=change,
                        severity=severity,
                        severity_score=SEVERITY_ORDER.get(severity, 0),
                        source_agent=f.agent_name,
                        source_tier=f.tier,
                    )
                )
        prioritized.sort(key=lambda c: c.severity_score, reverse=True)

        # ------------------------------------------------------------------
        # Blocking issues + re-run conditions.
        # ------------------------------------------------------------------
        blocking: list[str] = []
        if tier_a_fail:
            blocking.append(
                "Tier A failure: data integrity/identifiability not "
                "established. Re-run only after the underlying Tier A "
                "agents return PASS or CONDITIONAL_PASS."
            )
        for f in findings:
            if _is_fail(f.verdict):
                for fm in f.failure_modes or []:
                    if any(
                        kw in fm.lower()
                        for kw in (
                            "fatal",
                            "potentially fatal",
                            "ode",
                            "miscalibrated",
                            "leak",
                        )
                    ):
                        blocking.append(f"[{f.agent_name}] {fm}")

        rerun_conditions = [
            "all Tier A findings have verdict in {PASS, CONDITIONAL_PASS}",
            "all blocking_issues resolved with documented evidence",
            "any new pre-registered experiment is recorded before execution",
        ]
        if not any_clinical_validity:
            rerun_conditions.append(
                "external-cohort clinical validity evidence supplied to "
                "ClinicalTranslationSafetyAgent"
            )

        rationale = (
            "; ".join(rationale_parts)
            if rationale_parts
            else "all gates clear at the available evidence ceiling"
        )

        report = ChairReport(
            overall_grade=grade,
            blocking_issues=blocking,
            required_changes_prioritized=prioritized,
            re_run_conditions=rerun_conditions,
            rationale=rationale,
            per_tier_summary=per_tier,
            debate_log=debate_log,
        )

        return self._wrap(report, criteria_results, evidence)

    # ------------------------------------------------------------------
    # Conflict resolution.
    # ------------------------------------------------------------------
    def debate_resolution(
        self, findings: list[EvalFinding]
    ) -> list[dict[str, Any]]:
        """Resolve conflicts among findings.

        Rule: only **peer-reviewed-style evidence** (entries in
        ``finding.evidence`` flagged as ``"peer_reviewed": True`` or
        carrying a ``"citation"`` field) and **pre-registered experiments**
        (entries flagged as ``"pre_registered": True`` or appearing inside
        ``evidence['pre_registration']``) carry weight in conflict
        resolution. Anonymous or post-hoc opinion is logged but does not
        change the outcome.

        Returns a list of debate-log records.
        """
        log: list[dict[str, Any]] = []

        # Group findings by the criterion they touch (tier as a coarse proxy).
        by_tier: dict[str, list[EvalFinding]] = {}
        for f in findings:
            by_tier.setdefault(f.tier, []).append(f)

        for tier, group in by_tier.items():
            verdicts = {
                getattr(f.verdict, "value", str(f.verdict)).upper() for f in group
            }
            if len(verdicts) <= 1:
                continue
            # Conflict detected. Re-weight by admissible evidence.
            ranked: list[tuple[EvalFinding, int]] = []
            for f in group:
                weight = _admissible_evidence_weight(f)
                ranked.append((f, weight))
            ranked.sort(key=lambda fw: fw[1], reverse=True)
            winner, winner_weight = ranked[0]
            log.append(
                {
                    "tier": tier,
                    "conflicting_verdicts": sorted(verdicts),
                    "agents": [f.agent_name for f in group],
                    "resolution_rule": (
                        "peer-reviewed-style or pre-registered evidence "
                        "outweighs ungrounded claims"
                    ),
                    "winner_agent": winner.agent_name,
                    "winner_weight": winner_weight,
                    "winner_verdict": getattr(
                        winner.verdict, "value", str(winner.verdict)
                    ),
                }
            )
        return log

    # ------------------------------------------------------------------
    def _wrap(
        self,
        report: ChairReport,
        criteria_results: dict[str, Any],
        evidence: dict[str, Any],
    ) -> EvalFinding:
        """Pack a ChairReport into an EvalFinding."""
        verdict: Verdict
        if report.overall_grade >= 6:
            verdict = Verdict.PASS
        elif report.overall_grade >= 3:
            verdict = Verdict.CONDITIONAL
        else:
            verdict = Verdict.FAIL

        evidence = dict(evidence)
        evidence["chair_report"] = report.to_dict()

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=float(report.overall_grade) / 9.0,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=report.blocking_issues,
            required_changes=[c.description for c in report.required_changes_prioritized],
        )


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------


def _is_fail(v: Any) -> bool:
    val = getattr(v, "value", str(v)).upper()
    return val == "FAIL"


def _classify_severity(change: str, tier: str, verdict: Any) -> str:
    text = change.lower()
    if any(
        kw in text
        for kw in (
            "fatal",
            "blood-bank",
            "cardiac",
            "potentially fatal",
        )
    ):
        return "block_clinical"
    if "calibrate ode" in text or "leak" in text or "identifiability" in text:
        return "block_merge"
    if tier == "A" and _is_fail(verdict):
        return "block_merge"
    if tier == "C" and _is_fail(verdict):
        return "block_publication"
    if "clinical validity" in text or "prospective" in text:
        return "high"
    if "subgroup" in text or "ood" in text:
        return "high"
    if "report" in text or "document" in text:
        return "medium"
    return "low"


def _admissible_evidence_weight(f: EvalFinding) -> int:
    """Weight = count of admissible evidence items in a finding."""
    weight = 0
    ev = f.evidence or {}
    if not isinstance(ev, dict):
        return 0
    if ev.get("pre_registration"):
        weight += 5
    for v in ev.values():
        if isinstance(v, dict):
            if v.get("peer_reviewed"):
                weight += 3
            if v.get("pre_registered"):
                weight += 3
            if "citation" in v or "doi" in v:
                weight += 2
    return weight
