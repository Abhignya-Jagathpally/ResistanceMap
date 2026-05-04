"""Tiered orchestrator for the evaluation governance layer.

The :class:`EvalOrchestrator` runs registered :class:`EvalAgent` instances in
strict tier order:

1. **Tier A** -- run in parallel via :func:`asyncio.gather`. If *any* Tier A
   agent returns ``EvalVerdict.FAIL``, the orchestrator hard-stops: every
   Tier B / C / D agent is marked ``BLOCKED`` with the failing Tier A agent
   recorded as the blocking reason.
2. **Tier B** -- in parallel within the tier (assuming Tier A passed or only
   produced ``CONDITIONAL`` verdicts).
3. **Tier C** -- in parallel within the tier.
4. **Tier D** -- chair, runs last and consumes every prior finding.

Between Tier B and Tier C an :meth:`adversarial_round` hook is invoked. The
default implementation is a placeholder that records the architecture vs
baseline-adversary findings into the report metadata; the rules of the
debate are intentionally minimal here so that other agents (or future
revisions) can override the method without disturbing the orchestration
order.

The orchestrator writes a JSON audit trail under
``logs/evaluation/<run_id>/`` consisting of:

* ``report.json`` -- the full :class:`EvalReport`.
* ``findings.jsonl`` -- one finding per line in execution order.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from resistancemap.config import ResistanceMapConfig
from resistancemap.evaluation.base import EvalAgent, EvalFinding, EvalVerdict

logger = logging.getLogger(__name__)


_TIER_ORDER: tuple[str, ...] = ("A", "B", "C", "D")


@dataclass
class EvalReport:
    """Aggregated evaluation report for one run.

    Attributes:
        run_id: Stable identifier for the run (used in the audit trail path).
        per_tier_verdicts: Mapping ``tier -> {agent_name: verdict_string}``.
        findings: All findings (including BLOCKED) in execution order.
        overall_verdict: Conservative roll-up across tiers (FAIL > BLOCKED >
            CONDITIONAL > PASS; SKIPPED is treated as CONDITIONAL).
        maturity_grade: TRL-style 1--9 grade with a one-line justification.
        required_changes: Aggregated, de-duplicated remediation list.
        verification_chain: Ordered list of SHA-256 hashes (one per finding).
        timing: Mapping ``agent_name -> elapsed_seconds``.
        charter_notes: Free-form notes carried from the charter.
        metadata: Free-form bag (e.g. adversarial round results).
    """

    run_id: str
    per_tier_verdicts: dict[str, EvalVerdict] = field(default_factory=dict)
    per_tier_agent_verdicts: dict[str, dict[str, EvalVerdict]] = field(default_factory=dict)
    findings: list[EvalFinding] = field(default_factory=list)
    overall_verdict: EvalVerdict = EvalVerdict.CONDITIONAL
    maturity_grade: dict[str, Any] = field(default_factory=dict)
    required_changes: list[str] = field(default_factory=list)
    verification_chain: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    charter_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view of the report.

        Verdicts are coerced to their string ``.value`` form and findings
        are serialised via :meth:`EvalFinding.to_dict` so the result is
        suitable for ``json.dump``.
        """
        return {
            "run_id": self.run_id,
            "per_tier_verdicts": {
                tier: (v.value if isinstance(v, EvalVerdict) else v)
                for tier, v in self.per_tier_verdicts.items()
            },
            "per_tier_agent_verdicts": {
                tier: {name: (v.value if isinstance(v, EvalVerdict) else v)
                       for name, v in agents.items()}
                for tier, agents in self.per_tier_agent_verdicts.items()
            },
            "findings": [f.to_dict() for f in self.findings],
            "overall_verdict": (
                self.overall_verdict.value
                if isinstance(self.overall_verdict, EvalVerdict)
                else self.overall_verdict
            ),
            "maturity_grade": self.maturity_grade,
            "required_changes": self.required_changes,
            "verification_chain": self.verification_chain,
            "timing": self.timing,
            "charter_notes": self.charter_notes,
            "metadata": self.metadata,
        }


class EvalOrchestrator:
    """Tiered, hard-stopping orchestrator for evaluation agents.

    Attributes:
        agents: Mapping ``tier -> [EvalAgent, ...]``.
        log_root: Root directory for evaluation audit trails.
    """

    def __init__(self, log_root: Path | str = Path("logs/evaluation")) -> None:
        """Initialise the orchestrator.

        Args:
            log_root: Root directory under which per-run audit folders are
                created. Defaults to ``logs/evaluation``.
        """
        self.agents: dict[str, list[EvalAgent]] = {tier: [] for tier in _TIER_ORDER}
        self.log_root = Path(log_root)

    # ----------------------------------------------------------- registration

    def add_agent(self, agent: EvalAgent) -> None:
        """Register an :class:`EvalAgent` under its declared tier.

        Args:
            agent: The evaluation agent to register.

        Raises:
            ValueError: If the agent's tier is not one of A/B/C/D.
        """
        if agent.tier not in _TIER_ORDER:
            raise ValueError(
                f"Cannot register {agent.name}: tier {agent.tier!r} not in {_TIER_ORDER}"
            )
        self.agents[agent.tier].append(agent)
        logger.debug(
            "EvalOrchestrator: registered %s under tier %s", agent.name, agent.tier
        )

    # ----------------------------------------------------------------- run

    async def run(
        self,
        config: ResistanceMapConfig,
        run_id: str | None = None,
    ) -> EvalReport:
        """Execute all registered agents in tier order.

        Args:
            config: ResistanceMap configuration to audit.
            run_id: Optional stable run identifier. Defaults to a UTC
                timestamp.

        Returns:
            A populated :class:`EvalReport`. The report is also written to
            ``log_root/<run_id>/report.json`` and ``findings.jsonl``.
        """
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = EvalReport(run_id=run_id)
        report.charter_notes = []

        all_findings: list[EvalFinding] = []
        hard_stop_reason: str | None = None

        for tier in _TIER_ORDER:
            tier_agents = self.agents.get(tier, [])
            if not tier_agents:
                report.per_tier_verdicts[tier] = {}
                continue

            if hard_stop_reason is not None:
                # Mark every remaining agent BLOCKED without running it.
                blocked = [
                    EvalFinding(
                        agent_name=ag.name,
                        tier=ag.tier,
                        verdict=EvalVerdict.BLOCKED,
                        score=0.0,
                        failure_modes=[hard_stop_reason],
                        required_changes=[
                            "Resolve the upstream Tier A failure, then re-run."
                        ],
                    )
                    for ag in tier_agents
                ]
                for ag, finding in zip(tier_agents, blocked):
                    self._record(report, ag, finding, all_findings, elapsed=0.0)
                continue

            tier_findings = await self._run_tier(tier_agents, config, report)
            all_findings.extend(tier_findings)

            # Hard-stop check: any Tier A FAIL halts everything.
            if tier == "A":
                failed = [f for f in tier_findings if f.verdict == EvalVerdict.FAIL]
                if failed:
                    names = ", ".join(f.agent_name for f in failed)
                    hard_stop_reason = (
                        f"Tier A hard-stop: {names} returned FAIL"
                    )
                    logger.error("EvalOrchestrator: %s", hard_stop_reason)

            # Adversarial round between Tier B and Tier C.
            if tier == "B" and hard_stop_reason is None:
                arch = next(
                    (f for f in tier_findings if "architecture" in f.agent_name),
                    None,
                )
                baseline = next(
                    (f for f in tier_findings if "baseline" in f.agent_name),
                    None,
                )
                if arch is not None and baseline is not None:
                    debate = self.adversarial_round(arch, baseline)
                    report.metadata.setdefault("adversarial_rounds", []).append(debate)

        # Tier D chair has by now produced its finding(s); roll up.
        self._finalize(report, all_findings)
        self._write_audit_trail(report)
        return report

    # --------------------------------------------------------- tier helpers

    async def _run_tier(
        self,
        tier_agents: list[EvalAgent],
        config: ResistanceMapConfig,
        report: EvalReport,
    ) -> list[EvalFinding]:
        """Run all agents in a single tier in parallel.

        Args:
            tier_agents: Agents belonging to the tier.
            config: ResistanceMap configuration.
            report: The report being assembled (mutated in-place for
                timing / verification entries).

        Returns:
            The list of findings produced by this tier (in agent order).
        """
        tier_label = tier_agents[0].tier
        logger.info(
            "EvalOrchestrator: tier %s (%d agents)", tier_label, len(tier_agents)
        )

        # Gather upstream findings as the intake for this tier.
        intake: dict[str, EvalFinding] = {
            prior.agent_name: prior for prior in report.findings
        }
        # Tier D agents (chair) want the full list under a stable key.
        intake["findings"] = list(report.findings)

        async def _wrapped(agent: EvalAgent) -> tuple[EvalAgent, EvalFinding, float]:
            start = time.time()
            try:
                result = await agent._safe_execute(intake, config)
                elapsed = time.time() - start
                # Prefer the live finding stuffed in metadata; fall back to
                # result.output if it is itself an EvalFinding (subclasses
                # that override execute() directly typically do this).
                finding = result.metadata.get("finding") if result.metadata else None
                if finding is None and isinstance(result.output, EvalFinding):
                    finding = result.output
                if finding is None:
                    finding = EvalFinding(
                        agent_name=agent.name,
                        tier=agent.tier,
                        verdict=EvalVerdict.FAIL,
                        failure_modes=[
                            f"Agent did not return an EvalFinding (status={result.status.value})"
                        ],
                        required_changes=[
                            "Have the agent's assess() return an EvalFinding."
                        ],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("EvalOrchestrator: %s crashed", agent.name)
                elapsed = time.time() - start
                finding = EvalFinding(
                    agent_name=agent.name,
                    tier=agent.tier,
                    verdict=EvalVerdict.FAIL,
                    failure_modes=[f"{type(exc).__name__}: {exc}"],
                    required_changes=["Debug the unhandled exception."],
                )
            return agent, finding, elapsed

        results = await asyncio.gather(*(_wrapped(a) for a in tier_agents))

        produced: list[EvalFinding] = []
        for agent, finding, elapsed in results:
            self._record(report, agent, finding, produced, elapsed)
        return produced

    def _record(
        self,
        report: EvalReport,
        agent: EvalAgent,
        finding: EvalFinding,
        bucket: list[EvalFinding],
        elapsed: float,
    ) -> None:
        """Append one finding to the report and the per-tier bucket.

        Live :class:`EvalFinding` instances are stored on the report so that
        downstream consumers (tests, the chair, programmatic callers) can
        introspect verdict enums and required_changes lists directly.
        Serialisation to JSON happens at write time.
        """
        bucket.append(finding)
        report.findings.append(finding)
        report.timing[agent.name] = elapsed
        report.per_tier_agent_verdicts.setdefault(agent.tier, {})[agent.name] = finding.verdict
        # Roll up the worst per-agent verdict in this tier (FAIL > BLOCKED >
        # SKIPPED > CONDITIONAL > PASS) into the per-tier summary verdict.
        report.per_tier_verdicts[agent.tier] = self._tier_rollup(
            report.per_tier_agent_verdicts[agent.tier].values()
        )
        digest = hashlib.sha256(
            json.dumps(finding.to_dict(), sort_keys=True, default=str).encode()
        ).hexdigest()
        report.verification_chain.append(digest)
        logger.info(
            "EvalOrchestrator: %s -> %s (score=%.3f, hash=%s)",
            agent.name,
            finding.verdict.value,
            finding.score,
            digest[:8],
        )

    @staticmethod
    def _tier_rollup(verdicts: Any) -> EvalVerdict:
        """Reduce a tier's per-agent verdicts to a single worst-case verdict.

        Severity ordering (worst first): FAIL > BLOCKED > SKIPPED >
        CONDITIONAL > PASS. An empty input rolls up to CONDITIONAL.
        """
        order = (
            EvalVerdict.FAIL,
            EvalVerdict.BLOCKED,
            EvalVerdict.SKIPPED,
            EvalVerdict.CONDITIONAL,
            EvalVerdict.PASS,
        )
        seen = set(verdicts)
        if not seen:
            return EvalVerdict.CONDITIONAL
        for v in order:
            if v in seen:
                return v
        return EvalVerdict.CONDITIONAL

    # ----------------------------------------------------- adversarial hook

    def adversarial_round(
        self,
        arch_finding: EvalFinding | dict[str, Any],
        baseline_finding: EvalFinding | dict[str, Any],
    ) -> dict[str, Any]:
        """Placeholder architecture-vs-baseline debate hook.

        The default implementation does **not** modify either finding. It
        returns a structured debate record describing the rules and the
        outcome under those rules. Subclasses or future revisions can
        override this to implement, e.g., a multi-round critique loop.

        Rules (default):

        1. The architecture finding only "wins" if its score strictly
           exceeds the baseline finding by at least 0.05.
        2. Otherwise the round is recorded as a tie and the baseline's
           required_changes are merged into the architecture's.

        Args:
            arch_finding: The architecture agent's finding (Tier B).
            baseline_finding: The baseline-adversary agent's finding (Tier B).

        Returns:
            A dict describing the debate outcome.
        """
        arch_score = float(_get(arch_finding, "score", 0.0))
        base_score = float(_get(baseline_finding, "score", 0.0))
        margin = arch_score - base_score
        outcome = "architecture_wins" if margin >= 0.05 else "tie_or_baseline"
        record = {
            "rules": "arch must beat baseline by >= 0.05 to win",
            "arch_agent": _get(arch_finding, "agent_name", "unknown"),
            "baseline_agent": _get(baseline_finding, "agent_name", "unknown"),
            "arch_score": arch_score,
            "baseline_score": base_score,
            "margin": margin,
            "outcome": outcome,
        }
        logger.info(
            "EvalOrchestrator.adversarial_round: %s (margin=%.3f)", outcome, margin
        )
        return record

    # ----------------------------------------------------------- finalisation

    def _finalize(self, report: EvalReport, findings: list[EvalFinding]) -> None:
        """Compute roll-up verdict, maturity grade, and required_changes."""
        # Roll-up verdict.
        verdicts = {f.verdict for f in findings}
        if EvalVerdict.FAIL in verdicts:
            overall = EvalVerdict.FAIL
        elif EvalVerdict.BLOCKED in verdicts and EvalVerdict.PASS not in verdicts:
            overall = EvalVerdict.BLOCKED
        elif EvalVerdict.CONDITIONAL in verdicts or EvalVerdict.SKIPPED in verdicts:
            overall = EvalVerdict.CONDITIONAL
        elif EvalVerdict.PASS in verdicts:
            overall = EvalVerdict.PASS
        else:
            overall = EvalVerdict.CONDITIONAL
        report.overall_verdict = overall

        # Aggregate de-duplicated required changes.
        seen: set[str] = set()
        merged: list[str] = []
        for f in findings:
            for change in f.required_changes:
                if change not in seen:
                    seen.add(change)
                    merged.append(change)
        report.required_changes = merged

        # TRL-style maturity grade.
        report.maturity_grade = self._maturity_grade(findings, overall)

    @staticmethod
    def _maturity_grade(
        findings: list[EvalFinding], overall: EvalVerdict
    ) -> dict[str, Any]:
        """Return a TRL-style 1--9 maturity grade with a justification.

        The mapping is intentionally conservative: ResistanceMap is a
        research artefact and the highest grade reachable from a passing
        evaluation alone is TRL 4 (component validation in lab). Anything
        higher requires evidence the evaluation layer cannot itself
        manufacture (in-vivo, prospective, regulated study).
        """
        if not findings:
            return {
                "trl": 1,
                "justification": "No findings recorded; charter only.",
            }

        n = len(findings)
        n_pass = sum(1 for f in findings if f.verdict == EvalVerdict.PASS)
        n_fail = sum(1 for f in findings if f.verdict == EvalVerdict.FAIL)
        n_blocked = sum(1 for f in findings if f.verdict == EvalVerdict.BLOCKED)

        if overall == EvalVerdict.FAIL or n_blocked:
            trl = 1
            justification = (
                f"Hard-stop: {n_fail} FAIL / {n_blocked} BLOCKED of {n} findings."
            )
        elif overall == EvalVerdict.PASS and n_pass == n:
            trl = 4
            justification = (
                "All evaluation criteria PASS; component-level lab validation."
            )
        elif overall == EvalVerdict.PASS:
            trl = 3
            justification = (
                "PASS overall but not unanimous; analytical proof of concept."
            )
        elif overall == EvalVerdict.CONDITIONAL:
            trl = 2
            justification = (
                "Conditional pass; technology concept formulated, gaps remain."
            )
        else:
            trl = 1
            justification = "Basic principles only; insufficient evidence."

        return {
            "trl": trl,
            "justification": justification,
            "counts": {
                "total": n,
                "pass": n_pass,
                "fail": n_fail,
                "blocked": n_blocked,
                "skipped": sum(
                    1 for f in findings if f.verdict == EvalVerdict.SKIPPED
                ),
                "conditional": sum(
                    1 for f in findings if f.verdict == EvalVerdict.CONDITIONAL
                ),
            },
        }

    # ------------------------------------------------------------ audit log

    def _write_audit_trail(self, report: EvalReport) -> None:
        """Persist ``report.json`` and ``findings.jsonl`` under ``log_root``."""
        run_dir = self.log_root / report.run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(
                "EvalOrchestrator: cannot create audit dir %s: %s", run_dir, exc
            )
            return

        report_path = run_dir / "report.json"
        findings_path = run_dir / "findings.jsonl"

        try:
            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            with open(findings_path, "w") as f:
                for finding in report.findings:
                    serialised = (
                        finding.to_dict() if isinstance(finding, EvalFinding) else finding
                    )
                    f.write(json.dumps(serialised, default=str) + "\n")
            logger.info(
                "EvalOrchestrator: wrote audit trail to %s", run_dir
            )
        except OSError as exc:
            logger.error(
                "EvalOrchestrator: failed to write audit trail: %s", exc
            )


def _get(finding_or_dict: EvalFinding | dict[str, Any], key: str, default: Any) -> Any:
    """Tolerant accessor for either an :class:`EvalFinding` or a dict view."""
    if isinstance(finding_or_dict, EvalFinding):
        return getattr(finding_or_dict, key, default)
    if isinstance(finding_or_dict, dict):
        return finding_or_dict.get(key, default)
    return default
