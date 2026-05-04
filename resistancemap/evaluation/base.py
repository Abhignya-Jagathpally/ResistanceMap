"""Base class for evaluation governance agents.

Evaluation agents extend :class:`~resistancemap.agents.base.BaseAgent` so the
existing zero-trust verification, hashing, and DAG plumbing can be reused.
The new pieces are:

* a tier annotation (``A``/``B``/``C``/``D``),
* an :meth:`EvalAgent.assess` abstract method that returns an
  :class:`EvalFinding` instead of a free-form dict, and
* small helpers (``skip_if_data_missing``, ``record_evidence``,
  ``fail_closed_gate``) so individual agents stay short and uniform.

The :class:`EvalFinding` dataclass is the cross-agent contract: every
evaluation agent in this layer returns one, and the orchestrator aggregates
them into the final :class:`~resistancemap.evaluation.orchestrator.EvalReport`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from resistancemap.agents.base import AgentResult, AgentState, BaseAgent
from resistancemap.config import ResistanceMapConfig

logger = logging.getLogger(__name__)


class EvalVerdict(Enum):
    """Possible verdicts for an evaluation finding.

    Members:
        PASS: All required criteria met; agent endorses the artefact.
        FAIL: At least one *required* criterion failed; downstream
            consumers must treat the artefact as unfit.
        CONDITIONAL: Soft criteria failed but no required criterion did;
            artefact is usable subject to ``required_changes``.
        SKIPPED: Inputs / data are absent. The agent did not run its main
            logic; it produced a memo describing what is missing instead.
            **Never** use SKIPPED to silently hide a failure.
        BLOCKED: A higher-tier hard-stop prevented this agent from running.
            Set by the orchestrator, not by the agent itself.
    """

    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


# Alias for agents / tests that import the shorter name.
Verdict = EvalVerdict


@dataclass
class EvalFinding:
    """Structured finding emitted by every evaluation agent.

    Attributes:
        agent_name: Name of the producing agent (matches ``BaseAgent.name``).
        tier: Tier letter, one of ``"A"``, ``"B"``, ``"C"``, ``"D"``.
        verdict: An :class:`EvalVerdict` member.
        score: Aggregated weighted score in [0, 1]. Reported even when
            ``verdict == SKIPPED`` (in which case it is typically 0.0).
        criteria_results: Mapping ``criterion -> raw value`` for each rubric
            criterion the agent evaluated.
        evidence: Mapping ``key -> evidence payload`` recording the artefacts
            (paths, metric dicts, hashes, TODO stubs) the agent relied on.
        failure_modes: Concrete failure modes the agent observed or
            anticipated. These are the *things that went wrong*, not
            recommendations.
        required_changes: Concrete actions that, if applied, would let this
            agent re-run with a higher verdict. These are the
            *recommendations*.
    """

    agent_name: str
    tier: str
    verdict: EvalVerdict
    score: float = 0.0
    criteria_results: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    required_changes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Coerce ``verdict`` from string and clamp ``score`` into [0, 1].

        ``None`` scores (e.g. from agents that intentionally do not assign
        a numeric grade) are coerced to 0.0 rather than raising. This keeps
        partial findings constructible during ad-hoc instantiation.
        """
        if isinstance(self.verdict, str):
            self.verdict = EvalVerdict(self.verdict)
        if self.score is None:
            self.score = 0.0
        self.score = max(0.0, min(1.0, float(self.score)))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (used by the orchestrator)."""
        return {
            "agent_name": self.agent_name,
            "tier": self.tier,
            "verdict": self.verdict.value,
            "score": self.score,
            "criteria_results": self.criteria_results,
            "evidence": _stringify(self.evidence),
            "failure_modes": list(self.failure_modes),
            "required_changes": list(self.required_changes),
        }


def _stringify(obj: Any) -> Any:
    """Recursively coerce ``Path`` and other non-JSON values into strings."""
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


class EvalAgent(BaseAgent):
    """Base class for ResistanceMap evaluation agents.

    Subclasses must:

    1. set the class attribute :attr:`tier` to one of ``"A"``, ``"B"``,
       ``"C"``, ``"D"``;
    2. implement :meth:`assess`, returning an :class:`EvalFinding`.

    The base class wraps :meth:`assess` in :meth:`execute` so the agent
    plugs into the existing :class:`~resistancemap.agents.orchestrator.Orchestrator`
    and zero-trust hashing without further work.
    """

    #: Tier letter. Subclasses **must** override (or pass via __init__).
    tier: str = ""

    def __init__(
        self,
        name: str,
        tier: str | None = None,
        dependencies: list[str] | None = None,
    ) -> None:
        """Initialise the evaluation agent.

        Args:
            name: Unique agent identifier.
            tier: Optional explicit tier letter ("A"|"B"|"C"|"D"). If
                provided, becomes an instance attribute that overrides the
                class-level :attr:`tier`. This is convenient for fixtures and
                ad-hoc agents that do not warrant a dedicated subclass.
            dependencies: Optional list of upstream agent names. Defaults to
                an empty list (most Tier A agents are roots).
        """
        super().__init__(name=name, dependencies=dependencies)
        if tier is not None:
            self.tier = tier
        if self.tier not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"{type(self).__name__}: tier must be one of A/B/C/D, got {self.tier!r}"
            )
        self._evidence_buffer: dict[str, Any] = {}

    # ----------------------------------------------------------------- API

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        """Run the evaluation logic and return a structured finding.

        Subclasses should normally override this method. Advanced subclasses
        (e.g. test fixtures) may instead override :meth:`execute` directly,
        in which case this method is unused. The default implementation
        raises :class:`NotImplementedError` so an unconfigured agent fails
        loudly rather than silently producing a stub finding.

        Args:
            intake: Inputs prepared by the orchestrator. For Tier A this is
                normally an empty dict; for downstream tiers it contains
                upstream :class:`EvalFinding` objects keyed by agent name.
            config: The ResistanceMap configuration to audit.

        Returns:
            A populated :class:`EvalFinding`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override assess() or execute()"
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> AgentResult:
        """Adapter from :class:`BaseAgent` ``execute`` to :meth:`assess`.

        The returned :class:`AgentResult` carries the serialised finding as
        its ``output`` (so the orchestrator can hash it) and the live
        :class:`EvalFinding` instance under ``metadata['finding']`` so
        downstream evaluation agents can use it directly.
        """
        try:
            finding = await self.assess(inputs, config)
        except Exception as exc:  # noqa: BLE001 -- intentional broad catch
            logger.exception("%s: assess() raised", self.name)
            finding = EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=EvalVerdict.FAIL,
                failure_modes=[f"assess() raised {type(exc).__name__}: {exc}"],
                required_changes=["Fix the exception in the agent's assess()."],
            )

        output = finding.to_dict()
        return self._make_result(
            status=AgentState.COMPLETED,
            output=output,
            metadata={"finding": finding, "tier": self.tier},
        )

    # ------------------------------------------------------------- helpers

    def skip_if_data_missing(
        self,
        paths: Iterable[Path],
        memo: str | None = None,
    ) -> EvalFinding | None:
        """Return a SKIPPED finding if any of ``paths`` does not exist.

        This is the canonical "no fabricated data" escape hatch: when an
        evaluation agent's required inputs are absent, it must produce a
        finding that documents *what* is missing, not invent stand-ins.

        Args:
            paths: Required filesystem paths.
            memo: Optional human-readable memo to attach as
                ``failure_modes[0]``. A default is provided.

        Returns:
            A SKIPPED :class:`EvalFinding` listing the missing paths, or
            ``None`` if every path exists (caller should proceed normally).
        """
        missing = [Path(p) for p in paths if not Path(p).exists()]
        if not missing:
            return None

        message = memo or (
            "Required inputs are missing on disk; agent skipped to avoid "
            "fabricating data."
        )
        logger.warning(
            "%s: skipping (%d missing path(s)): %s",
            self.name,
            len(missing),
            [str(p) for p in missing],
        )
        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=EvalVerdict.SKIPPED,
            score=0.0,
            criteria_results={},
            evidence={"missing_paths": [str(p) for p in missing]},
            failure_modes=[message],
            required_changes=[
                f"Provide the missing input at {p}" for p in missing
            ],
        )

    def record_evidence(self, key: str, value: Any) -> None:
        """Buffer a piece of evidence for the next finding.

        Subclasses can call this freely during :meth:`assess` and then read
        :attr:`_evidence_buffer` (or pass it through directly) when building
        the :class:`EvalFinding`.

        Args:
            key: Stable identifier for the evidence.
            value: Any JSON-serialisable payload (Paths are auto-stringified).
        """
        self._evidence_buffer[key] = value

    def fail_closed_gate(
        self,
        reason: str,
        required_changes: list[str] | None = None,
    ) -> EvalFinding:
        """Build a FAIL finding for a hard-gate violation.

        Used when an agent encounters a condition that *must* block the
        downstream pipeline (e.g. patient ID overlap between train and
        test). The orchestrator's hard-stop logic relies on these returning
        ``EvalVerdict.FAIL``.

        Args:
            reason: Single human-readable reason; recorded in
                ``failure_modes``.
            required_changes: Optional remediation steps; defaults to a
                generic instruction to address the reason.

        Returns:
            A FAIL :class:`EvalFinding`.
        """
        logger.error("%s: fail-closed gate tripped: %s", self.name, reason)
        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=EvalVerdict.FAIL,
            score=0.0,
            evidence=dict(self._evidence_buffer),
            failure_modes=[reason],
            required_changes=required_changes or [f"Address: {reason}"],
        )
