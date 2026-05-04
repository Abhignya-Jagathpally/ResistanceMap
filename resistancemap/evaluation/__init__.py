"""ResistanceMap evaluation governance layer.

A multi-agent layer that audits the ResistanceMap pipeline against three
testable scientific claims (state, when, through-which-pathway). This package
is **orthogonal** to the training DAG under ``resistancemap.agents``: it
consumes the pipeline's configs and outputs and produces a graded, auditable
assessment without modifying the training pipeline itself.

The layer is structured as a tiered DAG:

* **Tier A** -- foundational fitness gates (data adequacy, measurement
  integration, bias / fairness / shift). Hard-stop tier: a Tier A FAIL marks
  every downstream agent BLOCKED.
* **Tier B** -- methodology / architecture / baselines (out of scope for this
  scaffold but discoverable via :class:`EvalOrchestrator`).
* **Tier C** -- forecasting & calibration / leakage and falsification.
* **Tier D** -- governance chair that consumes all findings and emits the
  final :class:`EvalReport`.

The public API exported here is intentionally narrow so other agents can
import the contracts (``EvalFinding`` shape, verdict enum, ``Charter``) without
pulling implementation details.
"""

from __future__ import annotations

from resistancemap.evaluation.base import (
    EvalAgent,
    EvalFinding,
    EvalVerdict,
    Verdict,
)
from resistancemap.evaluation.charter import (
    Charter,
    Claim,
    load_default_charter,
)
from resistancemap.evaluation.orchestrator import (
    EvalOrchestrator,
    EvalReport,
)
from resistancemap.evaluation.rubric import Rubric

__all__ = [
    "EvalAgent",
    "EvalFinding",
    "EvalVerdict",
    "Verdict",
    "EvalOrchestrator",
    "EvalReport",
    "Charter",
    "Claim",
    "load_default_charter",
    "Rubric",
]
