"""Weighted rubric for evaluation governance.

A :class:`Rubric` is a 3-level table:

    tier -> agent -> criterion -> {weight, evidence_type, required, ...}

Evaluation agents call :meth:`Rubric.score` to record a single criterion's
value and receive the weighted contribution back. :meth:`Rubric.tabulate`
returns a flat view suitable for logging or for the Tier D chair.

Pandas is *optional*: if it is importable, ``tabulate`` returns a
``DataFrame``; otherwise it returns a dict-of-dicts with the same shape so
the rubric remains usable in minimal environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


_VALID_EVIDENCE_TYPES = {
    "file",
    "metric",
    "schema",
    "document",
    "hash",
    "todo",
    "report",
    "table",
    "plot",
    "manifest",
}


@dataclass
class _ScoreEntry:
    """Single recorded score for ``(agent, criterion)``.

    Attributes:
        value: Raw value supplied by the agent. Numeric values in [0, 1] are
            interpreted directly; bools are coerced to 1.0 / 0.0; anything
            else is interpreted via :meth:`Rubric._coerce`.
        evidence: Free-form evidence payload (path, metric dict, hash, ...).
        weighted: ``value * weight`` after coercion.
    """

    value: Any
    evidence: Any
    weighted: float


@dataclass
class Rubric:
    """Weighted scoring rubric loaded from YAML.

    Attributes:
        table: Nested dict ``tier -> agent -> criterion -> spec``.
        scores: Recorded scores keyed by ``(tier, agent, criterion)``.
    """

    table: dict[str, dict[str, dict[str, dict[str, Any]]]] = field(default_factory=dict)
    scores: dict[tuple[str, str, str], _ScoreEntry] = field(default_factory=dict)

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "Rubric":
        """Build a :class:`Rubric` from an in-memory dict spec.

        Two schemas are accepted:

        1. The native ``tier -> agent -> criterion -> spec`` layout used by
           the YAML rubric on disk.
        2. A flatter ``{"agents": {agent_name: {"criteria": {criterion: spec}}}}``
           layout that is convenient for tests and ad-hoc rubrics. The flat
           layout is auto-promoted to a single synthetic tier ``"_default"``.

        Weights may be any non-negative float; the sum-to-one warning still
        fires when the convention is violated.
        """
        if not isinstance(spec, dict):
            raise ValueError(f"Rubric spec must be a mapping, got {type(spec)}")

        if "agents" in spec and isinstance(spec["agents"], dict):
            table: dict[str, dict[str, dict[str, dict[str, Any]]]] = {"_default": {}}
            for agent_name, agent_spec in spec["agents"].items():
                if not isinstance(agent_spec, dict):
                    raise ValueError(
                        f"agents.{agent_name} must be a mapping, got {type(agent_spec)}"
                    )
                criteria = agent_spec.get("criteria", {})
                if not isinstance(criteria, dict):
                    raise ValueError(
                        f"agents.{agent_name}.criteria must be a mapping"
                    )
                table["_default"][agent_name] = {
                    crit: dict(crit_spec) if isinstance(crit_spec, dict) else {"weight": float(crit_spec)}
                    for crit, crit_spec in criteria.items()
                }
        else:
            # Native tier-keyed layout (or empty).
            table = {tier: dict(agents) for tier, agents in spec.items()}

        rubric = cls(table=table)
        rubric._validate()
        return rubric

    def to_dict(self) -> dict[str, Any]:
        """Return the rubric table as a JSON-serialisable dict.

        The output uses the native ``tier -> agent -> criterion -> spec``
        layout regardless of which schema was used to build the rubric.
        """
        return {
            tier_name: {
                agent_name: {
                    crit_name: dict(crit_spec)
                    for crit_name, crit_spec in agents.items()
                }
                for agent_name, agents in tier.items()
            }
            for tier_name, tier in self.table.items()
        }

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Rubric":
        """Load a rubric from a YAML file.

        Args:
            path: Path to a rubric YAML in the schema documented in
                ``rubric.yaml``.

        Returns:
            A :class:`Rubric` populated from disk.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the YAML is structurally invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Rubric YAML not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            raise ValueError(f"Rubric YAML root must be a mapping, got {type(raw)}")

        rubric = cls(table=raw)
        rubric._validate()
        logger.debug(
            "Loaded rubric from %s: %d tiers", path, len(rubric.table)
        )
        return rubric

    def to_yaml(self, path: Path | str) -> None:
        """Persist the rubric (without recorded scores) back to YAML.

        Args:
            path: Destination path. Parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.table, f, sort_keys=False)
        logger.debug("Wrote rubric to %s", path)

    @classmethod
    def load_default(cls) -> "Rubric":
        """Load the rubric shipped next to this module (``rubric.yaml``)."""
        default_path = Path(__file__).parent / "rubric.yaml"
        return cls.from_yaml(default_path)

    # ----------------------------------------------------------- validation

    def _validate(self) -> None:
        """Lightweight structural validation of ``self.table``."""
        for tier_name, agents in self.table.items():
            if not isinstance(agents, dict):
                raise ValueError(
                    f"Tier {tier_name!r}: expected mapping of agent->criteria"
                )
            for agent_name, criteria in agents.items():
                if not isinstance(criteria, dict):
                    raise ValueError(
                        f"{tier_name}.{agent_name}: expected criteria mapping"
                    )
                total_weight = 0.0
                for crit_name, spec in criteria.items():
                    if not isinstance(spec, dict):
                        raise ValueError(
                            f"{tier_name}.{agent_name}.{crit_name}: spec must be a mapping"
                        )
                    weight = float(spec.get("weight", 0.0))
                    if weight < 0.0:
                        raise ValueError(
                            f"{tier_name}.{agent_name}.{crit_name}: weight {weight} must be non-negative"
                        )
                    total_weight += weight
                    ev_type = spec.get("evidence_type", "metric")
                    if ev_type not in _VALID_EVIDENCE_TYPES:
                        raise ValueError(
                            f"{tier_name}.{agent_name}.{crit_name}: "
                            f"evidence_type {ev_type!r} not in {_VALID_EVIDENCE_TYPES}"
                        )
                if criteria and abs(total_weight - 1.0) > 0.05:
                    logger.warning(
                        "Rubric %s.%s weights sum to %.3f, not 1.0",
                        tier_name,
                        agent_name,
                        total_weight,
                    )

    # -------------------------------------------------------------- scoring

    def _locate(
        self, agent: str, criterion: str
    ) -> tuple[str, dict[str, Any]] | tuple[None, None]:
        """Locate ``(tier_name, spec)`` for an ``(agent, criterion)`` pair."""
        for tier_name, agents in self.table.items():
            if agent in agents and criterion in agents[agent]:
                return tier_name, agents[agent][criterion]
        return None, None

    @staticmethod
    def _coerce(value: Any) -> float:
        """Coerce an arbitrary value into a [0, 1] score.

        Bools and numerics map directly. Strings ``pass``/``fail`` map to
        1.0/0.0. Anything else is treated as 0.0 with a warning. This
        method is intentionally permissive: agents are expected to do their
        own scoring and pass through a normalised value.
        """
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            v = float(value)
            return max(0.0, min(1.0, v))
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("pass", "ok", "yes", "true"):
                return 1.0
            if v in ("fail", "no", "false"):
                return 0.0
        logger.warning("Rubric._coerce: cannot coerce %r to score, using 0.0", value)
        return 0.0

    def score(
        self,
        agent: Any = None,
        criterion: str | None = None,
        value: Any = None,
        evidence: Any = None,
    ) -> Any:
        """Record a score, or compute weighted means from a nested raw dict.

        This method has two call modes:

        **Pointwise:** ``score(agent, criterion, value, evidence=None)``
        records one ``(agent, criterion)`` value and returns the
        ``value * weight`` contribution. ``0.0`` is returned (with a
        warning) if the criterion is unknown to the rubric.

        **Bulk:** ``score(raw_dict)`` where ``raw_dict`` has the shape
        ``{agent_name: {criterion_name: raw_value, ...}, ...}``. Returns a
        ``{agent_name: weighted_mean}`` dict where each weighted mean is
        ``sum(weight * coerced_value) / sum(weight)`` across the agent's
        criteria. This is the convention used by the rubric tests and by
        the Tier D chair when grading.
        """
        # Bulk mode: a single dict-of-dicts argument.
        if isinstance(agent, dict) and criterion is None and value is None:
            raw = agent
            results: dict[str, float] = {}
            for ag_name, ag_scores in raw.items():
                if not isinstance(ag_scores, dict):
                    raise ValueError(
                        f"score(raw_dict): expected mapping for agent "
                        f"{ag_name!r}, got {type(ag_scores)}"
                    )
                weighted_sum = 0.0
                weight_sum = 0.0
                for crit_name, raw_val in ag_scores.items():
                    _, spec = self._locate(ag_name, crit_name)
                    if spec is None:
                        logger.warning(
                            "Rubric.score(bulk): unknown criterion %s.%s, skipping",
                            ag_name,
                            crit_name,
                        )
                        continue
                    w = float(spec.get("weight", 0.0))
                    coerced = self._coerce(raw_val)
                    weighted_sum += w * coerced
                    weight_sum += w
                results[ag_name] = (
                    weighted_sum / weight_sum if weight_sum > 0 else 0.0
                )
            return results

        # Pointwise mode (legacy/native).
        if not isinstance(agent, str) or criterion is None:
            raise TypeError(
                "Rubric.score requires either (agent: str, criterion: str, value, ...) "
                "or a single nested-dict positional arg."
            )
        tier_name, spec = self._locate(agent, criterion)
        if spec is None:
            logger.warning(
                "Rubric.score: unknown criterion %s.%s, ignoring", agent, criterion
            )
            return 0.0
        weight = float(spec.get("weight", 0.0))
        coerced = self._coerce(value)
        weighted = coerced * weight
        self.scores[(tier_name, agent, criterion)] = _ScoreEntry(
            value=value, evidence=evidence, weighted=weighted
        )
        return weighted

    def agent_total(self, agent: str) -> float:
        """Sum the weighted scores recorded for one agent."""
        return sum(
            entry.weighted
            for (_, ag, _), entry in self.scores.items()
            if ag == agent
        )

    def tabulate(self) -> Any:
        """Tabulate recorded scores.

        Returns:
            A pandas DataFrame if pandas is importable, otherwise a
            ``dict[(tier, agent, criterion)] -> {value, weight, weighted, evidence}``
            structure with identical content.
        """
        rows = []
        for (tier_name, agent, criterion), entry in self.scores.items():
            spec = self.table.get(tier_name, {}).get(agent, {}).get(criterion, {})
            rows.append(
                {
                    "tier": tier_name,
                    "agent": agent,
                    "criterion": criterion,
                    "value": entry.value,
                    "weight": float(spec.get("weight", 0.0)),
                    "weighted": entry.weighted,
                    "evidence_type": spec.get("evidence_type"),
                    "required": bool(spec.get("required", False)),
                    "evidence": entry.evidence,
                }
            )

        try:
            import pandas as pd  # noqa: WPS433 (optional import)

            return pd.DataFrame(rows)
        except ImportError:
            logger.debug("pandas not available; tabulate() returning dict-of-dicts")
            return {
                (row["tier"], row["agent"], row["criterion"]): row for row in rows
            }
