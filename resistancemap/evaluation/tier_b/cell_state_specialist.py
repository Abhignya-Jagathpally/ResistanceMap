"""Tier B: Cell-State / Trajectory Specialist agent.

Evaluates whether the latent "resistance states" recovered by the VAE +
trajectory pipeline are *biologically coherent* (consistent with known
hematologic / multiple-myeloma transcriptional programs and lineage
constraints) versus mere statistical clusters.

This agent never fabricates data. If the upstream VAE / trajectory outputs
are not present in ``intake``, it returns a SKIPPED verdict that explains
exactly which artefacts are missing.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

from resistancemap.agents.base import AgentState
from resistancemap.config import ResistanceMapConfig
from resistancemap.evaluation.base import EvalAgent, EvalFinding, Verdict

logger = logging.getLogger(__name__)


# Hematologic / MM-relevant transcriptional programs we expect to surface
# if "resistance states" are biologically coherent. These are the program
# *names*, not synthetic gene lists — caller-supplied gene scores are matched
# against these labels.
EXPECTED_PROGRAMS: tuple[str, ...] = (
    "stress_response",
    "emt_like",
    "metabolic_reprogramming",
    "interferon_response",
    "dna_damage_repair",
    "plasma_cell_identity",
    "proliferation",
    "unfolded_protein_response",
)


class CellStateTrajectoryAgent(EvalAgent):
    """Audit cell-state and trajectory outputs for biological coherence.

    Required intake keys (any missing -> SKIPPED, never fabricated):

    - ``vae_latents``      : np.ndarray of shape (n_cells, latent_dim)
    - ``cell_metadata``    : iterable of dicts with at least ``batch`` and
                             ``lineage`` keys per cell
    - ``state_assignments``: np.ndarray of integer state labels per cell
    - ``program_scores``   : dict[str, np.ndarray] mapping program name ->
                             per-cell score (e.g., from a signature scorer)
    - ``trajectory_transitions``: optional 2D array (n_states, n_states)
                                  of transition probabilities

    Optional intake key:

    - ``batch_split_states``: dict[batch_id, np.ndarray] giving the per-cell
      state assignment when re-fit per batch (used for batch robustness).
    """

    tier = "B"

    REQUIRED_INTAKE_KEYS: tuple[str, ...] = (
        "vae_latents",
        "cell_metadata",
        "state_assignments",
        "program_scores",
    )

    def __init__(self) -> None:
        super().__init__(
            name="cell_state_trajectory",
            dependencies=[],
        )

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        """Soft-verify: missing inputs lead to SKIPPED, not error."""
        if not isinstance(inputs, dict):
            return False, "intake must be a dict"
        return True, ""

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        """Evaluate cell-state coherence; skip-gracefully on missing data."""
        logger.info("CellStateTrajectoryAgent: starting biological-coherence audit")

        missing = [k for k in self.REQUIRED_INTAKE_KEYS if k not in intake]
        if missing:
            msg = (
                "Required upstream artefacts are missing from intake: "
                f"{missing}. This agent does NOT fabricate latents or state "
                "assignments; rerun the VAE + trajectory stages and re-supply."
            )
            logger.warning("CellStateTrajectoryAgent: SKIPPED -- %s", msg)
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=Verdict.SKIPPED,
                score=0.0,
                criteria_results={"intake": {"missing_keys": missing}},
                evidence={"reason": "missing_upstream_outputs"},
                failure_modes=[msg],
                required_changes=[
                    f"Provide all of {list(self.REQUIRED_INTAKE_KEYS)} in intake "
                    "before re-running CellStateTrajectoryAgent."
                ],
            )

        latents = np.asarray(intake["vae_latents"])
        states = np.asarray(intake["state_assignments"])
        metadata = list(intake["cell_metadata"])
        program_scores = dict(intake["program_scores"])
        transitions = intake.get("trajectory_transitions")

        criteria_results: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # 1. Lineage consistency: every cell with the same lineage should
        # cluster predominantly into a small number of states.
        lineage_check = self._check_lineage_consistency(states, metadata)
        criteria_results["lineage_consistency"] = lineage_check
        if not lineage_check["passed"]:
            failure_modes.append(
                "State assignments violate lineage constraints: cells of the "
                "same lineage are scattered across many states."
            )
            required_changes.append(
                "Add a lineage-aware regularizer to the VAE or post-hoc merge "
                "states that split a single lineage."
            )

        # 2. Batch robustness: state definitions should be stable across batches
        batch_check = self._check_batch_robustness(
            states, metadata, intake.get("batch_split_states")
        )
        criteria_results["batch_robustness"] = batch_check
        if not batch_check["passed"]:
            failure_modes.append(
                "State definitions are not robust across batches; clusters may "
                "be technical artefacts rather than biological states."
            )
            required_changes.append(
                "Run per-batch re-fits and report Adjusted Rand Index between "
                "batch-wise state assignments before claiming biological states."
            )

        # 3. Program coherence: each state should be enriched for at least one
        # known transcriptional program.
        program_check = self._check_program_enrichment(states, program_scores)
        criteria_results["program_enrichment"] = program_check
        if not program_check["passed"]:
            failure_modes.append(
                "One or more states have no significant enrichment for any "
                "known hematologic / MM transcriptional program."
            )
            required_changes.append(
                "Either annotate the unenriched states with a candidate program "
                "(supported by a literature reference) or merge them as "
                "statistical artefacts."
            )

        # 4. Transition plausibility (only if transitions provided)
        if transitions is not None:
            trans_check = self._check_transition_plausibility(np.asarray(transitions))
            criteria_results["transition_plausibility"] = trans_check
            if not trans_check["passed"]:
                failure_modes.append(
                    "Transition matrix violates basic stochasticity / "
                    "irreducibility constraints."
                )
                required_changes.append(
                    "Re-estimate transitions on a longer time window or "
                    "regularize toward row-stochastic, irreducible matrices."
                )

        passed = sum(1 for v in criteria_results.values() if v.get("passed"))
        total = len(criteria_results)
        score = passed / total if total else 0.0

        if score == 1.0:
            verdict = Verdict.PASS
        elif score >= 0.5:
            verdict = Verdict.CONDITIONAL
        else:
            verdict = Verdict.FAIL

        evidence = {
            "n_cells": int(latents.shape[0]) if latents.ndim >= 1 else 0,
            "n_states": int(np.unique(states).size),
            "programs_evaluated": list(program_scores.keys()),
        }

        finding = EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )
        logger.info(
            "CellStateTrajectoryAgent: verdict=%s score=%.2f n_states=%d",
            verdict, score, evidence["n_states"],
        )
        return finding

    # ------------------------------------------------------------------
    # Sub-checks
    # ------------------------------------------------------------------

    def _check_lineage_consistency(
        self, states: np.ndarray, metadata: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Each lineage should be concentrated in a small number of states."""
        if len(metadata) != states.shape[0]:
            return {
                "passed": False,
                "reason": "metadata length does not match state assignments",
            }
        lineages = [m.get("lineage", "unknown") for m in metadata]
        unique_lineages = set(lineages)
        max_states_per_lineage = max(1, int(np.unique(states).size * 0.5))
        violations = []
        purity_scores: dict[str, float] = {}
        for lin in unique_lineages:
            mask = np.array([l == lin for l in lineages])
            if mask.sum() == 0:
                continue
            sub_states = states[mask]
            unique_sub, counts = np.unique(sub_states, return_counts=True)
            purity = float(counts.max() / counts.sum())
            purity_scores[str(lin)] = purity
            if unique_sub.size > max_states_per_lineage:
                violations.append(str(lin))
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "purity_per_lineage": purity_scores,
            "max_states_per_lineage_threshold": max_states_per_lineage,
        }

    def _check_batch_robustness(
        self,
        states: np.ndarray,
        metadata: list[dict[str, Any]],
        batch_split_states: Any | None,
    ) -> dict[str, Any]:
        """Compare per-batch state assignments via a label-overlap proxy."""
        if batch_split_states is None:
            # Without per-batch re-fits we still report batch composition;
            # passing requires evidence of robustness, so we mark CONDITIONAL.
            batches = [m.get("batch", "unknown") for m in metadata]
            unique_batches = set(batches)
            return {
                "passed": False,
                "reason": "no per-batch re-fit provided; cannot verify robustness",
                "n_batches_observed": len(unique_batches),
            }

        # Compute mean overlap of joint vs per-batch assignments.
        overlaps: dict[str, float] = {}
        for batch_id, per_batch in batch_split_states.items():
            per_batch = np.asarray(per_batch)
            mask = np.array([m.get("batch", "unknown") == batch_id for m in metadata])
            if mask.sum() == 0 or per_batch.shape[0] != mask.sum():
                continue
            joint_sub = states[mask]
            # Label-agnostic overlap: pair frequency
            n = joint_sub.shape[0]
            same_joint = (joint_sub[:, None] == joint_sub[None, :])
            same_batch = (per_batch[:, None] == per_batch[None, :])
            agree = float(((same_joint & same_batch).sum() - n) / max(1, n * (n - 1)))
            overlaps[str(batch_id)] = agree
        mean_overlap = float(np.mean(list(overlaps.values()))) if overlaps else 0.0
        return {
            "passed": mean_overlap >= 0.6,
            "mean_pair_agreement": mean_overlap,
            "per_batch_agreement": overlaps,
            "threshold": 0.6,
        }

    def _check_program_enrichment(
        self,
        states: np.ndarray,
        program_scores: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Each state should have at least one known program above its mean."""
        unknown_programs = [
            p for p in program_scores.keys() if p not in EXPECTED_PROGRAMS
        ]
        unique_states = np.unique(states)
        per_state: dict[int, list[str]] = {}
        unenriched = []
        for s in unique_states:
            mask = states == s
            enriched_here: list[str] = []
            for prog, scores in program_scores.items():
                scores = np.asarray(scores)
                if scores.shape[0] != states.shape[0]:
                    continue
                if scores[mask].mean() > scores.mean():
                    enriched_here.append(prog)
            per_state[int(s)] = enriched_here
            if not enriched_here:
                unenriched.append(int(s))
        return {
            "passed": len(unenriched) == 0,
            "enriched_programs_per_state": per_state,
            "unenriched_states": unenriched,
            "unrecognized_program_labels": unknown_programs,
            "expected_programs": list(EXPECTED_PROGRAMS),
        }

    def _check_transition_plausibility(self, T: np.ndarray) -> dict[str, Any]:
        """Sanity-check transition matrix shape, stochasticity, irreducibility."""
        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            return {"passed": False, "reason": "transitions must be square 2D"}
        row_sums = T.sum(axis=1)
        stochastic = bool(np.allclose(row_sums, 1.0, atol=1e-3))
        non_negative = bool((T >= 0).all())
        # Irreducibility proxy: every state reachable from every other in
        # at most n_states power steps.
        n = T.shape[0]
        reach = (T > 0).astype(np.int64)
        powered = reach.copy()
        for _ in range(n):
            powered = (powered @ reach > 0).astype(np.int64)
        irreducible = bool(powered.all())
        return {
            "passed": stochastic and non_negative and irreducible,
            "row_stochastic": stochastic,
            "non_negative": non_negative,
            "irreducible": irreducible,
        }

    # ------------------------------------------------------------------
    # BaseAgent compatibility shim
    # ------------------------------------------------------------------

    async def execute(self, inputs, config):  # type: ignore[override]
        finding = await self.assess(inputs, config)
        return self._make_result(
            AgentState.COMPLETED,
            output=finding.__dict__ if hasattr(finding, "__dict__") else finding,
            metadata={"verdict": str(finding.verdict), "score": finding.score, "finding": finding},
        )
