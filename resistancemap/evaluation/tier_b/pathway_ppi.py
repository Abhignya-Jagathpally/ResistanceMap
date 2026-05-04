"""Tier B: Pathway / PPI Reasoning agent.

Audits whether attributed pathways are *graph-grounded* — i.e., walks /
neighborhoods on the PPI graph and protein complexes — versus *post-hoc
gene lists* assembled after the fact. Uses
``resistancemap.evaluation.metrics.pathway_stability`` to compute
stability under edge dropout and held-out pathway generalization.

Like the cell-state agent, this agent never fabricates inputs: if upstream
attribution callables / arrays are missing it returns SKIPPED.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from resistancemap.agents.base import AgentState
from resistancemap.config import ResistanceMapConfig
from resistancemap.evaluation.base import EvalAgent, EvalFinding, Verdict
from resistancemap.evaluation.metrics.pathway_stability import (
    attribution_overlap,
    edge_dropout_stability,
    pathway_holdout_generalization,
)

logger = logging.getLogger(__name__)


class PathwayPPIReasoningAgent(EvalAgent):
    """Audit pathway attributions for graph-grounded structure and stability.

    Required intake keys (any missing -> SKIPPED, never fabricated):

    - ``ppi_edges``       : np.ndarray of shape (E, 2) or (E, 3); edges over
                            the PPI used by the trained model
    - ``attribute_fn``    : callable taking a (sub-)edge array and returning
                            an attribution dict / 1D array, used to probe
                            the model under graph perturbations
    - ``baseline_attribution``: dict[gene_id, score] or 1D ndarray, the
                                attribution under the unperturbed PPI
    - ``alternative_attribution``: dict[gene_id, score] from a re-run with
                                   a different PPI build (e.g., a 2nd
                                   STRING release)
    - ``heldout_pathway_ids``: iterable of pathway ids withheld from training
    - ``heldout_predictions`` : 1D ndarray of model predictions on those
                                held-out pathways
    - ``heldout_model_fn``    : callable taking pathway ids -> 1D scores

    Optional:

    - ``pathway_walk_evidence``: dict with at least ``walks_traced`` (int)
      and ``walks_total`` (int), evidence that pathways were derived from
      actual graph walks rather than post-hoc gene lists.
    """

    tier = "B"

    REQUIRED_INTAKE_KEYS: tuple[str, ...] = (
        "ppi_edges",
        "attribute_fn",
        "baseline_attribution",
        "alternative_attribution",
        "heldout_pathway_ids",
        "heldout_predictions",
        "heldout_model_fn",
    )

    def __init__(self) -> None:
        super().__init__(
            name="pathway_ppi_reasoning",
            dependencies=[],
        )

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(inputs, dict):
            return False, "intake must be a dict"
        return True, ""

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        """Run pathway / PPI graph-grounding audit."""
        logger.info(
            "PathwayPPIReasoningAgent: starting pathway grounding audit "
            "(ppi_proteins=%d, ppi_confidence=%.2f)",
            config.protein_net.ppi_proteins,
            config.data.ppi_confidence,
        )

        missing = [k for k in self.REQUIRED_INTAKE_KEYS if k not in intake]
        if missing:
            msg = (
                f"Required upstream artefacts missing: {missing}. This agent "
                "does NOT fabricate PPI edges, attributions, or heldout sets; "
                "rerun the protein-net stage and re-supply."
            )
            logger.warning("PathwayPPIReasoningAgent: SKIPPED -- %s", msg)
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=Verdict.SKIPPED,
                score=0.0,
                criteria_results={"intake": {"missing_keys": missing}},
                evidence={
                    "reason": "missing_upstream_outputs",
                    "config_ppi_proteins": config.protein_net.ppi_proteins,
                    "config_ppi_confidence": config.data.ppi_confidence,
                },
                failure_modes=[msg],
                required_changes=[
                    f"Provide all of {list(self.REQUIRED_INTAKE_KEYS)} in intake "
                    "before re-running PathwayPPIReasoningAgent."
                ],
            )

        criteria_results: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        ppi_edges = np.asarray(intake["ppi_edges"])
        attribute_fn = intake["attribute_fn"]
        baseline = intake["baseline_attribution"]
        alternative = intake["alternative_attribution"]
        heldout_ids = intake["heldout_pathway_ids"]
        heldout_preds = np.asarray(intake["heldout_predictions"], dtype=np.float64)
        heldout_model_fn = intake["heldout_model_fn"]

        # 1. Graph-walk provenance: pathways must originate from graph walks,
        # not gene lists pasted after the fact.
        walk_evidence = intake.get("pathway_walk_evidence")
        criteria_results["graph_walk_provenance"] = self._check_walk_provenance(
            walk_evidence
        )
        if not criteria_results["graph_walk_provenance"]["passed"]:
            failure_modes.append(
                "Reported pathways are not traceable to actual walks on the "
                "PPI; they may be post-hoc gene lists rather than graph-"
                "grounded reasoning."
            )
            required_changes.append(
                "Record the provenance of every reported pathway: which graph "
                "walks (or message-passing rounds) produced it, and report "
                "walks_traced / walks_total."
            )

        # 2. Cross-PPI-build consistency: attributions should overlap across
        # alternative PPI builds (e.g., STRING v11 vs v12).
        cross_overlap = attribution_overlap(baseline, alternative, top_k=50)
        criteria_results["cross_ppi_consistency"] = {
            **cross_overlap,
            "threshold_jaccard": 0.4,
            "passed": (
                not np.isnan(cross_overlap["jaccard"])
                and cross_overlap["jaccard"] >= 0.4
            ),
        }
        if not criteria_results["cross_ppi_consistency"]["passed"]:
            failure_modes.append(
                "Attributions are inconsistent across alternative PPI builds; "
                "the pathway story changes when the underlying graph changes."
            )
            required_changes.append(
                "Re-run pathway attribution against multiple PPI releases and "
                "report only attributions that survive Jaccard >= 0.4 across "
                "builds."
            )

        # 3. Edge-dropout stability via the metrics module.
        try:
            stability = edge_dropout_stability(
                attribute_fn=attribute_fn,
                ppi_edges=ppi_edges,
                dropout_rates=(0.05, 0.1, 0.2),
                n_trials=10,
                seed=0,
            )
            stability_passed = self._stability_passed(stability)
        except Exception as exc:  # never fabricate; record the failure.
            logger.exception("edge_dropout_stability failed")
            stability = {"error": str(exc)}
            stability_passed = False
        criteria_results["edge_dropout_stability"] = {
            **(stability if isinstance(stability, dict) else {}),
            "passed": stability_passed,
        }
        if not stability_passed:
            failure_modes.append(
                "Pathway attributions degrade rapidly under PPI edge dropout; "
                "reasoning may be brittle to graph noise."
            )
            required_changes.append(
                "Either ensemble attributions over edge-dropout perturbations "
                "or report only the subset of attributions whose Jaccard CI "
                "lower bound exceeds 0.5 at 10% dropout."
            )

        # 4. Held-out pathway generalization via the metrics module.
        try:
            corr = pathway_holdout_generalization(
                model_fn=heldout_model_fn,
                heldout_pathway_ids=heldout_ids,
                predictions=heldout_preds,
            )
            holdout_passed = bool(corr >= 0.3) and not np.isnan(corr)
        except Exception as exc:
            logger.exception("pathway_holdout_generalization failed")
            corr = float("nan")
            holdout_passed = False
        criteria_results["heldout_generalization"] = {
            "pearson_corr": float(corr),
            "threshold": 0.3,
            "passed": holdout_passed,
        }
        if not holdout_passed:
            failure_modes.append(
                "Held-out pathways do not generalize: the model cannot reproduce "
                "its own pathway scores when those pathways are excluded from "
                "training."
            )
            required_changes.append(
                "Either expand training pathway coverage or report pathway-level "
                "predictions only on pathways that generalize (Pearson >= 0.3)."
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
            "ppi_proteins": int(config.protein_net.ppi_proteins),
            "ppi_confidence": float(config.data.ppi_confidence),
            "n_edges_provided": int(ppi_edges.shape[0]),
            "n_heldout_pathways": int(heldout_preds.shape[0]),
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
            "PathwayPPIReasoningAgent: verdict=%s score=%.2f passed=%d/%d",
            verdict, score, passed, total,
        )
        return finding

    # ------------------------------------------------------------------
    # Sub-checks
    # ------------------------------------------------------------------

    def _check_walk_provenance(self, walk_evidence: Any) -> dict[str, Any]:
        if not isinstance(walk_evidence, dict):
            return {
                "passed": False,
                "reason": "no walk provenance provided",
            }
        traced = int(walk_evidence.get("walks_traced", 0))
        total = int(walk_evidence.get("walks_total", 0))
        ratio = traced / total if total > 0 else 0.0
        return {
            "passed": ratio >= 0.8 and total > 0,
            "walks_traced": traced,
            "walks_total": total,
            "trace_ratio": ratio,
            "threshold": 0.8,
        }

    def _stability_passed(self, stability: dict[str, Any]) -> bool:
        """Pass if Jaccard CI lower bound stays >= 0.5 at 10% dropout."""
        key = f"{0.1:.3f}"
        if key not in stability:
            return False
        sub = stability[key]
        if not isinstance(sub, dict):
            return False
        jacc = sub.get("jaccard", {})
        if not isinstance(jacc, dict):
            return False
        ci_low = jacc.get("ci_low", float("nan"))
        return bool(ci_low >= 0.5)

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
