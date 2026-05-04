"""Tier A bias / fairness / shift agent.

Audits the configured cohorts for selection bias, ancestry representation,
SES proxies, and train -> deploy domain shift. The actual numerical
computations (subgroup ECE, MMD / energy distance shift scores) are
delegated to a forthcoming ``resistancemap.evaluation.metrics.shift`` module
that another agent is producing in parallel; this agent records explicit
TODO evidence stubs and ``required_changes`` so the gap is visible without
forcing a hard dependency that does not yet exist.
"""

from __future__ import annotations

import logging
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from resistancemap.config import ResistanceMapConfig
from resistancemap.evaluation.base import EvalAgent, EvalFinding, EvalVerdict

logger = logging.getLogger(__name__)


class BiasFairnessShiftAgent(EvalAgent):
    """Audits selection bias, fairness, and train/deploy shift.

    The agent does not run shift detection itself; it calls into the
    optional ``resistancemap.evaluation.metrics.shift`` module if it exists,
    and otherwise records the call as a TODO with concrete remediation.
    """

    tier: str = "A"

    def __init__(self) -> None:
        """Initialise the agent with no upstream dependencies."""
        super().__init__(name="eval_bias_fairness_shift", dependencies=[])

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        """Run the bias / fairness / shift audit.

        Args:
            intake: Unused (Tier A root).
            config: ResistanceMap configuration.

        Returns:
            An :class:`EvalFinding`. Returns SKIPPED if the cohort manifests
            required to compute subgroup statistics are absent.
        """
        del intake

        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # We require *some* manifest to even attempt subgroup statistics.
        # We never invent demographic columns.
        manifest_path = (
            Path(config.data.mmrf_commpass_dir) / "patient_manifest.csv"
        )
        ancestry_doc = (
            Path(config.data.mmrf_commpass_dir) / "ancestry_distribution.md"
        )

        # ---- 1. Selection bias audit ------------------------------------
        criteria_results["selection_bias_audit"] = (
            "documented" if ancestry_doc.exists() else "undocumented"
        )
        evidence["selection_bias_doc"] = str(ancestry_doc)
        if not ancestry_doc.exists():
            failure_modes.append(
                "No selection-bias / ancestry distribution memo present for "
                "the configured cohorts. Public datasets (CCLE, GDSC, MMRF) "
                "all have known enrolment skews that must be acknowledged."
            )
            required_changes.append(
                f"Author a selection-bias memo at {ancestry_doc} covering "
                "CCLE cell-line ancestry, GDSC sourcing, and MMRF "
                "enrolment criteria."
            )

        # ---- 2. Ancestry representation ---------------------------------
        criteria_results["ancestry_representation"] = (
            "documented" if manifest_path.exists() else "todo"
        )
        evidence["patient_manifest"] = str(manifest_path)
        if not manifest_path.exists():
            failure_modes.append(
                "Per-patient ancestry / demographic columns are not "
                "available; subgroup calibration cannot be computed."
            )
            required_changes.append(
                f"Provide a patient manifest at {manifest_path} including "
                "self-reported ancestry and a documented missingness rate."
            )

        # ---- 3. SES proxies ---------------------------------------------
        criteria_results["ses_proxy_handling"] = "todo"
        evidence["ses_proxy_handling"] = {
            "status": "TODO",
            "note": (
                "If insurance / geography / income proxies are present, "
                "their handling must be documented; if absent, that absence "
                "must itself be documented."
            ),
        }
        required_changes.append(
            "Document SES proxy availability and handling (or explicit "
            "absence) for every cohort."
        )

        # ---- 4. Train/deploy shift --------------------------------------
        # Defer to metrics/shift.py if it exists; otherwise TODO stub.
        shift_status, shift_evidence = self._call_shift_metric(config)
        criteria_results["train_deploy_shift"] = shift_status
        evidence["train_deploy_shift"] = shift_evidence
        if shift_status == "todo":
            required_changes.append(
                "Implement resistancemap.evaluation.metrics.shift.compute_shift "
                "and re-run; this agent will pick it up automatically."
            )

        # ---- 5. Subgroup calibration hook -------------------------------
        criteria_results["subgroup_calibration"] = "todo"
        evidence["subgroup_calibration"] = {
            "status": "TODO",
            "note": (
                "Defer to a downstream calibration agent (Tier C). This "
                "agent only declares the requirement; it does not compute "
                "subgroup ECE itself."
            ),
        }
        required_changes.append(
            "Tier C forecasting agent must report ECE per ancestry subgroup, "
            "not just overall."
        )

        # ---- Verdict roll-up --------------------------------------------
        # We never PASS without manifests; the best we can return is
        # CONDITIONAL with explicit required_changes.
        if not manifest_path.exists() and not ancestry_doc.exists():
            verdict = EvalVerdict.SKIPPED
            failure_modes.insert(
                0,
                "Both ancestry memo and patient manifest are absent; "
                "agent skipped to avoid fabricating subgroup tables.",
            )
        else:
            verdict = EvalVerdict.CONDITIONAL

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=0.0 if verdict == EvalVerdict.SKIPPED else 0.25,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _call_shift_metric(
        config: ResistanceMapConfig,
    ) -> tuple[str, dict[str, Any]]:
        """Try to import and call the shift metric module.

        Returns:
            ``(status, evidence)``. ``status`` is ``"todo"`` when the
            module does not yet exist (the common case during scaffolding),
            ``"available"`` when the module is importable, and ``"error"``
            when the call raised. We never fabricate a numeric shift value.
        """
        try:
            spec = importlib_util.find_spec(
                "resistancemap.evaluation.metrics.shift"
            )
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            return "todo", {
                "status": "TODO",
                "note": (
                    "resistancemap.evaluation.metrics.shift not yet on path; "
                    "another agent is implementing it. Stub recorded."
                ),
            }
        try:
            module = importlib_util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            compute = getattr(module, "compute_shift", None)
            if compute is None:
                return "todo", {
                    "status": "TODO",
                    "note": (
                        "Module exists but compute_shift() is not defined."
                    ),
                }
            # We do not fabricate inputs. The actual call is the responsibility
            # of a downstream agent that has the loaded matrices.
            return "available", {
                "status": "available",
                "module": "resistancemap.evaluation.metrics.shift",
                "callable": "compute_shift",
                "note": (
                    "Module is importable; downstream agents should call "
                    "compute_shift() with real train and deploy matrices."
                ),
                "configured_drugs": list(config.data.target_drugs),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("BiasFairnessShiftAgent: shift import failed")
            return "error", {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
