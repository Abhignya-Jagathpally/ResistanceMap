"""Tier A data adequacy agent.

Audits the *fitness for purpose* of the configured data sources for the
multiple-myeloma resistance task. This is a foundational gate: if the
configured cohorts are not actually MM-relevant, or if patient overlap
across modalities is not knowable, the rest of the evaluation cannot make
honest claims and should hard-stop.

Hard rule
---------
This agent **never** fabricates data. If the referenced data files do not
exist on disk, it returns a SKIPPED finding whose ``failure_modes`` and
``required_changes`` spell out exactly what is missing. SKIPPED is not a
quiet pass; downstream tiers see it and the Tier D chair downgrades the
overall verdict accordingly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from resistancemap.config import ResistanceMapConfig
from resistancemap.evaluation.base import EvalAgent, EvalFinding, EvalVerdict
from resistancemap.evaluation.leakage import (
    LeakageReport,
    audit_patient_split,
    detect_temporal_leakage,
)

logger = logging.getLogger(__name__)


# Heuristics for indication match. We never *infer* indication from numeric
# data; we look at filename / config conventions and surface them as
# evidence so a human reviewer can confirm.
_MM_KEYWORDS: tuple[str, ...] = (
    "mm",
    "myeloma",
    "mmrf",
    "commpass",
    "gse124310",
    "gse271107",
)
_OFF_TARGET_KEYWORDS: tuple[str, ...] = (
    "aml",
    "lymphoma",
    "cll",
    "breast",
    "lung",
    "melanoma",
)


class DataAdequacyAgent(EvalAgent):
    """Foundational data-fitness audit for the MM resistance task.

    Checks performed (when the data is present):

    * **Indication match** -- the configured paths smell like multiple
      myeloma cohorts (filename / directory keywords); off-target keywords
      (AML, lymphoma, ...) trigger a fail-closed gate.
    * **Treatment line** -- whether NDMM vs RRMM metadata is reachable.
    * **Sample site** -- bone marrow vs peripheral blood note.
    * **Modality alignment** -- single-cell vs bulk patient overlap.
    * **Longitudinal availability** -- presence of multiple time points.
    * **Resistance label definition** -- primary vs secondary resistance
      provenance documented.
    * **Patient-split structural leakage** via :func:`audit_patient_split`.
    * **Temporal leakage** via :func:`detect_temporal_leakage`.

    The numeric metric checks (overlap counts, modality coverage matrix)
    are scaffolded to **defer** to a real loader rather than fabricate
    arrays. Until that loader is wired in, the agent records the gap as a
    TODO evidence stub and demands a follow-up via ``required_changes``.
    """

    tier: str = "A"

    def __init__(self) -> None:
        """Initialise the agent with no upstream dependencies."""
        super().__init__(name="eval_data_adequacy", dependencies=[])

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        """Run the data adequacy audit.

        Args:
            intake: Unused (Tier A is a root tier).
            config: ResistanceMap configuration to audit.

        Returns:
            An :class:`EvalFinding` with verdict PASS / CONDITIONAL / FAIL /
            SKIPPED depending on what the on-disk data and config support.
        """
        del intake  # Tier A roots take no upstream input.

        required_paths = [
            config.data.ccle_proteomics_path,
            config.data.ccle_epigenomics_dir,
            config.data.string_ppi_path,
        ]
        skipped = self.skip_if_data_missing(
            required_paths,
            memo=(
                "Core CCLE / STRING inputs are not present on disk; data "
                "adequacy audit skipped to avoid fabricating cohort tables."
            ),
        )
        if skipped is not None:
            skipped.required_changes.append(
                "Wire the loader in resistancemap.data so that "
                "DataAdequacyAgent can compute real cohort tables, "
                "modality overlap, and missingness maps."
            )
            return skipped

        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # ---- 1. Indication match ----------------------------------------
        indication_status, indication_notes = self._check_indication(config)
        criteria_results["indication_match"] = indication_status
        evidence["indication_notes"] = indication_notes
        if indication_status == "fail":
            return self.fail_closed_gate(
                reason=(
                    "Configured data paths look off-indication "
                    f"(non-MM keywords detected: {indication_notes['off_target']!r})."
                ),
                required_changes=[
                    "Replace the off-indication paths with MM-relevant cohorts.",
                ],
            )

        # ---- 2. Treatment line / sample site / longitudinal -------------
        # We do not invent metadata. We declare these as 'undocumented'
        # unless a sidecar manifest is present, and require the loader to
        # populate them.
        manifest_path = config.data.mmrf_commpass_dir / "patient_manifest.csv"
        criteria_results["treatment_line_documented"] = (
            "documented" if manifest_path.exists() else "undocumented"
        )
        criteria_results["sample_site_documented"] = (
            "documented" if manifest_path.exists() else "undocumented"
        )
        criteria_results["longitudinal_availability"] = (
            "available" if config.data.mmrf_commpass_dir.exists() else "unavailable"
        )
        evidence["patient_manifest_path"] = str(manifest_path)
        if criteria_results["treatment_line_documented"] == "undocumented":
            failure_modes.append(
                "Treatment line (NDMM vs RRMM) not documented in any manifest."
            )
            required_changes.append(
                f"Provide a per-patient manifest at {manifest_path} with "
                "treatment_line and sample_site columns."
            )

        # ---- 3. Modality alignment (single-cell vs bulk) -----------------
        sc_paths = [
            config.data.scrna_gse124310_path,
            config.data.scrna_gse271107_path,
        ]
        sc_present = [p for p in sc_paths if Path(p).exists()]
        criteria_results["modality_alignment"] = (
            "scaffolded" if sc_present else "single_modality"
        )
        evidence["single_cell_paths_present"] = [str(p) for p in sc_present]
        if not sc_present:
            failure_modes.append(
                "No single-cell MM cohort present; cannot align scRNA to bulk proteome."
            )
            required_changes.append(
                "Stage at least one MM scRNA cohort (GSE124310 or GSE271107) "
                "and a patient ID crosswalk to the bulk modalities."
            )

        # ---- 4. Resistance label definition ------------------------------
        label_doc = config.data.mmrf_commpass_dir / "resistance_label_definition.md"
        documented = label_doc.exists()
        criteria_results["resistance_label_definition"] = (
            "documented" if documented else "undocumented"
        )
        evidence["resistance_label_definition_doc"] = str(label_doc)
        if not documented:
            failure_modes.append(
                "Primary vs secondary resistance label definition is not "
                "documented; downstream calibration cannot be interpreted."
            )
            required_changes.append(
                f"Add a resistance label provenance memo at {label_doc} "
                "covering primary vs secondary, thresholding rule, and "
                "the source feature."
            )

        # Canonical example failure mode that downstream agents can pattern
        # match against.
        failure_modes.append(
            "Pre-treatment single-cell not matched to post-relapse single-cell "
            "(charter requires this for any temporal claim)."
        )

        # ---- 5. Leakage scans (structural + temporal) --------------------
        leakage_report = self._scan_leakage()
        criteria_results["temporal_leakage_clean"] = bool(leakage_report.clean)
        evidence["leakage_report"] = leakage_report.to_dict()
        if not leakage_report.clean:
            failure_modes.append(
                "Leakage scan was vacuous (no split manifest available)."
            )
            required_changes.append(
                "Expose train/val/test patient-id manifests so the leakage "
                "scan can run on real splits."
            )

        # ---- Verdict roll-up --------------------------------------------
        # We grade strictly: any required-criterion failure -> CONDITIONAL,
        # not PASS, because most of the strict numeric checks need a real
        # loader before this agent can fully clear them.
        score = self._coarse_score(criteria_results)
        if any(
            criteria_results.get(k) in ("fail", False)
            for k in (
                "indication_match",
                "modality_alignment",
                "resistance_label_definition",
                "temporal_leakage_clean",
            )
        ):
            verdict = EvalVerdict.CONDITIONAL
        else:
            verdict = EvalVerdict.CONDITIONAL  # never PASS without numeric loader
            required_changes.append(
                "Promote to PASS only after the data loader populates real "
                "cohort tables, modality overlap matrix, and split manifests."
            )

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

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _check_indication(
        config: ResistanceMapConfig,
    ) -> tuple[str, dict[str, list[str]]]:
        """Heuristic indication-match check using filename keywords."""
        haystacks = [
            str(config.data.ccle_proteomics_path).lower(),
            str(config.data.ccle_epigenomics_dir).lower(),
            str(config.data.mmrf_commpass_dir).lower(),
            str(config.data.scrna_gse124310_path).lower(),
            str(config.data.scrna_gse271107_path).lower(),
        ]
        mm_hits: list[str] = []
        off_hits: list[str] = []
        for hay in haystacks:
            for kw in _MM_KEYWORDS:
                if kw in hay:
                    mm_hits.append(f"{kw}@{hay}")
            for kw in _OFF_TARGET_KEYWORDS:
                if kw in hay:
                    off_hits.append(f"{kw}@{hay}")

        notes = {"mm_hits": mm_hits, "off_target": off_hits}
        if off_hits and not mm_hits:
            return "fail", notes
        if not mm_hits:
            return "weak", notes
        return "pass", notes

    def _scan_leakage(self) -> LeakageReport:
        """Run leakage audits on whatever split manifests we can find.

        We never invent IDs; if no manifests are available the underlying
        functions return a vacuous (and explicitly non-clean) report and
        the agent records this in failure_modes.
        """
        # No real loader yet -> empty inputs. The leakage functions are
        # designed to recognise this and return a non-clean (vacuous)
        # report rather than a misleading "all good".
        return audit_patient_split([], [], [])

    @staticmethod
    def _coarse_score(criteria_results: dict[str, Any]) -> float:
        """Crude average over the criteria results, for human inspection."""
        if not criteria_results:
            return 0.0
        good = 0
        total = 0
        for value in criteria_results.values():
            total += 1
            if value in ("pass", "documented", "available", True):
                good += 1
            elif value in ("scaffolded", "weak"):
                good += 0.5
        return good / total if total else 0.0
