"""Tier A measurement-integration agent.

Audits *how* the multi-modal measurements are integrated: cell-type vs
state resolution, batch effects, cross-modality alignment (same cells vs
imputed bridges), and whether the PPI graph actually covers the surface
targets of the drugs the pipeline is supposed to predict against.

As with every Tier A agent, this one refuses to manufacture data. When the
underlying matrices are not on disk, it returns a SKIPPED finding pointing
at the missing artefacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from resistancemap.config import ResistanceMapConfig
from resistancemap.evaluation.base import EvalAgent, EvalFinding, EvalVerdict

logger = logging.getLogger(__name__)


# Drug -> canonical surface / molecular target(s) for MM. We use this only
# to *report coverage* against config.data.target_drugs; we never decide a
# drug's mechanism here, only check that the PPI graph contains the
# proteins the pipeline would need to attribute through.
_DRUG_SURFACE_TARGETS: dict[str, tuple[str, ...]] = {
    "Daratumumab": ("CD38",),
    "Isatuximab": ("CD38",),
    "Elotuzumab": ("SLAMF7",),
    "Bortezomib": ("PSMB5", "PSMB1"),
    "Carfilzomib": ("PSMB5",),
    "Ixazomib": ("PSMB5",),
    "Lenalidomide": ("CRBN", "IKZF1", "IKZF3"),
    "Pomalidomide": ("CRBN", "IKZF1", "IKZF3"),
    "Thalidomide": ("CRBN",),
    "Dexamethasone": ("NR3C1",),
    "Talquetamab": ("GPRC5D",),
    "Teclistamab": ("TNFRSF17",),  # BCMA
}


class MeasurementIntegrationAgent(EvalAgent):
    """Audits cross-modality alignment and PPI coverage of drug targets."""

    tier: str = "A"

    def __init__(self) -> None:
        """Initialise the agent with no upstream dependencies."""
        super().__init__(name="eval_measurement_integration", dependencies=[])

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        """Run the measurement-integration audit.

        Args:
            intake: Unused (Tier A root).
            config: ResistanceMap configuration.

        Returns:
            An :class:`EvalFinding`. The PPI coverage check is computable
            from the STRING file alone, so it can return a real number even
            when single-cell data is absent.
        """
        del intake

        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # ---- 1. Cell-type vs state resolution ----------------------------
        # We do not have a single-cell loader yet; record the gap explicitly.
        criteria_results["cell_type_resolution"] = "todo"
        evidence["cell_type_resolution"] = {
            "status": "TODO",
            "note": (
                "Pending a single-cell loader that exposes per-cell type / "
                "state annotations. Resolution must be declared and matched "
                "to the resistance label granularity."
            ),
        }
        required_changes.append(
            "Wire a single-cell loader and declare cell-type vs state "
            "granularity in the data manifest."
        )

        # ---- 2. Batch effects -------------------------------------------
        criteria_results["batch_effects_assessed"] = "todo"
        evidence["batch_effects_assessed"] = {
            "status": "TODO",
            "metric": "kBET / iLISI not yet computed",
            "note": (
                "Defer to the batch-mixing metric provided by an upstream "
                "preprocessing agent; record the result here once produced."
            ),
        }
        required_changes.append(
            "Compute kBET / iLISI on the harmonised matrices and record it "
            "as evidence under batch_effects_assessed."
        )

        # ---- 3. Cross-modality alignment --------------------------------
        criteria_results["cross_modality_alignment"] = "todo"
        evidence["cross_modality_alignment"] = {
            "status": "TODO",
            "note": (
                "Same-cell vs imputed-bridge alignment must be declared "
                "with an explicit error bound. Currently undeclared."
            ),
        }
        required_changes.append(
            "Declare whether modalities share cells (paired) or are bridged "
            "by imputation, and bound the alignment error."
        )

        # ---- 4. PPI graph version + coverage of drug targets ------------
        ppi_path = Path(config.data.string_ppi_path)
        if not ppi_path.exists():
            skipped = self.skip_if_data_missing(
                [ppi_path],
                memo=(
                    "STRING PPI file is missing; cannot check graph version "
                    "or drug-target coverage."
                ),
            )
            assert skipped is not None
            skipped.failure_modes.extend(failure_modes)
            skipped.required_changes.extend(required_changes)
            return skipped

        ppi_proteins, ppi_edges = self._scan_ppi(ppi_path)
        criteria_results["ppi_graph_version"] = (
            "documented" if ppi_proteins else "unparseable"
        )
        evidence["ppi_graph"] = {
            "path": str(ppi_path),
            "n_proteins_detected": ppi_proteins,
            "n_edges_detected": ppi_edges,
            "confidence_cutoff": config.data.ppi_confidence,
            "expected_n_proteins": config.protein_net.ppi_proteins,
            "expected_n_edges": config.protein_net.ppi_edges,
        }
        if ppi_proteins == 0:
            failure_modes.append(
                f"PPI file at {ppi_path} could not be parsed for protein count."
            )
            required_changes.append(
                "Verify STRING file format (expected: tab/space separated, "
                "two protein columns + score) and re-run."
            )

        coverage = self._drug_target_coverage(
            ppi_path, ppi_proteins, config.data.target_drugs
        )
        criteria_results["drug_target_coverage"] = coverage["fraction"]
        evidence["drug_target_coverage"] = coverage
        if coverage["fraction"] < 0.8:
            failure_modes.append(
                f"PPI graph covers only {coverage['fraction']:.0%} of the "
                "surface targets of configured drugs; pathway attribution "
                "claims are unwarranted for the uncovered drugs."
            )
            required_changes.append(
                "Either upgrade the PPI graph (newer STRING release / "
                "lower confidence cutoff) or restrict target_drugs to the "
                "drugs whose targets are present in the graph."
            )

        # ---- Verdict roll-up --------------------------------------------
        score = float(coverage["fraction"]) * 0.5  # only the one numeric criterion
        verdict = EvalVerdict.CONDITIONAL
        if coverage["fraction"] >= 0.8 and ppi_proteins > 0:
            verdict = EvalVerdict.CONDITIONAL  # still TODO on the other 3 criteria
        else:
            verdict = EvalVerdict.CONDITIONAL  # explicit gaps recorded above

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

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _scan_ppi(ppi_path: Path) -> tuple[int, int]:
        """Stream the PPI file once and return ``(n_proteins, n_edges)``.

        We avoid loading the whole graph into memory and avoid pandas. The
        scan is intentionally tolerant of common STRING / TSV layouts: any
        line whose first two whitespace-separated tokens look like protein
        identifiers contributes one edge.
        """
        proteins: set[str] = set()
        edges = 0
        try:
            with open(ppi_path) as fh:
                for i, line in enumerate(fh):
                    if i == 0 and not line[:1].isalnum():
                        continue  # likely header / comment
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    a, b = parts[0], parts[1]
                    if not a or not b or a == b:
                        continue
                    proteins.add(a)
                    proteins.add(b)
                    edges += 1
        except OSError as exc:
            logger.error("MeasurementIntegrationAgent: cannot read PPI: %s", exc)
            return 0, 0
        return len(proteins), edges

    @staticmethod
    def _drug_target_coverage(
        ppi_path: Path,
        ppi_protein_count: int,
        target_drugs: list[str],
    ) -> dict[str, Any]:
        """Compute fraction of configured drugs whose targets sit in the PPI.

        We use a streaming substring scan rather than building a Python set
        of all proteins (cheaper for very large STRING files). The
        ``ppi_protein_count`` is taken on trust from a previous scan and
        used only as a sanity flag.
        """
        wanted: dict[str, tuple[str, ...]] = {}
        unknown_drugs: list[str] = []
        for drug in target_drugs:
            if drug in _DRUG_SURFACE_TARGETS:
                wanted[drug] = _DRUG_SURFACE_TARGETS[drug]
            else:
                unknown_drugs.append(drug)

        all_targets = {tgt for tgts in wanted.values() for tgt in tgts}
        found: set[str] = set()
        if ppi_protein_count and all_targets:
            try:
                with open(ppi_path) as fh:
                    for line in fh:
                        for tgt in all_targets - found:
                            if tgt in line:
                                found.add(tgt)
                        if len(found) == len(all_targets):
                            break
            except OSError as exc:
                logger.error(
                    "drug_target_coverage: cannot read PPI: %s", exc
                )

        per_drug = {
            drug: {
                "targets": list(tgts),
                "covered": [t for t in tgts if t in found],
                "fully_covered": all(t in found for t in tgts),
            }
            for drug, tgts in wanted.items()
        }
        n_total = len(wanted)
        n_covered = sum(1 for d in per_drug.values() if d["fully_covered"])
        fraction = n_covered / n_total if n_total else 0.0

        return {
            "fraction": fraction,
            "n_drugs_total": n_total,
            "n_drugs_fully_covered": n_covered,
            "per_drug": per_drug,
            "drugs_with_unknown_targets": unknown_drugs,
        }
