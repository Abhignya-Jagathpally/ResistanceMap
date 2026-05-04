"""Tier B evaluation agents for ResistanceMap.

Tier B agents perform *scientific-coherence* checks: they audit the
architecture for identifiability gaps, evaluate biological coherence of
recovered cell states, and verify that pathway attributions are graph-
grounded rather than post-hoc gene lists.

None of these agents fabricate inputs. When upstream artefacts are
missing, the agent returns a SKIPPED finding listing the missing keys.
"""

from __future__ import annotations

from resistancemap.evaluation.tier_b.architecture_auditor import (
    ArchitectureAuditorAgent,
)
from resistancemap.evaluation.tier_b.cell_state_specialist import (
    CellStateTrajectoryAgent,
)
from resistancemap.evaluation.tier_b.pathway_ppi import PathwayPPIReasoningAgent

__all__ = [
    "ArchitectureAuditorAgent",
    "CellStateTrajectoryAgent",
    "PathwayPPIReasoningAgent",
]
