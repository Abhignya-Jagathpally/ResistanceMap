"""Tier C evaluation agents.

Tier C agents perform domain-specific evaluation of model outputs:
- ForecastingUncertaintyAgent: probabilistic forecast quality, calibration, uncertainty
- BaselineAdversaryAgent: pre-registered comparisons against baselines
- ClinicalTranslationSafetyAgent: actionability and clinical-safety review

These agents consume model outputs (predictions, intervals, evidence) but never
fabricate data themselves.
"""

from __future__ import annotations

from resistancemap.evaluation.tier_c.forecasting_uncertainty import (
    ForecastingUncertaintyAgent,
)
from resistancemap.evaluation.tier_c.baseline_adversary import BaselineAdversaryAgent
from resistancemap.evaluation.tier_c.clinical_safety import (
    ClinicalTranslationSafetyAgent,
)

__all__ = [
    "ForecastingUncertaintyAgent",
    "BaselineAdversaryAgent",
    "ClinicalTranslationSafetyAgent",
]
