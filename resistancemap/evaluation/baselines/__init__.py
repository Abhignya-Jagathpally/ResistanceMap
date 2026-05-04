"""Baseline models for ResistanceMap evaluation.

These baselines are real, runnable models. They accept REAL data from the
caller and raise informative errors if no real data is given. None of them
fabricate data, return zero placeholders, or return random outputs.

Available baselines:

- :class:`ClinicalOnlyBaseline` -- penalized logistic regression on clinical
  features (age, ISS stage, cytogenetic risk).
- :class:`TranscriptomeOnlyBaseline` -- PCA via numpy SVD followed by a
  linear classifier/regressor.
- :class:`LastObservationBaseline` -- per-patient last-observation carry
  forward; requires longitudinal input.
- :class:`SimpleSurvivalBaseline` -- numpy Kaplan-Meier (or Cox if scipy is
  available) stratified by a single covariate.
"""

from __future__ import annotations

from resistancemap.evaluation.baselines.base import Baseline
from resistancemap.evaluation.baselines.clinical_only import ClinicalOnlyBaseline
from resistancemap.evaluation.baselines.transcriptome_only import (
    TranscriptomeOnlyBaseline,
)
from resistancemap.evaluation.baselines.last_observation import LastObservationBaseline
from resistancemap.evaluation.baselines.simple_survival import SimpleSurvivalBaseline

__all__ = [
    "Baseline",
    "ClinicalOnlyBaseline",
    "TranscriptomeOnlyBaseline",
    "LastObservationBaseline",
    "SimpleSurvivalBaseline",
]
