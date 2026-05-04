"""Pure-numpy evaluation metrics for ResistanceMap.

All metrics in this package are pure functions: numpy arrays in, float/dict
out, no side effects, explicit input validation. They never fabricate data;
callers must supply real arrays. ``scipy`` is optional.
"""

from __future__ import annotations

from resistancemap.evaluation.metrics.calibration import (
    brier_score_multiclass,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)
from resistancemap.evaluation.metrics.forecasting import (
    crps_ensemble,
    mean_absolute_calibration_error,
    pinball_loss,
)
from resistancemap.evaluation.metrics.pathway_stability import (
    attribution_overlap,
    edge_dropout_stability,
    pathway_holdout_generalization,
)
from resistancemap.evaluation.metrics.shift import (
    kolmogorov_smirnov_shift,
    maximum_mean_discrepancy,
    population_stability_index,
    shift_alarm,
)
from resistancemap.evaluation.metrics.survival import (
    concordance_index,
    integrated_brier_score,
    time_dependent_auc,
)

__all__ = [
    # calibration
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_curve",
    "brier_score_multiclass",
    # survival
    "integrated_brier_score",
    "time_dependent_auc",
    "concordance_index",
    # forecasting
    "pinball_loss",
    "crps_ensemble",
    "mean_absolute_calibration_error",
    # pathway stability
    "attribution_overlap",
    "edge_dropout_stability",
    "pathway_holdout_generalization",
    # shift
    "maximum_mean_discrepancy",
    "kolmogorov_smirnov_shift",
    "population_stability_index",
    "shift_alarm",
]
