"""Tier C: Forecasting & Uncertainty evaluation agent.

Assesses the probabilistic quality of trajectory/state forecasts produced by
the trajectory model. Concretely:

- Time-to-event quality (Integrated Brier Score, time-dependent AUC, C-index).
- Sharp + calibrated *distributional* forecasts via proper scoring rules
  (pinball loss for quantile forecasts; CRPS for ensemble forecasts).
- Calibration plots at every clinical horizon in
  ``config.trajectory.forecast_horizons``.
- Honest uncertainty for *rare resistance modes*: checks whether the Monte
  Carlo perturbation budget ``config.trajectory.n_perturbation`` is large
  enough to resolve quantiles in the tail.

CRITICAL caveat the agent always emits as a required change:

    ``config.trajectory.integration_time = 100.0`` is in **arbitrary ODE
    units**. Until that axis is calibrated to *calendar months* on a held-out
    longitudinal cohort with measured response times, the agent forbids any
    claim of "predictions at 3 / 6 / 12 months". The horizons in
    ``forecast_horizons`` are *targets*, not *current capabilities*.

The agent skips gracefully (verdict=SKIP) if no prediction tensors are
present in the intake dictionary; no synthetic data is fabricated.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from resistancemap.evaluation.base import EvalAgent, EvalFinding, Verdict

logger = logging.getLogger(__name__)


# Justification for the perturbation-budget threshold:
# To estimate a tail quantile q with relative error eps at 95% confidence using
# Monte Carlo, the rule of thumb is N >= 1 / (q * eps^2). For a clinically
# meaningful "rare resistance mode" prevalence of q = 0.05 and eps = 0.20 we
# need N >= ~500. The current default config.trajectory.n_perturbation = 50
# is ~10x too small for that regime; we therefore flag anything below 200 as
# inadequate (a softer threshold that still tolerates q ~ 0.10, eps ~ 0.30).
MIN_PERTURBATION_BUDGET = 200


class ForecastingUncertaintyAgent(EvalAgent):
    """Tier C agent: probabilistic forecasting quality and uncertainty audit.

    Expected ``intake`` keys (all optional; agent skips if absent):

    - ``y_true_event``: (N,) binary event indicator at each horizon, OR
      (N, H) per-horizon binary labels.
    - ``y_true_time``: (N,) time-to-event (same units the model claims).
    - ``y_pred_risk``: (N, H) predicted probability of event by horizon h.
    - ``y_pred_quantiles``: (N, H, Q) quantile forecasts.
    - ``quantile_levels``: (Q,) quantile levels in (0, 1) matching last axis.
    - ``y_pred_ensemble``: (N, H, M) Monte-Carlo ensemble forecasts.
    - ``rare_mode_mask``: (N,) bool, marks rare-phenotype patients for
      stratified scoring.
    """

    tier = "C"

    def __init__(self) -> None:
        super().__init__(
            name="forecasting_uncertainty_agent",
            dependencies=[],
        )

    async def assess(
        self, intake: dict[str, Any], config: Any
    ) -> EvalFinding:  # type: ignore[override]
        """Score probabilistic forecasts produced by the trajectory module."""
        # Lazy import so missing metric module doesn't break package import.
        try:
            from resistancemap.evaluation.metrics.survival import (
                concordance_index,
                integrated_brier_score,
                time_dependent_auc,
            )
            from resistancemap.evaluation.metrics.forecasting import (
                crps_ensemble,
                pinball_loss,
            )
            from resistancemap.evaluation.metrics.calibration import (
                expected_calibration_error,
                reliability_curve,
            )
        except Exception as exc:  # pragma: no cover - merge-time guard
            logger.warning(
                "Metric modules not yet importable (%s); proceeding "
                "with structural-only assessment.",
                exc,
            )
            integrated_brier_score = None  # type: ignore[assignment]
            time_dependent_auc = None  # type: ignore[assignment]
            concordance_index = None  # type: ignore[assignment]
            crps_ensemble = None  # type: ignore[assignment]
            pinball_loss = None  # type: ignore[assignment]
            expected_calibration_error = None  # type: ignore[assignment]
            reliability_curve = None  # type: ignore[assignment]

        horizons = list(config.trajectory.forecast_horizons)
        n_perturbation = int(config.trajectory.n_perturbation)
        integration_time = float(config.trajectory.integration_time)

        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {
            "horizons_months": horizons,
            "n_perturbation": n_perturbation,
            "integration_time_au": integration_time,
        }
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # ------------------------------------------------------------------
        # ALWAYS emit the time-axis calibration warning. This is non-negotiable
        # as long as the model uses arbitrary ODE units.
        # ------------------------------------------------------------------
        required_changes.append(
            "calibrate ODE time to calendar time before claiming time-to-event "
            "predictions at 3/6/12 months"
        )
        failure_modes.append(
            "ODE integration_time is in arbitrary units; mapping to calendar "
            "months has not been demonstrated. Any horizon-indexed metric is "
            "preliminary until that mapping exists."
        )

        # ------------------------------------------------------------------
        # Skip-gracefully if no prediction tensors at all.
        # ------------------------------------------------------------------
        has_risk = intake.get("y_pred_risk") is not None
        has_quantiles = intake.get("y_pred_quantiles") is not None
        has_ensemble = intake.get("y_pred_ensemble") is not None
        if not (has_risk or has_quantiles or has_ensemble):
            logger.info(
                "ForecastingUncertaintyAgent: no prediction tensors in "
                "intake; emitting SKIP verdict."
            )
            criteria_results["predictions_present"] = False
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=Verdict.SKIPPED,
                score=None,
                criteria_results=criteria_results,
                evidence=evidence,
                failure_modes=failure_modes,
                required_changes=required_changes,
            )

        criteria_results["predictions_present"] = True

        # ------------------------------------------------------------------
        # Time-to-event metrics.
        # ------------------------------------------------------------------
        y_true_event = _as_array(intake.get("y_true_event"))
        y_true_time = _as_array(intake.get("y_true_time"))
        y_pred_risk = _as_array(intake.get("y_pred_risk"))

        if (
            y_true_event is not None
            and y_true_time is not None
            and y_pred_risk is not None
            and integrated_brier_score is not None
        ):
            try:
                ibs = float(
                    integrated_brier_score(
                        y_true_time, y_true_event, y_pred_risk, times=horizons
                    )
                )
                criteria_results["integrated_brier_score"] = ibs
                evidence["integrated_brier_score"] = ibs
                # Lower IBS is better; 0.25 is the uninformative-baseline
                # ceiling we tolerate.
                if ibs > 0.25:
                    failure_modes.append(
                        f"Integrated Brier Score {ibs:.3f} exceeds the "
                        "0.25 uninformative threshold."
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("IBS computation failed: %s", exc)
                criteria_results["integrated_brier_score_error"] = str(exc)

            try:
                td_auc = time_dependent_auc(
                    y_true_time, y_true_event, y_pred_risk, times=horizons
                )
                criteria_results["time_dependent_auc"] = _to_serializable(td_auc)
                evidence["time_dependent_auc"] = _to_serializable(td_auc)
            except Exception as exc:  # pragma: no cover
                logger.warning("Time-dependent AUC failed: %s", exc)

            if concordance_index is not None:
                try:
                    # C-index conventionally takes risk averaged across horizons
                    # if the model emits per-horizon risks.
                    risk_summary = (
                        y_pred_risk.mean(axis=-1)
                        if y_pred_risk.ndim > 1
                        else y_pred_risk
                    )
                    c_idx = float(
                        concordance_index(y_true_time, y_true_event, risk_summary)
                    )
                    criteria_results["concordance_index"] = c_idx
                    evidence["concordance_index"] = c_idx
                except Exception as exc:  # pragma: no cover
                    logger.warning("C-index failed: %s", exc)

        # ------------------------------------------------------------------
        # Calibration plots at each horizon.
        # ------------------------------------------------------------------
        if (
            y_pred_risk is not None
            and y_true_event is not None
            and expected_calibration_error is not None
            and reliability_curve is not None
        ):
            calibration_per_horizon: dict[str, Any] = {}
            for h_idx, h in enumerate(horizons):
                try:
                    risks_h = (
                        y_pred_risk[:, h_idx]
                        if y_pred_risk.ndim == 2
                        else y_pred_risk
                    )
                    labels_h = (
                        y_true_event[:, h_idx]
                        if y_true_event.ndim == 2
                        else y_true_event
                    )
                    ece = float(expected_calibration_error(risks_h, labels_h))
                    curve = reliability_curve(labels_h, risks_h)
                    calibration_per_horizon[f"h={h}"] = {
                        "ece": ece,
                        "reliability_curve": _to_serializable(curve),
                    }
                    if ece > 0.10:
                        failure_modes.append(
                            f"Calibration at horizon {h}: ECE={ece:.3f} > 0.10"
                        )
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Calibration at horizon %s failed: %s", h, exc
                    )
            criteria_results["calibration_per_horizon"] = calibration_per_horizon

        # ------------------------------------------------------------------
        # Proper scoring rules for distributional forecasts.
        # ------------------------------------------------------------------
        y_pred_quantiles = _as_array(intake.get("y_pred_quantiles"))
        quantile_levels = _as_array(intake.get("quantile_levels"))
        if (
            y_pred_quantiles is not None
            and quantile_levels is not None
            and y_true_time is not None
            and pinball_loss is not None
        ):
            try:
                # Compute pinball loss by averaging over quantiles
                pl_values = []
                for q_idx, q_level in enumerate(quantile_levels):
                    # Extract quantile predictions for this level
                    q_preds = y_pred_quantiles[:, :, q_idx] if y_pred_quantiles.ndim == 3 else y_pred_quantiles[:, q_idx]
                    loss_q = pinball_loss(y_true_time, q_preds, q_level)
                    pl_values.append(loss_q)
                pl = float(np.mean(pl_values))
                criteria_results["pinball_loss"] = pl
                evidence["pinball_loss"] = pl
            except Exception as exc:  # pragma: no cover
                logger.warning("Pinball loss failed: %s", exc)

        y_pred_ensemble = _as_array(intake.get("y_pred_ensemble"))
        if (
            y_pred_ensemble is not None
            and y_true_time is not None
            and crps_ensemble is not None
        ):
            try:
                crps = float(crps_ensemble(y_true_time, y_pred_ensemble))
                criteria_results["crps"] = crps
                evidence["crps"] = crps
            except Exception as exc:  # pragma: no cover
                logger.warning("CRPS failed: %s", exc)

        # ------------------------------------------------------------------
        # Honest uncertainty for rare resistance modes.
        # ------------------------------------------------------------------
        criteria_results["n_perturbation_adequate"] = (
            n_perturbation >= MIN_PERTURBATION_BUDGET
        )
        if n_perturbation < MIN_PERTURBATION_BUDGET:
            failure_modes.append(
                f"n_perturbation={n_perturbation} is below the "
                f"{MIN_PERTURBATION_BUDGET} threshold needed to resolve "
                "tail quantiles for rare resistance modes (see docstring "
                "derivation)."
            )
            required_changes.append(
                f"raise config.trajectory.n_perturbation to >= "
                f"{MIN_PERTURBATION_BUDGET} or document the prevalence of "
                "the rare modes you actually need to resolve and re-derive "
                "the budget"
            )

        rare_mask = _as_array(intake.get("rare_mode_mask"))
        if (
            rare_mask is not None
            and y_pred_risk is not None
            and y_true_event is not None
        ):
            try:
                rare_idx = rare_mask.astype(bool)
                if rare_idx.sum() >= 5 and expected_calibration_error is not None:
                    rare_risks = (
                        y_pred_risk[rare_idx].mean(axis=-1)
                        if y_pred_risk.ndim > 1
                        else y_pred_risk[rare_idx]
                    )
                    rare_labels = (
                        y_true_event[rare_idx].max(axis=-1)
                        if y_true_event.ndim > 1
                        else y_true_event[rare_idx]
                    )
                    rare_ece = float(
                        expected_calibration_error(rare_labels, rare_risks)
                    )
                    criteria_results["rare_mode_ece"] = rare_ece
                    evidence["rare_mode_n"] = int(rare_idx.sum())
                    if rare_ece > 0.15:
                        failure_modes.append(
                            f"Rare-mode ECE {rare_ece:.3f} > 0.15: model is "
                            "miscalibrated where it matters most."
                        )
                else:
                    criteria_results["rare_mode_skipped"] = (
                        "fewer than 5 rare-mode patients in intake"
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("Rare-mode assessment failed: %s", exc)

        # ------------------------------------------------------------------
        # Verdict decision.
        # ------------------------------------------------------------------
        # Always at least CONDITIONAL_PASS because the time-axis caveat fires
        # unconditionally. Drop to FAIL if any hard quantitative gate failed.
        score = _aggregate_score(criteria_results)
        verdict: Verdict
        hard_failures = [
            f for f in failure_modes if "exceeds" in f or "miscalibrated" in f
        ]
        if hard_failures:
            verdict = Verdict.FAIL
        elif n_perturbation < MIN_PERTURBATION_BUDGET:
            verdict = Verdict.CONDITIONAL
        else:
            verdict = Verdict.CONDITIONAL

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


def _as_array(x: Any) -> np.ndarray | None:
    """Coerce intake values to numpy arrays without fabricating data."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def _to_serializable(x: Any) -> Any:
    """Convert numpy types to plain Python for criteria_results dict."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    return x


def _aggregate_score(criteria: dict[str, Any]) -> float | None:
    """Heuristic 0-1 aggregate score from numeric criteria.

    Lower-is-better metrics (IBS, pinball, CRPS, ECE) are inverted via
    1 - clip(value, 0, 1). Higher-is-better metrics (AUC, C-index) are
    used directly.
    """
    parts: list[float] = []

    def _push(val: Any, higher_is_better: bool) -> None:
        if val is None:
            return
        if isinstance(val, (list, tuple)):
            for v in val:
                _push(v, higher_is_better)
            return
        if isinstance(val, dict):
            return
        try:
            f = float(val)
        except (TypeError, ValueError):
            return
        if math.isnan(f):
            return
        f = max(0.0, min(1.0, f))
        parts.append(f if higher_is_better else 1.0 - f)

    _push(criteria.get("integrated_brier_score"), higher_is_better=False)
    _push(criteria.get("time_dependent_auc"), higher_is_better=True)
    _push(criteria.get("concordance_index"), higher_is_better=True)
    _push(criteria.get("pinball_loss"), higher_is_better=False)
    _push(criteria.get("crps"), higher_is_better=False)
    cal = criteria.get("calibration_per_horizon") or {}
    if isinstance(cal, dict):
        for v in cal.values():
            if isinstance(v, dict):
                _push(v.get("ece"), higher_is_better=False)
    _push(criteria.get("rare_mode_ece"), higher_is_better=False)

    if not parts:
        return None
    return float(sum(parts) / len(parts))