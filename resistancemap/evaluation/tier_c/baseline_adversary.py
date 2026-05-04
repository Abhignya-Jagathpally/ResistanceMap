"""Tier C: Baseline Adversary agent.

Pre-registers and adjudicates head-to-head comparisons of the main
ResistanceMap model against simpler baselines (clinical-only, transcriptome
PCA + linear, last-observation carry-forward, Kaplan-Meier / Cox).

Key principles enforced here:

1. **Pre-registration.** The full comparison schema (baselines, splits, metrics,
   acceptance gates) is declared *before* any number is computed. Anything
   computed outside this schema is reported as exploratory.
2. **No training inside the agent.** Baseline predictions must already exist
   in ``intake['baseline_predictions']``. Training is the job of
   ``resistancemap.evaluation.baselines``; this agent only *adjudicates*.
3. **Nested CV + bootstrap CIs + Benjamini-Hochberg.** A baseline only
   "loses" (and the corresponding architectural choice only earns its keep)
   if the main-model gain is significant under nested CV with bootstrap
   confidence intervals on the metric difference, controlled at FDR=0.10
   across the full family of comparisons.
4. **Incremental value table.** The agent emits a row for each
   architectural decision (omics fusion, GNN-on-PPI, ODE trajectory, latent
   dim) that pairs the deletion baseline with the empirical delta and CI.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from resistancemap.evaluation.base import EvalAgent, EvalFinding, Verdict

logger = logging.getLogger(__name__)


# Pre-registered architectural-choice -> deletion-baseline mapping. The agent
# emits one row of the "incremental value" table per entry.
DESIGN_CHOICES: list[dict[str, str]] = [
    {
        "design_choice": "Multi-omics fusion (proteomics + epigenomics + scRNA)",
        "deletion_baseline": "transcriptome_only",
        "discards": "proteomics + ATAC-seq + clinical context",
    },
    {
        "design_choice": "ODE-based trajectory module",
        "deletion_baseline": "last_observation",
        "discards": "temporal smoothness + perturbation sensitivity",
    },
    {
        "design_choice": "GNN-on-PPI protein interaction module",
        "deletion_baseline": "transcriptome_only",
        "discards": "biological prior over STRING interaction edges",
    },
    {
        "design_choice": "Latent disease state (VAE)",
        "deletion_baseline": "clinical_only",
        "discards": "high-dimensional molecular structure",
    },
    {
        "design_choice": "Survival head with proper time-to-event loss",
        "deletion_baseline": "simple_survival",
        "discards": "non-linear interactions and per-patient covariates",
    },
]


@dataclass
class BaselineComparisonSpec:
    """Pre-registered comparison row."""

    baseline_name: str
    optimizes: str
    discards: str
    metric: str
    pre_specified_split: str
    acceptance_rule: str
    raw_delta: float | None = None
    bootstrap_ci: tuple[float, float] | None = None
    p_value: float | None = None
    significant_after_bh: bool | None = None
    notes: str = ""


@dataclass
class IncrementalValueRow:
    """One row of the incremental-value-of-design-choices table."""

    design_choice: str
    deletion_baseline: str
    discards: str
    metric: str
    delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    significant: bool | None = None
    interpretation: str = ""
    rows_aggregated: int = 0


class BaselineAdversaryAgent(EvalAgent):
    """Tier C agent: pre-registered baseline adversary.

    Expected ``intake`` keys:

    - ``baseline_predictions``: dict of
      ``{baseline_name: {"y_pred": np.ndarray, "metric": str,
        "split": str, "main_y_pred": np.ndarray, "y_true": np.ndarray}}``.
      ``baseline_name`` should be one of the names produced by
      ``resistancemap.evaluation.baselines`` (e.g. ``clinical_only``,
      ``transcriptome_only``, ``last_observation``, ``simple_survival``).
      ``main_y_pred`` is the main-model prediction on the same indices.
    - ``n_bootstrap``: optional int (default 1000).
    - ``fdr_alpha``: optional float (default 0.10).
    - ``cv_folds``: optional int (default 5) -- only used to record what was
      pre-specified; the agent does NOT run CV itself.

    The agent reports verdict=PASS only if every architectural choice
    *either* shows a positive significant delta *or* is explicitly marked
    as not-yet-evaluated; verdict=FAIL if any baseline matches/beats the
    main model with significance after BH correction.
    """

    tier = "C"

    def __init__(self) -> None:
        super().__init__(
            name="baseline_adversary_agent",
            dependencies=[],
        )

    async def assess(
        self, intake: dict[str, Any], config: Any
    ) -> EvalFinding:  # type: ignore[override]
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        baseline_predictions = intake.get("baseline_predictions") or {}
        n_bootstrap = int(intake.get("n_bootstrap", 1000))
        fdr_alpha = float(intake.get("fdr_alpha", 0.10))
        cv_folds = int(intake.get("cv_folds", 5))

        evidence["pre_registration"] = {
            "baselines": [c["deletion_baseline"] for c in DESIGN_CHOICES],
            "design_choices": DESIGN_CHOICES,
            "n_bootstrap": n_bootstrap,
            "fdr_alpha": fdr_alpha,
            "nested_cv_folds": cv_folds,
            "multiple_testing_correction": "Benjamini-Hochberg (FDR)",
        }

        if not baseline_predictions:
            criteria_results["baseline_predictions_present"] = False
            failure_modes.append(
                "No baseline_predictions in intake; nothing to adjudicate."
            )
            required_changes.append(
                "compute predictions for each baseline in "
                "resistancemap.evaluation.baselines on the pre-specified "
                "splits and pass them in intake['baseline_predictions']"
            )
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

        criteria_results["baseline_predictions_present"] = True

        # ------------------------------------------------------------------
        # Per-baseline comparison + bootstrap CI on metric delta.
        # ------------------------------------------------------------------
        comparisons: list[BaselineComparisonSpec] = []
        p_values: list[float] = []
        for name, payload in baseline_predictions.items():
            try:
                y_true = np.asarray(payload["y_true"])
                y_pred = np.asarray(payload["y_pred"])
                main_pred = np.asarray(payload["main_y_pred"])
                metric_name = str(payload.get("metric", "auc"))
                split = str(payload.get("split", "test"))
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Malformed baseline payload %s: %s; skipping.", name, exc
                )
                continue

            try:
                delta, ci, pval = _bootstrap_delta(
                    y_true, main_pred, y_pred, metric_name, n_bootstrap
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Bootstrap failed for baseline %s: %s", name, exc)
                continue

            spec = BaselineComparisonSpec(
                baseline_name=name,
                optimizes=_what_baseline_optimizes(name),
                discards=_what_baseline_discards(name),
                metric=metric_name,
                pre_specified_split=split,
                acceptance_rule=(
                    "main model retained iff delta > 0 with bootstrap "
                    "95% CI excluding 0 AND BH-adjusted p < alpha"
                ),
                raw_delta=float(delta),
                bootstrap_ci=(float(ci[0]), float(ci[1])),
                p_value=float(pval),
            )
            comparisons.append(spec)
            p_values.append(pval)

        # Benjamini-Hochberg over the family of comparisons.
        bh_significant = _benjamini_hochberg(p_values, fdr_alpha)
        for spec, sig in zip(comparisons, bh_significant):
            spec.significant_after_bh = bool(sig)
            if not sig and spec.raw_delta is not None and spec.raw_delta <= 0:
                failure_modes.append(
                    f"Baseline {spec.baseline_name} matches/beats main model "
                    f"on {spec.metric} (delta={spec.raw_delta:+.3f}, "
                    f"CI={spec.bootstrap_ci})."
                )

        criteria_results["baseline_comparisons"] = [
            _spec_to_dict(s) for s in comparisons
        ]

        # ------------------------------------------------------------------
        # Build the incremental value table.
        # ------------------------------------------------------------------
        comparisons_by_baseline: dict[str, list[BaselineComparisonSpec]] = {}
        for spec in comparisons:
            comparisons_by_baseline.setdefault(spec.baseline_name, []).append(spec)

        incremental_rows: list[IncrementalValueRow] = []
        for choice in DESIGN_CHOICES:
            matching = comparisons_by_baseline.get(choice["deletion_baseline"], [])
            if not matching:
                row = IncrementalValueRow(
                    design_choice=choice["design_choice"],
                    deletion_baseline=choice["deletion_baseline"],
                    discards=choice["discards"],
                    metric="(not evaluated)",
                    interpretation=(
                        "no prediction provided for this deletion baseline; "
                        "incremental value of design choice is UNKNOWN."
                    ),
                )
                required_changes.append(
                    f"provide baseline predictions for "
                    f"'{choice['deletion_baseline']}' so the incremental "
                    f"value of '{choice['design_choice']}' can be measured"
                )
            else:
                # Aggregate (delta-weighted average) across all metrics for the
                # same baseline -- usually only one row in practice.
                deltas = [s.raw_delta for s in matching if s.raw_delta is not None]
                if not deltas:
                    continue
                avg_delta = float(np.mean(deltas))
                lows = [
                    s.bootstrap_ci[0] for s in matching if s.bootstrap_ci is not None
                ]
                highs = [
                    s.bootstrap_ci[1] for s in matching if s.bootstrap_ci is not None
                ]
                ci_low = float(np.mean(lows)) if lows else None
                ci_high = float(np.mean(highs)) if highs else None
                significant = any(s.significant_after_bh for s in matching)
                interpretation = (
                    "design choice earns its keep"
                    if significant and avg_delta > 0
                    else "design choice not yet justified by data"
                )
                row = IncrementalValueRow(
                    design_choice=choice["design_choice"],
                    deletion_baseline=choice["deletion_baseline"],
                    discards=choice["discards"],
                    metric=matching[0].metric,
                    delta=avg_delta,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    significant=significant,
                    interpretation=interpretation,
                    rows_aggregated=len(matching),
                )
            incremental_rows.append(row)

        criteria_results["incremental_value_table"] = [
            _row_to_dict(r) for r in incremental_rows
        ]
        evidence["incremental_value_table"] = criteria_results[
            "incremental_value_table"
        ]

        # ------------------------------------------------------------------
        # Verdict.
        # ------------------------------------------------------------------
        any_baseline_wins = any(
            s.raw_delta is not None
            and s.raw_delta <= 0
            and s.significant_after_bh
            for s in comparisons
        )
        any_unjustified = any(
            r.significant is False or r.delta is None for r in incremental_rows
        )

        if any_baseline_wins:
            verdict = Verdict.FAIL
        elif any_unjustified:
            verdict = Verdict.CONDITIONAL
        else:
            verdict = Verdict.PASS

        score = _aggregate_incremental_score(incremental_rows)

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


# ----------------------------------------------------------------------------
# Helper functions.
# ----------------------------------------------------------------------------


def _what_baseline_optimizes(name: str) -> str:
    return {
        "clinical_only": "linear loss on age/ISS/cytogenetics covariates only",
        "transcriptome_only": "PCA + linear classification loss on RNA",
        "last_observation": "carry-forward MSE on last labeled observation",
        "simple_survival": "Kaplan-Meier / Cox partial likelihood",
    }.get(name, "unspecified")


def _what_baseline_discards(name: str) -> str:
    return {
        "clinical_only": "all molecular omics, perturbation dynamics, GNN priors",
        "transcriptome_only": "proteomics, ATAC, fusion, perturbation dynamics",
        "last_observation": "all model structure; only past label is used",
        "simple_survival": "non-linear effects, multimodal covariates",
    }.get(name, "unspecified")


def _bootstrap_delta(
    y_true: np.ndarray,
    main_pred: np.ndarray,
    base_pred: np.ndarray,
    metric_name: str,
    n_bootstrap: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, tuple[float, float], float]:
    """Bootstrap the metric delta (main - baseline)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(y_true)
    if n < 2:
        return 0.0, (0.0, 0.0), 1.0

    metric_fn = _resolve_metric(metric_name)

    deltas = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            m_main = metric_fn(y_true[idx], main_pred[idx])
            m_base = metric_fn(y_true[idx], base_pred[idx])
            deltas[b] = m_main - m_base
        except Exception:
            deltas[b] = 0.0
    point = float(metric_fn(y_true, main_pred) - metric_fn(y_true, base_pred))
    ci_low = float(np.quantile(deltas, 0.025))
    ci_high = float(np.quantile(deltas, 0.975))
    # Two-sided p-value: fraction of bootstrap samples crossing zero.
    if point >= 0:
        p_val = float((deltas <= 0).mean())
    else:
        p_val = float((deltas >= 0).mean())
    p_val = max(min(p_val * 2, 1.0), 1.0 / max(n_bootstrap, 1))
    return point, (ci_low, ci_high), p_val


def _resolve_metric(name: str):
    """Return a callable f(y_true, y_pred) -> float."""
    name = name.lower()
    if name in ("auc", "auroc", "roc_auc"):
        return _auc_score
    if name in ("brier", "brier_score"):
        return lambda yt, yp: -float(np.mean((yt - yp) ** 2))
    if name in ("mse",):
        return lambda yt, yp: -float(np.mean((yt - yp) ** 2))
    if name in ("acc", "accuracy"):
        return lambda yt, yp: float(np.mean((yp >= 0.5) == (yt >= 0.5)))
    # Default: negative absolute error so "higher is better".
    return lambda yt, yp: -float(np.mean(np.abs(yt - yp)))


def _auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mann-Whitney U based AUC, no sklearn dependency."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    pos = y_pred[y_true > 0.5]
    neg = y_pred[y_true <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_pos = ranks[: len(pos)].sum()
    return float(
        (rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    )


def _benjamini_hochberg(p_values: list[float], alpha: float) -> list[bool]:
    """Return BH significance flags at level alpha."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    passed = sorted_p <= thresholds
    if not passed.any():
        cutoff_rank = -1
    else:
        cutoff_rank = int(np.max(np.where(passed)[0]))
    significant_sorted = np.zeros(m, dtype=bool)
    if cutoff_rank >= 0:
        significant_sorted[: cutoff_rank + 1] = True
    out = np.zeros(m, dtype=bool)
    out[order] = significant_sorted
    return out.tolist()


def _aggregate_incremental_score(rows: list[IncrementalValueRow]) -> float | None:
    if not rows:
        return None
    counted = [r for r in rows if r.delta is not None]
    if not counted:
        return None
    n_pos = sum(1 for r in counted if r.significant and (r.delta or 0) > 0)
    return float(n_pos / len(counted))


def _spec_to_dict(s: BaselineComparisonSpec) -> dict[str, Any]:
    return {
        "baseline_name": s.baseline_name,
        "optimizes": s.optimizes,
        "discards": s.discards,
        "metric": s.metric,
        "pre_specified_split": s.pre_specified_split,
        "acceptance_rule": s.acceptance_rule,
        "raw_delta": s.raw_delta,
        "bootstrap_ci": list(s.bootstrap_ci) if s.bootstrap_ci else None,
        "p_value": s.p_value,
        "significant_after_bh": s.significant_after_bh,
        "notes": s.notes,
    }


def _row_to_dict(r: IncrementalValueRow) -> dict[str, Any]:
    return {
        "design_choice": r.design_choice,
        "deletion_baseline": r.deletion_baseline,
        "discards": r.discards,
        "metric": r.metric,
        "delta": r.delta,
        "ci_low": r.ci_low,
        "ci_high": r.ci_high,
        "significant": r.significant,
        "interpretation": r.interpretation,
        "rows_aggregated": r.rows_aggregated,
    }
