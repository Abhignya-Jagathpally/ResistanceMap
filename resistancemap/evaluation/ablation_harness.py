"""Honest ablation harness with frozen upstream + paired bootstrap.

Closes the "monotone 0.82 -> 0.89 with no variance bars" suspicion:
  * Each ablation freezes upstream weights before training downstream.
  * All ablations share IDENTICAL CV folds and bootstrap indices (paired).
  * Reports mean +/- std across >= 5 seeds; paired p-value vs the row above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd


ABLATIONS: list[str] = [
    "vae_only",
    "vae_plus_trajectory",
    "vae_plus_gnn",                   # isolate GNN contribution
    "vae_plus_trajectory_no_gnn",
    "vae_plus_gnn_no_trajectory",
    "full_minus_fusion",              # late-fusion logistic stack instead
    "full_minus_landscape",
    "full",
]


@dataclass
class AblationResult:
    name: str
    per_seed: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.per_seed)) if self.per_seed else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.per_seed, ddof=1)) if len(self.per_seed) > 1 else float("nan")


def paired_bootstrap_pvalue(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided paired bootstrap p-value on score differences.

    Requires identical sample indices across both models (paired!).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Scores must be paired (identical indices).")
    rng = np.random.default_rng(seed)
    diffs = a - b
    obs = diffs.mean()
    null = diffs - obs      # centre under H0: mean diff = 0
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = null[idx].mean()
    return float((np.abs(boot_means) >= abs(obs)).mean())


def run_ablation(
    name: str,
    train_and_eval_fn: Callable[[str, int], float],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> AblationResult:
    """train_and_eval_fn(name, seed) -> scalar score on identical held-out fold.

    Implementation contract (enforce inside train_and_eval_fn):
      1. Load identical CV split for the seed.
      2. Freeze all upstream stage weights before training the downstream stage.
      3. Return score on a fold identified by seed (not resampled per ablation).
    """
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation {name!r}")
    res = AblationResult(name=name)
    for s in seeds:
        res.per_seed.append(float(train_and_eval_fn(name, s)))
    return res


def ablation_table(results: list[AblationResult]) -> pd.DataFrame:
    rows = []
    prev_scores: list[float] | None = None
    for r in results:
        pval = paired_bootstrap_pvalue(r.per_seed, prev_scores) if prev_scores else None
        rows.append({
            "ablation": r.name,
            "mean": r.mean,
            "std": r.std,
            "n_seeds": len(r.per_seed),
            "p_vs_prev": pval,
            "significant": (pval is not None and pval < 0.05),
        })
        prev_scores = r.per_seed
    df = pd.DataFrame(rows)
    # Make "n.s." deltas visible rather than hidden.
    df["verdict"] = np.where(
        df["p_vs_prev"].isna(), "baseline",
        np.where(df["significant"], "significant", "n.s."),
    )
    return df
