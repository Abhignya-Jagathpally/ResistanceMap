"""Per-drug metrics with bootstrap CIs.

Replaces the pooled AUROC headline that hid per-drug collapse on n=12/drug.
Reporting rule (enforced by ``require_per_drug_with_pooled``): no pooled
AUROC / AUPRC may be quoted without the per-drug table alongside it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class DrugMetric:
    drug: str
    n: int
    n_pos: int
    auroc: float
    auroc_lo: float
    auroc_hi: float
    auprc: float
    auprc_lo: float
    auprc_hi: float
    note: str = ""


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    if not values:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(np.mean(values)), float(lo), float(hi)


def per_drug_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    drug: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
    min_each_class: int = 3,
) -> list[DrugMetric]:
    """Per-drug AUROC/AUPRC with 95% percentile bootstrap intervals."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    drug = np.asarray(drug)
    rng = np.random.default_rng(seed)

    out: list[DrugMetric] = []
    for d in np.unique(drug):
        mask = drug == d
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum())
        n_neg = n - n_pos
        if n_pos < min_each_class or n_neg < min_each_class:
            out.append(DrugMetric(
                drug=str(d), n=n, n_pos=n_pos,
                auroc=float("nan"), auroc_lo=float("nan"), auroc_hi=float("nan"),
                auprc=float("nan"), auprc_lo=float("nan"), auprc_hi=float("nan"),
                note=f"insufficient class balance (n_pos={n_pos}, n_neg={n_neg})",
            ))
            continue
        auroc, auroc_lo, auroc_hi = _bootstrap_ci(
            y_true[mask], y_score[mask], roc_auc_score, n_boot, rng
        )
        auprc, auprc_lo, auprc_hi = _bootstrap_ci(
            y_true[mask], y_score[mask], average_precision_score, n_boot, rng
        )
        out.append(DrugMetric(
            drug=str(d), n=n, n_pos=n_pos,
            auroc=auroc, auroc_lo=auroc_lo, auroc_hi=auroc_hi,
            auprc=auprc, auprc_lo=auprc_lo, auprc_hi=auprc_hi,
        ))
    return out


def pooled_with_ci(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    rng = np.random.default_rng(seed)
    auroc, lo, hi = _bootstrap_ci(y_true, y_score, roc_auc_score, n_boot, rng)
    return {"auroc": auroc, "auroc_lo": lo, "auroc_hi": hi, "n": len(y_true)}


def require_per_drug_with_pooled(
    y_true, y_score, drug, **kwargs
) -> dict:
    """Return pooled + per-drug together. Prevents hiding per-drug collapse."""
    return {
        "pooled": pooled_with_ci(y_true, y_score, **kwargs),
        "per_drug": [asdict(m) for m in per_drug_metrics(y_true, y_score, drug, **kwargs)],
    }
