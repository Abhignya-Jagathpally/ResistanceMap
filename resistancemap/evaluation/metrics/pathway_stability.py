"""Pathway-attribution stability metrics for graph-grounded reasoning audits.

These metrics quantify whether attributions over a PPI are stable to graph
perturbations and whether reported pathways generalize to held-out subsets.
All functions accept caller-provided arrays / callables and never fabricate
data.

References
----------
Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting
    Model Predictions." NeurIPS.
Ghorbani, A., Abid, A., & Zou, J. (2019). "Interpretation of Neural Networks
    is Fragile." AAAI.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "attribution_overlap",
    "edge_dropout_stability",
    "pathway_holdout_generalization",
]


def _to_dict_or_array(attr: Any) -> Dict[Any, float]:
    """Coerce a 1D array or mapping to a {key: float} dict."""
    if isinstance(attr, dict):
        return {k: float(v) for k, v in attr.items()}
    arr = np.asarray(attr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("attribution arrays must be 1D when passed as ndarray")
    return {i: float(arr[i]) for i in range(arr.shape[0])}


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall-tau-b without scipy. O(N^2) — fine for attributions of <~10k items."""
    n = x.shape[0]
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        s = np.sign(dx) * np.sign(dy)
        concordant += int((s > 0).sum())
        discordant += int((s < 0).sum())
        tie_x += int(((dx == 0) & (dy != 0)).sum())
        tie_y += int(((dy == 0) & (dx != 0)).sum())
    denom = np.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _is_set_like_attribution(attr: Any) -> bool:
    """True if ``attr`` is a list/tuple/set of feature identifiers (no scores).

    Detects the simple "ranked feature list" form (e.g. ``["TP53", "KRAS"]``)
    that the rubric / pathway tests use, distinct from the score-bearing
    forms (1D numeric array or ``{id: score}`` dict).
    """
    if isinstance(attr, dict):
        return False
    if isinstance(attr, np.ndarray):
        return False
    if isinstance(attr, (list, tuple, set, frozenset)):
        # If every entry is non-numeric, treat as a feature set.
        for item in attr:
            if isinstance(item, (int, float, np.integer, np.floating, bool)):
                return False
        return True
    return False


def attribution_overlap(
    attribution_a: Any,
    attribution_b: Any,
    top_k: Optional[int] = None,
) -> Any:
    """Compare two feature attributions: Jaccard, Kendall-tau, top-k overlap.

    This function has two modes:

    **Set mode** -- when both inputs are plain iterables of feature
    identifiers (e.g. ``["TP53", "KRAS"]``), the function returns a single
    float: the Jaccard index ``|A ∩ B| / |A ∪ B|``. This is the form used
    by simple ranked-list comparisons.

    **Score mode** -- when at least one input is a numeric array or a
    ``{feature: score}`` dict, the function returns a richer dict with
    ``jaccard``, ``kendall_tau``, ``top_k_overlap``, and ``n_shared``.

    Args:
        attribution_a: Either an iterable of feature ids (set mode) or a
            score container (score mode).
        attribution_b: Same convention as ``attribution_a``.
        top_k: If given, restrict the score-mode comparison to the top-k
            features of each attribution by absolute score.

    Returns:
        Float (set mode) or dict of overlap statistics (score mode).
    """
    if _is_set_like_attribution(attribution_a) and _is_set_like_attribution(attribution_b):
        set_a = set(attribution_a)
        set_b = set(attribution_b)
        union = set_a | set_b
        if not union:
            return float("nan")
        return len(set_a & set_b) / len(union)

    a = _to_dict_or_array(attribution_a)
    b = _to_dict_or_array(attribution_b)

    shared_keys = sorted(set(a) & set(b), key=lambda k: str(k))
    if not shared_keys:
        return {
            "jaccard": float("nan"),
            "kendall_tau": float("nan"),
            "top_k_overlap": float("nan"),
            "n_shared": 0,
        }

    a_vec = np.array([a[k] for k in shared_keys], dtype=np.float64)
    b_vec = np.array([b[k] for k in shared_keys], dtype=np.float64)

    tau = _kendall_tau(a_vec, b_vec)

    n = len(shared_keys)
    k = top_k if top_k is not None else n
    k = max(1, min(k, n))
    a_top = set(np.argsort(-np.abs(a_vec))[:k].tolist())
    b_top = set(np.argsort(-np.abs(b_vec))[:k].tolist())
    union = a_top | b_top
    inter = a_top & b_top
    jaccard = len(inter) / len(union) if union else float("nan")
    overlap = len(inter) / k

    return {
        "jaccard": float(jaccard),
        "kendall_tau": float(tau),
        "top_k_overlap": float(overlap),
        "n_shared": int(n),
    }


def _bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05,
                  rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean of a sequence of floats."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if rng is None:
        rng = np.random.default_rng(0)
    means = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2)),
    }


def edge_dropout_stability(
    attribute_fn: Callable[[Any], Any],
    ppi_edges: np.ndarray,
    dropout_rates: Sequence[float] = (0.05, 0.1, 0.2),
    n_trials: int = 10,
    seed: int = 0,
) -> Dict[str, Any]:
    """Stability of attributions under random PPI edge dropout.

    For each dropout rate, repeatedly delete that fraction of PPI edges,
    re-run ``attribute_fn`` on the perturbed edge set, and measure the
    Jaccard / top-k overlap with the unperturbed attribution. Returns
    bootstrap CIs over the n_trials replicates.

    Args:
        attribute_fn: Callable taking an edge array and returning an
            attribution (1D ndarray or dict).
        ppi_edges: PPI edges, shape (E, 2) or (E, 3) (head, tail, [weight]).
        dropout_rates: Edge dropout fractions in [0, 1).
        n_trials: Independent dropout draws per rate.
        seed: Base RNG seed (offset per trial for reproducibility).

    Returns:
        Dict keyed by str(rate) with sub-dicts ``jaccard``, ``top_k_overlap``
        (each with mean / ci_low / ci_high), plus ``baseline_keys`` count.
    """
    edges = np.asarray(ppi_edges)
    if edges.ndim != 2 or edges.shape[1] < 2:
        raise ValueError("ppi_edges must be 2D with shape (E, 2) or (E, 3)")
    for r in dropout_rates:
        if not 0.0 <= r < 1.0:
            raise ValueError(f"dropout_rates must lie in [0, 1); got {r}")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = np.random.default_rng(seed)
    baseline = _to_dict_or_array(attribute_fn(edges))

    results: Dict[str, Any] = {"baseline_keys": len(baseline)}
    for rate in dropout_rates:
        jaccards = []
        overlaps = []
        n_keep = max(1, int(round((1.0 - rate) * edges.shape[0])))
        for trial in range(n_trials):
            t_rng = np.random.default_rng(seed + trial + 1 + int(rate * 1e6))
            keep = t_rng.choice(edges.shape[0], size=n_keep, replace=False)
            sub = edges[np.sort(keep)]
            try:
                perturbed = _to_dict_or_array(attribute_fn(sub))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "edge_dropout_stability: attribute_fn failed at rate=%.3f trial=%d: %s",
                    rate, trial, exc,
                )
                continue
            ov = attribution_overlap(baseline, perturbed)
            if not np.isnan(ov["jaccard"]):
                jaccards.append(ov["jaccard"])
            if not np.isnan(ov["top_k_overlap"]):
                overlaps.append(ov["top_k_overlap"])
        results[f"{rate:.3f}"] = {
            "jaccard": _bootstrap_ci(jaccards, rng=rng),
            "top_k_overlap": _bootstrap_ci(overlaps, rng=rng),
            "n_completed_trials": len(jaccards),
        }
    return results


def pathway_holdout_generalization(
    model_fn: Callable[[Iterable[Any]], np.ndarray],
    heldout_pathway_ids: Iterable[Any],
    predictions: np.ndarray,
) -> float:
    """Compare model predictions on held-out pathways against caller predictions.

    Computes Pearson correlation between ``model_fn(heldout_pathway_ids)`` and
    ``predictions`` aligned by index. Caller is responsible for providing
    real predictions; this function does not fabricate any data.

    Args:
        model_fn: Callable producing a 1D array of pathway-level scores
            for the given pathway ids.
        heldout_pathway_ids: Iterable of pathway identifiers held out from
            training; passed verbatim to ``model_fn``.
        predictions: Reference predictions, shape (P,), aligned with
            ``heldout_pathway_ids`` order.

    Returns:
        Pearson correlation in [-1, 1]; raises if shapes mismatch.
    """
    ids = list(heldout_pathway_ids)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 1:
        raise ValueError("predictions must be 1D")
    if len(ids) != predictions.shape[0]:
        raise ValueError(
            f"heldout_pathway_ids and predictions length mismatch: "
            f"{len(ids)} vs {predictions.shape[0]}"
        )
    if predictions.size < 2:
        raise ValueError("need at least 2 pathways for correlation")
    model_out = np.asarray(model_fn(ids), dtype=np.float64)
    if model_out.shape != predictions.shape:
        raise ValueError(
            f"model_fn output shape {model_out.shape} must match predictions {predictions.shape}"
        )
    a = predictions - predictions.mean()
    b = model_out - model_out.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)
