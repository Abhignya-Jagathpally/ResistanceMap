"""Evaluation metrics for ResistanceMap pipeline.

Combines metrics from:
    - MyeloMemory: VAE reconstruction, latent space quality, stability calibration
    - R2: Survival analysis (C-index, IBS, AUC(t))
    - ResistanceMap-specific: trajectory accuracy, target hit rate

Provides comprehensive evaluation across all pipeline stages.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        mean_squared_error,
    )
    from scipy.stats import spearmanr
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sksurv.metrics import (
        concordance_index_censored,
        integrated_brier_score,
        cumulative_dynamic_auc,
    )
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


# ============================================================================
# VAE Metrics (from MyeloMemory)
# ============================================================================

def reconstruction_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute mean squared error for VAE reconstruction.

    Args:
        predicted: (N, E) reconstructed epigenomic profiles.
        target: (N, E) true epigenomic profiles.

    Returns:
        Scalar MSE.
    """
    return torch.nn.functional.mse_loss(predicted, target).item()


def latent_space_metrics(
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> dict[str, float]:
    """Compute latent space quality metrics.

    Args:
        mu: (N, L) mean vectors.
        log_var: (N, L) log-variance vectors.

    Returns:
        Dict with 'mean_kl', 'active_units', 'latent_variance'.
    """
    # KL divergence per dimension
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    mean_kl = kl_per_dim.mean().item()

    # Active units: dimensions where KL > 0.01 (not collapsed)
    kl_per_unit = kl_per_dim.mean(dim=0)
    active_units = (kl_per_unit > 0.01).sum().item()

    # Overall latent variance
    latent_var = mu.var(dim=0).mean().item()

    return {
        "mean_kl": mean_kl,
        "active_units": int(active_units),
        "total_units": mu.shape[1],
        "latent_variance": latent_var,
    }


def stability_calibration_metrics(
    predicted_scores: torch.Tensor,
    drug_sensitivity_variance: torch.Tensor,
) -> dict[str, float]:
    """Evaluate stability scorer calibration.

    The stability score should be inversely correlated with drug sensitivity
    variance (high stability → consistent resistance → low variance).

    Args:
        predicted_scores: (N,) stability scores.
        drug_sensitivity_variance: (N,) variance of IC50 across drugs.

    Returns:
        Dict with 'spearman_rho', 'calibration_mse'.
    """
    # Filter NaN
    mask = ~(torch.isnan(predicted_scores) | torch.isnan(drug_sensitivity_variance))
    pred = predicted_scores[mask].cpu().numpy()
    target = drug_sensitivity_variance[mask].cpu().numpy()

    if len(pred) < 3:
        return {"spearman_rho": 0.0, "calibration_mse": float("inf")}

    # Stability should be inversely correlated with drug variance
    rho = 0.0
    if HAS_SKLEARN:
        rho, _ = spearmanr(pred, target)
        rho = float(rho) if not np.isnan(rho) else 0.0

    # Target: high stability → low variance (inverted and normalized)
    target_norm = 1.0 - (target - target.min()) / (target.max() - target.min() + 1e-8)
    cal_mse = float(mean_squared_error(target_norm, pred)) if HAS_SKLEARN else 0.0

    return {
        "spearman_rho": rho,
        "calibration_mse": cal_mse,
    }


# ============================================================================
# Survival Metrics (from R2)
# ============================================================================

def concordance_index(
    y_event: np.ndarray,
    y_time: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Harrell's C-index (concordance index).

    Measures discrimination: probability that model correctly ranks
    survival times for a pair of subjects.

    Args:
        y_event: (N,) event indicator (1 = event, 0 = censored).
        y_time: (N,) survival times.
        predictions: (N,) predicted risk scores.

    Returns:
        C-index in [0, 1]. 0.5 = random, 1.0 = perfect discrimination.
    """
    if not HAS_SKSURV:
        logger.warning("sksurv not available; returning 0.0")
        return 0.0

    c_index, _, _, _, _ = concordance_index_censored(
        y_event.astype(bool),
        y_time,
        predictions,
    )
    return float(c_index)


def time_dependent_auc(
    y_event: np.ndarray,
    y_time: np.ndarray,
    predictions: np.ndarray,
    times: np.ndarray | None = None,
) -> dict[float, float]:
    """Time-dependent AUC (dynamic AUC).

    Measures discrimination at specific timepoints.

    Args:
        y_event: (N,) event indicator.
        y_time: (N,) survival times.
        predictions: (N,) predicted risk scores.
        times: Timepoints for evaluation. If None, use percentiles of event times.

    Returns:
        Dict mapping timepoint → AUC value.
    """
    if not HAS_SKSURV:
        logger.warning("sksurv not available; returning empty dict")
        return {}

    if times is None:
        event_times = y_time[y_event.astype(bool)]
        times = np.percentile(event_times, [25, 50, 75])

    auc_dict = {}
    for t in times:
        try:
            auc, _, _ = cumulative_dynamic_auc(
                y_event.astype(bool),
                y_time,
                predictions,
                times=np.array([t]),
            )
            auc_dict[t] = auc[0]
        except Exception:
            auc_dict[t] = np.nan

    return auc_dict


def integrated_brier_score(
    y_event: np.ndarray,
    y_time: np.ndarray,
    survival_probs: np.ndarray,
    times: np.ndarray | None = None,
) -> float:
    """Integrated Brier Score (IBS) over time.

    Measures calibration: mean squared error between predicted and observed
    survival probabilities integrated over time.

    Args:
        y_event: (N,) event indicator.
        y_time: (N,) survival times.
        survival_probs: (N, T) predicted survival probabilities at T timepoints.
        times: Timepoints. If None, use percentiles of event times.

    Returns:
        IBS scalar.
    """
    if not HAS_SKSURV:
        logger.warning("sksurv not available; returning NaN")
        return np.nan

    if times is None:
        times = np.percentile(y_time[y_event.astype(bool)], [25, 50, 75])

    try:
        ibs = integrated_brier_score(
            y_event.astype(bool),
            y_time,
            survival_probs,
            times=times,
        )
        return float(ibs)
    except Exception as e:
        logger.warning(f"IBS computation failed: {e}")
        return np.nan


# ============================================================================
# ResistanceMap-specific Metrics
# ============================================================================

def trajectory_accuracy(
    predicted_states: list[str],
    actual_states: list[str],
) -> float:
    """Compute how well predicted resistance trajectories match actual evolution.

    Args:
        predicted_states: Predicted resistance state sequence.
        actual_states: Observed resistance state sequence.

    Returns:
        Fraction of timepoints where prediction matched actual state.
    """
    assert len(predicted_states) == len(actual_states), \
        f"Length mismatch: {len(predicted_states)} vs {len(actual_states)}"

    if len(predicted_states) == 0:
        return 0.0

    n_correct = sum(1 for p, a in zip(predicted_states, actual_states) if p == a)
    return n_correct / len(predicted_states)


def target_hit_rate(
    predicted_targets: list[str],
    validated_targets: list[str],
) -> float:
    """Compute fraction of predicted targets that are validated drug targets.

    Args:
        predicted_targets: List of predicted intervention target protein names.
        validated_targets: List of known/validated drug target protein names.

    Returns:
        Fraction of predictions that are in the validated set.
    """
    if len(predicted_targets) == 0:
        return 0.0

    validated_set = set(validated_targets)
    n_hits = sum(1 for t in predicted_targets if t in validated_set)
    return n_hits / len(predicted_targets)


def drug_resistance_metrics(
    predicted_ic50: torch.Tensor,
    true_ic50: torch.Tensor,
    drug_names: list[str],
    resistance_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute per-drug and aggregate resistance prediction metrics.

    Args:
        predicted_ic50: (N, D) predicted IC50 values.
        true_ic50: (N, D) true IC50 values (may contain NaN).
        drug_names: List of D drug names.
        resistance_threshold: IC50 threshold for binary resistant/sensitive.

    Returns:
        Dict with per-drug AUROC, AUPRC, MSE, and aggregate metrics.
    """
    if not HAS_SKLEARN:
        logger.warning("sklearn not available; returning empty metrics")
        return {}

    metrics = {}
    all_aurocs = []

    for d, drug in enumerate(drug_names):
        pred = predicted_ic50[:, d]
        true = true_ic50[:, d]

        # Filter NaN
        mask = ~torch.isnan(true)
        if mask.sum() < 5:
            continue

        pred_np = pred[mask].cpu().numpy()
        true_np = true[mask].cpu().numpy()

        # MSE
        mse = float(mean_squared_error(true_np, pred_np))
        metrics[f"{drug}/mse"] = mse

        # Binary classification metrics
        binary_true = (true_np > resistance_threshold).astype(int)
        if len(np.unique(binary_true)) == 2:
            try:
                auroc = float(roc_auc_score(binary_true, pred_np))
                auprc = float(average_precision_score(binary_true, pred_np))
                metrics[f"{drug}/auroc"] = auroc
                metrics[f"{drug}/auprc"] = auprc
                all_aurocs.append(auroc)
            except Exception:
                pass

    if all_aurocs:
        metrics["mean_auroc"] = float(np.mean(all_aurocs))

    return metrics


def compute_full_metrics(
    predictions: list[Any],
    dataset: Any | None = None,
    split: str = "test",
    actual_states: list[str] | None = None,
    validated_targets: list[str] | None = None,
) -> dict[str, float]:
    """Compute all metrics for the full ResistanceMap pipeline output.

    Args:
        predictions: List of LandscapeResult objects.
        dataset: Optional dataset with ground truth (for drug sensitivity metrics).
        split: Which split was evaluated.
        actual_states: Optional ground-truth resistance states for trajectory metrics.
        validated_targets: Optional list of known drug targets for target evaluation.

    Returns:
        Comprehensive metrics dict.
    """
    metrics = {"split": split, "n_samples": len(predictions)}

    if not predictions:
        return metrics

    # Aggregate confidence scores
    confidence_scores = torch.tensor([p.confidence_score for p in predictions])
    metrics["mean_confidence"] = confidence_scores.mean().item()
    metrics["std_confidence"] = confidence_scores.std().item()

    # State distribution
    states = [p.resistance_state for p in predictions]
    unique_states = set(states)
    for state in unique_states:
        count = sum(1 for s in states if s == state)
        metrics[f"n_{state}"] = count

    # Trajectory accuracy if available
    if actual_states is not None:
        predicted_states = [p.resistance_state for p in predictions]
        if len(predicted_states) == len(actual_states):
            traj_acc = trajectory_accuracy(predicted_states, actual_states)
            metrics["trajectory_accuracy"] = traj_acc
            logger.info(f"Trajectory accuracy: {traj_acc:.3f}")

    # Target hit rate if available
    if validated_targets is not None and predictions:
        all_predicted_targets = []
        for p in predictions:
            for target_name, _, _ in p.top_intervention_targets:
                all_predicted_targets.append(target_name)
        if all_predicted_targets:
            hit_rate = target_hit_rate(all_predicted_targets, validated_targets)
            metrics["target_hit_rate"] = hit_rate
            logger.info(f"Target hit rate: {hit_rate:.3f}")

    # Bootstrap confidence intervals for key metrics
    n_bootstrap = 1000
    n = len(predictions)
    if n >= 10:
        rng = np.random.RandomState(42)
        boot_conf = np.zeros(n_bootstrap)
        conf_np = confidence_scores.numpy()
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_conf[b] = conf_np[idx].mean()
        metrics["mean_confidence_ci95_low"] = float(np.percentile(boot_conf, 2.5))
        metrics["mean_confidence_ci95_high"] = float(np.percentile(boot_conf, 97.5))

    logger.info(
        f"Pipeline metrics ({split}): {len(predictions)} samples, "
        f"mean_confidence={metrics['mean_confidence']:.3f}"
    )

    return metrics
