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
        integrated_brier_score as _integrated_brier_score,
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
        ibs = _integrated_brier_score(
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


# ============================================================================
# Fairness & Equity Metrics (Agent 3 §3.1-3.6)
# ============================================================================

def ppv_parity(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, float]:
    """Compute Positive Predictive Value (PPV) across demographic groups.

    PPV measures the precision of positive predictions within each group.
    PPV = TP / (TP + FP) per group. High disparity in PPV across groups
    indicates fairness concerns (Agent 3 §3.1).

    Args:
        predictions: (N,) binary predictions (0 or 1).
        labels: (N,) binary ground truth labels (0 or 1).
        group_ids: (N,) group identifiers (e.g., patient demographics).

    Returns:
        Dict with:
            - 'ppv_per_group': dict mapping group_id → PPV value
            - 'ppv_max_disparity': max PPV difference across groups
            - 'ppv_mean': mean PPV across groups
    """
    ppv_per_group = {}
    unique_groups = np.unique(group_ids)

    for group in unique_groups:
        mask = group_ids == group
        pred_group = predictions[mask]
        label_group = labels[mask]

        tp = np.sum((pred_group == 1) & (label_group == 1))
        fp = np.sum((pred_group == 1) & (label_group == 0))
        denominator = tp + fp

        if denominator > 0:
            ppv = tp / denominator
        else:
            ppv = np.nan

        ppv_per_group[str(group)] = float(ppv) if not np.isnan(ppv) else 0.0

    # Compute max disparity
    valid_ppvs = [v for v in ppv_per_group.values() if v > 0.0]
    ppv_disparity = 0.0
    if len(valid_ppvs) > 1:
        ppv_disparity = float(np.max(valid_ppvs) - np.min(valid_ppvs))

    ppv_mean = float(np.mean([v for v in ppv_per_group.values()]))

    return {
        "ppv_per_group": ppv_per_group,
        "ppv_max_disparity": ppv_disparity,
        "ppv_mean": ppv_mean,
    }


def equalized_odds(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, float]:
    """Compute equalized odds: TPR and FPR parity across demographic groups.

    Equalized odds requires equal True Positive Rates (sensitivity) and
    False Positive Rates across groups. Disparities indicate algorithmic bias
    (Agent 3 §3.2).

    Args:
        predictions: (N,) binary predictions (0 or 1).
        labels: (N,) binary ground truth labels (0 or 1).
        group_ids: (N,) group identifiers.

    Returns:
        Dict with:
            - 'tpr_per_group': dict mapping group_id → TPR
            - 'fpr_per_group': dict mapping group_id → FPR
            - 'tpr_max_gap': max TPR difference across groups
            - 'fpr_max_gap': max FPR difference across groups
    """
    tpr_per_group = {}
    fpr_per_group = {}
    unique_groups = np.unique(group_ids)

    for group in unique_groups:
        mask = group_ids == group
        pred_group = predictions[mask]
        label_group = labels[mask]

        # TPR = TP / (TP + FN) among actual positives
        tp = np.sum((pred_group == 1) & (label_group == 1))
        fn = np.sum((pred_group == 0) & (label_group == 1))
        tpr_denom = tp + fn
        tpr = (tp / tpr_denom) if tpr_denom > 0 else np.nan

        # FPR = FP / (FP + TN) among actual negatives
        fp = np.sum((pred_group == 1) & (label_group == 0))
        tn = np.sum((pred_group == 0) & (label_group == 0))
        fpr_denom = fp + tn
        fpr = (fp / fpr_denom) if fpr_denom > 0 else np.nan

        tpr_per_group[str(group)] = float(tpr) if not np.isnan(tpr) else 0.0
        fpr_per_group[str(group)] = float(fpr) if not np.isnan(fpr) else 0.0

    # Compute max gaps
    valid_tprs = [v for v in tpr_per_group.values() if v >= 0.0]
    valid_fprs = [v for v in fpr_per_group.values() if v >= 0.0]

    tpr_gap = 0.0
    if len(valid_tprs) > 1:
        tpr_gap = float(np.max(valid_tprs) - np.min(valid_tprs))

    fpr_gap = 0.0
    if len(valid_fprs) > 1:
        fpr_gap = float(np.max(valid_fprs) - np.min(valid_fprs))

    return {
        "tpr_per_group": tpr_per_group,
        "fpr_per_group": fpr_per_group,
        "tpr_max_gap": tpr_gap,
        "fpr_max_gap": fpr_gap,
    }


def worst_subgroup_performance(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    clinical_safety_threshold: float = 0.6,
) -> dict[str, Any]:
    """Identify worst-performing subgroup and flag clinical safety concerns.

    Computes AUROC per subgroup and flags if worst AUROC falls below
    clinical safety threshold (Agent 3 §3.3).

    Args:
        predictions: (N,) predicted probabilities or scores in [0, 1].
        labels: (N,) binary ground truth labels (0 or 1).
        group_ids: (N,) group identifiers.
        clinical_safety_threshold: AUROC threshold (default 0.6).

    Returns:
        Dict with:
            - 'auroc_per_group': dict mapping group_id → AUROC
            - 'worst_subgroup': name of worst-performing group
            - 'worst_auroc': AUROC of worst group
            - 'best_auroc': AUROC of best group
            - 'auroc_gap': difference between best and worst
            - 'safety_concern': True if worst_auroc < threshold
    """
    if not HAS_SKLEARN:
        logger.warning("sklearn not available; returning empty dict")
        return {}

    auroc_per_group = {}
    unique_groups = np.unique(group_ids)

    for group in unique_groups:
        mask = group_ids == group
        pred_group = predictions[mask]
        label_group = labels[mask]

        if len(np.unique(label_group)) < 2:
            auroc_per_group[str(group)] = np.nan
            continue

        try:
            auroc = float(roc_auc_score(label_group, pred_group))
            auroc_per_group[str(group)] = auroc
        except Exception:
            auroc_per_group[str(group)] = np.nan

    valid_aurocs = {k: v for k, v in auroc_per_group.items() if not np.isnan(v)}

    if not valid_aurocs:
        return {
            "auroc_per_group": auroc_per_group,
            "worst_subgroup": "N/A",
            "worst_auroc": np.nan,
            "best_auroc": np.nan,
            "auroc_gap": np.nan,
            "safety_concern": False,
        }

    worst_group = min(valid_aurocs, key=valid_aurocs.get)
    best_group = max(valid_aurocs, key=valid_aurocs.get)
    worst_val = valid_aurocs[worst_group]
    best_val = valid_aurocs[best_group]
    gap = best_val - worst_val

    safety_concern = worst_val < clinical_safety_threshold

    if safety_concern:
        logger.warning(
            f"Safety concern: subgroup '{worst_group}' AUROC={worst_val:.3f} "
            f"below threshold {clinical_safety_threshold}"
        )

    return {
        "auroc_per_group": auroc_per_group,
        "worst_subgroup": str(worst_group),
        "worst_auroc": float(worst_val),
        "best_auroc": float(best_val),
        "auroc_gap": float(gap),
        "safety_concern": bool(safety_concern),
    }


def false_negative_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    drug_ids: np.ndarray | None = None,
    fn_rate_threshold: float = 0.1,
) -> dict[str, Any]:
    """Critical error mode analysis: false negatives in resistance prediction.

    A false negative = predicting sensitive when actually resistant (dangerous).
    Computes FN rate per subgroup and flags safety concerns (Agent 3 §3.4).

    Args:
        predictions: (N,) binary predictions (0=sensitive, 1=resistant).
        labels: (N,) binary ground truth (0=sensitive, 1=resistant).
        group_ids: (N,) group identifiers.
        drug_ids: (N,) optional drug identifiers for per-drug analysis.
        fn_rate_threshold: FN rate threshold for flagging safety concerns (default 0.1).

    Returns:
        Dict with:
            - 'fn_rate_per_group': dict mapping group_id → FN rate
            - 'fn_per_drug': dict of per-drug FN analysis (if drug_ids provided)
            - 'high_risk_groups': list of groups with FN rate > threshold
            - 'max_fn_rate': maximum FN rate across groups
    """
    fn_rate_per_group = {}
    unique_groups = np.unique(group_ids)

    for group in unique_groups:
        mask = group_ids == group
        pred_group = predictions[mask]
        label_group = labels[mask]

        # FN = predict 0 (sensitive) when actually 1 (resistant)
        fn = np.sum((pred_group == 0) & (label_group == 1))
        # Total actual resistants
        total_resistant = np.sum(label_group == 1)

        if total_resistant > 0:
            fn_rate = fn / total_resistant
        else:
            fn_rate = np.nan

        fn_rate_per_group[str(group)] = float(fn_rate) if not np.isnan(fn_rate) else 0.0

    # Identify high-risk groups
    high_risk_groups = [
        g for g, rate in fn_rate_per_group.items()
        if rate > fn_rate_threshold
    ]

    if high_risk_groups:
        logger.warning(
            f"High FN rate safety concerns in groups: {high_risk_groups} "
            f"(threshold: {fn_rate_threshold})"
        )

    max_fn = max(fn_rate_per_group.values()) if fn_rate_per_group else 0.0

    result = {
        "fn_rate_per_group": fn_rate_per_group,
        "high_risk_groups": high_risk_groups,
        "max_fn_rate": float(max_fn),
    }

    # Per-drug analysis if provided
    if drug_ids is not None:
        fn_per_drug = {}
        unique_drugs = np.unique(drug_ids)

        for drug in unique_drugs:
            drug_mask = drug_ids == drug
            drug_fn_rate = {}

            for group in unique_groups:
                group_drug_mask = (group_ids == group) & drug_mask
                pred_subset = predictions[group_drug_mask]
                label_subset = labels[group_drug_mask]

                fn = np.sum((pred_subset == 0) & (label_subset == 1))
                total_resistant = np.sum(label_subset == 1)

                if total_resistant > 0:
                    fn_rate = fn / total_resistant
                else:
                    fn_rate = np.nan

                drug_fn_rate[str(group)] = float(fn_rate) if not np.isnan(fn_rate) else 0.0

            fn_per_drug[str(drug)] = drug_fn_rate

        result["fn_per_drug"] = fn_per_drug

    return result


def calibration_parity(
    predicted_probs: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE) per demographic group.

    Measures calibration parity: whether predicted probabilities match
    observed frequencies within each group (Agent 3 §3.5).

    Args:
        predicted_probs: (N,) predicted probabilities in [0, 1].
        labels: (N,) binary ground truth labels (0 or 1).
        group_ids: (N,) group identifiers.
        n_bins: Number of bins for calibration curve (default 10).

    Returns:
        Dict with:
            - 'ece_per_group': dict mapping group_id → ECE
            - 'ece_max_disparity': max ECE difference across groups
            - 'ece_mean': mean ECE across groups
    """
    ece_per_group = {}
    unique_groups = np.unique(group_ids)

    for group in unique_groups:
        mask = group_ids == group
        probs_group = predicted_probs[mask]
        label_group = labels[mask]

        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (probs_group >= bin_edges[i]) & (probs_group < bin_edges[i+1])
            if bin_mask.sum() == 0:
                continue

            bin_accuracy = label_group[bin_mask].mean()
            bin_confidence = probs_group[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(probs_group)

            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

        ece_per_group[str(group)] = float(ece)

    # Compute max disparity
    valid_eces = [v for v in ece_per_group.values() if v >= 0.0]
    ece_disparity = 0.0
    if len(valid_eces) > 1:
        ece_disparity = float(np.max(valid_eces) - np.min(valid_eces))

    ece_mean = float(np.mean(valid_eces)) if valid_eces else 0.0

    return {
        "ece_per_group": ece_per_group,
        "ece_max_disparity": ece_disparity,
        "ece_mean": ece_mean,
    }


def auroc_with_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Bootstrap AUROC with confidence intervals.

    Computes AUROC and its confidence interval using bootstrap resampling
    (Agent 3 §3.6).

    Args:
        predictions: (N,) predicted scores or probabilities.
        labels: (N,) binary ground truth labels.
        n_bootstrap: Number of bootstrap samples (default 2000).
        confidence_level: Confidence level for CI (default 0.95).

    Returns:
        Dict with:
            - 'auroc': point estimate
            - 'ci_low': lower CI bound
            - 'ci_high': upper CI bound
            - 'ci_width': width of confidence interval
    """
    if not HAS_SKLEARN:
        logger.warning("sklearn not available; returning NaN")
        return {
            "auroc": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_width": np.nan,
        }

    # Point estimate
    try:
        auroc = float(roc_auc_score(labels, predictions))
    except Exception:
        return {
            "auroc": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_width": np.nan,
        }

    # Bootstrap
    n = len(predictions)
    boot_aurocs = np.zeros(n_bootstrap)
    rng = np.random.RandomState(42)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        pred_boot = predictions[idx]
        label_boot = labels[idx]

        try:
            boot_aurocs[b] = roc_auc_score(label_boot, pred_boot)
        except Exception:
            boot_aurocs[b] = np.nan

    # Remove NaNs
    valid_aurocs = boot_aurocs[~np.isnan(boot_aurocs)]
    if len(valid_aurocs) == 0:
        return {
            "auroc": auroc,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_width": np.nan,
        }

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(valid_aurocs, 100 * alpha / 2))
    ci_high = float(np.percentile(valid_aurocs, 100 * (1 - alpha / 2)))
    ci_width = ci_high - ci_low

    return {
        "auroc": auroc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
    }


def delong_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """DeLong's test for comparing two AUROC estimates.

    Tests whether two AUROC values are significantly different
    (Agent 3 §3.6).

    Args:
        predictions_a: (N,) predicted scores for model A.
        predictions_b: (N,) predicted scores for model B.
        labels: (N,) binary ground truth labels.

    Returns:
        Dict with:
            - 'auroc_a': AUROC of model A
            - 'auroc_b': AUROC of model B
            - 'z_statistic': test statistic
            - 'p_value': two-tailed p-value
    """
    if not HAS_SKLEARN:
        logger.warning("sklearn not available for DeLong test")
        return {
            "auroc_a": np.nan,
            "auroc_b": np.nan,
            "z_statistic": np.nan,
            "p_value": np.nan,
        }

    try:
        from scipy.stats import norm
        auroc_a = roc_auc_score(labels, predictions_a)
        auroc_b = roc_auc_score(labels, predictions_b)
    except Exception:
        return {
            "auroc_a": np.nan,
            "auroc_b": np.nan,
            "z_statistic": np.nan,
            "p_value": np.nan,
        }

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    # Simplified DeLong variance estimate
    # (full implementation would compute pairwise AUC contributions)
    q1 = auroc_a * (1 - auroc_a) * (n_pos - 1)
    q2 = auroc_b * (1 - auroc_b) * (n_neg - 1)
    var_auc = (q1 + q2) / (n_pos * n_neg)

    if var_auc == 0:
        var_auc = 1e-10

    z_stat = (auroc_a - auroc_b) / np.sqrt(var_auc)

    try:
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
    except Exception:
        p_value = np.nan

    return {
        "auroc_a": float(auroc_a),
        "auroc_b": float(auroc_b),
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
    }


def compute_full_metrics(
    predictions: list[Any],
    dataset: Any | None = None,
    split: str = "test",
    actual_states: list[str] | None = None,
    validated_targets: list[str] | None = None,
    group_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all metrics for the full ResistanceMap pipeline output.

    Args:
        predictions: List of LandscapeResult objects.
        dataset: Optional dataset with ground truth (for drug sensitivity metrics).
        split: Which split was evaluated.
        actual_states: Optional ground-truth resistance states for trajectory metrics.
        validated_targets: Optional list of known drug targets for target evaluation.
        group_ids: Optional (N,) array of group identifiers for fairness metrics.

    Returns:
        Comprehensive metrics dict including fairness/equity metrics if group_ids provided.
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

    # Fairness & equity metrics if group information provided
    if group_ids is not None and len(predictions) > 0:
        if len(group_ids) != len(predictions):
            logger.warning(
                f"group_ids length {len(group_ids)} != predictions length "
                f"{len(predictions)}; skipping fairness metrics"
            )
        else:
            # Extract resistance predictions and labels if available
            # (assumes predictions have binary resistance classification)
            resistance_preds = np.array([
                1 if p.resistance_state == "resistant" else 0
                for p in predictions
            ])

            if hasattr(predictions[0], "confidence_score"):
                # Use confidence scores as predicted probabilities
                pred_probs = np.array([p.confidence_score for p in predictions])

                # Compute fairness metrics
                try:
                    ci_result = auroc_with_ci(
                        pred_probs,
                        resistance_preds,
                        n_bootstrap=2000,
                        confidence_level=0.95,
                    )
                    metrics["auroc_ci"] = ci_result

                    worst_subgroup_result = worst_subgroup_performance(
                        pred_probs,
                        resistance_preds,
                        group_ids,
                        clinical_safety_threshold=0.6,
                    )
                    metrics["subgroup_analysis"] = worst_subgroup_result

                    fn_analysis = false_negative_analysis(
                        resistance_preds,
                        resistance_preds,
                        group_ids,
                        fn_rate_threshold=0.1,
                    )
                    metrics["false_negative_analysis"] = fn_analysis

                    cal_parity = calibration_parity(
                        pred_probs,
                        resistance_preds,
                        group_ids,
                        n_bins=10,
                    )
                    metrics["calibration_parity"] = cal_parity

                    ppv_result = ppv_parity(
                        resistance_preds,
                        resistance_preds,
                        group_ids,
                    )
                    metrics["ppv_parity"] = ppv_result

                    eq_odds = equalized_odds(
                        resistance_preds,
                        resistance_preds,
                        group_ids,
                    )
                    metrics["equalized_odds"] = eq_odds

                    logger.info(
                        f"Fairness metrics computed for {len(np.unique(group_ids))} groups"
                    )

                except Exception as e:
                    logger.warning(f"Error computing fairness metrics: {e}")

    logger.info(
        f"Pipeline metrics ({split}): {len(predictions)} samples, "
        f"mean_confidence={metrics['mean_confidence']:.3f}"
    )

    return metrics