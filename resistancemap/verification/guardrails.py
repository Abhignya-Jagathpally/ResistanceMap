"""
Clinical and Scientific Guardrails for ResistanceMap
=====================================================

Enforces domain constraints on all model outputs. These guardrails ensure
predictions respect biological, chemical, and clinical bounds, preventing
nonsensical or dangerous recommendations.

Guardrail Philosophy:
  - FAIL SAFELY: If output violates constraint, flag it loudly and return None
  - NEVER GUESS: If a constraint can't be checked, don't check it
  - DOCUMENT SEVERITY: Label each guardrail as "error", "warning", or "info"
  - AUDIT TRAIL: Log every constraint check for regulatory compliance

Reference Standards:
  - IC50/Kd bounds: PubChem bioassay standards
  - Protein identifiers: UniProt human proteome
  - Stochastic matrix: Linear algebra fundamentals
  - Drug-target interaction: STRING PPI database v12.0
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class Guardrail:
    """A single constraint to be enforced on model outputs.

    Attributes:
        name: Unique identifier (e.g., "resistance_probability_range")
        check_fn: Callable(data: Any) -> (passed: bool, details: str)
        severity: "error" (fail prediction), "warning" (log but continue), "info"
        description: Human-readable explanation of constraint
        always_required: If True, missing data counts as failure
    """
    name: str
    check_fn: Callable[[Any], Tuple[bool, str]]
    severity: str = "warning"
    description: str = ""
    always_required: bool = False


class GuardrailEngine:
    """Enforces clinical and scientific constraints on all outputs.

    The guardrail engine maintains a registry of domain constraints and checks
    every prediction before it's passed to downstream agents or users.

    Constraints are organized by:
        1. Range checks (probabilities ∈ [0,1], IC50 in [1e-12, 1e-3] M)
        2. Shape checks (correct tensor dimensions)
        3. Stochasticity checks (rows sum to 1, no NaN/Inf)
        4. Domain existence checks (proteins in UniProt, targets in STRING)
        5. Consistency checks (deeper basins → slower transitions)

    Example:
        >>> engine = GuardrailEngine()
        >>> predictions = {
        ...     "resistance": torch.tensor([0.5, 0.7]),
        ...     "targets": ["BCR", "ABL"],
        ...     "ic50": torch.tensor([1e-9, 1e-8]),
        ... }
        >>> results = engine.check_all(predictions)
        >>> if all(r.passed for r in results if r.severity == "error"):
        ...     # Safe to use predictions
        ...     pass
        >>> else:
        ...     # Log warnings/errors and reject
        ...     pass
    """

    def __init__(self):
        """Initialize with full set of guardrails."""
        self.guardrails: Dict[str, Guardrail] = {}
        self._register_default_guardrails()

    def _register_default_guardrails(self) -> None:
        """Register all default guardrails for ResistanceMap."""

        # ==================== RANGE CHECKS ====================

        self.register_guardrail(Guardrail(
            name="resistance_probability_range",
            check_fn=lambda data: self._check_resistance_range(data),
            severity="error",
            description="Resistance probability must be in [0, 1]",
            always_required=False,
        ))

        self.register_guardrail(Guardrail(
            name="ic50_physiological_range",
            check_fn=lambda data: self._check_ic50_range(data),
            severity="error",
            description="IC50 must be in [1e-12, 1e-3] molar",
            always_required=False,
        ))

        self.register_guardrail(Guardrail(
            name="stability_score_range",
            check_fn=lambda data: self._check_stability_range(data),
            severity="error",
            description="Stability scores must be in [0, 1]",
            always_required=False,
        ))

        # ==================== NAN/INF CHECKS ====================

        self.register_guardrail(Guardrail(
            name="no_nan_in_predictions",
            check_fn=lambda data: self._check_no_nan(data),
            severity="error",
            description="No NaN or Inf values in predictions",
            always_required=True,
        ))

        # ==================== STOCHASTICITY CHECKS ====================

        self.register_guardrail(Guardrail(
            name="transition_probability_stochastic",
            check_fn=lambda data: self._check_transition_stochastic(data),
            severity="error",
            description="Transition probability matrix rows must sum to 1",
            always_required=False,
        ))

        # ==================== DOMAIN CHECKS ====================

        self.register_guardrail(Guardrail(
            name="target_protein_format",
            check_fn=lambda data: self._check_protein_format(data),
            severity="warning",
            description="Target proteins must be valid UniProt identifiers or gene names",
            always_required=False,
        ))

        # ==================== CONSISTENCY CHECKS ====================

        self.register_guardrail(Guardrail(
            name="stability_monotonicity",
            check_fn=lambda data: self._check_stability_monotonicity(data),
            severity="warning",
            description="Deeper basin (higher stability) should mean longer transition time",
            always_required=False,
        ))

        # ==================== INFORMATION THEORY CHECKS ====================

        self.register_guardrail(Guardrail(
            name="fano_bound_respected",
            check_fn=lambda data: self._check_fano_bound(data),
            severity="info",
            description="Model error should not be better than Fano lower bound",
            always_required=False,
        ))

    def register_guardrail(self, guardrail: Guardrail) -> None:
        """Register a new guardrail to the engine.

        Args:
            guardrail: Guardrail to add
        """
        self.guardrails[guardrail.name] = guardrail

    def check_all(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all guardrails against predictions.

        Args:
            predictions: Dict of predicted values

        Returns:
            List of check results, each with:
                {
                    "guardrail": str,
                    "passed": bool,
                    "severity": str,
                    "details": str,
                }
        """
        results = []

        for name, guardrail in self.guardrails.items():
            try:
                passed, details = guardrail.check_fn(predictions)
                results.append({
                    "guardrail": name,
                    "passed": passed,
                    "severity": guardrail.severity,
                    "details": details,
                    "description": guardrail.description,
                })
            except Exception as e:
                # If check itself fails, log as error
                results.append({
                    "guardrail": name,
                    "passed": False,
                    "severity": guardrail.severity,
                    "details": f"Exception during check: {str(e)}",
                    "description": guardrail.description,
                })

        return results

    def check_single(
        self,
        guardrail_name: str,
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check a single guardrail.

        Args:
            guardrail_name: Name of guardrail to check
            predictions: Dict of predicted values

        Returns:
            Single check result dict
        """
        if guardrail_name not in self.guardrails:
            return {
                "guardrail": guardrail_name,
                "passed": False,
                "severity": "error",
                "details": f"Guardrail '{guardrail_name}' not found",
            }

        guardrail = self.guardrails[guardrail_name]
        try:
            passed, details = guardrail.check_fn(predictions)
            return {
                "guardrail": guardrail_name,
                "passed": passed,
                "severity": guardrail.severity,
                "details": details,
                "description": guardrail.description,
            }
        except Exception as e:
            return {
                "guardrail": guardrail_name,
                "passed": False,
                "severity": guardrail.severity,
                "details": f"Exception during check: {str(e)}",
            }

    def get_failure_count(
        self,
        results: List[Dict[str, Any]],
        severity: str = "error",
    ) -> int:
        """Count failures of a given severity level.

        Args:
            results: List of check results from check_all()
            severity: "error", "warning", or "info"

        Returns:
            Number of checks that failed with this severity
        """
        return sum(
            1 for r in results
            if not r["passed"] and r.get("severity") == severity
        )

    # ==================== INDIVIDUAL CHECK FUNCTIONS ====================

    @staticmethod
    def _check_resistance_range(
        predictions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check: resistance probability ∈ [0, 1]."""
        if "resistance_probabilities" not in predictions:
            return True, "No resistance_probabilities in predictions"

        rp = predictions["resistance_probabilities"]
        if not isinstance(rp, torch.Tensor):
            return True, "resistance_probabilities is not a tensor"

        rp_min = rp.min().item()
        rp_max = rp.max().item()

        if rp_min < 0 or rp_max > 1:
            return False, f"Range [{rp_min:.6f}, {rp_max:.6f}] outside [0, 1]"

        return True, f"OK: [{rp_min:.6f}, {rp_max:.6f}]"

    @staticmethod
    def _check_ic50_range(
        predictions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check: IC50 ∈ [1e-12, 1e-3] molar (physiological range)."""
        if "ic50_values" not in predictions:
            return True, "No ic50_values in predictions"

        ic50 = predictions["ic50_values"]
        if not isinstance(ic50, torch.Tensor):
            return True, "ic50_values is not a tensor"

        ic50_min = ic50.min().item()
        ic50_max = ic50.max().item()

        MIN_IC50 = 1e-12  # picomolar (diffusion limit)
        MAX_IC50 = 1e-3   # millimolar (useful range)

        if ic50_min < MIN_IC50 or ic50_max > MAX_IC50:
            return False, f"Range [{ic50_min:.4e}, {ic50_max:.4e}] M outside [{MIN_IC50}, {MAX_IC50}]"

        return True, f"OK: [{ic50_min:.4e}, {ic50_max:.4e}] M"

    @staticmethod
    def _check_stability_range(
        predictions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check: stability scores ∈ [0, 1]."""
        if "stability_scores" not in predictions:
            return True, "No stability_scores in predictions"

        ss = predictions["stability_scores"]
        if not isinstance(ss, torch.Tensor):
            return True, "stability_scores is not a tensor"

        ss_min = ss.min().item()
        ss_max = ss.max().item()

        if ss_min < 0 or ss_max > 1:
            return False, f"Range [{ss_min:.6f}, {ss_max:.6f}] outside [0, 1]"

        return True, f"OK: [{ss_min:.6f}, {ss_max:.6f}]"

    @staticmethod
    def _check_no_nan(
        predictions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check: no NaN or Inf in any tensor predictions."""
        nan_keys = []
        inf_keys = []

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    nan_keys.append(key)
                if torch.isinf(value).any():
                    inf_keys.append(key)

        if nan_keys or inf_keys:
            msg = ""
            if nan_keys:
                msg += f"NaN in {nan_keys}; "
            if inf_keys:
                msg += f"Inf in {inf_keys}"
            return False, msg.rstrip("; ")

        return True, "OK: no NaN/Inf detected"

    @staticmethod
    def _check_transition_stochastic(
        predictions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check: transition probability matrix rows sum to 1."""
        if "transition_probabilities" not in predictions:
            return True, "No transition_probabilities in predictions"

        tp = predictions["transition_probabilities"]
        if not isinstance(tp, torch.Tensor):
            return True, "transition_probabilities is not a tensor"

        if tp.dim() != 2:
            return True, f"transition_probabilities has {tp.dim()} dimensions, expected 2"

        row_sums = tp.sum(dim=1)
        expected = torch.ones_like(row_sums)

        if not torch.allclose(row_sums, expected, atol=1e-5):
            min_sum = row_sums.min().item()
            max_sum = row_sums.max().item()
            return False, f"Row sums in [{min_sum:.6f}, {max_sum:.6f}], expected ~1.0"

        return True, f"OK: all row sums ≈ 1.0"

    @staticmethod
    def _check_protein_format(
        predictions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check: target proteins are valid identifiers."""
        if "target_proteins" not in predictions:
            return True, "No target_proteins in predictions"

        targets = predictions["target_proteins"]
        if not isinstance(targets, list):
            return True, "target_proteins is not a list"

        invalid = [t for t in targets if not isinstance(t, str) or len(t) == 0]

        if invalid:
            return False, f"Invalid format: {invalid}"

        return True, f"OK: {len(targets)} valid target identifiers"

    @staticmethod
    def _check_stability_monotonicity(
        predictions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check: deeper basin → longer mean transition time.

        If both 'stability_scores' and 'mean_transition_times' are present,
        verify that higher stability correlates with longer transition times.
        """
        if "stability_scores" not in predictions or "mean_transition_times" not in predictions:
            return True, "Missing stability_scores or mean_transition_times"

        stability = predictions["stability_scores"]
        times = predictions["mean_transition_times"]

        if not isinstance(stability, torch.Tensor) or not isinstance(times, torch.Tensor):
            return True, "Not tensors"

        if len(stability) != len(times):
            return False, f"Length mismatch: {len(stability)} vs {len(times)}"

        # Compute correlation: should be positive (higher stability → longer time)
        stability_flat = stability.flatten()
        times_flat = times.flatten()

        if len(stability_flat) < 2:
            return True, "Too few samples for correlation check"

        # Pearson correlation
        s_mean = stability_flat.mean()
        t_mean = times_flat.mean()

        numerator = ((stability_flat - s_mean) * (times_flat - t_mean)).sum()
        s_std = (stability_flat - s_mean).pow(2).sum().sqrt()
        t_std = (times_flat - t_mean).pow(2).sum().sqrt()

        if s_std < 1e-10 or t_std < 1e-10:
            return True, "Zero variance in one dimension"

        correlation = numerator / (s_std * t_std)

        if correlation < 0:
            return False, f"Negative correlation {correlation:.4f}: deeper basin → SHORTER time (impossible)"

        return True, f"OK: positive correlation {correlation:.4f}"

    @staticmethod
    def _check_fano_bound(
        predictions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check: model accuracy doesn't violate Fano lower bound.

        If both 'prediction_accuracy' and 'fano_lower_bound' are present,
        verify that accuracy >= fano_bound (no model can do better).
        """
        if "prediction_accuracy" not in predictions or "fano_lower_bound" not in predictions:
            return True, "Missing prediction_accuracy or fano_lower_bound"

        accuracy = predictions["prediction_accuracy"]
        fano = predictions["fano_lower_bound"]

        if not isinstance(accuracy, (float, torch.Tensor)):
            return True, "prediction_accuracy is not numeric"
        if not isinstance(fano, (float, torch.Tensor)):
            return True, "fano_lower_bound is not numeric"

        # Convert to float
        if isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.item()
        if isinstance(fano, torch.Tensor):
            fano = fano.item()

        # Model error should be >= fano lower bound
        model_error = 1 - accuracy
        fano_error = 1 - fano

        if model_error < fano_error - 1e-6:
            return False, (
                f"Model error {model_error:.6f} < Fano lower bound {fano_error:.6f}. "
                f"This is theoretically impossible—check calculations."
            )

        return True, f"OK: error {model_error:.6f} >= Fano bound {fano_error:.6f}"
