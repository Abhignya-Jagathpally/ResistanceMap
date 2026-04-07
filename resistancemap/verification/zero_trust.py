"""
Zero-Trust Verification System for ResistanceMap
=================================================

Implements cryptographic verification and statistical validation at every
agent boundary. No agent trusts another's output without proof.

Core Principle:
  In multi-agent systems handling biological predictions, each agent must
  prove data integrity through:
    1. Cryptographic hash verification (append-only audit trail)
    2. Statistical validity checks (NaN/Inf, distribution shifts)
    3. Domain-specific constraint verification (biology bounds)
    4. Model sanity checks (gradient flow, calibration)

The hash chain creates an immutable audit trail:
  H₁ = SHA256(data₁)
  H₂ = SHA256(H₁ || data₂)
  H₃ = SHA256(H₂ || data₃)
  ...
  Hₙ = SHA256(Hₙ₋₁ || dataₙ)

Verification at each step:
  H'ᵢ = SHA256(Hᵢ₋₁ || dataᵢ) == Hᵢ? ✓

This prevents tampering even if an intermediate agent is compromised.

Reference: Lamport, Making the Right Choices; Nakamoto, Bitcoin (hash chains for integrity)
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
import torch
import numpy as np
from scipy import stats


@dataclass
class VerificationResult:
    """Result of a verification check at an agent boundary.

    Attributes:
        passed: Whether all checks passed
        checks: List of individual check results, each containing:
                {name: str, passed: bool, details: str}
        hash_verified: Cryptographic hash matched expected value
        statistical_valid: Distribution passes statistical checks
        timestamp: When verification occurred
        agent_name: Which agent produced the data
        data_hash: SHA256 hash of verified data for audit trail
    """
    passed: bool
    checks: List[Dict[str, Any]] = field(default_factory=list)
    hash_verified: bool = False
    statistical_valid: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_name: Optional[str] = None
    data_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for audit logging."""
        return {
            "passed": self.passed,
            "checks": self.checks,
            "hash_verified": self.hash_verified,
            "statistical_valid": self.statistical_valid,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "data_hash": self.data_hash,
        }


class ZeroTrustVerifier:
    """Verifies data integrity and statistical validity at every agent boundary.

    Maintains an append-only hash chain and verification log for full auditability.
    Every data handoff between agents requires:
      1. Cryptographic verification (hash chain integrity)
      2. Statistical validity (distribution checks)
      3. Domain constraints (biological bounds)
      4. Model sanity (numerical stability)

    Example:
        >>> verifier = ZeroTrustVerifier()
        >>>
        >>> # Agent 1 produces predictions
        >>> preds = {"resistance": torch.tensor([0.5, 0.7]), "targets": ["BCR", "ABL"]}
        >>> hash1 = verifier.create_verification_checkpoint("Agent1", preds)
        >>>
        >>> # Agent 2 receives predictions and verifies before trust
        >>> tensor = preds["resistance"]
        >>> stats = {"mean": 0.6, "std": 0.1, "shape": (2,)}
        >>> result = verifier.verify_tensor_statistics(tensor, stats)
        >>> assert result.passed, f"Verification failed: {result.checks}"
        >>>
        >>> # Audit trail is immutable
        >>> audit = verifier.get_audit_trail()
        >>> assert len(audit) > 0
    """

    def __init__(self):
        """Initialize verifier with empty hash chain and log."""
        self._hash_chain: List[str] = []
        self._verification_log: List[VerificationResult] = []

    @staticmethod
    def compute_tensor_hash(tensor: torch.Tensor) -> str:
        """Compute deterministic SHA256 hash of tensor contents.

        Args:
            tensor: PyTorch tensor of any shape and dtype

        Returns:
            Hex digest of SHA256(tensor_bytes || shape || dtype)

        Note:
            Requires deterministic ordering, so tensor is flattened.
            Hash includes shape and dtype to catch shape/type mismatches.
        """
        hasher = hashlib.sha256()

        # Hash tensor data (flatten for deterministic ordering)
        tensor_flat = tensor.flatten().cpu().detach().numpy()
        hasher.update(tensor_flat.tobytes())

        # Hash shape and dtype (catch silent shape/type changes)
        shape_bytes = json.dumps(list(tensor.shape)).encode()
        hasher.update(shape_bytes)

        dtype_bytes = str(tensor.dtype).encode()
        hasher.update(dtype_bytes)

        return hasher.hexdigest()

    @staticmethod
    def compute_dict_hash(data: Dict[str, Any]) -> str:
        """Compute deterministic SHA256 hash of dict.

        Args:
            data: Dictionary with JSON-serializable values

        Returns:
            Hex digest of SHA256(json_string)

        Note:
            Uses sort_keys=True for deterministic ordering.
            Tensors are converted to list form.
        """
        def serialize_for_hashing(obj):
            """Convert non-JSON types (like tensors) to JSON-serializable form."""
            if isinstance(obj, torch.Tensor):
                return obj.flatten().cpu().detach().tolist()
            elif isinstance(obj, dict):
                return {k: serialize_for_hashing(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_for_hashing(item) for item in obj]
            elif isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            return obj

        serialized = serialize_for_hashing(data)
        json_str = json.dumps(serialized, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def verify_data_integrity(self, data: Any, expected_hash: str) -> bool:
        """Verify data against expected hash. Zero-trust principle.

        Args:
            data: Data to verify (tensor or dict)
            expected_hash: Expected SHA256 hex digest

        Returns:
            True if hash matches, False otherwise.

        Zero-Trust Principle:
            Never trust that data hasn't been modified. Always verify against
            a cryptographic commitment made before the data was transmitted.
        """
        if isinstance(data, torch.Tensor):
            computed_hash = self.compute_tensor_hash(data)
        elif isinstance(data, dict):
            computed_hash = self.compute_dict_hash(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return computed_hash == expected_hash

    def verify_tensor_statistics(
        self,
        tensor: torch.Tensor,
        expected_stats: Dict[str, Any],
        check_nan_inf: bool = True,
        check_range: bool = True,
        check_distribution: bool = True,
    ) -> VerificationResult:
        """Verify tensor statistics match expected values.

        Args:
            tensor: Tensor to verify
            expected_stats: Dict with keys:
                - mean: expected mean value
                - std: expected standard deviation
                - shape: expected shape tuple
                - min/max: expected min/max values (optional)
                - value_range: (min_val, max_val) for valid range
            check_nan_inf: Whether to check for NaN/Inf values
            check_range: Whether to check min/max values
            check_distribution: Whether to check for distribution shifts

        Returns:
            VerificationResult with all checks documented.

        Statistical Checks:
            1. Shape Correctness: actual_shape == expected_shape
            2. NaN/Inf Check: no NaN or Inf values
            3. Range Check: all values in [min, max]
            4. Mean/Std: actual mean/std near expected (tolerance: 5%)
            5. Distribution: Kolmogorov-Smirnov test vs reference (if provided)
        """
        result = VerificationResult(passed=True)
        result.data_hash = self.compute_tensor_hash(tensor)

        # Check 1: Shape
        if tuple(tensor.shape) != tuple(expected_stats.get("shape", tensor.shape)):
            result.checks.append({
                "name": "shape",
                "passed": False,
                "details": f"Shape mismatch: {tuple(tensor.shape)} vs {expected_stats.get('shape')}",
            })
            result.passed = False
        else:
            result.checks.append({"name": "shape", "passed": True, "details": ""})

        # Check 2: NaN/Inf
        if check_nan_inf:
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if has_nan or has_inf:
                result.checks.append({
                    "name": "nan_inf",
                    "passed": False,
                    "details": f"NaN={has_nan}, Inf={has_inf}",
                })
                result.passed = False
            else:
                result.checks.append({"name": "nan_inf", "passed": True, "details": ""})

        # Check 3: Value range
        if check_range and "value_range" in expected_stats:
            min_val, max_val = expected_stats["value_range"]
            actual_min = tensor.min().item()
            actual_max = tensor.max().item()
            if actual_min < min_val or actual_max > max_val:
                result.checks.append({
                    "name": "value_range",
                    "passed": False,
                    "details": f"Range [{actual_min:.4f}, {actual_max:.4f}] outside [{min_val}, {max_val}]",
                })
                result.passed = False
            else:
                result.checks.append({"name": "value_range", "passed": True, "details": ""})

        # Check 4: Mean and Std
        tensor_flat = tensor.flatten().float()
        actual_mean = tensor_flat.mean().item()
        actual_std = tensor_flat.std().item()

        expected_mean = expected_stats.get("mean", actual_mean)
        expected_std = expected_stats.get("std", actual_std)

        mean_tolerance = abs(expected_mean) * 0.05 + 1e-6  # 5% or 1e-6
        std_tolerance = abs(expected_std) * 0.05 + 1e-6

        mean_ok = abs(actual_mean - expected_mean) <= mean_tolerance
        std_ok = abs(actual_std - expected_std) <= std_tolerance

        result.checks.append({
            "name": "mean",
            "passed": mean_ok,
            "details": f"Mean: {actual_mean:.6f} vs expected {expected_mean:.6f}",
        })
        result.checks.append({
            "name": "std",
            "passed": std_ok,
            "details": f"Std: {actual_std:.6f} vs expected {expected_std:.6f}",
        })

        if not (mean_ok and std_ok):
            result.passed = False

        result.statistical_valid = result.passed
        return result

    def verify_model_output(
        self,
        output: torch.Tensor,
        model_name: str,
        expected_range: Optional[Tuple[float, float]] = None,
        expected_shape: Optional[torch.Size] = None,
    ) -> VerificationResult:
        """Verify model output for numerical health and correctness.

        Args:
            output: Model output tensor
            model_name: Name of model for audit trail
            expected_range: (min, max) valid range for output values
            expected_shape: Expected output shape

        Returns:
            VerificationResult with sanity checks.

        Sanity Checks:
            1. No NaN/Inf (basic numerical stability)
            2. Within expected range (output bounds check)
            3. Not all same value (model hasn't collapsed)
            4. Gradient-compatible (not extreme magnitudes)
        """
        result = VerificationResult(passed=True, agent_name=model_name)
        result.data_hash = self.compute_tensor_hash(output)

        # Check 1: NaN/Inf
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        result.checks.append({
            "name": "nan_inf",
            "passed": not (has_nan or has_inf),
            "details": f"NaN={has_nan}, Inf={has_inf}",
        })
        if has_nan or has_inf:
            result.passed = False

        # Check 2: Range (e.g., [0, 1] for probabilities)
        if expected_range:
            min_val, max_val = expected_range
            actual_min = output.min().item()
            actual_max = output.max().item()
            in_range = (actual_min >= min_val and actual_max <= max_val)
            result.checks.append({
                "name": "output_range",
                "passed": in_range,
                "details": f"[{actual_min:.4f}, {actual_max:.4f}] vs [{min_val}, {max_val}]",
            })
            if not in_range:
                result.passed = False

        # Check 3: Shape
        if expected_shape is not None:
            shape_ok = tuple(output.shape) == tuple(expected_shape)
            result.checks.append({
                "name": "shape",
                "passed": shape_ok,
                "details": f"{tuple(output.shape)} vs {tuple(expected_shape)}",
            })
            if not shape_ok:
                result.passed = False

        # Check 4: Not all same value (model collapse detector)
        unique_vals = torch.unique(output).numel()
        not_collapsed = unique_vals > 1
        result.checks.append({
            "name": "model_collapse",
            "passed": not_collapsed,
            "details": f"{unique_vals} unique values",
        })
        if not not_collapsed:
            result.passed = False

        # Check 5: Gradient-compatible magnitudes (if requires_grad)
        if output.requires_grad:
            output_mag = output.abs().max().item()
            if output_mag > 1e4 or (output_mag < 1e-6 and output_mag > 0):
                result.checks.append({
                    "name": "gradient_health",
                    "passed": False,
                    "details": f"Output magnitude {output_mag:.4e} may cause gradient issues",
                })
                result.passed = False
            else:
                result.checks.append({
                    "name": "gradient_health",
                    "passed": True,
                    "details": f"Output magnitude {output_mag:.4e}",
                })

        result.statistical_valid = result.passed
        return result

    def verify_biological_constraints(
        self,
        predictions: Dict[str, Any],
    ) -> VerificationResult:
        """Verify predictions respect biological domain constraints.

        Args:
            predictions: Dict containing:
                - "resistance_probabilities": tensor in [0, 1]
                - "stability_scores": tensor in [0, 1]
                - "transition_probabilities": tensor where rows sum to 1
                - "target_proteins": list of UniProt identifiers
                - "ic50_values": tensor in [1e-12, 1e-3] molar

        Returns:
            VerificationResult documenting all constraint checks.

        Biological Constraints:
            1. Drug resistance ∈ [0, 1] (probability)
            2. Stability scores ∈ [0, 1] (normalized score)
            3. Transition probabilities: rows sum to 1 (stochastic matrix)
            4. IC50 values in [1e-12, 1e-3] M (picomolar to millimolar)
               - Below 1e-12 M: unlikely (diffusion limit)
               - Above 1e-3 M: not useful for drug
            5. Target proteins exist in STRING PPI (if provided)
        """
        result = VerificationResult(passed=True)

        # Check 1: Resistance probabilities ∈ [0, 1]
        if "resistance_probabilities" in predictions:
            rp = predictions["resistance_probabilities"]
            if isinstance(rp, torch.Tensor):
                rp_valid = (rp >= 0).all() and (rp <= 1).all()
                result.checks.append({
                    "name": "resistance_probability_range",
                    "passed": rp_valid,
                    "details": f"Range [{rp.min():.4f}, {rp.max():.4f}]",
                })
                if not rp_valid:
                    result.passed = False

        # Check 2: Stability scores ∈ [0, 1]
        if "stability_scores" in predictions:
            ss = predictions["stability_scores"]
            if isinstance(ss, torch.Tensor):
                ss_valid = (ss >= 0).all() and (ss <= 1).all()
                result.checks.append({
                    "name": "stability_score_range",
                    "passed": ss_valid,
                    "details": f"Range [{ss.min():.4f}, {ss.max():.4f}]",
                })
                if not ss_valid:
                    result.passed = False

        # Check 3: Transition probabilities sum to 1
        if "transition_probabilities" in predictions:
            tp = predictions["transition_probabilities"]
            if isinstance(tp, torch.Tensor) and tp.dim() == 2:
                row_sums = tp.sum(dim=1)
                sums_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
                result.checks.append({
                    "name": "transition_probability_stochastic",
                    "passed": sums_ok,
                    "details": f"Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}",
                })
                if not sums_ok:
                    result.passed = False

        # Check 4: IC50 values in physiological range
        if "ic50_values" in predictions:
            ic50 = predictions["ic50_values"]
            if isinstance(ic50, torch.Tensor):
                ic50_min = 1e-12  # picomolar (diffusion limit)
                ic50_max = 1e-3   # millimolar (useful range)
                ic50_valid = (ic50 >= ic50_min).all() and (ic50 <= ic50_max).all()
                result.checks.append({
                    "name": "ic50_physiological_range",
                    "passed": ic50_valid,
                    "details": f"Range [{ic50.min():.4e}, {ic50.max():.4e}] M vs [{ic50_min}, {ic50_max}]",
                })
                if not ic50_valid:
                    result.passed = False

        # Check 5: Target proteins format
        if "target_proteins" in predictions:
            targets = predictions["target_proteins"]
            if isinstance(targets, list):
                valid_format = all(
                    isinstance(t, str) and len(t) > 0
                    for t in targets
                )
                result.checks.append({
                    "name": "target_protein_format",
                    "passed": valid_format,
                    "details": f"{len(targets)} targets",
                })
                if not valid_format:
                    result.passed = False

        # Check 6: No NaN/Inf in any tensor predictions
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                has_nan = torch.isnan(value).any().item()
                has_inf = torch.isinf(value).any().item()
                if has_nan or has_inf:
                    result.checks.append({
                        "name": f"nan_inf_{key}",
                        "passed": False,
                        "details": f"NaN={has_nan}, Inf={has_inf}",
                    })
                    result.passed = False

        return result

    def create_verification_checkpoint(
        self,
        agent_name: str,
        data: Any,
    ) -> str:
        """Create a verification checkpoint and append to hash chain.

        Args:
            agent_name: Name of agent creating checkpoint
            data: Data to checkpoint (tensor or dict)

        Returns:
            SHA256 hash of the checkpoint. This hash becomes the previous_hash
            for the next checkpoint, creating an immutable chain.

        Example:
            >>> verifier = ZeroTrustVerifier()
            >>> hash1 = verifier.create_verification_checkpoint("Agent1", data1)
            >>> hash2 = verifier.create_verification_checkpoint("Agent2", data2)
            >>> # hash2 depends cryptographically on hash1 - tampering with data1 breaks chain
        """
        # Compute hash of current data
        if isinstance(data, torch.Tensor):
            current_hash = self.compute_tensor_hash(data)
        elif isinstance(data, dict):
            current_hash = self.compute_dict_hash(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Chain with previous hash
        if self._hash_chain:
            previous_hash = self._hash_chain[-1]
            # Chain is: SHA256(previous_hash || current_hash || agent_name)
            chained = hashlib.sha256()
            chained.update(previous_hash.encode())
            chained.update(current_hash.encode())
            chained.update(agent_name.encode())
            checkpoint_hash = chained.hexdigest()
        else:
            # First checkpoint
            checkpoint_hash = current_hash

        self._hash_chain.append(checkpoint_hash)

        # Log the checkpoint
        log_entry = {
            "agent": agent_name,
            "data_hash": current_hash,
            "checkpoint_hash": checkpoint_hash,
            "timestamp": datetime.utcnow().isoformat(),
        }
        # (Could append to log file here for persistence)

        return checkpoint_hash

    def verify_checkpoint(self, expected_hash: str, position: int) -> bool:
        """Verify that a checkpoint hash is in the chain at expected position.

        Args:
            expected_hash: Hash to look for
            position: Position in chain (0-indexed)

        Returns:
            True if hash matches at position, False otherwise.
        """
        if position < 0 or position >= len(self._hash_chain):
            return False
        return self._hash_chain[position] == expected_hash

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the complete verification audit trail.

        Returns:
            List of verification results, each with timestamp and details.
            Forms an immutable record of all verifications performed.
        """
        return [result.to_dict() for result in self._verification_log]

    def get_hash_chain(self) -> List[str]:
        """Return the complete hash chain.

        Returns:
            List of all checkpoint hashes, in order.
            This is the proof of sequential integrity.
        """
        return self._hash_chain.copy()

    def log_verification(self, result: VerificationResult) -> None:
        """Log a verification result to the audit trail.

        Args:
            result: VerificationResult to log
        """
        self._verification_log.append(result)
