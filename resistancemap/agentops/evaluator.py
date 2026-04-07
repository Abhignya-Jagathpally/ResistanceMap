"""
Evaluation layer for ResistanceMap — task completion, guardrails, accuracy metrics.

Tracks guardrail violations, task results, accuracy checks, and clinical appropriateness.
Computes aggregate evaluation metrics for performance monitoring.
"""

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class GuardrailViolation:
    """Record of a guardrail being triggered."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_name: str = ""
    violation_type: str = ""  # e.g., "hallucination", "token_limit", "safety"
    severity: str = ""  # "low", "medium", "high", "critical"
    description: str = ""
    input_hash: str = ""  # hash of the input that triggered violation

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "description": self.description,
            "input_hash": self.input_hash,
        }


@dataclass
class TaskResult:
    """Record of a task execution result."""

    task_id: str = ""
    status: str = ""  # "completed", "failed", "partial"
    verified: bool = False
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "verified": self.verified,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AccuracyCheck:
    """Record of a factual accuracy check."""

    claim: str = ""
    source: str = ""  # reference source/database
    verified: bool = False
    confidence: float = 0.0  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "claim": self.claim,
            "source": self.source,
            "verified": self.verified,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ClinicalCheck:
    """Record of a clinical appropriateness review."""

    recommendation: str = ""
    appropriate: bool = False
    reviewer: str = ""  # who verified
    notes: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "recommendation": self.recommendation,
            "appropriate": self.appropriate,
            "reviewer": self.reviewer,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FirstPassCheck:
    """Record of first-pass approval."""

    output_id: str = ""
    approved: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "output_id": self.output_id,
            "approved": self.approved,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics."""

    task_completion_rate: float = 0.0  # completed / attempted
    guardrail_violation_rate: float = 0.0  # violations / total_operations
    factual_accuracy_rate: float = 0.0  # verified_correct / total_claims
    clinical_appropriateness: float = 0.0  # clinically_valid / total_recommendations
    first_pass_approval_rate: float = 0.0  # approved_first_try / total_outputs

    # Violation breakdown
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    violations_by_severity: Dict[str, int] = field(default_factory=dict)

    # Timestamp
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "task_completion_rate": self.task_completion_rate,
            "guardrail_violation_rate": self.guardrail_violation_rate,
            "factual_accuracy_rate": self.factual_accuracy_rate,
            "clinical_appropriateness": self.clinical_appropriateness,
            "first_pass_approval_rate": self.first_pass_approval_rate,
            "violations_by_type": self.violations_by_type,
            "violations_by_severity": self.violations_by_severity,
            "computed_at": self.computed_at.isoformat(),
        }


class Evaluator:
    """
    Evaluation layer for tracking and measuring agent performance.

    Monitors:
    - Task completion (success/failure)
    - Guardrail violations (safety, accuracy, clinical)
    - Factual accuracy (claims verified against sources)
    - Clinical appropriateness (expert review)
    - First-pass approval rates (quality metric)
    """

    def __init__(self):
        self._violations: List[GuardrailViolation] = []
        self._task_results: List[TaskResult] = []
        self._accuracy_checks: List[AccuracyCheck] = []
        self._clinical_checks: List[ClinicalCheck] = []
        self._first_pass_checks: List[FirstPassCheck] = []
        self._lock = threading.Lock()

    def record_task_result(
        self,
        task_id: str,
        status: str,
        verified: bool = False,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Record the result of a task execution.

        Args:
            task_id: Unique task identifier.
            status: Task status ("completed", "failed", "partial").
            verified: Whether task was verified/validated.
            metadata: Optional additional metadata.
        """
        if metadata is None:
            metadata = {}

        result = TaskResult(
            task_id=task_id,
            status=status,
            verified=verified,
            metadata=metadata,
        )

        with self._lock:
            self._task_results.append(result)

    def record_guardrail_violation(
        self,
        agent_name: str,
        violation_type: str,
        severity: str,
        description: str,
        input_hash: Optional[str] = None,
    ) -> None:
        """
        Record a guardrail violation.

        Args:
            agent_name: Name of agent that violated guardrail.
            violation_type: Type of violation (e.g., "hallucination", "safety").
            severity: Severity level ("low", "medium", "high", "critical").
            description: Human-readable description.
            input_hash: Optional hash of input that triggered violation.
        """
        if input_hash is None:
            input_hash = ""

        violation = GuardrailViolation(
            agent_name=agent_name,
            violation_type=violation_type,
            severity=severity,
            description=description,
            input_hash=input_hash,
        )

        with self._lock:
            self._violations.append(violation)

    def record_accuracy_check(
        self,
        claim: str,
        source: str,
        verified: bool,
        confidence: float = 1.0,
    ) -> None:
        """
        Record a factual accuracy check.

        Args:
            claim: The claim being verified.
            source: Reference source/database used for verification.
            verified: Whether claim was verified as correct.
            confidence: Confidence in the verification (0.0 to 1.0).
        """
        check = AccuracyCheck(
            claim=claim,
            source=source,
            verified=verified,
            confidence=confidence,
        )

        with self._lock:
            self._accuracy_checks.append(check)

    def record_clinical_check(
        self,
        recommendation: str,
        appropriate: bool,
        reviewer: str = "",
        notes: str = "",
    ) -> None:
        """
        Record a clinical appropriateness review.

        Args:
            recommendation: The clinical recommendation being reviewed.
            appropriate: Whether recommendation is clinically appropriate.
            reviewer: Name/ID of reviewer.
            notes: Optional additional notes.
        """
        check = ClinicalCheck(
            recommendation=recommendation,
            appropriate=appropriate,
            reviewer=reviewer,
            notes=notes,
        )

        with self._lock:
            self._clinical_checks.append(check)

    def record_first_pass(self, output_id: str, approved: bool) -> None:
        """
        Record whether output was approved on first pass.

        Args:
            output_id: Unique output identifier.
            approved: Whether output was approved first-try.
        """
        check = FirstPassCheck(output_id=output_id, approved=approved)

        with self._lock:
            self._first_pass_checks.append(check)

    def get_metrics(self) -> EvaluationMetrics:
        """
        Compute aggregate evaluation metrics.

        Returns:
            EvaluationMetrics object with all computed rates and counts.
        """
        with self._lock:
            # Task completion rate
            num_tasks = len(self._task_results)
            completed = sum(
                1
                for t in self._task_results
                if t.status in ("completed", "partial")
            )
            task_completion_rate = (
                completed / num_tasks if num_tasks > 0 else 0.0
            )

            # Guardrail violation rate (violations per operation)
            num_violations = len(self._violations)
            total_operations = (
                num_tasks + len(self._accuracy_checks) + len(self._clinical_checks)
            )
            violation_rate = (
                num_violations / total_operations
                if total_operations > 0
                else 0.0
            )

            # Factual accuracy rate
            num_accuracy_checks = len(self._accuracy_checks)
            verified_correct = sum(
                1 for a in self._accuracy_checks if a.verified
            )
            accuracy_rate = (
                verified_correct / num_accuracy_checks
                if num_accuracy_checks > 0
                else 0.0
            )

            # Clinical appropriateness
            num_clinical = len(self._clinical_checks)
            clinically_appropriate = sum(
                1 for c in self._clinical_checks if c.appropriate
            )
            clinical_rate = (
                clinically_appropriate / num_clinical if num_clinical > 0 else 0.0
            )

            # First-pass approval rate
            num_first_pass = len(self._first_pass_checks)
            approved_first_pass = sum(
                1 for f in self._first_pass_checks if f.approved
            )
            approval_rate = (
                approved_first_pass / num_first_pass
                if num_first_pass > 0
                else 0.0
            )

            # Violations by type
            violations_by_type = {}
            for v in self._violations:
                violations_by_type[v.violation_type] = (
                    violations_by_type.get(v.violation_type, 0) + 1
                )

            # Violations by severity
            violations_by_severity = {}
            for v in self._violations:
                violations_by_severity[v.severity] = (
                    violations_by_severity.get(v.severity, 0) + 1
                )

            return EvaluationMetrics(
                task_completion_rate=task_completion_rate,
                guardrail_violation_rate=violation_rate,
                factual_accuracy_rate=accuracy_rate,
                clinical_appropriateness=clinical_rate,
                first_pass_approval_rate=approval_rate,
                violations_by_type=violations_by_type,
                violations_by_severity=violations_by_severity,
            )

    def get_violations_by_agent(self) -> Dict[str, List[GuardrailViolation]]:
        """
        Get violations grouped by agent name.

        Returns:
            Dictionary mapping agent names to lists of violations.
        """
        with self._lock:
            by_agent = {}
            for v in self._violations:
                if v.agent_name not in by_agent:
                    by_agent[v.agent_name] = []
                by_agent[v.agent_name].append(v)

            return by_agent

    def get_violations_by_severity(self) -> Dict[str, List[GuardrailViolation]]:
        """
        Get violations grouped by severity level.

        Returns:
            Dictionary mapping severity to lists of violations.
        """
        with self._lock:
            by_severity = {}
            for v in self._violations:
                if v.severity not in by_severity:
                    by_severity[v.severity] = []
                by_severity[v.severity].append(v)

            return by_severity

    def get_all_violations(self) -> List[GuardrailViolation]:
        """Get all recorded violations."""
        with self._lock:
            return self._violations.copy()

    def get_all_task_results(self) -> List[TaskResult]:
        """Get all recorded task results."""
        with self._lock:
            return self._task_results.copy()

    def get_all_accuracy_checks(self) -> List[AccuracyCheck]:
        """Get all recorded accuracy checks."""
        with self._lock:
            return self._accuracy_checks.copy()

    def get_all_clinical_checks(self) -> List[ClinicalCheck]:
        """Get all recorded clinical checks."""
        with self._lock:
            return self._clinical_checks.copy()

    def get_all_first_pass_checks(self) -> List[FirstPassCheck]:
        """Get all recorded first-pass checks."""
        with self._lock:
            return self._first_pass_checks.copy()

    def export_violations_json(self) -> List[Dict]:
        """Export all violations as JSON-serializable dicts."""
        with self._lock:
            return [v.to_dict() for v in self._violations]

    def clear(self) -> None:
        """Clear all evaluation records (useful for testing)."""
        with self._lock:
            self._violations.clear()
            self._task_results.clear()
            self._accuracy_checks.clear()
            self._clinical_checks.clear()
            self._first_pass_checks.clear()
