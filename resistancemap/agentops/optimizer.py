"""
Optimization layer for ResistanceMap — efficiency metrics and improvement tracking.

Monitors token efficiency, retrieval precision, handoff success, flow efficiency,
and generates improvement suggestions based on metrics history.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class TokenUsageRecord:
    """Record of token usage in an operation."""

    input_tokens: int = 0
    output_tokens: int = 0
    useful_tokens: int = 0  # tokens in output that contributed to solution
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "useful_tokens": self.useful_tokens,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RetrievalRecord:
    """Record of a retrieval operation."""

    query: str = ""
    results_count: int = 0
    relevant_results_count: int = 0
    k: int = 10  # top-k retrieval
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def precision_at_k(self) -> float:
        """Precision@k metric."""
        return self.relevant_results_count / self.k if self.k > 0 else 0.0

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "results_count": self.results_count,
            "relevant_results_count": self.relevant_results_count,
            "k": self.k,
            "precision_at_k": self.precision_at_k,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HandoffRecord:
    """Record of an agent handoff."""

    from_agent: str = ""
    to_agent: str = ""
    success: bool = False
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FlowStepRecord:
    """Record of a step in the processing flow."""

    step_id: str = ""
    necessary: bool = False  # whether step was critical to solution
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "step_id": self.step_id,
            "necessary": self.necessary,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetricSnapshot:
    """Point-in-time snapshot of metrics."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


@dataclass
class OptimizationMetrics:
    """Aggregate optimization metrics."""

    prompt_token_efficiency: float = 0.0  # useful_output / total_input
    retrieval_precision_at_k: Dict[int, float] = field(
        default_factory=dict
    )  # P@1, P@5, P@10
    handoff_success_rate: float = 0.0  # successful / total
    flow_step_efficiency: float = 0.0  # necessary_steps / total_steps
    improvement_velocity: float = 0.0  # metric_delta / days

    # Details
    avg_handoff_latency_ms: float = 0.0
    total_tokens_saved: int = 0
    total_steps_eliminated: int = 0

    # Timestamp
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "prompt_token_efficiency": self.prompt_token_efficiency,
            "retrieval_precision_at_k": self.retrieval_precision_at_k,
            "handoff_success_rate": self.handoff_success_rate,
            "flow_step_efficiency": self.flow_step_efficiency,
            "improvement_velocity": self.improvement_velocity,
            "avg_handoff_latency_ms": self.avg_handoff_latency_ms,
            "total_tokens_saved": self.total_tokens_saved,
            "total_steps_eliminated": self.total_steps_eliminated,
            "computed_at": self.computed_at.isoformat(),
        }


class Optimizer:
    """
    Optimization layer for tracking efficiency and improvement metrics.

    Monitors:
    - Token efficiency (useful output per input token)
    - Retrieval precision (P@k)
    - Handoff success and latency
    - Flow step necessity and efficiency
    - Metric improvement velocity over time
    """

    def __init__(self):
        self._token_usage: List[TokenUsageRecord] = []
        self._retrieval_results: List[RetrievalRecord] = []
        self._handoffs: List[HandoffRecord] = []
        self._flow_steps: List[FlowStepRecord] = []
        self._metric_history: List[MetricSnapshot] = []
        self._lock = threading.Lock()

    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        useful_tokens: int = 0,
    ) -> None:
        """
        Record token usage in an operation.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            useful_tokens: Number of output tokens that contributed to solution.
                          If 0, defaults to output_tokens.
        """
        if useful_tokens == 0:
            useful_tokens = output_tokens

        record = TokenUsageRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            useful_tokens=useful_tokens,
        )

        with self._lock:
            self._token_usage.append(record)

    def record_retrieval(
        self,
        query: str,
        results: List,
        relevant_results: List,
        k: int = 10,
    ) -> None:
        """
        Record a retrieval operation.

        Args:
            query: The search query.
            results: List of retrieved results.
            relevant_results: List of relevant results from the retrieved set.
            k: The k parameter (how many results were requested).
        """
        record = RetrievalRecord(
            query=query,
            results_count=len(results),
            relevant_results_count=len(relevant_results),
            k=k,
        )

        with self._lock:
            self._retrieval_results.append(record)

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """
        Record an agent handoff.

        Args:
            from_agent: Name of agent passing off.
            to_agent: Name of agent receiving.
            success: Whether handoff was successful.
            latency_ms: Handoff latency in milliseconds.
        """
        record = HandoffRecord(
            from_agent=from_agent,
            to_agent=to_agent,
            success=success,
            latency_ms=latency_ms,
        )

        with self._lock:
            self._handoffs.append(record)

    def record_flow_step(
        self,
        step_id: str,
        necessary: bool,
        duration_ms: float,
    ) -> None:
        """
        Record a step in the processing flow.

        Args:
            step_id: Unique step identifier.
            necessary: Whether step was critical to solution.
            duration_ms: Duration of the step in milliseconds.
        """
        record = FlowStepRecord(
            step_id=step_id,
            necessary=necessary,
            duration_ms=duration_ms,
        )

        with self._lock:
            self._flow_steps.append(record)

    def record_metric_snapshot(self, metrics: Dict) -> None:
        """
        Record a point-in-time snapshot of metrics.

        Args:
            metrics: Dictionary of metric names to values.
        """
        snapshot = MetricSnapshot(metrics=metrics)

        with self._lock:
            self._metric_history.append(snapshot)

    def get_metrics(self) -> OptimizationMetrics:
        """
        Compute aggregate optimization metrics.

        Returns:
            OptimizationMetrics object with all computed rates and efficiencies.
        """
        with self._lock:
            # Token efficiency
            total_input = sum(t.input_tokens for t in self._token_usage)
            total_useful = sum(t.useful_tokens for t in self._token_usage)
            token_efficiency = (
                total_useful / total_input if total_input > 0 else 0.0
            )
            tokens_saved = total_input - total_useful

            # Retrieval precision at different k values
            precision_at_k = {}
            for k_val in [1, 5, 10]:
                precisions = []
                for r in self._retrieval_results:
                    if r.k >= k_val:
                        # Assume precision degrades linearly with position
                        precisions.append(r.precision_at_k)
                precision_at_k[k_val] = (
                    sum(precisions) / len(precisions)
                    if precisions
                    else 0.0
                )

            # Handoff success rate
            num_handoffs = len(self._handoffs)
            successful_handoffs = sum(1 for h in self._handoffs if h.success)
            handoff_success_rate = (
                successful_handoffs / num_handoffs if num_handoffs > 0 else 0.0
            )

            # Handoff average latency
            avg_latency = (
                sum(h.latency_ms for h in self._handoffs) / num_handoffs
                if num_handoffs > 0
                else 0.0
            )

            # Flow step efficiency
            num_steps = len(self._flow_steps)
            necessary_steps = sum(1 for s in self._flow_steps if s.necessary)
            step_efficiency = (
                necessary_steps / num_steps if num_steps > 0 else 0.0
            )
            steps_eliminated = num_steps - necessary_steps

            # Improvement velocity (change in token efficiency over time)
            improvement_velocity = self._compute_improvement_velocity(
                window_days=7
            )

            return OptimizationMetrics(
                prompt_token_efficiency=token_efficiency,
                retrieval_precision_at_k=precision_at_k,
                handoff_success_rate=handoff_success_rate,
                flow_step_efficiency=step_efficiency,
                improvement_velocity=improvement_velocity,
                avg_handoff_latency_ms=avg_latency,
                total_tokens_saved=tokens_saved,
                total_steps_eliminated=steps_eliminated,
            )

    def get_improvement_velocity(
        self, metric_name: str, window_days: int = 7
    ) -> float:
        """
        Compute improvement velocity for a specific metric.

        Velocity is (metric_delta / days), indicating rate of change.

        Args:
            metric_name: Name of metric to compute velocity for.
            window_days: Time window in days.

        Returns:
            Rate of change per day.
        """
        with self._lock:
            return self._compute_improvement_velocity(
                metric_name=metric_name, window_days=window_days
            )

    def _compute_improvement_velocity(
        self, metric_name: str = None, window_days: int = 7
    ) -> float:
        """
        Internal: compute improvement velocity.

        Args:
            metric_name: Specific metric to track. If None, uses token efficiency.
            window_days: Time window in days.

        Returns:
            Rate of change per day.
        """
        if not self._metric_history or len(self._metric_history) < 2:
            # Need at least 2 snapshots
            return 0.0

        now = datetime.utcnow()
        cutoff = now - timedelta(days=window_days)

        # Filter snapshots within window
        recent = [
            s
            for s in self._metric_history
            if s.timestamp >= cutoff
        ]

        if len(recent) < 2:
            return 0.0

        # Default to token efficiency if no metric specified
        if metric_name is None:
            total_input = sum(t.input_tokens for t in self._token_usage)
            total_useful = sum(t.useful_tokens for t in self._token_usage)
            metric_name = "token_efficiency"

        # Get first and last values
        first_snapshot = recent[0]
        last_snapshot = recent[-1]

        first_val = first_snapshot.metrics.get(metric_name, 0.0)
        last_val = last_snapshot.metrics.get(metric_name, 0.0)

        time_delta_days = (
            last_snapshot.timestamp - first_snapshot.timestamp
        ).total_seconds() / (24 * 3600)

        if time_delta_days == 0:
            return 0.0

        velocity = (last_val - first_val) / time_delta_days

        return velocity

    def suggest_optimizations(self) -> List[str]:
        """
        Generate optimization suggestions based on metrics.

        Returns:
            List of actionable improvement suggestions.
        """
        suggestions = []
        metrics = self.get_metrics()

        # Token efficiency suggestions
        if metrics.prompt_token_efficiency < 0.7:
            suggestions.append(
                "Token efficiency below 70%: Consider summarizing inputs or "
                "using more targeted prompts"
            )

        # Retrieval precision suggestions
        p_at_1 = metrics.retrieval_precision_at_k.get(1, 0.0)
        if p_at_1 < 0.5:
            suggestions.append(
                "Retrieval precision@1 below 50%: Improve query formulation "
                "or relevance ranking"
            )

        # Handoff success suggestions
        if metrics.handoff_success_rate < 0.9:
            suggestions.append(
                "Handoff success rate below 90%: Improve agent communication "
                "protocols or context passing"
            )

        # Flow efficiency suggestions
        if metrics.flow_step_efficiency < 0.7:
            suggestions.append(
                f"Flow step efficiency below 70%: {metrics.total_steps_eliminated} "
                "steps are unnecessary - simplify workflow"
            )

        # Improvement velocity suggestions
        if metrics.improvement_velocity < 0:
            suggestions.append(
                "Metrics are regressing: Recent changes may be degrading performance - "
                "consider reverting"
            )
        elif metrics.improvement_velocity > 0:
            suggestions.append(
                f"Positive improvement velocity: Continue current optimization approach"
            )

        if not suggestions:
            suggestions.append("All metrics are healthy - maintain current approach")

        return suggestions

    def get_all_token_usage(self) -> List[TokenUsageRecord]:
        """Get all token usage records."""
        with self._lock:
            return self._token_usage.copy()

    def get_all_retrieval_records(self) -> List[RetrievalRecord]:
        """Get all retrieval records."""
        with self._lock:
            return self._retrieval_results.copy()

    def get_all_handoff_records(self) -> List[HandoffRecord]:
        """Get all handoff records."""
        with self._lock:
            return self._handoffs.copy()

    def get_all_flow_steps(self) -> List[FlowStepRecord]:
        """Get all flow step records."""
        with self._lock:
            return self._flow_steps.copy()

    def clear(self) -> None:
        """Clear all optimization records (useful for testing)."""
        with self._lock:
            self._token_usage.clear()
            self._retrieval_results.clear()
            self._handoffs.clear()
            self._flow_steps.clear()
            self._metric_history.clear()
