"""
Dashboard for ResistanceMap AgentOps — unified observability, evaluation, and optimization view.

Combines metrics from all three layers (tracer, evaluator, optimizer) and provides
comprehensive reporting, JSON export, and terminal summaries.
"""

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .evaluator import Evaluator, EvaluationMetrics
from .optimizer import Optimizer, OptimizationMetrics
from .tracer import Tracer


@dataclass
class AgentPerformance:
    """Performance metrics for a single agent."""

    agent_name: str = ""
    num_operations: int = 0
    avg_operation_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    success_rate: float = 0.0
    avg_accuracy: float = 0.0
    violations_count: int = 0
    score: float = 0.0  # composite score 0-100


class AgentOpsDashboard:
    """
    Unified dashboard for observability, evaluation, and optimization.

    Combines data from three core layers:
    - Tracer: distributed execution tracing
    - Evaluator: task completion and guardrail tracking
    - Optimizer: efficiency metrics
    """

    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        evaluator: Optional[Evaluator] = None,
        optimizer: Optional[Optimizer] = None,
    ):
        self.tracer = tracer or Tracer.get_instance()
        self.evaluator = evaluator or Evaluator()
        self.optimizer = optimizer or Optimizer()
        self._lock = threading.Lock()

    def get_full_report(self) -> Dict:
        """
        Generate comprehensive report combining all three layers.

        Returns:
            Dictionary with all metrics, performance data, and insights.
        """
        with self._lock:
            tracer_metrics = self.tracer.export_metrics()
            eval_metrics = self.evaluator.get_metrics()
            opt_metrics = self.optimizer.get_metrics()

            agent_perf = self._compute_agent_leaderboard()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "observability": {
                    "total_traces": tracer_metrics["total_traces"],
                    "total_spans": tracer_metrics["total_spans"],
                    "total_duration_ms": tracer_metrics["total_duration_ms"],
                    "total_cost_usd": tracer_metrics["total_cost_usd"],
                    "total_tokens": tracer_metrics["total_tokens"],
                    "agent_handoff_latencies_ms": tracer_metrics.get(
                        "agent_handoff_latencies_ms", {}
                    ),
                    "tool_execution_latencies_ms": tracer_metrics.get(
                        "tool_execution_latencies_ms", {}
                    ),
                },
                "evaluation": {
                    "task_completion_rate": eval_metrics.task_completion_rate,
                    "guardrail_violation_rate": (
                        eval_metrics.guardrail_violation_rate
                    ),
                    "factual_accuracy_rate": eval_metrics.factual_accuracy_rate,
                    "clinical_appropriateness": (
                        eval_metrics.clinical_appropriateness
                    ),
                    "first_pass_approval_rate": (
                        eval_metrics.first_pass_approval_rate
                    ),
                    "violations_by_type": eval_metrics.violations_by_type,
                    "violations_by_severity": (
                        eval_metrics.violations_by_severity
                    ),
                },
                "optimization": {
                    "prompt_token_efficiency": (
                        opt_metrics.prompt_token_efficiency
                    ),
                    "retrieval_precision_at_k": (
                        opt_metrics.retrieval_precision_at_k
                    ),
                    "handoff_success_rate": opt_metrics.handoff_success_rate,
                    "flow_step_efficiency": opt_metrics.flow_step_efficiency,
                    "improvement_velocity": opt_metrics.improvement_velocity,
                    "total_tokens_saved": opt_metrics.total_tokens_saved,
                    "total_steps_eliminated": (
                        opt_metrics.total_steps_eliminated
                    ),
                },
                "agent_performance": [asdict(ap) for ap in agent_perf],
                "optimization_suggestions": (
                    self.optimizer.suggest_optimizations()
                ),
            }

    def export_json(self, path: str) -> None:
        """
        Export full report to JSON file.

        Args:
            path: File path to write JSON to.
        """
        report = self.get_full_report()
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def print_summary(self) -> None:
        """
        Print human-readable summary to terminal.
        """
        report = self.get_full_report()

        print("\n" + "=" * 80)
        print("AGENTOPS DASHBOARD SUMMARY")
        print("=" * 80)

        # Observability section
        print("\n[OBSERVABILITY]")
        obs = report["observability"]
        print(f"  Total Traces:           {obs['total_traces']}")
        print(f"  Total Spans:            {obs['total_spans']}")
        print(f"  Total Duration:         {obs['total_duration_ms']:.2f} ms")
        print(f"  Total Cost:             ${obs['total_cost_usd']:.4f}")
        print(f"  Total Tokens:           {obs['total_tokens']}")

        if obs["agent_handoff_latencies_ms"]:
            print("  Agent Handoff Latencies:")
            for k, v in obs["agent_handoff_latencies_ms"].items():
                print(f"    {k}: {v:.2f} ms")

        if obs["tool_execution_latencies_ms"]:
            print("  Tool Execution Latencies:")
            for k, v in obs["tool_execution_latencies_ms"].items():
                print(f"    {k}: {v:.2f} ms")

        # Evaluation section
        print("\n[EVALUATION]")
        eva = report["evaluation"]
        print(f"  Task Completion Rate:   {eva['task_completion_rate']:.1%}")
        print(
            f"  Guardrail Violation Rate: "
            f"{eva['guardrail_violation_rate']:.1%}"
        )
        print(f"  Factual Accuracy Rate:  {eva['factual_accuracy_rate']:.1%}")
        print(
            f"  Clinical Appropriateness: "
            f"{eva['clinical_appropriateness']:.1%}"
        )
        print(
            f"  First-Pass Approval Rate: "
            f"{eva['first_pass_approval_rate']:.1%}"
        )

        if eva["violations_by_severity"]:
            print("  Violations by Severity:")
            for k, v in eva["violations_by_severity"].items():
                print(f"    {k}: {v}")

        # Optimization section
        print("\n[OPTIMIZATION]")
        opt = report["optimization"]
        print(
            f"  Token Efficiency:       "
            f"{opt['prompt_token_efficiency']:.1%}"
        )
        print(f"  Handoff Success Rate:   {opt['handoff_success_rate']:.1%}")
        print(f"  Flow Step Efficiency:   {opt['flow_step_efficiency']:.1%}")
        print(
            f"  Improvement Velocity:   "
            f"{opt['improvement_velocity']:.4f} per day"
        )
        print(f"  Tokens Saved:           {opt['total_tokens_saved']}")
        print(f"  Steps Eliminated:       {opt['total_steps_eliminated']}")

        if opt["retrieval_precision_at_k"]:
            print("  Retrieval Precision:")
            for k, v in sorted(opt["retrieval_precision_at_k"].items()):
                print(f"    P@{k}: {v:.1%}")

        # Agent leaderboard
        print("\n[AGENT PERFORMANCE LEADERBOARD]")
        agents = report["agent_performance"]
        if agents:
            # Sort by score descending
            agents = sorted(agents, key=lambda a: a["score"], reverse=True)
            print(f"{'Rank':<5} {'Agent':<20} {'Score':<8} {'Ops':<6} {'Acc':<8}")
            print("-" * 50)
            for i, agent in enumerate(agents[:10], 1):
                print(
                    f"{i:<5} {agent['agent_name']:<20} "
                    f"{agent['score']:<8.1f} {agent['num_operations']:<6} "
                    f"{agent['avg_accuracy']:.1%}"
                )
        else:
            print("  No agent performance data available")

        # Suggestions
        print("\n[OPTIMIZATION SUGGESTIONS]")
        suggestions = report["optimization_suggestions"]
        for i, sugg in enumerate(suggestions, 1):
            print(f"  {i}. {sugg}")

        print("\n" + "=" * 80 + "\n")

    def _compute_agent_leaderboard(self) -> List[AgentPerformance]:
        """
        Compute performance metrics for each agent.

        Returns:
            List of AgentPerformance sorted by composite score.
        """
        # Build agent metrics from tracer
        agent_data = {}

        for trace in self.tracer.completed_traces:
            for span in trace.spans.values():
                agent = span.agent_name
                if agent not in agent_data:
                    agent_data[agent] = {
                        "durations": [],
                        "costs": [],
                        "token_counts": [],
                        "success_count": 0,
                        "total_count": 0,
                    }

                agent_data[agent]["durations"].append(span.duration_ms)
                agent_data[agent]["costs"].append(span.cost_usd)
                agent_data[agent]["token_counts"].append(span.token_count)
                agent_data[agent]["total_count"] += 1

                if span.status == "completed":
                    agent_data[agent]["success_count"] += 1

        # Build agent violations from evaluator
        violations_by_agent = self.evaluator.get_violations_by_agent()

        # Compute performance metrics
        performance = []

        for agent_name, data in agent_data.items():
            num_ops = data["total_count"]
            avg_duration = (
                sum(data["durations"]) / len(data["durations"])
                if data["durations"]
                else 0.0
            )
            total_cost = sum(data["costs"])
            success_rate = (
                data["success_count"] / num_ops if num_ops > 0 else 0.0
            )

            # Estimate accuracy from evaluator
            accuracy_checks = self.evaluator.get_all_accuracy_checks()
            agent_accuracy = 0.0
            if accuracy_checks:
                verified = sum(1 for a in accuracy_checks if a.verified)
                agent_accuracy = verified / len(accuracy_checks)

            violations = len(violations_by_agent.get(agent_name, []))

            # Composite score: weighted combination of metrics
            # success_rate (40%) + accuracy (30%) + low_violations (20%) + efficiency (10%)
            efficiency_score = max(
                0, 1 - (avg_duration / 5000)
            )  # normalize to 5s
            composite_score = (
                success_rate * 40
                + agent_accuracy * 30
                + (1 - min(violations / 10, 1)) * 20
                + efficiency_score * 10
            )

            perf = AgentPerformance(
                agent_name=agent_name,
                num_operations=num_ops,
                avg_operation_duration_ms=avg_duration,
                total_cost_usd=total_cost,
                success_rate=success_rate,
                avg_accuracy=agent_accuracy,
                violations_count=violations,
                score=composite_score,
            )
            performance.append(perf)

        # Sort by score descending
        performance.sort(key=lambda p: p.score, reverse=True)

        return performance

    def get_agent_leaderboard(self) -> List[Dict]:
        """
        Get ranked list of agents by performance score.

        Returns:
            List of agent performance dictionaries, sorted by score (highest first).
        """
        leaderboard = self._compute_agent_leaderboard()
        return [asdict(ap) for ap in leaderboard]

    def get_observability_metrics(self) -> Dict:
        """Get observability layer metrics."""
        return self.tracer.export_metrics()

    def get_evaluation_metrics(self) -> Dict:
        """Get evaluation layer metrics."""
        metrics = self.evaluator.get_metrics()
        return asdict(metrics)

    def get_optimization_metrics(self) -> Dict:
        """Get optimization layer metrics."""
        metrics = self.optimizer.get_metrics()
        return asdict(metrics)

    def clear_all(self) -> None:
        """Clear all metrics and history (useful for testing)."""
        with self._lock:
            self.tracer.clear()
            self.evaluator.clear()
            self.optimizer.clear()
