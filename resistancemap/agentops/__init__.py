"""
AgentOps Framework for ResistanceMap.

Core observability, evaluation, and optimization layer for agent-based systems.

Three main components:
1. Tracer (observability): End-to-end distributed tracing of pipeline execution
2. Evaluator (evaluation): Task completion, guardrails, accuracy, clinical checks
3. Optimizer (optimization): Token efficiency, retrieval precision, handoff success

Usage:
    from resistancemap.agentops import Tracer, Evaluator, Optimizer, AgentOpsDashboard

    # Initialize components
    tracer = Tracer.get_instance()
    evaluator = Evaluator()
    optimizer = Optimizer()
    dashboard = AgentOpsDashboard(tracer, evaluator, optimizer)

    # Start tracing
    trace = tracer.start_trace()
    with tracer.trace_agent("agent_name", "operation", trace.trace_id) as span:
        # Do work
        span.metadata["result"] = "success"

    # Record metrics
    evaluator.record_task_result("task_1", "completed")
    optimizer.record_token_usage(100, 50, 45)

    # View results
    dashboard.print_summary()
    dashboard.export_json("report.json")
"""

from .dashboard import AgentOpsDashboard, AgentPerformance
from .evaluator import (
    AccuracyCheck,
    ClinicalCheck,
    Evaluator,
    EvaluationMetrics,
    FirstPassCheck,
    GuardrailViolation,
    TaskResult,
)
from .optimizer import (
    FlowStepRecord,
    HandoffRecord,
    Optimizer,
    OptimizationMetrics,
    RetrievalRecord,
    TokenUsageRecord,
)
from .tracer import Span, Trace, Tracer

__all__ = [
    # Tracer and related
    "Tracer",
    "Trace",
    "Span",
    # Evaluator and related
    "Evaluator",
    "EvaluationMetrics",
    "GuardrailViolation",
    "TaskResult",
    "AccuracyCheck",
    "ClinicalCheck",
    "FirstPassCheck",
    # Optimizer and related
    "Optimizer",
    "OptimizationMetrics",
    "TokenUsageRecord",
    "RetrievalRecord",
    "HandoffRecord",
    "FlowStepRecord",
    # Dashboard
    "AgentOpsDashboard",
    "AgentPerformance",
]
