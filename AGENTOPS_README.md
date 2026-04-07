# AgentOps Framework for ResistanceMap

Production-grade observability, evaluation, and optimization layer for agent-based systems.

## Overview

AgentOps provides three integrated layers:

1. **Tracer** (observability) — Distributed tracing of pipeline execution
2. **Evaluator** (evaluation) — Task completion, guardrails, and accuracy tracking
3. **Optimizer** (optimization) — Efficiency metrics and improvement suggestions
4. **Dashboard** (reporting) — Unified view with JSON export and terminal summaries

## Quick Start

```python
from resistancemap.agentops import Tracer, Evaluator, Optimizer, AgentOpsDashboard

# Initialize components
tracer = Tracer.get_instance()  # Singleton
evaluator = Evaluator()
optimizer = Optimizer()
dashboard = AgentOpsDashboard(tracer, evaluator, optimizer)

# Start tracing
trace = tracer.start_trace()

# Option 1: Manual span management
span = tracer.start_span(trace.trace_id, "agent_name", "operation")
# ... do work ...
tracer.end_span(span.span_id, "completed", {"cost_usd": 0.001})
tracer.end_trace(trace.trace_id)

# Option 2: Context manager (recommended)
with tracer.trace_agent("agent_name", "operation", trace.trace_id) as span:
    # do work
    span.metadata["result"] = "success"

# Record evaluation metrics
evaluator.record_task_result("task_001", "completed", verified=True)
evaluator.record_guardrail_violation("agent_a", "hallucination", "medium", "Unsupported claim")
evaluator.record_accuracy_check("claim", "FactDB", verified=True, confidence=0.95)
evaluator.record_clinical_check("recommendation", appropriate=True, reviewer="Dr. Smith")
evaluator.record_first_pass("output_001", approved=True)

# Record optimization metrics
optimizer.record_token_usage(200, 100, 95)  # input, output, useful tokens
optimizer.record_retrieval("query", results, relevant_results, k=10)
optimizer.record_handoff("agent_a", "agent_b", success=True, latency_ms=50)
optimizer.record_flow_step("step_1", necessary=True, duration_ms=100)

# Get comprehensive report
dashboard.print_summary()
dashboard.export_json("report.json")

# Get specific metrics
eval_metrics = evaluator.get_metrics()  # EvaluationMetrics object
opt_metrics = optimizer.get_metrics()   # OptimizationMetrics object
tracer_metrics = tracer.export_metrics() # Dict with all tracer data

# Get agent leaderboard
leaderboard = dashboard.get_agent_leaderboard()
for agent in leaderboard:
    print(f"{agent['agent_name']}: {agent['score']}/100")
```

## Core Classes

### Tracer (Observability)

Distributed tracing with OpenTelemetry-inspired design.

**Key Classes:**
- `Span` — Unit of work with span_id, parent_id, agent_name, operation, timing, metadata
- `Trace` — Collection of spans forming a complete pipeline execution
- `Tracer` — Singleton managing all traces

**Key Methods:**
```python
tracer = Tracer.get_instance()

# Create and manage traces
trace = tracer.start_trace(trace_id=None)
span = tracer.start_span(trace.trace_id, agent_name, operation, parent_span_id=None)
tracer.end_span(span.span_id, status="completed", metadata={})
tracer.end_trace(trace.trace_id)

# Context manager
with tracer.trace_agent(agent_name, operation, trace_id) as span:
    # span is created and auto-closed

# Analytics
tracer.export_metrics()  # Dict with all observability data
trace.get_agent_handoff_latencies()  # Dict[str, float] ms
trace.get_tool_execution_latencies()  # Dict[str, float] ms
trace.get_cost_per_request()  # float USD
trace.get_critical_path()  # List[Span] longest sequential chain
```

### Evaluator (Evaluation)

Task completion, guardrails, accuracy, and clinical review tracking.

**Key Classes:**
- `GuardrailViolation` — Records safety/hallucination/limit violations
- `TaskResult` — Task completion status (completed/failed/partial)
- `AccuracyCheck` — Factual verification against sources
- `ClinicalCheck` — Clinical appropriateness review
- `EvaluationMetrics` — Aggregate rates and counts

**Key Methods:**
```python
evaluator = Evaluator()

# Record metrics
evaluator.record_task_result(task_id, status, verified=False, metadata={})
evaluator.record_guardrail_violation(agent_name, violation_type, severity, description)
evaluator.record_accuracy_check(claim, source, verified, confidence=1.0)
evaluator.record_clinical_check(recommendation, appropriate, reviewer, notes="")
evaluator.record_first_pass(output_id, approved)

# Get metrics
metrics = evaluator.get_metrics()
# Returns: EvaluationMetrics with:
#   - task_completion_rate (float)
#   - guardrail_violation_rate (float)
#   - factual_accuracy_rate (float)
#   - clinical_appropriateness (float)
#   - first_pass_approval_rate (float)
#   - violations_by_type (Dict[str, int])
#   - violations_by_severity (Dict[str, int])

# Analysis
violations_by_agent = evaluator.get_violations_by_agent()  # Dict[str, List[...]]
violations_by_severity = evaluator.get_violations_by_severity()  # Dict[str, List[...]]
```

### Optimizer (Optimization)

Token efficiency, retrieval precision, handoff success, and improvement velocity.

**Key Classes:**
- `TokenUsageRecord` — Input/output/useful tokens
- `RetrievalRecord` — Query, results, precision@k
- `HandoffRecord` — Agent-to-agent transfers and latency
- `FlowStepRecord` — Step necessity and duration
- `OptimizationMetrics` — Aggregate efficiency metrics

**Key Methods:**
```python
optimizer = Optimizer()

# Record metrics
optimizer.record_token_usage(input_tokens, output_tokens, useful_tokens=0)
optimizer.record_retrieval(query, results, relevant_results, k=10)
optimizer.record_handoff(from_agent, to_agent, success, latency_ms)
optimizer.record_flow_step(step_id, necessary, duration_ms)
optimizer.record_metric_snapshot({"metric_name": value, ...})

# Get metrics
metrics = optimizer.get_metrics()
# Returns: OptimizationMetrics with:
#   - prompt_token_efficiency (float) — useful_tokens / input_tokens
#   - retrieval_precision_at_k (Dict[int, float]) — P@1, P@5, P@10
#   - handoff_success_rate (float)
#   - flow_step_efficiency (float) — necessary_steps / total_steps
#   - improvement_velocity (float) — metric_delta / days

# Analytics
velocity = optimizer.get_improvement_velocity("metric_name", window_days=7)
suggestions = optimizer.suggest_optimizations()  # List[str]
```

### Dashboard (Reporting)

Unified view combining all three layers.

**Key Methods:**
```python
dashboard = AgentOpsDashboard(tracer, evaluator, optimizer)

# Get comprehensive report
report = dashboard.get_full_report()  # Dict with all metrics

# Export
dashboard.export_json("report.json")
dashboard.print_summary()  # Terminal-friendly output

# Get agent rankings
leaderboard = dashboard.get_agent_leaderboard()  # List[AgentPerformance]
# Each agent has: agent_name, num_operations, avg_duration_ms, 
#                 total_cost_usd, success_rate, avg_accuracy, 
#                 violations_count, score (0-100)

# Get individual layers
tracer_metrics = dashboard.get_observability_metrics()
eval_metrics = dashboard.get_evaluation_metrics()
opt_metrics = dashboard.get_optimization_metrics()

# Clear for testing
dashboard.clear_all()
```

## Metrics Summary

### Observability (Tracer)
- `total_traces` — Number of pipeline executions
- `total_spans` — Total units of work
- `total_duration_ms` — Cumulative execution time
- `total_cost_usd` — Total API/compute cost
- `total_tokens` — Cumulative LLM tokens
- `agent_handoff_latencies_ms` — Time between agent transitions
- `tool_execution_latencies_ms` — Per-tool average latency

### Evaluation (Evaluator)
- `task_completion_rate` — Completed / attempted (0-1)
- `guardrail_violation_rate` — Violations / operations (0-1)
- `factual_accuracy_rate` — Verified correct / total claims (0-1)
- `clinical_appropriateness` — Clinically valid / total reviews (0-1)
- `first_pass_approval_rate` — Approved first-try / total (0-1)
- `violations_by_type` — Count per violation type
- `violations_by_severity` — Count per severity level

### Optimization (Optimizer)
- `prompt_token_efficiency` — Useful output / input tokens (0-1)
- `retrieval_precision_at_k` — P@1, P@5, P@10 (0-1 each)
- `handoff_success_rate` — Successful / total handoffs (0-1)
- `flow_step_efficiency` — Necessary / total steps (0-1)
- `improvement_velocity` — Metric change per day (float)
- `total_tokens_saved` — Input - useful tokens (int)
- `total_steps_eliminated` — Unnecessary step count (int)

## Thread Safety

All components are thread-safe:
- Tracer uses lock for concurrent trace management
- Evaluator uses lock for concurrent metric recording
- Optimizer uses lock for concurrent efficiency tracking
- Context managers handle automatic cleanup

## Testing

```python
# Clear all metrics between tests
dashboard.clear_all()

# Create isolated instances
evaluator = Evaluator()
optimizer = Optimizer()
custom_dashboard = AgentOpsDashboard(tracer, evaluator, optimizer)
```

## Performance Considerations

- Spans are stored in memory; export regularly for large traces
- Lock contention is minimal due to short critical sections
- Metrics are computed on-demand; cache if needed
- JSON export is fast for typical report sizes (<5MB)

## Files

- `tracer.py` (460 lines) — Distributed tracing
- `evaluator.py` (450 lines) — Task & quality evaluation
- `optimizer.py` (509 lines) — Efficiency tracking
- `dashboard.py` (346 lines) — Unified reporting
- `__init__.py` (79 lines) — Package exports

**Total: 1844 lines, 19 classes, 81 methods**

No external dependencies beyond stdlib + torch.
