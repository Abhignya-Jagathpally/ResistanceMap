"""
Observability layer for ResistanceMap — end-to-end distributed tracing inspired by OpenTelemetry.

Tracks spans (units of work), traces (full pipeline executions), and provides metrics like:
- Agent handoff latencies
- Tool execution latencies
- Cost per request
- Critical path analysis
"""

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, List, Optional


@dataclass
class Span:
    """A single unit of work within a trace."""

    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    parent_id: Optional[str] = None
    agent_name: str = ""
    operation: str = ""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    status: str = "running"  # "running" | "completed" | "failed"

    metadata: Dict = field(default_factory=dict)
    children: List['Span'] = field(default_factory=list)

    def __post_init__(self):
        """Ensure start_time is set if not provided."""
        if self.start_time is None:
            self.start_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Duration of this span in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def cost_usd(self) -> float:
        """Extract cost from metadata."""
        return self.metadata.get("cost_usd", 0.0)

    @property
    def token_count(self) -> int:
        """Extract total token count from metadata."""
        input_tokens = self.metadata.get("input_tokens", 0)
        output_tokens = self.metadata.get("output_tokens", 0)
        return input_tokens + output_tokens

    def to_dict(self) -> Dict:
        """Serialize span to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "agent_name": self.agent_name,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
            "cost_usd": self.cost_usd,
            "token_count": self.token_count,
            "num_children": len(self.children),
        }


@dataclass
class Trace:
    """End-to-end trace of a pipeline execution."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spans: Dict[str, Span] = field(default_factory=dict)

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def __post_init__(self):
        """Ensure start_time is set."""
        if self.start_time is None:
            self.start_time = time.time()

    @property
    def total_duration_ms(self) -> float:
        """Total duration of the trace in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def total_cost(self) -> float:
        """Sum of all span costs."""
        return sum(span.cost_usd for span in self.spans.values())

    @property
    def total_tokens(self) -> int:
        """Sum of all span tokens."""
        return sum(span.token_count for span in self.spans.values())

    def get_agent_handoff_latencies(self) -> Dict[str, float]:
        """
        Compute latencies between agent handoffs.
        Maps "agent_A -> agent_B" to milliseconds elapsed between A's end and B's start.
        """
        latencies = {}
        span_list = sorted(self.spans.values(), key=lambda s: s.start_time)

        for i in range(len(span_list) - 1):
            current = span_list[i]
            next_span = span_list[i + 1]

            if current.agent_name and next_span.agent_name:
                if current.end_time is not None:
                    latency_ms = (next_span.start_time - current.end_time) * 1000
                    key = f"{current.agent_name} -> {next_span.agent_name}"
                    latencies[key] = latency_ms

        return latencies

    def get_tool_execution_latencies(self) -> Dict[str, float]:
        """
        Compute average execution latency per tool/operation.
        Maps "operation_name" to average milliseconds.
        """
        operation_times = {}
        for span in self.spans.values():
            key = span.operation or "unknown"
            if key not in operation_times:
                operation_times[key] = []
            operation_times[key].append(span.duration_ms)

        latencies = {}
        for op, times in operation_times.items():
            latencies[op] = sum(times) / len(times) if times else 0.0

        return latencies

    def get_cost_per_request(self) -> float:
        """Total cost of this trace (all spans)."""
        return self.total_cost

    def get_critical_path(self) -> List[Span]:
        """
        Return the longest sequential chain of spans.
        This represents the critical path through the trace.
        """
        if not self.spans:
            return []

        # Build parent -> children map
        children_map: Dict[Optional[str], List[Span]] = {}
        for span in self.spans.values():
            parent = span.parent_id
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(span)

        # Find root spans (parent_id is None)
        roots = children_map.get(None, [])

        # DFS to find longest path
        def longest_path(span: Span) -> List[Span]:
            children = children_map.get(span.span_id, [])
            if not children:
                return [span]

            longest = [span]
            for child in children:
                child_path = longest_path(child)
                if len(child_path) + 1 > len(longest):
                    longest = [span] + child_path

            return longest

        if not roots:
            return []

        critical = []
        for root in roots:
            path = longest_path(root)
            if len(path) > len(critical):
                critical = path

        return critical

    def to_dict(self) -> Dict:
        """Serialize trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "num_spans": len(self.spans),
            "spans": {sid: s.to_dict() for sid, s in self.spans.items()},
        }


class Tracer:
    """
    Global tracer singleton for distributed tracing.

    Thread-safe implementation with support for concurrent trace execution.
    Provides context manager support for automatic span management.
    """

    _instance: ClassVar[Optional['Tracer']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self):
        self.active_traces: Dict[str, Trace] = {}
        self.completed_traces: List[Trace] = []
        self._span_stack: Dict[int, List[str]] = {}  # thread_id -> [span_ids]
        self._thread_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'Tracer':
        """Get singleton instance of Tracer."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_trace(self, trace_id: Optional[str] = None) -> Trace:
        """
        Start a new trace.

        Args:
            trace_id: Optional explicit trace ID. Generated if not provided.

        Returns:
            The created Trace object.
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        trace = Trace(trace_id=trace_id)

        with self._thread_lock:
            self.active_traces[trace_id] = trace

        return trace

    def start_span(
        self,
        trace_id: str,
        agent_name: str,
        operation: str,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """
        Start a new span within a trace.

        Args:
            trace_id: The trace this span belongs to.
            agent_name: Name of the agent executing this span.
            operation: The operation being performed.
            parent_span_id: Optional parent span ID for nesting.

        Returns:
            The created Span object.
        """
        with self._thread_lock:
            if trace_id not in self.active_traces:
                raise ValueError(f"Trace {trace_id} not found")

            trace = self.active_traces[trace_id]
            span = Span(
                trace_id=trace_id,
                agent_name=agent_name,
                operation=operation,
                parent_id=parent_span_id,
            )

            trace.spans[span.span_id] = span

            # Track span stack for this thread
            thread_id = threading.get_ident()
            if thread_id not in self._span_stack:
                self._span_stack[thread_id] = []
            self._span_stack[thread_id].append(span.span_id)

        return span

    def end_span(
        self,
        span_id: str,
        status: str = "completed",
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        End a span and mark its status.

        Args:
            span_id: The span ID to end.
            status: Final status ("completed", "failed", etc.).
            metadata: Optional metadata to attach (tokens, cost, etc.).
        """
        if metadata is None:
            metadata = {}

        with self._thread_lock:
            # Find the span across all traces
            for trace in self.active_traces.values():
                if span_id in trace.spans:
                    span = trace.spans[span_id]
                    span.end_time = time.time()
                    span.status = status
                    span.metadata.update(metadata)

                    # Pop from span stack
                    thread_id = threading.get_ident()
                    if thread_id in self._span_stack and self._span_stack[thread_id]:
                        self._span_stack[thread_id].pop()

                    return

    def end_trace(self, trace_id: str) -> None:
        """
        End a trace and move it to completed list.

        Args:
            trace_id: The trace ID to end.
        """
        with self._thread_lock:
            if trace_id in self.active_traces:
                trace = self.active_traces.pop(trace_id)
                trace.end_time = time.time()
                self.completed_traces.append(trace)

    def export_metrics(self) -> Dict:
        """
        Export all observability metrics from completed traces.

        Returns:
            Dictionary with comprehensive metrics.
        """
        total_traces = len(self.completed_traces)
        total_spans = sum(len(t.spans) for t in self.completed_traces)
        total_cost = sum(t.total_cost for t in self.completed_traces)
        total_tokens = sum(t.total_tokens for t in self.completed_traces)
        total_time_ms = sum(t.total_duration_ms for t in self.completed_traces)

        # Aggregate latencies
        all_handoff_latencies = {}
        all_tool_latencies = {}

        for trace in self.completed_traces:
            for key, latency in trace.get_agent_handoff_latencies().items():
                if key not in all_handoff_latencies:
                    all_handoff_latencies[key] = []
                all_handoff_latencies[key].append(latency)

            for key, latency in trace.get_tool_execution_latencies().items():
                if key not in all_tool_latencies:
                    all_tool_latencies[key] = []
                all_tool_latencies[key].append(latency)

        # Average latencies
        avg_handoff_latencies = {
            k: sum(v) / len(v) for k, v in all_handoff_latencies.items()
        }
        avg_tool_latencies = {
            k: sum(v) / len(v) for k, v in all_tool_latencies.items()
        }

        return {
            "total_traces": total_traces,
            "total_spans": total_spans,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "total_duration_ms": total_time_ms,
            "avg_cost_per_trace": total_cost / total_traces if total_traces > 0 else 0,
            "avg_tokens_per_trace": total_tokens / total_traces if total_traces > 0 else 0,
            "avg_duration_ms": total_time_ms / total_traces if total_traces > 0 else 0,
            "agent_handoff_latencies_ms": avg_handoff_latencies,
            "tool_execution_latencies_ms": avg_tool_latencies,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_active_trace_ids(self) -> List[str]:
        """Get list of currently active trace IDs."""
        with self._thread_lock:
            return list(self.active_traces.keys())

    def get_completed_trace_ids(self) -> List[str]:
        """Get list of completed trace IDs."""
        with self._thread_lock:
            return [t.trace_id for t in self.completed_traces]

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a trace by ID (active or completed)."""
        with self._thread_lock:
            if trace_id in self.active_traces:
                return self.active_traces[trace_id]

            for trace in self.completed_traces:
                if trace.trace_id == trace_id:
                    return trace

        return None

    @contextmanager
    def trace_agent(self, agent_name: str, operation: str, trace_id: str = None):
        """
        Context manager for automatic span lifecycle management.

        Usage:
            tracer = Tracer.get_instance()
            trace = tracer.start_trace()
            with tracer.trace_agent("agent_name", "operation", trace.trace_id) as span:
                # do work
                span.metadata["result"] = "success"

        Args:
            agent_name: Name of the agent.
            operation: Operation name.
            trace_id: Trace ID (uses current active trace if not provided).

        Yields:
            The created Span object.
        """
        if trace_id is None:
            # Try to find an active trace
            with self._thread_lock:
                if self.active_traces:
                    trace_id = next(iter(self.active_traces.keys()))
                else:
                    raise ValueError("No active trace found. Start a trace first.")

        span = self.start_span(trace_id, agent_name, operation)
        try:
            yield span
            self.end_span(span.span_id, status="completed")
        except Exception as e:
            self.end_span(span.span_id, status="failed", metadata={"error": str(e)})
            raise

    def clear(self) -> None:
        """Clear all traces (useful for testing)."""
        with self._thread_lock:
            self.active_traces.clear()
            self.completed_traces.clear()
            self._span_stack.clear()
