"""DAG-based orchestrator for parallel agent execution.

Manages agent dependencies and executes agents with maximum parallelism
while respecting dependency constraints. Implements zero-trust verification
and complete audit trails.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from resistancemap.agents.base import BaseAgent, AgentState, AgentResult
from resistancemap.config import ResistanceMapConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentDAG:
    """Directed acyclic graph of agent dependencies.

    Attributes:
        agents: Dict of agent_name -> BaseAgent instance
        edges: Dict of agent_name -> [dependency_names] (what each agent depends on)
    """

    agents: dict[str, BaseAgent] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)

    def add_agent(self, agent: BaseAgent) -> None:
        """Register an agent and its dependencies in the DAG.

        Args:
            agent: BaseAgent instance to add

        Raises:
            ValueError: If agent name already exists or dependency not found later
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent {agent.name} already registered")

        self.agents[agent.name] = agent
        self.edges[agent.name] = agent.dependencies.copy()

    def topological_sort(self) -> list[list[str]]:
        """Topologically sort agents into layers for parallel execution.

        Returns a list of layers where each layer contains agents that can
        run in parallel (they don't depend on each other).

        Returns:
            List of lists: [[agent_names_layer_0], [agent_names_layer_1], ...]

        Raises:
            ValueError: If cycle detected in dependencies
        """
        # Check for missing dependencies
        all_agent_names = set(self.agents.keys())
        for agent_name, deps in self.edges.items():
            for dep in deps:
                if dep not in all_agent_names:
                    raise ValueError(
                        f"Agent {agent_name} depends on unknown agent {dep}"
                    )

        # Kahn's algorithm for topological sort with layer grouping
        in_degree = {name: len(self.edges[name]) for name in self.agents}
        layers = []
        processed = set()

        while len(processed) < len(self.agents):
            # Find all agents ready to execute (all deps satisfied)
            ready = [
                name
                for name in self.agents
                if name not in processed and in_degree[name] == 0
            ]

            if not ready:
                # No ready agents but not all processed = cycle
                remaining = set(self.agents.keys()) - processed
                raise ValueError(f"Cycle detected in agent dependencies: {remaining}")

            layers.append(ready)
            processed.update(ready)

            # Decrement in_degree for agents that depend on ready agents
            for ready_agent in ready:
                for other_agent, deps in self.edges.items():
                    if other_agent in deps:
                        in_degree[other_agent] -= 1

        return layers

    def get_ready_agents(self, completed: set[str]) -> list[str]:
        """Get agents that are ready to execute given completed agents.

        Args:
            completed: Set of agent names that have completed

        Returns:
            List of agent names whose dependencies are all satisfied
        """
        ready = []
        for agent_name, deps in self.edges.items():
            if agent_name not in completed and all(dep in completed for dep in deps):
                ready.append(agent_name)
        return ready

    def validate(self) -> tuple[bool, str]:
        """Validate DAG structure (no cycles, all deps exist).

        Returns:
            (is_valid, error_message) tuple
        """
        try:
            self.topological_sort()
            return True, ""
        except ValueError as e:
            return False, str(e)


@dataclass
class ExecutionPlan:
    """Execution plan with layered agent schedule.

    Attributes:
        layers: List of layers, each containing agent names to run in parallel
        agent_order: Flattened list of all agent names in execution order
    """

    layers: list[list[str]]
    agent_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute flattened agent order."""
        self.agent_order = [name for layer in self.layers for name in layer]


class Orchestrator:
    """Zero-trust parallel agent orchestrator.

    Manages DAG-based execution with maximum parallelism while maintaining
    data integrity through verification and audit trails.

    Attributes:
        dag: AgentDAG instance
        results: Dict of agent_name -> AgentResult
        execution_plan: ExecutionPlan for the DAG
        _verification_chain: List of hashes for audit trail
        _timing_log: Dict of agent_name -> (start_time, end_time)
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.dag = AgentDAG()
        self.results: dict[str, AgentResult] = {}
        self.execution_plan: ExecutionPlan | None = None
        self._verification_chain: list[str] = []
        self._timing_log: dict[str, tuple[float, float]] = {}

    def add_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: BaseAgent instance
        """
        self.dag.add_agent(agent)

    def prepare_execution(self) -> tuple[bool, str]:
        """Validate DAG and prepare execution plan.

        Must be called before run().

        Returns:
            (is_valid, error_message) tuple
        """
        is_valid, error_msg = self.dag.validate()
        if not is_valid:
            return False, error_msg

        try:
            layers = self.dag.topological_sort()
            self.execution_plan = ExecutionPlan(layers=layers)
            return True, ""
        except Exception as e:
            return False, str(e)

    async def run(self, config: ResistanceMapConfig) -> dict[str, AgentResult]:
        """Execute all agents respecting dependencies with maximum parallelism.

        Algorithm:
        1. Topological sort to identify parallelizable layers
        2. For each layer, launch agents concurrently with asyncio.gather()
        3. After each agent completes, verify output hash (zero-trust)
        4. Pass verified outputs to dependent agents
        5. Record execution traces and verification chain

        Args:
            config: ResistanceMapConfig object

        Returns:
            Dict of agent_name -> AgentResult

        Raises:
            RuntimeError: If prepare_execution() not called or DAG invalid
            Exception: If any agent fails (logged but not re-raised)
        """
        if self.execution_plan is None:
            raise RuntimeError("Call prepare_execution() before run()")

        start_time = time.time()
        logger.info(
            f"Orchestrator: Starting execution of {len(self.dag.agents)} agents "
            f"in {len(self.execution_plan.layers)} layers"
        )

        try:
            # Execute each layer
            for layer_idx, layer in enumerate(self.execution_plan.layers):
                logger.info(f"Orchestrator: Layer {layer_idx} ({len(layer)} agents)")

                # Run all agents in layer concurrently
                layer_results = await asyncio.gather(
                    *[
                        self._run_agent_isolated(
                            self.dag.agents[name], name, config
                        )
                        for name in layer
                    ],
                    return_exceptions=False,
                )

                # Store results and update verification chain
                for agent_name, result in zip(layer, layer_results):
                    self.results[agent_name] = result
                    self._verification_chain.append(result.verification_hash)

                    # Log result
                    if result.status == AgentState.COMPLETED:
                        logger.info(
                            f"  {agent_name}: COMPLETED "
                            f"(hash={result.verification_hash[:8]}...)"
                        )
                    else:
                        logger.error(
                            f"  {agent_name}: {result.status.value.upper()} "
                            f"({result.error or 'no error msg'})"
                        )

        except Exception as e:
            logger.exception("Orchestrator: Unexpected error during execution")
            raise

        elapsed = time.time() - start_time
        logger.info(
            f"Orchestrator: Execution complete in {elapsed:.1f}s. "
            f"Completed: {sum(1 for r in self.results.values() if r.status == AgentState.COMPLETED)}/{len(self.dag.agents)}"
        )

        return self.results

    async def _run_agent_isolated(
        self, agent: BaseAgent, agent_name: str, config: ResistanceMapConfig
    ) -> AgentResult:
        """Run a single agent with input preparation and output verification.

        Simulates isolated execution by:
        1. Gathering inputs from completed dependencies
        2. Preparing input dict with verified outputs
        3. Executing agent with zero-trust wrapper
        4. Recording execution timing

        Args:
            agent: BaseAgent instance
            agent_name: Name of agent (for lookup)
            config: ResistanceMapConfig

        Returns:
            AgentResult
        """
        start_time = time.time()

        try:
            # Prepare inputs from dependencies
            inputs = {}
            for dep_name in agent.dependencies:
                if dep_name not in self.results:
                    return agent._make_result(
                        AgentState.FAILED,
                        error=f"Dependency {dep_name} not executed yet",
                    )

                dep_result = self.results[dep_name]
                if dep_result.status != AgentState.COMPLETED:
                    return agent._make_result(
                        AgentState.FAILED,
                        error=f"Dependency {dep_name} failed: {dep_result.error}",
                    )

                # Zero-trust: Verify dependency output before using
                is_valid, error_msg = agent.verify_output(dep_result)
                if not is_valid:
                    return agent._make_result(
                        AgentState.FAILED,
                        error=f"Dependency {dep_name} failed verification: {error_msg}",
                    )

                inputs[dep_name] = dep_result.output

            # Execute agent with zero-trust wrapper
            result = await agent._safe_execute(inputs, config)

            # Record timing
            self._timing_log[agent_name] = (start_time, time.time())

            return result

        except Exception as e:
            logger.exception(f"_run_agent_isolated({agent_name}): Unexpected error")
            self._timing_log[agent_name] = (start_time, time.time())
            return agent._make_result(
                AgentState.FAILED,
                error=f"Unexpected error: {str(e)}",
            )

    def get_execution_plan(self) -> list[list[str]]:
        """Get the execution plan (agents grouped by parallelizable layers).

        Returns:
            List of layers, each containing agent names
        """
        if self.execution_plan is None:
            return []
        return self.execution_plan.layers

    def get_verification_chain(self) -> list[str]:
        """Get the audit trail of output hashes.

        Returns:
            List of SHA256 hashes in execution order
        """
        return self._verification_chain.copy()

    def get_timing_report(self) -> dict[str, dict[str, float]]:
        """Get execution timing for each agent.

        Returns:
            Dict of agent_name -> {start_time, end_time, elapsed_seconds}
        """
        report = {}
        for agent_name, (start, end) in self._timing_log.items():
            report[agent_name] = {
                "start_time": start,
                "end_time": end,
                "elapsed_seconds": end - start,
            }
        return report

    def get_results_summary(self) -> dict[str, Any]:
        """Get summary of all agent results.

        Returns:
            Dict with result counts and metadata
        """
        status_counts = defaultdict(int)
        for result in self.results.values():
            status_counts[result.status.value] += 1

        return {
            "total_agents": len(self.results),
            "status_counts": dict(status_counts),
            "verification_chain_length": len(self._verification_chain),
            "failed_agents": [
                name
                for name, result in self.results.items()
                if result.status == AgentState.FAILED
            ],
        }
