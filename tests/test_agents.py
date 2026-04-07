"""Unit and integration tests for the multi-agent orchestrator system.

Tests cover:
1. BaseAgent functionality (hashing, input/output verification)
2. AgentDAG topological sorting and cycle detection
3. Orchestrator execution and parallelism
4. Specialized agent implementations
5. Zero-trust verification chain integrity
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from resistancemap.agents.base import (
    AgentState,
    AgentResult,
    BaseAgent,
)
from resistancemap.agents.orchestrator import (
    AgentDAG,
    Orchestrator,
)
from resistancemap.agents.specialized import (
    DataValidationAgent,
    DataPrepAgent,
    VAEPretrainAgent,
    ValidationAgent,
)
from resistancemap.config import ResistanceMapConfig


# ============================================================================
# Test BaseAgent
# ============================================================================


class TestHashComputation:
    """Test hash computation for various data types."""

    def test_hash_dict(self):
        """Hash computation for dicts."""
        data1 = {"a": 1, "b": 2}
        data2 = {"a": 1, "b": 2}
        data3 = {"a": 1, "b": 3}

        hash1 = BaseAgent.compute_hash(data1)
        hash2 = BaseAgent.compute_hash(data2)
        hash3 = BaseAgent.compute_hash(data3)

        assert hash1 == hash2, "Same dict should have same hash"
        assert hash1 != hash3, "Different dict should have different hash"

    def test_hash_list(self):
        """Hash computation for lists."""
        data1 = [1, 2, 3]
        data2 = [1, 2, 3]
        data3 = [1, 2, 4]

        hash1 = BaseAgent.compute_hash(data1)
        hash2 = BaseAgent.compute_hash(data2)
        hash3 = BaseAgent.compute_hash(data3)

        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_tensor(self):
        """Hash computation for PyTorch tensors."""
        tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor3 = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

        hash1 = BaseAgent.compute_hash(tensor1)
        hash2 = BaseAgent.compute_hash(tensor2)
        hash3 = BaseAgent.compute_hash(tensor3)

        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_string(self):
        """Hash computation for strings."""
        hash1 = BaseAgent.compute_hash("test")
        hash2 = BaseAgent.compute_hash("test")
        hash3 = BaseAgent.compute_hash("different")

        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_none(self):
        """Hash computation for None."""
        hash1 = BaseAgent.compute_hash(None)
        hash2 = BaseAgent.compute_hash(None)

        assert hash1 == hash2


class TestAgentResult:
    """Test AgentResult data structure."""

    def test_agent_result_creation(self):
        """Create AgentResult with proper fields."""
        output = {"key": "value"}
        result = AgentResult(
            agent_name="test_agent",
            status=AgentState.COMPLETED,
            output=output,
            verification_hash=BaseAgent.compute_hash(output),
        )

        assert result.agent_name == "test_agent"
        assert result.status == AgentState.COMPLETED
        assert result.output == output
        assert result.error is None

    def test_agent_result_to_dict(self):
        """Serialize AgentResult to dict."""
        output = {"key": "value"}
        result = AgentResult(
            agent_name="test",
            status=AgentState.COMPLETED,
            output=output,
            verification_hash="abc123",
            metadata={"metric": 0.95},
        )

        result_dict = result.to_dict()

        assert result_dict["agent_name"] == "test"
        assert result_dict["status"] == "completed"
        assert result_dict["verification_hash"] == "abc123"
        assert "metric" in result_dict["metadata"]


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str, dependencies: list[str] | None = None):
        """Initialize mock agent."""
        super().__init__(name, dependencies or [])
        self.execute_called = False

    async def execute(self, inputs: dict[str, Any], config: Any) -> AgentResult:
        """Mock execute that returns a simple result."""
        self.execute_called = True
        output = {"mock_data": "test_value", "input_count": len(inputs)}
        return self._make_result(AgentState.COMPLETED, output, metadata={"mock": True})


class TestMockAgentExecution:
    """Test basic agent execution."""

    @pytest.mark.asyncio
    async def test_agent_execute(self):
        """Execute a mock agent."""
        agent = MockAgent("test_agent")
        result = await agent.execute({}, ResistanceMapConfig())

        assert agent.execute_called
        assert result.status == AgentState.COMPLETED
        assert result.agent_name == "test_agent"
        assert "mock_data" in result.output

    @pytest.mark.asyncio
    async def test_agent_safe_execute(self):
        """Test agent execution with verification."""
        agent = MockAgent("test_agent")
        result = await agent._safe_execute({}, ResistanceMapConfig())

        assert result.status == AgentState.COMPLETED
        assert result.verification_hash != ""

    @pytest.mark.asyncio
    async def test_agent_input_validation(self):
        """Test input validation."""
        agent = MockAgent("test_agent")

        # Invalid input (not dict)
        result = await agent._safe_execute("invalid", ResistanceMapConfig())

        assert result.status == AgentState.FAILED
        assert "validation failed" in result.error.lower()


# ============================================================================
# Test AgentDAG
# ============================================================================


class TestAgentDAG:
    """Test directed acyclic graph functionality."""

    def test_dag_add_agent(self):
        """Add agents to DAG."""
        dag = AgentDAG()
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2", dependencies=["agent1"])

        dag.add_agent(agent1)
        dag.add_agent(agent2)

        assert "agent1" in dag.agents
        assert "agent2" in dag.agents
        assert dag.edges["agent2"] == ["agent1"]

    def test_dag_duplicate_agent(self):
        """Prevent duplicate agent names."""
        dag = AgentDAG()
        agent = MockAgent("agent1")

        dag.add_agent(agent)

        with pytest.raises(ValueError, match="already registered"):
            dag.add_agent(agent)

    def test_dag_topological_sort(self):
        """Topological sort with parallelization."""
        dag = AgentDAG()

        # Create DAG:
        # Layer 0: a, b
        # Layer 1: c (depends on a), d (depends on b)
        # Layer 2: e (depends on c, d)

        agents = {
            "a": MockAgent("a"),
            "b": MockAgent("b"),
            "c": MockAgent("c", dependencies=["a"]),
            "d": MockAgent("d", dependencies=["b"]),
            "e": MockAgent("e", dependencies=["c", "d"]),
        }

        for agent in agents.values():
            dag.add_agent(agent)

        layers = dag.topological_sort()

        assert len(layers) == 3
        assert set(layers[0]) == {"a", "b"}  # Parallel layer
        assert set(layers[1]) == {"c", "d"}  # Parallel layer
        assert layers[2] == ["e"]

    def test_dag_cycle_detection(self):
        """Detect cycles in DAG."""
        dag = AgentDAG()

        # Create cycle: a -> b -> a
        a = MockAgent("a")
        b = MockAgent("b", dependencies=["a"])

        dag.add_agent(a)
        dag.add_agent(b)
        dag.edges["a"] = ["b"]  # Add cycle

        with pytest.raises(ValueError, match="Cycle detected"):
            dag.topological_sort()

    def test_dag_missing_dependency(self):
        """Detect missing dependencies."""
        dag = AgentDAG()

        agent = MockAgent("agent", dependencies=["nonexistent"])
        dag.add_agent(agent)

        with pytest.raises(ValueError, match="unknown agent"):
            dag.topological_sort()

    def test_dag_validate(self):
        """Validate DAG structure."""
        dag = AgentDAG()

        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2", dependencies=["agent1"])

        dag.add_agent(agent1)
        dag.add_agent(agent2)

        is_valid, error_msg = dag.validate()

        assert is_valid
        assert error_msg == ""


# ============================================================================
# Test Orchestrator
# ============================================================================


class TestOrchestrator:
    """Test orchestrator execution and parallelism."""

    def test_orchestrator_creation(self):
        """Create orchestrator."""
        orchestrator = Orchestrator()

        assert orchestrator.dag is not None
        assert len(orchestrator.results) == 0
        assert len(orchestrator._verification_chain) == 0

    def test_orchestrator_add_agents(self):
        """Add agents to orchestrator."""
        orchestrator = Orchestrator()

        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2", dependencies=["agent1"])

        orchestrator.add_agent(agent1)
        orchestrator.add_agent(agent2)

        assert "agent1" in orchestrator.dag.agents
        assert "agent2" in orchestrator.dag.agents

    def test_orchestrator_prepare_execution(self):
        """Prepare execution plan."""
        orchestrator = Orchestrator()

        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2", dependencies=["agent1"])

        orchestrator.add_agent(agent1)
        orchestrator.add_agent(agent2)

        is_valid, error_msg = orchestrator.prepare_execution()

        assert is_valid
        assert orchestrator.execution_plan is not None
        assert len(orchestrator.execution_plan.layers) == 2

    def test_orchestrator_get_execution_plan(self):
        """Get execution plan layers."""
        orchestrator = Orchestrator()

        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")
        agent3 = MockAgent("agent3", dependencies=["agent1", "agent2"])

        orchestrator.add_agent(agent1)
        orchestrator.add_agent(agent2)
        orchestrator.add_agent(agent3)

        orchestrator.prepare_execution()
        plan = orchestrator.get_execution_plan()

        assert len(plan) == 2
        assert set(plan[0]) == {"agent1", "agent2"}
        assert plan[1] == ["agent3"]

    @pytest.mark.asyncio
    async def test_orchestrator_run(self):
        """Run orchestrator with mock agents."""
        orchestrator = Orchestrator()

        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2", dependencies=["agent1"])

        orchestrator.add_agent(agent1)
        orchestrator.add_agent(agent2)

        is_valid, _ = orchestrator.prepare_execution()
        assert is_valid

        config = ResistanceMapConfig()
        results = await orchestrator.run(config)

        assert len(results) == 2
        assert results["agent1"].status == AgentState.COMPLETED
        assert results["agent2"].status == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_orchestrator_verification_chain(self):
        """Verify verification chain is recorded."""
        orchestrator = Orchestrator()

        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2", dependencies=["agent1"])

        orchestrator.add_agent(agent1)
        orchestrator.add_agent(agent2)

        orchestrator.prepare_execution()
        await orchestrator.run(ResistanceMapConfig())

        chain = orchestrator.get_verification_chain()

        assert len(chain) == 2
        for hash_value in chain:
            assert len(hash_value) == 64  # SHA256 hex is 64 chars

    @pytest.mark.asyncio
    async def test_orchestrator_timing_report(self):
        """Get timing report for execution."""
        orchestrator = Orchestrator()

        agent = MockAgent("agent1")
        orchestrator.add_agent(agent)

        orchestrator.prepare_execution()
        await orchestrator.run(ResistanceMapConfig())

        timing = orchestrator.get_timing_report()

        assert "agent1" in timing
        assert "start_time" in timing["agent1"]
        assert "end_time" in timing["agent1"]
        assert "elapsed_seconds" in timing["agent1"]
        assert timing["agent1"]["elapsed_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_orchestrator_results_summary(self):
        """Get results summary."""
        orchestrator = Orchestrator()

        agent = MockAgent("agent1")
        orchestrator.add_agent(agent)

        orchestrator.prepare_execution()
        await orchestrator.run(ResistanceMapConfig())

        summary = orchestrator.get_results_summary()

        assert summary["total_agents"] == 1
        assert summary["status_counts"]["completed"] == 1
        assert len(summary["failed_agents"]) == 0


# ============================================================================
# Test Specialized Agents
# ============================================================================


class TestSpecializedAgents:
    """Test specific specialized agent implementations."""

    @pytest.mark.asyncio
    async def test_data_validation_agent(self):
        """Test DataValidationAgent."""
        agent = DataValidationAgent()

        assert agent.name == "data_validation"
        assert agent.dependencies == []

        # Mock execution
        result = await agent.execute({}, ResistanceMapConfig())

        assert result.agent_name == "data_validation"
        assert result.status in [AgentState.COMPLETED, AgentState.FAILED]

    @pytest.mark.asyncio
    async def test_data_prep_agent(self):
        """Test DataPrepAgent."""
        agent = DataPrepAgent()

        assert agent.name == "data_prep"
        assert "data_validation" in agent.dependencies

        # Should fail without proper input
        result = await agent.execute({}, ResistanceMapConfig())

        assert result.status == AgentState.FAILED

    @pytest.mark.asyncio
    async def test_vae_pretrain_agent(self):
        """Test VAEPretrainAgent."""
        agent = VAEPretrainAgent()

        assert agent.name == "vae_pretrain"
        assert "data_prep" in agent.dependencies


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.asyncio
    async def test_three_layer_pipeline(self):
        """Test a 3-layer dependency pipeline."""
        orchestrator = Orchestrator()

        # Create pipeline: a -> b -> c
        a = MockAgent("a")
        b = MockAgent("b", dependencies=["a"])
        c = MockAgent("c", dependencies=["b"])

        for agent in [a, b, c]:
            orchestrator.add_agent(agent)

        is_valid, _ = orchestrator.prepare_execution()
        assert is_valid

        config = ResistanceMapConfig()
        results = await orchestrator.run(config)

        # All agents should complete
        assert all(r.status == AgentState.COMPLETED for r in results.values())

        # Verify order in chain
        chain = orchestrator.get_verification_chain()
        assert len(chain) == 3

    @pytest.mark.asyncio
    async def test_parallel_pipeline(self):
        """Test a pipeline with parallel execution."""
        orchestrator = Orchestrator()

        # Create pipeline:
        # Layer 0: a, b (parallel)
        # Layer 1: c (depends on a, b)

        a = MockAgent("a")
        b = MockAgent("b")
        c = MockAgent("c", dependencies=["a", "b"])

        for agent in [a, b, c]:
            orchestrator.add_agent(agent)

        is_valid, _ = orchestrator.prepare_execution()
        assert is_valid

        plan = orchestrator.get_execution_plan()
        assert len(plan) == 2
        assert len(plan[0]) == 2  # a, b in parallel

        config = ResistanceMapConfig()
        results = await orchestrator.run(config)

        # All should complete
        assert all(r.status == AgentState.COMPLETED for r in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
