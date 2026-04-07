"""ResistanceMap Multi-Agent Orchestrator System.

A parallel, zero-trust agent system that replaces sequential main.py execution.
Agents execute concurrently with maximum parallelism while maintaining data integrity
through zero-trust verification and isolated execution contexts.

Key Components:
    - BaseAgent: Abstract base class for all specialized agents
    - Orchestrator: DAG-based parallel execution engine
    - AgentDAG: Dependency graph for agents
    - Specialized agents: DataValidationAgent, DataPrepAgent, VAEPretrainAgent, etc.

Zero-Trust Principles:
    - All outputs verified via SHA256 hashing
    - Input validation before execution
    - Isolated execution contexts
    - Complete audit trail of hashes
"""

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
    VAEFinetuneAgent,
    ESM2EmbedAgent,
    TrajectoryAgent,
    ProteinNetAgent,
    FusionAgent,
    LandscapeAgent,
    ValidationAgent,
)

__all__ = [
    "AgentState",
    "AgentResult",
    "BaseAgent",
    "AgentDAG",
    "Orchestrator",
    "DataValidationAgent",
    "DataPrepAgent",
    "VAEPretrainAgent",
    "VAEFinetuneAgent",
    "ESM2EmbedAgent",
    "TrajectoryAgent",
    "ProteinNetAgent",
    "FusionAgent",
    "LandscapeAgent",
    "ValidationAgent",
]
