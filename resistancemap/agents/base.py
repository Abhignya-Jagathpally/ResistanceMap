"""Base Agent class and shared abstractions for ResistanceMap orchestrator.

Defines the abstract interface all specialized agents must implement, along with
zero-trust verification utilities and agent state management.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Lifecycle states for agent execution."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_VERIFICATION = "waiting_verification"


@dataclass
class AgentResult:
    """Result object returned by BaseAgent.execute().

    Attributes:
        agent_name: Name of the agent that produced this result
        status: Final state of the agent (COMPLETED, FAILED, etc.)
        output: The actual result (dict, tensor, or any serializable object)
        verification_hash: SHA256 hash of output for zero-trust verification
        metadata: Additional info (timing, resource usage, validation details)
        error: Error message if status == FAILED
    """

    agent_name: str
    status: AgentState
    output: Any
    verification_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        """Ensure status is AgentState enum."""
        if isinstance(self.status, str):
            self.status = AgentState(self.status)

    def to_dict(self) -> dict:
        """Serialize result to dict (for logging/checkpointing)."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "verification_hash": self.verification_hash,
            "metadata": self.metadata,
            "error": self.error,
        }


class BaseAgent(ABC):
    """Abstract base class for all ResistanceMap agents.

    Defines the interface for specialized agents (DataValidationAgent,
    VAEPretrainAgent, etc.) and provides zero-trust verification utilities.

    Attributes:
        name: Unique identifier for this agent
        dependencies: Names of agents this agent depends on
    """

    def __init__(self, name: str, dependencies: list[str] | None = None):
        """Initialize agent.

        Args:
            name: Unique agent identifier
            dependencies: List of agent names this agent depends on (for DAG ordering)
        """
        self.name = name
        self.dependencies = dependencies or []

    @abstractmethod
    async def execute(self, inputs: dict[str, Any], config: Any) -> AgentResult:
        """Execute the agent's core logic.

        This method must be implemented by all subclasses. It should:
        1. Validate inputs via verify_inputs()
        2. Perform the actual work
        3. Compute output hash via compute_hash()
        4. Return AgentResult with verification_hash set

        Args:
            inputs: Input dictionary (typically outputs from dependent agents)
            config: ResistanceMapConfig object

        Returns:
            AgentResult with status, output, and verification_hash
        """
        pass

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        """Verify input data quality and integrity.

        Zero-trust principle: All inputs must be explicitly validated.
        Subclasses should override to add domain-specific checks.

        Args:
            inputs: Dictionary of inputs to verify

        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(inputs, dict):
            return False, "inputs must be a dictionary"
        return True, ""

    def verify_output(
        self, result: AgentResult, expected_hash: str | None = None
    ) -> tuple[bool, str]:
        """Verify output integrity via hash comparison.

        Args:
            result: AgentResult to verify
            expected_hash: Optional expected hash to compare against

        Returns:
            (is_valid, error_message) tuple
        """
        computed_hash = self.compute_hash(result.output)
        if computed_hash != result.verification_hash:
            return False, f"Hash mismatch: {computed_hash} != {result.verification_hash}"

        if expected_hash and expected_hash != computed_hash:
            return False, f"Expected hash {expected_hash}, got {computed_hash}"

        return True, ""

    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute SHA256 hash of data for zero-trust verification.

        Handles various data types:
        - Torch tensors: serialized to bytes
        - Dicts/lists: JSON serialized
        - Strings/primitives: encoded to UTF-8
        - Objects: pickle serialized

        Args:
            data: Data to hash

        Returns:
            SHA256 hex digest
        """
        h = hashlib.sha256()

        if isinstance(data, torch.Tensor):
            # Serialize tensor to bytes
            h.update(str(data.shape).encode())
            h.update(str(data.dtype).encode())
            h.update(data.detach().cpu().numpy().tobytes())

        elif isinstance(data, (dict, list)):
            # JSON serialize (preserves order)
            h.update(json.dumps(data, sort_keys=True, default=str).encode())

        elif isinstance(data, str):
            h.update(data.encode())

        elif isinstance(data, (int, float, bool)):
            h.update(str(data).encode())

        elif data is None:
            h.update(b"None")

        else:
            # Fallback: try string representation
            h.update(str(data).encode())

        return h.hexdigest()

    def _make_result(
        self,
        status: AgentState,
        output: Any | None = None,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> AgentResult:
        """Helper to create AgentResult with automatic hash computation.

        Args:
            status: Final agent state
            output: Result output (optional if status == FAILED)
            error: Error message if status == FAILED
            metadata: Additional metadata dict

        Returns:
            AgentResult with verification_hash automatically computed
        """
        verification_hash = self.compute_hash(output) if output is not None else ""
        return AgentResult(
            agent_name=self.name,
            status=status,
            output=output,
            verification_hash=verification_hash,
            metadata=metadata or {},
            error=error,
        )

    async def _safe_execute(
        self, inputs: dict[str, Any], config: Any
    ) -> AgentResult:
        """Wrapper around execute() that adds zero-trust checks and error handling.

        Subclasses should implement execute() directly and not call this.

        Args:
            inputs: Input dictionary
            config: ResistanceMapConfig

        Returns:
            AgentResult (status=COMPLETED or FAILED)
        """
        try:
            # Verify inputs
            is_valid, error_msg = self.verify_inputs(inputs)
            if not is_valid:
                logger.error(f"{self.name}: Input validation failed: {error_msg}")
                return self._make_result(
                    AgentState.FAILED,
                    error=f"Input validation failed: {error_msg}",
                )

            # Execute
            logger.info(f"{self.name}: Starting execution")
            result = await self.execute(inputs, config)

            # Verify output if not already failed
            if result.status == AgentState.COMPLETED:
                is_valid, error_msg = self.verify_output(result)
                if not is_valid:
                    logger.error(f"{self.name}: Output verification failed: {error_msg}")
                    result.status = AgentState.FAILED
                    result.error = error_msg

            logger.info(f"{self.name}: Execution complete (status={result.status.value})")
            return result

        except Exception as e:
            logger.exception(f"{self.name}: Unexpected error during execution")
            return self._make_result(
                AgentState.FAILED,
                error=f"Unexpected error: {str(e)}",
            )
