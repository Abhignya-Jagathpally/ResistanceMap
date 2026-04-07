"""Example usage of the ResistanceMap multi-agent orchestrator.

Demonstrates:
1. Registering agents with the orchestrator
2. Preparing execution plan (DAG topological sort)
3. Running agents with parallel execution
4. Collecting results and verification chain
5. Generating timing and summary reports

Run with:
    python -m resistancemap.agents.example_usage
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from resistancemap.agents import (
    Orchestrator,
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
from resistancemap.config import ResistanceMapConfig, load_config
from resistancemap.utils.logging_utils import setup_logger, log_summary

logger = logging.getLogger("resistancemap")


async def main():
    """Run the complete ResistanceMap pipeline with orchestrator."""

    # Setup logging
    setup_logger(log_dir=Path("logs"), wandb_project=None)
    logger.info("=" * 70)
    logger.info("ResistanceMap Multi-Agent Orchestrator")
    logger.info("=" * 70)

    # Load configuration
    config = load_config(Path("config.yaml"))
    logger.info(f"Configuration loaded from config.yaml")

    # Initialize orchestrator
    orchestrator = Orchestrator()
    logger.info("Orchestrator initialized")

    # Register all agents
    agents = [
        DataValidationAgent(),
        DataPrepAgent(),
        VAEPretrainAgent(),
        VAEFinetuneAgent(),
        ESM2EmbedAgent(),
        TrajectoryAgent(),
        ProteinNetAgent(),
        FusionAgent(),
        LandscapeAgent(),
        ValidationAgent(),
    ]

    for agent in agents:
        orchestrator.add_agent(agent)
        logger.info(f"Registered agent: {agent.name}")

    # Prepare execution plan
    logger.info("\nPreparing execution plan...")
    is_valid, error_msg = orchestrator.prepare_execution()

    if not is_valid:
        logger.error(f"Failed to prepare execution: {error_msg}")
        return False

    # Display execution plan
    plan = orchestrator.get_execution_plan()
    logger.info(f"\nExecution plan ({len(plan)} layers):")
    for layer_idx, layer in enumerate(plan):
        logger.info(f"  Layer {layer_idx}: {', '.join(layer)}")

    # Run orchestrator
    logger.info("\nStarting parallel execution...")
    try:
        results = await orchestrator.run(config)
    except Exception as e:
        logger.exception("Orchestrator execution failed")
        return False

    # Collect results
    summary = orchestrator.get_results_summary()
    logger.info(f"\nExecution summary:")
    logger.info(f"  Total agents: {summary['total_agents']}")
    logger.info(f"  Completed: {summary['status_counts'].get('completed', 0)}")
    logger.info(f"  Failed: {summary['status_counts'].get('failed', 0)}")

    if summary.get("failed_agents"):
        logger.error(f"  Failed agents: {', '.join(summary['failed_agents'])}")

    # Get verification chain
    verification_chain = orchestrator.get_verification_chain()
    logger.info(f"\nVerification chain ({len(verification_chain)} hashes):")
    for agent_name, hash_value in zip(
        orchestrator.execution_plan.agent_order, verification_chain
    ):
        logger.info(f"  {agent_name}: {hash_value[:16]}...")

    # Get timing report
    timing_report = orchestrator.get_timing_report()
    logger.info(f"\nTiming report:")
    total_time = sum(t["elapsed_seconds"] for t in timing_report.values())
    for agent_name, timing in sorted(
        timing_report.items(), key=lambda x: x[1]["elapsed_seconds"], reverse=True
    )[:5]:
        logger.info(f"  {agent_name}: {timing['elapsed_seconds']:.3f}s")
    logger.info(f"  Total execution time: {total_time:.1f}s")

    # Print detailed results
    logger.info("\nDetailed agent results:")
    for agent_name in orchestrator.execution_plan.agent_order:
        result = results[agent_name]
        status_emoji = "✓" if result.status.value == "completed" else "✗"
        logger.info(
            f"{status_emoji} {agent_name}: {result.status.value} | "
            f"hash={result.verification_hash[:8]}..."
        )
        if result.metadata:
            for key, value in result.metadata.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key}: {value}")

    # Save results to file
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save verification chain
    with open(checkpoint_dir / "verification_chain.json", "w") as f:
        json.dump(
            {
                "agents": orchestrator.execution_plan.agent_order,
                "hashes": verification_chain,
                "timestamp": str(Path("checkpoints").stat().st_mtime),
            },
            f,
            indent=2,
        )
    logger.info(f"\nVerification chain saved to {checkpoint_dir / 'verification_chain.json'}")

    # Save execution plan
    with open(checkpoint_dir / "execution_plan.json", "w") as f:
        json.dump(
            {
                "layers": plan,
                "total_agents": len(orchestrator.dag.agents),
                "agent_dependencies": {
                    agent.name: agent.dependencies for agent in agents
                },
            },
            f,
            indent=2,
        )
    logger.info(f"Execution plan saved to {checkpoint_dir / 'execution_plan.json'}")

    # Save results summary
    with open(checkpoint_dir / "results_summary.json", "w") as f:
        json.dump(
            {
                "summary": summary,
                "timing_report": timing_report,
                "agent_results": {
                    name: result.to_dict() for name, result in results.items()
                },
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Results summary saved to {checkpoint_dir / 'results_summary.json'}")

    logger.info("\n" + "=" * 70)
    logger.info("Orchestrator execution complete!")
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
