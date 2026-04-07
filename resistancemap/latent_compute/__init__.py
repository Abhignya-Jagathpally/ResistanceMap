"""
Latent compute scheduling module for ResistanceMap.

This module handles scheduling and execution of computationally expensive
operations as background jobs, including:
- ESM-2 protein embeddings (7853 proteins)
- PPI subgraph extraction and GNN training
- Trajectory ensemble generation
- Multi-task model training

Operations are scheduled with priority queues, resource constraints,
and dependency tracking.
"""

from resistancemap.latent_compute.scheduler import (
    ComputeJob,
    JobStatus,
    LatentComputeScheduler,
)

__all__ = [
    "ComputeJob",
    "JobStatus",
    "LatentComputeScheduler",
]
