"""
Latent compute scheduler for ResistanceMap.

Manages scheduling and execution of expensive background operations
with priority queues, resource budgets, and dependency tracking.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any, Callable, List, Tuple, Set
import heapq

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ComputeJob:
    """
    Represents a single compute job.

    Jobs are scheduled in a priority queue and executed respecting
    resource constraints and dependencies.
    """

    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation: str = ""
    priority: int = 5  # Lower = higher priority
    estimated_flops: float = 0.0
    estimated_memory_gb: float = 0.0
    status: JobStatus = field(default=JobStatus.QUEUED)
    result: Any = None
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    # Execution details
    fn: Optional[Callable] = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # job IDs this depends on

    def __lt__(self, other: "ComputeJob") -> bool:
        """Enable sorting in priority queue (lower priority value = higher priority)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.submitted_at < other.submitted_at

    def duration_seconds(self) -> Optional[float]:
        """Get job execution duration."""
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def wait_time_seconds(self) -> float:
        """Get time from submission to start."""
        if self.started_at is None:
            return (datetime.utcnow() - self.submitted_at).total_seconds()
        return (self.started_at - self.submitted_at).total_seconds()


class LatentComputeScheduler:
    """
    Schedules heavy compute operations with priority queues and resource budgets.

    Operations are executed asynchronously respecting:
    - Job priority (lower value = higher priority)
    - GPU memory budget
    - Maximum concurrent jobs
    - Job dependencies
    """

    def __init__(
        self,
        gpu_memory_budget_gb: float = 24.0,
        max_concurrent_jobs: int = 2,
        max_queue_size: int = 1000,
    ):
        """
        Initialize scheduler.

        Args:
            gpu_memory_budget_gb: Available GPU memory in GB
            max_concurrent_jobs: Maximum concurrent jobs
            max_queue_size: Maximum job queue size
        """
        self.gpu_memory_budget_gb = gpu_memory_budget_gb
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queue_size = max_queue_size

        self._job_queue: List[ComputeJob] = []
        self._running_jobs: Dict[str, ComputeJob] = {}
        self._completed_jobs: Dict[str, ComputeJob] = {}
        self._failed_jobs: Dict[str, ComputeJob] = {}
        self._all_jobs: Dict[str, ComputeJob] = {}

        self._gpu_memory_used_gb = 0.0
        self._scheduler_running = False
        self._scheduler_task: Optional[asyncio.Task] = None

        self.logger = logger

    def submit(
        self,
        operation: str,
        fn: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        priority: int = 5,
        estimated_flops: float = 0.0,
        estimated_memory_gb: float = 0.0,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """
        Submit a job to the scheduler.

        Args:
            operation: Operation name (e.g., "esm2_embedding")
            fn: Callable to execute
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Priority level (lower = higher priority)
            estimated_flops: Estimated FLOPs for the operation
            estimated_memory_gb: Estimated GPU memory in GB
            dependencies: List of job IDs this depends on

        Returns:
            Job ID
        """
        if len(self._all_jobs) >= self.max_queue_size:
            raise RuntimeError(f"Job queue full (max: {self.max_queue_size})")

        job = ComputeJob(
            operation=operation,
            fn=fn,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            estimated_flops=estimated_flops,
            estimated_memory_gb=estimated_memory_gb,
            dependencies=dependencies or [],
        )

        self._all_jobs[job.job_id] = job
        heapq.heappush(self._job_queue, job)

        self.logger.info(
            f"Job submitted: {job.job_id} ({operation}, priority={priority}, "
            f"mem={estimated_memory_gb:.1f}GB)"
        )

        return job.job_id

    async def start(self) -> None:
        """Start the scheduler main loop."""
        if self._scheduler_running:
            self.logger.warning("Scheduler already running")
            return

        self._scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        self.logger.info("Scheduler started")

    async def stop(self, wait_for_completion: bool = False) -> None:
        """
        Stop the scheduler.

        Args:
            wait_for_completion: If True, wait for running jobs to complete
        """
        self._scheduler_running = False

        if wait_for_completion:
            while self._running_jobs:
                await asyncio.sleep(0.1)

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Scheduler stopped")

    async def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                # Check for completed jobs
                await self._check_completed_jobs()

                # Submit new jobs if resources available
                await self._submit_available_jobs()

                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1.0)

    async def _check_completed_jobs(self) -> None:
        """Check for completed jobs and update status."""
        running_job_ids = list(self._running_jobs.keys())

        for job_id in running_job_ids:
            job = self._running_jobs[job_id]
            # Placeholder: would check actual task status in production
            # For now, we just track manually completed jobs
            pass

    async def _submit_available_jobs(self) -> None:
        """
        Submit next available job if resources allow.

        Respects:
        - Max concurrent jobs
        - GPU memory budget
        - Job dependencies
        """
        while len(self._running_jobs) < self.max_concurrent_jobs and self._job_queue:
            # Peek at next job (without removing)
            if not self._job_queue:
                break

            next_job = self._job_queue[0]

            # Check if dependencies are satisfied
            if not self._are_dependencies_satisfied(next_job):
                break

            # Check if memory available
            if (
                self._gpu_memory_used_gb + next_job.estimated_memory_gb
                > self.gpu_memory_budget_gb
            ):
                break

            # Remove from queue and start execution
            heapq.heappop(self._job_queue)
            await self._execute_job(next_job)

    def _are_dependencies_satisfied(self, job: ComputeJob) -> bool:
        """Check if all dependencies of a job are satisfied."""
        for dep_id in job.dependencies:
            if dep_id not in self._completed_jobs:
                return False
        return True

    async def _execute_job(self, job: ComputeJob) -> None:
        """
        Execute a job.

        Args:
            job: Job to execute
        """
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        self._running_jobs[job.job_id] = job

        self.logger.info(f"Job starting: {job.job_id} ({job.operation})")

        try:
            # Update GPU memory
            self._gpu_memory_used_gb += job.estimated_memory_gb

            # Execute function
            if asyncio.iscoroutinefunction(job.fn):
                result = await job.fn(*job.args, **job.kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: job.fn(*job.args, **job.kwargs),
                )

            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            self._completed_jobs[job.job_id] = job
            del self._running_jobs[job.job_id]

            self.logger.info(
                f"Job completed: {job.job_id} ({job.operation}) "
                f"in {job.duration_seconds():.2f}s"
            )

        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()

            self._failed_jobs[job.job_id] = job
            del self._running_jobs[job.job_id]

            self.logger.error(f"Job failed: {job.job_id} - {e}")

        finally:
            self._gpu_memory_used_gb -= job.estimated_memory_gb

    def get_job_status(self, job_id: str) -> Optional[ComputeJob]:
        """Get job status and metadata."""
        return self._all_jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job if it hasn't started.

        Returns True if cancelled, False if already running/completed.
        """
        if job_id not in self._all_jobs:
            return False

        job = self._all_jobs[job_id]

        if job.status == JobStatus.RUNNING:
            self.logger.warning(f"Cannot cancel running job: {job_id}")
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            self.logger.warning(f"Cannot cancel {job.status.value} job: {job_id}")
            return False

        # Remove from queue
        self._job_queue = [j for j in self._job_queue if j.job_id != job_id]
        heapq.heapify(self._job_queue)

        job.status = JobStatus.CANCELLED
        self.logger.info(f"Job cancelled: {job_id}")

        return True

    def estimate_completion_time(self, job_id: str) -> Optional[float]:
        """
        Estimate time until job completion in seconds.

        Returns None if job is already completed or not found.
        """
        job = self._all_jobs.get(job_id)
        if job is None:
            return None

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return 0.0

        if job.status == JobStatus.RUNNING and job.started_at is not None:
            # Very rough estimate based on position in queue
            # In production, would use historical data
            return 60.0  # Placeholder

        if job.status == JobStatus.QUEUED:
            position = sum(
                1 for j in self._job_queue
                if j.priority <= job.priority and j.submitted_at <= job.submitted_at
            )
            # Average 30 seconds per job
            return float(position * 30)

        return None

    # Pre-configured latent compute operations

    def schedule_esm2_embedding(
        self,
        sequences: List[str],
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        cache_dir: str = "./models",
        priority: int = 3,
    ) -> str:
        """
        Schedule ESM-2 protein embedding computation.

        Args:
            sequences: List of protein sequences
            model_name: HuggingFace model identifier
            cache_dir: Cache directory for model
            priority: Job priority (lower = higher priority)

        Returns:
            Job ID
        """
        n_sequences = len(sequences)
        # Rough estimates: ~1 TFLOP per sequence, ~12 GB peak
        estimated_flops = float(n_sequences * 1e12)
        estimated_memory = min(20.0, 4.0 + n_sequences * 0.002)

        async def _embed_esm2(seqs: List[str], model: str, cache: str) -> Dict:
            """Compute ESM-2 embeddings."""
            # Placeholder: would load model and compute embeddings
            return {
                "n_sequences": len(seqs),
                "embedding_dim": 650,
                "embeddings": None,  # Actual embeddings in production
            }

        return self.submit(
            operation="esm2_embedding",
            fn=_embed_esm2,
            args=(sequences, model_name, cache_dir),
            priority=priority,
            estimated_flops=estimated_flops,
            estimated_memory_gb=estimated_memory,
        )

    def schedule_ppi_subgraph_extraction(
        self,
        protein_list: List[str],
        ppi_network_path: str = "./data/string_ppi_v12.pkl",
        priority: int = 4,
    ) -> str:
        """
        Schedule PPI subgraph extraction for protein list.

        Args:
            protein_list: List of protein names
            ppi_network_path: Path to STRING PPI network
            priority: Job priority

        Returns:
            Job ID
        """
        n_proteins = len(protein_list)
        # Rough estimates: ~100 GFLOP for subgraph extraction
        estimated_flops = float(n_proteins * 100e9)
        estimated_memory = min(8.0, 2.0 + n_proteins * 0.001)

        async def _extract_subgraph(proteins: List[str], ppi_path: str) -> Dict:
            """Extract PPI subgraph."""
            # Placeholder
            return {
                "n_proteins": len(proteins),
                "n_edges": 0,
                "subgraph": None,
            }

        return self.submit(
            operation="ppi_subgraph_extraction",
            fn=_extract_subgraph,
            args=(protein_list, ppi_network_path),
            priority=priority,
            estimated_flops=estimated_flops,
            estimated_memory_gb=estimated_memory,
        )

    def schedule_trajectory_ensemble(
        self,
        initial_states: Any,  # torch.Tensor in production
        n_samples: int = 1000,
        n_steps: int = 100,
        priority: int = 6,
    ) -> str:
        """
        Schedule trajectory ensemble generation for latent space.

        Args:
            initial_states: Initial state tensors (batch_size, latent_dim)
            n_samples: Number of trajectory samples
            n_steps: Number of steps per trajectory
            priority: Job priority

        Returns:
            Job ID
        """
        # Rough estimates for 1000 samples x 100 steps in latent space
        estimated_flops = float(n_samples * n_steps * 1e8)
        estimated_memory = min(16.0, 4.0 + (n_samples * n_steps * 0.0001))

        async def _generate_trajectories(
            states: Any,
            n_samp: int,
            n_st: int,
        ) -> Dict:
            """Generate ensemble trajectories."""
            # Placeholder
            return {
                "n_trajectories": n_samp,
                "n_steps": n_st,
                "trajectories": None,
            }

        return self.submit(
            operation="trajectory_ensemble",
            fn=_generate_trajectories,
            args=(initial_states, n_samples, n_steps),
            priority=priority,
            estimated_flops=estimated_flops,
            estimated_memory_gb=estimated_memory,
        )

    # Utility methods

    def queue_depth(self) -> int:
        """Get current job queue depth."""
        return len(self._job_queue)

    def running_count(self) -> int:
        """Get number of running jobs."""
        return len(self._running_jobs)

    def completed_count(self) -> int:
        """Get number of completed jobs."""
        return len(self._completed_jobs)

    def failed_count(self) -> int:
        """Get number of failed jobs."""
        return len(self._failed_jobs)

    def gpu_memory_utilization(self) -> float:
        """Get GPU memory utilization percentage."""
        return (self._gpu_memory_used_gb / self.gpu_memory_budget_gb) * 100

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics."""
        return {
            "queue_depth": self.queue_depth(),
            "running_jobs": self.running_count(),
            "completed_jobs": self.completed_count(),
            "failed_jobs": self.failed_count(),
            "total_jobs": len(self._all_jobs),
            "gpu_memory_used_gb": self._gpu_memory_used_gb,
            "gpu_memory_budget_gb": self.gpu_memory_budget_gb,
            "gpu_utilization_percent": self.gpu_memory_utilization(),
            "scheduler_running": self._scheduler_running,
        }

    def get_running_jobs_summary(self) -> List[Dict[str, Any]]:
        """Get summary of currently running jobs."""
        return [
            {
                "job_id": job.job_id,
                "operation": job.operation,
                "status": job.status.value,
                "memory_gb": job.estimated_memory_gb,
                "elapsed_seconds": (datetime.utcnow() - job.started_at).total_seconds()
                if job.started_at
                else None,
            }
            for job in self._running_jobs.values()
        ]

    def get_queue_preview(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get preview of next jobs in queue."""
        result = []
        for i, job in enumerate(sorted(self._job_queue)[:limit]):
            result.append(
                {
                    "position": i + 1,
                    "job_id": job.job_id,
                    "operation": job.operation,
                    "priority": job.priority,
                    "memory_gb": job.estimated_memory_gb,
                    "dependencies": len(job.dependencies),
                }
            )
        return result
