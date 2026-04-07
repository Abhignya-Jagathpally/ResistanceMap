"""Structured logging utilities for ResistanceMap pipeline.

Directly reused from MyeloMemory's logging.py. Provides:
    - Consistent console + file logging
    - Optional Weights & Biases integration
    - Stage timing for performance monitoring
    - Structured metric logging

File named logging_utils.py to avoid collision with stdlib logging module.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

# Module-level state for stage timing
_stage_start_times: dict[str, float] = {}

# Optional W&B
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def setup_logger(
    log_dir: str | Path | None = None,
    wandb_project: str | None = None,
    config: dict[str, Any] | None = None,
) -> logging.Logger:
    """Configure the root logger for the ResistanceMap pipeline.

    Sets up console handler with structured formatting and optionally
    initializes Weights & Biases for experiment tracking.

    Args:
        log_dir: Directory for log files (default: current directory).
        wandb_project: W&B project name (optional).
        config: Optional config dict to log to W&B.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("resistancemap")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_dir / 'pipeline.log'}")

    # W&B initialization
    if HAS_WANDB and wandb_project:
        try:
            wandb_config = config or {}
            wandb.init(
                project=wandb_project,
                config={
                    "landscape_hidden_dim": wandb_config.get("hidden_dim", 256),
                    "landscape_n_drugs": wandb_config.get("n_drugs", 6),
                    "landscape_n_proteins": wandb_config.get("n_proteins", 100),
                },
                reinit=True,
            )
            logger.info(f"W&B initialized: project={wandb_project}")
        except Exception as e:
            logger.warning(f"W&B init failed (continuing without): {e}")

    return logger


def log_stage_start(stage_name: str) -> None:
    """Log the start of a pipeline stage and begin timing.

    Args:
        stage_name: Name of the pipeline stage (e.g., 'landscape_train').
    """
    logger = logging.getLogger("resistancemap")
    _stage_start_times[stage_name] = time.time()
    logger.info(f">>> STAGE START: {stage_name}")

    if HAS_WANDB and wandb.run is not None:
        wandb.log({f"stage/{stage_name}/started": 1})


def log_stage_end(stage_name: str, metrics: dict[str, Any] | None = None) -> None:
    """Log the completion of a pipeline stage with timing and metrics.

    Args:
        stage_name: Name of the pipeline stage.
        metrics: Optional dict of metrics to log.
    """
    logger = logging.getLogger("resistancemap")

    elapsed = time.time() - _stage_start_times.get(stage_name, time.time())
    minutes = elapsed / 60

    metrics_str = ""
    if metrics:
        metrics_str = " | " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )

    logger.info(f"<<< STAGE END: {stage_name} ({minutes:.1f} min){metrics_str}")

    if HAS_WANDB and wandb.run is not None:
        log_data = {
            f"stage/{stage_name}/elapsed_minutes": minutes,
            f"stage/{stage_name}/completed": 1,
        }
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"stage/{stage_name}/{k}"] = v
        wandb.log(log_data)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to both console and W&B.

    Args:
        metrics: Dict of metric name → value.
        step: Optional global step for x-axis alignment.
    """
    logger = logging.getLogger("resistancemap")
    metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f"Metrics: {metrics_str}")

    if HAS_WANDB and wandb.run is not None:
        wandb.log(metrics, step=step)


def log_summary(title: str, summary_dict: dict[str, Any]) -> None:
    """Log a summary section with key results.

    Args:
        title: Summary title.
        summary_dict: Dict of key → value pairs.
    """
    logger = logging.getLogger("resistancemap")
    logger.info(f"\n{'='*60}")
    logger.info(f"  {title}")
    logger.info(f"{'='*60}")
    for key, value in summary_dict.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info(f"{'='*60}\n")
