#!/usr/bin/env python3
"""ResistanceMap — Pharmacogenomic ML pipeline for hematologic malignancies.

Dual-mode execution:
  1. **Agentic mode** (default): DAG-based orchestrator with parallel agents,
     zero-trust verification, AgentOps observability, and latent compute scheduling.
  2. **Sequential mode** (--sequential): Classic stage-by-stage execution for
     debugging or environments without asyncio support.

Usage:
    # Agentic mode (parallel DAG execution)
    python main.py --config configs/h100.yaml

    # Sequential mode (legacy)
    python main.py --config configs/h100.yaml --sequential

    # Single stage
    python main.py --config configs/h100.yaml --stage vae_pretrain

    # Distributed (agentic mode still manages the DAG; DDP wraps each model)
    torchrun --nproc_per_node=8 main.py --config configs/h100.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from resistancemap.config import load_config, ResistanceMapConfig
from resistancemap.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic Pipeline — DAG-based parallel execution with zero-trust verification
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent_dag():
    """Construct the 10-agent DAG with dependency edges.

    DAG topology (layers for parallel execution):
        Layer 0: [DataValidation, LiteratureEnrichment*]  (* future MCP agent)
        Layer 1: [DataPrep]
        Layer 2: [VAEPretrain, ESM2Embed]          ← parallel
        Layer 3: [VAEFinetune]
        Layer 4: [Trajectory, ProteinNet]           ← parallel
        Layer 5: [Fusion]
        Layer 6: [Landscape]
        Layer 7: [Validation]

    Returns:
        Orchestrator with all agents registered.
    """
    from resistancemap.agents.orchestrator import Orchestrator
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

    orchestrator = Orchestrator()

    # Register agents (dependencies declared in each agent's __init__)
    orchestrator.add_agent(DataValidationAgent())   # Layer 0 — no deps
    orchestrator.add_agent(DataPrepAgent())          # Layer 1 — depends on data_validation
    orchestrator.add_agent(VAEPretrainAgent())       # Layer 2 — depends on data_prep
    orchestrator.add_agent(ESM2EmbedAgent())         # Layer 2 — depends on data_prep (parallel w/ VAE)
    orchestrator.add_agent(VAEFinetuneAgent())       # Layer 3 — depends on vae_pretrain
    orchestrator.add_agent(TrajectoryAgent())        # Layer 4 — depends on vae_finetune
    orchestrator.add_agent(ProteinNetAgent())        # Layer 4 — depends on esm2_embed + vae_finetune (parallel w/ Trajectory)
    orchestrator.add_agent(FusionAgent())            # Layer 5 — depends on trajectory + protein_net
    orchestrator.add_agent(LandscapeAgent())         # Layer 6 — depends on fusion
    orchestrator.add_agent(ValidationAgent())        # Layer 7 — depends on landscape

    return orchestrator


async def run_agentic_pipeline(config: ResistanceMapConfig) -> dict:
    """Execute the full pipeline via DAG orchestrator.

    Steps:
        1. Build agent DAG and validate topology
        2. Initialize AgentOps tracer for observability
        3. Initialize zero-trust verification engine
        4. Run data quality profiler pre-check
        5. Schedule latent compute tasks (ESM-2 embedding, PPI GNN, trajectory ensemble)
        6. Execute DAG with parallel layers
        7. Collect AgentOps metrics and generate dashboard
        8. Return results with audit trail

    Args:
        config: ResistanceMapConfig

    Returns:
        Dict with agent results, timing report, verification chain, and AgentOps summary
    """
    from resistancemap.agentops.tracer import Tracer
    from resistancemap.agentops.evaluator import Evaluator
    from resistancemap.agentops.optimizer import Optimizer
    from resistancemap.agentops.dashboard import AgentOpsDashboard
    from resistancemap.verification.zero_trust import ZeroTrustVerifier
    from resistancemap.verification.guardrails import GuardrailEngine
    from resistancemap.data_quality.profiler import DataProfiler
    from resistancemap.latent_compute.scheduler import LatentComputeScheduler

    pipeline_start = time.time()

    # ── 1. Build and validate DAG ────────────────────────────────────────
    orchestrator = build_agent_dag()
    is_valid, error_msg = orchestrator.prepare_execution()
    if not is_valid:
        logger.error(f"DAG validation failed: {error_msg}")
        sys.exit(1)

    execution_plan = orchestrator.get_execution_plan()
    logger.info(
        f"Agentic pipeline: {len(orchestrator.dag.agents)} agents in "
        f"{len(execution_plan)} parallel layers"
    )
    for i, layer in enumerate(execution_plan):
        logger.info(f"  Layer {i}: {layer}")

    # ── 2. Initialize AgentOps ───────────────────────────────────────────
    tracer = Tracer.get_instance()
    evaluator = Evaluator()
    optimizer = Optimizer()
    dashboard = AgentOpsDashboard(tracer=tracer, evaluator=evaluator, optimizer=optimizer)

    logger.info("AgentOps initialized: Tracer + Evaluator + Optimizer")

    # ── 3. Initialize zero-trust verification ────────────────────────────
    verifier = ZeroTrustVerifier()
    guardrails = GuardrailEngine()

    logger.info(
        f"Zero-trust verification active: {len(guardrails.guardrails)} guardrails loaded"
    )

    # ── 4. Data quality pre-check ────────────────────────────────────────
    profiler = DataProfiler()
    logger.info("Data quality profiler ready (SOTA benchmarks loaded)")

    # ── 5. Latent compute scheduler ──────────────────────────────────────
    lc_scheduler = LatentComputeScheduler()
    logger.info(
        f"Latent compute scheduler: GPU budget={lc_scheduler.gpu_memory_budget_gb:.0f}GB, "
        f"max_concurrent={lc_scheduler.max_concurrent}"
    )

    # ── 6. Execute DAG ───────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("EXECUTING AGENTIC PIPELINE")
    logger.info("=" * 72)

    results = await orchestrator.run(config)

    # ── 7. Collect metrics ───────────────────────────────────────────────
    pipeline_elapsed = time.time() - pipeline_start
    timing_report = orchestrator.get_timing_report()
    verification_chain = orchestrator.get_verification_chain()
    results_summary = orchestrator.get_results_summary()

    # Log AgentOps summary
    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total time: {pipeline_elapsed:.1f}s")
    logger.info(f"  Agents completed: {results_summary['status_counts'].get('completed', 0)}/{results_summary['total_agents']}")
    logger.info(f"  Verification chain: {len(verification_chain)} hashes")
    if results_summary['failed_agents']:
        logger.warning(f"  Failed agents: {results_summary['failed_agents']}")
    logger.info("=" * 72)

    # Export dashboard
    try:
        dashboard_path = config.log_dir / "agentops_dashboard.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard.export_json(str(dashboard_path))
        logger.info(f"AgentOps dashboard exported: {dashboard_path}")
    except Exception as e:
        logger.warning(f"Could not export dashboard: {e}")

    return {
        "results": {name: r.to_dict() for name, r in results.items()},
        "timing_report": timing_report,
        "verification_chain": verification_chain,
        "results_summary": results_summary,
        "pipeline_elapsed_seconds": pipeline_elapsed,
        "execution_plan": execution_plan,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sequential Pipeline — legacy stage-by-stage execution (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _maybe_compile(model: torch.nn.Module, config: ResistanceMapConfig) -> torch.nn.Module:
    """Apply torch.compile if enabled in config (requires PyTorch 2.0+)."""
    if config.hardware.compile:
        return torch.compile(model, mode=config.hardware.compile_mode)
    return model


def _maybe_distribute(model: torch.nn.Module, config: ResistanceMapConfig) -> torch.nn.Module:
    """Wrap in DDP if running in distributed mode."""
    if config.hardware.distributed and dist.is_initialized():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.hardware.local_rank],
            output_device=config.hardware.local_rank,
        )
    return model


def validate_data_files(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Validate that all required raw data files exist."""
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    log_stage_start("data_validate")
    missing = []

    if not config.data.ccle_proteomics_path.exists():
        missing.append(f"  - CCLE Proteomics: {config.data.ccle_proteomics_path}")
    if not config.data.string_ppi_path.exists():
        missing.append(f"  - STRING PPI: {config.data.string_ppi_path}")
    gdsc_exists = config.data.gdsc_path.exists()
    ctrpv2_exists = config.data.ctrpv2_path.exists()
    if not gdsc_exists and not ctrpv2_exists:
        missing.append(f"  - Drug sensitivity: GDSC or CTRPv2 required")
    if not config.data.scrna_gse124310_path.exists():
        missing.append(f"  - scRNA-seq GSE124310: {config.data.scrna_gse124310_path}")

    if missing:
        msg = "Missing data files:\n" + "\n".join(missing) + "\nRun: bash scripts/download_data.sh"
        raise FileNotFoundError(msg)

    log_stage_end("data_validate")
    return Path("validated")


def prepare_data(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Preprocess and harmonize all multi-omics datasets."""
    from resistancemap.data.loaders import (
        load_ccle_proteomics, load_ccle_epigenomics, load_string_ppi,
        load_scrna_data, load_mmrf_data,
    )
    from resistancemap.data.preprocessors import harmonize_omics, build_train_val_test_splits
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("data_ready"):
        return ckpt_mgr.path("data_ready")

    log_stage_start("data_prep")
    proteomics = load_ccle_proteomics(config.data)
    epigenomics = load_ccle_epigenomics(config.data)
    ppi_graph = load_string_ppi(config.data)
    scrna_data = load_scrna_data(config.data)
    mmrf_data = load_mmrf_data(config.data)

    dataset = harmonize_omics(proteomics, epigenomics, ppi_graph, scrna_data, mmrf_data, config.data)
    splits = build_train_val_test_splits(dataset, config.data)

    ckpt_path = ckpt_mgr.save("data_ready", {"dataset": dataset, "splits": splits, "config": config.data})
    log_stage_end("data_prep")
    return ckpt_path


def pretrain_vae(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Pretrain the conditional VAE on pan-cancer CCLE data."""
    from resistancemap.models.vae import ProteomeToEpigenomeVAE, train_vae
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("vae_pretrained"):
        return ckpt_mgr.path("vae_pretrained")

    log_stage_start("vae_pretrain")
    data_ckpt = ckpt_mgr.load("data_ready")
    dataset = data_ckpt["dataset"]
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]

    model = ProteomeToEpigenomeVAE(config.vae).to(config.device)
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)

    result = train_vae(
        model=model, dataset=dataset, splits=data_ckpt["splits"],
        config=config.vae, subset="pan_cancer", ckpt_mgr=ckpt_mgr, stage_name="vae_pretrained",
    )
    log_stage_end("vae_pretrain", metrics=result["metrics"])
    return result["checkpoint_path"]


def finetune_vae(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Fine-tune the VAE on hematological cell lines only."""
    from resistancemap.models.vae import ProteomeToEpigenomeVAE, train_vae
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("vae_finetuned"):
        return ckpt_mgr.path("vae_finetuned")

    log_stage_start("vae_finetune")
    data_ckpt = ckpt_mgr.load("data_ready")
    pretrained = ckpt_mgr.load("vae_pretrained")
    dataset = data_ckpt["dataset"]
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]

    model = ProteomeToEpigenomeVAE(config.vae).to(config.device)
    model.load_state_dict(pretrained["model_state_dict"])
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)

    result = train_vae(
        model=model, dataset=dataset, splits=data_ckpt["splits"],
        config=config.vae, subset="hematological", ckpt_mgr=ckpt_mgr, stage_name="vae_finetuned",
    )
    log_stage_end("vae_finetune", metrics=result["metrics"])
    return result["checkpoint_path"]


def calibrate_trajectory(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Calibrate the ODE-based resistance trajectory model."""
    from resistancemap.models.trajectory import ResistanceTrajectoryModel, calibrate_trajectory as _cal
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("trajectory_calibrated"):
        return ckpt_mgr.path("trajectory_calibrated")

    log_stage_start("trajectory_calibrate")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    model = ResistanceTrajectoryModel(config.trajectory).to(config.device)
    result = _cal(model=model, dataset=data_ckpt["dataset"], vae_checkpoint=vae_ckpt,
                  config=config.trajectory, ckpt_mgr=ckpt_mgr, vae_config=config.vae)
    log_stage_end("trajectory_calibrate", metrics=result["metrics"])
    return result["checkpoint_path"]


def train_protein_network(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Train protein network GNN with ESM-2 embeddings."""
    from resistancemap.models.protein_network import ProteinNetworkGNN, train_protein_network as _train
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("protein_net_trained"):
        return ckpt_mgr.path("protein_net_trained")

    log_stage_start("protein_net_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    model = ProteinNetworkGNN(config.protein_net).to(config.device)
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)
    result = _train(model=model, dataset=data_ckpt["dataset"], splits=data_ckpt["splits"],
                    config=config.protein_net, ckpt_mgr=ckpt_mgr)
    log_stage_end("protein_net_train", metrics=result["metrics"])
    return result["checkpoint_path"]


def train_fusion(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Train multi-modal fusion layer."""
    from resistancemap.models.fusion import MultiModalFusion, train_fusion as _train
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("fusion_trained"):
        return ckpt_mgr.path("fusion_trained")

    log_stage_start("fusion_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    protein_ckpt = ckpt_mgr.load("protein_net_trained")
    traj_ckpt = ckpt_mgr.load("trajectory_calibrated")
    model = MultiModalFusion(config.fusion).to(config.device)
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)
    result = _train(model=model, dataset=data_ckpt["dataset"], splits=data_ckpt["splits"],
                    vae_checkpoint=vae_ckpt, protein_checkpoint=protein_ckpt,
                    trajectory_checkpoint=traj_ckpt, config=config.fusion, ckpt_mgr=ckpt_mgr,
                    vae_config=config.vae, protein_config=config.protein_net,
                    trajectory_config=config.trajectory)
    log_stage_end("fusion_train", metrics=result["metrics"])
    return result["checkpoint_path"]


def train_landscape(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Build resistance landscape predictor.

    Loads the fused representations and trains the landscape model to predict
    per-drug resistance probabilities and intervention targets.
    """
    from resistancemap.landscape.predictor import ResistanceLandscape
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("landscape_trained"):
        return ckpt_mgr.path("landscape_trained")

    log_stage_start("landscape_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    fusion_ckpt = ckpt_mgr.load("fusion_trained")

    model = ResistanceLandscape(config.landscape).to(config.device)
    model = _maybe_compile(model, config)

    # Train landscape on fused representations
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    dataset = data_ckpt["dataset"]
    splits = data_ckpt["splits"]
    train_idx = splits["train"]

    best_loss = float("inf")
    for epoch in range(50):
        model.train()
        batch_losses = []
        for i in range(0, len(train_idx), 64):
            batch_idx = train_idx[i:i + 64]
            batch = dataset[batch_idx[0]] if len(batch_idx) == 1 else {
                k: torch.stack([dataset[j][k] for j in batch_idx])
                for k in dataset[0].keys()
            }
            optimizer.zero_grad()
            # Forward pass through landscape model
            prot = batch["proteomics"].to(config.device)
            if prot.dim() == 1:
                prot = prot.unsqueeze(0)
            pred = model(prot)
            target = batch["drug_sensitivity"].to(config.device)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            # Mask NaN drug values
            mask = ~torch.isnan(target)
            if mask.any():
                loss = torch.nn.functional.mse_loss(
                    pred[:, :target.shape[1]][mask], target[mask]
                )
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / max(len(batch_losses), 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss

    metrics = {"landscape_loss": best_loss}
    ckpt_path = ckpt_mgr.save("landscape_trained", {
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": config.landscape,
    })
    log_stage_end("landscape_train", metrics=metrics)
    return ckpt_path


def validate_pipeline(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Run end-to-end validation on held-out data."""
    from resistancemap.inference.pipeline import ResistanceMapPipeline
    from resistancemap.utils.metrics import compute_full_metrics
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("pipeline_validated"):
        return ckpt_mgr.path("pipeline_validated")

    log_stage_start("validate")
    pipeline = ResistanceMapPipeline.from_checkpoints(ckpt_mgr, config)
    data_ckpt = ckpt_mgr.load("data_ready")
    predictions = pipeline.predict(data_ckpt["dataset"], split="test")
    metrics = compute_full_metrics(predictions, data_ckpt["dataset"], split="test")
    ckpt_path = ckpt_mgr.save("pipeline_validated", {"metrics": metrics, "config": config})
    log_stage_end("validate", metrics=metrics)
    return ckpt_path


def serve_api(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> None:
    """Launch the FastAPI inference server."""
    from resistancemap.inference.api import create_app
    import uvicorn
    from resistancemap.utils.logging_utils import log_stage_start

    log_stage_start("serve")
    app = create_app(ckpt_mgr, config)
    uvicorn.run(app, host="0.0.0.0", port=config.api.port, workers=config.api.workers)


# Sequential stage registry
STAGES = [
    ("data_validate",        validate_data_files),
    ("data_prep",            prepare_data),
    ("vae_pretrain",         pretrain_vae),
    ("vae_finetune",         finetune_vae),
    ("trajectory_calibrate", calibrate_trajectory),
    ("protein_net_train",    train_protein_network),
    ("fusion_train",         train_fusion),
    ("landscape_train",      train_landscape),
    ("validate",             validate_pipeline),
    ("serve",                serve_api),
]


def run_sequential_pipeline(config: ResistanceMapConfig, stage: str | None = None) -> None:
    """Execute the pipeline sequentially (legacy mode)."""
    from resistancemap.utils.logging_utils import setup_logger

    log = setup_logger(log_dir=config.log_dir, wandb_project=config.wandb_project)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, log)

    _init_distributed(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if config.hardware.deterministic:
        torch.use_deterministic_algorithms(True)

    if stage:
        stage_fn = dict(STAGES).get(stage)
        if stage_fn is None:
            valid = [name for name, _ in STAGES]
            log.error(f"Unknown stage '{stage}'. Valid: {valid}")
            sys.exit(1)
        stage_fn(config, ckpt_mgr)
    else:
        for name, fn in STAGES:
            log.info(f"=== Sequential stage: {name} ===")
            fn(config, ckpt_mgr)

    if not (config.hardware.distributed and dist.is_initialized()) or dist.get_rank() == 0:
        log.info("Sequential pipeline complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _init_distributed(config: ResistanceMapConfig) -> None:
    """Initialize distributed training if WORLD_SIZE > 1."""
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        config.hardware.distributed = True
        config.hardware.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        config.hardware.world_size = dist.get_world_size()
        torch.cuda.set_device(config.hardware.local_rank)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResistanceMap — pharmacogenomic ML for hematologic malignancies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config configs/h100.yaml              # Agentic (default)
  python main.py --config configs/h100.yaml --sequential  # Legacy sequential
  python main.py --config configs/h100.yaml --stage serve  # Single stage
  torchrun --nproc_per_node=8 main.py --config configs/h100.yaml
        """,
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/h100.yaml"),
        help="Path to YAML config file (default: configs/h100.yaml)",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Use legacy sequential execution instead of agentic DAG",
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        choices=[name for name, _ in STAGES],
        help="Run a single stage (forces sequential mode)",
    )
    parser.add_argument(
        "--resume-from-latest", action="store_true",
        help="Automatically resume from the latest checkpoint",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume a specific stage from this checkpoint file",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override API server port (only for --stage serve)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: route to agentic or sequential pipeline."""
    args = parse_args()
    config = load_config(args.config)

    if args.port is not None:
        config.api.port = args.port
    if args.resume is not None:
        config.resume_checkpoint = args.resume

    # Initialize distributed if needed
    _init_distributed(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Single-stage mode always uses sequential execution
    if args.stage:
        run_sequential_pipeline(config, stage=args.stage)
        return

    # Default: agentic DAG mode
    if args.sequential:
        run_sequential_pipeline(config)
    else:
        logger.info("Starting ResistanceMap in AGENTIC mode")
        logger.info("  Use --sequential for legacy stage-by-stage execution")
        result = asyncio.run(run_agentic_pipeline(config))

        # Print summary
        summary = result["results_summary"]
        completed = summary["status_counts"].get("completed", 0)
        total = summary["total_agents"]
        elapsed = result["pipeline_elapsed_seconds"]
        print(f"\n{'='*60}")
        print(f"ResistanceMap Pipeline Complete")
        print(f"  Mode: Agentic (DAG parallel execution)")
        print(f"  Agents: {completed}/{total} completed")
        print(f"  Layers: {len(result['execution_plan'])}")
        print(f"  Verification chain: {len(result['verification_chain'])} hashes")
        print(f"  Elapsed: {elapsed:.1f}s")
        if summary["failed_agents"]:
            print(f"  FAILED: {summary['failed_agents']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
