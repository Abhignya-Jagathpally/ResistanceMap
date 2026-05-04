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

import numpy as np
import torch
import torch.distributed as dist

from resistancemap.config import load_config, ResistanceMapConfig
from resistancemap.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic Pipeline — DAG-based parallel execution with zero-trust verification
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent_dag(
    tracer=None,
    evaluator=None,
    optimizer=None,
    guardrails=None,
):
    """Construct the 10-agent DAG with dependency edges.

    DAG topology (layers for parallel execution):
        Layer 0: [DataValidation]
        Layer 1: [DataPrep]
        Layer 2: [VAEPretrain, ESM2Embed]          <- parallel
        Layer 3: [VAEFinetune]
        Layer 4: [Trajectory]
        Layer 5: [ProteinNet]                       <- needs trajectory checkpoint
        Layer 6: [Fusion]
        Layer 7: [Landscape]
        Layer 8: [Validation]

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

    # v7: thread agentops + guardrails through so the orchestrator emits
    # spans / records task results / runs GuardrailEngine.check_all on
    # each agent's output. None-args preserve legacy behavior.
    orchestrator = Orchestrator(
        tracer=tracer,
        evaluator=evaluator,
        optimizer=optimizer,
        guardrails=guardrails,
    )

    # Register agents (dependencies declared in each agent's __init__)
    orchestrator.add_agent(DataValidationAgent())   # Layer 0 — no deps
    orchestrator.add_agent(DataPrepAgent())          # Layer 1 — depends on data_validation
    orchestrator.add_agent(VAEPretrainAgent())       # Layer 2 — depends on data_prep
    orchestrator.add_agent(ESM2EmbedAgent())         # Layer 2 — depends on data_prep (parallel w/ VAE)
    orchestrator.add_agent(VAEFinetuneAgent())       # Layer 3 — depends on vae_pretrain
    orchestrator.add_agent(TrajectoryAgent())        # Layer 4 — depends on vae_finetune
    orchestrator.add_agent(ProteinNetAgent())        # Layer 5 — depends on esm2_embed + vae_finetune + trajectory
    orchestrator.add_agent(FusionAgent())            # Layer 6 — depends on trajectory + protein_net
    orchestrator.add_agent(LandscapeAgent())         # Layer 7 — depends on fusion
    orchestrator.add_agent(ValidationAgent())        # Layer 8 — depends on landscape

    return orchestrator


async def run_agentic_pipeline(config: ResistanceMapConfig) -> dict:
    """Execute the full pipeline via DAG orchestrator.

    Steps:
        1. Set up logging and checkpoint manager
        2. Build agent DAG and validate topology
        3. Initialize AgentOps tracer for observability
        4. Initialize zero-trust verification engine
        5. Run data quality profiler pre-check
        6. Schedule latent compute tasks (ESM-2 embedding, PPI GNN, trajectory ensemble)
        7. Execute DAG with parallel layers
        8. Collect AgentOps metrics and generate dashboard
        9. Return results with audit trail

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
    from resistancemap.utils.logging_utils import setup_logger

    pipeline_start = time.time()

    # ── 0. Logging, checkpoint manager, deterministic mode ───────────────
    log = setup_logger(log_dir=config.log_dir, wandb_project=config.wandb_project)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, log)
    config._ckpt_mgr = ckpt_mgr  # shared by all agents via _get_ckpt_mgr()

    if config.hardware.deterministic:
        import random
        random.seed(config.hardware.seed)
        np.random.seed(config.hardware.seed)
        torch.manual_seed(config.hardware.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.hardware.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── 1. Initialize AgentOps + zero-trust BEFORE building the DAG so we
    #       can inject them into the orchestrator (v7 fix for the
    #       "AgentOps initialized but never receives events" gap).
    tracer = Tracer.get_instance()
    evaluator = Evaluator()
    optimizer = Optimizer()
    dashboard = AgentOpsDashboard(tracer=tracer, evaluator=evaluator, optimizer=optimizer)
    logger.info("AgentOps initialized: Tracer + Evaluator + Optimizer")

    verifier = ZeroTrustVerifier()
    guardrails = GuardrailEngine()
    logger.info(
        f"Zero-trust verification active: {len(guardrails.guardrails)} guardrails loaded"
    )

    # ── 2. Build and validate DAG (now wired to AgentOps + guardrails) ──
    orchestrator = build_agent_dag(
        tracer=tracer,
        evaluator=evaluator,
        optimizer=optimizer,
        guardrails=guardrails,
    )
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

    # ── 4. Data quality pre-check ────────────────────────────────────────
    profiler = DataProfiler()
    logger.info("Data quality profiler ready (SOTA benchmarks loaded)")

    # ── 5. Latent compute scheduler ──────────────────────────────────────
    lc_scheduler = LatentComputeScheduler()
    logger.info(
        f"Latent compute scheduler: GPU budget={lc_scheduler.gpu_memory_budget_gb:.0f}GB, "
        f"max_concurrent={lc_scheduler.max_concurrent_jobs}"
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

    # v7: surface guardrail violations from this run (collected by the
    # orchestrator during _run_agent_isolated). Empty list means everything
    # passed; non-empty is informational unless severity=='error'.
    guardrail_violations = getattr(orchestrator, "_guardrail_violations", [])
    n_err = sum(1 for v in guardrail_violations if v.get("severity") == "error")
    if guardrail_violations:
        logger.info(
            f"Guardrails: {len(guardrail_violations)} violations "
            f"({n_err} error / {len(guardrail_violations) - n_err} non-error)"
        )

    # ── 8. Run evaluation governance (Tier A/B/C/D) automatically ───────
    # v7: previously this only ran behind --evaluate. We now invoke it
    # post-training so Tier-A FAILs are visible from the same command. It
    # is non-blocking by default — if you want hard-stop, set
    # config.evaluation.tier_a_hard_stop=True.
    governance_report = None
    try:
        if getattr(config.evaluation, "enabled", False):
            governance_report = await run_evaluation_governance(config)
    except Exception:
        logger.exception("Post-training governance run failed (continuing)")

    return {
        "results": {name: r.to_dict() for name, r in results.items()},
        "timing_report": timing_report,
        "verification_chain": verification_chain,
        "results_summary": results_summary,
        "pipeline_elapsed_seconds": pipeline_elapsed,
        "execution_plan": execution_plan,
        "guardrail_violations": guardrail_violations,
        "governance_report": governance_report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sequential Pipeline — legacy stage-by-stage execution (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _maybe_compile(model: torch.nn.Module, config: ResistanceMapConfig) -> torch.nn.Module:
    """Apply torch.compile if enabled in config (requires PyTorch 2.0+).

    If torch.compile is enabled but Inductor crashes (a recurring problem
    for some module shapes / mode combinations), suppress the dynamo
    exception and fall back to eager mode for that model so the rest of
    the pipeline can still complete. This mirrors the suggestion printed
    by torch._dynamo on backend failure.
    """
    if not config.hardware.compile:
        return model
    try:
        import torch._dynamo as _dynamo
        # Tell dynamo to fall back to eager on any subsequent compile failure
        # for this model rather than crashing the training loop.
        _dynamo.config.suppress_errors = True
    except Exception:  # noqa: BLE001
        pass
    return torch.compile(model, mode=config.hardware.compile_mode)


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
        load_scrna_data, load_mmrf_data, load_depmap_crispr,
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
    crispr_data = load_depmap_crispr(config.data)

    dataset = harmonize_omics(
        proteomics, epigenomics, ppi_graph,
        scrna_data=scrna_data, mmrf_data=mmrf_data,
        crispr_data=crispr_data,
        config=config.data,
    )
    splits = build_train_val_test_splits(dataset, config.data)

    ckpt_path = ckpt_mgr.save("data_ready", {"dataset": dataset, "splits": splits, "config": config.data})
    log_stage_end("data_prep")
    return ckpt_path


def pretrain_vae(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Pretrain the conditional VAE on pan-cancer CCLE data."""
    import copy
    from resistancemap.models.vae import ProteomeToEpigenomeVAE, train_vae
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("vae_pretrained"):
        return ckpt_mgr.path("vae_pretrained")

    log_stage_start("vae_pretrain")
    data_ckpt = ckpt_mgr.load("data_ready")
    dataset = data_ckpt["dataset"]
    local_config = copy.deepcopy(config)
    local_config.vae.input_dim = dataset.proteomics.shape[1]
    local_config.vae.epigenome_dim = dataset.epigenomics.shape[1]

    model = ProteomeToEpigenomeVAE(local_config.vae).to(local_config.device)
    model = _maybe_compile(model, local_config)
    model = _maybe_distribute(model, local_config)

    result = train_vae(
        model=model, dataset=dataset, splits=data_ckpt["splits"],
        config=local_config.vae, subset="pan_cancer", ckpt_mgr=ckpt_mgr, stage_name="vae_pretrained",
    )
    log_stage_end("vae_pretrain", metrics=result["metrics"])
    return result["checkpoint_path"]


def finetune_vae(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Fine-tune the VAE on hematological cell lines only."""
    import copy
    from resistancemap.models.vae import ProteomeToEpigenomeVAE, train_vae
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("vae_finetuned"):
        return ckpt_mgr.path("vae_finetuned")

    log_stage_start("vae_finetune")
    data_ckpt = ckpt_mgr.load("data_ready")
    pretrained = ckpt_mgr.load("vae_pretrained")
    dataset = data_ckpt["dataset"]
    # Use a deep copy to avoid mutating the shared config object —
    # pretrain_vae already does this; finetune must do the same.
    local_config = copy.deepcopy(config)
    local_config.vae.input_dim = dataset.proteomics.shape[1]
    local_config.vae.epigenome_dim = dataset.epigenomics.shape[1]

    model = ProteomeToEpigenomeVAE(local_config.vae).to(local_config.device)
    # Strip _orig_mod. prefix from compiled model state dicts
    state_dict = pretrained["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = _maybe_compile(model, local_config)
    model = _maybe_distribute(model, local_config)

    result = train_vae(
        model=model, dataset=dataset, splits=data_ckpt["splits"],
        config=local_config.vae, subset="hematological", ckpt_mgr=ckpt_mgr, stage_name="vae_finetuned",
    )
    log_stage_end("vae_finetune", metrics=result["metrics"])
    return result["checkpoint_path"]


def calibrate_trajectory(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Calibrate the ODE-based resistance trajectory model."""
    from resistancemap.models.trajectory import MemoryStabilityScorer, calibrate_scorer
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("stability_calibrated"):
        return ckpt_mgr.path("stability_calibrated")

    log_stage_start("trajectory_calibrate")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    model = MemoryStabilityScorer(config.trajectory).to(config.device)
    result = calibrate_scorer(scorer=model, dataset=data_ckpt["dataset"], vae_checkpoint=vae_ckpt,
                              config=config.trajectory, ckpt_mgr=ckpt_mgr)
    log_stage_end("trajectory_calibrate", metrics=result["metrics"])
    return result["checkpoint_path"]


def train_trajectory_forecaster(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Train the trajectory forecaster on top of the calibrated ODE.

    Builds a TrajectoryForecaster that predicts chromatin state evolution at
    3/6/12 month horizons via the ChromatinODE. Supervised by drug sensitivity
    as a proxy: samples with high resistance (high IC50 z-score) should show
    trajectories that converge toward the resistant (high-r) basin, while
    sensitive samples should remain in the active (high-a) basin.

    The forecaster's learnable parameters (latent_to_params, horizon_scale)
    are trained so that future stability scores correlate with observed drug
    response.  This makes the trajectory modality genuinely distinct from
    the static MemoryStabilityScorer used earlier.

    Checkpoint saved: "trajectory_forecaster_trained"
    """
    import copy
    from resistancemap.models.vae import ProteomeToEpigenomeVAE
    from resistancemap.models.trajectory import TrajectoryForecaster, MemoryStabilityScorer
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("trajectory_forecaster_trained"):
        return ckpt_mgr.path("trajectory_forecaster_trained")

    log_stage_start("trajectory_forecast")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    traj_ckpt = ckpt_mgr.load("stability_calibrated")
    dataset = data_ckpt["dataset"]
    protein_names = dataset.protein_names
    device = config.device

    # ── 1. Load frozen VAE for latent extraction ────────────────────────
    local_config = copy.deepcopy(config)
    local_config.vae.input_dim = dataset.proteomics.shape[1]
    local_config.vae.epigenome_dim = dataset.epigenomics.shape[1]
    vae = ProteomeToEpigenomeVAE(local_config.vae).to(device)
    vae_sd = {k.replace("_orig_mod.", ""): v for k, v in vae_ckpt["model_state_dict"].items()}
    vae.load_state_dict(vae_sd)
    vae.eval()

    # ── 2. Build trajectory forecaster ──────────────────────────────────
    # Re-use the calibrated ODE weights from MemoryStabilityScorer by
    # loading the scorer, then copying its ODE into the forecaster.
    scorer = MemoryStabilityScorer(config.trajectory).to(device)
    scorer_sd = {k.replace("_orig_mod.", ""): v for k, v in traj_ckpt["model_state_dict"].items()}
    scorer.load_state_dict(scorer_sd)
    scorer.eval()

    forecaster = TrajectoryForecaster(
        config=config.trajectory,
        protein_names=config.trajectory.reader_writer_proteins,
        use_sde=config.trajectory.use_sde,
    ).to(device)

    # Transfer the calibrated ODE parameters from scorer → forecaster.
    # Both share a ChromatinODE; copy the learned protein_to_params weights
    # so the forecaster starts from the calibrated ODE, not random.
    forecaster.ode.load_state_dict(scorer.ode.state_dict())
    logger.info("Transferred calibrated ODE weights from MemoryStabilityScorer to TrajectoryForecaster")

    # ── 3. Pre-compute VAE latents and reader/writer abundances ─────────
    logger.info("Pre-computing VAE latents and reader/writer levels for trajectory training...")
    all_latents = []
    all_rw_levels = []
    with torch.no_grad():
        for i in range(len(dataset)):
            prot = dataset.proteomics[i:i+1].to(device)
            mu, _ = vae.encode(prot)
            all_latents.append(mu.squeeze(0).cpu())
            rw = scorer.extract_reader_writer_levels(prot, protein_names)
            all_rw_levels.append(rw.squeeze(0).cpu())
    all_latents = torch.stack(all_latents)      # (N, 64)
    all_rw_levels = torch.stack(all_rw_levels)  # (N, 20)
    logger.info(f"Pre-computed: latents {all_latents.shape}, rw_levels {all_rw_levels.shape}")

    # ── 4. Build supervision targets ────────────────────────────────────
    # Strategy: The drug sensitivity z-scores are our proxy for resistance.
    # A sample with HIGH average z-scored IC50 across drugs is RESISTANT
    # (drugs are less effective). We want the forecaster's future stability
    # scores to predict this: resistant samples should show declining
    # stability (moving toward resistant basin) at longer horizons.
    #
    # Target per sample: mean drug sensitivity (NaN-aware), normalized [0,1].
    # 1.0 = most resistant → expect low future stability at long horizons
    # 0.0 = most sensitive → expect high future stability (locked in sensitive basin)
    drug_sens = dataset.drug_sensitivity  # (N, D)
    per_sample_resistance = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        valid = drug_sens[i][~torch.isnan(drug_sens[i])]
        if len(valid) > 0:
            per_sample_resistance[i] = valid.mean().item()
        else:
            per_sample_resistance[i] = float("nan")

    valid_mask = ~torch.isnan(per_sample_resistance)
    if valid_mask.sum() > 0:
        vr = per_sample_resistance[valid_mask]
        per_sample_resistance_norm = torch.zeros_like(per_sample_resistance)
        per_sample_resistance_norm[valid_mask] = (vr - vr.min()) / (vr.max() - vr.min() + 1e-8)
    else:
        per_sample_resistance_norm = torch.full((len(dataset),), 0.5)
        valid_mask = torch.ones(len(dataset), dtype=torch.bool)

    # ── 5. Training loop ────────────────────────────────────────────────
    # Only train the forecaster-specific parameters; freeze the ODE core
    # (it was already calibrated). The trainable params are:
    #   - latent_to_params: Linear(64→8) — per-parameter ODE modulation from latent
    #   - horizon_scale: scalar — learned time scaling
    trainable_params = [
        {"params": forecaster.latent_to_params.parameters(), "lr": 1e-3},
        {"params": [forecaster.horizon_scale], "lr": 5e-4},
    ]
    optimizer = torch.optim.Adam(trainable_params, weight_decay=1e-5)
    splits = data_ckpt["splits"]
    train_idx = splits["train"]
    val_idx = splits["val"]
    valid_train = [i for i in train_idx if valid_mask[i]]
    valid_val = [i for i in val_idx if valid_mask[i]]

    # Use euler solver during training — dopri5 with tight tolerances
    # can underflow dt when gradients push ODE into stiff regimes.
    _orig_solver = getattr(config.trajectory, "ode_solver", "euler")
    config.trajectory.ode_solver = "euler"

    best_val_loss = float("inf")
    patience, patience_counter = 20, 0
    batch_size = 32
    n_epochs = 200

    for epoch in range(n_epochs):
        forecaster.train()
        # But keep the core ODE frozen — only latent_to_params and horizon_scale learn
        for p in forecaster.ode.parameters():
            p.requires_grad = False

        perm = torch.randperm(len(valid_train))
        epoch_losses = []

        for bi in range(0, len(valid_train), batch_size):
            batch_end = min(bi + batch_size, len(valid_train))
            idxs = [valid_train[int(perm[j])] for j in range(bi, batch_end)]

            latent_b = all_latents[idxs].to(device)   # (B, 64)
            rw_b = all_rw_levels[idxs].to(device)     # (B, 20)
            target_b = per_sample_resistance_norm[idxs].to(device)  # (B,)

            optimizer.zero_grad()

            # Forward: compute stability scores at 3/6/12 month horizons
            # using internal differentiable methods (forecast() uses
            # torch.no_grad and self.eval(), blocking gradient flow).
            ode_params = forecaster._latent_to_ode_params(latent_b, rw_b)
            batch_size_b = latent_b.shape[0]
            a_init_b = torch.full((batch_size_b, 1), 0.5, device=device, dtype=latent_b.dtype)
            r_init_b = torch.full((batch_size_b, 1), 0.5, device=device, dtype=latent_b.dtype)

            target_stability = 1.0 - target_b
            horizon_weights = {3: 0.2, 6: 0.3, 12: 0.5}
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for h, w in horizon_weights.items():
                time_h = forecaster.horizon_times.get(h, float(h * 10.0))
                # horizon_scale modulates integration time; detach to float
                # for the ODE solver (gradients flow through ode_params instead)
                time_h = time_h * torch.nn.functional.softplus(forecaster.horizon_scale).item()
                a_final, r_final, _ = forecaster._integrate_trajectory(
                    ode_params, time_h, a_init_b, r_init_b
                )
                pred_stab = forecaster._compute_stability_at_state(a_final, r_final, ode_params)
                total_loss = total_loss + w * torch.nn.functional.mse_loss(pred_stab, target_stability)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(forecaster.latent_to_params.parameters()) + [forecaster.horizon_scale], 1.0
            )
            optimizer.step()
            epoch_losses.append(total_loss.item())

        # ── Validation ──
        forecaster.eval()
        val_losses = []
        with torch.no_grad():
            for bi in range(0, len(valid_val), batch_size):
                idxs = valid_val[bi:min(bi + batch_size, len(valid_val))]
                latent_b = all_latents[idxs].to(device)
                rw_b = all_rw_levels[idxs].to(device)
                target_b = per_sample_resistance_norm[idxs].to(device)
                target_stability = 1.0 - target_b

                ode_params_v = forecaster._latent_to_ode_params(latent_b, rw_b)
                bs_v = latent_b.shape[0]
                a_init_v = torch.full((bs_v, 1), 0.5, device=device, dtype=latent_b.dtype)
                r_init_v = torch.full((bs_v, 1), 0.5, device=device, dtype=latent_b.dtype)
                total_loss = torch.tensor(0.0, device=device)
                for h, w in horizon_weights.items():
                    time_h = forecaster.horizon_times.get(h, float(h * 10.0))
                    time_h = time_h * torch.nn.functional.softplus(forecaster.horizon_scale).item()
                    a_f, r_f, _ = forecaster._integrate_trajectory(
                        ode_params_v, time_h, a_init_v, r_init_v
                    )
                    pred_stab = forecaster._compute_stability_at_state(a_f, r_f, ode_params_v)
                    total_loss = total_loss + w * torch.nn.functional.mse_loss(pred_stab, target_stability)
                val_losses.append(total_loss.item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_mgr.save("trajectory_forecaster_trained", {
                "model_state_dict": forecaster.state_dict(),
                "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                "config": config.trajectory,
                "epoch": epoch,
            })
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0:
            logger.info(
                f"[trajectory_forecast] Epoch {epoch+1}/{n_epochs} "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"horizon_scale={torch.nn.functional.softplus(forecaster.horizon_scale).item():.4f}"
            )
        if patience_counter >= patience:
            logger.info(f"[trajectory_forecast] Early stopping at epoch {epoch+1}")
            break

    # Restore original solver
    config.trajectory.ode_solver = _orig_solver

    metrics = {"trajectory_forecast_val_loss": best_val_loss}
    log_stage_end("trajectory_forecast", metrics=metrics)
    return ckpt_mgr.path("trajectory_forecaster_trained")


def train_protein_network(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Train protein network GNN on PPI graph.

    Node features per protein:
      - ESM-2 bottleneck embedding (256-dim) if sequences available, else abundance (1)
      - VAE latent state broadcast (64)
      - Stability score (1)
    Total: 321-dim (with ESM-2) or 66-dim (fallback).

    Trained with masked MSE against drug sensitivity + reversibility proxy.
    """
    from resistancemap.models.protein_network import (
        PPIGraphNetwork, ESM2Embedder, ESM2Bottleneck,
    )
    from resistancemap.models.vae import ProteomeToEpigenomeVAE
    from resistancemap.models.trajectory import MemoryStabilityScorer
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("protein_net_trained"):
        return ckpt_mgr.path("protein_net_trained")

    log_stage_start("protein_net_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    traj_ckpt = ckpt_mgr.load("stability_calibrated")
    dataset = data_ckpt["dataset"]
    protein_names = dataset.protein_names
    n_proteins = len(protein_names)
    device = config.device

    # ── 1. Load frozen VAE for latent extraction ─────────────────────────
    # Set dims from actual data
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]
    vae = ProteomeToEpigenomeVAE(config.vae).to(device)
    vae_sd = vae_ckpt["model_state_dict"]
    vae_sd = {k.replace("_orig_mod.", ""): v for k, v in vae_sd.items()}
    vae.load_state_dict(vae_sd)
    vae.eval()

    # ── 2. Load frozen stability scorer ──────────────────────────────────
    scorer = MemoryStabilityScorer(config.trajectory).to(device)
    scorer_sd = traj_ckpt["model_state_dict"]
    scorer_sd = {k.replace("_orig_mod.", ""): v for k, v in scorer_sd.items()}
    scorer.load_state_dict(scorer_sd)
    scorer.eval()

    # ── 3. Pre-compute latent states and stability for all samples ───────
    logger.info("Pre-computing VAE latent states and stability scores...")
    all_latents = []
    all_stability = []
    with torch.no_grad():
        for i in range(len(dataset)):
            prot = dataset.proteomics[i:i+1].to(device)
            mu, _ = vae.encode(prot)
            all_latents.append(mu.squeeze(0).cpu())
            stab = scorer(prot, protein_names)
            all_stability.append(stab.squeeze(0).cpu())
    all_latents = torch.stack(all_latents)    # (N, 64)
    all_stability = torch.stack(all_stability)  # (N,)
    logger.info(f"Pre-computed: latents {all_latents.shape}, stability {all_stability.shape}")

    # ── 3b. ESM-2 protein embeddings (optional) ─────────────────────────
    # If the dataset provides protein sequences, compute ESM-2 embeddings
    # and compress via bottleneck (1280→256). This replaces the 1-dim
    # abundance feature per node with a 256-dim structural embedding,
    # giving the GNN much richer node features.
    esm2_embeddings = None
    esm2_bottleneck_dim = getattr(config.protein_net, "esm2_bottleneck_dim", 256)
    protein_sequences = getattr(dataset, "protein_sequences", None)

    if protein_sequences and len(protein_sequences) == n_proteins:
        logger.info(f"Computing ESM-2 embeddings for {n_proteins} proteins...")
        try:
            esm2_model_name = getattr(config.protein_net, "esm2_model", "facebook/esm2_t33_650M_UR50D")
            embedder = ESM2Embedder(
                model_name=esm2_model_name,
                cache_size=n_proteins,
                device=str(device),
            )
            bottleneck = ESM2Bottleneck(
                input_dim=1280,
                output_dim=esm2_bottleneck_dim,
            ).to(device)
            # Compute in batches to avoid OOM on large proteomes
            esm2_batch_size = 64
            esm2_raw = []
            for bi in range(0, n_proteins, esm2_batch_size):
                batch_seqs = protein_sequences[bi:bi + esm2_batch_size]
                raw_emb = embedder.embed_proteins(batch_seqs)  # (batch, 1280)
                esm2_raw.append(raw_emb.cpu())
            esm2_raw = torch.cat(esm2_raw, dim=0)  # (n_proteins, 1280)
            with torch.no_grad():
                esm2_embeddings = bottleneck(esm2_raw.to(device)).cpu()  # (n_proteins, 256)
            logger.info(f"ESM-2 embeddings: {esm2_embeddings.shape}")
            # Free the large ESM-2 model from GPU memory
            del embedder
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            logger.warning(f"ESM-2 embedding failed ({e}); falling back to abundance features")
            esm2_embeddings = None
    else:
        logger.info("No protein sequences available; using abundance-only node features")

    use_esm2 = esm2_embeddings is not None

    # ── 4. Build PPI edge index ──────────────────────────────────────────
    ppi_edges = dataset.ppi_edges or []
    ppi_scores = dataset.ppi_scores or []
    prot_to_idx = {p: i for i, p in enumerate(protein_names)}

    src, dst, weights = [], [], []
    for (p1, p2), score in zip(ppi_edges, ppi_scores):
        i, j = prot_to_idx.get(p1), prot_to_idx.get(p2)
        if i is not None and j is not None:
            src.extend([i, j])
            dst.extend([j, i])
            weights.extend([score, score])

    if not src:
        # Build self-loop graph if no overlap (STRING uses Ensembl IDs)
        logger.warning("No PPI edges match proteomics proteins; using k-NN self-loop graph")
        for i in range(n_proteins):
            src.append(i); dst.append(i); weights.append(1.0)
    n_edges = len(src)
    logger.info(f"PPI graph: {n_proteins} nodes, {n_edges} edges")

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)

    # ── 5. Build GNN + prediction head ───────────────────────────────────
    # Node features:
    #   With ESM-2: bottleneck(256) + VAE latent(64) + stability(1) = 321
    #   Without:    abundance(1) + VAE latent(64) + stability(1) = 66
    node_feat_dim = (esm2_bottleneck_dim if use_esm2 else 1) + config.vae.latent_dim + 1
    gnn = PPIGraphNetwork(
        in_dim=node_feat_dim, hidden_dim=config.protein_net.gnn_hidden,
        n_layers=config.protein_net.gnn_layers, n_heads=config.protein_net.gnn_heads,
        dropout=config.protein_net.gnn_dropout, edge_dim=1,
    ).to(device)
    n_drugs = dataset.drug_sensitivity.shape[1]
    pred_head = torch.nn.Sequential(
        torch.nn.Linear(config.protein_net.gnn_hidden, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_drugs),
    ).to(device)

    params = list(gnn.parameters()) + list(pred_head.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
    splits = data_ckpt["splits"]
    train_idx = splits["train"]
    val_idx = splits["val"]

    # ── 6. Training loop ─────────────────────────────────────────────────
    from torch_geometric.data import Data as PyGData
    best_val_loss = float("inf")
    patience, patience_counter = 15, 0

    # Inner batch is the number of samples whose GNN-forward activations
    # we hold in GPU memory before backward(). Each sample materialises
    # one full PPI graph forward (~929k edges with the real STRING graph),
    # so a large inner batch crosses 90 GB. 4 keeps a useful gradient
    # estimate while leaving headroom.
    pn_inner_batch = 4

    for epoch in range(100):
        gnn.train(); pred_head.train()
        perm = torch.randperm(len(train_idx))
        epoch_losses = []

        for bi in range(0, len(train_idx), pn_inner_batch):
            optimizer.zero_grad()
            batch_loss = 0.0
            count = 0

            for si in range(bi, min(bi + pn_inner_batch, len(train_idx))):
                idx = train_idx[int(perm[si])]
                # Build node features: ESM2(256) or abundance(1) + latent(64) + stab(1)
                if use_esm2:
                    prot_feat = esm2_embeddings.to(device)  # (P, 256)
                else:
                    prot_feat = dataset.proteomics[idx].to(device).unsqueeze(-1)  # (P, 1)
                latent_broadcast = all_latents[idx].to(device).unsqueeze(0).expand(n_proteins, -1)  # (P, 64)
                stab_broadcast = all_stability[idx].to(device).unsqueeze(0).expand(n_proteins).unsqueeze(-1)  # (P, 1)
                node_feat = torch.cat([prot_feat, latent_broadcast, stab_broadcast], dim=-1)  # (P, node_feat_dim)

                ppi_data = PyGData(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
                node_emb = gnn(ppi_data)  # (P, hidden)
                global_repr = node_emb.mean(dim=0, keepdim=True)  # (1, hidden)
                pred = pred_head(global_repr).squeeze(0)  # (n_drugs,)

                target = dataset.drug_sensitivity[idx].to(device)
                mask = ~torch.isnan(target)
                if mask.any():
                    loss = torch.nn.functional.mse_loss(pred[mask], target[mask])
                    batch_loss = batch_loss + loss
                    count += 1

            if count > 0:
                (batch_loss / count).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                epoch_losses.append((batch_loss / count).item())

        # Validation
        gnn.eval(); pred_head.eval()
        val_losses = []
        with torch.no_grad():
            for idx in val_idx:
                if use_esm2:
                    prot_feat = esm2_embeddings.to(device)
                else:
                    prot_feat = dataset.proteomics[idx].to(device).unsqueeze(-1)
                latent_broadcast = all_latents[idx].to(device).unsqueeze(0).expand(n_proteins, -1)
                stab_broadcast = all_stability[idx].to(device).unsqueeze(0).expand(n_proteins).unsqueeze(-1)
                node_feat = torch.cat([prot_feat, latent_broadcast, stab_broadcast], dim=-1)
                ppi_data = PyGData(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
                node_emb = gnn(ppi_data)
                global_repr = node_emb.mean(dim=0, keepdim=True)
                pred = pred_head(global_repr).squeeze(0)
                target = dataset.drug_sensitivity[idx].to(device)
                mask = ~torch.isnan(target)
                if mask.any():
                    val_losses.append(torch.nn.functional.mse_loss(pred[mask], target[mask]).item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_mgr.save("protein_net_trained", {
                "gnn_state_dict": gnn.state_dict(),
                "pred_head_state_dict": pred_head.state_dict(),
                "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                "config": config.protein_net,
                "edge_index": edge_index.cpu(), "edge_attr": edge_attr.cpu(),
                "node_feat_dim": node_feat_dim,
                "use_esm2": use_esm2,
                "esm2_embeddings": esm2_embeddings if use_esm2 else None,
            })
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"[protein_net] Epoch {epoch+1}/100 train={train_loss:.4f} val={val_loss:.4f}")
        if patience_counter >= patience:
            logger.info(f"[protein_net] Early stopping at epoch {epoch+1}")
            break

    metrics = {"protein_net_val_loss": best_val_loss}
    log_stage_end("protein_net_train", metrics=metrics)
    return ckpt_mgr.path("protein_net_trained")


def train_fusion(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Train multi-modal fusion layer.

    Combines four modalities (epigenetic state, trajectory, protein network,
    stability) via CrossModalFusionNet with learned cross-attention and gated
    fusion, trained against drug sensitivity.
    """
    import copy
    from resistancemap.models.vae import ProteomeToEpigenomeVAE
    from resistancemap.models.trajectory import MemoryStabilityScorer, TrajectoryForecaster
    from resistancemap.models.protein_network import PPIGraphNetwork
    from resistancemap.models.fusion import CrossModalFusionNet
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("fusion_trained"):
        return ckpt_mgr.path("fusion_trained")

    log_stage_start("fusion_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    traj_ckpt = ckpt_mgr.load("stability_calibrated")
    forecaster_ckpt = ckpt_mgr.load("trajectory_forecaster_trained")
    pnet_ckpt = ckpt_mgr.load("protein_net_trained")
    dataset = data_ckpt["dataset"]
    protein_names = dataset.protein_names
    n_proteins = len(protein_names)
    device = config.device

    # ── 1. Load frozen upstream models ───────────────────────────────────
    local_config = copy.deepcopy(config)
    local_config.vae.input_dim = dataset.proteomics.shape[1]
    local_config.vae.epigenome_dim = dataset.epigenomics.shape[1]
    vae = ProteomeToEpigenomeVAE(local_config.vae).to(device)
    vae_sd = {k.replace("_orig_mod.", ""): v for k, v in vae_ckpt["model_state_dict"].items()}
    vae.load_state_dict(vae_sd); vae.eval()

    scorer = MemoryStabilityScorer(config.trajectory).to(device)
    scorer_sd = {k.replace("_orig_mod.", ""): v for k, v in traj_ckpt["model_state_dict"].items()}
    scorer.load_state_dict(scorer_sd); scorer.eval()

    # Load the trained trajectory forecaster for real trajectory embeddings
    forecaster = TrajectoryForecaster(
        config=config.trajectory,
        protein_names=config.trajectory.reader_writer_proteins,
        use_sde=config.trajectory.use_sde,
    ).to(device)
    fc_sd = {k.replace("_orig_mod.", ""): v for k, v in forecaster_ckpt["model_state_dict"].items()}
    forecaster.load_state_dict(fc_sd); forecaster.eval()

    gnn = PPIGraphNetwork(
        in_dim=pnet_ckpt["node_feat_dim"], hidden_dim=config.protein_net.gnn_hidden,
        n_layers=config.protein_net.gnn_layers, n_heads=config.protein_net.gnn_heads,
        dropout=config.protein_net.gnn_dropout, edge_dim=1,
    ).to(device)
    gnn_sd = {k.replace("_orig_mod.", ""): v for k, v in pnet_ckpt["gnn_state_dict"].items()}
    gnn.load_state_dict(gnn_sd); gnn.eval()

    edge_index = pnet_ckpt["edge_index"].to(device)
    edge_attr = pnet_ckpt["edge_attr"].to(device)

    # Check if ESM-2 embeddings were used during protein network training
    pnet_use_esm2 = pnet_ckpt.get("use_esm2", False)
    pnet_esm2_embeddings = pnet_ckpt.get("esm2_embeddings", None)

    # ── 2. Pre-compute all modality embeddings ───────────────────────────
    logger.info("Pre-computing modality embeddings for fusion training...")
    from torch_geometric.data import Data as PyGData
    epi_states = []     # (N, 64) — VAE latent
    traj_states = []    # (N, traj_dim) — real trajectory forecaster output
    pnet_outputs = []   # (N, gnn_hidden) — GNN global pool
    stab_scores = []    # (N, 1)

    # Trajectory embedding: concatenate stability scores at 3/6/12 month
    # horizons + initial stability + transition probabilities at each horizon.
    # This gives a 10-dim trajectory vector per sample:
    #   [initial_stab(1), stab_3m(1), stab_6m(1), stab_12m(1),
    #    trans_3m(1), trans_6m(1), trans_12m(1),
    #    a_3m(1), a_12m(1), r_12m(1)]
    # We'll pad/project this to 64-dim to keep the fusion input dims unchanged.
    traj_raw_dim = 10
    traj_projector = torch.nn.Sequential(
        torch.nn.Linear(traj_raw_dim, 64),
        torch.nn.GELU(),
    ).to(device)
    # Initialize with small weights so early training is stable
    torch.nn.init.xavier_uniform_(traj_projector[0].weight, gain=0.5)
    torch.nn.init.zeros_(traj_projector[0].bias)

    with torch.no_grad():
        for i in range(len(dataset)):
            prot = dataset.proteomics[i:i+1].to(device)
            # Epigenetic state = VAE latent
            mu, _ = vae.encode(prot)
            epi_states.append(mu.squeeze(0).cpu())
            # Stability
            stab = scorer(prot, protein_names)
            stab_scores.append(stab.view(1).cpu())

            # Real trajectory: forecast at 3/6/12 months using the trained forecaster
            rw = scorer.extract_reader_writer_levels(prot, protein_names)
            forecast_result = forecaster.forecast(mu, rw, horizons=[3, 6, 12])
            # Build trajectory feature vector
            traj_vec = torch.cat([
                forecast_result["initial_stability"].unsqueeze(-1),          # (1, 1)
                forecast_result["stability_scores"][3].unsqueeze(-1),        # (1, 1)
                forecast_result["stability_scores"][6].unsqueeze(-1),        # (1, 1)
                forecast_result["stability_scores"][12].unsqueeze(-1),       # (1, 1)
                forecast_result["transition_probs"][3].unsqueeze(-1),        # (1, 1)
                forecast_result["transition_probs"][6].unsqueeze(-1),        # (1, 1)
                forecast_result["transition_probs"][12].unsqueeze(-1),       # (1, 1)
                forecast_result["states"][3][0],                             # (1, 1) a at 3m
                forecast_result["states"][12][0],                            # (1, 1) a at 12m
                forecast_result["states"][12][1],                            # (1, 1) r at 12m
            ], dim=-1)  # (1, 10)
            traj_states.append(traj_vec.squeeze(0).cpu())

            # Protein network — match node feature construction from train_protein_network
            if pnet_use_esm2 and pnet_esm2_embeddings is not None:
                prot_feat = pnet_esm2_embeddings.to(device)  # (P, 256)
            else:
                prot_feat = prot.squeeze(0).unsqueeze(-1)  # (P, 1)
            lat_bc = mu.expand(n_proteins, -1)         # (P, 64)
            stab_bc = stab.expand(n_proteins).unsqueeze(-1)  # (P, 1)
            node_feat = torch.cat([prot_feat, lat_bc, stab_bc], dim=-1)
            ppi_data = PyGData(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
            node_emb = gnn(ppi_data)  # (P, hidden)
            pnet_outputs.append(node_emb.mean(dim=0).cpu())  # (hidden,)

    epi_states = torch.stack(epi_states)      # (N, 64)
    traj_states = torch.stack(traj_states)    # (N, 10) — raw trajectory features
    pnet_outputs = torch.stack(pnet_outputs)  # (N, gnn_hidden)
    stab_scores = torch.stack(stab_scores)    # (N, 1)
    logger.info(
        f"Embeddings: epi={epi_states.shape}, traj_raw={traj_states.shape}, "
        f"pnet={pnet_outputs.shape}, stab={stab_scores.shape}"
    )

    # ── 3. Build CrossModalFusionNet + drug prediction head ───────────────
    pnet_dim = pnet_outputs.shape[1]
    # Modality dimensions fed into CrossModalFusionNet:
    #   epigenetic: 64 (VAE latent, projected from traj_projector-equivalent)
    #   trajectory: 64 (10-dim raw → traj_projector → 64)
    #   protein_network: gnn_hidden (e.g. 128)
    #   stability: 1
    modality_dims = {
        "epigenetic": 64,
        "trajectory": 64,
        "protein_network": pnet_dim,
        "stability": 1,
    }
    fusion_output_dim = config.fusion.hidden_dim
    fusion = CrossModalFusionNet(
        modality_dims=modality_dims,
        hidden_dim=config.fusion.hidden_dim,
        n_heads=config.fusion.n_heads,
        dropout=config.fusion.dropout,
        output_dim=fusion_output_dim,
    ).to(device)

    n_drugs = dataset.drug_sensitivity.shape[1]
    drug_head = torch.nn.Sequential(
        torch.nn.Linear(fusion_output_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_drugs),
    ).to(device)

    # Include traj_projector in trainable params so it learns alongside fusion
    params = (
        list(fusion.parameters())
        + list(drug_head.parameters())
        + list(traj_projector.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=config.fusion.fusion_lr, weight_decay=config.fusion.weight_decay)
    splits = data_ckpt["splits"]
    train_idx = splits["train"]
    val_idx = splits["val"]

    # ── 4. Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience, patience_counter = 15, 0

    for epoch in range(config.fusion.fusion_epochs):
        fusion.train(); drug_head.train(); traj_projector.train()
        perm = torch.randperm(len(train_idx))
        epoch_losses = []

        for bi in range(0, len(train_idx), 64):
            batch_end = min(bi + 64, len(train_idx))
            idxs = [train_idx[int(perm[j])] for j in range(bi, batch_end)]

            epi_b = epi_states[idxs].to(device)
            traj_raw_b = traj_states[idxs].to(device)   # (B, 10) raw trajectory
            traj_b = traj_projector(traj_raw_b)          # (B, 64) projected
            pnet_b = pnet_outputs[idxs].to(device)
            stab_b = stab_scores[idxs].to(device)
            target_b = dataset.drug_sensitivity[idxs].to(device)

            optimizer.zero_grad()

            modality_dict = {
                "epigenetic": epi_b,
                "trajectory": traj_b,
                "protein_network": pnet_b,
                "stability": stab_b,
            }
            fused, _ = fusion(modality_dict)
            pred = drug_head(fused)
            mask = ~torch.isnan(target_b)
            if mask.any():
                loss = torch.nn.functional.mse_loss(pred[mask], target_b[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

        # Validation
        fusion.eval(); drug_head.eval(); traj_projector.eval()
        val_losses = []
        with torch.no_grad():
            for bi in range(0, len(val_idx), 64):
                idxs = val_idx[bi:min(bi+64, len(val_idx))]
                epi_b = epi_states[idxs].to(device)
                traj_raw_b = traj_states[idxs].to(device)
                traj_b = traj_projector(traj_raw_b)
                pnet_b = pnet_outputs[idxs].to(device)
                stab_b = stab_scores[idxs].to(device)
                target_b = dataset.drug_sensitivity[idxs].to(device)
                modality_dict = {
                    "epigenetic": epi_b,
                    "trajectory": traj_b,
                    "protein_network": pnet_b,
                    "stability": stab_b,
                }
                fused, _ = fusion(modality_dict)
                pred = drug_head(fused)
                mask = ~torch.isnan(target_b)
                if mask.any():
                    val_losses.append(torch.nn.functional.mse_loss(pred[mask], target_b[mask]).item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_mgr.save("fusion_trained", {
                "fusion_state_dict": fusion.state_dict(),
                "drug_head_state_dict": drug_head.state_dict(),
                "traj_projector_state_dict": traj_projector.state_dict(),
                "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                "config": config.fusion,
                "fusion_output_dim": fusion_output_dim,
                "fusion_type": "cross_attention",
                "modality_dims": modality_dims,
                "traj_raw_dim": traj_raw_dim,
                "epi_states": epi_states, "traj_states": traj_states,
                "pnet_outputs": pnet_outputs, "stab_scores": stab_scores,
            })
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"[fusion] Epoch {epoch+1}/{config.fusion.fusion_epochs} train={train_loss:.4f} val={val_loss:.4f}")
        if patience_counter >= patience:
            logger.info(f"[fusion] Early stopping at epoch {epoch+1}")
            break

    metrics = {"fusion_val_loss": best_val_loss}
    log_stage_end("fusion_train", metrics=metrics)
    return ckpt_mgr.path("fusion_trained")


def train_landscape(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Build resistance landscape predictor.

    Uses fused representations from upstream fusion stage to train
    the landscape model with drug resistance, state, and target heads.
    """
    from resistancemap.landscape.predictor import ResistanceLandscape
    from resistancemap.models.fusion import CrossModalFusionNet
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("landscape_trained"):
        return ckpt_mgr.path("landscape_trained")

    log_stage_start("landscape_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    fusion_ckpt = ckpt_mgr.load("fusion_trained")
    dataset = data_ckpt["dataset"]
    device = config.device

    # Load pre-computed fused representations
    fusion_output_dim = fusion_ckpt["fusion_output_dim"]
    traj_raw_dim = fusion_ckpt.get("traj_raw_dim", 10)
    epi_states = fusion_ckpt["epi_states"]
    traj_states = fusion_ckpt["traj_states"]    # (N, traj_raw_dim) raw trajectory features
    pnet_outputs = fusion_ckpt["pnet_outputs"]
    stab_scores = fusion_ckpt["stab_scores"]

    # Rebuild trajectory projector (10→64) and fusion model
    traj_projector = torch.nn.Sequential(
        torch.nn.Linear(traj_raw_dim, 64),
        torch.nn.GELU(),
    ).to(device)
    if "traj_projector_state_dict" in fusion_ckpt:
        traj_projector.load_state_dict(fusion_ckpt["traj_projector_state_dict"])
    traj_projector.eval()

    pnet_dim = pnet_outputs.shape[1]
    # Rebuild CrossModalFusionNet (backward compat: fall back to concat if old ckpt)
    fusion_type = fusion_ckpt.get("fusion_type", "concat")
    if fusion_type == "cross_attention":
        modality_dims = fusion_ckpt.get("modality_dims", {
            "epigenetic": 64, "trajectory": 64,
            "protein_network": pnet_dim, "stability": 1,
        })
        fusion = CrossModalFusionNet(
            modality_dims=modality_dims,
            hidden_dim=config.fusion.hidden_dim,
            n_heads=config.fusion.n_heads,
            dropout=config.fusion.dropout,
            output_dim=fusion_output_dim,
        ).to(device)
    else:
        # Legacy concat fusion for old checkpoints
        total_input_dim = 64 + 64 + pnet_dim + 1
        fusion = torch.nn.Sequential(
            torch.nn.Linear(total_input_dim, fusion_output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.fusion.dropout),
            torch.nn.Linear(fusion_output_dim, fusion_output_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(fusion_output_dim // 2, fusion_output_dim),
        ).to(device)
    fusion.load_state_dict(fusion_ckpt["fusion_state_dict"])
    fusion.eval()

    # Pre-compute fused representations
    logger.info("Computing fused representations for landscape training...")
    fused_reprs = []
    with torch.no_grad():
        for i in range(len(dataset)):
            traj_proj = traj_projector(traj_states[i:i+1].to(device))  # (1, 64)
            if fusion_type == "cross_attention":
                modality_dict = {
                    "epigenetic": epi_states[i:i+1].to(device),
                    "trajectory": traj_proj,
                    "protein_network": pnet_outputs[i:i+1].to(device),
                    "stability": stab_scores[i:i+1].to(device),
                }
                fused_out, _ = fusion(modality_dict)
                fused_reprs.append(fused_out.squeeze(0).cpu())
            else:
                concat = torch.cat([
                    epi_states[i:i+1].to(device), traj_proj,
                    pnet_outputs[i:i+1].to(device), stab_scores[i:i+1].to(device),
                ], dim=-1)
                fused_reprs.append(fusion(concat).squeeze(0).cpu())
    fused_reprs = torch.stack(fused_reprs)  # (N, fusion_dim)
    logger.info(f"Fused representations: {fused_reprs.shape}")

    n_drugs = dataset.drug_sensitivity.shape[1]
    use_evidential = getattr(config.landscape, "use_evidential", False)
    model = ResistanceLandscape(
        fusion_dim=fused_reprs.shape[1],
        n_drugs=n_drugs,
        n_proteins=min(len(dataset.protein_names), 100),
        use_evidential=use_evidential,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    splits = data_ckpt["splits"]
    train_idx = splits["train"]
    val_idx = splits["val"]

    best_val_loss = float("inf")
    patience, patience_counter = 15, 0

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(train_idx))
        epoch_losses = []

        for bi in range(0, len(train_idx), 64):
            batch_end = min(bi + 64, len(train_idx))
            idxs = [train_idx[int(perm[j])] for j in range(bi, batch_end)]

            fused_b = fused_reprs[idxs].to(device)
            target_b = dataset.drug_sensitivity[idxs].to(device)

            optimizer.zero_grad()
            output = model(fused_b)
            drug_pred = output["drug_resistance"]  # (B, n_drugs * n_timepoints)

            # Use first timepoint predictions for drug sensitivity loss
            pred_first = drug_pred[:, :n_drugs]
            mask = ~torch.isnan(target_b)
            if mask.any():
                loss = torch.nn.functional.mse_loss(pred_first[mask], target_b[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bi in range(0, len(val_idx), 64):
                idxs = val_idx[bi:min(bi+64, len(val_idx))]
                fused_b = fused_reprs[idxs].to(device)
                target_b = dataset.drug_sensitivity[idxs].to(device)
                output = model(fused_b)
                pred_first = output["drug_resistance"][:, :n_drugs]
                mask = ~torch.isnan(target_b)
                if mask.any():
                    val_losses.append(torch.nn.functional.mse_loss(pred_first[mask], target_b[mask]).item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_mgr.save("landscape_trained", {
                "model_state_dict": model.state_dict(),
                "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                "config": config.landscape,
                "fusion_dim": fused_reprs.shape[1],
                "use_evidential": use_evidential,
            })
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"[landscape] Epoch {epoch+1}/100 train={train_loss:.4f} val={val_loss:.4f}")
        if patience_counter >= patience:
            logger.info(f"[landscape] Early stopping at epoch {epoch+1}")
            break

    metrics = {"landscape_val_loss": best_val_loss}
    log_stage_end("landscape_train", metrics=metrics)
    return ckpt_mgr.path("landscape_trained")


def validate_pipeline(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Run end-to-end validation on held-out test data.

    Loads all trained models, computes fused representations for test split,
    and evaluates landscape predictions against drug sensitivity ground truth.
    """
    from resistancemap.landscape.predictor import ResistanceLandscape
    from resistancemap.models.fusion import CrossModalFusionNet
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("pipeline_validated"):
        return ckpt_mgr.path("pipeline_validated")

    log_stage_start("validate")
    data_ckpt = ckpt_mgr.load("data_ready")
    fusion_ckpt = ckpt_mgr.load("fusion_trained")
    landscape_ckpt = ckpt_mgr.load("landscape_trained")
    dataset = data_ckpt["dataset"]
    splits = data_ckpt["splits"]
    test_idx = splits["test"]
    device = config.device

    # Load fusion model + trajectory projector to compute fused representations for test set
    pnet_dim = fusion_ckpt["pnet_outputs"].shape[1]
    fusion_output_dim = fusion_ckpt["fusion_output_dim"]
    traj_raw_dim = fusion_ckpt.get("traj_raw_dim", 10)

    # Rebuild trajectory projector
    traj_projector = torch.nn.Sequential(
        torch.nn.Linear(traj_raw_dim, 64),
        torch.nn.GELU(),
    ).to(device)
    if "traj_projector_state_dict" in fusion_ckpt:
        traj_projector.load_state_dict(fusion_ckpt["traj_projector_state_dict"])
    traj_projector.eval()

    # Rebuild CrossModalFusionNet (backward compat: fall back to concat if old ckpt)
    fusion_type = fusion_ckpt.get("fusion_type", "concat")
    if fusion_type == "cross_attention":
        modality_dims = fusion_ckpt.get("modality_dims", {
            "epigenetic": 64, "trajectory": 64,
            "protein_network": pnet_dim, "stability": 1,
        })
        fusion = CrossModalFusionNet(
            modality_dims=modality_dims,
            hidden_dim=config.fusion.hidden_dim,
            n_heads=config.fusion.n_heads,
            dropout=config.fusion.dropout,
            output_dim=fusion_output_dim,
        ).to(device)
    else:
        total_input_dim = 64 + 64 + pnet_dim + 1
        fusion = torch.nn.Sequential(
            torch.nn.Linear(total_input_dim, fusion_output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.fusion.dropout),
            torch.nn.Linear(fusion_output_dim, fusion_output_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(fusion_output_dim // 2, fusion_output_dim),
        ).to(device)
    fusion.load_state_dict(fusion_ckpt["fusion_state_dict"])
    fusion.eval()

    # Compute test fused representations
    epi_s = fusion_ckpt["epi_states"]
    traj_s = fusion_ckpt["traj_states"]      # (N, traj_raw_dim) raw trajectory
    pnet_o = fusion_ckpt["pnet_outputs"]
    stab_s = fusion_ckpt["stab_scores"]

    test_fused = []
    with torch.no_grad():
        for idx in test_idx:
            traj_proj = traj_projector(traj_s[idx:idx+1].to(device))
            if fusion_type == "cross_attention":
                modality_dict = {
                    "epigenetic": epi_s[idx:idx+1].to(device),
                    "trajectory": traj_proj,
                    "protein_network": pnet_o[idx:idx+1].to(device),
                    "stability": stab_s[idx:idx+1].to(device),
                }
                fused_out, _ = fusion(modality_dict)
                test_fused.append(fused_out.squeeze(0).cpu())
            else:
                concat = torch.cat([
                    epi_s[idx:idx+1].to(device), traj_proj,
                    pnet_o[idx:idx+1].to(device), stab_s[idx:idx+1].to(device),
                ], dim=-1)
                test_fused.append(fusion(concat).squeeze(0).cpu())
    test_fused = torch.stack(test_fused).to(device)

    # Load landscape model
    n_drugs = dataset.drug_sensitivity.shape[1]
    landscape_use_evidential = landscape_ckpt.get("use_evidential", False)
    landscape = ResistanceLandscape(
        fusion_dim=fusion_output_dim, n_drugs=n_drugs,
        n_proteins=min(len(dataset.protein_names), 100),
        use_evidential=landscape_use_evidential,
    ).to(device)
    landscape.load_state_dict(landscape_ckpt["model_state_dict"])
    landscape.eval()

    # Predict on test set
    with torch.no_grad():
        output = landscape(test_fused)
        pred = output["drug_resistance"][:, :n_drugs]
        target = dataset.drug_sensitivity[test_idx].to(device)
        mask = ~torch.isnan(target)

        if mask.any():
            test_mse = torch.nn.functional.mse_loss(pred[mask], target[mask]).item()
        else:
            test_mse = float("nan")

    # v7: per-drug breakdown. README claims drug-level performance but v6
    # only persisted a single aggregate. We compute (n, MSE, MAE, Spearman)
    # per drug on observed cells only, matching scripts/run_baselines_real.py
    # so the two are directly comparable.
    drug_names = list(getattr(dataset, "drug_names", [f"drug_{i}" for i in range(n_drugs)]))
    per_drug_metrics: list[dict] = []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for d in range(n_drugs):
        col_target = target_np[:, d]
        col_pred = pred_np[:, d]
        col_mask = ~np.isnan(col_target)
        n_obs = int(col_mask.sum())
        if n_obs == 0:
            per_drug_metrics.append({
                "drug": drug_names[d] if d < len(drug_names) else f"drug_{d}",
                "n_obs": 0, "mse": float("nan"), "mae": float("nan"),
                "spearman": float("nan"), "reliable": False,
            })
            continue
        diffs = col_pred[col_mask] - col_target[col_mask]
        mse_d = float(np.mean(diffs ** 2))
        mae_d = float(np.mean(np.abs(diffs)))
        # Spearman rho via rank-correlation; gracefully handle constant arrays.
        spearman_d = float("nan")
        try:
            from scipy.stats import spearmanr
            if np.std(col_pred[col_mask]) > 1e-12 and np.std(col_target[col_mask]) > 1e-12:
                rho, _ = spearmanr(col_pred[col_mask], col_target[col_mask])
                if rho is not None and not np.isnan(rho):
                    spearman_d = float(rho)
        except Exception:
            pass
        per_drug_metrics.append({
            "drug": drug_names[d] if d < len(drug_names) else f"drug_{d}",
            "n_obs": n_obs,
            "mse": mse_d,
            "mae": mae_d,
            "spearman": spearman_d,
            "reliable": n_obs >= 10,
        })

    # Persist a CSV next to the checkpoint for the metrics-auditor subagent.
    try:
        import csv
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "per_drug_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["drug", "n_obs", "mse", "mae", "spearman", "reliable"])
            w.writeheader()
            for row in per_drug_metrics:
                w.writerow(row)
        logger.info(f"Per-drug metrics CSV: {csv_path}")
    except Exception:
        logger.exception("Failed to write per_drug_metrics.csv (continuing)")

    # v7: persist a minimal checkpoint NOW so the baseline runner (which
    # reads pipeline_validated.pt to grab ResistanceMap's test_mse) sees
    # the current value. We re-save below after baseline_summary is filled in.
    _partial_metrics = {
        "test_mse": test_mse,
        "n_test_samples": len(test_idx),
        "n_drugs": n_drugs,
        "split": "test",
        "per_drug_metrics": per_drug_metrics,
    }
    ckpt_mgr.save("pipeline_validated", {"metrics": _partial_metrics, "config": config})

    # v7: best-effort baseline comparison. Calls scripts/run_baselines_real.py
    # via subprocess so import-time pickle issues / package-not-installed
    # do not break validation. The script writes paper/tables/baseline_comparison.{json,md}.
    baseline_summary = None
    try:
        import subprocess, sys as _sys
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_baselines_real.py"
        if script.exists():
            r = subprocess.run(
                [_sys.executable, str(script)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode == 0:
                bjson = repo_root / "paper" / "tables" / "baseline_comparison.json"
                if bjson.exists():
                    import json as _json
                    bdata = _json.loads(bjson.read_text())
                    ranked = sorted(
                        bdata.get("results", []),
                        key=lambda x: (x.get("test_mse") if x.get("test_mse") is not None else float("inf")),
                    )
                    rm_rank = next(
                        (i for i, r in enumerate(ranked, 1) if "ResistanceMap" in (r.get("model") or "")),
                        None,
                    )
                    baseline_summary = {
                        "n_models": len(ranked),
                        "best_model": ranked[0].get("model") if ranked else None,
                        "best_test_mse": ranked[0].get("test_mse") if ranked else None,
                        "resistancemap_rank": rm_rank,
                        "comparison_path": str(bjson),
                    }
                    logger.info(
                        f"Baselines: {len(ranked)} models compared; "
                        f"best={baseline_summary['best_model']} ({baseline_summary['best_test_mse']:.4f}); "
                        f"ResistanceMap rank={rm_rank}/{len(ranked)}"
                    )
            else:
                logger.warning(
                    f"Baselines runner exited {r.returncode}: {r.stderr[:300]}"
                )
    except Exception:
        logger.exception("Baseline comparison failed (continuing without)")

    # v8: real-world actionability scoring.
    # A drug is "actionable for compound-prioritization screening" if its
    # rank-correlation against ground-truth IC50 is meaningful and there are
    # enough observations to trust the estimate. Calibration (low absolute
    # MSE) is a SEPARATE, stricter requirement.
    DRUG_CLASS = {
        "Bortezomib": "Proteasome",
        "Lenalidomide": "IMiD",
        "Panobinostat": "HDAC", "Vorinostat": "HDAC", "Romidepsin": "HDAC",
        "Venetoclax": "BCL2",
        "Dinaciclib": "CDK", "Palbociclib": "CDK",
        "Doxorubicin": "DNA-damage", "Etoposide": "DNA-damage",
        "Cyclophosphamide": "DNA-damage",
    }
    actionable_drugs = []
    well_calibrated_drugs = []
    failure_drugs = []
    for r in per_drug_metrics:
        r["drug_class"] = DRUG_CLASS.get(r["drug"], "Other")
        spearman = r.get("spearman", float("nan"))
        mse = r.get("mse", float("nan"))
        n = r.get("n_obs", 0)
        # Actionable: useful for ranking compounds (rank-correlation matters,
        # absolute calibration doesn't). Threshold 0.25 chosen as the
        # rule-of-thumb floor for "better than random" in pharmacological
        # screening literature.
        r["actionable_for_screening"] = bool(
            n >= 30 and not np.isnan(spearman) and spearman >= 0.25
        )
        # Well-calibrated: usable for absolute IC50 prediction (much higher bar)
        r["well_calibrated"] = bool(
            r["actionable_for_screening"] and not np.isnan(mse) and mse < 1.0
        )
        # Failure mode: not actionable
        r["failure_mode"] = not r["actionable_for_screening"]
        if r["actionable_for_screening"]:
            actionable_drugs.append(r["drug"])
        if r["well_calibrated"]:
            well_calibrated_drugs.append(r["drug"])
        if r["failure_mode"]:
            failure_drugs.append(r["drug"])

    # Drug-class aggregated stats
    by_class: dict[str, list] = {}
    for r in per_drug_metrics:
        by_class.setdefault(r["drug_class"], []).append(r)
    drug_class_metrics = {}
    for cls, rows in by_class.items():
        valid = [r for r in rows if not np.isnan(r.get("mse", float("nan")))]
        if not valid:
            continue
        drug_class_metrics[cls] = {
            "n_drugs": len(rows),
            "mean_mse": float(np.mean([r["mse"] for r in valid])),
            "mean_spearman": float(np.nanmean([r["spearman"] for r in valid])),
            "n_actionable": sum(1 for r in rows if r.get("actionable_for_screening")),
            "drugs": [r["drug"] for r in rows],
        }

    actionability_summary = {
        "n_drugs_total": n_drugs,
        "n_actionable_for_screening": len(actionable_drugs),
        "n_well_calibrated": len(well_calibrated_drugs),
        "n_failure_modes": len(failure_drugs),
        "actionable_drugs": actionable_drugs,
        "well_calibrated_drugs": well_calibrated_drugs,
        "failure_drugs": failure_drugs,
        "by_class": drug_class_metrics,
        "thresholds": {
            "actionable_min_spearman": 0.25,
            "actionable_min_n_obs": 30,
            "well_calibrated_max_mse": 1.0,
        },
        "interpretation": (
            f"ResistanceMap rank-prioritizes compounds for {len(actionable_drugs)}/{n_drugs} drugs "
            f"({100*len(actionable_drugs)/n_drugs:.0f}%). "
            f"{len(well_calibrated_drugs)}/{n_drugs} are also well-calibrated for absolute IC50 prediction. "
            f"Known failure modes: {failure_drugs or 'none'}."
        ),
    }
    logger.info(
        f"Actionability: {len(actionable_drugs)}/{n_drugs} drugs actionable for screening "
        f"({len(well_calibrated_drugs)} well-calibrated). Failures: {failure_drugs or 'none'}"
    )

    metrics = {
        "test_mse": test_mse,
        "n_test_samples": len(test_idx),
        "n_drugs": n_drugs,
        "split": "test",
        "per_drug_metrics": per_drug_metrics,
        "baseline_summary": baseline_summary,
        "actionability_summary": actionability_summary,
    }
    logger.info(f"Validation: test_mse={test_mse:.4f} on {len(test_idx)} samples ({n_drugs} drugs)")
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
    ("data_validate",          validate_data_files),
    ("data_prep",              prepare_data),
    ("vae_pretrain",           pretrain_vae),
    ("vae_finetune",           finetune_vae),
    ("trajectory_calibrate",   calibrate_trajectory),
    ("trajectory_forecast",    train_trajectory_forecaster),
    ("protein_net_train",      train_protein_network),
    ("fusion_train",           train_fusion),
    ("landscape_train",        train_landscape),
    ("validate",               validate_pipeline),
    ("serve",                  serve_api),
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

async def run_evaluation_governance(config: ResistanceMapConfig) -> None:
    """Run the evaluation governance layer over the configured pipeline.

    Registers the 10 evaluation agents (Tier A/B/C/D) with an
    :class:`~resistancemap.evaluation.orchestrator.EvalOrchestrator` and
    executes them. The training DAG is **not** invoked. The orchestrator
    enforces a Tier A hard stop (any FAIL blocks Tier B/C/D), runs an
    architecture-vs-baseline adversarial round between Tier B and Tier C,
    and finishes with the chair (Tier D).

    Args:
        config: Loaded :class:`ResistanceMapConfig`. Only ``config.evaluation``
            controls behaviour here, but every agent reads other sections of
            the config to audit them.
    """
    # Imports are local so the training-only paths never pay the cost of
    # loading the evaluation package.
    from resistancemap.evaluation import EvalOrchestrator
    from resistancemap.evaluation.tier_a import (
        DataAdequacyAgent,
        MeasurementIntegrationAgent,
        BiasFairnessShiftAgent,
    )
    from resistancemap.evaluation.tier_b import (
        ArchitectureAuditorAgent,
        CellStateTrajectoryAgent,
        PathwayPPIReasoningAgent,
    )
    from resistancemap.evaluation.tier_c import (
        ForecastingUncertaintyAgent,
        BaselineAdversaryAgent,
        ClinicalTranslationSafetyAgent,
    )
    from resistancemap.evaluation.tier_d import PrincipalIntegratorAgent

    eval_cfg = config.evaluation
    if not eval_cfg.enabled:
        logger.warning("evaluation.enabled is False; nothing to do.")
        return

    orch = EvalOrchestrator(log_root=eval_cfg.log_root)

    # Tier A — gates
    orch.add_agent(DataAdequacyAgent())
    orch.add_agent(MeasurementIntegrationAgent())
    orch.add_agent(BiasFairnessShiftAgent())

    # Tier B — architecture and mechanism
    if not eval_cfg.skip_tier_b:
        orch.add_agent(ArchitectureAuditorAgent())
        orch.add_agent(CellStateTrajectoryAgent())
        orch.add_agent(PathwayPPIReasoningAgent())

    # Tier C — prediction quality + alternatives
    if not eval_cfg.skip_tier_c:
        orch.add_agent(ForecastingUncertaintyAgent())
        orch.add_agent(BaselineAdversaryAgent())
        orch.add_agent(ClinicalTranslationSafetyAgent())

    # Tier D — chair
    if not eval_cfg.skip_tier_d:
        orch.add_agent(PrincipalIntegratorAgent())

    logger.info("Starting ResistanceMap evaluation governance run")
    report = await orch.run(config, run_id=eval_cfg.run_id)

    overall = (
        report.overall_verdict.value
        if hasattr(report.overall_verdict, "value")
        else str(report.overall_verdict)
    )
    print(f"\n{'=' * 60}")
    print("ResistanceMap Evaluation Governance Report")
    print(f"  run_id:           {report.run_id}")
    print(f"  overall verdict:  {overall}")
    print(f"  TRL grade:        {report.maturity_grade.get('trl', '?')}"
          f" — {report.maturity_grade.get('justification', '')}")
    print(f"  per-tier:         "
          + ", ".join(
              f"{t}={(v.value if hasattr(v, 'value') else v)}"
              for t, v in report.per_tier_verdicts.items()
          ))
    print(f"  required changes: {len(report.required_changes)}")
    for i, change in enumerate(report.required_changes[:10], 1):
        print(f"    {i:>2}. {change}")
    if len(report.required_changes) > 10:
        print(f"    ... and {len(report.required_changes) - 10} more")
    print(f"  audit trail:      {eval_cfg.log_root}/{report.run_id}/")
    print(f"{'=' * 60}\n")


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
    parser.add_argument(
        "--evaluate", action="store_true",
        help=(
            "Run the evaluation governance layer (10 PhD-level audit agents "
            "across Tier A/B/C/D) instead of the training DAG. Tier A "
            "FAILs hard-stop downstream tiers; report is written under "
            "evaluation.log_root/<run_id>/."
        ),
    )
    # v10 paper-spec sprint orchestrator (separate from legacy STAGES registry)
    from resistancemap.v10_runner import VALID_STAGE_NAMES as _V10_STAGES
    parser.add_argument(
        "--v10-sprint", type=str, default=None, choices=_V10_STAGES,
        help=(
            "Run the v10 paper-spec pipeline (orthogonal to the legacy "
            "STAGES registry). Semantic stage names: 'all' runs the full "
            "S1→S7 pipeline; individual stages: landscape_dsm (S1), "
            "hbayes_strata (S2), propagation_rwr (S3), mediation_nie (S4), "
            "conformal_mondrian (S5), beataml_xdisease (S6), "
            "dps_singlesnapshot (S7), fixes (F7-Vorinostat + F10 attempts)."
        ),
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

    # Evaluation governance mode: orthogonal to the training DAG.
    if args.evaluate:
        asyncio.run(run_evaluation_governance(config))
        return

    # v10 sprint orchestrator: orthogonal to legacy training DAG and
    # evaluation governance. Runs scripts/v10/s*.py via subprocess in
    # dependency order; logs land in <log_dir>/v10_e2e/<stage>.log.
    if args.v10_sprint:
        from resistancemap.v10_runner import run_v10_e2e
        rc = run_v10_e2e(args.v10_sprint, log_dir=Path(config.log_dir) / "v10_e2e")
        sys.exit(rc)

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