"""Specialized sub-agents for ResistanceMap pipeline.

Implements 10 domain-specific agents that call the real training functions
from resistancemap.main for actual model training with DAG-based parallel
execution, zero-trust verification, and complete audit trails.

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

Each agent wraps the corresponding sequential function from
resistancemap.main via asyncio.to_thread() so the event loop stays
responsive while GPU-bound work runs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from resistancemap.agents.base import BaseAgent, AgentState, AgentResult
from resistancemap.config import ResistanceMapConfig
from resistancemap.utils.checkpoint import CheckpointManager
from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

logger = logging.getLogger(__name__)


def _get_ckpt_mgr(config: ResistanceMapConfig) -> CheckpointManager:
    """Get the shared CheckpointManager attached to config by run_agentic_pipeline."""
    if hasattr(config, "_ckpt_mgr") and config._ckpt_mgr is not None:
        return config._ckpt_mgr
    return CheckpointManager(config.checkpoint_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 0
# ═══════════════════════════════════════════════════════════════════════════════

class DataValidationAgent(BaseAgent):
    """Validates all input files exist and have expected schemas."""

    def __init__(self):
        super().__init__(name="data_validation", dependencies=[])

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import validate_data_files
            ckpt_mgr = _get_ckpt_mgr(config)

            await asyncio.to_thread(validate_data_files, config, ckpt_mgr)

            output = {
                "valid": True,
                "errors": [],
                "paths_checked": {
                    "proteomics": str(config.data.ccle_proteomics_path),
                    "ppi": str(config.data.string_ppi_path),
                    "gdsc": str(config.data.gdsc_path),
                },
            }
            metadata = {"validation_timestamp": time.time()}
            return self._make_result(AgentState.COMPLETED, output, metadata=metadata)

        except FileNotFoundError as e:
            logger.error(f"Data validation failed: {e}")
            return self._make_result(
                AgentState.FAILED, error=f"Data validation failed: {e}",
            )
        except Exception as e:
            logger.exception("DataValidationAgent failed")
            return self._make_result(
                AgentState.FAILED, error=f"Data validation error: {e}",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1
# ═══════════════════════════════════════════════════════════════════════════════

class DataPrepAgent(BaseAgent):
    """Harmonizes multi-omics data (proteomics, epigenomics, PPI, scRNA, MMRF)."""

    def __init__(self):
        super().__init__(name="data_prep", dependencies=["data_validation"])

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if "data_validation" not in inputs:
            return False, "Missing data_validation output"
        val = inputs["data_validation"]
        if not isinstance(val, dict) or not val.get("valid"):
            return False, "Data validation did not pass"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import prepare_data
            ckpt_mgr = _get_ckpt_mgr(config)

            result_path = await asyncio.to_thread(prepare_data, config, ckpt_mgr)

            # Read back stats from the checkpoint
            ckpt = ckpt_mgr.load("data_ready")
            ds = ckpt["dataset"]
            output = {
                "checkpoint_path": str(result_path),
                "proteomics_shape": tuple(ds.proteomics.shape),
                "epigenomics_shape": tuple(ds.epigenomics.shape),
                "n_samples": len(ds),
                "n_proteins": ds.proteomics.shape[1],
            }
            metadata = {
                "n_samples": len(ds),
                "n_proteins": ds.proteomics.shape[1],
            }
            return self._make_result(AgentState.COMPLETED, output, metadata=metadata)

        except Exception as e:
            logger.exception("DataPrepAgent failed")
            return self._make_result(AgentState.FAILED, error=f"Data prep failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2  (VAE pretrain + ESM-2 embed run in parallel)
# ═══════════════════════════════════════════════════════════════════════════════

class VAEPretrainAgent(BaseAgent):
    """Pretrains the proteome-to-epigenome conditional VAE on pan-cancer CCLE data."""

    def __init__(self):
        super().__init__(name="vae_pretrain", dependencies=["data_prep"])

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if "data_prep" not in inputs:
            return False, "Missing data_prep output"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import pretrain_vae
            ckpt_mgr = _get_ckpt_mgr(config)

            result_path = await asyncio.to_thread(pretrain_vae, config, ckpt_mgr)

            ckpt = ckpt_mgr.load("vae_pretrained")
            metrics = ckpt.get("metrics", {})
            output = {
                "checkpoint_path": str(result_path),
                "model_type": "ProteomeToEpigenomeVAE",
                "metrics": metrics,
            }
            return self._make_result(AgentState.COMPLETED, output, metadata=metrics)

        except Exception as e:
            logger.exception("VAEPretrainAgent failed")
            return self._make_result(AgentState.FAILED, error=f"VAE pretrain failed: {e}")


class ESM2EmbedAgent(BaseAgent):
    """Validates ESM-2 protein embeddings availability.

    The current PPI-GNN training builds node features from
    protein abundance + VAE latent + stability scores.  This agent
    pre-validates protein metadata so downstream ProteinNetAgent can
    proceed immediately.  Full ESM-2 forward-pass integration is a
    planned enhancement.
    """

    def __init__(self):
        super().__init__(name="esm2_embed", dependencies=["data_prep"])

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if "data_prep" not in inputs:
            return False, "Missing data_prep output"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        log_stage_start("esm2_embed")
        try:
            ckpt_mgr = _get_ckpt_mgr(config)
            ckpt = ckpt_mgr.load("data_ready")
            ds = ckpt["dataset"]
            protein_names = ds.protein_names

            # v7: be explicit about whether this layer actually produces ESM-2
            # embeddings or just validates protein metadata. The dataset only
            # has FASTA sequences when the user supplied them upstream; CCLE
            # proteomics by itself does not. main.train_protein_network() is
            # the only place that runs the real ESM-2 forward, and only when
            # ds.protein_sequences is non-empty.
            has_sequences = bool(getattr(ds, "protein_sequences", None))
            embeddings_produced = False  # this agent never produces them itself

            if has_sequences:
                logger.info(
                    f"ESM-2 embed: {len(protein_names)} proteins ready for "
                    f"{config.protein_net.esm2_model}; sequences present — "
                    "real embeddings will be produced inside ProteinNetAgent."
                )
            else:
                logger.warning(
                    f"ESM-2 embed: {len(protein_names)} proteins validated, "
                    "but dataset.protein_sequences is empty. ProteinNetAgent "
                    "will fall back to abundance/latent/stability features "
                    "(no real ESM-2 forward pass). To enable, populate "
                    "dataset.protein_sequences with FASTA sequences during data_prep."
                )

            output = {
                "n_proteins": len(protein_names),
                "esm2_model": config.protein_net.esm2_model,
                "embedding_dim": config.protein_net.esm2_dim,
                "protein_names_sample": protein_names[:10],
                "has_sequences": has_sequences,
                "embeddings_produced_here": embeddings_produced,
            }
            metadata = {
                "n_proteins": len(protein_names),
                "esm2_model": config.protein_net.esm2_model,
                "embedding_dim": config.protein_net.esm2_dim,
                "has_sequences": has_sequences,
                "embeddings_produced_here": embeddings_produced,
            }
            log_stage_end("esm2_embed", metadata)
            return self._make_result(AgentState.COMPLETED, output, metadata=metadata)

        except Exception as e:
            logger.exception("ESM2EmbedAgent failed")
            return self._make_result(AgentState.FAILED, error=f"ESM-2 embed failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3
# ═══════════════════════════════════════════════════════════════════════════════

class VAEFinetuneAgent(BaseAgent):
    """Fine-tunes the VAE on hematological cell lines."""

    def __init__(self):
        super().__init__(name="vae_finetune", dependencies=["vae_pretrain"])

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if "vae_pretrain" not in inputs:
            return False, "Missing vae_pretrain output"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import finetune_vae
            ckpt_mgr = _get_ckpt_mgr(config)

            result_path = await asyncio.to_thread(finetune_vae, config, ckpt_mgr)

            ckpt = ckpt_mgr.load("vae_finetuned")
            metrics = ckpt.get("metrics", {})
            output = {
                "checkpoint_path": str(result_path),
                "model_type": "ProteomeToEpigenomeVAE",
                "finetuned": True,
                "metrics": metrics,
            }
            return self._make_result(AgentState.COMPLETED, output, metadata=metrics)

        except Exception as e:
            logger.exception("VAEFinetuneAgent failed")
            return self._make_result(AgentState.FAILED, error=f"VAE finetune failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryAgent(BaseAgent):
    """Calibrates the ODE-based resistance trajectory (MemoryStabilityScorer)."""

    def __init__(self):
        super().__init__(name="trajectory", dependencies=["vae_finetune"])

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if "vae_finetune" not in inputs:
            return False, "Missing vae_finetune output"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import calibrate_trajectory, train_trajectory_forecaster
            ckpt_mgr = _get_ckpt_mgr(config)

            calib_path = await asyncio.to_thread(calibrate_trajectory, config, ckpt_mgr)
            forecaster_path = await asyncio.to_thread(train_trajectory_forecaster, config, ckpt_mgr)

            calib_ckpt = ckpt_mgr.load("stability_calibrated")
            forecaster_ckpt = ckpt_mgr.load("trajectory_forecaster_trained")
            metrics = {**calib_ckpt.get("metrics", {}), **forecaster_ckpt.get("metrics", {})}
            output = {
                "checkpoint_path": str(calib_path),
                "forecaster_checkpoint_path": str(forecaster_path),
                "model_type": "MemoryStabilityScorer+TrajectoryForecaster",
                "calibrated": True,
                "trained": True,
                "metrics": metrics,
            }
            return self._make_result(AgentState.COMPLETED, output, metadata=metrics)

        except Exception as e:
            logger.exception("TrajectoryAgent failed")
            return self._make_result(AgentState.FAILED, error=f"Trajectory calibration/forecaster failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 5  (ProteinNet needs trajectory checkpoint for stability scores)
# ═══════════════════════════════════════════════════════════════════════════════

class ProteinNetAgent(BaseAgent):
    """Trains protein interaction GNN on the STRING PPI graph.

    Node features: protein abundance (1) + VAE latent (64) + stability (1) = 66-dim.
    Requires checkpoints from esm2_embed (protein metadata), vae_finetune (latent),
    AND trajectory (stability scores).
    """

    def __init__(self):
        super().__init__(
            name="protein_net",
            dependencies=["esm2_embed", "vae_finetune", "trajectory"],
        )

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        for dep in ("esm2_embed", "vae_finetune", "trajectory"):
            if dep not in inputs:
                return False, f"Missing {dep} output"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import train_protein_network
            ckpt_mgr = _get_ckpt_mgr(config)

            result_path = await asyncio.to_thread(train_protein_network, config, ckpt_mgr)

            ckpt = ckpt_mgr.load("protein_net_trained")
            metrics = ckpt.get("metrics", {})
            output = {
                "checkpoint_path": str(result_path),
                "model_type": "PPIGraphNetwork",
                "metrics": metrics,
            }
            return self._make_result(AgentState.COMPLETED, output, metadata=metrics)

        except Exception as e:
            logger.exception("ProteinNetAgent failed")
            return self._make_result(AgentState.FAILED, error=f"Protein net training failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 6
# ═══════════════════════════════════════════════════════════════════════════════

class FusionAgent(BaseAgent):
    """Trains cross-modal fusion (epi + trajectory + protein-net + stability)."""

    def __init__(self):
        super().__init__(
            name="fusion",
            dependencies=["trajectory", "protein_net"],
        )

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if "trajectory" not in inputs or "protein_net" not in inputs:
            return False, "Missing trajectory or protein_net output"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import train_fusion
            ckpt_mgr = _get_ckpt_mgr(config)

            result_path = await asyncio.to_thread(train_fusion, config, ckpt_mgr)

            ckpt = ckpt_mgr.load("fusion_trained")
            metrics = ckpt.get("metrics", {})
            output = {
                "checkpoint_path": str(result_path),
                "model_type": "MultiModalFusion",
                "metrics": metrics,
            }
            return self._make_result(AgentState.COMPLETED, output, metadata=metrics)

        except Exception as e:
            logger.exception("FusionAgent failed")
            return self._make_result(AgentState.FAILED, error=f"Fusion training failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 7
# ═══════════════════════════════════════════════════════════════════════════════

class LandscapeAgent(BaseAgent):
    """Trains the resistance landscape predictor."""

    def __init__(self):
        super().__init__(name="landscape", dependencies=["fusion"])

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if "fusion" not in inputs:
            return False, "Missing fusion output"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import train_landscape
            ckpt_mgr = _get_ckpt_mgr(config)

            result_path = await asyncio.to_thread(train_landscape, config, ckpt_mgr)

            ckpt = ckpt_mgr.load("landscape_trained")
            metrics = ckpt.get("metrics", {})
            output = {
                "checkpoint_path": str(result_path),
                "model_type": "ResistanceLandscape",
                "metrics": metrics,
            }
            return self._make_result(AgentState.COMPLETED, output, metadata=metrics)

        except Exception as e:
            logger.exception("LandscapeAgent failed")
            return self._make_result(AgentState.FAILED, error=f"Landscape training failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 8
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationAgent(BaseAgent):
    """End-to-end validation on held-out test data with SOTA comparison."""

    def __init__(self):
        super().__init__(name="validation", dependencies=["landscape"])

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        if "landscape" not in inputs:
            return False, "Missing landscape output"
        return True, ""

    async def execute(self, inputs: dict[str, Any], config: ResistanceMapConfig) -> AgentResult:
        try:
            from resistancemap.main import validate_pipeline
            ckpt_mgr = _get_ckpt_mgr(config)

            result_path = await asyncio.to_thread(validate_pipeline, config, ckpt_mgr)

            ckpt = ckpt_mgr.load("pipeline_validated")
            metrics = ckpt.get("metrics", {})
            output = {
                "checkpoint_path": str(result_path),
                "validation_passed": True,
                "metrics": metrics,
            }
            return self._make_result(AgentState.COMPLETED, output, metadata=metrics)

        except Exception as e:
            logger.exception("ValidationAgent failed")
            return self._make_result(AgentState.FAILED, error=f"Validation failed: {e}")
