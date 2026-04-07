"""End-to-end ResistanceMap inference pipeline (L0 to L5).

Adapted from MyeloMemory's pipeline.py. Orchestrates:
    L0: Data preparation (proteomics normalization)
    L1: VAE pretraining/finetuning
    L2: Stability scoring (resistance memory)
    L3: Protein network analysis (GNN)
    L4: Multi-omics fusion
    L5: Resistance landscape prediction

Given raw proteomics data, produces LandscapeResult with drug resistance
predictions, intervention targets, and basin-of-attraction maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from resistancemap.landscape.predictor import (
    ResistanceLandscape,
    ResistanceLandscapePredictor,
    LandscapeResult,
    TargetRanker,
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalLandscape:
    """Container for temporal resistance landscape trajectory.

    Attributes:
        timepoints: List of timepoint labels (e.g., ["baseline", "3m", "6m"]).
        landscapes: List of LandscapeResult, one per timepoint.
        trajectory_state_sequence: Observed sequence of resistance states.
        predicted_state_sequence: Model's prediction of resistance state evolution.
        trajectory_accuracy: Fraction of predicted states matching observed (if available).
    """

    timepoints: list[str]
    landscapes: list[LandscapeResult]
    trajectory_state_sequence: list[str]
    predicted_state_sequence: list[str]
    trajectory_accuracy: float | None = None


class ResistanceMapPipeline:
    """Full inference pipeline wiring L0–L5 for ResistanceMap.

    Usage:
        pipeline = ResistanceMapPipeline.from_checkpoints(ckpt_mgr, config)
        result = pipeline.predict_single(proteomics_dict)
        trajectory = pipeline.predict_trajectory(proteomics_timeseries)

    Args:
        landscape_model: Trained ResistanceLandscape module (L5).
        landscape_predictor: ResistanceLandscapePredictor instance.
        drug_names: List of target drug names.
        protein_names: List of protein names in the network.
        state_names: List of resistance state names.
        device: Torch device for inference.
        config: ResistanceMapConfig (optional).
    """

    def __init__(
        self,
        landscape_model: ResistanceLandscape,
        landscape_predictor: ResistanceLandscapePredictor,
        drug_names: list[str],
        protein_names: list[str],
        state_names: list[str] | None = None,
        device: str = "cpu",
        config: Any | None = None,
    ) -> None:
        """Initialize the ResistanceMapPipeline.

        Args:
            landscape_model: Trained L5 model.
            landscape_predictor: Predictor wrapper for L5.
            drug_names: Drug names.
            protein_names: Protein names.
            state_names: Resistance state names.
            device: Torch device.
            config: Optional configuration dict.
        """
        self.landscape_model = landscape_model.to(device).eval()
        self.landscape_predictor = landscape_predictor
        self.drug_names = drug_names
        self.protein_names = protein_names
        self.state_names = state_names or ["sensitive", "intermediate", "resistant"]
        self.device = device
        self.config = config or {}

        logger.info(
            f"ResistanceMapPipeline initialized: "
            f"{len(drug_names)} drugs, {len(protein_names)} proteins, "
            f"{len(self.state_names)} resistance states"
        )

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_dir: str | None = None,
        config: Any | None = None,
    ) -> "ResistanceMapPipeline":
        """Load all pipeline stages from saved checkpoints.

        Args:
            checkpoint_dir: Directory containing stage checkpoints.
            config: Pipeline configuration.

        Returns:
            Initialized ResistanceMapPipeline ready for inference.

        Note:
            In production, this would load L0-L4 models from checkpoints
            and compose them. For demonstration, we create a minimal
            pipeline with L5 only.
        """
        config = config or {}
        device = config.get("device", "cpu")

        # In a full implementation:
        # - Load L1 VAE from "vae_finetuned.pt"
        # - Load L2 stability scorer from "stability_calibrated.pt"
        # - Load L3 GNN from "gnn_trained.pt"
        # - Load L4 fusion module from "fusion_trained.pt"
        # - Load L5 landscape model from "landscape_trained.pt"

        drug_names = config.get("drug_names", [
            "Bortezomib", "Lenalidomide", "Dexamethasone",
            "Carfilzomib", "Pomalidomide", "Daratumumab"
        ])
        protein_names = config.get("protein_names", [f"PROT_{i}" for i in range(100)])
        state_names = config.get("state_names", ["sensitive", "intermediate", "resistant"])

        # Create L5 model
        fusion_dim = config.get("fusion_dim", 128)
        n_drugs = len(drug_names)
        n_proteins = len(protein_names)
        landscape_model = ResistanceLandscape(
            fusion_dim=fusion_dim,
            n_drugs=n_drugs,
            n_timepoints=3,
            n_states=len(state_names),
            n_proteins=n_proteins,
            hidden_dim=256,
            dropout_rate=0.2,
        ).to(device)

        # Create target ranker
        known_drug_targets = config.get("known_drug_targets", {})
        ppi_centrality = config.get("ppi_centrality", {})
        target_ranker = TargetRanker(
            known_drug_targets=known_drug_targets,
            ppi_centrality=ppi_centrality,
            device=device,
        )

        # Create landscape predictor
        landscape_predictor = ResistanceLandscapePredictor(
            model=landscape_model,
            drug_names=drug_names,
            protein_names=protein_names,
            state_names=state_names,
            target_ranker=target_ranker,
            device=device,
        )

        logger.info("Pipeline loaded from configuration (L5 landscape model ready)")

        return cls(
            landscape_model=landscape_model,
            landscape_predictor=landscape_predictor,
            drug_names=drug_names,
            protein_names=protein_names,
            state_names=state_names,
            device=device,
            config=config,
        )

    @torch.no_grad()
    def predict_single(
        self,
        proteomics: dict[str, float] | torch.Tensor,
        sample_id: str | None = None,
    ) -> LandscapeResult:
        """Run inference on a single sample's proteomics profile.

        In a full implementation, this would:
        1. Normalize proteomics (L0)
        2. Pass through VAE encoder (L1)
        3. Compute stability score (L2)
        4. Run through GNN (L3)
        5. Fuse representations (L4)
        6. Predict resistance landscape (L5)

        For now, demonstrates L5 prediction on a mock fused representation.

        Args:
            proteomics: Dict mapping protein_name → abundance, or (n_proteins,) tensor.
            sample_id: Optional sample identifier.

        Returns:
            LandscapeResult with predictions.
        """
        # Convert to tensor if needed
        if isinstance(proteomics, dict):
            prot_vector = torch.zeros(len(self.protein_names), dtype=torch.float32)
            for i, prot_name in enumerate(self.protein_names):
                if prot_name in proteomics:
                    prot_vector[i] = proteomics[prot_name]
            proteomics = prot_vector
        else:
            proteomics = torch.as_tensor(proteomics, dtype=torch.float32)

        # Mock: use proteomics as fused representation (in production, run L0-L4)
        fused_repr = proteomics[:self.landscape_model.fusion_dim].to(self.device)

        # Ensure correct shape
        if fused_repr.shape[0] < self.landscape_model.fusion_dim:
            padding = torch.zeros(
                self.landscape_model.fusion_dim - fused_repr.shape[0],
                device=self.device
            )
            fused_repr = torch.cat([fused_repr, padding])
        else:
            fused_repr = fused_repr[:self.landscape_model.fusion_dim]

        result = self.landscape_predictor.predict_single(fused_repr, sample_id=sample_id)
        logger.info(f"Single sample prediction: {result.resistance_state} (confidence={result.confidence_score:.3f})")
        return result

    @torch.no_grad()
    def predict_batch(
        self,
        dataset: Any,
        batch_size: int = 32,
    ) -> list[LandscapeResult]:
        """Run inference on a batch dataset.

        Args:
            dataset: Dataset with proteomics profiles.
            batch_size: Batch size for processing.

        Returns:
            List of LandscapeResult, one per sample.
        """
        results = []
        n_samples = len(dataset)

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_samples = []

            for j in range(i, batch_end):
                sample = dataset[j]
                if isinstance(sample, dict) and "proteomics" in sample:
                    batch_samples.append(sample["proteomics"])
                else:
                    batch_samples.append(sample)

            # Stack into tensor
            if isinstance(batch_samples[0], torch.Tensor):
                batch_tensor = torch.stack(batch_samples).to(self.device)
            else:
                batch_tensor = torch.tensor(batch_samples, dtype=torch.float32).to(self.device)

            # Truncate to fusion_dim
            fusion_dim = self.landscape_model.fusion_dim
            batch_tensor = batch_tensor[:, :fusion_dim]

            # Pad if needed
            if batch_tensor.shape[1] < fusion_dim:
                padding = torch.zeros(
                    batch_tensor.shape[0],
                    fusion_dim - batch_tensor.shape[1],
                    device=self.device
                )
                batch_tensor = torch.cat([batch_tensor, padding], dim=1)

            # Predict batch
            batch_results = self.landscape_predictor.predict_batch(batch_tensor)
            results.extend(batch_results)

        logger.info(f"Batch prediction complete: {len(results)} samples")
        return results

    @torch.no_grad()
    def predict_trajectory(
        self,
        proteomics_timeseries: list[dict[str, float] | torch.Tensor],
        timepoint_labels: list[str] | None = None,
        actual_states: list[str] | None = None,
    ) -> TemporalLandscape:
        """Predict resistance landscape trajectory over time.

        Given multiple proteomics snapshots, predict the full resistance
        state trajectory and basin-of-attraction evolution.

        Args:
            proteomics_timeseries: List of proteomics profiles (one per timepoint).
            timepoint_labels: Optional labels for timepoints (e.g., ["baseline", "3m", "6m"]).
            actual_states: Optional ground truth resistance states for comparison.

        Returns:
            TemporalLandscape with predicted trajectory.
        """
        if timepoint_labels is None:
            timepoint_labels = [f"T{i}" for i in range(len(proteomics_timeseries))]

        assert len(proteomics_timeseries) == len(timepoint_labels), \
            f"Timeseries length {len(proteomics_timeseries)} != labels length {len(timepoint_labels)}"

        landscapes = []
        predicted_states = []

        for t, (prot, label) in enumerate(zip(proteomics_timeseries, timepoint_labels)):
            landscape = self.predict_single(prot, sample_id=f"trajectory_t{t}")
            landscapes.append(landscape)
            predicted_states.append(landscape.resistance_state)

        # Compute trajectory accuracy if ground truth available
        trajectory_accuracy = None
        if actual_states is not None:
            assert len(actual_states) == len(predicted_states), \
                f"Actual states length {len(actual_states)} != predicted {len(predicted_states)}"
            n_correct = sum(1 for a, p in zip(actual_states, predicted_states) if a == p)
            trajectory_accuracy = n_correct / len(actual_states)

        logger.info(
            f"Trajectory prediction: {len(landscapes)} timepoints, "
            f"states={predicted_states}, accuracy={trajectory_accuracy}"
        )

        return TemporalLandscape(
            timepoints=timepoint_labels,
            landscapes=landscapes,
            trajectory_state_sequence=actual_states or [],
            predicted_state_sequence=predicted_states,
            trajectory_accuracy=trajectory_accuracy,
        )
