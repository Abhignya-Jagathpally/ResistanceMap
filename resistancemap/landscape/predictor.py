"""Resistance Landscape Predictor: L5 module for resistance state prediction.

Adapted from MyeloMemory architecture. Takes fused representation from L4
and produces per-drug resistance probabilities, intervention targets, and
basin-of-attraction maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LandscapeResult:
    """Container for a single sample's resistance landscape prediction.

    Attributes:
        sample_id: Optional identifier for the sample.
        fused_representation: (fusion_dim,) fused multi-omics embedding from L4.
        drug_resistance_3m: Dict mapping drug name → P(resistant) at 3 months.
        drug_resistance_6m: Dict mapping drug name → P(resistant) at 6 months.
        drug_resistance_12m: Dict mapping drug name → P(resistant) at 12 months.
        top_intervention_targets: List of (protein_name, resistance_contribution_score, actionability_score).
        resistance_state: Predicted current resistance state (e.g., "sensitive", "intermediate", "resistant").
        basin_of_attraction: Predicted future resistance state toward which tumor is heading.
        transition_probabilities: (n_states, n_states) matrix of state-to-state transition probs.
        confidence_score: Overall confidence in the prediction (0-1).
        epistemic_uncertainty: Model uncertainty (0-1), from Dirichlet when evidential=True.
        aleatoric_uncertainty: Data uncertainty (0-1), from expected categorical when evidential=True.
        evidence_strength: Sum of Dirichlet concentration parameters (evidence support).
    """

    sample_id: str | None = None
    fused_representation: torch.Tensor | None = None
    drug_resistance_3m: dict[str, float] = field(default_factory=dict)
    drug_resistance_6m: dict[str, float] = field(default_factory=dict)
    drug_resistance_12m: dict[str, float] = field(default_factory=dict)
    top_intervention_targets: list[tuple[str, float, float]] = field(default_factory=list)
    resistance_state: str = "unknown"
    basin_of_attraction: str = "unknown"
    transition_probabilities: torch.Tensor | None = None
    confidence_score: float = 0.0
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    evidence_strength: float = 0.0


class TargetRanker:
    """Ranks intervention targets by actionability and resistance contribution.

    Combines:
    - Per-protein resistance contribution scores from L3
    - Known drug target status (literature/databases)
    - Protein-protein interaction (PPI) centrality

    Attributes:
        known_drug_targets: Dict mapping protein name → list of drugs it's a target for.
        ppi_centrality: Dict mapping protein name → centrality score in PPI network.
        device: Torch device.
    """

    def __init__(
        self,
        known_drug_targets: dict[str, list[str]] | None = None,
        ppi_centrality: dict[str, float] | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the TargetRanker.

        Args:
            known_drug_targets: Mapping of protein → list of drugs targeting it.
            ppi_centrality: Mapping of protein → centrality score (0-1).
            device: Torch device.
        """
        self.known_drug_targets = known_drug_targets or {}
        self.ppi_centrality = ppi_centrality or {}
        self.device = device

    def rank_targets(
        self,
        protein_names: list[str],
        resistance_scores: torch.Tensor,
        top_k: int = 10,
    ) -> list[tuple[str, float, float]]:
        """Rank proteins by actionability for intervention.

        Args:
            protein_names: List of protein names.
            resistance_scores: (n_proteins,) per-protein resistance contribution scores.
            top_k: Number of top targets to return.

        Returns:
            List of (protein_name, resistance_contribution, actionability_score).
        """
        assert len(protein_names) == len(resistance_scores), \
            f"Protein names length {len(protein_names)} != scores length {len(resistance_scores)}"

        if isinstance(resistance_scores, torch.Tensor):
            resistance_scores = resistance_scores.cpu().numpy()

        results = []

        for prot_name, res_score in zip(protein_names, resistance_scores):
            # Resistance contribution (0-1)
            contrib = float(res_score)

            # Actionability: weighted combination of drug target status + PPI centrality
            is_drug_target = 1.0 if prot_name in self.known_drug_targets else 0.0
            ppi_cent = self.ppi_centrality.get(prot_name, 0.0)

            actionability = 0.6 * is_drug_target + 0.4 * ppi_cent

            results.append((prot_name, contrib, actionability))

        # Sort by combined score: resistance * actionability
        results.sort(
            key=lambda x: x[1] * x[2],  # resistance_contribution * actionability
            reverse=True
        )

        return results[:top_k]


class EvidentialResistanceHead(nn.Module):
    """Evidential output head for resistance state prediction using Dirichlet distribution.

    Instead of standard softmax, outputs Dirichlet concentration parameters α.
    The predictive distribution is Dir(α), and uncertainty is derived from:
    - Epistemic (model) uncertainty: K / sum(α), where K = n_states
    - Aleatoric (data) uncertainty: expected entropy of the categorical distribution

    The confidence in the prediction is calibrated as: S / (S + K), where S = sum(α).
    """

    def __init__(self, in_features: int, n_states: int) -> None:
        """Initialize the evidential resistance head.

        Args:
            in_features: Number of input features.
            n_states: Number of resistance states.
        """
        super().__init__()
        self.n_states = n_states

        # Network to produce Dirichlet concentration parameters
        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_states),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning alpha, probability, and uncertainty.

        Args:
            x: (batch_size, in_features) input tensor.

        Returns:
            Tuple of:
            - alpha: (batch_size, n_states) Dirichlet concentration parameters
            - probs: (batch_size, n_states) expected probabilities (α / sum(α))
            - uncertainties: (batch_size,) epistemic uncertainty per sample
        """
        # Output logits and convert to concentration parameters via softplus
        logits = self.network(x)
        alpha = torch.nn.functional.softplus(logits) + 1.0  # Ensure α > 0

        # Expected probabilities: α / sum(α)
        S = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / S

        # Epistemic uncertainty: K / sum(α)
        K = float(self.n_states)
        epistemic_unc = K / (S.squeeze(-1))

        return alpha, probs, epistemic_unc

    def compute_aleatoric_uncertainty(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute aleatoric (data) uncertainty from alpha.

        Aleatoric uncertainty is the expected entropy under the Dirichlet distribution.

        Args:
            alpha: (batch_size, n_states) concentration parameters.

        Returns:
            (batch_size,) aleatoric uncertainty per sample, guaranteed non-negative.
        """
        S = torch.sum(alpha, dim=-1)

        # Expected entropy: sum( -p_i * psi(alpha_i - psi(S)) )
        # Approximation: (K - S) / (S * (S + 1)) where K = n_states
        K = float(self.n_states)
        aleatoric = (K - S) / (S * (S + 1.0))

        # Ensure non-negative: clamp to [0, inf)
        aleatoric = torch.clamp(aleatoric, min=0.0)

        return aleatoric


class ResistanceLandscape(nn.Module):
    """Neural module predicting resistance landscape and intervention targets.

    L5 layer that:
    1. Takes fused representation from L4
    2. Predicts per-drug resistance at multiple timepoints (3, 6, 12 months)
    3. Produces basin-of-attraction map (which resistance state system is heading to)
    4. Computes state-to-state transition probability matrix
    5. Ranks intervention targets by resistance contribution + actionability

    Optionally supports evidential uncertainty quantification via Dirichlet distribution
    for state prediction, providing epistemic and aleatoric uncertainty estimates.

    ARCHITECTURAL NOTE: This landscape predictor produces predictions that may conflict with
    the trajectory forecaster's outputs. Downstream consumers should use one or the other,
    or explicitly combine them with a documented fusion strategy. The landscape output is
    for current state prediction and future resistance trajectory, while trajectory forecaster
    predicts time-indexed survival/event probabilities.

    Architecture:
    - Input: fused_dim (from L4)
    - Hidden layers: 256 -> 128 -> 64
    - Heads:
        - Drug resistance head: (n_drugs * n_timepoints,) outputs → sigmoid
        - State classification head: (n_states,) outputs → softmax OR EvidentialResistanceHead
        - Transition matrix head: (n_states * n_states,) outputs → softmax
        - Target ranking head: (n_proteins,) resistance contribution scores

    Args:
        fusion_dim: Dimension of fused representation from L4.
        n_drugs: Number of drugs to predict (default 6).
        n_timepoints: Number of timepoints for resistance prediction (default 3: 3m, 6m, 12m).
        n_states: Number of resistance states (default 3: sensitive, intermediate, resistant).
        n_proteins: Number of proteins in the network.
        hidden_dim: Size of hidden layers (default 256).
        dropout_rate: Dropout probability (default 0.2).
        use_evidential: If True, use EvidentialResistanceHead for state prediction (default False).
    """

    def __init__(
        self,
        fusion_dim: int,
        n_drugs: int = 6,
        n_timepoints: int = 3,
        n_states: int = 3,
        n_proteins: int = 100,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        use_evidential: bool = False,
    ) -> None:
        super().__init__()

        self.fusion_dim = fusion_dim
        self.n_drugs = n_drugs
        self.n_timepoints = n_timepoints
        self.n_states = n_states
        self.n_proteins = n_proteins
        self.hidden_dim = hidden_dim
        self.use_evidential = use_evidential

        # Shared encoder: fused_dim -> hidden_dim -> hidden_dim/2
        self.encoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Drug resistance head: predict P(resistant) at each timepoint per drug
        self.drug_resistance_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_drugs * n_timepoints),
        )

        # State classification head: predict current resistance state
        if use_evidential:
            self.state_head = EvidentialResistanceHead(hidden_dim // 2, n_states)
        else:
            self.state_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, n_states),
            )

        # Transition matrix head: predict state-to-state transitions
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_states * n_states),
        )

        # Target ranking head: per-protein resistance contribution scores
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_proteins),
        )

    def forward(self, fused_representation: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the resistance landscape predictor.

        Args:
            fused_representation: (batch_size, fusion_dim) fused embedding from L4.

        Returns:
            Dict with keys:
                - "drug_resistance": (batch_size, n_drugs * n_timepoints) probs
                - "resistance_state": (batch_size, n_states) logits (or probs if evidential)
                - "transition_matrix": (batch_size, n_states, n_states) softmax probs
                - "target_scores": (batch_size, n_proteins) resistance contribution scores
                - "alpha" (if evidential): (batch_size, n_states) Dirichlet parameters
                - "epistemic_uncertainty" (if evidential): (batch_size,) model uncertainty
                - "aleatoric_uncertainty" (if evidential): (batch_size,) data uncertainty
        """
        batch_size = fused_representation.shape[0]

        # Shared encoding
        encoded = self.encoder(fused_representation)

        # Drug resistance predictions (logits)
        drug_logits = self.drug_resistance_head(encoded)
        drug_probs = torch.sigmoid(drug_logits)

        # Current resistance state (logits or evidential)
        if self.use_evidential:
            alpha, state_probs, epistemic_unc = self.state_head(encoded)
            aleatoric_unc = self.state_head.compute_aleatoric_uncertainty(alpha)
            state_output = state_probs
        else:
            state_logits = self.state_head(encoded)
            state_output = state_logits
            alpha = None
            epistemic_unc = None
            aleatoric_unc = None

        # State-to-state transitions
        trans_logits = self.transition_head(encoded)
        trans_logits_reshaped = trans_logits.view(batch_size, self.n_states, self.n_states)
        transition_probs = torch.softmax(trans_logits_reshaped, dim=-1)

        # Target scores (0-1, via sigmoid)
        target_logits = self.target_head(encoded)
        target_scores = torch.sigmoid(target_logits)

        output = {
            "drug_resistance": drug_probs,
            "resistance_state": state_output,
            "transition_matrix": transition_probs,
            "target_scores": target_scores,
        }

        if self.use_evidential:
            output["alpha"] = alpha
            output["epistemic_uncertainty"] = epistemic_unc
            output["aleatoric_uncertainty"] = aleatoric_unc

        return output


class ResistanceLandscapePredictor:
    """End-to-end predictor for resistance landscape results.

    Takes a fused representation from L4 and the trained ResistanceLandscape model,
    and produces a LandscapeResult with all predictions.

    Args:
        model: Trained ResistanceLandscape module.
        drug_names: List of drug names (length n_drugs).
        protein_names: List of protein names (length n_proteins).
        state_names: List of resistance state names (length n_states).
        target_ranker: TargetRanker instance for ranking intervention targets.
        device: Torch device.
    """

    def __init__(
        self,
        model: ResistanceLandscape,
        drug_names: list[str],
        protein_names: list[str],
        state_names: list[str] | None = None,
        target_ranker: TargetRanker | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the ResistanceLandscapePredictor.

        Args:
            model: Trained ResistanceLandscape module.
            drug_names: List of drug names.
            protein_names: List of protein names.
            state_names: State names (default: ["sensitive", "intermediate", "resistant"]).
            target_ranker: TargetRanker for ranking intervention targets.
            device: Torch device.
        """
        self.model = model.to(device).eval()
        self.drug_names = drug_names
        self.protein_names = protein_names
        self.state_names = state_names or ["sensitive", "intermediate", "resistant"]
        self.target_ranker = target_ranker or TargetRanker(device=device)
        self.device = device

        assert len(drug_names) == model.n_drugs, \
            f"Drug names length {len(drug_names)} != model n_drugs {model.n_drugs}"
        assert len(protein_names) == model.n_proteins, \
            f"Protein names length {len(protein_names)} != model n_proteins {model.n_proteins}"
        assert len(self.state_names) == model.n_states, \
            f"State names length {len(self.state_names)} != model n_states {model.n_states}"

    @torch.no_grad()
    def predict_single(
        self,
        fused_representation: torch.Tensor,
        sample_id: str | None = None,
        protein_resistance_scores: torch.Tensor | None = None,
    ) -> LandscapeResult:
        """Predict resistance landscape for a single sample.

        Args:
            fused_representation: (fusion_dim,) tensor from L4.
            sample_id: Optional sample identifier.
            protein_resistance_scores: (n_proteins,) per-protein resistance scores for ranking.

        Returns:
            LandscapeResult with all predictions and optional uncertainty estimates.
        """
        # Add batch dimension
        batch_input = fused_representation.unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(batch_input)

        drug_probs = outputs["drug_resistance"].squeeze(0)  # (n_drugs * n_timepoints,)
        state_output = outputs["resistance_state"].squeeze(0)  # (n_states,)
        transition_matrix = outputs["transition_matrix"].squeeze(0)  # (n_states, n_states)
        target_scores = outputs["target_scores"].squeeze(0)  # (n_proteins,)

        # Parse drug resistance by timepoint
        # Reshape flat output (n_drugs * n_timepoints,) to (n_drugs, n_timepoints)
        drug_probs_np = drug_probs.cpu().numpy()
        drug_probs_reshaped = drug_probs_np.reshape(len(self.drug_names), self.model.n_timepoints)
        drug_resistance_3m = {}
        drug_resistance_6m = {}
        drug_resistance_12m = {}

        for i, drug_name in enumerate(self.drug_names):
            drug_resistance_3m[drug_name] = float(drug_probs_reshaped[i, 0])
            drug_resistance_6m[drug_name] = float(drug_probs_reshaped[i, 1])
            drug_resistance_12m[drug_name] = float(drug_probs_reshaped[i, 2])

        # Handle evidential vs standard state output
        if self.model.use_evidential:
            # state_output is already probs from evidential head
            state_probs = state_output.cpu().numpy()
            alpha = outputs["alpha"].squeeze(0)  # (n_states,)
            epistemic_unc = float(outputs["epistemic_uncertainty"].squeeze(0).cpu())
            aleatoric_unc = float(outputs["aleatoric_uncertainty"].squeeze(0).cpu())
            evidence_strength = float(alpha.sum().cpu())

            # Calibrated confidence: S / (S + K) where S = sum(alpha), K = n_states
            K = float(self.model.n_states)
            calibrated_confidence = evidence_strength / (evidence_strength + K)
        else:
            # Standard softmax over logits
            state_probs = torch.softmax(state_output, dim=0).cpu().numpy()
            epistemic_unc = 0.0
            aleatoric_unc = 0.0
            evidence_strength = 0.0
            calibrated_confidence = float(state_probs[int(np.argmax(state_probs))])

        # Determine current state (argmax)
        current_state_idx = int(np.argmax(state_probs))
        resistance_state = self.state_names[current_state_idx]

        # Determine basin of attraction (column with highest total transition prob)
        trans_np = transition_matrix.cpu().numpy()
        basin_idx = int(np.argmax(trans_np.mean(axis=0)))  # Average column
        basin_of_attraction = self.state_names[basin_idx]

        # Rank intervention targets
        if protein_resistance_scores is not None:
            ranked_targets = self.target_ranker.rank_targets(
                self.protein_names,
                protein_resistance_scores,
                top_k=10,
            )
        else:
            ranked_targets = self.target_ranker.rank_targets(
                self.protein_names,
                target_scores,
                top_k=10,
            )

        return LandscapeResult(
            sample_id=sample_id,
            fused_representation=fused_representation.cpu(),
            drug_resistance_3m=drug_resistance_3m,
            drug_resistance_6m=drug_resistance_6m,
            drug_resistance_12m=drug_resistance_12m,
            top_intervention_targets=ranked_targets,
            resistance_state=resistance_state,
            basin_of_attraction=basin_of_attraction,
            transition_probabilities=transition_matrix.cpu(),
            confidence_score=calibrated_confidence,
            epistemic_uncertainty=epistemic_unc,
            aleatoric_uncertainty=aleatoric_unc,
            evidence_strength=evidence_strength,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        fused_representations: torch.Tensor,
        sample_ids: list[str] | None = None,
        protein_resistance_scores: torch.Tensor | None = None,
    ) -> list[LandscapeResult]:
        """Predict resistance landscape for a batch of samples.

        Args:
            fused_representations: (batch_size, fusion_dim) tensor.
            sample_ids: Optional list of sample identifiers.
            protein_resistance_scores: (batch_size, n_proteins) optional per-protein scores.

        Returns:
            List of LandscapeResult, one per sample.
        """
        batch_size = fused_representations.shape[0]
        if sample_ids is None:
            sample_ids = [None] * batch_size

        # Single vectorized forward pass (expensive GPU operation)
        outputs = self.model(fused_representations.to(self.device))

        # Extract all tensors at once
        drug_probs = outputs["drug_resistance"]  # (batch_size, n_drugs * n_timepoints)
        state_output = outputs["resistance_state"]  # (batch_size, n_states)
        transition_matrix = outputs["transition_matrix"]  # (batch_size, n_states, n_states)
        target_scores = outputs["target_scores"]  # (batch_size, n_proteins)

        # Handle evidential outputs if present
        if self.model.use_evidential:
            alpha = outputs["alpha"]  # (batch_size, n_states)
            epistemic_uncertainty = outputs["epistemic_uncertainty"]  # (batch_size,)
            aleatoric_uncertainty = outputs["aleatoric_uncertainty"]  # (batch_size,)
            K = float(self.model.n_states)
            state_probs = outputs["resistance_state"].cpu().numpy()  # Already normalized from evidential
        else:
            alpha = None
            epistemic_uncertainty = None
            aleatoric_uncertainty = None
            # Compute softmax for standard mode
            state_probs = torch.softmax(state_output, dim=1).cpu().numpy()  # (batch_size, n_states)

        # Lightweight post-processing loop: build results
        results = []
        for i in range(batch_size):
            # Extract drug resistance by timepoint for this sample
            # Reshape flat output (n_drugs * n_timepoints,) to (n_drugs, n_timepoints)
            drug_probs_np = drug_probs[i].cpu().numpy()
            drug_probs_reshaped = drug_probs_np.reshape(len(self.drug_names), self.model.n_timepoints)
            drug_resistance_3m = {}
            drug_resistance_6m = {}
            drug_resistance_12m = {}

            for j, drug_name in enumerate(self.drug_names):
                drug_resistance_3m[drug_name] = float(drug_probs_reshaped[j, 0])
                drug_resistance_6m[drug_name] = float(drug_probs_reshaped[j, 1])
                drug_resistance_12m[drug_name] = float(drug_probs_reshaped[j, 2])

            # Determine current state (argmax)
            current_state_idx = int(np.argmax(state_probs[i]))
            resistance_state = self.state_names[current_state_idx]

            # Determine basin of attraction
            trans_np = transition_matrix[i].cpu().numpy()
            basin_idx = int(np.argmax(trans_np.mean(axis=0)))
            basin_of_attraction = self.state_names[basin_idx]

            # Compute confidence and uncertainties
            if self.model.use_evidential:
                alpha_i = alpha[i].cpu()
                epistemic_unc = float(epistemic_uncertainty[i].cpu())
                aleatoric_unc = float(aleatoric_uncertainty[i].cpu())
                evidence_strength = float(alpha_i.sum().cpu())
                calibrated_confidence = evidence_strength / (evidence_strength + K)
            else:
                epistemic_unc = 0.0
                aleatoric_unc = 0.0
                evidence_strength = 0.0
                calibrated_confidence = float(state_probs[i, current_state_idx])

            # Rank intervention targets (lightweight per-sample operation)
            if protein_resistance_scores is not None:
                prot_scores = protein_resistance_scores[i]
            else:
                prot_scores = target_scores[i]

            ranked_targets = self.target_ranker.rank_targets(
                self.protein_names,
                prot_scores,
                top_k=10,
            )

            result = LandscapeResult(
                sample_id=sample_ids[i],
                fused_representation=fused_representations[i].cpu(),
                drug_resistance_3m=drug_resistance_3m,
                drug_resistance_6m=drug_resistance_6m,
                drug_resistance_12m=drug_resistance_12m,
                top_intervention_targets=ranked_targets,
                resistance_state=resistance_state,
                basin_of_attraction=basin_of_attraction,
                transition_probabilities=transition_matrix[i].cpu(),
                confidence_score=calibrated_confidence,
                epistemic_uncertainty=epistemic_unc,
                aleatoric_uncertainty=aleatoric_unc,
                evidence_strength=evidence_strength,
            )
            results.append(result)

        return results