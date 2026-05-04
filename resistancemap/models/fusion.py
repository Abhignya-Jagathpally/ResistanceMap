"""L4 Multi-Modal Fusion layer for ResistanceMap.

Combines four modalities into a unified representation for landscape prediction:
    - Epigenetic state (64-dim)
    - Trajectory state (64-dim)
    - Protein network output (256-dim)
    - Stability score (1-dim)

Supports two fusion modes:
    1. Cross-attention (default): Multi-head cross-modal attention with gating
    2. Tensor fusion (alternative): Outer product fusion

Components sourced from:
    - CrossModalAttentionBlock: r4/pipeline4/models/attention_fusion.py (multi-head + FFN + LayerNorm)
    - CrossModalFusionNet: r4/pipeline4/models/attention_fusion.py (adapted for 4 modalities)
    - TensorFusion: r4/graph_ml/fusion_model.py (outer product fusion)

Enhancements:
    - ResidualBatchCorrector: Learns batch-specific residual corrections
    - Missing modality handling: Automatic detection and fallback embeddings
    - Configurable modality dimensions: Supports expanded latent spaces
    - Domain adaptation: Optional MMD-based alignment between domains
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResidualBatchCorrector(nn.Module):
    """Learns batch-specific residual corrections for features.

    Corrects batch effects by learning a per-batch residual correction.
    Uses batch embeddings combined with linear residual computation.

    Args:
        feature_dim: Dimension of input features.
        n_batches: Number of distinct batches.
        embedding_dim: Dimension of batch embeddings (default: feature_dim // 2).
    """

    def __init__(
        self,
        feature_dim: int,
        n_batches: int,
        embedding_dim: int | None = None,
    ):
        """Initialize residual batch corrector.

        Args:
            feature_dim: Feature dimension.
            n_batches: Number of batches.
            embedding_dim: Embedding dimension for batch representation.
        """
        super().__init__()
        if embedding_dim is None:
            embedding_dim = max(8, feature_dim // 2)

        self.feature_dim = feature_dim
        self.n_batches = n_batches
        self.embedding_dim = embedding_dim

        # Batch embeddings
        self.batch_embeddings = nn.Embedding(n_batches, embedding_dim)

        # Linear transformation from batch embedding to residual
        self.residual_proj = nn.Linear(embedding_dim, feature_dim)

        # Scaling factor for residual contribution
        self.register_parameter(
            "residual_scale",
            nn.Parameter(torch.ones(1) * 0.1),
        )

        logger.info(
            f"Initialized ResidualBatchCorrector: "
            f"feature_dim={feature_dim}, n_batches={n_batches}, embedding_dim={embedding_dim}"
        )

    def forward(
        self,
        features: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply batch correction to features.

        Args:
            features: (batch, feature_dim) feature tensor.
            batch_ids: (batch,) batch identifiers (0 to n_batches-1).

        Returns:
            Tensor of shape (batch, feature_dim) with batch correction applied.
        """
        # Get batch embeddings
        batch_embeds = self.batch_embeddings(batch_ids)  # (batch, embedding_dim)

        # Compute residual correction
        residual = self.residual_proj(batch_embeds)  # (batch, feature_dim)

        # Apply scaled correction
        corrected = features + self.residual_scale * residual

        return corrected


class CrossModalAttentionBlock(nn.Module):
    """Multi-head cross-attention between query and key-value modalities (from R4).

    Architecture:
        - Multi-head attention (query vs key/value)
        - Residual connection + LayerNorm
        - Feed-forward network + residual + LayerNorm

    Adapted from: r4/pipeline4/models/attention_fusion.py (CrossModalAttentionBlock)

    Args:
        dim: Feature dimension (all modalities projected to this).
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        """Initialize cross-modal attention block.

        Args:
            dim: Feature dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: Query tensor (batch, seq_len, dim) or (batch, 1, dim) for single-token.
            key_value: Key and value tensor (batch, seq_len, dim) or concatenation of modalities.

        Returns:
            Tuple of:
                - attended: (batch, seq_len, dim) attended features with residuals
                - attn_weights: (batch, heads, seq_len, seq_len) attention weights
        """
        attn_out, attn_weights = self.attention(query, key_value, key_value)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class CrossModalFusionNet(nn.Module):
    """Cross-modal fusion network for multiple modalities (adapted from R4).

    Architecture:
        - Per-modality projection to common hidden_dim
        - Cross-attention blocks (each modality attends to others)
        - Gated fusion (learned weights for each modality)
        - Output projection to latent space

    Features:
        - Missing modality detection and fallback embeddings
        - Optional domain adaptation via MMD loss
        - Automatic masking of missing (all-zero) modalities

    Adapted from: r4/pipeline4/models/attention_fusion.py (CrossModalFusionNet)

    Args:
        modality_dims: Dict mapping modality name to input dimension.
        hidden_dim: Common hidden dimension for all modalities.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        output_dim: Final output dimension (default: hidden_dim).
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        output_dim: int | None = None,
    ):
        """Initialize cross-modal fusion network.

        Args:
            modality_dims: Dict of modality name → input dimension.
            hidden_dim: Common hidden dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            output_dim: Output dimension (default: hidden_dim).
        """
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim

        self.modality_names = sorted(modality_dims.keys())
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Per-modality projection to hidden_dim
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for name, dim in modality_dims.items()
        })

        # Default embeddings for missing modalities
        self.default_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
            for name in self.modality_names
        })

        # Cross-attention blocks (each modality attends to others)
        self.cross_attns = nn.ModuleDict({
            name: CrossModalAttentionBlock(hidden_dim, n_heads, dropout)
            for name in self.modality_names
        })

        # Gated fusion: learned weights for each modality
        n_mod = len(self.modality_names)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * n_mod, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mod),
            nn.Softmax(dim=-1),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        logger.info(
            f"Initialized CrossModalFusionNet: "
            f"modalities={list(modality_dims.keys())}, "
            f"hidden_dim={hidden_dim}, output_dim={output_dim}"
        )

    def _is_missing_modality(
        self,
        modality_tensor: torch.Tensor,
        threshold: float = 1e-6,
    ) -> bool:
        """Check if a modality is missing (all-zero or near-zero).

        Args:
            modality_tensor: Tensor to check.
            threshold: Threshold for considering a value as zero.

        Returns:
            True if modality is missing, False otherwise.
        """
        return (modality_tensor.abs().max() < threshold).item()

    @staticmethod
    def compute_mmd_loss(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        kernel: str = "rbf",
        bandwidth: float = 1.0,
    ) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy (MMD) loss between domains.

        Used for domain adaptation to align source and target distributions.

        Args:
            source_features: (batch, dim) features from source domain.
            target_features: (batch, dim) features from target domain.
            kernel: Kernel type ("rbf" or "linear").
            bandwidth: Bandwidth for RBF kernel.

        Returns:
            Scalar tensor representing MMD loss.
        """
        def compute_kernel(x, y, kernel_type="rbf", bw=1.0):
            """Compute kernel matrix between x and y."""
            if kernel_type == "rbf":
                pairwise_sq_dist = torch.cdist(x, y) ** 2
                return torch.exp(-pairwise_sq_dist / (2 * bw ** 2))
            elif kernel_type == "linear":
                return torch.mm(x, y.t())
            else:
                raise ValueError(f"Unknown kernel: {kernel_type}")

        # Compute kernel matrices
        Kxx = compute_kernel(source_features, source_features, kernel, bandwidth)
        Kyy = compute_kernel(target_features, target_features, kernel, bandwidth)
        Kxy = compute_kernel(source_features, target_features, kernel, bandwidth)

        # MMD^2 = E[K(X,X)] - 2*E[K(X,Y)] + E[K(Y,Y)]
        mmd_sq = Kxx.mean() - 2 * Kxy.mean() + Kyy.mean()

        return torch.clamp(mmd_sq, min=0.0).sqrt()

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        modality_mask: dict[str, bool] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through cross-modal fusion.

        Args:
            modalities: Dict mapping modality name to tensor (batch, modality_dim).
            modality_mask: Optional dict of modality_name -> is_present (bool).
                          If None, automatically detects missing modalities.

        Returns:
            Tuple of:
                - fused: (batch, output_dim) fused representation
                - attn_weights: Dict mapping modality name to attention weights
        """
        batch_size = modalities[self.modality_names[0]].shape[0]

        # Detect missing modalities if not provided
        if modality_mask is None:
            modality_mask = {
                name: not self._is_missing_modality(modalities[name])
                for name in self.modality_names
            }

        # 1. Project each modality to hidden_dim, using defaults for missing ones
        projected = {}
        for name in self.modality_names:
            if modality_mask[name]:
                # Modality is present, project it
                x = self.projections[name](modalities[name])  # (batch, hidden)
            else:
                # Modality is missing, use default embedding
                x = self.default_embeddings[name].expand(batch_size, -1)  # (batch, hidden)
            projected[name] = x.unsqueeze(1)  # (batch, 1, hidden)

        # 2. Cross-attention: each modality attends to concatenation of others
        attended = {}
        attn_weights = {}
        for name in self.modality_names:
            # Concatenate all modalities except current one
            others = torch.cat(
                [projected[n] for n in self.modality_names if n != name], dim=1,
            )  # (batch, n_modalities-1, hidden)

            out, weights = self.cross_attns[name](projected[name], others)
            attended[name] = out.squeeze(1)  # (batch, hidden)
            attn_weights[name] = weights

        # 3. Gated fusion: learned combination of attended modalities
        concat = torch.cat([attended[n] for n in self.modality_names], dim=-1)  # (batch, hidden * n_mod)
        gates = self.gate(concat)  # (batch, n_modalities)

        # Apply masking to gates for missing modalities (non-inplace to preserve gradients)
        mask = torch.ones_like(gates)
        for i, name in enumerate(self.modality_names):
            if not modality_mask[name]:
                mask[:, i] = 0.0
        gates = gates * mask

        # Renormalize gates after masking
        gate_sum = gates.sum(dim=-1, keepdim=True)
        gate_sum = torch.clamp(gate_sum, min=1e-8)
        gates = gates / gate_sum

        fused = torch.zeros_like(attended[self.modality_names[0]])  # (batch, hidden)
        for i, name in enumerate(self.modality_names):
            fused += gates[:, i:i+1] * attended[name]

        # 4. Output projection
        output = self.output_proj(fused)

        return output, attn_weights


class TensorFusion(nn.Module):
    """Tensor outer product fusion for multi-modal representations (from R4).

    Computes outer product of modality pairs and flattens for downstream processing.
    Adapted from: r4/graph_ml/fusion_model.py (TensorFusion with correct einsum)

    Args:
        modality_dims: Dict mapping modality name to input dimension.
        hidden_dim: Intermediate hidden dimension.
        output_dim: Output dimension (default: hidden_dim).
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        hidden_dim: int = 128,
        output_dim: int | None = None,
    ):
        """Initialize tensor fusion.

        Args:
            modality_dims: Dict of modality name → input dimension.
            hidden_dim: Hidden dimension for fusion.
            output_dim: Output dimension (default: hidden_dim).
        """
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim

        self.modality_names = sorted(modality_dims.keys())
        self.hidden_dim = hidden_dim

        # Projections for each modality
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in modality_dims.items()
        })

        # Fused dimension: outer product of all modalities flattened
        n_mod = len(modality_dims)
        self.fused_dim = hidden_dim ** n_mod if n_mod <= 2 else hidden_dim * n_mod  # Avoid explosion for >2 modalities

        # Fusion FC layers
        self.fc = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

        logger.info(
            f"Initialized TensorFusion: "
            f"modalities={list(modality_dims.keys())}, "
            f"fused_dim={self.fused_dim}, output_dim={output_dim}"
        )

    def forward(self, modalities: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through tensor fusion.

        Args:
            modalities: Dict mapping modality name to tensor (batch, modality_dim).

        Returns:
            Tensor of shape (batch, output_dim).
        """
        # Project modalities
        projected = {
            name: self.projections[name](modalities[name])
            for name in self.modality_names
        }  # Each (batch, hidden)

        # Outer product for first two modalities, then concatenate others
        if len(self.modality_names) == 2:
            # Outer product for pairwise fusion
            m1, m2 = [projected[n] for n in self.modality_names]
            fused = torch.einsum("bi,bj->bij", m1, m2)  # (batch, hidden, hidden)
            fused = fused.reshape(fused.shape[0], -1)  # (batch, hidden^2)
        else:
            # For >2 modalities, concatenate projected features
            fused = torch.cat([projected[n] for n in self.modality_names], dim=-1)  # (batch, hidden * n_mod)

        return self.fc(fused)


class ResistanceMapFusion(nn.Module):
    """Multi-modal fusion for ResistanceMap landscape prediction (L4).

    Combines four modalities into unified representation:
        - epigenetic_state: (batch, 64) [or configurable]
        - trajectory_state: (batch, 64) [or configurable]
        - protein_network_output: (batch, 256) [or configurable]
        - stability_score: (batch, 1) [or configurable]

    Supports two fusion modes:
        1. "cross_attention" (default): Multi-head cross-attention with gating
        2. "tensor": Outer product tensor fusion
        3. "concat": Simple concatenation

    Features:
        - Optional batch correction via ResidualBatchCorrector
        - Configurable modality dimensions
        - Automatic missing modality handling

    Args:
        hidden_dim: Hidden dimension for fusion layers.
        output_dim: Output dimension for landscape predictor.
        fusion_type: Fusion mode ("cross_attention", "tensor", "concat").
        n_heads: Number of attention heads (for cross-attention mode).
        dropout: Dropout probability.
        modality_dims: Optional dict to override default modality dimensions.
        batch_correction: Whether to enable batch correction (default False).
        n_batches: Number of batches (only used if batch_correction=True).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 128,
        fusion_type: str = "cross_attention",
        n_heads: int = 4,
        dropout: float = 0.2,
        modality_dims: dict[str, int] | None = None,
        batch_correction: bool = False,
        n_batches: int | None = None,
    ):
        """Initialize ResistanceMap fusion layer.

        Args:
            hidden_dim: Hidden dimension for intermediate layers.
            output_dim: Output dimension.
            fusion_type: "cross_attention" (default), "tensor", or "concat".
            n_heads: Number of attention heads (cross-attention only).
            dropout: Dropout probability.
            modality_dims: Override default modality dimensions. Dict of name -> dim.
            batch_correction: Enable batch-specific residual correction.
            n_batches: Number of batches (required if batch_correction=True).
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.output_dim = output_dim
        self.batch_correction = batch_correction

        # Define modalities (use provided or defaults)
        if modality_dims is None:
            self.modality_dims = {
                "epigenetic": 64,
                "trajectory": 64,
                "protein_network": 256,
                "stability": 1,
            }
        else:
            self.modality_dims = modality_dims.copy()

        # Initialize batch correctors if needed
        self.batch_correctors = None
        if batch_correction:
            if n_batches is None:
                raise ValueError("n_batches must be provided if batch_correction=True")
            self.batch_correctors = nn.ModuleDict({
                name: ResidualBatchCorrector(
                    feature_dim=dim,
                    n_batches=n_batches,
                )
                for name, dim in self.modality_dims.items()
            })

        # Initialize fusion layers based on type
        if fusion_type == "cross_attention":
            self.fusion = CrossModalFusionNet(
                modality_dims=self.modality_dims,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                output_dim=output_dim,
            )
        elif fusion_type == "tensor":
            self.fusion = TensorFusion(
                modality_dims=self.modality_dims,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )
        elif fusion_type == "concat":
            # Concatenation fusion
            total_dim = sum(self.modality_dims.values())
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose 'cross_attention', 'tensor', or 'concat'.")

        logger.info(
            f"Initialized ResistanceMapFusion: "
            f"fusion_type={fusion_type}, hidden_dim={hidden_dim}, output_dim={output_dim}, "
            f"batch_correction={batch_correction}"
        )

    def forward(
        self,
        epigenetic_state: torch.Tensor,
        trajectory_state: torch.Tensor,
        protein_network_output: torch.Tensor,
        stability_score: torch.Tensor,
        batch_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through multi-modal fusion.

        Args:
            epigenetic_state: (batch, epigenetic_dim) epigenetic features from VAE.
            trajectory_state: (batch, trajectory_dim) trajectory features from temporal model.
            protein_network_output: (batch, protein_network_dim) protein network propagation output.
            stability_score: (batch, stability_dim) stability score from ODE model.
            batch_ids: Optional (batch,) batch identifiers for batch correction.

        Returns:
            Dict with keys:
                - 'fused_representation': (batch, output_dim) unified representation
                - 'attention_weights': (if cross_attention mode) dict of attention matrices
        """
        # Validate input shapes
        batch_size = epigenetic_state.shape[0]
        assert trajectory_state.shape[0] == batch_size
        assert protein_network_output.shape[0] == batch_size
        assert stability_score.shape[0] == batch_size

        # Prepare modality dict
        modalities = {
            "epigenetic": epigenetic_state,
            "trajectory": trajectory_state,
            "protein_network": protein_network_output,
            "stability": stability_score,
        }

        # Apply batch correction if enabled
        if self.batch_correction and batch_ids is not None:
            for name in modalities.keys():
                if name in self.batch_correctors:
                    modalities[name] = self.batch_correctors[name](
                        modalities[name], batch_ids
                    )

        # Fusion
        if self.fusion_type == "cross_attention":
            fused, attn_weights = self.fusion(modalities)
            return {
                "fused_representation": fused,
                "attention_weights": attn_weights,
            }
        elif self.fusion_type == "tensor":
            fused = self.fusion(modalities)
            return {
                "fused_representation": fused,
            }
        elif self.fusion_type == "concat":
            # Concatenation mode
            concat_features = torch.cat(
                [modalities[name] for name in sorted(modalities.keys())], dim=-1
            )
            fused = self.fusion(concat_features)
            return {
                "fused_representation": fused,
            }

    @property
    def modality_info(self) -> dict[str, int]:
        """Return modality dimensions."""
        return self.modality_dims.copy()

    def get_fusion_info(self) -> str:
        """Get human-readable fusion configuration."""
        return (
            f"ResistanceMapFusion(type={self.fusion_type}, "
            f"modalities={list(self.modality_dims.keys())}, "
            f"output_dim={self.output_dim}, "
            f"batch_correction={self.batch_correction})"
        )


class MultiModalFusionPipeline(nn.Module):
    """High-level pipeline for integrated multi-modal fusion.

    Combines protein network propagator output with other modalities
    and produces landscape-ready representation.

    Args:
        hidden_dim: Hidden dimension.
        output_dim: Output dimension for landscape prediction.
        fusion_type: Type of fusion ("cross_attention", "tensor", "concat").
        modality_dims: Optional dict to override default modality dimensions.
        batch_correction: Whether to enable batch correction (default False).
        n_batches: Number of batches (required if batch_correction=True).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 128,
        fusion_type: str = "cross_attention",
        modality_dims: dict[str, int] | None = None,
        batch_correction: bool = False,
        n_batches: int | None = None,
    ):
        """Initialize multi-modal fusion pipeline.

        Args:
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            fusion_type: Fusion type.
            modality_dims: Optional override for modality dimensions.
            batch_correction: Enable batch correction.
            n_batches: Number of batches (required if batch_correction=True).
        """
        super().__init__()
        self.fusion_layer = ResistanceMapFusion(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            fusion_type=fusion_type,
            modality_dims=modality_dims,
            batch_correction=batch_correction,
            n_batches=n_batches,
        )

    def forward(
        self,
        epigenetic_state: torch.Tensor,
        trajectory_state: torch.Tensor,
        protein_network_output: torch.Tensor,
        stability_score: torch.Tensor,
        batch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            epigenetic_state: (batch, epigenetic_dim) epigenetic features.
            trajectory_state: (batch, trajectory_dim) trajectory features.
            protein_network_output: (batch, protein_network_dim) protein network output.
            stability_score: (batch, stability_dim) stability scores.
            batch_ids: Optional (batch,) batch identifiers for batch correction.

        Returns:
            Tensor of shape (batch, output_dim) ready for landscape prediction.
        """
        result = self.fusion_layer(
            epigenetic_state,
            trajectory_state,
            protein_network_output,
            stability_score,
            batch_ids=batch_ids,
        )
        return result["fused_representation"]