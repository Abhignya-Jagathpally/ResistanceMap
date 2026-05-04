"""Module 1: Proteome-to-Epigenome Conditional Variational Autoencoder.

Adapted from MyeloMemory/models/vae.py

Given a proteomic profile (protein abundances), infers the latent epigenetic
memory state and reconstructs the expected epigenomic profile (ATAC-seq peaks,
histone modifications).

Architecture:
    Encoder: proteomics → [hidden layers] → (mu, log_var) in R^latent_dim
    Decoder: z ~ N(mu, var) → [hidden layers] → reconstructed epigenomics

Key design choices:
    - GELU activation (smoother gradients than ReLU for biological data)
    - Cyclical KL annealing (prevents posterior collapse)
    - Gradient checkpointing (fits batch_size=512 on H100 80GB)
    - bf16 mixed precision throughout
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from resistancemap.config import VAEConfig
from resistancemap.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


def _iter_parents(root: nn.Module, child_name: str):
    """Yield ancestor modules for a named child in a module hierarchy.

    Given a dotted name like 'decoder.mean_head', yields the modules
    corresponding to 'decoder' — i.e., all intermediate parents between
    *root* and the leaf.
    """
    parts = child_name.split(".")
    current = root
    for part in parts[:-1]:
        current = getattr(current, part)
        yield current


class _GradientReversalFunction(torch.autograd.Function):
    """Autograd function implementing gradient reversal.

    Custom autograd.Function that negates gradients during backpropagation.
    This is necessary because PyTorch's autograd system does not call
    nn.Module.backward() — it only respects torch.autograd.Function.backward().
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Forward pass: identity function.

        Args:
            ctx: Context object for storing values during backward.
            x: Input tensor.
            lambda_: Gradient scaling factor.

        Returns:
            x.clone() to ensure proper gradient tracking.
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass: negate gradients scaled by lambda_.

        Args:
            ctx: Context object with stored lambda_.
            grad_output: Gradient from downstream.

        Returns:
            Tuple of (reversed_grad, None) where None is for lambda_ gradient.
        """
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for domain-adversarial training.

    Implements the gradient reversal technique from Ganin et al. (2016).
    During forward pass, acts as identity. During backward pass, negates
    gradients scaled by lambda_.

    This enables domain-adversarial training by making the latent space
    unable to discriminate between domains, thus learning domain-invariant
    representations.

    Args:
        lambda_: Scaling factor for gradient reversal (default 1.0).
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: apply gradient reversal via autograd.Function.

        Args:
            x: Input tensor.

        Returns:
            Input tensor with gradient reversal applied during backprop.
        """
        return _GradientReversalFunction.apply(x, self.lambda_)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Applies conditional affine transformation to features:
        output = gamma(conditioning) * input + beta(conditioning)

    where gamma and beta are learned functions of conditioning input.
    Enables fine-grained modulation of latent representations conditioned
    on external information (e.g., epigenome features).

    Args:
        conditioning_dim: Dimension of conditioning input.
        feature_dim: Dimension of features to modulate.
    """

    def __init__(self, conditioning_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # Learn gamma and beta from conditioning
        self.fc_gamma = nn.Linear(conditioning_dim, feature_dim)
        self.fc_beta = nn.Linear(conditioning_dim, feature_dim)

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM transformation.

        Args:
            x: (B, feature_dim) input features.
            conditioning: (B, conditioning_dim) conditioning input.

        Returns:
            (B, feature_dim) modulated features.
        """
        gamma = self.fc_gamma(conditioning)
        beta = self.fc_beta(conditioning)
        return gamma * x + beta


class ConditionalDomainDiscriminator(nn.Module):
    """Domain classifier with gradient reversal for adversarial training.

    Takes latent representations and classifies domain (e.g., cell_line vs patient).
    Uses GradientReversalLayer to prevent the encoder from easily discriminating
    between domains, encouraging learning of domain-invariant representations.

    Architecture:
        latent_dim → ReLU → 128 → LeakyReLU → Dropout(0.3)
                  → 64 → LeakyReLU → Dropout(0.3)
                  → n_domains (logits)

    Args:
        latent_dim: Dimension of input latent vectors.
        n_domains: Number of domain classes (default 2 for binary cell_line/patient).
    """

    def __init__(self, latent_dim: int, n_domains: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_domains = n_domains

        # Gradient reversal on input
        self.grl = GradientReversalLayer(lambda_=1.0)

        # MLP architecture
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, n_domains),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Classify domain from latent vector.

        Args:
            z: (B, latent_dim) latent representations.

        Returns:
            (B, n_domains) logits for domain classification.
        """
        z_reversed = self.grl(z)
        return self.mlp(z_reversed)


class _EncoderBlock(nn.Module):
    """Single encoder layer: Linear → BatchNorm → GELU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_bn: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))


class _DecoderBlock(nn.Module):
    """Single decoder layer: Linear → BatchNorm → GELU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_bn: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))


class StochasticDecoder(nn.Module):
    """Stochastic decoder outputting distribution parameters over epigenomics.

    Instead of a deterministic reconstruction, this decoder outputs the parameters
    of a distribution (mean and log-variance) over the epigenomic space. This
    captures the biological reality that multiple chromatin states can be consistent
    with the same proteomics profile.

    The decoder learns to output:
    - A mean vector (epigenome_dim,) representing the expected epigenomic state
    - A log-variance vector (epigenome_dim,) capturing per-feature aleatoric
      uncertainty (model's epistemic uncertainty about what chromatin state
      corresponds to a given proteomics profile)

    Architecture:
        z (latent) → [hidden layers] → h_final
        h_final → mean_head → mean (epigenome_dim,)
        h_final → logvar_head → log_var (epigenome_dim,)

    Args:
        latent_dim: Dimension of input latent vector.
        epigenome_dim: Dimension of output epigenomic space.
        decoder_hidden_dims: List of hidden layer dimensions.
        dropout: Dropout rate.
        use_batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        latent_dim: int,
        epigenome_dim: int,
        decoder_hidden_dims: list[int],
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.epigenome_dim = epigenome_dim

        # Build shared decoder backbone
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.append(
                _DecoderBlock(prev_dim, hidden_dim, dropout, use_batch_norm)
            )
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Output heads for mean and log-variance
        self.mean_head = nn.Linear(prev_dim, epigenome_dim)
        self.logvar_head = nn.Linear(prev_dim, epigenome_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to distribution parameters.

        Args:
            z: (B, latent_dim) latent memory state vector.

        Returns:
            Tuple of (mean, log_var):
            - mean: (B, epigenome_dim) expected epigenomic profile
            - log_var: (B, epigenome_dim) log-variance (aleatoric uncertainty)
        """
        h = self.decoder(z)
        mean = self.mean_head(h)
        log_var = self.logvar_head(h)
        return mean, log_var


class ProteomeToEpigenomeVAE(nn.Module):
    """Conditional VAE mapping proteomics → latent memory state → epigenomics.

    The latent space (64-dim by default, scalable to 256 for expanded capacity)
    represents the inferred epigenetic memory state: a compressed representation
    of the chromatin landscape that would produce the observed proteomic profile.

    Supports optional features:
    - FiLM conditioning: modulate encoder representations with epigenome features
    - Domain-adversarial training: learn domain-invariant latent representations

    Args:
        config: VAEConfig with architecture and training hyperparameters.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        # Use expanded_latent_dim if specified, otherwise use latent_dim
        self.latent_dim = (
            config.expanded_latent_dim
            if config.expanded_latent_dim is not None
            else config.latent_dim
        )
        self.use_stochastic_decoder = getattr(config, 'use_stochastic_decoder', False)
        self.conditioning_dim = getattr(config, 'conditioning_dim', 0)
        self.domain_adversarial = getattr(config, 'domain_adversarial', False)

        # Build encoder
        encoder_layers = []
        prev_dim = config.input_dim
        self.encoder_blocks = nn.ModuleList()

        for hidden_dim in config.encoder_hidden_dims:
            block = _EncoderBlock(prev_dim, hidden_dim, config.dropout, config.use_batch_norm)
            encoder_layers.append(block)
            self.encoder_blocks.append(block)

            # Add FiLM conditioning layer if conditioning_dim > 0
            if self.conditioning_dim > 0:
                encoder_layers.append(FiLMLayer(self.conditioning_dim, hidden_dim))

            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent projections
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, self.latent_dim)

        # Build decoder - either stochastic or deterministic
        if self.use_stochastic_decoder:
            self.decoder = StochasticDecoder(
                latent_dim=self.latent_dim,
                epigenome_dim=config.epigenome_dim,
                decoder_hidden_dims=config.decoder_hidden_dims,
                dropout=config.dropout,
                use_batch_norm=config.use_batch_norm,
            )
            self.output_head = None  # Stochastic decoder has its own heads
        else:
            # Standard deterministic decoder
            decoder_layers = []
            prev_dim = self.latent_dim
            for hidden_dim in config.decoder_hidden_dims:
                decoder_layers.append(
                    _DecoderBlock(prev_dim, hidden_dim, config.dropout, config.use_batch_norm)
                )
                prev_dim = hidden_dim
            self.decoder = nn.Sequential(*decoder_layers)
            self.output_head = nn.Linear(prev_dim, config.epigenome_dim)

        # Domain discriminator for adversarial training (optional)
        if self.domain_adversarial:
            self.domain_discriminator = ConditionalDomainDiscriminator(
                latent_dim=self.latent_dim, n_domains=2
            )
        else:
            self.domain_discriminator = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers.

        Skip StochasticDecoder modules — they handle their own
        logvar_head initialization which must be preserved.
        """
        for name, module in self.named_modules():
            # Skip StochasticDecoder and its children; it initializes itself
            if isinstance(module, StochasticDecoder):
                continue
            if isinstance(module, nn.Linear):
                # Check this Linear is not inside a StochasticDecoder
                is_stochastic_child = any(
                    isinstance(parent, StochasticDecoder)
                    for parent in _iter_parents(self, name)
                )
                if is_stochastic_child:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std.

        Args:
            mu: (B, latent_dim) mean of the approximate posterior.
            log_var: (B, latent_dim) log variance.

        Returns:
            z: (B, latent_dim) sampled latent vector.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic at inference

    def decode(
        self, z: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to reconstructed epigenomics.

        Args:
            z: (B, latent_dim) latent memory state vector.

        Returns:
            If deterministic decoder:
                (B, epigenome_dim) reconstructed epigenomic profile.
            If stochastic decoder:
                Tuple of (mean, log_var), each (B, epigenome_dim).
        """
        if self.use_stochastic_decoder:
            # Stochastic decoder returns (mean, log_var)
            return self.decoder(z)
        else:
            # Deterministic decoder returns reconstruction
            h = self.decoder(z)
            return self.output_head(h)

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Full forward pass: encode → sample → decode.

        Args:
            x: (B, P) protein abundance tensor.
            conditioning: Optional (B, conditioning_dim) conditioning input for FiLM layers.

        Returns:
            If deterministic decoder:
                Tuple of (reconstruction, mu, log_var).
            If stochastic decoder:
                Tuple of (recon_mean, recon_logvar, mu, log_var) where:
                - recon_mean, recon_logvar: distribution parameters over epigenomics
                - mu, log_var: latent distribution parameters
        """
        mu, log_var = self.encode(x, conditioning)
        z = self.reparameterize(mu, log_var)
        decoder_out = self.decode(z)

        if self.use_stochastic_decoder:
            recon_mean, recon_logvar = decoder_out
            return recon_mean, recon_logvar, mu, log_var
        else:
            recon = decoder_out
            return recon, mu, log_var

    def encode(
        self, x: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode proteomics to latent distribution parameters.

        Args:
            x: (B, P) protein abundance tensor.
            conditioning: Optional (B, conditioning_dim) conditioning input for FiLM.

        Returns:
            Tuple of (mu, log_var), each (B, latent_dim).
        """
        if self.conditioning_dim > 0 and conditioning is None:
            raise ValueError(
                f"conditioning required (conditioning_dim={self.conditioning_dim}) "
                "but not provided"
            )

        if self.config.gradient_checkpointing and self.training:
            if self.conditioning_dim > 0:
                # Checkpointing with conditioning
                h = torch.utils.checkpoint.checkpoint(
                    self._encode_with_conditioning,
                    x, conditioning,
                    use_reentrant=False
                )
            else:
                h = torch.utils.checkpoint.checkpoint(
                    self.encoder, x, use_reentrant=False
                )
        else:
            if self.conditioning_dim > 0:
                h = self._encode_with_conditioning(x, conditioning)
            else:
                h = self.encoder(x)

        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def _encode_with_conditioning(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Helper for encoding with FiLM conditioning.

        Args:
            x: (B, P) protein abundance tensor.
            conditioning: (B, conditioning_dim) conditioning input.

        Returns:
            (B, hidden_dim) encoded representation.
        """
        h = x
        film_idx = 0
        for layer in self.encoder:
            if isinstance(layer, FiLMLayer):
                h = layer(h, conditioning)
                film_idx += 1
            else:
                h = layer(h)
        return h

    def domain_classify(self, z: torch.Tensor) -> torch.Tensor:
        """Classify domain from latent representation.

        Args:
            z: (B, latent_dim) latent vector.

        Returns:
            (B, 2) logits for domain classification (if domain_adversarial=True).

        Raises:
            RuntimeError: If model was not initialized with domain_adversarial=True.
        """
        if self.domain_discriminator is None:
            raise RuntimeError(
                "domain_classify() requires model to be initialized with "
                "domain_adversarial=True in config"
            )
        return self.domain_discriminator(z)

    def get_memory_state(self, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """Extract the latent memory state (mu) without sampling.

        This is the primary output used by downstream modules.

        Args:
            x: (B, P) protein abundance tensor.
            conditioning: Optional (B, conditioning_dim) conditioning input for FiLM.

        Returns:
            (B, latent_dim) deterministic memory state embedding.
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x, conditioning)
        return mu

    def get_epigenome_reconstruction_with_uncertainty(
        self, x: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get epigenomic reconstruction with aleatoric uncertainty estimates.

        Only available when using stochastic decoder. Returns the distribution
        parameters p(epigenome | proteome).

        Args:
            x: (B, P) protein abundance tensor.
            conditioning: Optional (B, conditioning_dim) conditioning input for FiLM.

        Returns:
            If stochastic decoder:
                Tuple of (mean, log_var):
                - mean: (B, epigenome_dim) expected epigenomic profile
                - log_var: (B, epigenome_dim) log-variance (aleatoric uncertainty)
            If deterministic decoder:
                Returns (reconstruction, zeros) with deterministic reconstruction

        Raises:
            RuntimeError: If called on model without stochastic decoder enabled.
        """
        if not self.use_stochastic_decoder:
            raise RuntimeError(
                "get_epigenome_reconstruction_with_uncertainty() requires "
                "use_stochastic_decoder=True in config"
            )

        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x, conditioning)
            z = self.reparameterize(mu, torch.zeros_like(mu))  # Use mean for deterministic path
            recon_mean, recon_logvar = self.decoder(z)

        return recon_mean, recon_logvar


def _kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL divergence from N(mu, var) to N(0, I).

    Returns:
        Scalar KL loss (mean over batch).
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def _stochastic_reconstruction_loss(
    target: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood loss for stochastic decoder.

    Assumes a Gaussian distribution over the reconstruction:
        p(x | z) = N(x | mean, diag(var))
        NLL = 0.5 * (log(var) + (x - mean)^2 / var)

    This captures aleatoric uncertainty: the model's uncertainty about which
    epigenomic state corresponds to the proteomics profile.

    Args:
        target: (B, D) target epigenomic profile.
        mean: (B, D) predicted mean of reconstruction distribution.
        log_var: (B, D) predicted log-variance (per-feature aleatoric uncertainty).

    Returns:
        Scalar NLL loss (mean over batch and features).
    """
    var = torch.exp(log_var)
    # Clip variance to avoid numerical issues
    var = torch.clamp(var, min=1e-8)
    nll = 0.5 * (log_var + (target - mean) ** 2 / var)
    return torch.mean(nll)


def _cyclical_kl_weight(
    step: int, total_steps: int, n_cycles: int, ratio: float, max_weight: float
) -> float:
    """Compute cyclical KL annealing weight (β-annealing schedule).

    Implements the cyclical annealing schedule from Fu et al. (2019) for the
    β-VAE framework. The KL divergence term is weighted by a schedule that
    cycles between 0 and max_weight, preventing posterior collapse by allowing
    the model to initially focus on reconstruction, then gradually enforcing
    the KL constraint.

    β-VAE interpretation:
    - β controls the information bottleneck: lower β → more latent capacity
    - max_weight (β) drives toward a more regularized latent space
    - Annealing schedule: allows model to learn good reconstruction before
      enforcing distributional constraints

    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        n_cycles: Number of annealing cycles (e.g., 4 for 4 ramps up and plateaus).
        ratio: Fraction of each cycle spent in the ramp-up phase (e.g., 0.5).
        max_weight: Maximum KL weight β. Controls information bottleneck tightness.
                   Common values: 0.1 (loose), 1.0 (standard VAE), 10.0 (tight).

    Returns:
        KL weight (β) for the current step, in range [0, max_weight].
    """
    cycle_length = total_steps // n_cycles
    position_in_cycle = step % cycle_length
    ramp_length = int(cycle_length * ratio)

    if position_in_cycle < ramp_length:
        return max_weight * (position_in_cycle / ramp_length)
    return max_weight


def extract_latent_trajectory(
    model: ProteomeToEpigenomeVAE,
    proteomics_timeseries: torch.Tensor,
) -> torch.Tensor:
    """Extract latent trajectory from a time series of proteomics snapshots.

    Takes a sequence of proteomics measurements over time and encodes each
    into the VAE latent space, producing a trajectory in the 64-dimensional
    memory state space.

    Args:
        model: Trained ProteomeToEpigenomeVAE model.
        proteomics_timeseries: (T, P) tensor of protein abundances over T timepoints.
                               Can also be (B, T, P) for batch processing.

    Returns:
        Latent trajectory: (T, 64) or (B, T, 64) memory state embeddings.
        For 2D input, returns (T, latent_dim).
        For 3D input, returns (B, T, latent_dim).
    """
    model.eval()

    input_shape = proteomics_timeseries.shape
    is_batched = len(input_shape) == 3

    if is_batched:
        batch_size, time_steps, n_proteins = input_shape
        # Reshape to (B*T, P)
        flat_input = proteomics_timeseries.view(batch_size * time_steps, n_proteins)
    else:
        time_steps, n_proteins = input_shape
        flat_input = proteomics_timeseries

    with torch.no_grad():
        # Encode all timepoints at once
        mu, _ = model.encode(flat_input)

    # Reshape back to trajectory form
    if is_batched:
        trajectory = mu.view(batch_size, time_steps, model.latent_dim)
    else:
        trajectory = mu.view(time_steps, model.latent_dim)

    logger.info(
        f"Extracted latent trajectory with shape {trajectory.shape} "
        f"(latent_dim={model.latent_dim})"
    )

    return trajectory


def train_vae(
    model: nn.Module,
    dataset: Any,
    splits: dict[str, list[int]],
    config: VAEConfig,
    subset: str,
    ckpt_mgr: CheckpointManager,
    stage_name: str,
) -> dict[str, Any]:
    """Train the VAE on specified data subset.

    Args:
        model: ProteomeToEpigenomeVAE (possibly DDP-wrapped).
        dataset: MultiOmicsDataset.
        splits: Train/val/test index splits.
        config: VAE hyperparameters.
        subset: 'pan_cancer' or 'hematological'.
        ckpt_mgr: Checkpoint manager for saving.
        stage_name: Name for the checkpoint file.

    Returns:
        Dict with 'checkpoint_path' and 'metrics'.
    """
    # Select subset
    if subset == "hematological":
        heme_lineages = {
            "Myeloid", "Lymphoid",
            "haematopoietic_and_lymphoid_tissue",
            "blood", "lymphocyte", "plasma_cell",
        }
        indices = [
            i for i in splits["train"]
            if dataset.lineage[i] in heme_lineages
        ]
        val_indices = [
            i for i in splits["val"]
            if dataset.lineage[i] in heme_lineages
        ]
    else:
        indices = splits["train"]
        val_indices = splits["val"]

    epochs = config.pretrain_epochs if subset == "pan_cancer" else config.finetune_epochs
    lr = config.pretrain_lr if subset == "pan_cancer" else config.finetune_lr

    effective_batch_size = min(config.batch_size, len(indices))
    train_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=min(8, 2 if len(indices) < 100 else 8),
        pin_memory=True,
        drop_last=len(indices) > effective_batch_size,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda")

    total_steps = epochs * max(len(train_loader), 1)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    kl_weight = 0.0

    device = next(model.parameters()).device

    # Check if model has domain discriminator and can use adversarial training
    has_domain_discriminator = hasattr(model, 'domain_discriminator') and model.domain_discriminator is not None
    adv_weight = getattr(config, 'domain_adversarial_weight', 0.1)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_adv = 0.0

        for batch in train_loader:
            proteomics = batch["proteomics"].to(device, non_blocking=True)
            epigenomics = batch["epigenomics"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):
                model_output = model(proteomics)

                # Handle both deterministic and stochastic decoder outputs
                if len(model_output) == 4:
                    # Stochastic decoder: (recon_mean, recon_logvar, mu, log_var)
                    recon_mean, recon_logvar, mu, log_var = model_output
                    recon_loss = _stochastic_reconstruction_loss(
                        epigenomics, recon_mean, recon_logvar
                    )
                else:
                    # Deterministic decoder: (recon, mu, log_var)
                    recon, mu, log_var = model_output
                    recon_loss = F.mse_loss(recon, epigenomics)

                kl_loss = _kl_divergence(mu, log_var)

                kl_weight = _cyclical_kl_weight(
                    global_step, total_steps,
                    config.kl_anneal_cycles,
                    config.kl_anneal_ratio,
                    config.kl_weight_max,
                )
                loss = recon_loss + kl_weight * kl_loss

                # Domain-adversarial loss (if enabled and domain labels available)
                domain_loss = 0.0
                if has_domain_discriminator and "domain_labels" in batch:
                    domain_labels = batch["domain_labels"].to(device, non_blocking=True)
                    domain_logits = model.domain_classify(mu)
                    domain_loss = F.cross_entropy(domain_logits, domain_labels)
                    loss = loss + adv_weight * domain_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            if domain_loss != 0.0:
                epoch_adv += domain_loss.item() if isinstance(domain_loss, torch.Tensor) else domain_loss
            global_step += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                proteomics = batch["proteomics"].to(device, non_blocking=True)
                epigenomics = batch["epigenomics"].to(device, non_blocking=True)

                with autocast("cuda", dtype=torch.bfloat16):
                    model_output = model(proteomics)

                    # Handle both deterministic and stochastic decoder outputs
                    if len(model_output) == 4:
                        # Stochastic decoder: (recon_mean, recon_logvar, mu, log_var)
                        recon_mean, recon_logvar, mu, log_var = model_output
                        recon_loss = _stochastic_reconstruction_loss(
                            epigenomics, recon_mean, recon_logvar
                        )
                    else:
                        # Deterministic decoder: (recon, mu, log_var)
                        recon, mu, log_var = model_output
                        recon_loss = F.mse_loss(recon, epigenomics)

                    kl_loss = _kl_divergence(mu, log_var)
                    val_loss += (recon_loss + kl_loss).item()

        avg_train = epoch_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        log_msg = (
            f"[{stage_name}] Epoch {epoch + 1}/{epochs} — "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"recon={epoch_recon / max(len(train_loader), 1):.4f} "
            f"kl={epoch_kl / max(len(train_loader), 1):.4f} "
            f"kl_weight={kl_weight:.4f}"
        )
        if has_domain_discriminator and epoch_adv > 0.0:
            log_msg += f" adv={epoch_adv / max(len(train_loader), 1):.4f}"
        logger.info(log_msg)

        # Early stopping + checkpoint best
        if avg_val < best_val_loss - config.min_delta:
            best_val_loss = avg_val
            patience_counter = 0

            # Save best checkpoint
            raw_model = model.module if hasattr(model, "module") else model
            ckpt_mgr.save(stage_name, {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_val_loss,
            })
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    metrics = {
        "best_val_loss": best_val_loss,
        "final_epoch": epoch + 1,
        "total_steps": global_step,
    }

    return {
        "checkpoint_path": ckpt_mgr.path(stage_name),
        "metrics": metrics,
    }
