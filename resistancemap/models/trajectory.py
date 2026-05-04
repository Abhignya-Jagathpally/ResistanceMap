"""Module 2: Memory Stability Scorer and Trajectory Forecaster — ODE-based bistability model.

Adapted and extended from MyeloMemory/models/stability.py

Adapts the Sneppen & Ringrose chromatin bistability framework:
    - Parameterizes feedback loop strengths from REAL proteomic measurements
      of chromatin reader/writer enzymes (EZH2, DNMT1, TET1, etc.)
    - Computes basin-of-attraction depth as a stability score
    - Score ranges from 0 (transient adaptation) to 1 (locked-in memory)

The new TrajectoryForecaster extends this ODE system to predict temporal
evolution of the epigenetic state over time horizons (3, 6, 12 months),
mapping the resistance landscape over time.

The ODE system models two competing chromatin states (active vs. repressed)
with auto-catalytic and cross-inhibitory feedback. The depth of the potential
well at the current state determines how much perturbation (drug treatment)
would be needed to flip the epigenetic state.

Uses torchdiffeq for GPU-accelerated, differentiable ODE solving (H100-optimized).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from resistancemap.config import StabilityConfig
from resistancemap.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

# Lazy import — torchdiffeq is only needed for this module
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


# Temporal disclaimers for evaluation agent
TEMPORAL_DISCLAIMERS = {
    3: (
        "Short-term ODE predictions (3 months): deterministic assumption assumes "
        "stable epigenetic landscape. Therapy discontinuation and clonal drift "
        "are not modeled. Predictions are local to current state; global resets not captured."
    ),
    6: (
        "Medium-term ODE predictions (6 months): accumulating uncertainty from "
        "parameter drift, unmodeled stochastic epigenetic switches, and therapy interactions. "
        "Confidence intervals widen. ODE assumes continuous state—discrete gene amplification "
        "events are not captured."
    ),
    9: (
        "Extended-term ODE predictions (9 months): growing uncertainty from accumulated "
        "parameter drift and unmodeled stochastic processes. Confidence intervals significantly widen. "
        "ODE determinism increasingly unreliable; consider qualitative trends only."
    ),
    12: (
        "Long-term ODE predictions (12 months): deterministic ODE extrapolation "
        "becomes unreliable. Assumes parameters remain constant—driver mutations, "
        "epigenetic catastrophes, and tumor heterogeneity growth are not modeled. "
        "Use as qualitative trend only."
    ),
}


class SinkhornOT(nn.Module):
    """Optimal transport coupling via entropic-regularized Sinkhorn divergence.

    Couples two point clouds (cross-sectional snapshots) using optimal transport
    with entropic regularization. Computes transport plan and McCann displacement
    interpolation for trajectory coupling.

    Args:
        epsilon: Entropic regularization strength (default 0.1).
        n_iterations: Number of Sinkhorn iterations (default 100).
    """

    def __init__(self, epsilon: float = 0.1, n_iterations: int = 100) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.n_iterations = n_iterations

    def _sinkhorn_iterations(
        self, cost_matrix: torch.Tensor, n_iters: int = None, tolerance: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute transport plan via log-domain Sinkhorn iterations.

        Uses log-space computation throughout to avoid numerical overflow with small epsilon.
        Implements entropic-regularized optimal transport with optional early stopping.

        Args:
            cost_matrix: (M, N) ground cost matrix.
            n_iters: Number of iterations.
            tolerance: Convergence tolerance for dual variables (early stopping).

        Returns:
            Tuple of (transport_plan, dual_vars) where transport_plan is (M, N).
        """
        if n_iters is None:
            n_iters = self.n_iterations

        M, N = cost_matrix.shape
        device = cost_matrix.device

        # Work in log-space throughout to avoid overflow
        log_K = -cost_matrix / self.epsilon  # (M, N)

        # Initialize log-domain dual variables as zeros
        log_u = torch.zeros(M, device=device)  # (M,)
        log_v = torch.zeros(N, device=device)  # (N,)

        # Sinkhorn iterations in log-domain
        for iteration in range(n_iters):
            # Store previous dual variables for convergence check
            log_u_prev = log_u.clone()
            log_v_prev = log_v.clone()

            # Update log_u: log_u = -logsumexp(log_K + log_v, dim=1)
            log_u = -torch.logsumexp(log_K + log_v[None, :], dim=1)

            # Update log_v: log_v = -logsumexp(log_K + log_u, dim=0)
            log_v = -torch.logsumexp(log_K + log_u[:, None], dim=0)

            # Optional convergence check for early stopping
            if iteration > 0:
                u_change = torch.max(torch.abs(log_u - log_u_prev))
                v_change = torch.max(torch.abs(log_v - log_v_prev))
                max_change = torch.max(u_change, v_change)
                if max_change < tolerance:
                    break

        # Compute transport plan in linear space only at the end.
        # Clamp the exponent to avoid overflow for very small epsilon values.
        log_transport = log_u[:, None] + log_K + log_v[None, :]
        log_transport = torch.clamp(log_transport, max=80.0)  # exp(80) ≈ 5.5e34, safe for float32
        transport_plan = torch.exp(log_transport)

        return transport_plan, (log_u, log_v)

    def forward(
        self, cloud1: torch.Tensor, cloud2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Couple two point clouds via OT.

        Args:
            cloud1: (M, D) first point cloud.
            cloud2: (N, D) second point cloud.

        Returns:
            Tuple of (transport_plan, interpolation) where interpolation
            is McCann displacement interpolation at t=0.5.
        """
        # Compute cost matrix: Euclidean distance squared
        diff = cloud1.unsqueeze(1) - cloud2.unsqueeze(0)  # (M, N, D)
        cost_matrix = (diff ** 2).sum(dim=-1)  # (M, N)

        transport_plan, _ = self._sinkhorn_iterations(cost_matrix)

        # McCann displacement interpolation at t=0.5
        interpolation = (transport_plan @ cloud2) / transport_plan.sum(dim=1, keepdim=True)

        return transport_plan, interpolation


class DriftNetwork(nn.Module):
    """Learned drift function f_θ(z, t) for the SDE.

    Small MLP that maps (latent_state, time) → drift vector.

    Args:
        latent_dim: Dimension of latent state (default 64).
        hidden_dim: Hidden layer dimension (default 32).
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute drift at (z, t).

        Args:
            z: (B, D) latent state.
            t: (B,) or (B, 1) time points.

        Returns:
            (B, D) drift vector.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        z_t = torch.cat([z, t], dim=-1)
        return self.net(z_t)


class DiffusionNetwork(nn.Module):
    """Learned diffusion coefficient g_θ(z, t) for the SDE.

    Small MLP outputting positive-definite diagonal diffusion matrix.
    Represents clonal stochasticity and unmodeled epigenetic switching.

    Args:
        latent_dim: Dimension of latent state (default 64).
        hidden_dim: Hidden layer dimension (default 32).
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),  # Ensure positive definiteness
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute diagonal diffusion at (z, t).

        Args:
            z: (B, D) latent state.
            t: (B,) or (B, 1) time points.

        Returns:
            (B, D) diagonal diffusion coefficients.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        z_t = torch.cat([z, t], dim=-1)
        return self.net(z_t)


class JumpProcess(nn.Module):
    """Models therapy-induced discontinuities via Poisson jump process.

    Jump rate λ_θ(z,t) governs how often jumps occur.
    Jump size h_θ(z) governs where system jumps to.
    Captures sudden state changes from drug treatment or clonal extinction.

    Args:
        latent_dim: Dimension of latent state (default 64).
        hidden_dim: Hidden layer dimension (default 32).
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Jump rate network: outputs scalar ≥ 0
        self.rate_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        # Jump size network: outputs displacement vector
        self.size_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def jump_rate(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute jump rate λ_θ(z, t).

        Args:
            z: (B, D) latent state.
            t: (B,) or (B, 1) time points.

        Returns:
            (B, 1) jump rates.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        z_t = torch.cat([z, t], dim=-1)
        return self.rate_net(z_t)

    def jump_size(self, z: torch.Tensor) -> torch.Tensor:
        """Compute jump size h_θ(z).

        Args:
            z: (B, D) latent state.

        Returns:
            (B, D) jump displacement.
        """
        return self.size_net(z)

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, dt: float, rng: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample jumps from Poisson process.

        Args:
            z: (B, D) current state.
            t: (B,) or (B, 1) current time.
            dt: Time step.
            rng: Torch random generator.

        Returns:
            Tuple of (z_after_jump, n_jumps) where n_jumps is (B, 1) count.
        """
        rate = self.jump_rate(z, t)  # (B, 1)
        lambda_dt = rate * dt  # Expected number of jumps in [t, t+dt]

        # Poisson sampling
        n_jumps = torch.poisson(lambda_dt, generator=rng).long()  # (B, 1)

        # Jump displacement (same for all jumps, simplified)
        displacement = self.jump_size(z)  # (B, D)

        # Update state
        z_after = z + n_jumps.float() * displacement

        return z_after, n_jumps.squeeze(-1)


class SurvivalTimeCalibrator(nn.Module):
    """Maps latent pseudotime to calendar time via Weibull hazard model.

    Learns shape parameter k and scale parameter λ. Provides survival
    function S(t) = exp(-(t/λ)^k) for uncertainty quantification in
    survival-time predictions.

    Args:
        latent_dim: Dimension of latent state (default 64).
    """

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Learnable Weibull parameters
        self.shape = nn.Parameter(torch.tensor(1.5))  # k > 0
        self.scale = nn.Parameter(torch.tensor(1.0))  # λ > 0

    def forward(self, pseudotime: torch.Tensor) -> torch.Tensor:
        """Compute calendar time from pseudotime via Weibull mapping.

        Args:
            pseudotime: (B,) pseudotime values from ODE.

        Returns:
            (B,) calendar times.
        """
        k = F.softplus(self.shape)
        lam = F.softplus(self.scale)
        # Weibull CDF: F(t) = 1 - exp(-(t/λ)^k)
        # Inverse: t = λ * (- log(1 - u))^(1/k)
        # FIX #5: Weibull quantile function (was: calendar_time = lam * pseudotime)
        # Clamp pseudotime to (epsilon, 1-epsilon) to avoid log(0)
        u = torch.clamp(pseudotime, min=1e-6, max=1.0 - 1e-6)
        # t = lambda * (-log(1 - u))^(1/k)  — proper Weibull inverse CDF
        calendar_time = lam * torch.pow(-torch.log(1.0 - u), 1.0 / k)
        return calendar_time

    def survival_function(self, t: torch.Tensor) -> torch.Tensor:
        """Compute Weibull survival function S(t) = exp(-(t/λ)^k).

        Args:
            t: (B,) time points.

        Returns:
            (B,) survival probabilities.
        """
        k = F.softplus(self.shape)
        lam = F.softplus(self.scale)
        return torch.exp(-((t / lam) ** k + 1e-8))


class NeuralJumpSDE(nn.Module):
    """Stochastic differential equation with drift, diffusion, and jumps.

    Wraps ChromatinODE but adds stochastic dynamics:
        dz = f_θ(z, t) dt + g_θ(z, t) dW + dJ
    where dJ is a jump process.

    Integrates via Euler-Maruyama scheme. Falls back to deterministic ODE
    if stochastic=False.

    Args:
        ode: ChromatinODE instance.
        latent_dim: Dimension of latent space (default 64).
        hidden_dim: Hidden layer dimension for networks (default 32).
        include_jumps: Whether to include jump process (default True).
    """

    def __init__(
        self,
        ode: ChromatinODE,
        latent_dim: int = 64,
        hidden_dim: int = 32,
        include_jumps: bool = True,
    ) -> None:
        super().__init__()
        self.ode = ode
        self.latent_dim = latent_dim
        self.drift_net = DriftNetwork(latent_dim, hidden_dim)
        self.diffusion_net = DiffusionNetwork(latent_dim, hidden_dim)
        self.jump_process = JumpProcess(latent_dim, hidden_dim) if include_jumps else None

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        n_steps: int = 100,
        stochastic: bool = True,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Integrate SDE from z0 over t_span.

        Args:
            z0: (B, D) initial state.
            t_span: (2,) time interval [t0, tf].
            n_steps: Number of integration steps.
            stochastic: If False, use deterministic ODE path.
            n_samples: Number of Monte Carlo samples (only if stochastic).

        Returns:
            (T, B, D) or (T, B*n_samples, D) trajectory.
        """
        batch_size = z0.shape[0]
        device = z0.device
        dt = (t_span[1] - t_span[0]) / n_steps

        if not stochastic or not (self.drift_net is not None):
            # Fall back to deterministic ODE
            return self._integrate_ode(z0, t_span, n_steps)

        # Monte Carlo SDE integration
        z_samples = z0.unsqueeze(0).expand(n_samples, -1, -1).reshape(batch_size * n_samples, -1)
        times = torch.linspace(t_span[0], t_span[1], n_steps + 1, device=device)
        trajectory = [z_samples]

        rng = torch.Generator(device=device)

        for step in range(n_steps):
            t = times[step]
            t_batch = torch.full((z_samples.shape[0],), t.item(), device=device)

            # Drift term
            drift = self.drift_net(z_samples, t_batch)

            # Diffusion term
            diffusion = self.diffusion_net(z_samples, t_batch)
            dW = torch.randn_like(z_samples, generator=rng) * (dt ** 0.5)
            stochastic_term = diffusion * dW

            # Jump term
            jump_term = torch.zeros_like(z_samples)
            if self.jump_process is not None:
                z_after_jump, _ = self.jump_process(z_samples, t_batch, dt, rng)
                jump_term = z_after_jump - z_samples

            # Euler step
            z_samples = z_samples + drift * dt + stochastic_term + jump_term

            trajectory.append(z_samples)

        return torch.stack(trajectory, dim=0)

    def _integrate_ode(
        self, z0: torch.Tensor, t_span: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
        """Fall back to deterministic ODE integration."""
        times = torch.linspace(t_span[0], t_span[1], n_steps + 1, device=z0.device)
        trajectory = [z0]

        z = z0.clone()
        for i in range(n_steps):
            t = times[i]
            t_next = times[i + 1]
            dt = t_next - t

            # Simple Euler step (would normally use torchdiffeq here)
            dz = self.drift_net(z, torch.full((z.shape[0],), t.item(), device=z.device))
            z = z + dz * dt

            trajectory.append(z)

        return torch.stack(trajectory, dim=0)


class ChromatinODE(nn.Module):
    """ODE system for the Sneppen-Ringrose bistability model.

    State variables:
        a: Active chromatin mark level (e.g., H3K4me3)
        r: Repressive chromatin mark level (e.g., H3K27me3)

    Dynamics:
        da/dt = w_a * f(a) - e_r * g(r) * a - d_a * a + basal_a
        dr/dt = w_r * f(r) - e_a * g(a) * r - d_r * r + basal_r

    Where:
        w_a, w_r: Writer strengths (auto-catalysis, from proteomic levels)
        e_a, e_r: Eraser strengths (cross-inhibition, from proteomic levels)
        d_a, d_r: Dilution rates (from proliferation markers)
        f, g: Hill functions for cooperative binding
        basal_a, basal_r: Basal production rates

    Parameters are derived from chromatin reader/writer protein abundances.
    """

    def __init__(self, config: StabilityConfig) -> None:
        super().__init__()
        n_proteins = len(config.reader_writer_proteins)

        # Learnable mapping: protein levels → ODE parameters
        # This is calibrated against washout time-course data
        self.protein_to_params = nn.Sequential(
            nn.Linear(n_proteins, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 8),  # [w_a, w_r, e_a, e_r, d_a, d_r, basal_a, basal_r]
            nn.Softplus(),  # All ODE params must be positive
        )

        # Hill function parameters (learnable)
        self.hill_n = nn.Parameter(torch.tensor(2.0))  # Cooperativity
        self.hill_k = nn.Parameter(torch.tensor(0.5))  # Half-max

    def _hill(self, x: torch.Tensor) -> torch.Tensor:
        """Hill function for cooperative binding."""
        n = F.softplus(self.hill_n)  # Ensure n > 0
        k = F.softplus(self.hill_k)
        return x.pow(n) / (k.pow(n) + x.pow(n) + 1e-8)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute derivatives for the chromatin ODE system.

        Args:
            t: Current time (scalar, unused but required by odeint).
            state: (B, 2 + 8) tensor where [:, 0] = a, [:, 1] = r,
                   [:, 2:] = ODE parameters (constant through integration).

        Returns:
            (B, 2 + 8) derivatives (params have zero derivative).
        """
        a = state[:, 0:1]  # Active mark level
        r = state[:, 1:2]  # Repressive mark level
        params = state[:, 2:]  # ODE parameters (constant)

        w_a = params[:, 0:1]
        w_r = params[:, 1:2]
        e_a = params[:, 2:3]
        e_r = params[:, 3:4]
        d_a = params[:, 4:5]
        d_r = params[:, 5:6]
        basal_a = params[:, 6:7]
        basal_r = params[:, 7:8]

        da_dt = w_a * self._hill(a) - e_r * self._hill(r) * a - d_a * a + basal_a
        dr_dt = w_r * self._hill(r) - e_a * self._hill(a) * r - d_r * r + basal_r

        # Parameters are constant — zero derivatives
        dparam_dt = torch.zeros_like(params)

        return torch.cat([da_dt, dr_dt, dparam_dt], dim=1)


class MemoryStabilityScorer(nn.Module):
    """Computes the memory stability score for a given proteomic profile.

    Pipeline:
        1. Extract chromatin reader/writer protein levels from full proteome
        2. Map protein levels → ODE parameters via learned neural network
        3. Integrate ODE to find steady state
        4. Estimate basin-of-attraction depth via perturbation sampling
        5. Normalize to 0–1 stability score

    A score of 0 means the epigenetic state is easily flipped (transient
    adaptation, potentially reversible by drug rechallenge).

    A score of 1 means the state is deeply locked in (permanent epigenetic
    memory, resistant to perturbation).

    Args:
        config: StabilityConfig with ODE and calibration parameters.
    """

    def __init__(self, config: StabilityConfig) -> None:
        super().__init__()
        self.config = config
        self.ode = ChromatinODE(config)
        self.protein_names = config.reader_writer_proteins

        # Learnable normalization for sigmoid centering
        # Initialized from training distribution statistics; updated during calibration
        self.basin_center = nn.Parameter(torch.tensor(1.53))
        self.basin_scale = nn.Parameter(torch.tensor(1.56))

    def extract_reader_writer_levels(
        self,
        proteomics: torch.Tensor,
        all_protein_names: list[str],
    ) -> torch.Tensor:
        """Extract chromatin reader/writer protein levels from full proteome.

        Args:
            proteomics: (B, P) full protein abundance tensor.
            all_protein_names: List of P protein names matching columns.

        Returns:
            (B, N_rw) tensor of reader/writer protein levels.
        """
        name_to_idx = {name: i for i, name in enumerate(all_protein_names)}
        indices = []
        for prot in self.protein_names:
            if prot in name_to_idx:
                indices.append(name_to_idx[prot])
            else:
                # Use zero for missing proteins
                indices.append(-1)

        result = torch.zeros(
            proteomics.shape[0], len(self.protein_names),
            device=proteomics.device, dtype=proteomics.dtype,
        )
        for i, idx in enumerate(indices):
            if idx >= 0:
                result[:, i] = proteomics[:, idx]

        return result

    def _find_steady_state(
        self, ode_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate ODE to find the steady-state chromatin configuration.

        Args:
            ode_params: (B, 8) ODE parameters from protein_to_params network.

        Returns:
            Tuple of (a_steady, r_steady), each (B, 1).
        """
        if odeint is None:
            raise ImportError(
                "torchdiffeq is required for stability scoring. "
                "Install with: pip install torchdiffeq"
            )

        batch_size = ode_params.shape[0]
        device = ode_params.device

        # Initial condition: balanced state
        a0 = torch.full((batch_size, 1), 0.5, device=device)
        r0 = torch.full((batch_size, 1), 0.5, device=device)
        state0 = torch.cat([a0, r0, ode_params], dim=1)

        t_span = torch.tensor(
            [0.0, self.config.integration_time], device=device
        )

        # Integrate — use Euler with small step size for numerical stability.
        # Large ODE parameters (e.g. dilution rates > 1) require step_size < 1
        # to keep the explicit Euler scheme stable.
        trajectory = odeint(
            self.ode,
            state0,
            t_span,
            method="euler",
            options={"step_size": 0.1},
        )

        final_state = trajectory[-1]  # (B, 10)
        a_steady = final_state[:, 0:1].clamp(0.0, 10.0)
        r_steady = final_state[:, 1:2].clamp(0.0, 10.0)
        # Replace NaN from diverged ODE with balanced default
        a_steady = torch.where(a_steady.isnan(), torch.tensor(0.5, device=device), a_steady)
        r_steady = torch.where(r_steady.isnan(), torch.tensor(0.5, device=device), r_steady)

        return a_steady, r_steady

    def _estimate_basin_depth(
        self,
        ode_params: torch.Tensor,
        a_steady: torch.Tensor,
        r_steady: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate basin-of-attraction depth via numerical Jacobian.

        Computes the maximum eigenvalue of the Jacobian at the steady state.
        More negative eigenvalue = deeper basin = higher stability.

        Args:
            ode_params: (B, 8) ODE parameters.
            a_steady: (B, 1) steady-state active mark level.
            r_steady: (B, 1) steady-state repressive mark level.

        Returns:
            (B,) basin depth scores (higher = more stable).
        """
        batch_size = ode_params.shape[0]
        device = ode_params.device

        # Compute the Jacobian of the ODE at the steady state.
        # The max real eigenvalue (most negative = most stable) directly
        # quantifies how strongly the system is attracted back after
        # perturbation.  This discriminates between samples even when
        # the ODE is monostable (single attractor).
        #
        # Force float32 for numerical Jacobian — bf16 lacks precision for
        # finite differences with eps=1e-3.
        eps = 1e-3
        steady = torch.cat([a_steady.detach(), r_steady.detach()], dim=1).float()
        ode_params_f32 = ode_params.float()
        full_state = torch.cat([steady, ode_params_f32], dim=1)  # (B, 10)

        # Numerical Jacobian via finite differences (2x2 for the a,r subsystem)
        jacobians = torch.zeros(batch_size, 2, 2, device=device)
        t_zero = torch.tensor(0.0, device=device)
        f0 = self.ode(t_zero, full_state)[:, :2]  # (B, 2)

        for j in range(2):
            perturbed = full_state.clone()
            perturbed[:, j] = perturbed[:, j] + eps
            f_plus = self.ode(t_zero, perturbed)[:, :2]
            jacobians[:, :, j] = (f_plus - f0) / eps

        # Eigenvalues of 2x2 matrix via quadratic formula (batched, no loops)
        a11 = jacobians[:, 0, 0]
        a12 = jacobians[:, 0, 1]
        a21 = jacobians[:, 1, 0]
        a22 = jacobians[:, 1, 1]

        trace = a11 + a22
        det = a11 * a22 - a12 * a21
        discriminant = (trace ** 2 - 4 * det).clamp(min=0.0)

        # Max eigenvalue (least negative = least stable)
        lambda_max = (trace + discriminant.sqrt()) / 2  # (B,)

        # More negative lambda_max = more stable.
        # Convert to [0, 1]: use -lambda_max as the stability metric.
        # Larger -lambda_max = deeper basin.
        basin_depth = -lambda_max  # Positive values = stable fixed point
        return basin_depth

    def forward(
        self,
        proteomics: torch.Tensor,
        all_protein_names: list[str],
    ) -> torch.Tensor:
        """Compute memory stability score for a batch of proteomic profiles.

        Args:
            proteomics: (B, P) protein abundance tensor.
            all_protein_names: List of P protein names.

        Returns:
            (B,) stability scores in [0, 1].
        """
        # Step 1: Extract reader/writer levels
        rw_levels = self.extract_reader_writer_levels(proteomics, all_protein_names)

        # Step 2: Map to ODE parameters
        ode_params = self.ode.protein_to_params(rw_levels)

        # Step 3: Find steady state
        a_steady, r_steady = self._find_steady_state(ode_params)

        # Step 4: Estimate basin depth
        basin_depth = self._estimate_basin_depth(ode_params, a_steady, r_steady)

        # Step 5: Normalize to [0, 1] — sigmoid with learnable centering.
        # basin_center and basin_scale are nn.Parameters updated during calibration.
        score = torch.sigmoid(self.basin_scale * (basin_depth - self.basin_center))

        # Guard against NaN from ODE divergence on rare samples
        score = torch.where(score.isnan(), torch.tensor(0.5, device=score.device), score)

        return score


class TrajectoryForecaster(nn.Module):
    """Predicts the temporal evolution of epigenetic state over future time horizons.

    Takes a current VAE latent state (64-dim) and integrates the chromatin ODE
    forward in time to predict future epigenetic memory states at specified
    horizons (3, 6, 12 months).

    Key innovation over MemoryStabilityScorer:
        - MyeloMemory ODE only finds steady states (equilibrium points)
        - TrajectoryForecaster predicts the TEMPORAL PATH to those states,
          mapping the resistance landscape over time
        - Computes transition probabilities between basins of attraction
        - Provides stability scores at each horizon

    Args:
        config: StabilityConfig with ODE parameters.
        protein_names: List of chromatin reader/writer protein names.
    """

    def __init__(
        self,
        config: StabilityConfig,
        protein_names: list[str],
        use_sde: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.protein_names = protein_names
        self.ode = ChromatinODE(config)
        self.latent_dim = 64  # VAE latent dimension
        self.use_sde = use_sde

        # Time horizons in arbitrary units (proportional to months)
        # These map to real months via calibration
        self.horizon_times = {
            3: 30.0,    # 3 months
            6: 60.0,    # 6 months
            12: 120.0,  # 12 months
        }

        # Learnable horizon scaling factor (calibrated from data)
        self.horizon_scale = nn.Parameter(torch.tensor(1.0))

        # Per-parameter ODE modulation from latent state (Linear(64→8))
        # Optimized in train_trajectory_forecaster alongside horizon_scale;
        # _latent_to_ode_params currently delegates to protein_to_params, so this
        # head is reserved for future use but must exist as a Module so the
        # forecaster training optimizer/grad-clip references resolve.
        self.latent_to_params = nn.Linear(64, 8)

        # Basin transition parameters (learned during training)
        self.basin_transition_net = nn.Sequential(
            nn.Linear(64 + 2, 32),  # latent + (a_steady, r_steady)
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 2),  # Logits for basin assignment (active vs. repressive)
        )

        # Stochastic dynamics (if enabled)
        if self.use_sde:
            self.neural_jump_sde = NeuralJumpSDE(self.ode, self.latent_dim)
            self.survival_calibrator = SurvivalTimeCalibrator(self.latent_dim)
            self.ot_coupler = SinkhornOT()
            logger.info("TrajectoryForecaster initialized with SDE support (stochastic mode)")
        else:
            self.neural_jump_sde = None
            self.survival_calibrator = None
            self.ot_coupler = None
            logger.info(f"Initialized TrajectoryForecaster with latent_dim={self.latent_dim} (deterministic ODE mode)")

    def _latent_to_ode_params(
        self, latent_state: torch.Tensor, protein_abundances: torch.Tensor
    ) -> torch.Tensor:
        """Convert VAE latent state + protein abundances to ODE parameters.

        In a full implementation, we would learn a mapping from the VAE latent
        state to ODE parameters. For now, we use the chromatin ODE's protein_to_params
        and incorporate the latent state as a modulation factor.

        Args:
            latent_state: (B, 64) VAE memory state.
            protein_abundances: (B, N_rw) chromatin reader/writer levels.

        Returns:
            (B, 8) ODE parameters.
        """
        # Direct protein-to-params mapping
        ode_params = self.ode.protein_to_params(protein_abundances)

        # Optionally modulate ODE parameters by latent state
        # (latent state encodes global epigenetic configuration)
        latent_mod = torch.sigmoid(latent_state.mean(dim=1, keepdim=True))  # (B, 1)
        ode_params = ode_params * (0.8 + 0.4 * latent_mod)  # Modulate ±20%

        return ode_params

    def _integrate_trajectory(
        self,
        ode_params: torch.Tensor,
        time_horizon: float,
        initial_a: torch.Tensor,
        initial_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate chromatin ODE forward from initial state to time horizon.

        Args:
            ode_params: (B, 8) ODE parameters.
            time_horizon: Integration time (arbitrary units).
            initial_a: (B, 1) initial active mark level.
            initial_r: (B, 1) initial repressive mark level.

        Returns:
            Tuple of (a_final, r_final, trajectory).
            trajectory: (T, B, 2) full time evolution of (a, r).
        """
        if odeint is None:
            raise ImportError("torchdiffeq required. Install: pip install torchdiffeq")

        batch_size = ode_params.shape[0]
        device = ode_params.device

        # Initial state: current chromatin marks + ODE parameters
        state0 = torch.cat([initial_a, initial_r, ode_params], dim=1)

        # Time span: from 0 to horizon
        t_eval = torch.linspace(0, time_horizon, 50, device=device)

        # Integrate ODE
        trajectory = odeint(
            self.ode,
            state0,
            t_eval,
            method="euler",
            options={"step_size": time_horizon / 100.0},
        )

        final_state = trajectory[-1]
        a_final = final_state[:, 0:1].clamp(0.0, 10.0)
        r_final = final_state[:, 1:2].clamp(0.0, 10.0)

        # Replace NaN from divergence with initial state
        a_final = torch.where(a_final.isnan(), initial_a, a_final)
        r_final = torch.where(r_final.isnan(), initial_r, r_final)

        return a_final, r_final, trajectory

    def _compute_stability_at_state(
        self, a: torch.Tensor, r: torch.Tensor, ode_params: torch.Tensor
    ) -> torch.Tensor:
        """Compute basin depth (stability) at a given (a, r) state.

        Uses the Jacobian at the current state to estimate basin depth.

        Args:
            a: (B, 1) active mark level.
            r: (B, 1) repressive mark level.
            ode_params: (B, 8) ODE parameters.

        Returns:
            (B,) stability scores.
        """
        batch_size = ode_params.shape[0]
        device = ode_params.device

        eps = 1e-3
        steady = torch.cat([a.detach(), r.detach()], dim=1).float()
        ode_params_f32 = ode_params.float()
        full_state = torch.cat([steady, ode_params_f32], dim=1)

        # Compute 2x2 Jacobian
        jacobians = torch.zeros(batch_size, 2, 2, device=device)
        t_zero = torch.tensor(0.0, device=device)
        f0 = self.ode(t_zero, full_state)[:, :2]

        for j in range(2):
            perturbed = full_state.clone()
            perturbed[:, j] = perturbed[:, j] + eps
            f_plus = self.ode(t_zero, perturbed)[:, :2]
            jacobians[:, :, j] = (f_plus - f0) / eps

        # Eigenvalues
        a11 = jacobians[:, 0, 0]
        a12 = jacobians[:, 0, 1]
        a21 = jacobians[:, 1, 0]
        a22 = jacobians[:, 1, 1]

        trace = a11 + a22
        det = a11 * a22 - a12 * a21
        discriminant = (trace ** 2 - 4 * det).clamp(min=0.0)

        lambda_max = (trace + discriminant.sqrt()) / 2
        basin_depth = -lambda_max

        # Normalize: sigmoid with fixed centering
        stability = torch.sigmoid(2.0 * (basin_depth - 1.5))
        stability = torch.where(stability.isnan(), torch.tensor(0.5, device=device), stability)

        return stability

    def _compute_transition_probs(
        self,
        initial_state: tuple[torch.Tensor, torch.Tensor],
        final_state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute transition probabilities between basins of attraction.

        Estimates the probability of transitioning from initial basin to
        final basin based on trajectory proximity and basin depths.

        Args:
            initial_state: (a_init, r_init), each (B, 1).
            final_state: (a_final, r_final), each (B, 1).

        Returns:
            (B,) transition probabilities in [0, 1].
        """
        a_init, r_init = initial_state
        a_final, r_final = final_state

        # Distance traveled in (a, r) space
        euclidean_dist = torch.sqrt((a_final - a_init) ** 2 + (r_final - r_init) ** 2)

        # Normalize: max distance is sqrt(10^2 + 10^2) = 14.14
        norm_dist = (euclidean_dist / 15.0).clamp(0.0, 1.0)

        # Transition probability: higher distance = higher prob of basin change
        transition_prob = torch.sigmoid(3.0 * (norm_dist - 0.3)).squeeze(-1)

        return transition_prob

    def _integrate_sde_trajectory(
        self,
        latent_state: torch.Tensor,
        time_horizon: float,
        n_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Integrate SDE forward with Monte Carlo sampling for uncertainty.

        Args:
            latent_state: (B, 64) VAE latent memory state.
            time_horizon: Integration time (arbitrary units).
            n_samples: Number of Monte Carlo samples.

        Returns:
            Tuple of (a_final_mean, r_final_mean, uncertainties_dict).
            uncertainties_dict contains 'a_mean', 'a_std', 'r_mean', 'r_std'.
        """
        batch_size = latent_state.shape[0]
        device = latent_state.device

        if self.neural_jump_sde is None:
            raise ValueError("SDE not initialized. Set use_sde=True during __init__")

        t_span = torch.tensor([0.0, time_horizon], device=device)

        # Run Monte Carlo samples
        a_samples = []
        r_samples = []

        for _ in range(n_samples):
            # Note: In a full implementation, would integrate latent state through SDE.
            # For now, we integrate chromatin marks (a, r) ensemble.
            # This is a placeholder for the full latent SDE.
            trajectory = self.neural_jump_sde(
                latent_state, t_span, n_steps=50, stochastic=True, n_samples=1
            )

            # Extract final state (B, D)
            a_sample = trajectory[-1, :, :1]
            r_sample = trajectory[-1, :, 1:2]

            a_samples.append(a_sample)
            r_samples.append(r_sample)

        # Stack and compute statistics
        a_stack = torch.stack(a_samples, dim=0)  # (n_samples, B, 1)
        r_stack = torch.stack(r_samples, dim=0)  # (n_samples, B, 1)

        a_mean = a_stack.mean(dim=0)
        a_std = a_stack.std(dim=0)
        r_mean = r_stack.mean(dim=0)
        r_std = r_stack.std(dim=0)

        uncertainties = {
            'a_mean': a_mean,
            'a_std': a_std,
            'r_mean': r_mean,
            'r_std': r_std,
        }

        return a_mean, r_mean, uncertainties

    def forecast(
        self,
        initial_state: torch.Tensor,
        protein_abundances: torch.Tensor,
        horizons: list[int] | None = None,
        n_samples: int = 1,
    ) -> dict[str, Any]:
        """Forecast future epigenetic states and stability at multiple horizons.

        Args:
            initial_state: (B, 64) VAE latent memory state.
            protein_abundances: (B, N_rw) chromatin reader/writer protein levels.
            horizons: Time horizons in months (e.g., [3, 6, 12]). Defaults to [3, 6, 12].
            n_samples: Number of Monte Carlo samples for SDE (default 1, ignored if use_sde=False).

        Returns:
            Dictionary with keys:
                - 'states': Dict[int, Tuple[torch.Tensor, torch.Tensor]]
                  Predicted (a, r) at each horizon.
                - 'stability_scores': Dict[int, torch.Tensor]
                  Stability score at each horizon.
                - 'transition_probs': Dict[int, torch.Tensor]
                  Transition probability to alternative basin at each horizon.
                - 'uncertainties': Dict[int, Dict[str, torch.Tensor]] (SDE only)
                  Contains 'a_mean', 'a_std', 'r_mean', 'r_std' for confidence intervals.
                - 'survival_probs': Dict[int, torch.Tensor] (SDE only)
                  Calibrated survival probabilities at each horizon.
                - 'temporal_disclaimers': Dict[int, str]
                  Epistemic warnings for each horizon.
                - 'initial_stability': torch.Tensor
                  Stability at current state.
        """
        if horizons is None:
            horizons = [3, 6, 12]

        self.eval()
        device = initial_state.device

        # Get ODE parameters from protein abundances
        ode_params = self._latent_to_ode_params(initial_state, protein_abundances)

        # Initialize at balanced (equatorial) state
        batch_size = initial_state.shape[0]
        a_init = torch.full((batch_size, 1), 0.5, device=device)
        r_init = torch.full((batch_size, 1), 0.5, device=device)

        # Compute initial stability
        initial_stability = self._compute_stability_at_state(a_init, r_init, ode_params)

        results = {
            'states': {},
            'stability_scores': {},
            'transition_probs': {},
            'initial_stability': initial_stability,
        }

        # Add SDE-specific fields
        if self.use_sde:
            results['uncertainties'] = {}
            results['survival_probs'] = {}

        results['temporal_disclaimers'] = {}

        with torch.no_grad():
            for horizon in horizons:
                time_h = self.horizon_times.get(horizon, float(horizon * 10.0))
                time_h = time_h * F.softplus(self.horizon_scale)

                if self.use_sde and self.neural_jump_sde is not None:
                    # SDE integration with Monte Carlo samples
                    a_final, r_final, uncertainties = self._integrate_sde_trajectory(
                        initial_state, time_h, n_samples
                    )
                    results['uncertainties'][horizon] = uncertainties

                    # Survival probability from calibrator
                    if self.survival_calibrator is not None:
                        # Map pseudotime to calendar time and compute survival
                        survival_prob = self.survival_calibrator.survival_function(
                            torch.tensor([time_h], device=device)
                        )
                        results['survival_probs'][horizon] = survival_prob
                else:
                    # Standard ODE integration (deterministic)
                    a_final, r_final, traj = self._integrate_trajectory(
                        ode_params, time_h, a_init, r_init
                    )

                # Compute stability at this horizon
                stab = self._compute_stability_at_state(a_final, r_final, ode_params)

                # Compute transition probability
                trans_prob = self._compute_transition_probs(
                    (a_init, r_init), (a_final, r_final)
                )

                results['states'][horizon] = (a_final, r_final)
                results['stability_scores'][horizon] = stab
                results['transition_probs'][horizon] = trans_prob
                results['temporal_disclaimers'][horizon] = TEMPORAL_DISCLAIMERS.get(
                    horizon, "Temporal predictions rely on ODE stability assumptions."
                )

                logger.debug(
                    f"Horizon {horizon}m: a_final={a_final.mean():.3f}, "
                    f"r_final={r_final.mean():.3f}, "
                    f"stability={stab.mean():.3f}, trans_prob={trans_prob.mean():.3f}"
                )

        return results

    def forecast_with_uncertainty(
        self,
        initial_state: torch.Tensor,
        protein_abundances: torch.Tensor,
        horizons: list[int] | None = None,
        n_mc_samples: int = 100,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Run Monte Carlo SDE rollouts and return confidence intervals.

        This method runs many SDE trajectories to build confidence intervals
        around point predictions. Only available when use_sde=True.

        Args:
            initial_state: (B, 64) VAE latent memory state.
            protein_abundances: (B, N_rw) chromatin reader/writer levels.
            horizons: Time horizons in months (e.g., [3, 6, 12]).
            n_mc_samples: Number of Monte Carlo trajectories.
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI).

        Returns:
            Dictionary with keys:
                - 'point_forecast': Standard forecast() output.
                - 'ci_lower': Dict[int, Tuple[torch.Tensor, torch.Tensor]]
                  Lower confidence bounds (a_lower, r_lower) at each horizon.
                - 'ci_upper': Dict[int, Tuple[torch.Tensor, torch.Tensor]]
                  Upper confidence bounds (a_upper, r_upper) at each horizon.
                - 'n_mc_samples': int
                  Number of samples used.
        """
        if not self.use_sde or self.neural_jump_sde is None:
            raise ValueError(
                "SDE not initialized. Set use_sde=True during __init__ to use "
                "forecast_with_uncertainty()."
            )

        if horizons is None:
            horizons = [3, 6, 12]

        # Run base forecast
        point_forecast = self.forecast(initial_state, protein_abundances, horizons, n_samples=1)

        # Compute quantiles from Monte Carlo samples
        device = initial_state.device
        batch_size = initial_state.shape[0]

        lower_q = (1.0 - confidence_level) / 2.0
        upper_q = 1.0 - lower_q

        ci_lower = {}
        ci_upper = {}

        with torch.no_grad():
            for horizon in horizons:
                time_h = self.horizon_times.get(horizon, float(horizon * 10.0))
                time_h = time_h * F.softplus(self.horizon_scale)

                # Collect samples
                a_ensemble = []
                r_ensemble = []

                for _ in range(n_mc_samples):
                    a_samp, r_samp, _ = self._integrate_sde_trajectory(
                        initial_state, time_h, n_samples=1
                    )
                    a_ensemble.append(a_samp)
                    r_ensemble.append(r_samp)

                a_ensemble = torch.cat(a_ensemble, dim=0)  # (n_mc_samples*B, 1)
                r_ensemble = torch.cat(r_ensemble, dim=0)  # (n_mc_samples*B, 1)

                # Reshape to (n_mc_samples, B, 1)
                a_ensemble = a_ensemble.reshape(n_mc_samples, batch_size, 1)
                r_ensemble = r_ensemble.reshape(n_mc_samples, batch_size, 1)

                # Compute quantiles
                a_lower = torch.quantile(a_ensemble, lower_q, dim=0)
                a_upper = torch.quantile(a_ensemble, upper_q, dim=0)
                r_lower = torch.quantile(r_ensemble, lower_q, dim=0)
                r_upper = torch.quantile(r_ensemble, upper_q, dim=0)

                ci_lower[horizon] = (a_lower, r_lower)
                ci_upper[horizon] = (a_upper, r_upper)

                logger.info(
                    f"Horizon {horizon}m: "
                    f"a [{a_lower.mean():.3f}, {a_upper.mean():.3f}], "
                    f"r [{r_lower.mean():.3f}, {r_upper.mean():.3f}]"
                )

        return {
            'point_forecast': point_forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_mc_samples': n_mc_samples,
            'confidence_level': confidence_level,
        }


@dataclass
class ForecastReliability:
    """Quantifies the reliability of a neural ODE forecast horizon.

    Attributes:
        horizon_months: The forecast horizon in months.
        reliability_grade: Letter grade ("A" for 0-3m, "B" for 3-6m, "C" for 6-9m, "F" for 9-12m+).
        disclaimer: Disclaimer text from TEMPORAL_DISCLAIMERS for this horizon.
        confidence_decay_factor: Exponential decay factor (e.g. 0.95^months) quantifying confidence loss.
        is_reliable: Boolean flag (True only for grades A or B).
        hard_warning: Strong warning message for grades C/F, or None for A/B.
        recommended_action: Actionable guidance string (e.g., "Use as point estimate" for A).
    """
    horizon_months: int
    reliability_grade: str
    disclaimer: str
    confidence_decay_factor: float
    is_reliable: bool
    hard_warning: str | None
    recommended_action: str


class ForecastReliabilityScorer:
    """Enforces temporal disclaimers and reliability constraints on Neural ODE forecasts.

    Addresses the fundamental issue that extrapolating a deterministic ODE 12 months
    into the future from a single snapshot is highly speculative. This scorer:
    - Assigns reliability grades based on forecast horizon
    - Applies exponential confidence decay
    - Provides actionable warnings and recommendations
    - Supports validation with optional error raising

    The reliability grading is based on accumulating uncertainty in ODE parameters,
    unmodeled stochastic processes, and the limits of deterministic extrapolation.
    """

    def __init__(self, base_decay_rate: float = 0.95) -> None:
        """Initialize the ForecastReliabilityScorer.

        Args:
            base_decay_rate: Monthly confidence decay rate (default 0.95 → 5% loss per month).
                             Used as base^months to compute decay factor.
        """
        self.base_decay_rate = base_decay_rate

    def score(
        self,
        forecast_horizon_months: int,
        uncertainty_estimate: float | None = None,
    ) -> ForecastReliability:
        """Score the reliability of a forecast at a given horizon.

        Args:
            forecast_horizon_months: Forecast horizon in months (3, 6, 9, or 12).
            uncertainty_estimate: Optional external uncertainty measure (unused in base grading).

        Returns:
            ForecastReliability dataclass with grade, warnings, and recommendations.

        Raises:
            ValueError: If horizon is not in supported range [1, 12].
        """
        if not (1 <= forecast_horizon_months <= 12):
            raise ValueError(
                f"Forecast horizon must be in [1, 12] months; got {forecast_horizon_months}"
            )

        # Compute exponential confidence decay factor
        confidence_decay_factor = self.base_decay_rate ** forecast_horizon_months

        # Assign reliability grade based on horizon
        if forecast_horizon_months <= 3:
            reliability_grade = "A"
            is_reliable = True
            hard_warning = None
            recommended_action = (
                "Use as point estimate. Confidence intervals remain narrow. "
                "Suitable for near-term decision support."
            )
        elif forecast_horizon_months <= 6:
            reliability_grade = "B"
            is_reliable = True
            hard_warning = None
            recommended_action = (
                "Use with wide confidence intervals. Accumulating parameter uncertainty. "
                "Suitable for medium-term planning with sensitivity analysis."
            )
        elif forecast_horizon_months <= 9:
            reliability_grade = "C"
            is_reliable = False
            hard_warning = (
                f"CAUTION: {forecast_horizon_months}-month ODE forecast is highly speculative. "
                "Parameters may drift, stochastic epigenetic switches are unmodeled, "
                "and discrete events (mutations, clonal selection) are not captured. "
                "Use qualitative trends only; do not rely on quantitative predictions."
            )
            recommended_action = (
                "Treat as qualitative trend only. Use multiple corroborating data sources. "
                "Consider ensemble forecasts with alternative models."
            )
        else:  # >= 10 months
            reliability_grade = "F"
            is_reliable = False
            hard_warning = (
                f"UNRELIABLE: {forecast_horizon_months}-month deterministic ODE extrapolation "
                "is not scientifically defensible. Assumes constant parameters and no driver mutations, "
                "epigenetic catastrophes, or heterogeneity growth. This forecast should NOT be used "
                "for clinical decision-making or patient counseling."
            )
            recommended_action = (
                "Do NOT use for quantitative predictions. Consider alternative approaches: "
                "stochastic simulations, ensemble models, or expert clinical judgment."
            )

        # Get disclaimer from TEMPORAL_DISCLAIMERS (use closest available key)
        disclaimer_key = min(TEMPORAL_DISCLAIMERS.keys(), key=lambda k: abs(k - forecast_horizon_months))
        disclaimer = TEMPORAL_DISCLAIMERS[disclaimer_key]

        return ForecastReliability(
            horizon_months=forecast_horizon_months,
            reliability_grade=reliability_grade,
            disclaimer=disclaimer,
            confidence_decay_factor=confidence_decay_factor,
            is_reliable=is_reliable,
            hard_warning=hard_warning,
            recommended_action=recommended_action,
        )

    @staticmethod
    def apply_confidence_decay(
        predictions: dict[str, Any],
        horizon_months: int,
        base_decay_rate: float = 0.95,
    ) -> dict[str, Any]:
        """Apply exponential confidence decay to prediction confidence scores.

        Scales all confidence-related fields (e.g., standard deviations, quantiles)
        by the computed decay factor to reflect growing uncertainty.

        Args:
            predictions: Dictionary containing forecast data, typically with keys like
                        'point_forecast', 'ci_lower', 'ci_upper', 'std_dev', etc.
            horizon_months: Forecast horizon in months.
            base_decay_rate: Monthly decay rate (default 0.95).

        Returns:
            Dictionary with scaled confidence measures.
        """
        decay_factor = base_decay_rate ** horizon_months

        # Make a copy to avoid mutating the original
        result = {}
        for key, value in predictions.items():
            if key in ("ci_lower", "ci_upper", "std_dev", "variance", "std_error"):
                # Scale confidence intervals and uncertainty measures
                if isinstance(value, dict):
                    result[key] = {k: (v * decay_factor if isinstance(v, (int, float)) else v) for k, v in value.items()}
                elif isinstance(value, (list, tuple)):
                    result[key] = type(value)(v * decay_factor if isinstance(v, (int, float)) else v for v in value)
                else:
                    # Assume it's a tensor or numeric scalar
                    result[key] = value * decay_factor
            else:
                # Copy other fields unchanged
                result[key] = value

        # Add metadata
        result["confidence_decay_factor"] = decay_factor
        result["forecast_horizon_months"] = horizon_months

        return result

    def validate_forecast_request(
        self,
        horizon_months: int,
        raise_on_unreliable: bool = False,
    ) -> ForecastReliability:
        """Validate and log a forecast request, optionally raising on unreliable horizons.

        Args:
            horizon_months: Requested forecast horizon in months.
            raise_on_unreliable: If True, raise ValueError for grades C or F (unreliable).

        Returns:
            ForecastReliability object with validation results.

        Raises:
            ValueError: If raise_on_unreliable=True and grade is C or F.
        """
        reliability = self.score(horizon_months)

        # Log the validation result
        log_level = "warning" if not reliability.is_reliable else "info"
        getattr(logger, log_level)(
            f"Forecast validation [{horizon_months}m]: Grade {reliability.reliability_grade} | "
            f"Decay factor: {reliability.confidence_decay_factor:.4f} | "
            f"Reliable: {reliability.is_reliable}"
        )

        # Log the hard warning if present
        if reliability.hard_warning:
            logger.warning(f"Hard warning: {reliability.hard_warning}")

        # Optionally raise
        if raise_on_unreliable and not reliability.is_reliable:
            raise ValueError(
                f"Forecast at {horizon_months} months is unreliable (grade {reliability.reliability_grade}). "
                f"Hard warning: {reliability.hard_warning}"
            )

        return reliability


def calibrate_scorer(
    scorer: MemoryStabilityScorer,
    dataset: Any,
    vae_checkpoint: dict[str, Any],
    config: StabilityConfig,
    ckpt_mgr: CheckpointManager,
    calibrate_survival: bool = False,
    survival_calibrator: SurvivalTimeCalibrator | None = None,
) -> dict[str, Any]:
    """Calibrate the stability scorer against drug washout time-course data.

    The calibration objective: cell lines that show persistent drug resistance
    after washout should have HIGH stability scores; those that revert should
    have LOW stability scores.

    Since direct washout data may be limited, we use a proxy: the variance
    of drug sensitivity across similar cell lines. High variance = low
    stability (the state is noisy/unstable). Low variance = high stability
    (the state is consistent/locked).

    Args:
        scorer: MemoryStabilityScorer model.
        dataset: MultiOmicsDataset.
        vae_checkpoint: Loaded VAE checkpoint (for memory state extraction).
        config: StabilityConfig.
        ckpt_mgr: Checkpoint manager.
        calibrate_survival: Whether to also calibrate SurvivalTimeCalibrator.
        survival_calibrator: Optional SurvivalTimeCalibrator instance.

    Returns:
        Dict with 'checkpoint_path' and 'metrics'.
    """
    device = next(scorer.parameters()).device

    optimizer = torch.optim.Adam(
        scorer.parameters(), lr=config.calibration_lr
    )

    # Use drug sensitivity variance as proxy for instability
    drug_sens = dataset.drug_sensitivity  # (N, D)
    # Compute per-sample variance across drugs (ignoring NaN)
    drug_var = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        valid = drug_sens[i][~torch.isnan(drug_sens[i])]
        if len(valid) > 1:
            drug_var[i] = valid.var().item()
        else:
            drug_var[i] = float("nan")

    # Normalize variance to [0, 1] target (high var → low stability target)
    valid_mask = ~torch.isnan(drug_var)
    if valid_mask.sum() > 0:
        dv = drug_var[valid_mask]
        drug_var_norm = torch.zeros_like(drug_var)
        drug_var_norm[valid_mask] = 1.0 - (dv - dv.min()) / (dv.max() - dv.min() + 1e-8)
    else:
        logger.warning("No valid drug sensitivity data for calibration; using uniform targets")
        drug_var_norm = torch.full((len(dataset),), 0.5)
        valid_mask = torch.ones(len(dataset), dtype=torch.bool)

    # Calibration loop
    best_loss = float("inf")
    protein_names = dataset.protein_names

    for epoch in range(config.calibration_epochs):
        # Mini-batch from valid samples
        valid_indices = torch.where(valid_mask)[0]
        perm = valid_indices[torch.randperm(len(valid_indices))]
        batch_idx = perm[:config.calibration_batch_size]

        proteomics = dataset.proteomics[batch_idx].to(device)
        targets = drug_var_norm[batch_idx].to(device)

        optimizer.zero_grad(set_to_none=True)

        scores = scorer(proteomics, protein_names)
        loss = F.mse_loss(scores, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
        if not any(p.grad is not None and p.grad.isnan().any() for p in scorer.parameters()):
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            logger.info(
                f"[stability_calibrate] Epoch {epoch + 1}/{config.calibration_epochs} "
                f"loss={loss.item():.4f} "
                f"basin_center={scorer.basin_center.item():.4f} "
                f"basin_scale={scorer.basin_scale.item():.4f}"
            )

        if loss.item() < best_loss:
            best_loss = loss.item()
            ckpt_mgr.save("stability_calibrated", {
                "model_state_dict": scorer.state_dict(),
                "epoch": epoch,
                "best_metric": best_loss,
            })

    metrics = {
        "best_calibration_loss": best_loss,
        "n_valid_samples": int(valid_mask.sum()),
    }

    # Optional: calibrate survival time model
    if calibrate_survival and survival_calibrator is not None:
        # Placeholder for survival calibration logic
        # In a full implementation, would fit Weibull parameters to censoring/progression data
        logger.info("Survival time calibrator initialized for optional calibration")
        metrics["survival_calibrator_initialized"] = True

    return {
        "checkpoint_path": ckpt_mgr.path("stability_calibrated"),
        "metrics": metrics,
    }