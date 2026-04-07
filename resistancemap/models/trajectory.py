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

    def __init__(self, config: StabilityConfig, protein_names: list[str]) -> None:
        super().__init__()
        self.config = config
        self.protein_names = protein_names
        self.ode = ChromatinODE(config)
        self.latent_dim = 64  # VAE latent dimension

        # Time horizons in arbitrary units (proportional to months)
        # These map to real months via calibration
        self.horizon_times = {
            3: 30.0,    # 3 months
            6: 60.0,    # 6 months
            12: 120.0,  # 12 months
        }

        # Learnable horizon scaling factor (calibrated from data)
        self.horizon_scale = nn.Parameter(torch.tensor(1.0))

        # Basin transition parameters (learned during training)
        self.basin_transition_net = nn.Sequential(
            nn.Linear(64 + 2, 32),  # latent + (a_steady, r_steady)
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 2),  # Logits for basin assignment (active vs. repressive)
        )

        logger.info(f"Initialized TrajectoryForecaster with latent_dim={self.latent_dim}")

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

    def forecast(
        self,
        initial_state: torch.Tensor,
        protein_abundances: torch.Tensor,
        horizons: list[int] | None = None,
    ) -> dict[str, Any]:
        """Forecast future epigenetic states and stability at multiple horizons.

        Args:
            initial_state: (B, 64) VAE latent memory state.
            protein_abundances: (B, N_rw) chromatin reader/writer protein levels.
            horizons: Time horizons in months (e.g., [3, 6, 12]). Defaults to [3, 6, 12].

        Returns:
            Dictionary with keys:
                - 'states': Dict[int, Tuple[torch.Tensor, torch.Tensor]]
                  Predicted (a, r) at each horizon.
                - 'stability_scores': Dict[int, torch.Tensor]
                  Stability score at each horizon.
                - 'transition_probs': Dict[int, torch.Tensor]
                  Transition probability to alternative basin at each horizon.
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

        with torch.no_grad():
            for horizon in horizons:
                time_h = self.horizon_times.get(horizon, float(horizon * 10.0))
                time_h = time_h * F.softplus(self.horizon_scale)

                # Integrate ODE to horizon
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

                logger.debug(
                    f"Horizon {horizon}m: a_final={a_final.mean():.3f}, "
                    f"r_final={r_final.mean():.3f}, "
                    f"stability={stab.mean():.3f}, trans_prob={trans_prob.mean():.3f}"
                )

        return results


def calibrate_scorer(
    scorer: MemoryStabilityScorer,
    dataset: Any,
    vae_checkpoint: dict[str, Any],
    config: StabilityConfig,
    ckpt_mgr: CheckpointManager,
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

    return {
        "checkpoint_path": ckpt_mgr.path("stability_calibrated"),
        "metrics": metrics,
    }
