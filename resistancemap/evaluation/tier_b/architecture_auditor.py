"""Tier B: Architecture Auditor agent.

Maps every training component of the ResistanceMap pipeline to a precise
*scientific function* and emits a structured per-module audit covering
identifiability, what is identified vs assumed, and the ablation that
*should* falsify each design choice.

This agent does not run any training. It reads ``ResistanceMapConfig`` and
emits a finding consumed by Tier C/D agents.
"""

from __future__ import annotations

import logging
from typing import Any

from resistancemap.agents.base import AgentState
from resistancemap.config import ResistanceMapConfig
from resistancemap.evaluation.base import EvalAgent, EvalFinding, Verdict

logger = logging.getLogger(__name__)


class ArchitectureAuditorAgent(EvalAgent):
    """Audit each ResistanceMap training component for scientific coherence.

    Module roles assumed by this agent (one row per criterion in the finding):

    - **VAE latent**       : state summary of the proteomic / epigenomic profile.
    - **ODE layer**        : continuous-time dynamics of the latent state.
    - **GNN (PPI)**        : pathway / interaction context for protein effects.
    - **Fusion**           : how heterogeneous modalities are combined.

    For each module the auditor records:
        identified         : claims that the data + loss can recover.
        assumed            : claims baked into the architecture and *not*
                             tested by training.
        identifiability    : the explicit identifiability gap and what
                             constraint or experiment would close it.
        ablation_plan      : what each ablation is supposed to *break* — not
                             a list of ablations we have, but the falsification
                             prediction tied to each.
    """

    tier = "B"

    def __init__(self) -> None:
        super().__init__(
            name="architecture_auditor",
            dependencies=[],
        )

    def verify_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str]:
        """No upstream training output is required; this is a static audit."""
        return True, ""

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        """Run the architecture audit and return a structured EvalFinding."""
        logger.info("ArchitectureAuditorAgent: starting structured audit")

        criteria_results: dict[str, dict[str, Any]] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # ---------- VAE latent ----------
        vae_audit = self._audit_vae(config)
        criteria_results["vae_latent"] = vae_audit
        if vae_audit["identifiability_gap"]:
            failure_modes.append(
                "VAE latent geometry is only identified up to rotation/sign "
                "without anchoring constraints."
            )
            required_changes.append(
                "Add a latent identifiability constraint (e.g., orthogonal "
                "anchors, contrastive supervision, or fixed reference samples) "
                "before any latent-axis interpretation is reported."
            )

        # ---------- ODE / trajectory ----------
        ode_audit = self._audit_trajectory(config)
        criteria_results["ode_dynamics"] = ode_audit
        if ode_audit["arbitrary_time_units"]:
            failure_modes.append(
                f"trajectory.integration_time={config.trajectory.integration_time} "
                "is in *arbitrary units*; calendar-time `when`-claims are not "
                "supported until the integrator is calibrated to clinical time."
            )
            required_changes.append(
                "Calibrate ODE integration time to calendar time using a paired "
                "longitudinal dataset before emitting any time-stamped resistance "
                "forecasts."
            )

        # ---------- GNN / PPI context ----------
        gnn_audit = self._audit_protein_net(config)
        criteria_results["gnn_pathway_context"] = gnn_audit
        if gnn_audit["confounded_attribution_risk"]:
            failure_modes.append(
                "GNN attributions are confounded with PPI degree until a "
                "degree-controlled null model is run."
            )
            required_changes.append(
                "Run pathway attribution against a degree-preserving rewired "
                "PPI null and report effect-size relative to that null."
            )

        # ---------- Fusion ----------
        fusion_audit = self._audit_fusion(config)
        criteria_results["fusion"] = fusion_audit
        if fusion_audit["modality_collapse_risk"]:
            failure_modes.append(
                "Cross-attention fusion can collapse onto a single dominant "
                "modality without per-modality dropout ablations."
            )
            required_changes.append(
                "Run leave-one-modality-out ablations and report the predicted "
                "performance drop for each modality; missing or zero-drop "
                "modalities indicate collapse."
            )

        n_modules = len(criteria_results)
        n_clean = sum(
            1 for r in criteria_results.values() if not r.get("issues_found")
        )
        score = float(n_clean) / float(n_modules)

        if score == 1.0:
            verdict = Verdict.PASS
        elif score >= 0.5:
            verdict = Verdict.CONDITIONAL
        else:
            verdict = Verdict.FAIL

        evidence = {
            "n_modules_audited": n_modules,
            "modules_clean": n_clean,
            "config_signature": {
                "vae.latent_dim": config.vae.latent_dim,
                "trajectory.integration_time": config.trajectory.integration_time,
                "trajectory.ode_solver": config.trajectory.ode_solver,
                "protein_net.gnn_conv_type": config.protein_net.gnn_conv_type,
                "fusion.fusion_type": config.fusion.fusion_type,
            },
        }

        finding = EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )
        logger.info(
            "ArchitectureAuditorAgent: verdict=%s score=%.2f modules=%d",
            verdict, score, n_modules,
        )
        return finding

    # ------------------------------------------------------------------
    # Per-module audit helpers
    # ------------------------------------------------------------------

    def _audit_vae(self, config: ResistanceMapConfig) -> dict[str, Any]:
        """Audit the VAE latent (scientific function: state summary)."""
        latent_dim = config.vae.latent_dim
        return {
            "scientific_function": "state summary of proteomic+epigenomic profile",
            "identified": [
                "Marginal data likelihood under the chosen reconstruction loss",
                f"A latent of dimension {latent_dim} that minimizes ATAC + "
                "histone reconstruction error",
            ],
            "assumed": [
                "Gaussian prior over latents (no tissue/donor stratification)",
                "ATAC + H3K4me3 + H3K27me3 fully describe the epigenomic state",
                "Latent axes are *interpretable* (they are not, by default)",
            ],
            "identifiability": (
                "VAE latents are only identified up to an orthogonal rotation "
                "and per-axis sign flip. Any plot or claim that pins meaning "
                "to a *specific* latent axis is unfounded without an anchor "
                "(e.g., contrastive supervision, identifiable VAE / iVAE, or "
                "a fixed set of reference samples)."
            ),
            "ablation_plan": {
                "drop_h3k27me3_decoder": (
                    "should *break* recovery of repressed-chromatin states; if "
                    "performance is unchanged, the H3K27me3 head is decorative."
                ),
                "shuffle_protein_to_epigenome_pairs": (
                    "should collapse the conditional generation quality to the "
                    "marginal; if not, the model is ignoring conditioning."
                ),
                "freeze_latent_to_PCA": (
                    "should hurt downstream resistance prediction if and only "
                    "if non-linear latent geometry matters."
                ),
            },
            "identifiability_gap": True,
            "issues_found": True,
        }

    def _audit_trajectory(self, config: ResistanceMapConfig) -> dict[str, Any]:
        """Audit the ODE layer (scientific function: latent dynamics)."""
        t_end = config.trajectory.integration_time
        horizons = list(config.trajectory.forecast_horizons)
        return {
            "scientific_function": "continuous-time dynamics of the resistance latent state",
            "identified": [
                "Local vector field of the latent ODE up to a time reparameterization",
                f"Stationary points reachable within integration_time={t_end}",
            ],
            "assumed": [
                "The drift function is autonomous (no exogenous treatment time series)",
                "Cells follow population-average dynamics (no per-patient drift)",
                f"Output horizons {horizons} are in months and aligned with calendar time",
            ],
            "identifiability": (
                "ODE flows are identifiable only up to monotonic time "
                "reparameterization. ``integration_time`` is in arbitrary units; "
                "without a paired longitudinal dataset (e.g., MMRF CoMMpass "
                "serial timepoints) we cannot map ODE-time to months. Any "
                "`when`-claim must be deferred until calibration is done."
            ),
            "ablation_plan": {
                "replace_ode_with_linear_drift": (
                    "should *break* curvature-dependent forecasts; if not, the "
                    "ODE adds no value over a linear extrapolator."
                ),
                "permute_initial_conditions": (
                    "should destroy per-patient forecasts; if not, the model "
                    "is forecasting the population mean."
                ),
                "halve_integration_time": (
                    "should change long-horizon forecasts but not short-horizon; "
                    "if both change identically, time has no physical meaning."
                ),
            },
            "arbitrary_time_units": True,
            "issues_found": True,
        }

    def _audit_protein_net(self, config: ResistanceMapConfig) -> dict[str, Any]:
        """Audit the PPI GNN (scientific function: pathway context)."""
        return {
            "scientific_function": "pathway / interaction context for protein-level effects",
            "identified": [
                f"A {config.protein_net.gnn_layers}-layer "
                f"{config.protein_net.gnn_conv_type.upper()} embedding over "
                f"{config.protein_net.ppi_proteins} proteins",
                "Neighborhood smoothness consistent with the PPI edge set",
            ],
            "assumed": [
                "STRING PPI edges represent functional interactions, not "
                "literature-citation bias",
                "Edge weights from the STRING combined score are calibrated",
                "ESM-2 embeddings encode binding context relevant to MM "
                "resistance, not just sequence similarity",
            ],
            "identifiability": (
                "Attention-based GNN attributions are confounded with node "
                "*degree* in the PPI: high-degree hubs accumulate attribution "
                "regardless of biology. Without a degree-preserving null we "
                "cannot tell pathway signal from hub artifact."
            ),
            "ablation_plan": {
                "swap_ppi_for_degree_preserving_rewire": (
                    "should *break* pathway-level attributions; if not, the "
                    "model is keying off node degree alone."
                ),
                "drop_esm2_embeddings_use_one_hot": (
                    "should hurt OOD generalization to held-out proteins; if "
                    "not, the ESM-2 features are decorative."
                ),
                "edge_confidence_threshold_sweep": (
                    f"varying ppi_confidence around the current "
                    f"{config.data.ppi_confidence} should produce a "
                    "monotonic stability curve; non-monotonicity flags "
                    "overfitting to the chosen cutoff."
                ),
            },
            "confounded_attribution_risk": True,
            "issues_found": True,
        }

    def _audit_fusion(self, config: ResistanceMapConfig) -> dict[str, Any]:
        """Audit the multi-modal fusion (scientific function: combine modalities)."""
        return {
            "scientific_function": "combine VAE latent, trajectory state, and pathway context into a unified resistance prediction",
            "identified": [
                f"Cross-modal attention weights of dimension {config.fusion.hidden_dim}",
                "Per-task prediction error contribution per modality (only if "
                "leave-one-out ablations are run)",
            ],
            "assumed": [
                "All modalities are observed for every patient (no MNAR handling)",
                f"{config.fusion.fusion_type} attention is the right inductive bias",
                "Modalities are exchangeable in the attention layer",
            ],
            "identifiability": (
                "Cross-attention fusion can be dominated by the lowest-noise "
                "modality, leaving the others with negligible gradient. Without "
                "leave-one-modality-out ablations we cannot tell whether the "
                "fusion is multi-modal or merely re-weighting one modality."
            ),
            "ablation_plan": {
                "leave_one_modality_out": (
                    "each held-out modality should produce a measurable, "
                    "non-trivial drop in held-out performance; a zero-drop "
                    "modality indicates collapse."
                ),
                "shuffle_modality_alignment": (
                    "should destroy patient-level predictions if modalities are "
                    "actually being fused; if not, fusion is just averaging."
                ),
                "swap_cross_attention_for_concat": (
                    "should match cross-attention performance only on tasks "
                    "where modality interactions are linear."
                ),
            },
            "modality_collapse_risk": True,
            "issues_found": True,
        }

    # ------------------------------------------------------------------
    # BaseAgent compatibility shim
    # ------------------------------------------------------------------

    async def execute(self, inputs, config):  # type: ignore[override]
        """BaseAgent.execute shim — delegates to ``assess`` and wraps the result."""
        finding = await self.assess(inputs, config)
        return self._make_result(
            AgentState.COMPLETED,
            output=finding.__dict__ if hasattr(finding, "__dict__") else finding,
            metadata={"verdict": str(finding.verdict), "score": finding.score, "finding": finding},
        )
