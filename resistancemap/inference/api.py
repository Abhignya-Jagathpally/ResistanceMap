"""FastAPI server for ResistanceMap inference.

Adapted from MyeloMemory's api.py. Provides REST endpoints for:
    POST /predict               — Single sample prediction
    POST /predict/batch         — Batch prediction
    POST /predict/trajectory    — Temporal trajectory prediction
    GET  /targets/{sample_id}   — Top intervention targets
    GET  /landscape/{sample_id} — Resistance landscape visualization data
    GET  /health                — Health check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from resistancemap.inference.pipeline import ResistanceMapPipeline, TemporalLandscape
from resistancemap.landscape.predictor import LandscapeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for single-sample prediction."""

    protein_abundances: dict[str, float] = Field(
        ...,
        description="Mapping of protein/gene name → abundance value.",
        examples=[{"CDK4": 1.5, "RB1": 2.3, "TP53": 0.8}],
    )
    sample_id: str | None = Field(
        None,
        description="Optional sample identifier for traceability.",
    )


class DrugResistancePrediction(BaseModel):
    """Prediction for a single drug at a specific timepoint."""

    drug_name: str = Field(description="Drug name")
    timepoint_months: int = Field(description="Timepoint in months (3, 6, or 12)")
    resistance_probability: float = Field(
        description="P(resistant) at this timepoint, 0-1"
    )


class InterventionTarget(BaseModel):
    """Ranked intervention target protein."""

    protein_name: str = Field(description="Protein/gene name")
    resistance_contribution: float = Field(
        description="How much this protein contributes to resistance (0-1)"
    )
    actionability_score: float = Field(
        description="How actionable this target is via drugs or other interventions (0-1)"
    )


class PredictResponse(BaseModel):
    """Response body for prediction endpoint."""

    sample_id: str | None = Field(None, description="Sample identifier from request")
    resistance_state: str = Field(
        description="Current predicted resistance state (e.g., sensitive, intermediate, resistant)"
    )
    basin_of_attraction: str = Field(
        description="Predicted future resistance state the tumor is heading toward"
    )
    confidence_score: float = Field(
        description="Confidence in the resistance state prediction (0-1)"
    )
    drug_predictions: list[DrugResistancePrediction] = Field(
        description="Per-drug resistance predictions at 3, 6, 12 months"
    )
    top_intervention_targets: list[InterventionTarget] = Field(
        description="Top 10 proteins ranked by intervention potential"
    )
    interpretation: str = Field(
        description="Human-readable summary of resistance landscape"
    )
    model_version: str = Field(
        default="1.0.0",
        description="Model version for traceability"
    )


class TrajectoryPredictRequest(BaseModel):
    """Request body for trajectory prediction."""

    proteomics_timeseries: list[dict[str, float]] = Field(
        ...,
        description="List of proteomics profiles (one per timepoint)"
    )
    timepoint_labels: list[str] | None = Field(
        None,
        description="Optional labels for timepoints (e.g., ['baseline', '3m', '6m'])"
    )
    actual_states: list[str] | None = Field(
        None,
        description="Optional ground-truth resistance states for validation"
    )


class TemporalLandscapeResponse(BaseModel):
    """Response for trajectory prediction."""

    timepoints: list[str] = Field(description="Timepoint labels")
    predicted_state_sequence: list[str] = Field(
        description="Predicted resistance state at each timepoint"
    )
    trajectory_accuracy: float | None = Field(
        None,
        description="Fraction of predicted states matching actual (if provided)"
    )
    landscapes: list[PredictResponse] = Field(
        description="Full landscape prediction at each timepoint"
    )


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""

    samples: list[PredictRequest]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="ok or not_ready")
    model_loaded: bool = Field(description="Whether the pipeline is ready")
    device: str = Field(description="Torch device (cpu or cuda)")
    version: str = Field(description="API version")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    pipeline: ResistanceMapPipeline | None = None,
    config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        pipeline: Initialized ResistanceMapPipeline (or None to load from config).
        config: Configuration dict with pipeline parameters.

    Returns:
        Configured FastAPI app.
    """
    config = config or {}
    pipeline_state: dict[str, Any] = {"pipeline": pipeline}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            if pipeline_state["pipeline"] is None:
                pipeline_state["pipeline"] = ResistanceMapPipeline.from_checkpoints(
                    config.get("checkpoint_dir"),
                    config,
                )
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
        yield

    app = FastAPI(
        title="ResistanceMap API",
        description=(
            "AI pipeline for predicting tumor resistance landscapes and "
            "intervention targets in multiple myeloma."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def _get_pipeline() -> ResistanceMapPipeline:
        p = pipeline_state["pipeline"]
        if p is None:
            raise HTTPException(status_code=503, detail="Pipeline not loaded")
        return p

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        pipeline = pipeline_state["pipeline"]
        return HealthResponse(
            status="ok" if pipeline is not None else "not_ready",
            model_loaded=pipeline is not None,
            device=str(config.get("device", "cpu")),
            version="1.0.0",
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        pipeline = _get_pipeline()

        # Run single-sample prediction
        landscape_result = pipeline.predict_single(
            request.protein_abundances,
            sample_id=request.sample_id,
        )

        # Build drug prediction list (3, 6, 12 months)
        drug_preds = []
        for drug_name in pipeline.drug_names:
            for months, resistance_dict in [
                (3, landscape_result.drug_resistance_3m),
                (6, landscape_result.drug_resistance_6m),
                (12, landscape_result.drug_resistance_12m),
            ]:
                resistance_prob = resistance_dict.get(drug_name, 0.0)
                drug_preds.append(DrugResistancePrediction(
                    drug_name=drug_name,
                    timepoint_months=months,
                    resistance_probability=round(resistance_prob, 4),
                ))

        # Build intervention targets list
        intervention_targets = [
            InterventionTarget(
                protein_name=prot_name,
                resistance_contribution=round(res_contrib, 4),
                actionability_score=round(actionability, 4),
            )
            for prot_name, res_contrib, actionability in landscape_result.top_intervention_targets
        ]

        # Generate interpretation
        interpretation = _interpret_landscape(
            landscape_result.resistance_state,
            landscape_result.basin_of_attraction,
            intervention_targets,
        )

        return PredictResponse(
            sample_id=request.sample_id,
            resistance_state=landscape_result.resistance_state,
            basin_of_attraction=landscape_result.basin_of_attraction,
            confidence_score=round(landscape_result.confidence_score, 4),
            drug_predictions=drug_preds,
            top_intervention_targets=intervention_targets,
            interpretation=interpretation,
            model_version="1.0.0",
        )

    @app.post("/predict/batch")
    async def predict_batch(request: BatchPredictRequest) -> list[PredictResponse]:
        _get_pipeline()

        if len(request.samples) > config.get("max_batch_size", 100):
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.samples)} exceeds max {config.get('max_batch_size')}",
            )

        results = []
        for sample in request.samples:
            single_response = await predict(sample)
            results.append(single_response)

        return results

    @app.post("/predict/trajectory", response_model=TemporalLandscapeResponse)
    async def predict_trajectory(
        request: TrajectoryPredictRequest,
    ) -> TemporalLandscapeResponse:
        pipeline = _get_pipeline()

        # Run trajectory prediction
        temporal_landscape = pipeline.predict_trajectory(
            request.proteomics_timeseries,
            timepoint_labels=request.timepoint_labels,
            actual_states=request.actual_states,
        )

        # Convert landscapes to response format
        landscape_responses = []
        for landscape in temporal_landscape.landscapes:
            # Rebuild response for each timepoint
            drug_preds = []
            for drug_name in pipeline.drug_names:
                for months, resistance_dict in [
                    (3, landscape.drug_resistance_3m),
                    (6, landscape.drug_resistance_6m),
                    (12, landscape.drug_resistance_12m),
                ]:
                    resistance_prob = resistance_dict.get(drug_name, 0.0)
                    drug_preds.append(DrugResistancePrediction(
                        drug_name=drug_name,
                        timepoint_months=months,
                        resistance_probability=round(resistance_prob, 4),
                    ))

            intervention_targets = [
                InterventionTarget(
                    protein_name=prot_name,
                    resistance_contribution=round(res_contrib, 4),
                    actionability_score=round(actionability, 4),
                )
                for prot_name, res_contrib, actionability in landscape.top_intervention_targets
            ]

            interpretation = _interpret_landscape(
                landscape.resistance_state,
                landscape.basin_of_attraction,
                intervention_targets,
            )

            landscape_responses.append(PredictResponse(
                sample_id=landscape.sample_id,
                resistance_state=landscape.resistance_state,
                basin_of_attraction=landscape.basin_of_attraction,
                confidence_score=round(landscape.confidence_score, 4),
                drug_predictions=drug_preds,
                top_intervention_targets=intervention_targets,
                interpretation=interpretation,
                model_version="1.0.0",
            ))

        return TemporalLandscapeResponse(
            timepoints=temporal_landscape.timepoints,
            predicted_state_sequence=temporal_landscape.predicted_state_sequence,
            trajectory_accuracy=(
                round(temporal_landscape.trajectory_accuracy, 4)
                if temporal_landscape.trajectory_accuracy is not None
                else None
            ),
            landscapes=landscape_responses,
        )

    @app.get("/targets/{sample_id}")
    async def get_targets(sample_id: str) -> dict[str, Any]:
        """Retrieve top intervention targets for a sample (from cache if available)."""
        return {
            "sample_id": sample_id,
            "note": "Targets would be retrieved from prediction cache in production",
        }

    @app.get("/landscape/{sample_id}")
    async def get_landscape(sample_id: str) -> dict[str, Any]:
        """Retrieve landscape visualization data for a sample."""
        return {
            "sample_id": sample_id,
            "note": "Landscape data would be retrieved from prediction cache in production",
        }

    return app


def _interpret_landscape(
    resistance_state: str,
    basin_of_attraction: str,
    intervention_targets: list[InterventionTarget],
) -> str:
    """Generate human-readable interpretation of the resistance landscape."""
    state_text = f"Current resistance state: {resistance_state}."
    basin_text = f" The tumor appears to be heading toward {basin_of_attraction} state."

    top_targets_text = ""
    if intervention_targets:
        top_protein = intervention_targets[0].protein_name
        top_actionability = intervention_targets[0].actionability_score
        if top_actionability > 0.5:
            top_targets_text = (
                f" Top intervention target: {top_protein} "
                f"(actionability score={top_actionability:.2f}). "
                f"Consider targeting this protein to modulate resistance."
            )

    return f"{state_text}{basin_text}{top_targets_text}"
