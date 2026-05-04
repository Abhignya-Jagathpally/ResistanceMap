"""Configuration dataclasses for ResistanceMap pipeline.

All hyperparameters, paths, and hardware settings are defined here.
Loaded from YAML via load_config().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml


@dataclass
class DataConfig:
    """Paths and parameters for data loading and preprocessing."""

    # Raw data directories
    ccle_proteomics_path: Path = Path("data/raw/ccle_proteomics.csv")
    ccle_epigenomics_dir: Path = Path("data/raw/ccle_epigenomics/")
    string_ppi_path: Path = Path("data/raw/string_ppi.txt")
    gdsc_path: Path = Path("data/raw/gdsc_drug_sensitivity.csv")
    ctrpv2_path: Path = Path("data/raw/ctrpv2_drug_sensitivity.csv")

    # Single-cell and patient data
    scrna_gse124310_path: Path = Path("data/raw/gse124310.h5ad")  # MM patient samples
    scrna_gse271107_path: Path = Path("data/raw/gse271107.h5ad")  # Lenalidomide response
    mmrf_commpass_dir: Path = Path("data/raw/mmrf_commpass/")  # Clinical + genomic data

    # Pre-aggregated scRNA pseudobulk (per-(sample, disease_stage)).
    # Built by scripts/build_scrna_summary.py. When present, harmonize_omics
    # attaches it to MultiOmicsDataset as a longitudinal HD/MGUS/SMM/MM axis.
    scrna_summary_path: Path = Path("checkpoints/scrna_summary.pt")

    # Tier-2 baseline datasets used by similar models. Optional — pipeline
    # degrades gracefully when absent. See docs/DATASETS_AND_BENCHMARKS.md
    # for the strategic rationale (DrugCell/MOLI/PaccMann use CRISPR + PRISM
    # alongside CCLE proteomics; HMCL is the MM-specific cell-line cohort).
    depmap_crispr_path: Path = Path("data/raw/depmap/CRISPRGeneEffect.csv")
    hmcl_keats_dir: Path = Path("data/raw/hmcl_keats/")
    prism_path: Path = Path("data/raw/prism/secondary-screen-dose-response-curve-parameters.csv")

    # Preprocessing
    min_coverage: float = 0.7  # Drop proteins missing in >30% of samples
    imputation: str = "knn"  # knn | median | zero
    normalization: str = "quantile"  # quantile | zscore | log2
    ppi_confidence: float = 0.7  # STRING combined score cutoff

    # Single-cell QC thresholds
    scrna_min_genes: int = 200
    scrna_max_genes: int = 8000
    scrna_min_counts: int = 1000
    scrna_max_mt: float = 0.2  # Max mitochondrial percentage

    # Splits
    test_fraction: float = 0.15
    val_fraction: float = 0.15
    random_seed: int = 42

    # Drug targets for trajectory modeling.
    # Restricted to MM-relevant compounds that exist in the GDSC cell-line
    # screen; see configs/default.yaml for the rationale and the standard-
    # of-care drugs that intentionally cannot live here.
    target_drugs: list[str] = field(default_factory=lambda: [
        "Bortezomib", "Lenalidomide", "Panobinostat", "Vorinostat",
        "Romidepsin", "Venetoclax", "Dinaciclib", "Palbociclib",
        "Doxorubicin", "Etoposide", "Cyclophosphamide",
    ])


@dataclass
class VAEConfig:
    """Hyperparameters for the Proteome-to-Epigenome conditional VAE."""

    # Architecture
    input_dim: int = 8000  # Number of proteins in feature vector
    epigenome_dim: int = 50000  # Number of ATAC-seq peaks to reconstruct
    latent_dim: int = 64  # Resistance state embedding dimension
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "gelu"

    # Training
    pretrain_epochs: int = 200
    finetune_epochs: int = 100
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 512
    gradient_clip_norm: float = 1.0

    # KL annealing (cyclical)
    kl_anneal_cycles: int = 4
    kl_anneal_ratio: float = 0.5
    kl_weight_max: float = 1.0

    # Reconstruction loss weights
    atac_weight: float = 1.0
    h3k4me3_weight: float = 0.5
    h3k27me3_weight: float = 0.5

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4

    # Domain-adversarial training (Agent 2 §2.4)
    conditioning_dim: int = 0  # FiLM conditioning dim (0 = disabled)
    domain_adversarial: bool = False
    adversarial_weight: float = 0.1
    adversarial_warmup_epochs: int = 10

    # Cross-modal VAE enhancements
    use_stochastic_decoder: bool = False  # Use stochastic decoder for uncertainty quantification
    expanded_latent_dim: Optional[int] = None  # Override latent_dim (e.g., 128, 256) to reduce bottleneck


@dataclass
class TrajectoryConfig:
    """Hyperparameters for ODE-based resistance trajectory modeling."""

    # ODE solver
    ode_solver: str = "euler"  # euler | dopri5 | rk4
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-7
    integration_time: float = 100.0  # Arbitrary time units for trajectory

    # Forecasting horizons
    forecast_horizons: list[int] = field(default_factory=lambda: [3, 6, 12])  # months

    # Monte Carlo sampling for uncertainty quantification
    n_perturbation: int = 50  # Samples for trajectory uncertainty

    # Calibration
    calibration_lr: float = 1e-3
    calibration_epochs: int = 500
    calibration_batch_size: int = 64

    # Output range
    score_range: tuple[float, float] = (0.0, 1.0)

    # Chromatin reader/writer proteins for ODE parameterization
    reader_writer_proteins: list[str] = field(default_factory=lambda: [
        "EZH2", "KDM6A", "KDM6B", "KMT2A", "KMT2D",
        "DNMT1", "DNMT3A", "DNMT3B", "TET1", "TET2",
        "HDAC1", "HDAC2", "KAT2A", "KAT2B", "EP300",
        "BRD4", "SMARCA4", "ARID1A", "SUZ12", "EED",
    ])

    # Neural Jump-SDE (Agent 4 §4.1-4.7)
    use_sde: bool = False
    sde_drift_hidden: int = 128
    sde_diffusion_hidden: int = 64
    sde_jump_hidden: int = 64
    sde_dt: float = 0.01
    sde_n_mc_samples: int = 50
    survival_calibrate: bool = False


# Alias for backward compatibility with trajectory module
StabilityConfig = TrajectoryConfig


@dataclass
class ProteinNetConfig:
    """Hyperparameters for protein network with ESM-2 embeddings."""

    # ESM-2 language model
    esm2_model: str = "facebook/esm2_t33_650M_UR50D"
    esm2_dim: int = 1280  # ESM2-T33 output dimension

    # Graph neural network
    gnn_hidden: int = 256
    gnn_layers: int = 4
    gnn_heads: int = 8  # For multi-head attention
    gnn_dropout: float = 0.2
    gnn_conv_type: str = "gat"  # gat | gcn | graphsage

    # PPI graph statistics
    ppi_proteins: int = 7853  # Typical STRING human network size
    ppi_edges: int = 460000  # Typical edge count

    # Anti-over-smoothing (Agent 2 §2.3, Agent 5 §5.1-5.8)
    dropedge_rate: float = 0.1
    use_pairnorm: bool = True
    use_jumping_knowledge: bool = True
    jk_mode: str = "cat"  # cat | max | lstm
    use_graphmask: bool = False
    graphmask_sparsity_weight: float = 0.01

    # ESM-2 bottleneck (Agent 5 §5.2)
    esm2_bottleneck_dim: int = 256

    # Drug-conditioned attention (Agent 5 §5.3)
    drug_embedding_dim: int = 0  # 0 = disabled
    n_drug_types: int = 11

    # Evidential classification (Agent 5 §5.6)
    use_evidential_head: bool = False
    evidential_n_classes: int = 3

    # Phosphoproteomics (Agent 5 §5.4)
    phospho_dim: int = 0  # 0 = disabled

    # STRING debiasing (Agent 5 §5.7)
    string_debias_textmining: bool = True
    string_debias_threshold: float = 0.7


@dataclass
class FusionConfig:
    """Hyperparameters for multi-modal fusion."""

    # Cross-attention fusion
    hidden_dim: int = 128
    n_heads: int = 4
    dropout: float = 0.2
    fusion_type: str = "cross_attention"  # cross_attention | concat | gated

    # Training
    fusion_lr: float = 5e-4
    fusion_epochs: int = 150
    weight_decay: float = 1e-4

    # Batch correction (Agent 2 §2.4, Agent 3 §3.3)
    batch_correction: bool = False
    n_batches: int = 10
    batch_embed_dim: int = 16

    # Missing modality handling (Agent 3 §3.3)
    handle_missing_modalities: bool = True


@dataclass
class LandscapeConfig:
    """Hyperparameters for resistance landscape visualization."""

    # Top resistance mechanism identification
    n_top_targets: int = 20
    confidence_threshold: float = 0.8

    # Visualization
    visualization: bool = True
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Evidential landscape (Agent 6 §6.4)
    use_evidential: bool = False


@dataclass
class APIConfig:
    """API server configuration."""

    port: int = 8000
    workers: int = 4
    host: str = "0.0.0.0"


@dataclass
class HardwareConfig:
    """GPU and distributed training settings."""

    device: str = "cuda"
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    compile: bool = True  # torch.compile
    compile_mode: str = "reduce-overhead"  # default | reduce-overhead | max-autotune
    pin_memory: bool = True
    num_workers: int = 8  # DataLoader workers
    prefetch_factor: int = 4

    # Distributed
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

    # Reproducibility
    deterministic: bool = False
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Evaluation governance layer settings.

    These flags only affect the orthogonal evaluation governance layer
    under :mod:`resistancemap.evaluation`; they have no effect on the
    training DAG. The training pipeline runs unchanged regardless of how
    these are set.
    """

    enabled: bool = True

    # Audit trail destination (one subdirectory per run_id).
    log_root: Path = Path("logs/evaluation")

    # Optional explicit run identifier; defaults to a UTC timestamp.
    run_id: Optional[str] = None

    # Rubric source. None falls back to the default rubric.yaml shipped
    # with the evaluation package.
    rubric_path: Optional[Path] = None

    # Tier gating: a Tier A FAIL always hard-stops downstream tiers. Set
    # to False to *demote* a Tier A FAIL into a CONDITIONAL warning
    # (intended only for offline rubric debugging — never for releases).
    tier_a_hard_stop: bool = True

    # Tier opt-outs (use to skip tiers when the necessary intake is not
    # yet available; the chair downgrades the report accordingly).
    skip_tier_b: bool = False
    skip_tier_c: bool = False
    skip_tier_d: bool = False


@dataclass
class ResistanceMapConfig:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    protein_net: ProteinNetConfig = field(default_factory=ProteinNetConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    landscape: LandscapeConfig = field(default_factory=LandscapeConfig)
    api: APIConfig = field(default_factory=APIConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    wandb_project: str = "resistancemap"
    resume_checkpoint: Optional[Path] = None

    @property
    def device(self) -> torch.device:
        """Get torch device based on configuration."""
        if self.hardware.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda", self.hardware.local_rank)
        return torch.device("cpu")

    @property
    def amp_dtype(self) -> torch.dtype:
        """Get automatic mixed precision dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.hardware.dtype, torch.bfloat16)


def load_config(path: Path) -> ResistanceMapConfig:
    """Load configuration from a YAML file, with defaults for missing fields.

    Args:
        path: Path to YAML config file.

    Returns:
        Fully populated ResistanceMapConfig.

    Raises:
        FileNotFoundError: If path does not exist (returns defaults).
    """
    if not path.exists():
        return ResistanceMapConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = ResistanceMapConfig()

    # Map nested YAML keys to dataclass fields
    section_map = {
        "data": (config.data, DataConfig),
        "vae": (config.vae, VAEConfig),
        "trajectory": (config.trajectory, TrajectoryConfig),
        "protein_net": (config.protein_net, ProteinNetConfig),
        "fusion": (config.fusion, FusionConfig),
        "landscape": (config.landscape, LandscapeConfig),
        "api": (config.api, APIConfig),
        "hardware": (config.hardware, HardwareConfig),
        "evaluation": (config.evaluation, EvaluationConfig),
    }

    for section_name, (section_obj, section_cls) in section_map.items():
        if section_name in raw:
            for key, value in raw[section_name].items():
                if hasattr(section_obj, key):
                    # Convert string paths to Path objects. Note: with
                    # `from __future__ import annotations`, dataclass field
                    # types are strings, so compare by name.
                    field_type = section_cls.__dataclass_fields__[key].type
                    type_name = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", str(field_type))
                    if (type_name == "Path" or "Path" in type_name) and value is not None:
                        value = Path(value)
                    elif type_name == "float" and isinstance(value, str):
                        value = float(value)
                    elif type_name == "int" and isinstance(value, str):
                        value = int(value)
                    setattr(section_obj, key, value)

    # Top-level fields
    for key in ("checkpoint_dir", "log_dir", "wandb_project"):
        if key in raw:
            value = raw[key]
            if key.endswith("_dir"):
                value = Path(value)
            setattr(config, key, value)

    return config