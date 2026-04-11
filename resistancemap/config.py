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

    # Drug targets for trajectory modeling
    target_drugs: list[str] = field(default_factory=lambda: [
        "Bortezomib", "Lenalidomide", "Dexamethasone",
        "Carfilzomib", "Pomalidomide", "Daratumumab",
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
    }

    for section_name, (section_obj, section_cls) in section_map.items():
        if section_name in raw:
            for key, value in raw[section_name].items():
                if hasattr(section_obj, key):
                    # Convert string paths to Path objects
                    field_type = section_cls.__dataclass_fields__[key].type
                    if field_type == Path or field_type == "Path":
                        value = Path(value)
                    setattr(section_obj, key, value)

    # Top-level fields
    for key in ("checkpoint_dir", "log_dir", "wandb_project"):
        if key in raw:
            value = raw[key]
            if key.endswith("_dir"):
                value = Path(value)
            setattr(config, key, value)

    return config
