"""Dataset loaders for CCLE, single-cell, MMRF CoMMpass, and drug sensitivity.

Each loader returns a standardized dictionary with tensors and metadata.
All loaders handle missing data, format conversion, and basic validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from resistancemap.config import DataConfig

logger = logging.getLogger(__name__)


class MultiOmicsDataset(Dataset):
    """PyTorch dataset holding matched proteomics + epigenomics for cell lines and samples.

    Attributes:
        proteomics: (N, P) tensor of protein abundances.
        epigenomics: (N, E) tensor of ATAC-seq / histone mark signals.
        sample_ids: List of sample identifiers (cell line IDs or patient IDs).
        lineage: List of tissue lineage labels.
        drug_sensitivity: (N, D) tensor of IC50 values (NaN where missing).
        protein_names: List of protein/gene names for the P columns.
        epigenome_feature_names: List of epigenomic feature identifiers.
        ppi_edges: Optional list of (protein1, protein2) tuples.
        ppi_scores: Optional list of PPI confidence scores.
        source: Origin of samples ('ccle_cell_line', 'scrna_patient', 'mmrf').
    """

    def __init__(
        self,
        proteomics: torch.Tensor,
        epigenomics: torch.Tensor,
        sample_ids: list[str],
        lineage: list[str],
        drug_sensitivity: torch.Tensor,
        protein_names: list[str],
        epigenome_feature_names: list[str],
        ppi_edges: Optional[list[tuple[str, str]]] = None,
        ppi_scores: Optional[list[float]] = None,
        source: str = "ccle_cell_line",
    ) -> None:
        """Initialize MultiOmicsDataset.

        Args:
            proteomics: (N, P) protein abundance tensor.
            epigenomics: (N, E) epigenomic signal tensor.
            sample_ids: List of N sample identifiers.
            lineage: List of N lineage labels.
            drug_sensitivity: (N, D) IC50 tensor or (N, 1).
            protein_names: List of P protein names.
            epigenome_feature_names: List of E feature names.
            ppi_edges: Optional PPI edges for graph construction.
            ppi_scores: Optional PPI confidence scores.
            source: Source label for the data.
        """
        assert proteomics.shape[0] == epigenomics.shape[0] == len(sample_ids)
        self.proteomics = proteomics
        self.epigenomics = epigenomics
        self.sample_ids = sample_ids
        self.lineage = lineage
        self.drug_sensitivity = drug_sensitivity
        self.protein_names = protein_names
        self.epigenome_feature_names = epigenome_feature_names
        self.ppi_edges = ppi_edges
        self.ppi_scores = ppi_scores
        self.source = source

    def __len__(self) -> int:
        return self.proteomics.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "proteomics": self.proteomics[idx],
            "epigenomics": self.epigenomics[idx],
            "drug_sensitivity": self.drug_sensitivity[idx],
        }

    def subset_by_lineage(self, lineages: list[str]) -> "MultiOmicsDataset":
        """Return a new dataset filtered to specific tissue lineages."""
        mask = [lin in lineages for lin in self.lineage]
        indices = [i for i, m in enumerate(mask) if m]
        return MultiOmicsDataset(
            proteomics=self.proteomics[indices],
            epigenomics=self.epigenomics[indices],
            sample_ids=[self.sample_ids[i] for i in indices],
            lineage=[self.lineage[i] for i in indices],
            drug_sensitivity=self.drug_sensitivity[indices],
            protein_names=self.protein_names,
            epigenome_feature_names=self.epigenome_feature_names,
            ppi_edges=self.ppi_edges,
            ppi_scores=self.ppi_scores,
            source=self.source,
        )


def load_ccle_proteomics(config: DataConfig) -> dict[str, Any]:
    """Load CCLE proteomics data from DepMap.

    Expected format: CSV with cell lines as rows, proteins as columns.
    First column is cell line ID.

    Args:
        config: DataConfig with path to CCLE proteomics.

    Returns:
        Dict with keys: 'data' (DataFrame), 'sample_ids', 'protein_names'.
    """
    path = config.ccle_proteomics_path
    logger.info(f"Loading CCLE proteomics from {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"CCLE proteomics not found at {path}. "
            "Run scripts/download_data.sh first."
        )

    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded proteomics: {df.shape[0]} cell lines, {df.shape[1]} proteins")

    # Drop proteins with too much missing data
    coverage = df.notna().mean(axis=0)
    kept = coverage >= config.min_coverage
    df = df.loc[:, kept]
    logger.info(
        f"After coverage filter ({config.min_coverage:.0%}): "
        f"{df.shape[1]} proteins retained"
    )

    # Map UniProt IDs to gene symbols if available
    mapping_path = path.parent / "uniprot_hugo_mapping.csv"
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path)
        uniprot_to_gene = dict(zip(mapping_df["UniprotID"], mapping_df["Symbol"]))
        old_cols = df.columns.tolist()
        new_cols = [uniprot_to_gene.get(c, c) for c in old_cols]
        n_mapped = sum(1 for o, n in zip(old_cols, new_cols) if o != n)
        df.columns = new_cols
        df = df.loc[:, ~df.columns.duplicated()]
        logger.info(f"Mapped {n_mapped}/{len(old_cols)} UniProt IDs to gene symbols")

    return {
        "data": df,
        "sample_ids": df.index.tolist(),
        "protein_names": df.columns.tolist(),
    }


def load_ccle_epigenomics(config: DataConfig) -> dict[str, Any]:
    """Load CCLE epigenomic data (chromatin profiling, ATAC, histone marks).

    Expected directory structure:
        ccle_epigenomics/
        ├── chromatin_profiling.csv
        ├── atac_seq.csv
        ├── h3k4me3.csv
        └── h3k27me3.csv

    Args:
        config: DataConfig with epigenomics directory path.

    Returns:
        Dict with keys for each loaded assay, 'sample_ids', 'feature_names'.
    """
    epi_dir = config.ccle_epigenomics_dir
    logger.info(f"Loading CCLE epigenomics from {epi_dir}")

    if not epi_dir.exists():
        raise FileNotFoundError(
            f"CCLE epigenomics directory not found at {epi_dir}. "
            "Run scripts/download_data.sh first."
        )

    result = {}
    all_feature_names = []

    # Try chromatin profiling first
    chrom_path = epi_dir / "chromatin_profiling.csv"
    if chrom_path.exists():
        df = pd.read_csv(chrom_path)
        if "BroadID" in df.columns:
            df = df.set_index("BroadID")
        else:
            df = df.set_index(df.columns[0])
        df = df.select_dtypes(include=["number"])
        result["chromatin_profiling"] = df
        all_feature_names.extend([f"chromatin:{col}" for col in df.columns])
        logger.info(f"  chromatin_profiling: {df.shape[0]} lines, {df.shape[1]} features")

    # Load individual assay files
    for assay in ("atac_seq", "h3k4me3", "h3k27me3"):
        csv_path = epi_dir / f"{assay}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            result[assay] = df
            all_feature_names.extend([f"{assay}:{col}" for col in df.columns])
            logger.info(f"  {assay}: {df.shape[0]} lines, {df.shape[1]} features")

    # Determine sample IDs from first available source
    for key in ("chromatin_profiling", "atac_seq", "h3k4me3", "h3k27me3"):
        if key in result:
            result["sample_ids"] = result[key].index.tolist()
            break
    else:
        result["sample_ids"] = []

    if not all_feature_names:
        raise FileNotFoundError(
            f"No epigenomic data files found in {epi_dir}. "
            "Need at least one of: chromatin_profiling.csv, atac_seq.csv, "
            "h3k4me3.csv, h3k27me3.csv"
        )

    result["feature_names"] = all_feature_names
    return result


def load_string_ppi(config: DataConfig) -> dict[str, Any]:
    """Load STRING protein-protein interaction network.

    Expected format: TSV with columns [protein1, protein2, combined_score].

    Args:
        config: DataConfig with path to STRING PPI.

    Returns:
        Dict with 'edges', 'scores', 'proteins', 'num_edges', 'num_proteins'.
    """
    path = config.string_ppi_path
    logger.info(f"Loading STRING PPI network from {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"STRING PPI not found at {path}. "
            "Run scripts/download_data.sh first."
        )

    df = pd.read_csv(path, sep="\t")
    expected_cols = {"protein1", "protein2", "combined_score"}
    if not expected_cols.issubset(df.columns):
        df.columns = ["protein1", "protein2", "combined_score"]

    # Normalize STRING scores to 0-1 if needed
    if df["combined_score"].max() > 1.0:
        logger.info("  STRING scores on 0-1000 scale — normalizing to 0-1")
        df["combined_score"] = df["combined_score"] / 1000.0

    # Filter by confidence threshold
    threshold = config.ppi_confidence
    df = df[df["combined_score"] >= threshold]
    logger.info(
        f"PPI edges after confidence filter (>={threshold}): {len(df)}"
    )

    edges = list(zip(df["protein1"], df["protein2"]))
    scores = df["combined_score"].tolist()
    proteins = set(df["protein1"]) | set(df["protein2"])

    return {
        "edges": edges,
        "scores": scores,
        "proteins": proteins,
        "num_edges": len(edges),
        "num_proteins": len(proteins),
    }


def load_scrna_h5ad(path: Path, config: DataConfig) -> dict[str, Any]:
    """Load single-cell RNA-seq data from h5ad file.

    Expected to contain:
        - counts matrix (X)
        - optional cell metadata (obs)
        - optional feature metadata (var)

    Args:
        path: Path to .h5ad file.
        config: DataConfig with scRNA-seq QC parameters.

    Returns:
        Dict with 'adata' (AnnData object), 'sample_ids', 'gene_names'.
    """
    try:
        import anndata
    except ImportError:
        raise ImportError("anndata is required for scRNA-seq loading. Install with: pip install anndata")

    logger.info(f"Loading scRNA-seq from {path}")

    if not path.exists():
        raise FileNotFoundError(f"scRNA-seq file not found at {path}")

    adata = anndata.read_h5ad(path)
    logger.info(
        f"Loaded scRNA-seq: {adata.n_obs} cells, {adata.n_vars} genes"
    )

    # Basic QC thresholds can be applied downstream in preprocess_scrna()
    return {
        "adata": adata,
        "sample_ids": adata.obs.index.tolist() if hasattr(adata.obs, "index") else [f"cell_{i}" for i in range(adata.n_obs)],
        "gene_names": adata.var.index.tolist() if hasattr(adata.var, "index") else [f"gene_{i}" for i in range(adata.n_vars)],
    }


def load_drug_sensitivity(config: DataConfig) -> dict[str, Any]:
    """Load drug sensitivity data from GDSC and CTRPv2.

    Merges sources, preferring GDSC for duplicates.

    Args:
        config: DataConfig with paths to sensitivity data.

    Returns:
        Dict with 'data' (cell line × drug IC50 matrix), 'sample_ids', 'drug_names'.
    """
    logger.info("Loading drug sensitivity data")

    frames = []

    for name, path in [("GDSC", config.gdsc_path), ("CTRPv2", config.ctrpv2_path)]:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            logger.info(f"  {name}: {len(df)} records")
            df = _standardize_drug_columns(df, name)
            frames.append(df)
        else:
            logger.warning(f"  {name}: not found at {path}, skipping")

    if not frames:
        raise FileNotFoundError("No drug sensitivity data found.")

    combined = pd.concat(frames, ignore_index=True)

    # Filter to target drugs
    target = [d.lower() for d in config.target_drugs]
    if "drug_name" not in combined.columns:
        raise ValueError(
            "Drug sensitivity data has no 'drug_name' column after standardization."
        )
    combined["drug_lower"] = combined["drug_name"].str.lower()
    combined = combined[combined["drug_lower"].isin(target)]

    # Pivot to matrix
    pivot = combined.pivot_table(
        index="sample_id",
        columns="drug_name",
        values="ic50",
        aggfunc="median",
    )

    logger.info(
        f"Drug sensitivity matrix: {pivot.shape[0]} samples × {pivot.shape[1]} drugs"
    )

    return {
        "data": pivot,
        "sample_ids": pivot.index.tolist(),
        "drug_names": pivot.columns.tolist(),
    }


def _standardize_drug_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Remap drug sensitivity columns to standard format.

    Standard output: sample_id, drug_name, ic50.

    Args:
        df: Input DataFrame with various column names.
        source_name: Name of data source (for logging).

    Returns:
        DataFrame with standardized column names.
    """
    col_map = {}
    cols_lower = {c: c.lower().strip().replace(" ", "_") for c in df.columns}

    for orig, lower in cols_lower.items():
        if lower in ("arxspan_id", "depmap_id", "modelid", "patient_id"):
            col_map[orig] = "sample_id"
        elif lower in ("cell_line_name", "ccle_name", "cclename"):
            if "sample_id" not in col_map.values():
                col_map[orig] = "sample_id"
        elif lower == "drug_name":
            col_map[orig] = "drug_name"
        elif lower == "cpd_name" and "drug_name" not in col_map.values():
            col_map[orig] = "drug_name"
        elif lower in ("ic50_published", "ic50"):
            col_map[orig] = "ic50"
        elif lower == "log2.ic50" and "ic50" not in col_map.values():
            col_map[orig] = "log2_ic50"

    df = df.rename(columns=col_map)

    # Convert log2 IC50 to linear if needed
    if "ic50" not in df.columns and "log2_ic50" in df.columns:
        df["ic50"] = 2.0 ** df["log2_ic50"]

    logger.info(f"  {source_name}: standardized columns")
    return df


def load_scrna_data(config: DataConfig) -> dict[str, Any]:
    """Load all available scRNA-seq datasets.

    Loads GSE124310 (MM patient samples) and GSE271107 (lenalidomide response)
    if their files exist. Returns empty dict if neither is found.

    Args:
        config: DataConfig with scRNA-seq paths.

    Returns:
        Dict with loaded AnnData objects keyed by accession.
    """
    result = {}

    for name, path in [
        ("GSE124310", config.scrna_gse124310_path),
        ("GSE271107", config.scrna_gse271107_path),
    ]:
        if path.exists():
            data = load_scrna_h5ad(path, config)
            result[name] = data
            logger.info(f"Loaded scRNA-seq {name}: {data['adata'].n_obs} cells")
        else:
            logger.warning(f"scRNA-seq {name} not found at {path}, skipping")

    return result


def load_mmrf_data(config: DataConfig) -> dict[str, Any]:
    """Load MMRF CoMMpass clinical and genomic data.

    Looks for clinical.txt and gene_expression.tsv in the CoMMpass directory.
    Returns empty dict if directory doesn't exist or is empty.

    Args:
        config: DataConfig with MMRF CoMMpass directory path.

    Returns:
        Dict with clinical and expression data.
    """
    mmrf_dir = config.mmrf_commpass_dir
    result = {}

    if not mmrf_dir.exists():
        logger.warning(f"MMRF CoMMpass directory not found at {mmrf_dir}, skipping")
        return result

    # Clinical data
    clinical_path = mmrf_dir / "clinical.txt"
    if clinical_path.exists():
        df = pd.read_csv(clinical_path, sep="\t", low_memory=False)
        result["clinical"] = df
        logger.info(f"Loaded MMRF clinical: {len(df)} patients")

    # Gene expression
    expr_path = mmrf_dir / "gene_expression.tsv"
    if expr_path.exists():
        df = pd.read_csv(expr_path, sep="\t", index_col=0, low_memory=False)
        result["expression"] = df
        logger.info(f"Loaded MMRF expression: {df.shape}")

    if not result:
        logger.warning(f"No MMRF data files found in {mmrf_dir}")

    return result
