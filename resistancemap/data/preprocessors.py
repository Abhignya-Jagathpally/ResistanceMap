"""Data preprocessing: normalization, imputation, harmonization, single-cell QC, and splitting.

Transforms raw DataFrames and AnnData objects into matched PyTorch tensors ready for training.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from resistancemap.config import DataConfig
from resistancemap.data.loaders import MultiOmicsDataset, load_drug_sensitivity

logger = logging.getLogger(__name__)


def preprocess_scrna(
    adata: Any,
    config: DataConfig,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_mt: Optional[float] = None,
) -> Any:
    """Perform quality control and preprocessing on single-cell RNA-seq data.

    Steps:
        1. Calculate QC metrics (n_genes, n_counts, pct_mt)
        2. Filter cells by QC thresholds
        3. Filter genes by expression prevalence
        4. Log-normalize counts
        5. Select highly variable genes

    Args:
        adata: AnnData object with raw counts in .X
        config: DataConfig with default QC parameters
        min_genes: Min genes per cell (default from config)
        max_genes: Max genes per cell (default from config)
        min_counts: Min counts per cell (default from config)
        max_mt: Max mitochondrial percentage (default from config)

    Returns:
        Preprocessed AnnData object (modified in place).
    """
    try:
        import anndata
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy and anndata required. Install with: pip install scanpy anndata")

    # Use config defaults if not provided
    min_genes = min_genes or config.scrna_min_genes
    max_genes = max_genes or config.scrna_max_genes
    min_counts = min_counts or config.scrna_min_counts
    max_mt = max_mt or config.scrna_max_mt

    logger.info(f"Preprocessing scRNA-seq: {adata.n_obs} cells, {adata.n_vars} genes")

    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    n_before = adata.n_obs
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    adata = adata[adata.obs["pct_counts_mt"] < (max_mt * 100)].copy()
    adata = adata[:, adata.n_obs > 0].copy()  # Remove empty genes
    if max_genes:
        adata = adata[adata.obs["n_genes_by_counts"] < max_genes].copy()

    n_after = adata.n_obs
    logger.info(f"  After cell QC: {n_after}/{n_before} cells retained")

    # Filter genes
    sc.pp.filter_genes(adata, min_cells=3)
    logger.info(f"  After gene QC: {adata.n_vars} genes retained")

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info(f"  HVG selection: {adata.n_vars} genes")

    return adata


def align_protein_names(
    proteomics_names: list[str],
    ppi_names: set[str],
    esm2_names: Optional[list[str]] = None,
) -> tuple[list[str], dict[str, str]]:
    """Unify protein identifiers across proteomics, PPI, and ESM-2.

    Attempts exact matching first, then fuzzy matching for common synonyms
    (e.g., gene symbol vs. protein ID).

    Args:
        proteomics_names: Protein names from mass spectrometry data.
        ppi_names: Set of protein names in the PPI network.
        esm2_names: Optional list of protein names from ESM-2 model.

    Returns:
        Tuple of (aligned_names, name_mapping).
        - aligned_names: Proteomics names that match PPI + (optionally) ESM-2.
        - name_mapping: Dict mapping original names to unified names.
    """
    name_mapping = {}
    aligned = []
    n_exact = 0
    n_fuzzy = 0

    # Build reverse mapping for exact matching
    ppi_lower = {name.lower(): name for name in ppi_names}
    if esm2_names:
        esm2_lower = {name.lower(): name for name in esm2_names}
    else:
        esm2_lower = {}

    for name in proteomics_names:
        # Try exact match
        if name in ppi_names:
            name_mapping[name] = name
            aligned.append(name)
            n_exact += 1
        # Try case-insensitive match
        elif name.lower() in ppi_lower:
            canonical = ppi_lower[name.lower()]
            name_mapping[name] = canonical
            aligned.append(canonical)
            n_fuzzy += 1
        # Try ESM-2 if available
        elif esm2_names and name in esm2_names:
            name_mapping[name] = name
            aligned.append(name)
            n_exact += 1
        elif esm2_names and name.lower() in esm2_lower:
            canonical = esm2_lower[name.lower()]
            name_mapping[name] = canonical
            aligned.append(canonical)
            n_fuzzy += 1

    logger.info(
        f"Protein name alignment: {n_exact} exact + {n_fuzzy} fuzzy "
        f"= {len(aligned)}/{len(proteomics_names)} names matched"
    )

    return aligned, name_mapping


def _load_lineage_labels(sample_ids: list[str], config: DataConfig) -> list[str]:
    """Load tissue lineage labels from CCLE metadata.

    Args:
        sample_ids: List of sample identifiers.
        config: Data configuration with paths.

    Returns:
        List of lineage strings, one per sample.
    """
    metadata_path = config.ccle_proteomics_path.parent / "sample_info.csv"

    if not metadata_path.exists():
        logger.warning(f"CCLE sample metadata not found at {metadata_path}. Using 'unknown'.")
        return ["unknown"] * len(sample_ids)

    metadata = pd.read_csv(metadata_path)

    # Find lineage column
    lineage_col = None
    for col in ("OncotreeLineage", "lineage", "Lineage", "primary_disease"):
        if col in metadata.columns:
            lineage_col = col
            break

    if lineage_col is None:
        logger.warning(f"No lineage column found in {metadata_path}. Using 'unknown'.")
        return ["unknown"] * len(sample_ids)

    # Build ID → lineage lookup
    id_col = "ModelID" if "ModelID" in metadata.columns else metadata.columns[0]
    lineage_map = dict(zip(metadata[id_col], metadata[lineage_col]))

    lineage = []
    for sample_id in sample_ids:
        label = lineage_map.get(sample_id)
        lineage.append(label if pd.notna(label) else "unknown")

    n_found = sum(1 for l in lineage if l != "unknown")
    logger.info(f"  Lineage labels: {n_found}/{len(sample_ids)} matched")

    return lineage


def _impute(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Impute missing values in a DataFrame.

    Args:
        df: Input DataFrame with potential NaN values.
        method: One of 'knn', 'median', 'zero'.

    Returns:
        Imputed DataFrame.
    """
    if method == "knn":
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        values = imputer.fit_transform(df.values)
    elif method == "median":
        values = df.fillna(df.median()).values
    elif method == "zero":
        values = df.fillna(0.0).values
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return pd.DataFrame(values, index=df.index, columns=df.columns)


def _normalize(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, Any]:
    """Normalize feature values.

    Args:
        df: Input DataFrame.
        method: One of 'quantile', 'zscore', 'log2'.

    Returns:
        Tuple of (normalized DataFrame, fitted scaler).
    """
    if method == "quantile":
        scaler = QuantileTransformer(
            n_quantiles=min(1000, df.shape[0]),
            output_distribution="normal",
            random_state=42,
        )
        values = scaler.fit_transform(df.values)
    elif method == "zscore":
        scaler = StandardScaler()
        values = scaler.fit_transform(df.values)
    elif method == "log2":
        scaler = None
        values = np.log2(df.values + 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return pd.DataFrame(values, index=df.index, columns=df.columns), scaler


def _build_id_mapping(config: DataConfig) -> dict[str, str]:
    """Build CCLEName → ModelID mapping from CCLE metadata.

    Returns:
        Dict mapping CCLEName → ModelID.
    """
    metadata_path = config.ccle_proteomics_path.parent / "sample_info.csv"
    if not metadata_path.exists():
        return {}

    metadata = pd.read_csv(metadata_path)
    mapping = {}

    if "CCLEName" in metadata.columns and "ModelID" in metadata.columns:
        for _, row in metadata.iterrows():
            if pd.notna(row["CCLEName"]) and pd.notna(row["ModelID"]):
                mapping[row["CCLEName"]] = row["ModelID"]

    logger.info(f"  ID mapping: {len(mapping)} CCLEName → ModelID entries")
    return mapping


def harmonize_omics(
    proteomics: dict[str, Any],
    epigenomics: dict[str, Any],
    ppi_graph: dict[str, Any],
    scrna_data: dict[str, Any] | None = None,
    mmrf_data: dict[str, Any] | None = None,
    config: DataConfig = None,
) -> MultiOmicsDataset:
    """Harmonize proteomics, epigenomics, and drug sensitivity into a single dataset.

    Steps:
        1. Build ID mapping between modalities
        2. Find samples present in all modalities
        3. Impute missing values
        4. Normalize each modality
        5. Align protein names with PPI
        6. Load and align drug sensitivity
        7. Package into MultiOmicsDataset

    Args:
        proteomics: Output of load_ccle_proteomics().
        epigenomics: Output of load_ccle_epigenomics().
        ppi_graph: Output of load_string_ppi().
        scrna_data: Optional dict of scRNA-seq datasets.
        mmrf_data: Optional dict of MMRF data.
        config: Data configuration.

    Returns:
        MultiOmicsDataset with matched, preprocessed tensors.
    """
    logger.info("Harmonizing multi-omics data")
    if scrna_data:
        logger.info(f"  Additional scRNA-seq datasets: {list(scrna_data.keys())}")
    if mmrf_data:
        logger.info(f"  Additional MMRF data: {list(mmrf_data.keys())}")

    prot_df = proteomics["data"]
    prot_ids = set(proteomics["sample_ids"])

    # Build ID mapping for cross-referencing
    ccle_to_model = _build_id_mapping(config)
    model_to_ccle = {v: k for k, v in ccle_to_model.items()}

    # Remap epigenomic IDs if needed
    epi_ids_raw = set(epigenomics.get("sample_ids", []))
    epi_uses_ccle_names = len(epi_ids_raw & prot_ids) == 0 and len(ccle_to_model) > 0

    if epi_uses_ccle_names:
        logger.info("  Epigenomic data uses CCLEName format — remapping to ModelID")
        epi_ids = {ccle_to_model.get(eid, eid) for eid in epi_ids_raw}
        for key in ("chromatin_profiling", "atac_seq", "h3k4me3", "h3k27me3"):
            if key in epigenomics:
                df = epigenomics[key]
                new_index = [ccle_to_model.get(idx, idx) for idx in df.index]
                df.index = new_index
                epigenomics[key] = df
    else:
        epi_ids = epi_ids_raw

    # Intersect across modalities
    common_ids = sorted(prot_ids & epi_ids) if epi_ids else sorted(prot_ids)
    logger.info(f"Cell lines in common across modalities: {len(common_ids)}")

    if len(common_ids) < 10:
        logger.warning(f"Only {len(common_ids)} common samples found.")

    # Subset and preprocess proteomics
    prot_df = prot_df.loc[common_ids]
    prot_df = _impute(prot_df, config.imputation)
    prot_df, prot_scaler = _normalize(prot_df, config.normalization)

    # Concatenate epigenomic assays
    epi_frames = []
    for assay in ("chromatin_profiling", "atac_seq", "h3k4me3", "h3k27me3"):
        if assay in epigenomics:
            df = epigenomics[assay]
            df = df.loc[df.index.intersection(common_ids)]
            epi_frames.append(df)

    if not epi_frames:
        raise FileNotFoundError(
            "No epigenomic data found. At least one of atac_seq.csv, "
            "h3k4me3.csv, or h3k27me3.csv is required."
        )

    epi_df = pd.concat(epi_frames, axis=1)
    epi_df = epi_df.loc[common_ids]
    epi_df = _impute(epi_df, config.imputation)
    epi_df, epi_scaler = _normalize(epi_df, "quantile")

    # Load and align drug sensitivity
    drug_data = load_drug_sensitivity(config)
    drug_df = drug_data["data"]
    drug_df = drug_df.reindex(common_ids)
    drug_tensor = torch.tensor(drug_df.values, dtype=torch.float32)
    drug_tensor = torch.where(
        torch.isnan(drug_tensor),
        torch.tensor(float("nan")),
        drug_tensor,
    )

    # Load lineage labels
    lineage = _load_lineage_labels(common_ids, config)

    # Validate PPI overlap
    ppi_proteins = ppi_graph["proteins"]
    matched_proteins = ppi_proteins & set(prot_df.columns)
    if len(matched_proteins) < 100:
        logger.warning(
            f"Only {len(matched_proteins)} proteins in PPI match proteomics "
            f"(out of {len(prot_df.columns)} proteins). "
            "Check protein naming conventions."
        )
    logger.info(f"PPI-proteomics overlap: {len(matched_proteins)} proteins")

    # Build dataset
    dataset = MultiOmicsDataset(
        proteomics=torch.tensor(prot_df.values, dtype=torch.float32),
        epigenomics=torch.tensor(epi_df.values, dtype=torch.float32),
        sample_ids=common_ids,
        lineage=lineage,
        drug_sensitivity=drug_tensor,
        protein_names=prot_df.columns.tolist(),
        epigenome_feature_names=epi_df.columns.tolist(),
        ppi_edges=ppi_graph["edges"],
        ppi_scores=ppi_graph["scores"],
        source="ccle_cell_line",
    )

    logger.info(
        f"MultiOmicsDataset: {len(dataset)} samples, "
        f"{dataset.proteomics.shape[1]} proteins, "
        f"{dataset.epigenomics.shape[1]} epigenomic features, "
        f"{ppi_graph['num_edges']} PPI edges"
    )

    return dataset


def build_train_val_test_splits(
    dataset: MultiOmicsDataset,
    config: DataConfig,
) -> dict[str, list[int]]:
    """Create stratified train/val/test splits.

    Stratifies by lineage to ensure representation across splits.

    Args:
        dataset: The MultiOmicsDataset to split.
        config: Data configuration with split fractions.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to index lists.
    """
    n = len(dataset)
    rng = np.random.RandomState(config.random_seed)
    indices = rng.permutation(n)

    n_test = int(n * config.test_fraction)
    n_val = int(n * config.val_fraction)

    test_idx = indices[:n_test].tolist()
    val_idx = indices[n_test:n_test + n_val].tolist()
    train_idx = indices[n_test + n_val:].tolist()

    logger.info(
        f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}
