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
        drug_names: Optional[list[str]] = None,
        drug_target_mean: Optional[torch.Tensor] = None,
        drug_target_std: Optional[torch.Tensor] = None,
        patient_ids: Optional[list[str]] = None,
        scrna_pseudobulk: Optional[torch.Tensor] = None,
        scrna_stages: Optional[list[str]] = None,
        scrna_samples: Optional[list[str]] = None,
        scrna_sources: Optional[list[str]] = None,
        scrna_gene_names: Optional[list[str]] = None,
        crispr_effect: Optional[torch.Tensor] = None,
        crispr_gene_names: Optional[list[str]] = None,
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
            patient_ids: Optional list of patient IDs for group-stratified splits.
        """
        # Validate all 5 required input dimensions
        n_samples = proteomics.shape[0]
        assert (
            epigenomics.shape[0] == n_samples
        ), f"epigenomics shape {epigenomics.shape[0]} != proteomics {n_samples}"
        assert (
            len(sample_ids) == n_samples
        ), f"sample_ids count {len(sample_ids)} != proteomics {n_samples}"
        assert (
            len(lineage) == n_samples
        ), f"lineage count {len(lineage)} != proteomics {n_samples}"
        assert (
            drug_sensitivity.shape[0] == n_samples
        ), f"drug_sensitivity shape {drug_sensitivity.shape[0]} != proteomics {n_samples}"
        if patient_ids is not None:
            assert (
                len(patient_ids) == n_samples
            ), f"patient_ids count {len(patient_ids)} != proteomics {n_samples}"

        # Validate feature dimensions
        assert (
            len(protein_names) == proteomics.shape[1]
        ), f"protein_names count {len(protein_names)} != proteomics features {proteomics.shape[1]}"
        assert (
            len(epigenome_feature_names) == epigenomics.shape[1]
        ), f"epigenome_feature_names count {len(epigenome_feature_names)} != epigenomics features {epigenomics.shape[1]}"
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
        # Optional drug-sensitivity bookkeeping. ``drug_sensitivity`` may be
        # standardised (z-scored) for training; mean/std vectors let
        # downstream consumers undo the transform when reporting metrics.
        self.drug_names = drug_names
        self.drug_target_mean = drug_target_mean
        self.drug_target_std = drug_target_std
        self.patient_ids = patient_ids
        # Auxiliary scRNA pseudobulk side-channel (NOT aligned with proteomics rows;
        # patient-level stage trajectories from GSE124310 + GSE271107). Consumers
        # such as TrajectoryAgent can use this as a longitudinal disease-stage
        # anchor (HD → MGUS → SMM → MM). None when scrna_summary.pt is absent.
        self.scrna_pseudobulk = scrna_pseudobulk
        self.scrna_stages = scrna_stages
        self.scrna_samples = scrna_samples
        self.scrna_sources = scrna_sources
        self.scrna_gene_names = scrna_gene_names
        # DepMap CRISPR knockout phenotypes (Chronos / DEMETER scores).
        # Row-aligned with proteomics rows (same sample_ids); NaN where the
        # cell line has no CRISPR screen. Used by data-integration-validator
        # and proteomics-pathway-validator as a causal-grounding signal:
        # without perturbation data, attribution scores are association-only.
        self.crispr_effect = crispr_effect
        self.crispr_gene_names = crispr_gene_names

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
        patient_ids_subset = (
            [self.patient_ids[i] for i in indices]
            if self.patient_ids is not None
            else None
        )
        crispr_subset = (
            self.crispr_effect[indices] if self.crispr_effect is not None else None
        )
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
            drug_names=self.drug_names,
            drug_target_mean=self.drug_target_mean,
            drug_target_std=self.drug_target_std,
            patient_ids=patient_ids_subset,
            scrna_pseudobulk=self.scrna_pseudobulk,
            scrna_stages=self.scrna_stages,
            scrna_samples=self.scrna_samples,
            scrna_sources=self.scrna_sources,
            scrna_gene_names=self.scrna_gene_names,
            crispr_effect=crispr_subset,
            crispr_gene_names=self.crispr_gene_names,
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

    # Strip "GENE_SYMBOL (entrez_id)" decoration if present so that protein
    # names are clean HGNC symbols. This is the format DepMap ships for
    # CCLE expression data, and the trailing "(NNN)" makes naive PPI joins
    # fail silently. The regex anchors on a trailing space + parenthesised
    # numeric id so legitimate symbols containing spaces are left alone.
    import re as _re
    _entrez_decoration = _re.compile(r"\s*\(\d+\)\s*$")
    cleaned = [_entrez_decoration.sub("", str(c)).strip() for c in df.columns]
    n_stripped = sum(1 for o, n in zip(df.columns, cleaned) if o != n)
    if n_stripped:
        logger.info(
            f"Stripped '(entrez_id)' decoration from {n_stripped}/{len(cleaned)} protein names"
        )
        df.columns = cleaned
        # Drop any duplicates the strip introduced (e.g. paralog name collisions)
        df = df.loc[:, ~df.columns.duplicated()]

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

    df = pd.read_csv(path, sep=r"\s+", engine="python")
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

    # ── ENSP -> gene-symbol translation ──────────────────────────────────
    # STRING ships PPI edges keyed by Ensembl protein IDs (e.g.
    # "9606.ENSP00000000233") but the rest of the pipeline (proteomics,
    # drug-sensitivity, ESM2 embeddings) keys on HGNC gene symbols. Without
    # an explicit translation, the edge intersection with proteomics is
    # empty and the GNN degenerates to self-loops. We try a small list of
    # canonical info-file paths and fall back to the raw ENSP ids only if
    # nothing matches.
    ensp_to_symbol = _load_string_protein_info(path)
    if ensp_to_symbol:
        before = len(df)
        df = df.assign(
            protein1=df["protein1"].map(ensp_to_symbol),
            protein2=df["protein2"].map(ensp_to_symbol),
        ).dropna(subset=["protein1", "protein2"])
        logger.info(
            f"Translated STRING ENSP -> gene symbols: "
            f"{len(df)}/{before} edges retained after mapping"
        )
    else:
        logger.warning(
            "No STRING protein.info file found; PPI edges will remain "
            "ENSP-keyed and will likely not match proteomics gene symbols"
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


def _load_string_protein_info(ppi_path: Path) -> dict[str, str]:
    """Locate the STRING protein.info file and build an ENSP -> gene_symbol map.

    The info file ships separately from the links file. We probe a list of
    plausible locations relative to ``ppi_path`` (sibling, parent's
    ``string/`` subdir, parent itself) and accept the first one that
    parses. Returns an empty dict if nothing is found.

    File schema (whitespace-separated, .gz allowed):
        #string_protein_id  preferred_name  protein_size  annotation
    """
    candidates: list[Path] = []
    for parent in (ppi_path.parent, ppi_path.parent / "string", ppi_path.parent.parent / "string"):
        for name in (
            "9606.protein.info.v12.0.txt.gz",
            "9606.protein.info.v12.0.txt",
            "9606.protein.info.v11.5.txt.gz",
            "9606.protein.info.v11.0.txt.gz",
            "protein.info.txt.gz",
        ):
            candidates.append(parent / name)

    for cand in candidates:
        if not cand.exists():
            continue
        try:
            info_df = pd.read_csv(
                cand,
                sep="\t",
                comment=None,
                compression="infer",
                low_memory=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  Could not parse STRING info file {cand}: {exc}")
            continue
        # The header line starts with '#'; pandas keeps it but the first column
        # may be named '#string_protein_id'. Normalise both forms.
        cols = {c.lstrip("#").strip(): c for c in info_df.columns}
        if "string_protein_id" not in cols or "preferred_name" not in cols:
            logger.warning(
                f"  STRING info file {cand} missing expected columns; got {list(info_df.columns)}"
            )
            continue
        sp_col = cols["string_protein_id"]
        gn_col = cols["preferred_name"]
        mapping = dict(zip(info_df[sp_col].astype(str), info_df[gn_col].astype(str)))
        logger.info(
            f"  Loaded STRING protein info from {cand.name}: "
            f"{len(mapping)} ENSP -> gene_symbol entries"
        )
        return mapping

    return {}


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
    """Load drug sensitivity data from GDSC, CTRPv2, and PRISM.

    Merges sources; conflicting (sample, drug) pairs are aggregated by median
    in the downstream pivot. PRISM is included when ``config.prism_path``
    exists, providing the Broad Repurposing Hub's ~4,500-compound coverage
    on top of the GDSC + CTRPv2 baselines.

    Args:
        config: DataConfig with paths to sensitivity data.

    Returns:
        Dict with 'data' (cell line × drug IC50 matrix), 'sample_ids', 'drug_names'.
    """
    logger.info("Loading drug sensitivity data")

    frames = []
    sources = [
        ("GDSC", config.gdsc_path),
        ("CTRPv2", config.ctrpv2_path),
    ]
    # PRISM is optional and only present when the file is staged. Add as a
    # third source so target_drugs that exist in PRISM but not GDSC/CTRPv2
    # (e.g. clinical compounds with broader Broad screen coverage) get
    # pulled into the matrix.
    prism_path = getattr(config, "prism_path", None)
    if prism_path is not None:
        sources.append(("PRISM", Path(prism_path)))

    for name, path in sources:
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

    # Canonicalize drug_name to a single case per compound so cross-source
    # screens (GDSC: "Bortezomib", PRISM: "bortezomib") collapse into one
    # pivot column. Use the original config.target_drugs spelling as the
    # canonical form when available; otherwise fall back to title-case.
    canonical = {d.lower(): d for d in config.target_drugs}
    combined["drug_name"] = combined["drug_lower"].map(
        lambda s: canonical.get(s, s.title())
    )

    # Canonicalize sample_id to DepMap IDs *before* pivoting. GDSC ships
    # CELL_LINE_NAME strings ("22RV1"), PRISM ships depmap_id ("ACH-000956"),
    # CTRPv2 ships its own cell-line names. Without this step, the same
    # cell line shows up as multiple pivot rows and downstream reindex onto
    # a DepMap-keyed common_ids set silently drops the name-keyed rows
    # whenever any DepMap-keyed source is also present (the harmonize
    # fallback only triggers on zero-overlap, and PRISM gives non-zero
    # overlap by itself).
    sample_info_path = config.ccle_proteomics_path.parent / "sample_info.csv"
    combined = _map_sample_ids_to_depmap(combined, sample_info_path)

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

    # PRISM ships compound names in a generic 'name' column; only treat
    # it as the drug name when we know the source is PRISM (the column
    # is too generic to claim unconditionally).
    is_prism = source_name.upper() == "PRISM"

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
        elif is_prism and lower == "name" and "drug_name" not in col_map.values():
            col_map[orig] = "drug_name"
        elif lower in ("ic50_published", "ic50"):
            col_map[orig] = "ic50"
        elif lower == "log2.ic50" and "ic50" not in col_map.values():
            col_map[orig] = "log2_ic50"
        elif lower == "ln_ic50" and "ic50" not in col_map.values():
            col_map[orig] = "ln_ic50"

    df = df.rename(columns=col_map)

    # Convert log-scale IC50 to linear if needed
    if "ic50" not in df.columns and "log2_ic50" in df.columns:
        df["ic50"] = 2.0 ** df["log2_ic50"]
    elif "ic50" not in df.columns and "ln_ic50" in df.columns:
        import numpy as _np
        df["ic50"] = _np.exp(df["ln_ic50"])

    logger.info(f"  {source_name}: standardized columns")
    return df


_ACH_PREFIX = "ACH-"


def _map_sample_ids_to_depmap(
    df: pd.DataFrame, sample_info_path: Path
) -> pd.DataFrame:
    """Canonicalize the ``sample_id`` column to DepMap IDs (``ACH-NNNNNN``).

    Rows whose ``sample_id`` already starts with ``ACH-`` are passed through
    unchanged (PRISM, DepMap-keyed sources). For the remainder we look up
    each id against ``sample_info.csv``, choosing the alias column with the
    highest match count among ``stripped_cell_line_name``, ``cell_line_name``,
    ``CCLE_Name`` (in that order on ties — stripped names win because they
    canonicalize away whitespace and punctuation that GDSC keeps and DepMap
    drops).

    Rows that fail to map are kept with their original id and will simply be
    dropped by the downstream ``drug_df.reindex(common_ids)`` in
    ``harmonize_omics``. We never silently fabricate a DepMap ID; unmapped
    means unmapped.

    Args:
        df: DataFrame with at least a ``sample_id`` column.
        sample_info_path: Path to CCLE ``sample_info.csv`` (DepMap_ID + alias
            columns). When absent, the function logs a warning and returns
            ``df`` unchanged.

    Returns:
        ``df`` with ``sample_id`` rewritten in place where a DepMap ID was
        recoverable.
    """
    if "sample_id" not in df.columns:
        return df

    if not sample_info_path.exists():
        logger.warning(
            f"sample_info.csv not found at {sample_info_path}; "
            "drug-sensitivity sample IDs will not be canonicalized to DepMap IDs"
        )
        return df

    meta = pd.read_csv(sample_info_path)
    id_col = meta.columns[0]  # DepMap_ID by convention
    if not meta[id_col].astype(str).str.startswith(_ACH_PREFIX).any():
        logger.warning(
            f"sample_info.csv first column {id_col!r} does not look like "
            "DepMap IDs; skipping sample_id canonicalization"
        )
        return df

    # Identify rows already in DepMap form vs rows that need mapping.
    sids = df["sample_id"].astype(str)
    already_depmap_mask = sids.str.startswith(_ACH_PREFIX)
    n_pre_mapped = int(already_depmap_mask.sum())
    n_to_map = int((~already_depmap_mask).sum())

    if n_to_map == 0:
        logger.info(
            f"  Sample-ID canonicalization: all {n_pre_mapped} rows are "
            "already DepMap-keyed; nothing to map"
        )
        return df

    # Try alias columns in priority order; pick the one giving the most
    # matches against the rows that still need mapping. This is more robust
    # than the harmonize fallback, which short-circuits on first non-zero
    # overlap and can therefore pick a low-yield column when a higher-yield
    # one exists later in the list.
    alias_priority = ["stripped_cell_line_name", "cell_line_name", "CCLE_Name"]
    available = [c for c in alias_priority if c in meta.columns]
    if not available:
        logger.warning(
            f"sample_info.csv has none of {alias_priority}; "
            "skipping sample_id canonicalization"
        )
        return df

    needs_mapping = sids[~already_depmap_mask]
    needs_set = set(needs_mapping.astype(str).str.strip())

    best_col: Optional[str] = None
    best_map: dict[str, str] = {}
    best_hits = -1
    for col in available:
        alias_to_id = (
            meta[[col, id_col]]
            .dropna()
            .astype(str)
            .assign(_alias=lambda d: d[col].str.strip())
            .drop_duplicates(subset="_alias")
            .set_index("_alias")[id_col]
            .to_dict()
        )
        hits = sum(1 for s in needs_set if s in alias_to_id)
        if hits > best_hits:
            best_hits = hits
            best_map = alias_to_id
            best_col = col

    if best_hits <= 0 or best_col is None:
        logger.warning(
            "Sample-ID canonicalization: no alias column matched any "
            f"non-DepMap sample_id ({n_to_map} rows unmapped)"
        )
        return df

    # Apply only to the rows that were not already DepMap-keyed; leave PRISM
    # rows untouched even if their ACH id happens to also appear as an alias.
    mapped = sids.where(
        already_depmap_mask,
        sids.str.strip().map(lambda s: best_map.get(s, s)),
    )
    n_mapped = int((mapped != sids).sum())
    n_unmapped = n_to_map - n_mapped
    df = df.copy()
    df["sample_id"] = mapped.values
    logger.info(
        f"  Sample-ID canonicalization via {best_col!r}: "
        f"{n_pre_mapped} already-DepMap rows + {n_mapped}/{n_to_map} newly "
        f"mapped ({n_unmapped} unmappable rows kept verbatim and will be "
        "filtered by the downstream reindex)"
    )
    return df


def load_scrna_data(config: DataConfig) -> Optional[dict[str, Any]]:
    """Load all available scRNA-seq datasets.

    Loads GSE124310 (MM patient samples) and GSE271107 (lenalidomide response)
    if their files exist. Returns None if neither is found (graceful degradation
    for missing scRNA-seq data).

    Args:
        config: DataConfig with scRNA-seq paths.

    Returns:
        Dict with loaded AnnData objects keyed by accession, or None if no datasets found.
    """
    result = {}

    for name, path in [
        ("GSE124310", config.scrna_gse124310_path),
        ("GSE271107", config.scrna_gse271107_path),
    ]:
        if path.exists():
            try:
                data = load_scrna_h5ad(path, config)
                result[name] = data
                logger.info(f"Loaded scRNA-seq {name}: {data['adata'].n_obs} cells")
            except Exception as e:
                logger.warning(f"Failed to load scRNA-seq {name} from {path}: {e}")
        else:
            logger.debug(f"scRNA-seq {name} not found at {path}, skipping")

    # Return None if no datasets loaded (graceful degradation)
    return result if result else None


def load_depmap_crispr(config: DataConfig) -> Optional[dict[str, Any]]:
    """Load DepMap CRISPR gene-effect (Chronos) matrix.

    Format: CSV with cell lines (DepMap ModelID) as rows and genes as
    columns. DepMap ships gene names with the legacy ``"GENE_SYMBOL
    (entrez_id)"`` decoration; we strip the trailing ``(NNN)`` so the
    columns are clean HGNC symbols matching CCLE proteomics + STRING PPI.

    Returns ``None`` (graceful degradation) when the file is absent so
    the pipeline can still run on cell-line-only feature data.

    Args:
        config: DataConfig with ``depmap_crispr_path``.

    Returns:
        Dict with 'data' (DataFrame, ModelID rows × gene cols),
        'sample_ids', 'gene_names'. None if the file is missing.
    """
    path = getattr(config, "depmap_crispr_path", None)
    if path is None:
        logger.info("DepMap CRISPR path not configured; skipping")
        return None
    path = Path(path)
    if not path.exists():
        logger.info(f"DepMap CRISPR not found at {path} (graceful degradation)")
        return None

    logger.info(f"Loading DepMap CRISPR gene-effect from {path}")
    df = pd.read_csv(path, index_col=0, low_memory=False)
    logger.info(
        f"  CRISPR matrix: {df.shape[0]} cell lines × {df.shape[1]} genes"
    )

    # Strip "GENE (entrez_id)" decoration the same way load_ccle_proteomics
    # does so the gene columns join cleanly with proteomics + PPI.
    import re as _re
    _decoration = _re.compile(r"\s*\(\d+\)\s*$")
    cleaned = [_decoration.sub("", str(c)).strip() for c in df.columns]
    n_stripped = sum(1 for o, n in zip(df.columns, cleaned) if o != n)
    if n_stripped:
        logger.info(
            f"  Stripped '(entrez_id)' decoration from {n_stripped}/{len(cleaned)} CRISPR genes"
        )
        df.columns = cleaned
        df = df.loc[:, ~df.columns.duplicated()]

    return {
        "data": df,
        "sample_ids": df.index.tolist(),
        "gene_names": df.columns.tolist(),
    }


def load_mmrf_data(config: DataConfig) -> dict[str, Any]:
    """Load MMRF CoMMpass clinical and genomic data.

    Loads clinical.txt, gene_expression.tsv, and optionally mutation data from
    the CoMMpass directory. Extracts patient IDs, treatment info, and ISS staging
    from clinical data. Returns empty dict if directory doesn't exist (graceful degradation).

    Args:
        config: DataConfig with MMRF CoMMpass directory path.

    Returns:
        Dict with keys: 'clinical', 'expression', 'patient_ids', 'treatment_data'.
        Returns empty dict if directory missing or no data files found.
    """
    mmrf_dir = config.mmrf_commpass_dir
    result = {}

    if not mmrf_dir.exists():
        logger.info(f"MMRF CoMMpass directory not found at {mmrf_dir} (graceful degradation)")
        return result

    logger.info(f"Loading MMRF CoMMpass data from {mmrf_dir}")

    # Load clinical data
    clinical_path = mmrf_dir / "clinical.txt"
    clinical_df = None
    if clinical_path.exists():
        try:
            clinical_df = pd.read_csv(clinical_path, sep="\t", low_memory=False)
            result["clinical"] = clinical_df
            logger.info(f"Loaded MMRF clinical: {len(clinical_df)} patients, {len(clinical_df.columns)} features")

            # Extract patient IDs
            patient_id_col = None
            for col in ("patient_id", "PATIENT_ID", "Patient_ID", "patientID"):
                if col in clinical_df.columns:
                    patient_id_col = col
                    break
            if patient_id_col:
                result["patient_ids"] = clinical_df[patient_id_col].tolist()
                logger.info(f"Extracted {len(result['patient_ids'])} patient IDs")

            # Extract treatment data if available
            treatment_data = {}
            treatment_cols = [c for c in clinical_df.columns if "treatment" in c.lower() or "drug" in c.lower()]
            if treatment_cols:
                for col in treatment_cols:
                    treatment_data[col] = clinical_df[col].tolist()
                result["treatment_data"] = treatment_data
                logger.info(f"Extracted treatment data: {len(treatment_cols)} treatment columns")

            # Extract ISS staging if available
            iss_cols = [c for c in clinical_df.columns if "iss" in c.lower()]
            if iss_cols:
                result["iss_staging"] = {col: clinical_df[col].tolist() for col in iss_cols}
                logger.info(f"Extracted ISS staging from {len(iss_cols)} columns")
        except Exception as e:
            logger.error(f"Failed to load MMRF clinical data from {clinical_path}: {e}")

    # Load gene expression data
    expr_path = mmrf_dir / "gene_expression.tsv"
    if expr_path.exists():
        try:
            expr_df = pd.read_csv(expr_path, sep="\t", index_col=0, low_memory=False)
            result["expression"] = expr_df
            logger.info(f"Loaded MMRF gene expression: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
        except Exception as e:
            logger.error(f"Failed to load MMRF expression data from {expr_path}: {e}")

    # Load mutation data if available
    mutation_path = mmrf_dir / "mutations.tsv"
    if mutation_path.exists():
        try:
            mut_df = pd.read_csv(mutation_path, sep="\t", low_memory=False)
            result["mutations"] = mut_df
            logger.info(f"Loaded MMRF mutations: {len(mut_df)} records")
        except Exception as e:
            logger.warning(f"Failed to load MMRF mutation data from {mutation_path}: {e}")

    if not result:
        logger.warning(f"No MMRF data files found in {mmrf_dir}")
    else:
        logger.info(f"MMRF data loading complete: {len(result)} data types loaded")

    return result