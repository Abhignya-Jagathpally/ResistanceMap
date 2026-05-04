"""
resistancemap/data/mmrf_loader.py
=================================
Real data loader for MMRF CoMMpass (Multiple Myeloma Research Foundation
Relating Clinical Outcomes in MM to Personal Assessment of Genetic Profile).

Handles:
  - GDC flat files: RNA-seq HTSeq counts, MAF mutation calls, TSV clinical
  - Preprocessed AnnData (.h5ad) format
  - Gene name harmonization (HGNC symbols)
  - Treatment response label extraction
  - Longitudinal visit alignment
  - Stratified cohort splitting

Dataset: phs000748 -- 1,143 patients, WGS/WES/RNA-seq, longitudinal clinical
Available via GDC Data Portal and MMRF Researcher Gateway.

Requirements:
    pip install torch numpy pandas scipy anndata scanpy
"""

from __future__ import annotations

import os
import re
import glob
import json
import logging
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IMWG response criteria for multiple myeloma
RESPONDER_LABELS = {"sCR", "CR", "VGPR", "PR"}
NON_RESPONDER_LABELS = {"SD", "PD", "MR"}  # MR sometimes reported

# High-risk cytogenetics
HIGH_RISK_CYTOGENETICS = ["t(4;14)", "t(14;16)", "del(17p)", "gain(1q)"]

# Standard longitudinal timepoints (months from baseline)
STANDARD_TIMEPOINTS = [0, 3, 6, 12, 24]

# GDC file patterns
GDC_CLINICAL_PATTERN = "clinical.tsv"
GDC_RNASEQ_PATTERN = "*.htseq.counts"
GDC_MAF_PATTERN = "*.maf"

# MMRF Researcher Gateway file patterns
MMRF_CLINICAL_FILE = "MMRF_CoMMpass_IA*_PER_PATIENT.csv"
MMRF_VISIT_FILE = "MMRF_CoMMpass_IA*_PER_PATIENT_VISIT.csv"
MMRF_RNASEQ_FILE = "MMRF_CoMMpass_IA*_salmon_gene_tpm.tsv"
MMRF_MUTATION_FILE = "MMRF_CoMMpass_IA*_All_Canonical_Muts.tsv"
MMRF_TREATMENT_FILE = "MMRF_CoMMpass_IA*_STAND_ALONE_TRTRESP.csv"
MMRF_CYTOGENETICS_FILE = "MMRF_CoMMpass_IA*_CNA_LongInsert_FISH_CN.tsv"


# ---------------------------------------------------------------------------
# Gene Name Harmonization
# ---------------------------------------------------------------------------

class HGNCMapper:
    """Map gene identifiers to HGNC-approved symbols.

    Uses a prebuilt alias table. If no table is available, falls back to
    simple Ensembl-to-symbol mapping from the count file headers.
    """

    def __init__(self, alias_file: Optional[str] = None):
        self.alias_map: Dict[str, str] = {}
        self.ensembl_map: Dict[str, str] = {}
        if alias_file and os.path.exists(alias_file):
            self._load_alias_table(alias_file)

    def _load_alias_table(self, path: str) -> None:
        """Load HGNC alias table (TSV with columns: symbol, alias_symbol, prev_symbol, ensembl_gene_id)."""
        df = pd.read_csv(path, sep="\t", low_memory=False)
        if "symbol" in df.columns and "alias_symbol" in df.columns:
            for _, row in df.iterrows():
                symbol = str(row["symbol"]).strip()
                if pd.notna(row.get("alias_symbol")):
                    for alias in str(row["alias_symbol"]).split("|"):
                        self.alias_map[alias.strip().upper()] = symbol
                if pd.notna(row.get("prev_symbol")):
                    for prev in str(row["prev_symbol"]).split("|"):
                        self.alias_map[prev.strip().upper()] = symbol
                if pd.notna(row.get("ensembl_gene_id")):
                    self.ensembl_map[str(row["ensembl_gene_id"]).strip()] = symbol
        logger.info(f"Loaded {len(self.alias_map)} gene aliases, {len(self.ensembl_map)} Ensembl mappings")

    def harmonize(self, gene_id: str) -> str:
        """Map a gene identifier to its HGNC symbol."""
        gene_id_clean = gene_id.strip()
        # Strip Ensembl version suffix
        base_id = gene_id_clean.split(".")[0]
        if base_id in self.ensembl_map:
            return self.ensembl_map[base_id]
        upper = gene_id_clean.upper()
        if upper in self.alias_map:
            return self.alias_map[upper]
        return gene_id_clean

    def harmonize_index(self, genes: pd.Index) -> pd.Index:
        """Harmonize a pandas Index of gene names."""
        return pd.Index([self.harmonize(g) for g in genes])


# ---------------------------------------------------------------------------
# Clinical Data Parsers
# ---------------------------------------------------------------------------

def parse_gdc_clinical(clinical_dir: str) -> pd.DataFrame:
    """Parse GDC-format clinical TSV files.

    GDC exports clinical data as TSV with columns including:
    case_submitter_id, age_at_diagnosis, gender, iss_stage, etc.
    """
    tsv_files = glob.glob(os.path.join(clinical_dir, "**", "clinical.tsv"), recursive=True)
    if not tsv_files:
        tsv_files = glob.glob(os.path.join(clinical_dir, "**", "*.tsv"), recursive=True)
    if not tsv_files:
        raise FileNotFoundError(f"No clinical TSV files found in {clinical_dir}")

    dfs = []
    for f in tsv_files:
        try:
            df = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
            if "case_submitter_id" in df.columns or "submitter_id" in df.columns:
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not dfs:
        raise ValueError("No parseable clinical files found")

    clinical = pd.concat(dfs, ignore_index=True)
    id_col = "case_submitter_id" if "case_submitter_id" in clinical.columns else "submitter_id"
    clinical = clinical.drop_duplicates(subset=[id_col])
    clinical = clinical.rename(columns={id_col: "patient_id"})
    return clinical


def parse_mmrf_clinical(data_dir: str) -> pd.DataFrame:
    """Parse MMRF Researcher Gateway clinical CSV files.

    Combines per-patient demographics, treatment response, and cytogenetics.
    """
    # Find per-patient file
    patient_files = glob.glob(os.path.join(data_dir, MMRF_CLINICAL_FILE))
    if not patient_files:
        patient_files = glob.glob(os.path.join(data_dir, "**", "PER_PATIENT*.csv"), recursive=True)
    if not patient_files:
        raise FileNotFoundError(f"No MMRF per-patient file found in {data_dir}")

    patients = pd.read_csv(patient_files[0], low_memory=False)
    # Standardize patient ID column
    id_candidates = ["PUBLIC_ID", "public_id", "MMRF_ID", "Study_Visit_ID"]
    id_col = next((c for c in id_candidates if c in patients.columns), patients.columns[0])
    patients = patients.rename(columns={id_col: "patient_id"})

    # Parse treatment response
    trt_files = glob.glob(os.path.join(data_dir, MMRF_TREATMENT_FILE))
    if not trt_files:
        trt_files = glob.glob(os.path.join(data_dir, "**", "*TRTRESP*.csv"), recursive=True)
    if trt_files:
        trt = pd.read_csv(trt_files[0], low_memory=False)
        trt_id_col = next((c for c in id_candidates if c in trt.columns), trt.columns[0])
        trt = trt.rename(columns={trt_id_col: "patient_id"})
        patients = patients.merge(trt, on="patient_id", how="left", suffixes=("", "_trt"))

    # Parse cytogenetics
    cyto_files = glob.glob(os.path.join(data_dir, MMRF_CYTOGENETICS_FILE))
    if not cyto_files:
        cyto_files = glob.glob(os.path.join(data_dir, "**", "*FISH*.tsv"), recursive=True)
    if cyto_files:
        cyto = pd.read_csv(cyto_files[0], sep="\t", low_memory=False)
        cyto_id_col = next((c for c in id_candidates if c in cyto.columns), cyto.columns[0])
        cyto = cyto.rename(columns={cyto_id_col: "patient_id"})
        patients = patients.merge(cyto, on="patient_id", how="left", suffixes=("", "_cyto"))

    return patients


def extract_response_label(clinical: pd.DataFrame) -> pd.Series:
    """Extract binary response label from clinical data.

    Responder: sCR, CR, VGPR, PR  -> 1
    Non-responder: SD, PD          -> 0
    """
    # Try multiple column names for best response
    response_cols = [
        "bestresponse", "best_response", "BESTRESPONSE", "D_PT_bestresptyp",
        "best_confirmed_response", "BOR", "best_overall_response",
        "D_PT_bestresptyp_1", "line1_best_response",
    ]
    resp_col = None
    for col in response_cols:
        if col in clinical.columns:
            resp_col = col
            break
    if resp_col is None:
        # Search for columns containing 'response' or 'resp'
        candidates = [c for c in clinical.columns if "resp" in c.lower() and "best" in c.lower()]
        if candidates:
            resp_col = candidates[0]
        else:
            logger.warning("No response column found; returning NaN labels")
            return pd.Series(np.nan, index=clinical.index, name="response_label")

    labels = clinical[resp_col].astype(str).str.strip().str.upper()
    response = pd.Series(np.nan, index=clinical.index, name="response_label")

    for idx, val in labels.items():
        if val in {"SCR", "CR", "VGPR", "PR"}:
            response[idx] = 1.0
        elif val in {"SD", "PD", "MR"}:
            response[idx] = 0.0

    n_resp = (response == 1).sum()
    n_nonresp = (response == 0).sum()
    n_missing = response.isna().sum()
    logger.info(f"Response labels: {n_resp} responders, {n_nonresp} non-responders, {n_missing} missing")
    return response


def extract_cytogenetic_risk(clinical: pd.DataFrame) -> pd.DataFrame:
    """Extract high-risk cytogenetic features.

    Returns DataFrame with boolean columns for each aberration plus
    an overall risk score (0=standard, 1=high, 2=ultra-high).
    """
    cyto_cols_map = {
        "t(4;14)": ["FISH_t_4_14", "t_4_14", "D_PT_FISH_t414"],
        "t(14;16)": ["FISH_t_14_16", "t_14_16", "D_PT_FISH_t1416"],
        "del(17p)": ["FISH_del_17p", "del_17p", "D_PT_FISH_del17p", "del17p"],
        "gain(1q)": ["FISH_gain_1q", "gain_1q", "D_PT_FISH_gain1q", "amp1q"],
    }

    result = pd.DataFrame(index=clinical.index)
    for aberration, col_names in cyto_cols_map.items():
        found = False
        for col in col_names:
            if col in clinical.columns:
                vals = clinical[col].astype(str).str.strip().str.upper()
                result[aberration] = vals.isin(["YES", "TRUE", "1", "POSITIVE", "Y"])
                found = True
                break
        if not found:
            result[aberration] = False
            logger.debug(f"No column found for {aberration}")

    # Risk score: 0=standard, 1=high (1 factor), 2=ultra-high (2+ factors)
    n_factors = result.sum(axis=1)
    result["risk_score"] = np.where(n_factors >= 2, 2, np.where(n_factors >= 1, 1, 0))
    result["risk_category"] = np.where(
        n_factors >= 2, "ultra-high",
        np.where(n_factors >= 1, "high", "standard")
    )
    return result


def extract_iss_stage(clinical: pd.DataFrame) -> pd.Series:
    """Extract ISS (International Staging System) stage."""
    iss_cols = ["iss_stage", "ISS_STAGE", "D_PT_iss", "ISS", "iss"]
    for col in iss_cols:
        if col in clinical.columns:
            stage = clinical[col].astype(str).str.extract(r"(\d)")[0].astype(float)
            logger.info(f"ISS stages: {stage.value_counts().to_dict()}")
            return stage.rename("iss_stage")
    logger.warning("No ISS stage column found")
    return pd.Series(np.nan, index=clinical.index, name="iss_stage")


def extract_treatment_regimen(clinical: pd.DataFrame) -> pd.DataFrame:
    """Extract treatment regimen information.

    Returns DataFrame with columns: regimen_name, contains_PI (proteasome inhibitor),
    contains_IMiD (immunomodulatory), contains_dex (dexamethasone), line_number.
    """
    regimen_cols = [
        "treatment_regimen", "trtname", "TRTNAME", "D_TRT_trtname",
        "regimen", "line1_regimen", "D_PT_trtgrp",
    ]
    reg_col = None
    for col in regimen_cols:
        if col in clinical.columns:
            reg_col = col
            break

    result = pd.DataFrame(index=clinical.index)
    if reg_col is None:
        result["regimen_name"] = "unknown"
        result["contains_PI"] = False
        result["contains_IMiD"] = False
        result["contains_dex"] = False
        result["line_number"] = 1
        return result

    regimens = clinical[reg_col].astype(str).str.upper()
    result["regimen_name"] = clinical[reg_col]

    pi_drugs = ["BORTEZOMIB", "BTZ", "CARFILZOMIB", "CFZ", "IXAZOMIB", "IXA", "VELCADE"]
    imid_drugs = ["LENALIDOMIDE", "LEN", "POMALIDOMIDE", "POM", "THALIDOMIDE", "THAL", "REVLIMID"]
    dex_drugs = ["DEXAMETHASONE", "DEX"]

    result["contains_PI"] = regimens.apply(
        lambda x: any(d in x for d in pi_drugs) if pd.notna(x) else False
    )
    result["contains_IMiD"] = regimens.apply(
        lambda x: any(d in x for d in imid_drugs) if pd.notna(x) else False
    )
    result["contains_dex"] = regimens.apply(
        lambda x: any(d in x for d in dex_drugs) if pd.notna(x) else False
    )

    line_cols = ["line_number", "trtline", "D_TRT_trtline", "line"]
    result["line_number"] = 1
    for col in line_cols:
        if col in clinical.columns:
            result["line_number"] = pd.to_numeric(clinical[col], errors="coerce").fillna(1).astype(int)
            break

    return result


# ---------------------------------------------------------------------------
# RNA-seq Loaders
# ---------------------------------------------------------------------------

def load_gdc_htseq_counts(count_dir: str, gene_mapper: Optional[HGNCMapper] = None) -> pd.DataFrame:
    """Load HTSeq count files from GDC download.

    Each file is a two-column TSV (gene_id, count) named by sample UUID.
    Returns a DataFrame: rows=genes, columns=sample_ids.
    """
    count_files = glob.glob(os.path.join(count_dir, "**", "*.htseq.counts*"), recursive=True)
    if not count_files:
        count_files = glob.glob(os.path.join(count_dir, "**", "*.counts"), recursive=True)
    if not count_files:
        raise FileNotFoundError(f"No HTSeq count files found in {count_dir}")

    logger.info(f"Loading {len(count_files)} HTSeq count files")
    counts_dict = {}
    for f in count_files:
        sample_id = Path(f).stem.replace(".htseq.counts", "").replace(".counts", "")
        try:
            df = pd.read_csv(f, sep="\t", header=None, names=["gene_id", "count"],
                             index_col=0, comment="_")
            counts_dict[sample_id] = df["count"]
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    counts = pd.DataFrame(counts_dict)
    if gene_mapper:
        counts.index = gene_mapper.harmonize_index(counts.index)
    # Deduplicate genes by summing
    counts = counts.groupby(counts.index).sum()
    logger.info(f"Loaded RNA-seq: {counts.shape[0]} genes x {counts.shape[1]} samples")
    return counts


def load_mmrf_rnaseq(data_dir: str, gene_mapper: Optional[HGNCMapper] = None) -> pd.DataFrame:
    """Load MMRF salmon TPM or STAR count matrix.

    The MMRF provides a single matrix file with genes as rows, samples as columns.
    """
    rnaseq_files = glob.glob(os.path.join(data_dir, MMRF_RNASEQ_FILE))
    if not rnaseq_files:
        rnaseq_files = glob.glob(os.path.join(data_dir, "**", "*salmon*tpm*"), recursive=True)
    if not rnaseq_files:
        rnaseq_files = glob.glob(os.path.join(data_dir, "**", "*STAR*count*"), recursive=True)
    if not rnaseq_files:
        raise FileNotFoundError(f"No MMRF RNA-seq file found in {data_dir}")

    counts = pd.read_csv(rnaseq_files[0], sep="\t", index_col=0, low_memory=False)
    if gene_mapper:
        counts.index = gene_mapper.harmonize_index(counts.index)
    counts = counts.groupby(counts.index).sum()
    logger.info(f"Loaded MMRF RNA-seq: {counts.shape[0]} genes x {counts.shape[1]} samples")
    return counts


def load_mutation_data(data_dir: str, source: str = "gdc") -> pd.DataFrame:
    """Load somatic mutation data from MAF files.

    Returns a binary mutation matrix: rows=genes, columns=patients.
    Value=1 if gene has a non-silent mutation in that patient.
    """
    if source == "gdc":
        maf_files = glob.glob(os.path.join(data_dir, "**", "*.maf*"), recursive=True)
    else:
        maf_files = glob.glob(os.path.join(data_dir, MMRF_MUTATION_FILE))
        if not maf_files:
            maf_files = glob.glob(os.path.join(data_dir, "**", "*Muts*"), recursive=True)

    if not maf_files:
        raise FileNotFoundError(f"No MAF/mutation files found in {data_dir}")

    all_muts = []
    for f in maf_files:
        try:
            maf = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
            # Standardize column names
            col_map = {}
            for c in maf.columns:
                cl = c.lower()
                if "hugo" in cl or "gene" in cl:
                    col_map[c] = "gene"
                elif "tumor_sample" in cl or "sample" in cl:
                    col_map[c] = "sample_id"
                elif "variant_class" in cl:
                    col_map[c] = "variant_class"
            maf = maf.rename(columns=col_map)
            if "gene" in maf.columns and "sample_id" in maf.columns:
                all_muts.append(maf)
        except Exception as e:
            logger.warning(f"Failed to parse {f}: {e}")

    if not all_muts:
        raise ValueError("No parseable mutation files found")

    maf_combined = pd.concat(all_muts, ignore_index=True)

    # Filter non-silent mutations
    if "variant_class" in maf_combined.columns:
        silent_types = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR"}
        maf_combined = maf_combined[~maf_combined["variant_class"].isin(silent_types)]

    # Build binary matrix
    mutation_matrix = pd.crosstab(maf_combined["gene"], maf_combined["sample_id"])
    mutation_matrix = (mutation_matrix > 0).astype(int)
    logger.info(f"Mutation matrix: {mutation_matrix.shape[0]} genes x {mutation_matrix.shape[1]} samples")
    return mutation_matrix


# ---------------------------------------------------------------------------
# Sample-to-Patient Mapping
# ---------------------------------------------------------------------------

def build_sample_patient_map(data_dir: str) -> Dict[str, str]:
    """Build mapping from sample/aliquot IDs to patient IDs.

    Uses the MMRF sample manifest or GDC metadata JSON files.
    """
    mapping = {}

    # Try MMRF manifest
    manifest_files = glob.glob(os.path.join(data_dir, "**", "*manifest*"), recursive=True)
    manifest_files += glob.glob(os.path.join(data_dir, "**", "*sample*map*"), recursive=True)

    for f in manifest_files:
        try:
            df = pd.read_csv(f, sep=None, engine="python", low_memory=False)
            sample_col = next((c for c in df.columns if "sample" in c.lower()), None)
            patient_col = next((c for c in df.columns if any(x in c.lower() for x in ["patient", "public_id", "mmrf"])), None)
            if sample_col and patient_col:
                for _, row in df.iterrows():
                    mapping[str(row[sample_col])] = str(row[patient_col])
        except Exception:
            pass

    # Try GDC JSON metadata
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    for f in json_files:
        try:
            with open(f) as fh:
                meta = json.load(fh)
            if isinstance(meta, list):
                for entry in meta:
                    if "cases" in entry and "file_name" in entry:
                        for case in entry["cases"]:
                            mapping[entry["file_name"]] = case.get("submitter_id", "")
        except Exception:
            pass

    # MMRF convention: sample IDs start with MMRF_xxxx
    # e.g., MMRF_1234_1_BM_CD138pos_T1 -> MMRF_1234
    for key in list(mapping.keys()):
        match = re.match(r"(MMRF_\d+)", key)
        if match:
            mapping[key] = match.group(1)

    logger.info(f"Built sample-to-patient mapping: {len(mapping)} entries")
    return mapping


def map_samples_to_patients(
    expression: pd.DataFrame, sample_patient_map: Dict[str, str]
) -> pd.DataFrame:
    """Aggregate expression by patient (mean across samples)."""
    new_cols = {}
    for col in expression.columns:
        pid = sample_patient_map.get(col)
        if pid is None:
            match = re.match(r"(MMRF_\d+)", col)
            pid = match.group(1) if match else col
        if pid not in new_cols:
            new_cols[pid] = []
        new_cols[pid].append(col)

    patient_expr = pd.DataFrame(
        {pid: expression[cols].mean(axis=1) for pid, cols in new_cols.items()},
        index=expression.index,
    )
    logger.info(f"Mapped {expression.shape[1]} samples -> {patient_expr.shape[1]} patients")
    return patient_expr


# ---------------------------------------------------------------------------
# Longitudinal Visit Extraction
# ---------------------------------------------------------------------------

def extract_longitudinal_visits(
    data_dir: str, patient_ids: List[str]
) -> Dict[str, Dict[int, pd.Series]]:
    """Extract longitudinal data per patient per timepoint.

    Returns: {patient_id: {month: row_of_clinical_data}}
    Timepoints: 0 (baseline), 3, 6, 12, 24 months.
    """
    visit_files = glob.glob(os.path.join(data_dir, MMRF_VISIT_FILE))
    if not visit_files:
        visit_files = glob.glob(os.path.join(data_dir, "**", "*VISIT*.csv"), recursive=True)
    if not visit_files:
        logger.warning("No visit-level data files found")
        return {}

    visits = pd.read_csv(visit_files[0], low_memory=False)
    id_candidates = ["PUBLIC_ID", "public_id", "MMRF_ID"]
    id_col = next((c for c in id_candidates if c in visits.columns), visits.columns[0])
    visits = visits.rename(columns={id_col: "patient_id"})

    # Find timepoint column (days or months from diagnosis)
    time_col = None
    for col in ["VISITDY", "visit_day", "D_VJ_day", "days_from_diagnosis"]:
        if col in visits.columns:
            time_col = col
            break

    if time_col is None:
        logger.warning("No visit time column found")
        return {}

    visits["month"] = pd.to_numeric(visits[time_col], errors="coerce") / 30.44  # days -> months

    longitudinal = {}
    for pid in patient_ids:
        patient_visits = visits[visits["patient_id"] == pid].copy()
        if patient_visits.empty:
            continue
        longitudinal[pid] = {}
        for target_month in STANDARD_TIMEPOINTS:
            # Find closest visit within +/- 45 days
            diffs = (patient_visits["month"] - target_month).abs()
            min_idx = diffs.idxmin()
            if diffs[min_idx] <= 1.5:  # 1.5 months tolerance
                longitudinal[pid][target_month] = patient_visits.loc[min_idx]

    n_with_visits = len(longitudinal)
    avg_timepoints = np.mean([len(v) for v in longitudinal.values()]) if longitudinal else 0
    logger.info(f"Longitudinal data: {n_with_visits} patients, avg {avg_timepoints:.1f} timepoints")
    return longitudinal


# ---------------------------------------------------------------------------
# Stratified Cohort Splitting
# ---------------------------------------------------------------------------

def stratified_cohort_split(
    patient_ids: np.ndarray,
    response_labels: np.ndarray,
    iss_stages: np.ndarray,
    risk_scores: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split patients into discovery/validation/test stratified by ISS and risk.

    Creates a composite stratification variable from response, ISS stage,
    and cytogenetic risk to ensure balanced representation across splits.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)

    # Build composite strata
    strata = []
    for resp, iss, risk in zip(response_labels, iss_stages, risk_scores):
        r = int(resp) if not np.isnan(resp) else -1
        i = int(iss) if not np.isnan(iss) else -1
        strata.append(f"{r}_{i}_{int(risk)}")
    strata = np.array(strata)

    # Collapse rare strata (< 5 members) into an "other" bin
    from collections import Counter
    counts = Counter(strata)
    strata_collapsed = np.array([s if counts[s] >= 5 else "rare" for s in strata])

    # Stratified split using sklearn
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(patient_ids))
    # First split: train vs (val+test)
    val_test_frac = val_frac + test_frac
    try:
        train_idx, valtest_idx = train_test_split(
            idx, test_size=val_test_frac, stratify=strata_collapsed, random_state=seed
        )
    except ValueError:
        # Fallback to non-stratified if strata too small
        logger.warning("Stratification failed, using random split")
        train_idx, valtest_idx = train_test_split(
            idx, test_size=val_test_frac, random_state=seed
        )

    # Second split: val vs test
    relative_test_frac = test_frac / val_test_frac
    try:
        val_idx, test_idx = train_test_split(
            valtest_idx, test_size=relative_test_frac,
            stratify=strata_collapsed[valtest_idx], random_state=seed
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            valtest_idx, test_size=relative_test_frac, random_state=seed
        )

    logger.info(
        f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
        f"({len(train_idx)/len(idx):.0%}/{len(val_idx)/len(idx):.0%}/{len(test_idx)/len(idx):.0%})"
    )
    return patient_ids[train_idx], patient_ids[val_idx], patient_ids[test_idx]


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class MMRFCoMMpassDataset(Dataset):
        """PyTorch Dataset for MMRF CoMMpass multi-omics data."""

        def __init__(self, data_dir=None, h5ad_path=None, source="mmrf",
                     modalities=None, gene_mapper=None, cohort="all",
                     max_genes=2000, longitudinal=False, seed=42, transform=None):
            super().__init__()
            self.data_dir = data_dir
            self.h5ad_path = h5ad_path
            self.source = source
            self.modalities = modalities or ["expression", "mutation", "clinical"]
            self.gene_mapper = gene_mapper or HGNCMapper()
            self.cohort = cohort
            self.max_genes = max_genes
            self.longitudinal = longitudinal
            self.seed = seed
            self.transform = transform
            self.expression = None
            self.mutations = None
            self.clinical_features = None
            self.response_labels = None
            self.patient_ids = None
            self.gene_names = None
            self.metadata = {}
            self.longitudinal_data = {}
            self._train_ids = None
            self._val_ids = None
            self._test_ids = None
            self._active_indices = None
            self._load_data()

        def _load_data(self):
            if self.h5ad_path and os.path.exists(self.h5ad_path):
                self._load_from_h5ad()
            elif self.data_dir:
                if self.source == "gdc":
                    self._load_from_gdc()
                elif self.source == "mmrf":
                    self._load_from_mmrf()
                else:
                    raise ValueError(f"Unknown source: {self.source}")
            else:
                raise ValueError("Must provide either data_dir or h5ad_path")
            self._split_cohorts()

        def _load_from_h5ad(self):
            if not HAS_ANNDATA:
                raise ImportError("anndata required: pip install anndata")
            adata = ad.read_h5ad(self.h5ad_path)
            if sparse.issparse(adata.X):
                self.expression = np.asarray(adata.X.todense())
            else:
                self.expression = np.asarray(adata.X)
            self.gene_names = np.array(adata.var_names)
            self.patient_ids = np.array(adata.obs_names)
            resp_cols = ["response_label", "response", "drug_response", "label"]
            for col in resp_cols:
                if col in adata.obs.columns:
                    self.response_labels = adata.obs[col].values.astype(float)
                    break
            if self.response_labels is None:
                self.response_labels = np.full(len(self.patient_ids), np.nan)
            if "mutations" in (adata.obsm or {}):
                self.mutations = np.asarray(adata.obsm["mutations"])
            elif "mutation" in (adata.layers or {}):
                self.mutations = np.asarray(adata.layers["mutation"])
            self.metadata = {col: adata.obs[col].values for col in adata.obs.columns if col not in resp_cols}

        def _load_from_gdc(self):
            clinical = parse_gdc_clinical(self.data_dir)
            self.response_labels = extract_response_label(clinical).values
            iss = extract_iss_stage(clinical)
            cyto = extract_cytogenetic_risk(clinical)
            treatment = extract_treatment_regimen(clinical)
            self.patient_ids = clinical["patient_id"].values
            if "expression" in self.modalities:
                expr = load_gdc_htseq_counts(self.data_dir, self.gene_mapper)
                sample_map = build_sample_patient_map(self.data_dir)
                expr = map_samples_to_patients(expr, sample_map)
                common_patients = np.intersect1d(self.patient_ids, expr.columns)
                expr = expr[common_patients]
                self.expression = expr.values.T
                self.gene_names = np.array(expr.index)
            if "mutation" in self.modalities:
                try:
                    mut = load_mutation_data(self.data_dir, source="gdc")
                    mut_patients = np.intersect1d(self.patient_ids, mut.columns)
                    self.mutations = mut[mut_patients].values.T
                except FileNotFoundError:
                    logger.warning("No mutation data found")
            self._store_metadata(clinical, iss, cyto, treatment)

        def _load_from_mmrf(self):
            clinical = parse_mmrf_clinical(self.data_dir)
            self.response_labels = extract_response_label(clinical).values
            iss = extract_iss_stage(clinical)
            cyto = extract_cytogenetic_risk(clinical)
            treatment = extract_treatment_regimen(clinical)
            self.patient_ids = clinical["patient_id"].values
            if "expression" in self.modalities:
                expr = load_mmrf_rnaseq(self.data_dir, self.gene_mapper)
                sample_map = build_sample_patient_map(self.data_dir)
                expr = map_samples_to_patients(expr, sample_map)
                common_patients = np.intersect1d(self.patient_ids, expr.columns)
                if len(common_patients) == 0:
                    new_map = {}
                    for col in expr.columns:
                        match = re.match(r"(MMRF_\d+)", col)
                        if match:
                            new_map[match.group(1)] = new_map.get(match.group(1), [])
                            new_map[match.group(1)].append(col)
                    if new_map:
                        patient_expr = {}
                        for pid, cols in new_map.items():
                            patient_expr[pid] = expr[cols].mean(axis=1)
                        expr = pd.DataFrame(patient_expr)
                        common_patients = np.intersect1d(self.patient_ids, expr.columns)
                if len(common_patients) > 0:
                    expr = expr[common_patients]
                    gene_var = expr.var(axis=1)
                    top_genes = gene_var.nlargest(self.max_genes).index
                    expr = expr.loc[top_genes]
                    self.expression = expr.values.T
                    self.gene_names = np.array(expr.index)
                else:
                    logger.warning("No overlapping patient IDs between expression and clinical")
            if "mutation" in self.modalities:
                try:
                    mut = load_mutation_data(self.data_dir, source="mmrf")
                    common = np.intersect1d(self.patient_ids, mut.columns)
                    if len(common) > 0:
                        self.mutations = mut[common].values.T
                except FileNotFoundError:
                    logger.warning("No mutation data found")
            if self.longitudinal:
                self.longitudinal_data = extract_longitudinal_visits(self.data_dir, self.patient_ids.tolist())
            self._store_metadata(clinical, iss, cyto, treatment)

        def _store_metadata(self, clinical, iss, cyto, treatment):
            self.metadata["iss_stage"] = iss.values if isinstance(iss, pd.Series) else iss
            for col in cyto.columns:
                self.metadata[col] = cyto[col].values
            for col in treatment.columns:
                self.metadata[col] = treatment[col].values
            for age_col in ["age_at_diagnosis", "D_PT_age", "AGE", "age"]:
                if age_col in clinical.columns:
                    self.metadata["age"] = pd.to_numeric(clinical[age_col], errors="coerce").values
                    break
            for sex_col in ["gender", "sex", "D_PT_gender", "SEX"]:
                if sex_col in clinical.columns:
                    self.metadata["sex"] = clinical[sex_col].values
                    break

        def _split_cohorts(self):
            if self.patient_ids is None or len(self.patient_ids) == 0:
                return
            iss = self.metadata.get("iss_stage", np.full(len(self.patient_ids), np.nan))
            if isinstance(iss, pd.Series):
                iss = iss.values
            risk = self.metadata.get("risk_score", np.zeros(len(self.patient_ids)))
            if isinstance(risk, pd.Series):
                risk = risk.values
            resp = self.response_labels if self.response_labels is not None else np.full(len(self.patient_ids), np.nan)
            self._train_ids, self._val_ids, self._test_ids = stratified_cohort_split(
                self.patient_ids, resp, iss, risk, seed=self.seed
            )
            if self.cohort == "train":
                self._active_indices = np.isin(self.patient_ids, self._train_ids)
            elif self.cohort == "val":
                self._active_indices = np.isin(self.patient_ids, self._val_ids)
            elif self.cohort == "test":
                self._active_indices = np.isin(self.patient_ids, self._test_ids)
            else:
                self._active_indices = np.ones(len(self.patient_ids), dtype=bool)

        def __len__(self):
            if self._active_indices is not None:
                return int(self._active_indices.sum())
            return len(self.patient_ids) if self.patient_ids is not None else 0

        def __getitem__(self, idx):
            if self._active_indices is not None:
                real_indices = np.where(self._active_indices)[0]
                real_idx = real_indices[idx]
            else:
                real_idx = idx
            sample = {}
            if self.expression is not None:
                sample["expression"] = torch.tensor(self.expression[real_idx], dtype=torch.float32)
            if self.mutations is not None:
                sample["mutations"] = torch.tensor(self.mutations[real_idx], dtype=torch.float32)
            if self.clinical_features is not None:
                sample["clinical"] = torch.tensor(self.clinical_features[real_idx], dtype=torch.float32)
            if self.response_labels is not None:
                label = self.response_labels[real_idx]
                sample["label"] = torch.tensor(0.0 if np.isnan(label) else label, dtype=torch.float32)
                sample["label_mask"] = torch.tensor(not np.isnan(label), dtype=torch.bool)
            sample["patient_id"] = self.patient_ids[real_idx]
            for key in ["iss_stage", "risk_score", "risk_category", "age", "sex"]:
                if key in self.metadata:
                    val = self.metadata[key][real_idx]
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        sample[f"meta_{key}"] = torch.tensor(float(val), dtype=torch.float32)
            if self.transform:
                sample = self.transform(sample)
            return sample

        def get_split_dataset(self, split):
            clone = MMRFCoMMpassDataset.__new__(MMRFCoMMpassDataset)
            clone.__dict__.update(self.__dict__)
            clone.cohort = split
            if split == "train":
                clone._active_indices = np.isin(self.patient_ids, self._train_ids)
            elif split == "val":
                clone._active_indices = np.isin(self.patient_ids, self._val_ids)
            elif split == "test":
                clone._active_indices = np.isin(self.patient_ids, self._test_ids)
            else:
                clone._active_indices = np.ones(len(self.patient_ids), dtype=bool)
            return clone

        def get_features_labels(self, split="all"):
            ds = self.get_split_dataset(split) if split != "all" else self
            indices = np.where(ds._active_indices)[0] if ds._active_indices is not None else np.arange(len(self.patient_ids))
            features = []
            if self.expression is not None:
                features.append(self.expression[indices])
            if self.mutations is not None:
                features.append(self.mutations[indices])
            if not features:
                raise ValueError("No features loaded")
            X = np.concatenate(features, axis=1)
            y = self.response_labels[indices] if self.response_labels is not None else np.zeros(len(indices))
            mask = ~np.isnan(y)
            return X[mask], y[mask]

        @property
        def n_genes(self):
            return self.expression.shape[1] if self.expression is not None else 0

        @property
        def n_patients(self):
            return len(self)

        def summary(self):
            lines = [f"MMRFCoMMpassDataset (cohort={self.cohort})", f"  Patients: {self.n_patients}", f"  Genes: {self.n_genes}"]
            if self.mutations is not None:
                lines.append(f"  Mutation genes: {self.mutations.shape[1]}")
            if self.response_labels is not None:
                valid = ~np.isnan(self.response_labels)
                lines.append(f"  Response: {int((self.response_labels[valid]==1).sum())} R / {int((self.response_labels[valid]==0).sum())} NR / {int((~valid).sum())} missing")
            return "\n".join(lines)

    def collate_variable_length(batch):
        """Collate function for variable-length sequences."""
        result = {}
        keys = batch[0].keys()
        for key in keys:
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                if values[0].dim() == 0:
                    result[key] = torch.stack(values)
                elif all(v.shape == values[0].shape for v in values):
                    result[key] = torch.stack(values)
                else:
                    max_len = max(v.shape[0] for v in values)
                    padded = []
                    masks = []
                    for v in values:
                        pad_size = max_len - v.shape[0]
                        if v.dim() == 1:
                            padded.append(torch.nn.functional.pad(v, (0, pad_size)))
                        else:
                            padded.append(torch.nn.functional.pad(v, (0, 0, 0, pad_size)))
                        mask = torch.ones(max_len, dtype=torch.bool)
                        mask[v.shape[0]:] = False
                        masks.append(mask)
                    result[key] = torch.stack(padded)
                    result[f"{key}_mask"] = torch.stack(masks)
            elif isinstance(values[0], str):
                result[key] = values
            else:
                result[key] = values
        return result

else:
    class MMRFCoMMpassDataset:
        """Numpy-only fallback for MMRFCoMMpassDataset."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for MMRFCoMMpassDataset: pip install torch")

    def collate_variable_length(batch):
        raise ImportError("PyTorch required")


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_hgnc_mapper():
    mapper = HGNCMapper()
    assert mapper.harmonize("ENSG00000141510.5") == "ENSG00000141510"
    assert mapper.harmonize("TP53") == "TP53"
    print("  [PASS] test_hgnc_mapper")


def test_response_label_extraction():
    clinical = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4", "P5"],
        "bestresponse": ["CR", "VGPR", "PD", "SD", "PR"],
    })
    labels = extract_response_label(clinical)
    assert labels.iloc[0] == 1.0
    assert labels.iloc[2] == 0.0
    assert labels.iloc[4] == 1.0
    print("  [PASS] test_response_label_extraction")


def test_cytogenetic_risk():
    clinical = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "FISH_t_4_14": ["Yes", "No", "Yes"],
        "FISH_del_17p": ["No", "No", "Yes"],
    })
    cyto = extract_cytogenetic_risk(clinical)
    assert cyto.loc[0, "t(4;14)"] == True
    assert cyto.loc[2, "risk_score"] == 2
    assert cyto.loc[1, "risk_score"] == 0
    print("  [PASS] test_cytogenetic_risk")


def test_stratified_split():
    np.random.seed(42)
    n = 200
    pids = np.array([f"P{i}" for i in range(n)])
    resp = np.random.choice([0.0, 1.0], size=n, p=[0.3, 0.7])
    iss = np.random.choice([1.0, 2.0, 3.0], size=n)
    risk = np.random.choice([0, 1, 2], size=n)
    train, val, test = stratified_cohort_split(pids, resp, iss, risk)
    assert len(train) + len(val) + len(test) == n
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(train, val)) == 0
    assert abs(len(train) / n - 0.70) < 0.05
    print("  [PASS] test_stratified_split")


def run_tests():
    print("Running mmrf_loader tests...")
    test_hgnc_mapper()
    test_response_label_extraction()
    test_cytogenetic_risk()
    test_stratified_split()
    print("All mmrf_loader tests passed!")


if __name__ == "__main__":
    run_tests()
