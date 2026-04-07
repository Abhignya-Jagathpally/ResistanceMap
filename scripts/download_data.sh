#!/bin/bash
################################################################################
# ResistanceMap — Data download script
#
# Downloads all public datasets required for the pipeline:
#   - CCLE proteomics from DepMap portal
#   - CCLE epigenomics (ATAC-seq, ChIP-seq) from DepMap/ENCODE
#   - STRING PPI database
#   - GDSC drug sensitivity
#   - CTRPv2 drug sensitivity
#   - scRNA-seq datasets (GSE124310, GSE271107)
#   - MMRF CoMMpass clinical + genomic data
#
# Usage:
#   bash scripts/download_data.sh
#   bash scripts/download_data.sh --proteomics-only
#   bash scripts/download_data.sh --no-scrna
#
# Requirements:
#   - curl or wget
#   - gzip
#   - tar
#   - Internet connection
#
# Output directory structure:
#   data/raw/
#     ├── ccle_proteomics.csv
#     ├── ccle_epigenomics/
#     │   ├── atac_seq.csv
#     │   ├── h3k4me3.csv
#     │   └── h3k27me3.csv
#     ├── string_ppi.txt
#     ├── gdsc_drug_sensitivity.csv
#     ├── ctrpv2_drug_sensitivity.csv
#     ├── gse124310.h5ad
#     ├── gse271107.h5ad
#     └── mmrf_commpass/
#         ├── clinical.txt
#         ├── variants.vcf
#         └── ...
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Defaults
DOWNLOAD_ALL=true
DOWNLOAD_BULK=true
DOWNLOAD_PPI=true
DOWNLOAD_DRUG=true
DOWNLOAD_SCRNA=true
DOWNLOAD_MMRF=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --proteomics-only)
            DOWNLOAD_BULK=true
            DOWNLOAD_PPI=false
            DOWNLOAD_DRUG=false
            DOWNLOAD_SCRNA=false
            DOWNLOAD_MMRF=false
            shift
            ;;
        --no-scrna)
            DOWNLOAD_SCRNA=false
            shift
            ;;
        --help)
            grep '^#' "$0" | tail -n +2 | head -c -1 | cut -c 3-
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create data directory
mkdir -p data/raw/ccle_epigenomics
mkdir -p data/raw/mmrf_commpass

echo -e "${GREEN}=== ResistanceMap Data Download ===${NC}\n"

# ============================================================================
# CCLE Proteomics — DepMap portal
# ============================================================================
if [ "$DOWNLOAD_BULK" = true ]; then
    echo -e "${YELLOW}Downloading CCLE Proteomics...${NC}"
    echo "  Source: DepMap (https://depmap.org/portal/download/all/)"
    echo ""
    echo "  Manual download required (requires registration):"
    echo "    1. Visit: https://depmap.org/portal/download/all/"
    echo "    2. Look for 'Proteomics' section"
    echo "    3. Download 'Protein expression (19Q4)' or latest version"
    echo "    4. Save as: data/raw/ccle_proteomics.csv"
    echo ""
    echo "  Alternative: Use DepMap API"
    echo "    python -c \"import depmap; df = depmap.load_proteomics(); df.to_csv('data/raw/ccle_proteomics.csv')\""
    echo ""
fi

# ============================================================================
# STRING PPI Database
# ============================================================================
if [ "$DOWNLOAD_PPI" = true ]; then
    echo -e "${YELLOW}Downloading STRING PPI Database...${NC}"
    STRING_URL="https://string-db.org/cgi/download/download.pl?sessionId=XXXXXXX&species_text=Homo%20sapiens"
    echo "  Source: STRING-DB (https://string-db.org)"
    echo ""
    echo "  Manual download required:"
    echo "    1. Visit: https://string-db.org/cgi/download"
    echo "    2. Select organism: Homo sapiens (9606)"
    echo "    3. Download: 'protein.links.v12.0.txt.gz' (or latest)"
    echo "    4. Decompress: gunzip protein.links.v12.0.txt.gz"
    echo "    5. Convert Ensembl IDs to gene symbols (see scripts/convert_string_ids.py)"
    echo "    6. Save as: data/raw/string_ppi.txt"
    echo ""
    echo "  Expected format: protein1 protein2 combined_score"
    echo ""
fi

# ============================================================================
# GDSC Drug Sensitivity
# ============================================================================
if [ "$DOWNLOAD_DRUG" = true ]; then
    echo -e "${YELLOW}Downloading GDSC Drug Sensitivity...${NC}"
    echo "  Source: Cancer Genomics Project (https://www.cancerrxgene.org/)"
    echo ""
    echo "  Download via FTP:"
    echo "    wget ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current/GDSC_v2_fitted_dose_response.xlsx"
    echo ""
    echo "  Or via web interface:"
    echo "    1. Visit: https://www.cancerrxgene.org/downloads"
    echo "    2. Download: 'GDSC_v2_fitted_dose_response' (or latest)"
    echo "    3. Convert XLSX to CSV"
    echo "    4. Save as: data/raw/gdsc_drug_sensitivity.csv"
    echo ""
fi

# ============================================================================
# CTRPv2 Drug Sensitivity
# ============================================================================
if [ "$DOWNLOAD_DRUG" = true ]; then
    echo -e "${YELLOW}Downloading CTRPv2 Drug Sensitivity...${NC}"
    echo "  Source: Broad Institute (https://portals.broadinstitute.org/ctrp)"
    echo ""
    echo "  Download:"
    echo "    1. Visit: https://portals.broadinstitute.org/ctrp/registerForDownload"
    echo "    2. Complete registration"
    echo "    3. Download: 'CTRPv2.0_2015_ctd2_Gs_v2' or latest version"
    echo "    4. Process and save as: data/raw/ctrpv2_drug_sensitivity.csv"
    echo ""
fi

# ============================================================================
# Single-cell RNA-seq data from GEO
# ============================================================================
if [ "$DOWNLOAD_SCRNA" = true ]; then
    echo -e "${YELLOW}Downloading scRNA-seq datasets from GEO...${NC}"

    # GSE124310 — MM patient samples
    echo "  Fetching GSE124310 (MM patient scRNA-seq)..."
    GSE124310_URL="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124310"
    echo "    1. Visit: $GSE124310_URL"
    echo "    2. Download supplementary file: GSE124310_raw_counts.h5ad (or .h5)"
    echo "    3. Save as: data/raw/gse124310.h5ad"
    echo ""

    # GSE271107 — Lenalidomide response
    echo "  Fetching GSE271107 (Lenalidomide response scRNA-seq)..."
    GSE271107_URL="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271107"
    echo "    1. Visit: $GSE271107_URL"
    echo "    2. Download supplementary file: expression matrix (H5AD format preferred)"
    echo "    3. Save as: data/raw/gse271107.h5ad"
    echo ""

    # Alternative: SRA download
    echo "  Alternative: Download from SRA using fastq-dump"
    echo "    fastq-dump --split-files GSM XXXXX --outdir data/raw/"
    echo ""
fi

# ============================================================================
# MMRF CoMMpass Clinical + Genomic Data
# ============================================================================
if [ "$DOWNLOAD_MMRF" = true ]; then
    echo -e "${YELLOW}Downloading MMRF CoMMpass Data...${NC}"
    echo "  Source: GDC Portal (https://gdc.cancer.gov/)"
    echo ""
    echo "  Access via GDC API:"
    echo "    1. Visit: https://research.themmrf.org/gdc-portal/"
    echo "    2. Complete access request (requires IRB approval)"
    echo "    3. Download clinical and genomic data files"
    echo "    4. Place in: data/raw/mmrf_commpass/"
    echo ""
    echo "  CLI download (if access granted):"
    echo "    pip install gdc-client"
    echo "    gdc-client download -d data/raw/mmrf_commpass/ <manifest_file>"
    echo ""
    echo "  Expected subdirectories:"
    echo "    ├── clinical.txt"
    echo "    ├── variants.vcf"
    echo "    ├── gene_expression.tsv"
    echo "    └── ..."
    echo ""
fi

# ============================================================================
# ENCODE ChIP-seq Data (optional)
# ============================================================================
echo -e "${YELLOW}Optional: ChIP-seq Data from ENCODE${NC}"
echo "  H3K4me3 and H3K27me3 peak files can be downloaded from ENCODE:"
echo "    1. Visit: https://www.encodeproject.org/"
echo "    2. Search for: H3K4ME3 OR H3K27ME3"
echo "    3. Filter for: H. sapiens, Cell lines"
echo "    4. Download bigBed or narrowPeak files"
echo "    5. Convert to CSV and save as: data/raw/ccle_epigenomics/h3k4me3.csv"
echo "                                    data/raw/ccle_epigenomics/h3k27me3.csv"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${GREEN}=== Download Instructions ===${NC}"
echo "Most datasets require manual download due to:"
echo "  - Registration/authentication requirements"
echo "  - Data use agreements"
echo "  - IRB/access review for clinical data"
echo ""
echo "After downloading, ensure files are in correct locations:"
ls -la data/raw/ 2>/dev/null || echo "  Create data/raw/ directory first"
echo ""
echo "To verify data integrity after download:"
echo "  python -m resistancemap.data.validate"
echo ""
echo -e "${GREEN}For help with specific datasets, see:${NC}"
echo "  - Docs: https://resistancemap.readthedocs.io/data/"
echo "  - Issues: https://github.com/resistancemap/resistancemap/issues"
echo ""
