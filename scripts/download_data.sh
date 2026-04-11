#!/bin/bash
################################################################################
# ResistanceMap — Data download script
#
# Downloads all public datasets required for the pipeline:
#   - CCLE proteomics from DepMap portal
#   - CCLE epigenomics (ATAC-seq, chromatin profiling) from DepMap
#   - STRING PPI database v12.0
#   - GDSC drug sensitivity v2
#   - CTRPv2 drug sensitivity
#   - scRNA-seq datasets (GSE124310, GSE271107)
#   - MMRF CoMMpass clinical + genomic data
#
# Usage:
#   bash scripts/download_data.sh
#   bash scripts/download_data.sh --skip-scrna
#   bash scripts/download_data.sh --skip-gdsc
#   bash scripts/download_data.sh --help
#
# Requirements:
#   - curl or wget
#   - gzip
#   - tar
#   - Python 3.7+ (for format conversions and STRING ID conversion)
#   - pandas (for XLSX to CSV conversion)
#   - Internet connection
#
# Output directory structure:
#   data/raw/
#     ├── ccle_proteomics.csv
#     ├── ccle_epigenomics/
#     │   └── chromatin_profiling.csv
#     ├── string_ppi.txt
#     ├── gdsc_drug_sensitivity.csv
#     ├── ctrpv2_drug_sensitivity.csv
#     ├── gse124310.h5ad
#     ├── gse271107.h5ad
#     └── mmrf_commpass/
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
SKIP_SCRNA=false
SKIP_GDSC=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-scrna)
            SKIP_SCRNA=true
            shift
            ;;
        --skip-gdsc)
            SKIP_GDSC=true
            shift
            ;;
        --help)
            grep '^#' "$0" | tail -n +2 | head -c -1 | cut -c 3-
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper function for logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Helper function to download files
download_file() {
    local url="$1"
    local output="$2"
    local description="$3"

    if [ -f "$output" ]; then
        log_warn "$description already exists, skipping download"
        return 0
    fi

    log_info "Downloading $description..."
    log_info "  URL: $url"

    if command -v curl &> /dev/null; then
        if curl -L -f --progress-bar -o "$output" "$url"; then
            log_success "Downloaded: $output"
            return 0
        else
            log_error "Failed to download: $description"
            rm -f "$output"
            return 1
        fi
    elif command -v wget &> /dev/null; then
        if wget --show-progress -O "$output" "$url"; then
            log_success "Downloaded: $output"
            return 0
        else
            log_error "Failed to download: $description"
            rm -f "$output"
            return 1
        fi
    else
        log_error "Neither curl nor wget found. Please install one to download files."
        return 1
    fi
}

# Create data directory structure
log_info "Creating data directory structure..."
mkdir -p data/raw/ccle_epigenomics
mkdir -p data/raw/mmrf_commpass

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}       ResistanceMap Data Download Script${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# CCLE Proteomics — DepMap 23Q4
# ============================================================================
log_info "Processing CCLE Proteomics (DepMap 23Q4 Proteomics)..."
CCLE_PROT_URL="https://ndownloader.figshare.com/files/40449056"
CCLE_PROT_PATH="data/raw/ccle_proteomics.csv"

if download_file "$CCLE_PROT_URL" "$CCLE_PROT_PATH" "CCLE Proteomics"; then
    log_success "CCLE Proteomics ready at: $CCLE_PROT_PATH"
else
    log_error "CCLE Proteomics download failed"
fi
echo ""

# ============================================================================
# STRING PPI Database v12.0
# ============================================================================
log_info "Processing STRING PPI Database (v12.0, Human)..."
STRING_PPI_URL="https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
STRING_PPI_GZ="data/raw/string_ppi.txt.gz"
STRING_PPI_PATH="data/raw/string_ppi.txt"
STRING_ALIAS_URL="https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz"
STRING_ALIAS_GZ="data/raw/string_aliases.txt.gz"
STRING_ALIAS_TXT="data/raw/string_aliases.txt"

if [ ! -f "$STRING_PPI_PATH" ]; then
    # Download compressed PPI file
    if download_file "$STRING_PPI_URL" "$STRING_PPI_GZ" "STRING PPI links"; then
        log_info "Decompressing STRING PPI database..."
        gzip -df "$STRING_PPI_GZ"
        log_success "STRING PPI decompressed to: $STRING_PPI_PATH"
    fi

    # Download alias file for ID conversion
    if [ ! -f "$STRING_ALIAS_TXT" ]; then
        if download_file "$STRING_ALIAS_URL" "$STRING_ALIAS_GZ" "STRING protein aliases"; then
            log_info "Decompressing STRING aliases..."
            gzip -df "$STRING_ALIAS_GZ"
            log_success "STRING aliases decompressed"
        fi
    fi

    # Convert Ensembl IDs to gene symbols if alias file is available
    if [ -f "$STRING_ALIAS_TXT" ] && [ -f "$STRING_PPI_PATH" ]; then
        log_info "Converting STRING Ensembl IDs to gene symbols..."
        python3 << 'EOF'
import sys

def build_id_map(alias_file):
    """Build mapping from Ensembl IDs to gene symbols."""
    id_map = {}
    with open(alias_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                ensembl_id = parts[0]
                alias = parts[1]
                # Prefer gene symbols (typically shorter, uppercase)
                if alias and not alias.startswith('9606.'):
                    if ensembl_id not in id_map or len(alias) < len(id_map[ensembl_id]):
                        id_map[ensembl_id] = alias
    return id_map

# Build the ID mapping
alias_map = build_id_map('data/raw/string_aliases.txt')

# Convert PPI file
converted = 0
with open('data/raw/string_ppi.txt', 'r') as infile, \
     open('data/raw/string_ppi_converted.txt', 'w') as outfile:
    next(infile)  # Skip header

    # Write new header
    outfile.write('protein1\tprotein2\tcombined_score\n')

    for line in infile:
        parts = line.strip().split()
        if len(parts) >= 3:
            ensembl1 = parts[0]
            ensembl2 = parts[1]
            score = parts[2]

            # Map to gene symbols if available, otherwise keep Ensembl ID
            symbol1 = alias_map.get(ensembl1, ensembl1)
            symbol2 = alias_map.get(ensembl2, ensembl2)

            outfile.write(f'{symbol1}\t{symbol2}\t{score}\n')
            converted += 1

print(f'Converted {converted} protein interactions')

# Replace original with converted version
import os
os.replace('data/raw/string_ppi_converted.txt', 'data/raw/string_ppi.txt')
EOF
        log_success "STRING PPI conversion complete"
    fi
else
    log_warn "STRING PPI database already exists, skipping"
fi
echo ""

# ============================================================================
# GDSC Drug Sensitivity v2
# ============================================================================
if [ "$SKIP_GDSC" = false ]; then
    log_info "Processing GDSC Drug Sensitivity v2..."
    GDSC_URL="https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx"
    GDSC_XLSX="data/raw/gdsc_drug_sensitivity.xlsx"
    GDSC_CSV="data/raw/gdsc_drug_sensitivity.csv"

    if [ ! -f "$GDSC_CSV" ]; then
        if download_file "$GDSC_URL" "$GDSC_XLSX" "GDSC Drug Sensitivity XLSX"; then
            log_info "Converting GDSC XLSX to CSV..."
            python3 << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_excel('data/raw/gdsc_drug_sensitivity.xlsx', sheet_name=0)
    df.to_csv('data/raw/gdsc_drug_sensitivity.csv', index=False)
    print(f'Converted GDSC file: {len(df)} rows')
except Exception as e:
    print(f'Error converting GDSC file: {e}', file=sys.stderr)
    sys.exit(1)
EOF
            if [ $? -eq 0 ]; then
                log_success "GDSC CSV conversion complete"
                rm -f "$GDSC_XLSX"
            else
                log_error "GDSC conversion failed"
            fi
        fi
    else
        log_warn "GDSC Drug Sensitivity already exists, skipping"
    fi
else
    log_warn "Skipping GDSC download (--skip-gdsc flag set)"
fi
echo ""

# ============================================================================
# CCLE Epigenomics — DepMap 23Q4 Chromatin Profiling
# ============================================================================
log_info "Processing CCLE Epigenomics (DepMap 23Q4 Chromatin)..."
CCLE_EPIGG_URL="https://ndownloader.figshare.com/files/40449074"
CCLE_EPIGG_PATH="data/raw/ccle_epigenomics/chromatin_profiling.csv"

if download_file "$CCLE_EPIGG_URL" "$CCLE_EPIGG_PATH" "CCLE Chromatin Profiling"; then
    log_success "CCLE Epigenomics ready at: $CCLE_EPIGG_PATH"
else
    log_error "CCLE Epigenomics download failed"
fi
echo ""

# ============================================================================
# scRNA-seq from GEO (GSE124310, GSE271107)
# ============================================================================
if [ "$SKIP_SCRNA" = false ]; then
    log_info "Processing scRNA-seq datasets from GEO..."

    # GSE124310
    GSE124310_PATH="data/raw/gse124310.h5ad"
    if [ ! -f "$GSE124310_PATH" ]; then
        log_info "Attempting to download GSE124310 (MM patient samples)..."

        # Try direct supplementary file access
        GSE124310_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE124nnn/GSE124310/suppl/"

        log_warn "GSE124310: Requires manual download from GEO supplementary files"
        log_warn "  Visit: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124310"
        log_warn "  Download the count matrix or h5ad file and place at: $GSE124310_PATH"
    else
        log_success "GSE124310 already present"
    fi

    # GSE271107
    GSE271107_PATH="data/raw/gse271107.h5ad"
    if [ ! -f "$GSE271107_PATH" ]; then
        log_warn "GSE271107: Requires manual download from GEO supplementary files"
        log_warn "  Visit: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271107"
        log_warn "  Download the count matrix or h5ad file and place at: $GSE271107_PATH"
    else
        log_success "GSE271107 already present"
    fi
else
    log_warn "Skipping scRNA-seq downloads (--skip-scrna flag set)"
fi
echo ""

# ============================================================================
# CTRPv2 Drug Sensitivity
# ============================================================================
log_info "Processing CTRPv2 Drug Sensitivity..."
CTRPV2_URL="https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip"
CTRPV2_ZIP="data/raw/ctrpv2_dataset.zip"
CTRPV2_CSV="data/raw/ctrpv2_drug_sensitivity.csv"

if [ ! -f "$CTRPV2_CSV" ]; then
    log_warn "CTRPv2: Registration may be required"
    log_info "Attempting download from NCI CTD2 portal..."

    if download_file "$CTRPV2_URL" "$CTRPV2_ZIP" "CTRPv2 dataset"; then
        log_info "Extracting CTRPv2 archive..."
        unzip -q "$CTRPV2_ZIP" -d data/raw/ctrpv2_extracted/

        # Look for the response matrix file and convert to CSV
        log_warn "CTRPv2: Please review extracted files in data/raw/ctrpv2_extracted/ and convert response matrix to CSV"
    else
        log_warn "CTRPv2 download may require manual access from: https://portals.broadinstitute.org/ctrp"
    fi
else
    log_success "CTRPv2 already present"
fi
echo ""

# ============================================================================
# MMRF CoMMpass
# ============================================================================
log_info "Processing MMRF CoMMpass Data..."
if [ ! -d "data/raw/mmrf_commpass" ] || [ -z "$(ls -A data/raw/mmrf_commpass)" ]; then
    log_warn "MMRF CoMMpass requires IRB approval for access"
    echo ""
    echo -e "${YELLOW}To download MMRF CoMMpass data:${NC}"
    echo "  1. Visit: https://research.themmrf.org/"
    echo "  2. Complete access request form (requires IRB approval)"
    echo "  3. Download data files from GDC Portal"
    echo "  4. Place files in: data/raw/mmrf_commpass/"
    echo ""
    echo "  Alternatively, use GDC client (if access granted):"
    echo "    pip install gdc-client"
    echo "    gdc-client download -d data/raw/mmrf_commpass/ <manifest_file>"
    echo ""
else
    log_success "MMRF CoMMpass directory exists"
fi
echo ""

# ============================================================================
# Validation and Summary
# ============================================================================
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}       Download Validation Summary${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""

downloaded_files=()
missing_files=()

# Check CCLE Proteomics
if [ -f "data/raw/ccle_proteomics.csv" ]; then
    size=$(du -h "data/raw/ccle_proteomics.csv" | cut -f1)
    downloaded_files+=("✓ CCLE Proteomics ($size)")
else
    missing_files+=("✗ CCLE Proteomics")
fi

# Check STRING PPI
if [ -f "data/raw/string_ppi.txt" ]; then
    size=$(du -h "data/raw/string_ppi.txt" | cut -f1)
    downloaded_files+=("✓ STRING PPI ($size)")
else
    missing_files+=("✗ STRING PPI")
fi

# Check GDSC
if [ -f "data/raw/gdsc_drug_sensitivity.csv" ]; then
    size=$(du -h "data/raw/gdsc_drug_sensitivity.csv" | cut -f1)
    downloaded_files+=("✓ GDSC Drug Sensitivity ($size)")
else
    missing_files+=("✗ GDSC Drug Sensitivity")
fi

# Check CCLE Epigenomics
if [ -f "data/raw/ccle_epigenomics/chromatin_profiling.csv" ]; then
    size=$(du -h "data/raw/ccle_epigenomics/chromatin_profiling.csv" | cut -f1)
    downloaded_files+=("✓ CCLE Epigenomics ($size)")
else
    missing_files+=("✗ CCLE Epigenomics")
fi

# Check GSE124310
if [ -f "data/raw/gse124310.h5ad" ]; then
    size=$(du -h "data/raw/gse124310.h5ad" | cut -f1)
    downloaded_files+=("✓ GSE124310 scRNA-seq ($size)")
else
    missing_files+=("✗ GSE124310 scRNA-seq (manual download required)")
fi

# Check GSE271107
if [ -f "data/raw/gse271107.h5ad" ]; then
    size=$(du -h "data/raw/gse271107.h5ad" | cut -f1)
    downloaded_files+=("✓ GSE271107 scRNA-seq ($size)")
else
    missing_files+=("✗ GSE271107 scRNA-seq (manual download required)")
fi

# Check CTRPv2
if [ -f "data/raw/ctrpv2_drug_sensitivity.csv" ]; then
    size=$(du -h "data/raw/ctrpv2_drug_sensitivity.csv" | cut -f1)
    downloaded_files+=("✓ CTRPv2 Drug Sensitivity ($size)")
else
    missing_files+=("✗ CTRPv2 Drug Sensitivity")
fi

# Check MMRF CoMMpass
if [ -d "data/raw/mmrf_commpass" ] && [ ! -z "$(ls -A data/raw/mmrf_commpass 2>/dev/null)" ]; then
    count=$(ls -1 data/raw/mmrf_commpass | wc -l)
    downloaded_files+=("✓ MMRF CoMMpass ($count files)")
else
    missing_files+=("✗ MMRF CoMMpass (requires IRB approval)")
fi

echo -e "${GREEN}Downloaded:${NC}"
for file in "${downloaded_files[@]}"; do
    echo "  $file"
done
echo ""

if [ ${#missing_files[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing/Pending:${NC}"
    for file in "${missing_files[@]}"; do
        echo "  $file"
    done
    echo ""
fi

echo -e "${BLUE}Directory structure:${NC}"
tree -L 2 data/raw/ 2>/dev/null || find data/raw -type f -o -type d | head -20
echo ""

echo -e "${GREEN}Download complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify all required files are present in data/raw/"
echo "  2. Run data validation: python -m resistancemap.data.validate"
echo "  3. Process data: python scripts/prepare_data.py"
echo ""
