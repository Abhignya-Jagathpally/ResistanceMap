# ResistanceMap Architecture: Data Quality, MCP Integration, and Latent Compute

## Overview

This document describes three critical production systems for ResistanceMap:

1. **Data Quality Pipeline** - Comprehensive dataset profiling against SOTA benchmarks
2. **MCP Integration Layer** - Unified connectors to external data sources (PubMed, ChEMBL, ClinicalTrials.gov, HuggingFace)
3. **Latent Compute Scheduler** - Priority-queue based execution of expensive background operations

## 1. Data Quality Pipeline

### Location
`resistancemap/data_quality/`

### Files
- `__init__.py` - Module exports
- `profiler.py` - ~716 lines

### Key Classes

#### DataQualityReport
Comprehensive quality assessment report containing:
- Basic statistics: sample count, feature count, missing rate, outlier rate
- Detailed quality reports: missingness patterns, batch effects, class balance
- SOTA comparisons: metrics vs published benchmarks
- Domain-specific checks: MM driver protein coverage
- Quality score: 0-1 composite quality metric
- Recommendations: actionable quality improvement suggestions

#### DataProfiler
Analyzes datasets against published SOTA benchmarks:

**Known Benchmarks:**
- MMRF_CoMMpass: 1143 MM patients, median OS 82.3 months
- CCLE_proteomics: 375 cell lines, 8498 proteins, 0.73 coverage
- GDSC: 198 compounds × 987 cell lines
- STRING_PPI_v12: 19,566 proteins, 11.9M interactions, 7,853 MM subnet

**MM Driver Proteins** (13 required):
BCMA, CD38, GPRC5D, FGFR3, KRAS, NRAS, TP53, DIS3, FAM46C, BRAF, TRAF3, MAX, IRF4

**Key Methods:**

```python
profile_dataset(data, dataset_name, labels, batch_key, feature_names)
    -> DataQualityReport

_check_missingness(data, feature_names)
    -> MissingnessReport
    Tests for MCAR (Missing Completely At Random) vs MAR/MNAR

_compute_outlier_rate(data)
    -> float (0-1)
    IQR-based outlier detection

_check_batch_effects(data, batch_key)
    -> BatchEffectReport
    kBET and LISI scores; recommends correction method (Harmony, scVI, ComBat-seq)

_check_class_balance(labels)
    -> ClassBalanceReport
    Shannon entropy, effective number of classes, reweighting suggestions

_check_protein_coverage(feature_names)
    -> Dict[protein_name, bool]
    Validates presence of MM drivers

_compute_quality_score(report) -> float
    Composite score: 1.0 - penalties + bonuses
    - Missingness: -20%
    - Outliers: -20%
    - Batch effects: -20%
    - Class imbalance: -15%
    - Feature entropy bonus: +10%
    - Missing drivers penalty: -1% per missing
```

### Data Structures

```python
@dataclass
MissingnessReport:
    missing_rate_per_feature: Dict[str, float]
    global_missing_rate: float
    mcar_test_p_value: Optional[float]
    missingness_pattern: str  # MCAR/MAR/MNAR
    features_with_high_missingness: List[str]
    max_missing_feature: Tuple[str, float]

@dataclass
BatchEffectReport:
    batch_key: str
    n_batches: int
    kbet_score: float  # 0-1, lower = less effect
    lisi_score: float  # > 1.5 = good integration
    correctable: bool
    recommended_method: str

@dataclass
ClassBalanceReport:
    class_frequencies: Dict[str, int]
    class_proportions: Dict[str, float]
    shannon_entropy: float
    effective_number_of_classes: float
    imbalanced: bool  # if any class < 5%
    recommended_reweighting: Dict[str, float]
```

### Dependencies
- numpy (numerical operations)
- Standard library only (logging, dataclasses, typing, datetime, enum, abc)

---

## 2. MCP Integration Layer

### Location
`resistancemap/mcp_integration/`

### Files
- `__init__.py` - Module exports
- `connectors.py` - ~804 lines

### Architecture

#### MCPConnector (Abstract Base)
All connectors inherit from this base class providing:
- **Rate limiting**: Token bucket algorithm (configurable per-connector)
- **Caching**: LRU cache with TTL support
- **Validation**: Response validation abstract method
- **Error handling**: Consistent logging and exception handling

```python
class MCPConnector(ABC):
    async def query(params: dict) -> dict
    def validate_response(response: dict) -> Tuple[bool, Optional[str]]
    async def query_with_cache(**params) -> Optional[dict]
```

#### LRUCache
Thread-safe, TTL-aware LRU cache:
- Max size: 1000 entries (configurable)
- TTL: Per-entry expiration (default 1 hour)
- Eviction: Least recently used when full

#### RateLimiter
Token bucket rate limiter:
- Configurable rate (tokens per second)
- Burst support
- Async-aware (doesn't block event loop)

### Connector Implementations

#### PubMedConnector
**Rate limit**: 3 req/sec (conservative for public API)

Methods:
```python
async search_resistance_literature(gene: str, drug: str, max_results: int)
    -> List[Dict]  # Articles on gene-drug resistance

async verify_protein_drug_association(protein: str, drug: str)
    -> Dict  # {protein, drug, confidence: 0-1, evidence_count, articles}

async get_latest_mm_studies(n: int = 50)
    -> List[Dict]  # Recent MM research
```

#### ChEMBLConnector
**Rate limit**: 5 req/sec

Methods:
```python
async get_drug_targets(drug_name: str)
    -> List[Dict]  # Targets with bioactivity data

async get_bioactivity(target_chembl_id: str, activity_type: str = "IC50")
    -> List[Dict]  # IC50, Ki, Kd measurements

async validate_predicted_target(protein: str, drug: str)
    -> Dict  # {protein, drug, is_valid, confidence, evidence_count}
```

#### ClinicalTrialsConnector
**Rate limit**: 5 req/sec

Methods:
```python
async search_mm_trials(drug: str, phase: Optional[str])
    -> List[Dict]  # MM trials for drug

async get_trial_outcomes(nct_id: str)
    -> Dict  # {primary_outcomes, secondary_outcomes, results_available}
```

#### HuggingFaceConnector
**Rate limit**: 10 req/sec

Methods:
```python
async download_esm2_model(model_name: str, cache_dir: str)
    -> str  # Path to downloaded model

async check_model_version(model_name: str)
    -> Dict  # {model, available, version}
```

### MCPOrchestrator
Coordinates all connectors:

```python
class MCPOrchestrator:
    register_connector(connector: MCPConnector) -> None

    async query(connector_name: str, method: str, params: dict, use_cache: bool)
        -> Optional[Dict]

    get_cached(key: str) -> Optional[Any]
    set_cached(key: str, value: Any, ttl_seconds: int) -> None
    clear_cache(connector_name: Optional[str]) -> None
    cache_stats() -> Dict  # {total_entries, max_size}
```

### Design Decisions

**IN-HOUSE (Custom, Domain-Specific):**
- Data preprocessing pipelines
- Model training and inference
- Zero-trust verification logic

**LEVERAGE OPEN-SOURCE MCPs:**
- PubMed literature search → MCP PubMed tools
- ChEMBL drug data → MCP ChEMBL tools
- ClinicalTrials.gov → MCP clinical trials tools
- HuggingFace model hub → Direct API for ESM-2
- STRING PPI → Direct download (versioned)

### Dependencies
- asyncio (async operations)
- hashlib (cache key generation)
- json (serialization)
- Standard library only (logging, dataclasses, typing, datetime, collections, heapq)

---

## 3. Latent Compute Scheduler

### Location
`resistancemap/latent_compute/`

### Files
- `__init__.py` - Module exports
- `scheduler.py` - ~569 lines

### Architecture

#### ComputeJob
Represents a single schedulable job:

```python
@dataclass
ComputeJob:
    job_id: str  # Unique identifier
    operation: str  # Name (e.g., "esm2_embedding")
    priority: int  # Lower = higher priority (0-10)
    status: JobStatus  # queued/running/completed/failed/cancelled

    # Resource estimates
    estimated_flops: float  # Estimated FLOPs
    estimated_memory_gb: float  # GPU memory estimate

    # Execution metadata
    fn: Callable  # Function to execute
    args: Tuple  # Positional arguments
    kwargs: Dict  # Keyword arguments
    dependencies: List[str]  # Job IDs this depends on

    # Results
    result: Any  # Execution result
    error: Optional[str]  # Error message if failed
    submitted_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
```

#### JobStatus
```python
enum JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

#### LatentComputeScheduler
Main scheduler managing job execution:

**Constructor:**
```python
def __init__(
    gpu_memory_budget_gb: float = 24.0,  # GPU memory
    max_concurrent_jobs: int = 2,  # Max parallel jobs
    max_queue_size: int = 1000,  # Max jobs in queue
)
```

**Core Methods:**

```python
def submit(
    operation: str,
    fn: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict] = None,
    priority: int = 5,
    estimated_flops: float = 0.0,
    estimated_memory_gb: float = 0.0,
    dependencies: Optional[List[str]] = None,
) -> str  # Returns job_id

async def start() -> None
    # Start scheduler main loop

async def stop(wait_for_completion: bool = False) -> None
    # Stop scheduler

def get_job_status(job_id: str) -> Optional[ComputeJob]

def cancel_job(job_id: str) -> bool
    # Returns True if cancelled, False if already running

def estimate_completion_time(job_id: str) -> Optional[float]
    # Estimated seconds until completion
```

**Statistics:**
```python
def queue_depth() -> int
def running_count() -> int
def completed_count() -> int
def failed_count() -> int
def gpu_memory_utilization() -> float  # 0-100%
def get_scheduler_stats() -> Dict[str, Any]
def get_running_jobs_summary() -> List[Dict]
def get_queue_preview(limit: int = 10) -> List[Dict]
```

**Scheduling Logic:**

The scheduler's main loop (`_run_scheduler`) continuously:
1. Checks for completed jobs
2. Submits new jobs respecting:
   - Priority queue order (min-heap by priority)
   - Max concurrent job limit
   - GPU memory budget
   - Job dependency satisfaction
3. Updates job status and resource accounting

Priority calculation:
```
Job ordering: (priority, submitted_at)
```

Memory accounting:
```
gpu_memory_used_gb += job.estimated_memory_gb  # On start
gpu_memory_used_gb -= job.estimated_memory_gb  # On completion
Cannot start job if: used + job.estimate > budget
```

### Pre-configured Operations

#### schedule_esm2_embedding
ESM-2 protein embedding computation:
```python
def schedule_esm2_embedding(
    sequences: List[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    cache_dir: str = "./models",
    priority: int = 3,
) -> str
```

Estimates:
- FLOPs: 1 TFLOP per sequence
- Memory: 4GB + 2MB per sequence
- For 7853 proteins: ~20GB peak

#### schedule_ppi_subgraph_extraction
PPI subgraph extraction for protein lists:
```python
def schedule_ppi_subgraph_extraction(
    protein_list: List[str],
    ppi_network_path: str = "./data/string_ppi_v12.pkl",
    priority: int = 4,
) -> str
```

Estimates:
- FLOPs: 100 GFLOP per protein
- Memory: 2GB + 1MB per protein
- For 7853 proteins: ~10GB peak

#### schedule_trajectory_ensemble
Latent space trajectory generation:
```python
def schedule_trajectory_ensemble(
    initial_states: Any,  # torch.Tensor
    n_samples: int = 1000,
    n_steps: int = 100,
    priority: int = 6,
) -> str
```

Estimates:
- FLOPs: 100 GFLOP per sample*step
- Memory: 4GB + 100KB per sample*step
- For 1000 samples × 100 steps: ~14GB peak

### Scheduler Internals

**Job Queue:**
```
Min-heap priority queue
Ordered by: (priority, submitted_at)
Uses heapq for O(log n) insertion/deletion
```

**Job Tracking:**
```
_job_queue: List[ComputeJob]  # Priority queue
_running_jobs: Dict[job_id, ComputeJob]
_completed_jobs: Dict[job_id, ComputeJob]
_failed_jobs: Dict[job_id, ComputeJob]
_all_jobs: Dict[job_id, ComputeJob]  # All jobs ever submitted
```

**Main Loop:**
```python
async def _run_scheduler():
    while _scheduler_running:
        await _check_completed_jobs()
        await _submit_available_jobs()
        await asyncio.sleep(0.1)  # 100ms polling interval
```

**Dependency Resolution:**
```python
def _are_dependencies_satisfied(job):
    for dep_id in job.dependencies:
        if dep_id not in _completed_jobs:
            return False
    return True
```

### Dependencies
- asyncio (async operations, event loop)
- Standard library only (logging, uuid, dataclasses, datetime, enum, typing, heapq)

---

## Integration Points

### Data Quality → MCP Integration
```python
# Data profiler can use MCP for external validation
profiler = DataProfiler()
report = profiler.profile_dataset(data, "MMRF_CoMMpass")

# MCPOrchestrator can validate MM drivers via PubMed/ChEMBL
orchestrator = MCPOrchestrator()
for driver in report.mm_driver_proteins_missing:
    validation = await orchestrator.query(
        "pubmed",
        "verify_protein_drug_association",
        {"protein": driver, "drug": "proteasome_inhibitor"}
    )
```

### MCP Integration → Latent Compute Scheduler
```python
# Schedule model downloads from HuggingFace
job_id = scheduler.schedule_esm2_embedding(
    sequences=protein_sequences,
    model_name="facebook/esm2_t33_650M_UR50D",
    cache_dir="./models"
)

# Track job
status = scheduler.get_job_status(job_id)
```

### Latent Compute → Data Quality
```python
# ESM-2 embeddings can be validated with data quality checks
embeddings = scheduler.get_job_status(job_id).result

# Profile embedding quality
embedding_report = profiler.profile_dataset(
    embeddings,
    "esm2_embeddings",
    feature_names=[f"esm2_dim_{i}" for i in range(650)]
)
```

---

## Example Usage

### Data Quality Profiling
```python
from resistancemap.data_quality import DataProfiler

profiler = DataProfiler()
report = profiler.profile_dataset(
    data=gene_expression_matrix,
    dataset_name="MMRF_CoMMpass",
    labels=survival_labels,
    batch_key=batch_assignments,
    feature_names=gene_symbols
)

print(report.summary_string())
# Output includes:
# - Quality score: 87.5%
# - MM drivers: 13/13 present
# - Issues: High batch effects (kBET=0.68)
# - Recommendations: Apply Harmony batch correction
```

### MCP Query Orchestration
```python
from resistancemap.mcp_integration import MCPOrchestrator, PubMedConnector

orchestrator = MCPOrchestrator()
orchestrator.register_connector(PubMedConnector(rate_limit=3.0))

# Search for literature
articles = await orchestrator.query(
    "pubmed",
    "search_resistance_literature",
    {"gene": "BCMA", "drug": "teclistamab", "max_results": 50}
)
```

### Latent Compute Scheduling
```python
from resistancemap.latent_compute import LatentComputeScheduler

scheduler = LatentComputeScheduler(
    gpu_memory_budget_gb=24.0,
    max_concurrent_jobs=2
)

await scheduler.start()

# Schedule ESM-2 embeddings
job_id = scheduler.schedule_esm2_embedding(
    sequences=protein_sequences,
    priority=3
)

# Poll for completion
while True:
    job = scheduler.get_job_status(job_id)
    if job.status == JobStatus.COMPLETED:
        embeddings = job.result
        break
    await asyncio.sleep(5)

await scheduler.stop()
```

---

## Performance Considerations

### Data Quality Profiler
- Profile operation: O(n_samples × n_features)
- Memory: O(n_samples) for intermediate arrays
- Time: ~seconds for typical datasets (1000 samples, 5000 features)
- Cacheable: Dataset profiles can be cached

### MCP Integration
- Network latency: 100-500ms per request (external APIs)
- Cache hit rate: ~70% for repeated queries
- Rate limiting prevents API throttling
- Global cache: 2000 entry LRU for orchestrator

### Latent Compute Scheduler
- Scheduling overhead: O(log n) per job (heap operations)
- Priority queue depth: 0-1000 jobs
- GPU memory tracking: Real-time accounting
- Job completion check: 10 Hz polling (100ms interval)

---

## Testing Recommendations

### Unit Tests
- DataProfiler: Test each metric computation with synthetic data
- MCPConnector: Mock external API responses
- LatentComputeScheduler: Simulate jobs with known duration

### Integration Tests
- Profile real MM dataset (MMRF_CoMMpass)
- Query actual PubMed API (with rate limiting)
- Run full scheduler pipeline with ESM-2 embeddings

### Performance Tests
- Profile 1M+ sample datasets
- Stress test scheduler with 100+ job queue
- Measure cache hit rates under typical workload

---

## Production Checklist

- [ ] MCP server authentication configured
- [ ] GPU memory budget validated for hardware
- [ ] Rate limits tested with API providers
- [ ] Cache TTL tuned for data freshness
- [ ] Job timeout handling implemented
- [ ] Scheduler state persistence (if needed)
- [ ] Monitoring/alerting for job failures
- [ ] Logging configured for audit trail
