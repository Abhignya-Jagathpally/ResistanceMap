# ResistanceMap - Production Systems Delivery Index

## Quick Reference

### Module 1: Data Quality Pipeline
**Location:** `resistancemap/data_quality/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 25 | Module exports |
| `profiler.py` | 716 | Core quality profiling engine |

**Key Classes:**
- `DataQualityReport` - Comprehensive quality assessment (15+ fields)
- `DataProfiler` - SOTA benchmark comparisons
- `MissingnessReport` - MCAR/MAR/MNAR analysis
- `BatchEffectReport` - Batch effect quantification
- `ClassBalanceReport` - Class balance analysis & reweighting

**Core Method:**
```python
report = profiler.profile_dataset(
    data=matrix,           # np.ndarray (n_samples, n_features)
    dataset_name="string", # Dataset identifier
    labels=array,          # Optional class labels
    batch_key=array,       # Optional batch assignments
    feature_names=list,    # Optional feature names
    sample_names=list      # Optional sample names
)
# Returns: DataQualityReport with 50+ metrics
```

---

### Module 2: MCP Integration Layer
**Location:** `resistancemap/mcp_integration/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 30 | Module exports |
| `connectors.py` | 804 | Connector implementations |

**Core Classes:**
- `MCPConnector` (ABC) - Rate limiting, caching, validation
- `LRUCache` - Thread-safe TTL-aware cache
- `RateLimiter` - Token bucket rate limiter
- `PubMedConnector` - Literature search (3 req/sec)
- `ChEMBLConnector` - Drug-target data (5 req/sec)
- `ClinicalTrialsConnector` - Trial data (5 req/sec)
- `HuggingFaceConnector` - Model hub (10 req/sec)
- `MCPOrchestrator` - Unified connector management

**Core API:**
```python
# Initialize orchestrator
orchestrator = MCPOrchestrator()
orchestrator.register_connector(PubMedConnector())

# Execute query with built-in caching & rate limiting
result = await orchestrator.query(
    connector_name="pubmed",
    method="search_resistance_literature",
    params={"gene": "BCMA", "drug": "teclistamab", "max_results": 50}
)

# Cache management
orchestrator.clear_cache("pubmed")
stats = orchestrator.cache_stats()  # {total_entries, max_size}
```

**Available Methods:**

PubMedConnector:
- `search_resistance_literature(gene, drug, max_results)`
- `verify_protein_drug_association(protein, drug)`
- `get_latest_mm_studies(n)`

ChEMBLConnector:
- `get_drug_targets(drug_name)`
- `get_bioactivity(target_chembl_id, activity_type)`
- `validate_predicted_target(protein, drug)`

ClinicalTrialsConnector:
- `search_mm_trials(drug, phase)`
- `get_trial_outcomes(nct_id)`

HuggingFaceConnector:
- `download_esm2_model(model_name, cache_dir)`
- `check_model_version(model_name)`

---

### Module 3: Latent Compute Scheduler
**Location:** `resistancemap/latent_compute/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 25 | Module exports |
| `scheduler.py` | 569 | Priority queue scheduler |

**Core Classes:**
- `ComputeJob` - Job representation with metadata
- `JobStatus` (enum) - queued, running, completed, failed, cancelled
- `LatentComputeScheduler` - Priority-queue based executor

**Core API:**
```python
# Initialize scheduler
scheduler = LatentComputeScheduler(
    gpu_memory_budget_gb=24.0,  # GPU memory
    max_concurrent_jobs=2,      # Max parallel jobs
    max_queue_size=1000         # Max queue size
)

# Start scheduler main loop
await scheduler.start()

# Submit job
job_id = scheduler.schedule_esm2_embedding(
    sequences=protein_list,
    model_name="facebook/esm2_t33_650M_UR50D",
    cache_dir="./models",
    priority=3  # Lower = higher priority
)

# Monitor job
job = scheduler.get_job_status(job_id)
if job.status == JobStatus.COMPLETED:
    result = job.result

# Get statistics
stats = scheduler.get_scheduler_stats()
# {queue_depth, running_jobs, completed_jobs, failed_jobs, gpu_utilization_percent, ...}

# Stop scheduler
await scheduler.stop(wait_for_completion=True)
```

**Pre-configured Operations:**

1. `schedule_esm2_embedding(sequences, model, cache_dir, priority)`
   - ESM-2 protein embeddings
   - Estimates: 1 TFLOP/seq, 4GB + 2MB/seq

2. `schedule_ppi_subgraph_extraction(protein_list, ppi_path, priority)`
   - PPI subgraph extraction
   - Estimates: 100 GFLOP/protein, 2GB + 1MB/protein

3. `schedule_trajectory_ensemble(initial_states, n_samples, n_steps, priority)`
   - Latent space trajectories
   - Estimates: 100 GFLOP/(sample×step), 4GB + 100KB/(sample×step)

**Scheduling Constraints:**
- Priority queue: Lower value = higher priority
- GPU memory: Budget-aware scheduling
- Concurrency: Max 2 jobs running simultaneously
- Dependencies: Jobs can depend on other jobs

**Monitoring Methods:**
```python
scheduler.queue_depth()              # int
scheduler.running_count()            # int
scheduler.completed_count()          # int
scheduler.failed_count()             # int
scheduler.gpu_memory_utilization()   # float (0-100%)
scheduler.get_scheduler_stats()      # Dict[str, Any]
scheduler.get_running_jobs_summary() # List[Dict]
scheduler.get_queue_preview(limit)   # List[Dict]
```

---

## Integration Examples

### Example 1: Profile Dataset + External Validation
```python
from resistancemap.data_quality import DataProfiler
from resistancemap.mcp_integration import MCPOrchestrator, PubMedConnector

# Profile dataset
profiler = DataProfiler()
report = profiler.profile_dataset(data, "MMRF_CoMMpass", labels=labels)

# Validate missing drivers via PubMed
orchestrator = MCPOrchestrator()
orchestrator.register_connector(PubMedConnector())

for driver in report.mm_driver_proteins_missing:
    validation = await orchestrator.query(
        "pubmed",
        "verify_protein_drug_association",
        {"protein": driver, "drug": "proteasome_inhibitor"}
    )
    print(f"{driver}: confidence={validation['confidence']:.2f}")
```

### Example 2: Schedule Embeddings + Profile Quality
```python
from resistancemap.latent_compute import LatentComputeScheduler
from resistancemap.data_quality import DataProfiler

# Schedule ESM-2 embeddings
scheduler = LatentComputeScheduler()
await scheduler.start()

job_id = scheduler.schedule_esm2_embedding(
    sequences=protein_sequences,
    priority=3
)

# Wait for completion
while scheduler.get_job_status(job_id).status != JobStatus.COMPLETED:
    await asyncio.sleep(10)

# Profile embedding quality
embeddings = scheduler.get_job_status(job_id).result
profiler = DataProfiler()
embedding_quality = profiler.profile_dataset(
    embeddings,
    "esm2_embeddings",
    feature_names=[f"dim_{i}" for i in range(650)]
)
```

### Example 3: Full Pipeline
```python
# 1. Quality check raw data
raw_quality = profiler.profile_dataset(raw_data, "raw_mmrf")
if raw_quality.overall_quality_score < 0.7:
    print("Warning: Low quality data")
    for issue in raw_quality.quality_issues:
        print(f"  - {issue}")

# 2. Schedule expensive preprocessing
preprocess_job = scheduler.submit(
    operation="normalize_and_impute",
    fn=preprocess_function,
    args=(raw_data,),
    priority=2  # Higher priority than embeddings
)

# 3. Schedule ESM-2 embeddings (depends on preprocessing)
embedding_job = scheduler.schedule_esm2_embedding(
    sequences=protein_list,
    priority=3,
    dependencies=[preprocess_job]
)

# 4. Validate via literature
literature_results = await orchestrator.query(
    "pubmed",
    "get_latest_mm_studies",
    {"n": 50}
)

# 5. Monitor progress
print(scheduler.get_queue_preview(limit=5))
print(scheduler.get_scheduler_stats())
```

---

## Testing

### Quick Import Test
```bash
cd /sessions/ecstatic-upbeat-pascal/mnt/Claude_pzs8sxrjxfjjc/ResistanceMap
python3 -c "
from resistancemap.data_quality import DataProfiler
from resistancemap.mcp_integration import MCPOrchestrator
from resistancemap.latent_compute import LatentComputeScheduler
print('All imports successful!')
"
```

### Unit Test Template
```python
import numpy as np
from resistancemap.data_quality import DataProfiler

# Generate synthetic data
profiler = DataProfiler()
data = np.random.randn(100, 50)
feature_names = [f"gene_{i}" for i in range(50)]

# Profile
report = profiler.profile_dataset(
    data=data,
    dataset_name="test",
    feature_names=feature_names
)

# Verify
assert report.n_samples == 100
assert report.n_features == 50
assert 0 <= report.overall_quality_score <= 1
print("Test passed!")
```

---

## File Locations

```
/sessions/ecstatic-upbeat-pascal/mnt/Claude_pzs8sxrjxfjjc/ResistanceMap/

├── resistancemap/
│   ├── data_quality/
│   │   ├── __init__.py
│   │   └── profiler.py
│   ├── mcp_integration/
│   │   ├── __init__.py
│   │   └── connectors.py
│   └── latent_compute/
│       ├── __init__.py
│       └── scheduler.py
├── ARCHITECTURE.md          (Detailed architecture guide)
└── DELIVERY_INDEX.md        (This file)
```

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Lines | 2,169 |
| Data Quality | 741 |
| MCP Integration | 834 |
| Latent Compute | 594 |
| Classes | 19 |
| Functions | 60+ |
| Type Hints | 100% |
| Dependencies | numpy + stdlib |
| Python Version | 3.8+ |

---

## Next Steps

1. **Test with Real Data**
   - Profile MMRF_CoMMpass dataset
   - Validate MM driver coverage
   - Check batch effects (if applicable)

2. **Connect MCP Servers**
   - Set up PubMed authentication
   - Configure ChEMBL API access
   - Test ClinicalTrials.gov queries
   - Set HuggingFace cache directory

3. **Configure Scheduler**
   - Adjust GPU memory budget for your hardware
   - Test ESM-2 embedding scheduling
   - Monitor job completion times
   - Tune rate limits based on API responses

4. **Integration Testing**
   - Run full pipeline on sample data
   - Verify cross-module data flow
   - Stress test scheduler with 100+ jobs
   - Monitor cache hit rates

---

## Support

For detailed architecture documentation, see `ARCHITECTURE.md`.

For specific class/method details, refer to docstrings in source files.
