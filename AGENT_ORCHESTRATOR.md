# ResistanceMap Multi-Agent Orchestrator

Production-grade parallel agent system replacing sequential `main.py` execution with a DAG-based, zero-trust orchestrator for maximum parallelism and data integrity.

## Overview

The multi-agent orchestrator executes ResistanceMap's pipeline as a directed acyclic graph (DAG) of specialized agents, enabling:

- **Maximum Parallelism**: Agents with independent dependencies run concurrently
- **Zero-Trust Verification**: All outputs verified via SHA256 hashing
- **Isolated Execution**: Each agent runs in an isolated context
- **Complete Audit Trail**: Full verification chain recorded for compliance
- **Async-First**: Built on asyncio for high-concurrency execution

## Architecture

### Core Components

#### 1. BaseAgent (`agents/base.py`)
Abstract base class defining the agent interface:

```python
class BaseAgent(ABC):
    name: str                    # Unique identifier
    dependencies: list[str]      # Agent names this depends on

    async def execute(inputs, config) -> AgentResult
    def verify_inputs(inputs) -> (bool, str)
    def verify_output(result, expected_hash) -> (bool, str)
    @staticmethod
    def compute_hash(data) -> str  # SHA256
```

**Key Features:**
- Zero-trust: All outputs hashed via SHA256
- Type-aware hashing: Tensors, dicts, strings, objects
- Input/output validation hooks for subclasses
- Automatic error handling via `_safe_execute()`

#### 2. AgentDAG (`agents/orchestrator.py`)
Dependency graph management:

```python
class AgentDAG:
    agents: dict[name -> BaseAgent]
    edges: dict[name -> dependencies]

    def add_agent(agent)
    def topological_sort() -> list[layers]  # Kahn's algorithm
    def validate() -> (bool, str)          # Cycle detection
```

**Topological Sort Output:**
```
Layer 0 (parallel): [DataValidationAgent, LiteratureAgent]
Layer 1 (parallel): [DataPrepAgent]
Layer 2 (parallel): [VAEPretrainAgent, ESM2EmbedAgent]
Layer 3: [VAEFinetuneAgent]
Layer 4 (parallel): [TrajectoryAgent, ProteinNetAgent]
Layer 5: [FusionAgent]
Layer 6: [LandscapeAgent]
Layer 7: [ValidationAgent]
```

#### 3. Orchestrator (`agents/orchestrator.py`)
Main execution engine:

```python
class Orchestrator:
    dag: AgentDAG
    results: dict[name -> AgentResult]
    execution_plan: ExecutionPlan
    _verification_chain: list[hash]      # Audit trail

    async def run(config) -> dict[AgentResult]
    async def _run_agent_isolated(agent, inputs, config)
    def get_execution_plan() -> list[layers]
    def get_verification_chain() -> list[hash]
    def get_timing_report() -> dict[timing]
    def get_results_summary() -> dict[counts]
```

### Agent Execution Flow

```
1. prepare_execution()
   ├─ Validate DAG (no cycles, all deps exist)
   └─ Topological sort → ExecutionPlan with layers

2. run(config)
   └─ For each layer:
      ├─ Launch all agents concurrently (asyncio.gather)
      ├─ For each completed agent:
      │  ├─ Verify output hash (zero-trust)
      │  ├─ Record verification hash
      │  └─ Store AgentResult
      └─ Pass verified outputs to next layer

3. Results & Audit Trail
   ├─ get_verification_chain() → [hash0, hash1, ...]
   ├─ get_timing_report() → {agent: {start, end, elapsed}}
   └─ get_results_summary() → {status_counts, failed_agents}
```

## Specialized Agents

### 1. DataValidationAgent
**Dependencies:** None (Layer 0)

Validates all input files exist and have expected schemas.

**Checks:**
- Data directories exist (proteomics, epigenomics, PPI)
- CSV files readable with expected columns
- HDF5/H5AD files accessible
- Data shapes reasonable

**Output:**
```python
{
    "valid": bool,
    "errors": [str],
    "paths_checked": {
        "proteomics_path": str,
        "epigenomics_dir": str,
        "ppi_path": str,
        ...
    }
}
```

### 2. DataPrepAgent
**Dependencies:** DataValidationAgent (Layer 1)

Harmonizes omics data via `preprocessors.harmonize_omics()`.

**Operations:**
- Normalize proteomics and epigenomics
- Impute missing values
- Harmonize sample identifiers
- Record data quality metrics

**Output:**
```python
{
    "proteomics_shape": (n_samples, n_proteins),
    "epigenomics_shape": (n_samples, n_peaks),
    "samples": [...],
    "proteins": [...],
    "peaks": [...]
}
```

### 3. VAEPretrainAgent
**Dependencies:** DataPrepAgent (Layer 2)

Pretrains proteome-to-epigenome conditional VAE on large unlabeled dataset.

**Training:**
- Encoder: proteomics → latent space (64-dim)
- Decoder: latent → epigenomics reconstruction
- Loss: reconstruction + KL divergence
- KL annealing: cyclical to prevent posterior collapse

**Output:**
```python
{
    "model_type": "ProteomeToEpigenomeVAE",
    "latent_dim": 64,
    "encoder_hidden_dims": [...],
    "epoch": 200,
    "final_loss": 0.15,
}
```

### 4. VAEFinetuneAgent
**Dependencies:** VAEPretrainAgent (Layer 3)

Finetunes VAE on resistance-labeled data.

**Training:**
- Use pretrained encoder/decoder as initialization
- Finetune with lower learning rate (1e-4)
- 100 epochs (vs 200 for pretraining)
- Reduces overfitting via early stopping

**Output:**
```python
{
    "model_type": "ProteomeToEpigenomeVAE",
    "finetuned": true,
    "epoch": 300,  # 200 + 100
    "final_loss": 0.08,
}
```

### 5. ESM2EmbedAgent
**Dependencies:** DataPrepAgent (Layer 2, parallel with VAEPretrainAgent)

Generates protein embeddings via ESM-2 language model.

**Process:**
- Load ESM-2-T33 (650M parameters, 1280-dim output)
- Batch embed all ~7800 proteins from PPI network
- Cache embeddings for GNN training

**Output:**
```python
{
    "model": "facebook/esm2_t33_650M_UR50D",
    "n_proteins": 7853,
    "embedding_dim": 1280,
    "embedding_shape": (7853, 1280),
}
```

### 6. TrajectoryAgent
**Dependencies:** VAEFinetuneAgent (Layer 4)

Calibrates ODE-based trajectory forecaster.

**Calibration:**
- Use VAE latent space as trajectory anchor
- Fit ODE dynamics to resistance progression
- Monte Carlo sampling for uncertainty quantification
- Forecast resistance at 3, 6, 12 months

**Output:**
```python
{
    "model_type": "ODETrajectory",
    "ode_solver": "dopri5",
    "calibrated": true,
    "n_perturbations": 50,
}
```

### 7. ProteinNetAgent
**Dependencies:** ESM2EmbedAgent, VAEFinetuneAgent (Layer 4, parallel with TrajectoryAgent)

Trains protein interaction GNN.

**Architecture:**
- Input: ESM-2 embeddings (1280-dim)
- Graph: ~460k edges from STRING PPI
- Layers: 4 GAT layers with 8 heads
- Output: protein-level resistance representations

**Output:**
```python
{
    "model_type": "ProteinNetworkGNN",
    "gnn_conv_type": "gat",
    "n_layers": 4,
    "n_heads": 8,
    "n_edges": 460000,
    "trained": true,
}
```

### 8. FusionAgent
**Dependencies:** TrajectoryAgent, ProteinNetAgent (Layer 5)

Trains cross-modal fusion model combining trajectory and protein network.

**Fusion Type:**
- Cross-attention: Query (trajectory) × Key/Value (protein net)
- Hidden: 128-dim intermediate
- Heads: 4 attention heads
- Output: fused representations for landscape prediction

**Output:**
```python
{
    "model_type": "MultiModalFusion",
    "fusion_type": "cross_attention",
    "hidden_dim": 128,
    "trained": true,
}
```

### 9. LandscapeAgent
**Dependencies:** FusionAgent (Layer 6)

Trains resistance landscape predictor.

**Operations:**
- Identify top 20 resistance mechanisms
- Compute confidence scores (threshold: 0.8)
- Generate UMAP visualizations
- Create landscape heatmaps (drug vs. mechanism)

**Output:**
```python
{
    "model_type": "ResistanceLandscapePredictor",
    "n_top_targets": 20,
    "confidence_threshold": 0.8,
    "trained": true,
}
```

### 10. ValidationAgent
**Dependencies:** LandscapeAgent (Layer 7)

Comprehensive validation with SOTA comparison metrics.

**Metrics:**
- Reconstruction MSE (VAE)
- Trajectory RMSE
- Protein network F1 score
- Fusion Spearman correlation
- Landscape top-K accuracy
- SOTA baseline comparison (e.g., DeepDrug3D)

**Output:**
```python
{
    "reconstruction_mse": 0.12,
    "trajectory_rmse": 0.15,
    "protein_net_f1": 0.78,
    "fusion_spearman": 0.85,
    "landscape_top_k": 0.88,
    "sota_baseline_auc": 0.87,
    "improvement_over_sota": 0.04,
    "validation_passed": true,
}
```

## Zero-Trust Verification

### Hash Computation
All agent outputs hashed via SHA256:

```python
def compute_hash(data):
    if torch.Tensor:
        hash(shape + dtype + bytes)
    elif dict/list:
        hash(json.dumps(sorted))
    elif str:
        hash(encoded)
    else:
        hash(str_representation)
```

### Verification Chain
After each agent completes:
1. Compute output hash: `hash = compute_hash(agent.output)`
2. Verify integrity: `agent_result.verification_hash == hash`
3. Record in chain: `verification_chain.append(hash)`
4. Pass to dependents: Verify before use

### Audit Trail
```python
verification_chain = [
    "8f2a1c3d...",  # DataValidationAgent
    "c9e3b5f2...",  # DataPrepAgent
    "1a4d7e9c...",  # VAEPretrainAgent
    ...
]
```

## Usage

### 1. Basic Usage

```python
from resistancemap.agents import Orchestrator, *Agent
from resistancemap.config import ResistanceMapConfig
import asyncio

async def main():
    # Initialize
    orchestrator = Orchestrator()

    # Register agents
    orchestrator.add_agent(DataValidationAgent())
    orchestrator.add_agent(DataPrepAgent())
    # ... all agents

    # Prepare
    is_valid, error = orchestrator.prepare_execution()
    if not is_valid:
        raise ValueError(error)

    # Execute
    config = ResistanceMapConfig()
    results = await orchestrator.run(config)

    # Analyze
    plan = orchestrator.get_execution_plan()
    chain = orchestrator.get_verification_chain()
    timing = orchestrator.get_timing_report()
    summary = orchestrator.get_results_summary()

asyncio.run(main())
```

### 2. Example Script

See `agents/example_usage.py` for complete working example:

```bash
cd ResistanceMap
python -m resistancemap.agents.example_usage
```

Outputs:
- `checkpoints/verification_chain.json` - Audit trail
- `checkpoints/execution_plan.json` - DAG structure
- `checkpoints/results_summary.json` - All results & timing

## Performance Characteristics

### Parallelization
With the DAG, execution efficiency depends on agent durations:

```
Sequential (7 stages):
  T_total = T_0 + T_1 + T_2 + T_3 + T_4 + T_5 + T_6

Parallel (optimized):
  T_total = T_0 + T_1 + max(T_2, T_2) + T_3 + max(T_4, T_4) + T_5 + T_6
          = T_0 + T_1 + T_2 + T_3 + T_4 + T_5 + T_6 (if all balanced)
```

**Speedup** depends on layer composition:
- Layer 2: VAEPretrainAgent (200 epochs) vs ESM2EmbedAgent (parallel) → 2x speedup
- Layer 4: TrajectoryAgent vs ProteinNetAgent (parallel) → 2x speedup

### Memory Isolation
Each agent executes in an isolated context:
- Independent input dict for each agent
- Output verification prevents data corruption
- Failed agents don't affect dependents (explicit error handling)

## Error Handling

### Agent Failure
If an agent fails (status=FAILED):
- Error message recorded in `result.error`
- Dependent agents receive explicit error input
- Orchestrator continues execution
- Summary shows failed agents

### Verification Failure
If output hash verification fails:
- Agent status set to FAILED
- Error message: "Output verification failed: Hash mismatch"
- Dependents notified via error inputs

### Orchestrator Failure
If DAG invalid or execution fails:
- Exception raised to caller
- Results up to failure point still available
- Verification chain frozen at failure point

## Integration with Existing Pipeline

### Replacing main.py
Old sequential flow:
```python
# main.py (old)
run_data_validation()
run_data_prep()
run_vae_pretrain()
run_vae_finetune()
run_esm2_embed()  # Blocked waiting for VAE
run_trajectory()
run_protein_net()  # Blocked waiting for ESM2
```

New parallel flow:
```python
# orchestrator (new)
orchestrator = Orchestrator()
orchestrator.add_agent(DataValidationAgent())
# ... all agents
results = await orchestrator.run(config)
```

### Minimal Changes to Existing Code
- Agents call existing functions: `preprocessors.harmonize_omics()`, `models.vae.train_vae()`, etc.
- No changes needed to data loaders, models, or utilities
- Agent system is a thin orchestration layer above existing code

## Configuration

Agents inherit configuration from `ResistanceMapConfig`:
- `config.data.*` - Data paths and QC thresholds
- `config.vae.*` - VAE hyperparameters
- `config.trajectory.*` - ODE and forecasting settings
- `config.protein_net.*` - GNN hyperparameters
- `config.fusion.*` - Fusion model settings
- `config.landscape.*` - Landscape predictor settings
- `config.hardware.*` - GPU/distributed settings

Example `config.yaml`:
```yaml
data:
  ccle_proteomics_path: data/raw/ccle_proteomics.csv
  min_coverage: 0.7
  imputation: knn

vae:
  latent_dim: 64
  pretrain_epochs: 200
  finetune_epochs: 100

hardware:
  device: cuda
  dtype: bfloat16
  compile: true
```

## Logging & Monitoring

### Stage Logging
Each agent logs start/end via `log_stage_start()` and `log_stage_end()`:

```
2025-01-15 10:30:42 | INFO | >>> STAGE START: data_validation
2025-01-15 10:30:45 | INFO | <<< STAGE END: data_validation (0.1 min) | files_found=3
```

### W&B Integration
Optional Weights & Biases logging:
```python
setup_logger(log_dir="logs", wandb_project="resistancemap", config=config_dict)
```

Logs to W&B:
- Stage timing: `stage/{agent}/elapsed_minutes`
- Completion flags: `stage/{agent}/completed`
- Custom metrics: `stage/{agent}/{key}`

### Verification Chain Export
After execution:
```python
chain = orchestrator.get_verification_chain()
# Export for compliance/audit
with open("verification_chain.json", "w") as f:
    json.dump({"agents": agent_names, "hashes": chain}, f)
```

## Testing

### Unit Tests
Test individual agents:
```python
agent = DataValidationAgent()
result = await agent.execute({}, config)
assert result.status == AgentState.COMPLETED
```

### Integration Tests
Test DAG and orchestrator:
```python
orchestrator = Orchestrator()
# Register agents...
is_valid, _ = orchestrator.prepare_execution()
assert is_valid

results = await orchestrator.run(config)
assert all(r.status == COMPLETED for r in results.values())
```

### Verification Tests
Test zero-trust hashing:
```python
data = {"key": [1, 2, 3]}
hash1 = BaseAgent.compute_hash(data)
hash2 = BaseAgent.compute_hash(data)
assert hash1 == hash2

data["key"].append(4)
hash3 = BaseAgent.compute_hash(data)
assert hash3 != hash1  # Data changed, hash differs
```

## Future Extensions

### Checkpointing
Save/restore agent states:
```python
orchestrator.save_checkpoint("run_20250115.ckpt")
orchestrator.resume_checkpoint("run_20250115.ckpt")  # Skip completed agents
```

### Distributed Execution
Execute agents on different machines:
```python
# Plan for multiprocessing/Ray distribution
async def _run_agent_isolated_remote(agent, inputs):
    # Submit to remote executor
    future = ray.remote(agent.execute).remote(inputs, config)
    return await future
```

### Agent Versioning
Track agent code versions:
```python
@dataclass
class AgentResult:
    agent_version: str  # Code hash or semver
    framework_version: str  # PyTorch, etc.
```

### Custom Metrics
Agent-specific KPIs:
```python
class DataPrepAgent(BaseAgent):
    def execute(self, ...):
        metadata = {
            "n_duplicates_removed": 42,
            "imputation_rate": 0.15,
            "normalization_method": "quantile",
        }
```

## References

- DAG Theory: Kahn's algorithm for topological sort (1962)
- Zero-Trust: NIST ZT Architecture (SP 800-207)
- Async: Python asyncio documentation
- SHA256: FIPS PUB 180-4 (Secure Hash Standard)
