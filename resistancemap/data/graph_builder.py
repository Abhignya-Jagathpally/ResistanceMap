"""Build PyTorch Geometric graph objects from STRING PPI network with ESM-2 embeddings.

Constructs per-sample graphs where nodes are proteins with ESM-2 embeddings,
and edges are protein-protein interactions. Node features include proteomics,
VAE latent embeddings, and ESM-2 representations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency during data prep
try:
    from torch_geometric.data import Data, Batch
except ImportError:
    Data = None
    Batch = None


def build_protein_index(protein_names: list[str]) -> dict[str, int]:
    """Create a mapping from protein name to integer index.

    Args:
        protein_names: List of protein/gene names from proteomics data.

    Returns:
        Dict mapping protein name → integer index.
    """
    return {name: idx for idx, name in enumerate(protein_names)}


def build_edge_index(
    ppi_edges: list[tuple[str, str]],
    ppi_scores: list[float],
    protein_index: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert STRING PPI edges to PyG edge_index and edge_attr tensors.

    Only includes edges where BOTH proteins are in the protein index
    (i.e., present in proteomics data).

    Args:
        ppi_edges: List of (protein1, protein2) tuples from STRING.
        ppi_scores: List of confidence scores (0-1) for each edge.
        protein_index: Mapping from protein name → index.

    Returns:
        Tuple of (edge_index [2, E], edge_attr [E, 1]).
        - edge_index: Long tensor with shape (2, num_edges).
        - edge_attr: Float tensor with edge weights, shape (num_edges, 1).
    """
    src, dst, weights = [], [], []

    for (p1, p2), score in zip(ppi_edges, ppi_scores):
        if p1 in protein_index and p2 in protein_index:
            i, j = protein_index[p1], protein_index[p2]
            # Undirected: add both directions
            src.extend([i, j])
            dst.extend([j, i])
            weights.extend([score, score])

    if not src:
        logger.warning("No PPI edges matched protein names. Check naming conventions.")
        # Return self-loop graph as fallback
        n = len(protein_index)
        edge_index = torch.stack([torch.arange(n), torch.arange(n)])
        edge_attr = torch.ones(n, 1)
        return edge_index, edge_attr

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    logger.info(
        f"PPI graph: {len(protein_index)} nodes, {edge_index.shape[1]} directed edges"
    )

    return edge_index, edge_attr


def build_sample_graph(
    proteomics: torch.Tensor,
    esm2_embeddings: Optional[torch.Tensor] = None,
    memory_state: Optional[torch.Tensor] = None,
    stability_score: Optional[float] = None,
    edge_index: Optional[torch.Tensor] = None,
    edge_attr: Optional[torch.Tensor] = None,
    drug_sensitivity: Optional[torch.Tensor] = None,
) -> "Data":
    """Build a single PyG Data object for one sample.

    Node features are constructed by concatenating:
        - ESM-2 embeddings (1280 dims, if available)
        - Protein abundance from proteomics (P dims)
        - VAE latent memory state (L dims, broadcast to all nodes)
        - Stability score (1 dim, broadcast to all nodes)

    Args:
        proteomics: (P,) protein abundance vector for this sample.
        esm2_embeddings: (P, 1280) ESM-2 pre-computed embeddings, or None.
        memory_state: (L,) VAE latent vector, or None.
        stability_score: Scalar stability score, or None.
        edge_index: (2, E) PPI edge indices (shared across samples).
        edge_attr: (E, 1) edge weights.
        drug_sensitivity: (D,) IC50 values for target drugs, or None.

    Returns:
        PyG Data object with node features, edges, and labels.

    Raises:
        ImportError: If torch_geometric is not installed.
    """
    if Data is None:
        raise ImportError(
            "torch_geometric is required. Install with: pip install torch-geometric"
        )

    num_nodes = proteomics.shape[0]
    features = []

    # Add ESM-2 embeddings if available (high-dimensional protein representations)
    if esm2_embeddings is not None:
        assert esm2_embeddings.shape[0] == num_nodes, \
            f"ESM-2 embeddings {esm2_embeddings.shape[0]} != nodes {num_nodes}"
        features.append(esm2_embeddings)
    else:
        # If no ESM-2, at least use a learnable embedding per protein
        # (placeholder: will be replaced by actual ESM-2 in production)
        pass

    # Add proteomics abundance
    features.append(proteomics.unsqueeze(1))  # (P, 1)

    # Broadcast global VAE latent state to all nodes
    if memory_state is not None:
        memory_broadcast = memory_state.unsqueeze(0).expand(num_nodes, -1)  # (P, L)
        features.append(memory_broadcast)

    # Broadcast stability score to all nodes
    if stability_score is not None:
        stab_broadcast = torch.full(
            (num_nodes, 1),
            stability_score,
            device=proteomics.device,
            dtype=proteomics.dtype,
        )
        features.append(stab_broadcast)

    # Concatenate all features
    if features:
        node_x = torch.cat(features, dim=1)
    else:
        # Fallback: just use proteomics
        node_x = proteomics.unsqueeze(1)

    # Build PyG Data object
    data = Data(x=node_x)

    if edge_index is not None:
        data.edge_index = edge_index
    if edge_attr is not None:
        data.edge_attr = edge_attr
    if drug_sensitivity is not None:
        data.y = drug_sensitivity

    return data


def build_batch_graphs(
    proteomics_batch: torch.Tensor,
    esm2_embeddings_batch: Optional[torch.Tensor] = None,
    memory_states: Optional[torch.Tensor] = None,
    stability_scores: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    edge_attr: Optional[torch.Tensor] = None,
    drug_sensitivity_batch: Optional[torch.Tensor] = None,
) -> "Batch":
    """Build a batched PyG graph from multiple samples.

    All samples share the same PPI edge structure but have sample-specific
    node features (proteomics, ESM-2, memory state, stability).

    Args:
        proteomics_batch: (B, P) batch of protein abundance vectors.
        esm2_embeddings_batch: (B, P, 1280) batch of ESM-2 embeddings, or None.
        memory_states: (B, L) batch of VAE latent vectors, or None.
        stability_scores: (B,) batch of stability scores, or None.
        edge_index: (2, E) shared PPI edge indices, or None.
        edge_attr: (E, 1) shared edge weights, or None.
        drug_sensitivity_batch: (B, D) batch of IC50 targets, or None.

    Returns:
        PyG Batch object combining all samples.

    Raises:
        ImportError: If torch_geometric is not installed.
    """
    if Batch is None:
        raise ImportError(
            "torch_geometric is required. Install with: pip install torch-geometric"
        )

    graphs = []
    batch_size = proteomics_batch.shape[0]

    for i in range(batch_size):
        # Extract sample-specific data
        esm2 = esm2_embeddings_batch[i] if esm2_embeddings_batch is not None else None
        mem = memory_states[i] if memory_states is not None else None
        stab = stability_scores[i].item() if stability_scores is not None else None
        drug = drug_sensitivity_batch[i] if drug_sensitivity_batch is not None else None

        # Build individual graph
        g = build_sample_graph(
            proteomics=proteomics_batch[i],
            esm2_embeddings=esm2,
            memory_state=mem,
            stability_score=stab,
            edge_index=edge_index,
            edge_attr=edge_attr,
            drug_sensitivity=drug,
        )
        graphs.append(g)

    # Batch graphs (PyG handles node index offsets automatically)
    return Batch.from_data_list(graphs)


def load_esm2_embeddings(
    protein_names: list[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: str = "cuda",
) -> torch.Tensor:
    """Load ESM-2 protein language model embeddings for a list of proteins.

    ESM-2 provides contextualized embeddings capturing protein sequence information.
    This function uses a pre-trained model to generate embeddings.

    Args:
        protein_names: List of gene symbols or UniProt IDs.
        model_name: ESM-2 model identifier (e.g., "facebook/esm2_t33_650M_UR50D").
        device: Device for inference ('cuda' or 'cpu').

    Returns:
        Tensor of shape (len(protein_names), 1280) with ESM-2 embeddings.

    Note:
        In production, this would fetch sequences from UniProt and generate
        embeddings. For now, this is a placeholder that demonstrates the interface.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "transformers required for ESM-2. Install with: pip install transformers"
        )

    logger.info(f"Loading ESM-2 embeddings for {len(protein_names)} proteins")

    # Placeholder: return random embeddings with correct shape
    # In production, would fetch real sequences and embed them
    embeddings = torch.randn(len(protein_names), 1280, device=device)

    logger.info(f"ESM-2 embeddings: {embeddings.shape}")
    return embeddings
