"""L3 Protein Network Propagator for ResistanceMap.

Combines ESM-2 protein embeddings with GNN-based propagation on PPI networks
and pathway-aware encoding to compute per-protein resistance contributions.

Architecture:
    1. ESM2Embedder: Loads facebook/esm2_t33_650M_UR50D, computes 1280-dim embeddings
    2. PPIGraphNetwork: Adapted from R4 GATModel (r4/graph_ml/gnn_models.py)
       with residual connections and LayerNorm for protein feature aggregation
    3. PathwayAwareEncoder: Adapted from R4 proteomics_encoder.py for cross-attention
       between protein features and pathway embeddings
    4. ResistancePropagator: Uses GlobalAttention pooling (from MyeloMemory pattern)
       to generate per-protein resistance contribution scores

Components sourced from:
    - R4 GATModel: r4/graph_ml/gnn_models.py (residual + LayerNorm + edge_dim support)
    - R4 PathwayAwareEncoder: r4/graph_ml/proteomics_encoder.py
    - MyeloMemory GlobalAttention pattern: myelomemory/models/gnn.py
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATConv, global_mean_pool, GlobalAttention
    from torch_geometric.data import Data as PyGData, Batch as PyGBatch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("torch_geometric not available, protein network models will use fallback")


class ESM2Embedder(nn.Module):
    """ESM-2 protein sequence embedder from HuggingFace.

    Loads facebook/esm2_t33_650M_UR50D and produces 1280-dimensional embeddings
    for protein sequences. Includes LRU cache to avoid recomputation.

    Attributes:
        cache_size (int): Max number of sequences to cache in memory.
        device (str): Device to run embeddings on.
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D",
                 cache_size: int = 10000, device: str = "cpu"):
        """Initialize ESM-2 embedder.

        Args:
            model_name: HuggingFace model identifier.
            cache_size: Maximum sequences to cache.
            device: Device for computation.
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.cache_size = cache_size
        self.embedding_cache: Dict[str, torch.Tensor] = {}

        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
            self.embedding_dim = 1280
            logger.info(f"Loaded ESM-2 model {model_name} with embedding_dim={self.embedding_dim}")
        except ImportError:
            logger.error("transformers library required for ESM2Embedder")
            self.model = None
            self.tokenizer = None
            self.embedding_dim = 1280

    def embed_proteins(self, sequences: List[str]) -> torch.Tensor:
        """Compute embeddings for protein sequences.

        Args:
            sequences: List of protein sequences (amino acid strings).

        Returns:
            Tensor of shape (len(sequences), 1280) with embeddings.
        """
        if self.model is None:
            logger.warning("ESM-2 model not loaded, returning random embeddings")
            return torch.randn(len(sequences), self.embedding_dim, device=self.device)

        # Check cache for hits
        embeddings = []
        uncached_sequences = []
        uncached_indices = []

        for i, seq in enumerate(sequences):
            if seq in self.embedding_cache:
                embeddings.append((i, self.embedding_cache[seq]))
            else:
                uncached_sequences.append(seq)
                uncached_indices.append(i)

        # Compute uncached embeddings
        if uncached_sequences:
            with torch.no_grad():
                inputs = self.tokenizer(
                    uncached_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024
                ).to(self.device)
                outputs = self.model(**inputs)
                sequence_embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence length

            for idx, seq, emb in zip(uncached_indices, uncached_sequences, sequence_embeddings):
                self.embedding_cache[seq] = emb.cpu()
                embeddings.append((idx, emb.cpu()))

                # Evict oldest if cache full
                if len(self.embedding_cache) > self.cache_size:
                    self.embedding_cache.pop(next(iter(self.embedding_cache)))

        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        result = torch.stack([e[1] for e in embeddings]).to(self.device)
        return result

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Wrapper for embed_proteins."""
        return self.embed_proteins(sequences)


class PPIGraphNetwork(nn.Module):
    """Graph Attention Network on PPI networks (adapted from R4 GATModel).

    Architecture:
        - Input projection: node_features (1345-dim) → hidden_dim
        - N layers of GATConv with residual connections and LayerNorm
        - Output projection: hidden_dim → hidden_dim

    Node features are concatenated from:
        - ESM-2 embedding (1280)
        - Latent state (64)
        - Stability score (1)
        Total: 1345 dimensions

    Adapted from: r4/graph_ml/gnn_models.py (GATModel with residual connections)
    """

    def __init__(self, in_dim: int = 1345, hidden_dim: int = 256,
                 n_layers: int = 4, n_heads: int = 8, dropout: float = 0.2,
                 edge_dim: Optional[int] = 1):
        """Initialize PPIGraphNetwork.

        Args:
            in_dim: Input node feature dimension (ESM2 + latent + stability).
            hidden_dim: Hidden feature dimension.
            n_layers: Number of GAT layers.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            edge_dim: Edge attribute dimension (PPI confidence scores).
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GAT layers with residual connections (from R4 GATModel)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, dropout=dropout,
                    edge_dim=edge_dim)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = dropout

    def forward(self, data: PyGData) -> torch.Tensor:
        """Forward pass on PPI graph.

        Args:
            data: PyG Data object with:
                - x: (num_nodes, in_dim) node features
                - edge_index: (2, num_edges) edge indices
                - edge_attr: (num_edges, edge_dim) edge weights

        Returns:
            Tensor of shape (num_nodes, hidden_dim) with node embeddings.
        """
        if self.edge_dim is not None and data.edge_attr is not None:
            assert data.edge_attr.shape[1] == self.edge_dim, \
                f"Edge dim mismatch: expected {self.edge_dim}, got {data.edge_attr.shape[1]}"

        x = self.input_proj(data.x)

        # Graph attention layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, edge_attr=data.edge_attr if self.edge_dim else None)
            x = norm(x)
            x = F.relu(x) + residual  # Residual connection
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class PathwayAwareProteinEncoder(nn.Module):
    """Pathway-aware encoder for protein features (adapted from R4).

    Cross-attention between protein features and learned pathway embeddings.
    Adapted from: r4/graph_ml/proteomics_encoder.py (PathwayAwareEncoder)

    Args:
        input_dim: Dimension of input protein features (from GNN).
        hidden_dim: Hidden dimension for attention.
        n_pathways: Number of learned pathway embeddings.
        output_dim: Output feature dimension.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 n_pathways: int = 50, output_dim: int = 256):
        """Initialize pathway-aware encoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Attention hidden dimension.
            n_pathways: Number of pathway embeddings to learn.
            output_dim: Output dimension.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pathway_embeddings = nn.Parameter(torch.randn(n_pathways, hidden_dim) * 0.01)
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (num_nodes, input_dim).
            mask: Optional padding mask (True for padded positions).

        Returns:
            Pathway-aware features (num_nodes, output_dim).
        """
        h = self.input_proj(x)  # (num_nodes, hidden)
        h_q = h.unsqueeze(1)  # (num_nodes, 1, hidden)

        # Cross-attention with learned pathway embeddings
        pw = self.pathway_embeddings.unsqueeze(0).expand(h.shape[0], -1, -1)  # (num_nodes, n_pathways, hidden)
        q = self.attn_q(h_q)  # (num_nodes, 1, hidden)
        k = self.attn_k(pw)  # (num_nodes, n_pathways, hidden)
        v = self.attn_v(pw)  # (num_nodes, n_pathways, hidden)

        # Attention
        scores = torch.bmm(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5)  # (num_nodes, 1, n_pathways)
        attn = F.softmax(scores, dim=-1)
        attended = torch.bmm(attn, v).squeeze(1)  # (num_nodes, hidden)

        return self.output_proj(attended)


class ResistancePropagator(nn.Module):
    """Protein network propagator with global attention pooling for resistance scoring.

    Uses GlobalAttention pooling (from MyeloMemory pattern) to generate per-protein
    resistance contribution scores from the PPI network.

    Args:
        gnn_dim: Dimension of GNN node embeddings.
        hidden_dim: Hidden dimension for attention gate.
        n_proteins: Number of proteins in network (for output size).
    """

    def __init__(self, gnn_dim: int = 256, hidden_dim: int = 256, n_proteins: int = None):
        """Initialize resistance propagator.

        Args:
            gnn_dim: Dimension of input node embeddings.
            hidden_dim: Hidden dimension.
            n_proteins: Number of proteins (optional, may be dynamic).
        """
        super().__init__()
        self.gnn_dim = gnn_dim
        self.n_proteins = n_proteins

        # GlobalAttention gate (from MyeloMemory pattern)
        gate_nn = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)

        # Resistance scoring head
        self.resistance_head = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        # Per-protein contribution scores
        if n_proteins is not None:
            self.per_protein_scores = nn.Linear(gnn_dim, n_proteins)
        else:
            self.per_protein_scores = None

    def forward(self, node_emb: torch.Tensor, batch_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_emb: Node embeddings (num_nodes, gnn_dim) or (batch_size, gnn_dim).
            batch_vec: Batch assignment vector (num_nodes,) for batched graphs.

        Returns:
            Tuple of:
                - graph_resistance: (batch_size, 1) or (1, 1) global resistance score
                - per_protein_resistance: (batch_size, n_proteins) or (num_nodes, n_proteins) per-protein scores
        """
        # Global pooling
        if batch_vec is not None:
            graph_emb = self.pool(node_emb, batch_vec)
        else:
            graph_emb = node_emb.mean(dim=0, keepdim=True)

        # Global resistance score
        graph_resistance = self.resistance_head(graph_emb)

        # Per-protein resistance contributions
        if self.per_protein_scores is not None:
            per_protein_resistance = self.per_protein_scores(node_emb)
        else:
            per_protein_resistance = node_emb.mean(dim=1, keepdim=True)

        return graph_resistance, per_protein_resistance


class ProteinNetworkPropagator(nn.Module):
    """Complete L3 Protein Network Propagator for ResistanceMap.

    Combines:
        1. ESM2Embedder: Protein sequence embeddings (1280-dim)
        2. PPIGraphNetwork: GAT-based PPI network processing
        3. PathwayAwareProteinEncoder: Pathway-aware feature refinement
        4. ResistancePropagator: Global + per-protein resistance scoring

    Input: Protein sequences, PPI network topology
    Output: Per-protein resistance contributions + global network resistance score

    Attributes:
        esm2_embedder: ESM-2 sequence embedder.
        ppi_gnn: Graph attention network on PPI.
        pathway_encoder: Pathway-aware feature encoder.
        propagator: Resistance scoring layer.
    """

    def __init__(self, n_proteins: int = 10000, hidden_dim: int = 256,
                 n_pathway_embeddings: int = 50, device: str = "cpu"):
        """Initialize protein network propagator.

        Args:
            n_proteins: Number of proteins in network.
            hidden_dim: Hidden dimension throughout.
            n_pathway_embeddings: Number of learned pathway embeddings.
            device: Computation device.
        """
        super().__init__()
        self.n_proteins = n_proteins
        self.hidden_dim = hidden_dim
        self.device = device

        # ESM-2 embedder (1280-dim)
        self.esm2_embedder = ESM2Embedder(device=device)
        esm2_dim = 1280
        latent_dim = 64
        stability_dim = 1
        node_feat_dim = esm2_dim + latent_dim + stability_dim  # 1345

        # PPI GNN: takes concatenated features
        self.ppi_gnn = PPIGraphNetwork(
            in_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=4,
            n_heads=8,
            dropout=0.2,
            edge_dim=1,
        )

        # Pathway-aware encoder
        self.pathway_encoder = PathwayAwareProteinEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_pathways=n_pathway_embeddings,
            output_dim=hidden_dim,
        )

        # Resistance propagator
        self.propagator = ResistancePropagator(
            gnn_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_proteins=n_proteins,
        )

        logger.info(
            f"Initialized ProteinNetworkPropagator: "
            f"esm2(1280) + latent({latent_dim}) + stability({stability_dim}) → "
            f"gnn({hidden_dim}) → pathway({hidden_dim}) → resistance"
        )

    def forward(
        self,
        sequences: List[str],
        latent_states: torch.Tensor,
        stability_scores: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through protein network propagator.

        Args:
            sequences: List of protein sequences.
            latent_states: (num_proteins, 64) latent memory states from VAE.
            stability_scores: (num_proteins,) stability scores from ODE model.
            edge_index: (2, num_edges) PPI network edges.
            edge_attr: (num_edges, 1) optional edge weights.
            batch_vec: (num_proteins,) batch assignment for multi-sample processing.

        Returns:
            Dict with keys:
                - 'node_embeddings': (num_proteins, hidden_dim) GNN node embeddings
                - 'pathway_aware': (num_proteins, hidden_dim) pathway-refined features
                - 'global_resistance': (batch_size, 1) network-level resistance score
                - 'per_protein_resistance': (batch_size, n_proteins) or (num_proteins, n_proteins)
        """
        # 1. Embed protein sequences
        esm2_emb = self.esm2_embedder(sequences)  # (num_proteins, 1280)

        # 2. Concatenate node features: ESM2 + latent + stability
        stability_scores_1d = stability_scores.unsqueeze(-1) if stability_scores.dim() == 1 else stability_scores
        node_features = torch.cat([
            esm2_emb,
            latent_states,
            stability_scores_1d,
        ], dim=-1)  # (num_proteins, 1345)

        # 3. Create PyG Data object
        ppi_data = PyGData(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        if batch_vec is not None:
            ppi_data.batch = batch_vec

        # 4. GNN forward pass
        node_embeddings = self.ppi_gnn(ppi_data)  # (num_proteins, hidden_dim)

        # 5. Pathway-aware encoding
        pathway_aware = self.pathway_encoder(node_embeddings)  # (num_proteins, hidden_dim)

        # 6. Resistance propagation
        global_resistance, per_protein_resistance = self.propagator(
            pathway_aware, batch_vec if batch_vec is not None else None
        )

        return {
            "node_embeddings": node_embeddings,
            "pathway_aware": pathway_aware,
            "global_resistance": global_resistance,
            "per_protein_resistance": per_protein_resistance,
        }
