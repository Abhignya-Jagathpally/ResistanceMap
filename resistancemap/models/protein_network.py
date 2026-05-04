"""L3 Protein Network Propagator for ResistanceMap.

Combines ESM-2 protein embeddings with GNN-based propagation on PPI networks
and pathway-aware encoding to compute per-protein resistance contributions.

Architecture:
    1. ESM2Embedder: Loads facebook/esm2_t33_650M_UR50D, computes 1280-dim embeddings
    2. PPIGraphNetwork: Adapted from R4 GATModel (r4/graph_ml/gnn_models.py)
       with residual connections and LayerNorm for protein feature aggregation
       - Enhanced with DropEdge, PairNorm, JumpingKnowledge for over-smoothing
       - GraphMASK for causal edge attribution
    3. PathwayAwareEncoder: Adapted from R4 proteomics_encoder.py for cross-attention
       between protein features and pathway embeddings
    4. ResistancePropagator: Uses GlobalAttention pooling (from MyeloMemory pattern)
       to generate per-protein resistance contribution scores
    5. EvidentialClassificationHead: Dirichlet-based uncertainty quantification

Components sourced from:
    - R4 GATModel: r4/graph_ml/gnn_models.py (residual + LayerNorm + edge_dim support)
    - R4 PathwayAwareEncoder: r4/graph_ml/proteomics_encoder.py
    - MyeloMemory GlobalAttention pattern: myelomemory/models/gnn.py
"""

from __future__ import annotations

import logging

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


class DropEdge(nn.Module):
    """Randomly drops edges during training for regularization.

    Attributes:
        drop_rate: Probability of dropping each edge.
    """

    def __init__(self, drop_rate: float = 0.2):
        """Initialize DropEdge.

        Args:
            drop_rate: Dropout probability for edges (0.0 to 1.0).
        """
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Drop edges randomly during training.

        Args:
            edge_index: (2, num_edges) edge index tensor.

        Returns:
            Possibly reduced edge_index tensor.
        """
        if not self.training or self.drop_rate == 0.0:
            return edge_index

        n_edges = edge_index.shape[1]
        mask = torch.rand(n_edges, device=edge_index.device) > self.drop_rate
        return edge_index[:, mask]


class PairNorm(nn.Module):
    """PairNorm normalization (Zhao & Akoglu 2020).

    Maintains pairwise distances between node representations by centering
    and scaling to unit variance.

    Attributes:
        scale: Scaling factor for final variance.
    """

    def __init__(self, scale: float = 1.0):
        """Initialize PairNorm.

        Args:
            scale: Scaling factor for variance.
        """
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PairNorm to features.

        Args:
            x: (num_nodes, dim) feature tensor.

        Returns:
            Normalized features (num_nodes, dim).
        """
        # Center features
        x_centered = x - x.mean(dim=0, keepdim=True)

        # Scale to unit variance times scale factor
        std = x_centered.std(dim=0, keepdim=True) + 1e-8
        x_normalized = x_centered / std * self.scale

        return x_normalized


class ESM2Bottleneck(nn.Module):
    """Bottleneck to reduce ESM-2 embeddings from 1280-dim to 256-dim.

    Prevents high-dimensional ESM-2 embeddings from drowning out other modalities
    in concatenation.
    """

    def __init__(self, input_dim: int = 1280, output_dim: int = 256):
        """Initialize ESM2Bottleneck.

        Args:
            input_dim: Input dimension (1280 for ESM-2).
            output_dim: Output dimension (256).
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce dimensionality of ESM-2 embeddings.

        Args:
            x: (batch_size, 1280) ESM-2 embeddings.

        Returns:
            (batch_size, 256) reduced embeddings.
        """
        x = self.linear(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


class DrugConditionedAttention(nn.Module):
    """Modulates GAT attention scores based on drug identity.

    Learns a per-drug attention scaling that can increase or decrease
    the influence of drug-target edges.

    Attributes:
        drug_embedding_dim: Dimension of drug embeddings.
        hidden_dim: Hidden dimension for modulation network.
    """

    def __init__(self, drug_embedding_dim: int = 128, hidden_dim: int = 64):
        """Initialize DrugConditionedAttention.

        Args:
            drug_embedding_dim: Dimension of drug embeddings.
            hidden_dim: Hidden dimension for modulation MLP.
        """
        super().__init__()
        self.drug_embedding_dim = drug_embedding_dim
        self.gate = nn.Sequential(
            nn.Linear(drug_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, attention_scores: torch.Tensor, drug_embedding: torch.Tensor) -> torch.Tensor:
        """Modulate attention scores by drug embedding.

        Args:
            attention_scores: (num_edges, num_heads) attention scores from GAT.
            drug_embedding: (drug_embedding_dim,) drug embedding vector.

        Returns:
            Modulated attention scores (num_edges, num_heads).
        """
        modulation = self.gate(drug_embedding.unsqueeze(0))  # (1, 1)
        return attention_scores * modulation


class JumpingKnowledge(nn.Module):
    """Aggregates node representations from multiple GNN layers.

    Supports concatenation, max pooling, or LSTM-based aggregation
    across layers to combat over-smoothing.

    Attributes:
        mode: Aggregation mode ("cat", "max", "lstm").
        hidden_dim: Hidden dimension for projection and LSTM.
    """

    def __init__(self, mode: str = "cat", hidden_dim: int = 256, num_layers: int = 4):
        """Initialize JumpingKnowledge.

        Args:
            mode: Aggregation mode ("cat", "max", "lstm").
            hidden_dim: Hidden dimension.
            num_layers: Number of layers to aggregate.
        """
        super().__init__()
        assert mode in ["cat", "max", "lstm"], f"Invalid mode: {mode}"
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if mode == "cat":
            # Project concatenated features back to hidden_dim
            self.proj = nn.Linear(hidden_dim * num_layers, hidden_dim)
        elif mode == "lstm":
            # LSTM to aggregate across layers
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate representations from multiple layers.

        Args:
            layer_outputs: List of (num_nodes, hidden_dim) tensors, one per layer.

        Returns:
            Aggregated (num_nodes, hidden_dim) tensor.
        """
        if self.mode == "cat":
            # Concatenate across layers and project
            x = torch.cat(layer_outputs, dim=-1)  # (num_nodes, hidden_dim * num_layers)
            return self.proj(x)
        elif self.mode == "max":
            # Element-wise max across layers
            return torch.stack(layer_outputs, dim=0).max(dim=0).values
        elif self.mode == "lstm":
            # LSTM aggregation
            x = torch.stack(layer_outputs, dim=1)  # (num_nodes, num_layers, hidden_dim)
            _, (h_n, _) = self.lstm(x)  # h_n: (1, num_nodes, hidden_dim)
            return h_n.squeeze(0)  # (num_nodes, hidden_dim)


class GraphMASKLayer(nn.Module):
    """Learns a differentiable binary mask over edges for causal attribution.

    The mask is learned via a gate network that takes edge features
    and outputs per-edge mask values in [0, 1].

    Attributes:
        edge_dim: Dimension of edge features.
        hidden_dim: Hidden dimension for gate network.
    """

    def __init__(self, edge_dim: int = 1, hidden_dim: int = 64):
        """Initialize GraphMASKLayer.

        Args:
            edge_dim: Dimension of edge features.
            hidden_dim: Hidden dimension for gate MLP.
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._mask_cache = None

    def forward(self, edge_attention: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """Modulate edge attention with learned mask.

        Args:
            edge_attention: (num_edges,) or (num_edges, num_heads) attention scores.
            edge_attr: (num_edges, edge_dim) edge attributes. If None, uses learnable default.

        Returns:
            Masked attention (same shape as input).
        """
        if edge_attr is None:
            # Use uniform edge features if not provided
            edge_attr = torch.ones(edge_attention.shape[0], self.edge_dim,
                                   device=edge_attention.device)

        mask = self.gate(edge_attr)  # (num_edges, 1)
        self._mask_cache = mask.squeeze(-1)

        return edge_attention * mask

    def get_mask(self) -> torch.Tensor:
        """Get the learned edge mask for interpretability.

        Returns:
            (num_edges,) mask values in [0, 1].
        """
        if self._mask_cache is None:
            raise RuntimeError("get_mask() called before forward()")
        return self._mask_cache


class EvidentialClassificationHead(nn.Module):
    """Dirichlet-based evidential classification head (Sensoy et al. 2018).

    Outputs concentration parameters for a Dirichlet distribution,
    enabling uncertainty quantification beyond softmax predictions.

    Attributes:
        input_dim: Input feature dimension.
        n_classes: Number of output classes.
    """

    def __init__(self, input_dim: int = 256, n_classes: int = 3):
        """Initialize EvidentialClassificationHead.

        Args:
            input_dim: Input feature dimension.
            n_classes: Number of classes (default 3: sensitive/intermediate/resistant).
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        # MLP to logits
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Dirichlet concentration parameters.

        Args:
            x: (batch_size, input_dim) input features.

        Returns:
            (batch_size, n_classes) concentration parameters alpha.
        """
        logits = self.mlp(x)
        # Ensure alpha > 0 via softplus + 1
        alpha = F.softplus(logits) + 1.0
        return alpha

    def uncertainty(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute epistemic uncertainty from Dirichlet parameters.

        Epistemic uncertainty: K / sum(alpha) where K = n_classes

        Args:
            alpha: (batch_size, n_classes) concentration parameters.

        Returns:
            (batch_size, 1) uncertainty scores.
        """
        K = self.n_classes
        total_alpha = alpha.sum(dim=-1, keepdim=True)
        uncertainty = K / total_alpha
        return uncertainty


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
        self.embedding_cache: dict[str, torch.Tensor] = {}

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

    def embed_proteins(self, sequences: list[str]) -> torch.Tensor:
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

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """Wrapper for embed_proteins."""
        return self.embed_proteins(sequences)


class PPIGraphNetwork(nn.Module):
    """Graph Attention Network on PPI networks (adapted from R4 GATModel).

    Architecture:
        - Input projection: node_features (variable-dim) → hidden_dim
        - N layers of GATConv with residual connections and LayerNorm
        - Optional DropEdge regularization
        - Optional PairNorm for stable training
        - Optional JumpingKnowledge to combat over-smoothing
        - Optional GraphMASK for causal edge attribution
        - Output projection: hidden_dim → hidden_dim

    Node features are concatenated from:
        - ESM-2 embedding (1280 or reduced to 256)
        - Latent state (64)
        - Stability score (1)
        - Optional phosphoproteomics (phospho_dim)
        Total: variable dimensions based on configuration

    Adapted from: r4/graph_ml/gnn_models.py (GATModel with residual connections)
    """

    def __init__(self, in_dim: int = 321, hidden_dim: int = 256,
                 n_layers: int = 4, n_heads: int = 8, dropout: float = 0.2,
                 edge_dim: int | None = 1, dropedge_rate: float = 0.0,
                 use_pairnorm: bool = False, use_jumping_knowledge: bool = False,
                 use_graphmask: bool = False):
        """Initialize PPIGraphNetwork.

        Args:
            in_dim: Input node feature dimension (ESM2 + latent + stability + optional phospho).
            hidden_dim: Hidden feature dimension.
            n_layers: Number of GAT layers.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            edge_dim: Edge attribute dimension (PPI confidence scores).
            dropedge_rate: Edge dropout rate (0.0 = disabled).
            use_pairnorm: Whether to apply PairNorm after each conv layer.
            use_jumping_knowledge: Whether to aggregate representations across layers.
            use_graphmask: Whether to apply GraphMASK for causal edge attribution.
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.n_layers = n_layers
        self.use_pairnorm = use_pairnorm
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_graphmask = use_graphmask

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Optional DropEdge
        self.drop_edge = DropEdge(dropedge_rate) if dropedge_rate > 0.0 else None

        # GAT layers with residual connections (from R4 GATModel)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, dropout=dropout,
                    edge_dim=edge_dim)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Optional PairNorm
        if self.use_pairnorm:
            self.pairnorms = nn.ModuleList([PairNorm(scale=1.0) for _ in range(n_layers)])
        else:
            self.pairnorms = None

        # Optional GraphMASK
        if self.use_graphmask:
            self.graphmask = GraphMASKLayer(edge_dim=edge_dim if edge_dim else 1)
        else:
            self.graphmask = None

        # Optional JumpingKnowledge
        if self.use_jumping_knowledge:
            self.jumping_knowledge = JumpingKnowledge(mode="cat", hidden_dim=hidden_dim,
                                                       num_layers=n_layers)
        else:
            self.jumping_knowledge = None

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

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        layer_outputs = [] if self.use_jumping_knowledge else None

        # Graph attention layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Optional DropEdge
            if self.drop_edge is not None:
                edge_index_curr = self.drop_edge(edge_index)
            else:
                edge_index_curr = edge_index

            residual = x
            _edge_attr = edge_attr if (self.edge_dim is not None and self.edge_dim > 0 and edge_attr is not None) else None
            x = conv(x, edge_index_curr, edge_attr=_edge_attr)
            x = norm(x)

            # Optional PairNorm before ReLU
            if self.use_pairnorm:
                x = self.pairnorms[i](x)

            x = F.relu(x) + residual  # Residual connection
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Collect for JumpingKnowledge
            if self.use_jumping_knowledge:
                layer_outputs.append(x)

        # Optional JumpingKnowledge aggregation
        if self.use_jumping_knowledge:
            x = self.jumping_knowledge(layer_outputs)

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
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

        # Apply mask if provided
        if mask is not None:
            attended = attended * mask.unsqueeze(-1)  # Zero out masked positions

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

    def forward(self, node_emb: torch.Tensor, batch_vec: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
            per_protein_resistance = node_emb.mean(dim=-1, keepdim=True)  # average features per node

        return graph_resistance, per_protein_resistance


class ProteinNetworkPropagator(nn.Module):
    """Complete L3 Protein Network Propagator for ResistanceMap.

    Combines:
        1. ESM2Embedder: Protein sequence embeddings (1280-dim)
        2. ESM2Bottleneck: Reduces 1280-dim to 256-dim to prevent drowning
        3. PPIGraphNetwork: GAT-based PPI network processing with:
           - DropEdge for regularization
           - PairNorm for over-smoothing prevention
           - JumpingKnowledge for multi-layer aggregation
           - GraphMASK for causal edge attribution
        4. PathwayAwareProteinEncoder: Pathway-aware feature refinement
        5. ResistancePropagator: Global + per-protein resistance scoring
        6. EvidentialClassificationHead: Optional Dirichlet-based uncertainty

    Input: Protein sequences, PPI network topology, optional drug embedding/phosphoproteomics
    Output: Per-protein resistance contributions + global network resistance score

    Attributes:
        esm2_embedder: ESM-2 sequence embedder.
        esm2_bottleneck: Dimensionality reduction for ESM-2.
        ppi_gnn: Graph attention network on PPI.
        pathway_encoder: Pathway-aware feature encoder.
        propagator: Resistance scoring layer.
        evidential_head: Optional Dirichlet classification head.
    """

    def __init__(self, n_proteins: int = 10000, hidden_dim: int = 256,
                 n_pathway_embeddings: int = 50, device: str = "cpu",
                 use_evidential_head: bool = False, dropedge_rate: float = 0.0,
                 use_pairnorm: bool = False, use_jumping_knowledge: bool = False,
                 use_graphmask: bool = False, phospho_dim: int = 0):
        """Initialize protein network propagator.

        Args:
            n_proteins: Number of proteins in network.
            hidden_dim: Hidden dimension throughout.
            n_pathway_embeddings: Number of learned pathway embeddings.
            device: Computation device.
            use_evidential_head: Whether to add EvidentialClassificationHead.
            dropedge_rate: Edge dropout rate for GNN (0.0 = disabled).
            use_pairnorm: Whether to apply PairNorm in GNN.
            use_jumping_knowledge: Whether to aggregate across GNN layers.
            use_graphmask: Whether to use GraphMASK for causal attribution.
            phospho_dim: Phosphoproteomics feature dimension (0 = disabled).
        """
        super().__init__()
        self.n_proteins = n_proteins
        self.hidden_dim = hidden_dim
        self.device = device
        self.phospho_dim = phospho_dim

        # ESM-2 embedder (1280-dim)
        self.esm2_embedder = ESM2Embedder(device=device)
        esm2_dim = 1280
        esm2_bottleneck_dim = 256

        # ESM-2 bottleneck to reduce dimensionality
        self.esm2_bottleneck = ESM2Bottleneck(input_dim=esm2_dim, output_dim=esm2_bottleneck_dim)

        latent_dim = 64
        stability_dim = 1
        node_feat_dim = esm2_bottleneck_dim + latent_dim + stability_dim + phospho_dim  # 321 + phospho_dim

        # PPI GNN: takes concatenated features
        self.ppi_gnn = PPIGraphNetwork(
            in_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=4,
            n_heads=8,
            dropout=0.2,
            edge_dim=1,
            dropedge_rate=dropedge_rate,
            use_pairnorm=use_pairnorm,
            use_jumping_knowledge=use_jumping_knowledge,
            use_graphmask=use_graphmask,
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

        # Optional evidential classification head
        self.use_evidential_head = use_evidential_head
        if use_evidential_head:
            self.evidential_head = EvidentialClassificationHead(
                input_dim=hidden_dim, n_classes=3
            )
        else:
            self.evidential_head = None

        logger.info(
            f"Initialized ProteinNetworkPropagator: "
            f"esm2(1280) → bottleneck(256) + latent({latent_dim}) + stability({stability_dim})" +
            (f" + phospho({phospho_dim})" if phospho_dim > 0 else "") +
            f" → gnn({hidden_dim}) → pathway({hidden_dim}) → resistance"
        )
        if use_evidential_head:
            logger.info("  with EvidentialClassificationHead (Dirichlet-based)")
        if dropedge_rate > 0.0:
            logger.info(f"  with DropEdge (rate={dropedge_rate})")
        if use_pairnorm:
            logger.info("  with PairNorm")
        if use_jumping_knowledge:
            logger.info("  with JumpingKnowledge")
        if use_graphmask:
            logger.info("  with GraphMASK (causal attribution)")

    def forward(
        self,
        sequences: list[str],
        latent_states: torch.Tensor,
        stability_scores: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch_vec: torch.Tensor | None = None,
        drug_embedding: torch.Tensor | None = None,
        phosphoproteomics: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through protein network propagator.

        Args:
            sequences: List of protein sequences.
            latent_states: (num_proteins, 64) latent memory states from VAE.
            stability_scores: (num_proteins,) stability scores from ODE model.
            edge_index: (2, num_edges) PPI network edges.
            edge_attr: (num_edges, 1) optional edge weights.
            batch_vec: (num_proteins,) batch assignment for multi-sample processing.
            drug_embedding: (drug_embedding_dim,) optional drug embedding vector.
            phosphoproteomics: (num_proteins, phospho_dim) optional phosphoproteomics data.

        Returns:
            Dict with keys:
                - 'node_embeddings': (num_proteins, hidden_dim) GNN node embeddings
                - 'pathway_aware': (num_proteins, hidden_dim) pathway-refined features
                - 'global_resistance': (batch_size, 1) network-level resistance score
                - 'per_protein_resistance': (batch_size, n_proteins) or (num_proteins, n_proteins)
                - 'evidential_alpha': (batch_size, 3) if use_evidential_head=True
                - 'uncertainty': (batch_size, 1) epistemic uncertainty if use_evidential_head=True
        """
        # Validate input shapes
        if latent_states.dim() != 2:
            raise ValueError(f"Expected 2D tensor (B, D), got shape {latent_states.shape}")

        # 1. Embed protein sequences
        esm2_emb = self.esm2_embedder(sequences)  # (num_proteins, 1280)

        # 2. Apply ESM-2 bottleneck
        esm2_reduced = self.esm2_bottleneck(esm2_emb)  # (num_proteins, 256)

        # 3. Concatenate node features: ESM2_reduced + latent + stability + optional phosphoproteomics
        stability_scores_1d = stability_scores.unsqueeze(-1) if stability_scores.dim() == 1 else stability_scores

        node_features_list = [esm2_reduced, latent_states, stability_scores_1d]

        if phosphoproteomics is not None:
            assert phosphoproteomics.shape[0] == len(sequences), \
                f"Phosphoproteomics batch size {phosphoproteomics.shape[0]} != sequences {len(sequences)}"
            node_features_list.append(phosphoproteomics)

        node_features = torch.cat(node_features_list, dim=-1)

        # 4. Create PyG Data object
        ppi_data = PyGData(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        if batch_vec is not None:
            ppi_data.batch = batch_vec

        # 5. GNN forward pass
        node_embeddings = self.ppi_gnn(ppi_data)  # (num_proteins, hidden_dim)

        # 6. Pathway-aware encoding
        pathway_aware = self.pathway_encoder(node_embeddings)  # (num_proteins, hidden_dim)

        # 7. Resistance propagation
        global_resistance, per_protein_resistance = self.propagator(
            pathway_aware, batch_vec if batch_vec is not None else None
        )

        result = {
            "node_embeddings": node_embeddings,
            "pathway_aware": pathway_aware,
            "global_resistance": global_resistance,
            "per_protein_resistance": per_protein_resistance,
        }

        # 8. Optional evidential classification head
        if self.use_evidential_head and self.evidential_head is not None:
            alpha = self.evidential_head(pathway_aware)
            uncertainty = self.evidential_head.uncertainty(alpha)
            result["evidential_alpha"] = alpha
            result["uncertainty"] = uncertainty

        return result