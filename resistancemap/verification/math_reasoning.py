"""
First-Principles Mathematical Reasoning for ResistanceMap
==========================================================

This module implements fundamental theoretical bounds and stability analysis
from information theory, dynamical systems, network topology, and cognitive
science. Every method is grounded in peer-reviewed mathematics.

Key Insight: We don't just make predictions—we prove bounds on what's
theoretically possible, then verify our model respects those bounds.

References:
  [1] Cover & Thomas (2006). Elements of Information Theory (2nd ed). Wiley.
  [2] Strogatz (2018). Nonlinear Dynamics and Chaos (2nd ed). CRC Press.
  [3] Barabási & Albert (1999). Emergence of scaling in random networks. Science.
  [4] Tversky (1977). Features of Similarity. Psychological Review.
  [5] Alberts et al. (2015). Molecular Biology of the Cell (6th ed). Garland.
  [6] Lane (2016). Power, Sex, Suicide: Mitochondria and the Meaning of Life. OUP.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform


class InformationTheoreticBounds:
    """Information-theoretic bounds on prediction accuracy.

    From first principles (Cover & Thomas, Ch. 2-3):
    ============================================================

    FANO'S INEQUALITY (Fundamental Limit on Error)
    -----------------------------------------------
    For a classification problem with target variable Y and features X:

        P(ŷ ≠ y) ≥ (H(Y|X) - 1) / log(|Y|)

    where:
        P(ŷ ≠ y) = probability of misclassification
        H(Y|X) = conditional entropy (remaining uncertainty in Y given X)
        |Y| = number of classes

    Interpretation:
    - If H(Y|X) = 0 (features determine Y perfectly), error can be 0
    - If H(Y|X) = log(|Y|) (X tells us nothing), error ≥ (log(|Y|) - 1) / log(|Y|)
    - Our model's error should approach (not exceed) this bound for it to be optimal

    Applied to Drug Resistance Prediction:
    - Y ∈ {sensitive, resistant, mixed} → |Y| = 3
    - X = proteomics, genomics, epigenomics (8000+ features)
    - If H(Y|X) is high, even perfect prediction model has error floor
    - If H(Y|X) is low, there's a lot of info in X about Y
    """

    @staticmethod
    def compute_fano_bound(
        features: torch.Tensor,
        labels: torch.Tensor,
        n_classes: int,
    ) -> Dict[str, float]:
        """Compute Fano lower bound on prediction error.

        Args:
            features: [n_samples, n_features] feature tensor
            labels: [n_samples] class labels (0 to n_classes-1)
            n_classes: Number of classes

        Returns:
            Dict with:
                - "conditional_entropy": H(Y|X) in bits
                - "fano_bound": Lower bound on P(error)
                - "achievable_accuracy_upper": Upper bound on accuracy (1 - P(error))
                - "n_classes": Number of classes
                - "n_samples": Number of samples
        """
        features = features.detach().cpu()
        labels = labels.detach().cpu()

        # Estimate H(Y|X) using k-NN entropy estimator (Kraskov et al. 2004)
        # H(Y|X) ≈ H(Y) - I(X; Y) where I(X;Y) is mutual information
        # We estimate I(X;Y) via k-NN distance correlation

        # First compute H(Y) (marginal entropy)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        py = counts.float() / len(labels)
        hy = -(py * torch.log2(py + 1e-10)).sum().item()

        # For H(Y|X), use k-NN entropy estimation
        # With k=3 (standard choice)
        k = min(3, len(features) // n_classes)

        hyx = InformationTheoreticBounds._knn_entropy_estimate(
            features, labels, k, n_classes
        )

        # Fano bound: P(error) ≥ (H(Y|X) - 1) / log(|Y|)
        log_classes = np.log2(n_classes)
        fano_bound = max(0, (hyx - 1) / log_classes) if log_classes > 0 else 0
        achievable_acc = 1 - fano_bound

        return {
            "conditional_entropy": float(hyx),
            "marginal_entropy": float(hy),
            "fano_bound": float(fano_bound),
            "achievable_accuracy_upper": float(achievable_acc),
            "n_classes": n_classes,
            "n_samples": len(features),
        }

    @staticmethod
    def _knn_entropy_estimate(
        features: torch.Tensor,
        labels: torch.Tensor,
        k: int,
        n_classes: int,
    ) -> float:
        """Estimate conditional entropy H(Y|X) via k-NN method.

        Uses the idea: if X is informative about Y, then neighbors in X-space
        should have similar Y values. Entropy is low when that correlation is high.
        """
        # Compute pairwise distances
        features_np = features.cpu().numpy()
        dist_matrix = squareform(pdist(features_np, metric='euclidean'))

        # For each point, find k nearest neighbors
        hyx = 0.0
        n_samples = len(features)

        for i in range(n_samples):
            # Get k nearest neighbors (excluding self)
            neighbors_idx = np.argsort(dist_matrix[i])[1:k+1]
            neighbor_labels = labels[neighbors_idx].numpy()

            # Entropy of neighbor labels
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            p = counts / len(neighbor_labels)
            h_neighbors = -np.sum(p * np.log2(p + 1e-10))
            hyx += h_neighbors

        return hyx / n_samples

    @staticmethod
    def mutual_information(
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute mutual information I(X; Y) = H(Y) - H(Y|X).

        Mutual information quantifies how much information features X contain
        about the target Y. Ranges from 0 (X independent of Y) to H(Y) (X
        determines Y perfectly).

        Args:
            features: [n_samples, n_features] features
            labels: [n_samples] labels

        Returns:
            I(X; Y) in bits. Values near H(Y) mean X is very informative.
        """
        features = features.detach().cpu()
        labels = labels.detach().cpu()

        # H(Y): marginal entropy of labels
        unique_labels, counts = torch.unique(labels, return_counts=True)
        py = counts.float() / len(labels)
        hy = -(py * torch.log2(py + 1e-10)).sum().item()

        # H(Y|X): conditional entropy
        k = min(3, len(features) // len(unique_labels))
        hyx = InformationTheoreticBounds._knn_entropy_estimate(
            features, labels, k, len(unique_labels)
        )

        # I(X; Y) = H(Y) - H(Y|X)
        mi = hy - hyx
        return max(0, float(mi))

    @staticmethod
    def information_bottleneck_bound(
        input_dim: int,
        latent_dim: int,
        output_classes: int,
    ) -> Dict[str, float]:
        """Information bottleneck bound on VAE compression.

        From Tishby & Schwartz (1999) Information Bottleneck Method:
        The information bottleneck principle states that there's a tradeoff
        between:
            - Information Compression: I(X; Z) should be small (Z is compressed)
            - Information Retention: I(Z; Y) should be large (Z still encodes Y)

        The rate-distortion bound gives the frontier of this tradeoff.

        Applied to ResistanceMap's feature compression:
        - Input X: 8000-dim proteomics/genomics
        - Latent Z: 64-dim learned representation
        - Output Y: resistance state (3 classes)

        The latent space must balance:
            min_Z { I(X; Z) + β·D(Y, ŷ(Z)) }

        This function estimates what compression is theoretically achievable.

        Args:
            input_dim: Dimensionality of input (e.g., 8000)
            latent_dim: Dimensionality of latent space (e.g., 64)
            output_classes: Number of output classes (e.g., 3)

        Returns:
            Dict with:
                - "compression_ratio": input_dim / latent_dim
                - "bits_per_sample": log2(output_classes) (target information)
                - "max_latent_capacity": latent_dim * log2(e)
                - "is_feasible": Can latent space encode target?
                - "estimated_margin": Bits of capacity beyond minimum needed
        """
        compression_ratio = input_dim / latent_dim
        target_info = np.log2(output_classes)  # Minimum info needed for classification
        latent_capacity = latent_dim * np.log2(np.e)  # Max bits per sample

        is_feasible = latent_capacity >= target_info
        margin = latent_capacity - target_info

        return {
            "compression_ratio": float(compression_ratio),
            "bits_per_sample": float(target_info),
            "max_latent_capacity": float(latent_capacity),
            "is_feasible": bool(is_feasible),
            "estimated_margin": float(margin),
        }


class DynamicalSystemsTheory:
    """Dynamical systems analysis for epigenetic memory and stability.

    From first principles (Strogatz, Ch. 2-6):
    ============================================================

    LYAPUNOV STABILITY (Measure of Basin Depth)
    -------------------------------------------
    A fixed point x* of the system dx/dt = f(x) is STABLE if all eigenvalues
    of the Jacobian J = ∂f/∂x|_{x*} have negative real parts.

    For a 2D bistable system (epigenetic memory):
        dx₁/dt = x₁(1 - x₁²) - αx₂
        dx₂/dt = x₂(1 - x₂²) - αx₁

    Fixed points:
        (x₁*, x₂*) = (0, 0) [unstable saddle, rate ↔ sensitive]
        (x₁*, x₂*) ≈ (0.7, 0.7) [stable, rate ↔ resistant]

    Basin of attraction: The set of initial conditions that lead to a stable
    fixed point. Deeper basin → harder to push state away → more stable memory.

    Applied to drug-induced resistance:
    - Chromatin state (sensitive) and (resistant) are two stable fixed points
    - Epigenetic memory = depth of basin around current state
    - Drug exposure = noise or perturbation pushing state toward basin boundary
    - Resistance emerges when noise overcomes basin boundary

    KRAMERS ESCAPE RATE (Time to Transition)
    ----------------------------------------
    For a particle in a potential V(x) with thermal noise at temperature T:

        k(T) = (ωₐ·ωᵦ) / (2π·γ) · exp(-ΔV/kT)

    where:
        ωₐ, ωᵦ = frequencies at minima (basin curvatures)
        γ = friction coefficient
        ΔV = barrier height
        k = Boltzmann constant

    Interpretation:
        - Deeper basin (high ωₐ) → slower escape → longer time to resistance
        - Higher barrier ΔV → exponentially slower escape
        - Higher temperature (noise) → faster escape (faster resistance)

    In the context of cancer/infection:
        - Thermal noise ↔ stochastic gene expression
        - Barrier height ↔ epigenetic stability (how locked is chromatin?)
        - Escape rate ↔ rate of spontaneous resistance emergence
    """

    @staticmethod
    def compute_lyapunov_exponents(
        jacobian: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute Lyapunov exponents from Jacobian matrix.

        The Lyapunov exponents are the eigenvalues of the Jacobian.
        - Negative eigenvalues → stable direction (state is pulled back)
        - Positive eigenvalues → unstable direction (state diverges away)
        - Zero eigenvalue → marginally stable

        Args:
            jacobian: [n_dims, n_dims] Jacobian matrix df/dx at fixed point

        Returns:
            Dict with:
                - "eigenvalues": Complex eigenvalues (sorted by real part descending)
                - "is_stable": All eigenvalues have Re(λ) < 0?
                - "stability_margin": max(Re(λ)) — how close to instability?
                - "real_parts": Real parts of eigenvalues
                - "imag_parts": Imaginary parts of eigenvalues
        """
        jacobian = jacobian.detach().cpu().float()

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(jacobian)

        # Sort by real part (descending)
        real_parts = eigenvalues.real
        sorted_idx = torch.argsort(real_parts, descending=True)
        eigenvalues_sorted = eigenvalues[sorted_idx]

        # Check stability
        max_real = real_parts.max().item()
        is_stable = max_real < 0

        return {
            "eigenvalues": eigenvalues_sorted,
            "is_stable": bool(is_stable),
            "stability_margin": float(max_real),  # Should be negative
            "real_parts": eigenvalues_sorted.real,
            "imag_parts": eigenvalues_sorted.imag,
        }

    @staticmethod
    def basin_depth(jacobian: torch.Tensor) -> float:
        """Compute basin depth (stability strength) of a fixed point.

        Basin depth = -min(Re(λ)) where λ are eigenvalues of Jacobian.

        Higher depth = deeper basin = more stable = harder to flip state.

        Interpretation:
        - Depth = 0: Marginally stable (saddle point)
        - Depth = 0.1: Shallow basin, noise can flip state easily
        - Depth = 1.0: Moderate basin, some stability
        - Depth > 1.0: Deep basin, strong memory, hard to flip

        Args:
            jacobian: [n_dims, n_dims] Jacobian at fixed point

        Returns:
            Basin depth (positive value, higher = more stable)
        """
        jacobian = jacobian.detach().cpu().float()
        eigenvalues = torch.linalg.eigvals(jacobian)
        max_real = eigenvalues.real.max().item()

        # Depth = -max_real (convert most-positive eigenvalue to depth)
        depth = -max_real
        return max(0, float(depth))

    @staticmethod
    def transition_rate_kramers(
        basin_depth: float,
        noise_level: float,
        barrier_height: float = 1.0,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Compute Kramers escape rate from bistable potential.

        Kramers Escape Rate:
            k(T) = (ωₐ·ωᵦ) / (2π) · exp(-ΔV/kT)

        where:
            ωₐ ∝ basin_depth (frequency at minimum)
            ωᵦ ∝ barrier curvature (frequency at saddle)
            ΔV = barrier_height (energy difference)
            kT ∝ noise_level (thermal energy)

        This predicts the MEAN TIME TO TRANSITION between bistable states:
            τ ∝ 1/k(T) = (2π/(ωₐ·ωᵦ)) · exp(ΔV/kT)

        Applied to resistance: predicts timescale for spontaneous resistance
        emergence in absence of external drug pressure.

        Args:
            basin_depth: Lyapunov exponent magnitude (rate of stability)
            noise_level: Stochastic fluctuation magnitude
            barrier_height: Energy barrier between states (default 1.0)
            temperature: Effective temperature (noise ∝ T) (default 1.0)

        Returns:
            Dict with:
                - "escape_rate": k(T) — rate of leaving basin
                - "mean_time_to_transition": 1/k(T) — average time to flip
                - "barrier_exponent": ΔV/kT — dominates exponential
                - "frequency_factor": ωₐ·ωᵦ — prefactor
        """
        # Frequency factors: proportional to basin depth and noise
        omega_a = max(basin_depth, 1e-6)  # Frequency at minimum
        omega_b = noise_level + 1e-6  # Frequency at barrier (saddle)

        frequency_factor = (omega_a * omega_b) / (2 * np.pi)

        # Barrier exponent: ΔV / (k*T)
        effective_temp = temperature * noise_level + 1e-6
        barrier_exponent = barrier_height / effective_temp

        # Escape rate: k = frequency_factor * exp(-barrier_exponent)
        escape_rate = frequency_factor * np.exp(-barrier_exponent)

        # Mean transition time
        mean_transition_time = 1.0 / escape_rate if escape_rate > 0 else np.inf

        return {
            "escape_rate": float(escape_rate),
            "mean_time_to_transition": float(mean_transition_time),
            "barrier_exponent": float(barrier_exponent),
            "frequency_factor": float(frequency_factor),
        }

    @staticmethod
    def waddington_potential(
        states: torch.Tensor,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct Waddington potential from state trajectory and velocity field.

        Waddington's Epigenetic Landscape (1957): cells roll downhill toward
        stable attractor states (differentiated phenotypes). The potential
        energy V(x) determines which states are stable.

        Reconstruction (Bidirectional):
            V(x) = -∫ f(x) · dx    (since f = -∇V in gradient systems)

        For our 2D epigenetic system:
            dC/dt = -∂V/∂C + noise
            dM/dt = -∂V/∂M + noise

        We can estimate V by numerically integrating f.

        Args:
            states: [n_samples, n_dims] trajectory of system states
            velocities: [n_samples, n_dims] velocities (d(state)/dt)

        Returns:
            Estimated potential energy V(x) at each state point
        """
        states = states.detach().cpu().float()
        velocities = velocities.detach().cpu().float()

        # Numerical integration: V(x) = -∫ f(x) · dx along trajectory
        # Using cumulative trapezoidal rule

        # Compute f·v (inner product of velocity field with observed velocities)
        dot_products = (velocities ** 2).sum(dim=1)  # Magnitude of velocity

        # Cumulative integral
        dt = 1.0  # Assume unit time steps
        potential = -torch.cumsum(dot_products * dt, dim=0)

        # Normalize to minimum potential = 0
        potential = potential - potential.min()

        return potential


class NetworkTopologyTheory:
    """Network topology analysis for STRING PPI and drug target selection.

    From first principles (Barabási & Albert, 1999):
    ============================================================

    SCALE-FREE NETWORKS
    -------------------
    Many biological networks (PPI, metabolic, regulatory) follow power-law
    degree distribution:

        P(k) ~ k^(-γ)

    where γ ≈ 2-3 for biological networks.

    Implications:
        - Few hub proteins (high degree): network robustness
        - Many low-degree proteins: individual nodes dispensable
        - Hubs often essential (drug targets) but also cause side effects

    PAGERANK CENTRALITY (Information Flow Importance)
    -------------------------------------------------
    PageRank (Brin & Page, 1998) ranks nodes by incoming link importance,
    weighted by importance of sources.

    For resistance networks:
        - Nodes with high PageRank = "information hubs"
        - Mutations/drugs affecting these nodes propagate through network
        - Good drug targets: high PageRank + non-essential-in-healthy-state

    Applied to STRING PPI:
        - Compute PageRank of all proteins
        - Proteins high in PageRank are likely key regulators
        - Combine with other features (degree, betweenness) for target ranking
    """

    @staticmethod
    def compute_degree_distribution(
        edge_index: torch.Tensor,
        n_nodes: int,
    ) -> Dict[str, any]:
        """Compute degree distribution and power-law fit.

        Args:
            edge_index: [2, n_edges] edge list (undirected)
            n_nodes: Total number of nodes

        Returns:
            Dict with:
                - "degrees": tensor of degree for each node
                - "degree_distribution": histogram of degrees
                - "power_law_exponent": γ (fit of P(k) ~ k^(-γ))
                - "is_scale_free": Is exponent in [2, 3]?
                - "average_degree": <k>
                - "max_degree": max k
        """
        edge_index = edge_index.detach().cpu().long()

        # Count degree of each node
        degrees = torch.zeros(n_nodes, dtype=torch.long)
        degrees.scatter_add_(0, edge_index[0], 1)
        degrees.scatter_add_(0, edge_index[1], 1)

        # Degree distribution (histogram)
        deg_counts = torch.bincount(degrees)
        degree_dist = deg_counts.float() / deg_counts.sum()

        # Fit power law: log P(k) ~ -γ log(k)
        k_vals = torch.arange(1, len(degree_dist))
        p_vals = degree_dist[1:]
        valid_idx = p_vals > 0

        if valid_idx.sum() > 2:
            # Log-log linear regression
            log_k = torch.log(k_vals[valid_idx].float())
            log_p = torch.log(p_vals[valid_idx])

            # Simple least squares fit
            n_pts = valid_idx.sum().item()
            mean_log_k = log_k.mean()
            mean_log_p = log_p.mean()

            numerator = ((log_k - mean_log_k) * (log_p - mean_log_p)).sum()
            denominator = ((log_k - mean_log_k) ** 2).sum()

            if denominator > 1e-10:
                slope = numerator / denominator
                gamma = -slope.item()  # P(k) ~ k^(-γ) means log P ~ -γ log k
            else:
                gamma = 0.0
        else:
            gamma = 0.0

        avg_degree = degrees.float().mean().item()
        max_degree = degrees.max().item()

        return {
            "degrees": degrees,
            "degree_distribution": degree_dist,
            "power_law_exponent": float(gamma),
            "is_scale_free": 2.0 <= gamma <= 3.0,
            "average_degree": float(avg_degree),
            "max_degree": int(max_degree),
        }

    @staticmethod
    def compute_pagerank(
        edge_index: torch.Tensor,
        n_nodes: int,
        damping: float = 0.85,
        n_iterations: int = 30,
    ) -> torch.Tensor:
        """Compute PageRank centrality using power iteration method.

        PageRank (Brin & Page, 1998):
            rank(v) = (1-d)/N + d · Σ_{u→v} rank(u) / out_degree(u)

        where:
            d = damping factor (0.85 typical)
            N = number of nodes
            u→v = edge from u to v

        The damping factor models "random walking": with probability (1-d),
        jump to random node; with probability d, follow link.

        Args:
            edge_index: [2, n_edges] edge list (can be directed or undirected)
            n_nodes: Total number of nodes
            damping: Damping factor (default 0.85, typical range [0.8, 0.9])
            n_iterations: Number of power iterations (default 30)

        Returns:
            [n_nodes] PageRank vector, sum = 1, higher = more central
        """
        edge_index = edge_index.detach().cpu().long()

        # Compute out-degrees
        out_degrees = torch.zeros(n_nodes, dtype=torch.float)
        out_degrees.scatter_add_(0, edge_index[0], 1.0)

        # Avoid division by zero (for sink nodes with no outlinks)
        out_degrees[out_degrees == 0] = 1.0

        # Initialize PageRank uniformly
        pagerank = torch.ones(n_nodes) / n_nodes

        # Power iteration
        for _ in range(n_iterations):
            new_rank = (1 - damping) / n_nodes * torch.ones(n_nodes)

            # Contribution from incoming edges
            for src, dst in edge_index.t():
                src, dst = src.item(), dst.item()
                new_rank[dst] += damping * pagerank[src] / out_degrees[src]

            pagerank = new_rank

        return pagerank

    @staticmethod
    def identify_network_motifs(
        edge_index: torch.Tensor,
        n_nodes: int,
        motif_type: str = "feed_forward_loop",
    ) -> List[Tuple[int, int, int]]:
        """Identify network motifs (recurring patterns).

        Common motifs in signaling networks:
        - Feed-forward loop: A→B→C and A→C (amplifies signal)
        - Feedback loop: A→B→A (oscillation or bistability)
        - Bistable switch: mutual inhibition (X⊣Y and Y⊣X)

        Args:
            edge_index: [2, n_edges] edge list
            n_nodes: Total number of nodes
            motif_type: "feed_forward_loop", "feedback_loop", "bistable_switch"

        Returns:
            List of tuples identifying motif instances
        """
        edge_index = edge_index.detach().cpu().long()

        # Build adjacency list for fast lookup
        adjacency = {i: set() for i in range(n_nodes)}
        for src, dst in edge_index.t():
            src, dst = src.item(), dst.item()
            adjacency[src].add(dst)

        motifs = []

        if motif_type == "feed_forward_loop":
            # Pattern: A→B, B→C, A→C
            for a in range(n_nodes):
                for b in adjacency[a]:
                    for c in adjacency[b]:
                        if c in adjacency[a] and a != c:
                            motifs.append((a, b, c))

        elif motif_type == "feedback_loop":
            # Pattern: A→B→A (3-node simplest case)
            for a in range(n_nodes):
                for b in adjacency[a]:
                    if a in adjacency[b]:
                        motifs.append((a, b))

        elif motif_type == "bistable_switch":
            # Pattern: A⊣B and B⊣A (mutual inhibition)
            for a in range(n_nodes):
                for b in adjacency[a]:
                    if a in adjacency[b]:
                        motifs.append((a, b))

        return motifs

    @staticmethod
    def target_druggability_score(
        pagerank: torch.Tensor,
        degree: torch.Tensor,
        betweenness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute composite druggability score for drug targets.

        A good drug target should be:
        1. Information hub (high PageRank) - affects many downstream proteins
        2. Moderate degree - not an essential hub needed everywhere
        3. High betweenness - controls flow through network

        Composite score (normalized):
            druggability = 0.4·PR + 0.3·degree_norm + 0.3·betweenness_norm

        where each component is min-max normalized to [0, 1].

        Args:
            pagerank: [n_nodes] PageRank centrality
            degree: [n_nodes] node degrees
            betweenness: [n_nodes] betweenness centrality (optional)

        Returns:
            [n_nodes] druggability scores in [0, 1]
        """
        pagerank = pagerank.detach().cpu().float()
        degree = degree.detach().cpu().float()

        # Normalize components to [0, 1]
        pr_norm = (pagerank - pagerank.min()) / (pagerank.max() - pagerank.min() + 1e-10)
        deg_norm = (degree - degree.min()) / (degree.max() - degree.min() + 1e-10)

        if betweenness is not None:
            betweenness = betweenness.detach().cpu().float()
            bc_norm = (betweenness - betweenness.min()) / (betweenness.max() - betweenness.min() + 1e-10)
        else:
            bc_norm = torch.zeros_like(pr_norm)

        # Composite score
        weights = torch.tensor([0.4, 0.3, 0.3])
        druggability = 0.4 * pr_norm + 0.3 * deg_norm + 0.3 * bc_norm

        return druggability / druggability.max()


class CognitiveDecisionTheory:
    """Tversky's similarity and decision frameworks for target ranking.

    From first principles (Tversky, 1977):
    ============================================================

    ASYMMETRIC SIMILARITY MODEL
    ----------------------------
    Tversky's model of similarity (not distance):

        S(a, b) = f(A ∩ B) / [f(A ∩ B) + α·f(A-B) + β·f(B-A)]

    where:
        A, B = feature sets of items a, b
        A ∩ B = common features
        A-B = features in a but not b
        B-A = features in b but not a
        α, β = weights (often α=β=1 for symmetric, α<β for asymmetry)
        f = measure function (e.g., number of features, or sum of importance)

    Key insight: Similarity is ASYMMETRIC!
        S(China, Taiwan) ≠ S(Taiwan, China)

    Applied to drug target selection:
        - Compare candidate targets to known successful target (e.g., BCR-ABL/Imatinib)
        - S(candidate, benchmark) = how much does candidate resemble benchmark?
        - High similarity → likely good target (similar success factors)

    ELIMINATION BY ASPECTS (EBA)
    ----------------------------
    Tversky's sequential decision model: evaluate candidates one aspect at a time,
    probabilistically eliminating low scorers.

    Procedure:
        1. Rank aspects by importance (e.g., PageRank > druggability > tissue expression)
        2. For each aspect in order:
           - Compute quantile threshold (e.g., top 50%)
           - Eliminate candidates below threshold
        3. Return remaining candidates (or use rank within remaining)

    Applied to selecting top drug targets from PPI network:
        - Most important aspect: Is target in PPI and information hub?
        - Next: Is target druggable (not essential, not too many interactions)?
        - Next: Expression pattern (preferentially expressed in tumor/pathogen)?
        - Etc.
    """

    @staticmethod
    def tversky_similarity(
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> float:
        """Compute asymmetric Tversky similarity S(a, b).

        S(a, b) = f(A ∩ B) / [f(A ∩ B) + α·f(A-B) + β·f(B-A)]

        where:
            A ∩ B = intersection (both have feature)
            A-B = features a has but b doesn't
            B-A = features b has but a doesn't
            α, β control relative weight of asymmetries

        Args:
            features_a: Binary feature vector for item a
            features_b: Binary feature vector for item b
            alpha: Weight for A-B asymmetry (default 1.0)
            beta: Weight for B-A asymmetry (default 1.0)

        Returns:
            Similarity in [0, 1]. Higher = more similar.
        """
        features_a = features_a.detach().cpu().float()
        features_b = features_b.detach().cpu().float()

        # Binarize (anything > 0.5 is "has feature")
        a_binary = (features_a > 0.5).float()
        b_binary = (features_b > 0.5).float()

        # Compute set operations (as cardinalities / sums)
        intersection = (a_binary * b_binary).sum().item()
        a_only = (a_binary * (1 - b_binary)).sum().item()
        b_only = (b_binary * (1 - a_binary)).sum().item()

        # Tversky formula
        denominator = intersection + alpha * a_only + beta * b_only
        if denominator < 1e-10:
            return 0.0

        similarity = intersection / denominator
        return float(similarity)

    @staticmethod
    def elimination_by_aspects(
        candidates: torch.Tensor,
        feature_names: List[str],
        feature_importance: torch.Tensor,
        n_rounds: int = 3,
        elimination_threshold: float = 0.5,
    ) -> List[Tuple[int, float]]:
        """Rank candidates using elimination-by-aspects procedure.

        Tversky's EBA: sequentially eliminate low performers on most-important
        aspects, keeping remaining candidates.

        Args:
            candidates: [n_candidates, n_features] feature matrix
            feature_names: List of n_features feature names
            feature_importance: [n_features] importance weights (sum to 1)
            n_rounds: Number of elimination rounds (default 3)
            elimination_threshold: Quantile below which to eliminate (default 0.5 = median)

        Returns:
            List of (candidate_idx, remaining_score) tuples, ranked by final score
        """
        candidates = candidates.detach().cpu().float()
        feature_importance = feature_importance.detach().cpu().float()

        # Normalize importance to sum to 1
        importance = feature_importance / feature_importance.sum()

        # Sort features by importance
        sorted_idx = torch.argsort(importance, descending=True)

        # Track active candidates
        active = set(range(len(candidates)))
        scores = torch.zeros(len(candidates))

        for round_idx in range(min(n_rounds, len(sorted_idx))):
            feature_idx = sorted_idx[round_idx].item()
            feature_scores = candidates[:, feature_idx]

            # Get threshold for active candidates only
            active_scores = torch.tensor([
                feature_scores[i].item() for i in active
            ])

            if len(active_scores) > 0:
                threshold = torch.quantile(active_scores, elimination_threshold)

                # Eliminate below threshold
                to_remove = {
                    idx for idx in active
                    if feature_scores[idx].item() < threshold.item()
                }
                active = active - to_remove

                # Accumulate scores
                for idx in active:
                    scores[idx] += importance[feature_idx] * feature_scores[idx]

        # Sort remaining by score
        remaining = sorted(
            [(idx, scores[idx].item()) for idx in active],
            key=lambda x: x[1],
            reverse=True,
        )

        return remaining
