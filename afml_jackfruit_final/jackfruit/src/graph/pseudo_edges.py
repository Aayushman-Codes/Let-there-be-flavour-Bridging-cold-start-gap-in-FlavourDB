"""Pseudo-edge builder for cold-start ingredients.

Core idea: An ingredient with zero recipe co-occurrence edges (cold-start)
has no neighborhood for a GNN to aggregate over. We inject soft, weighted
pseudo-edges from the cold node to its chemically most similar known
ingredients. The weights are Tanimoto similarity scores.

This gives the GNN aggregator a non-empty neighborhood of chemically
relevant nodes, letting it produce meaningful embeddings even for
ingredients it has never seen in recipes.

Design choices:
- Pseudo-edges are directed: cold -> neighbor (and optionally reversed).
- Edge weights are Tanimoto similarity, optionally temperature-scaled.
- A threshold floor drops very low-similarity edges (noise).
- Top-k keeps the neighborhood sparse and focused.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_pseudo_edges(
    cold_node_ids: List[int],
    similarity_topk: Dict[int, List[Tuple[int, float]]],
    top_k: int = 10,
    temperature: float = 1.0,
    threshold: float = 0.05,
    include_reverse: bool = True,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Build pseudo-edges from cold-start nodes to their nearest chemical neighbors.

    Args:
        cold_node_ids: Node IDs of cold-start ingredients (no real edges).
        similarity_topk: Pre-computed top-k similarities from
            compute_sparse_topk_similarity. Maps cold_node_id to
            [(neighbor_id, tanimoto_score), ...].
        top_k: Maximum pseudo-edges per cold node (already enforced in
            similarity computation, but re-enforced here for safety).
        temperature: Temperature scaling. sim_scaled = sim^(1/T).
            T>1 flattens differences (more uniform weights).
            T<1 sharpens (winner-take-all). T=1 is no change.
        threshold: Drop pseudo-edges with (scaled) similarity below this.
        include_reverse: If True, add reverse edges (neighbor -> cold).
            This lets the GNN also propagate information from the cold
            node back to neighbors, which can help if the cold node
            has useful chemistry features.

    Returns:
        Tuple of:
        - edge_list: List of (src, dst) tuples
        - edge_weights: List of float weights (Tanimoto scores)
    """
    edge_list: List[Tuple[int, int]] = []
    edge_weights: List[float] = []

    for cold_id in cold_node_ids:
        neighbors = similarity_topk.get(cold_id, [])

        for neighbor_id, raw_sim in neighbors[:top_k]:
            # Temperature scaling
            if temperature != 1.0 and raw_sim > 0:
                scaled_sim = float(raw_sim ** (1.0 / temperature))
            else:
                scaled_sim = float(raw_sim)

            if scaled_sim < threshold:
                continue

            # Forward edge: cold -> neighbor
            edge_list.append((cold_id, neighbor_id))
            edge_weights.append(scaled_sim)

            # Reverse edge: neighbor -> cold
            if include_reverse:
                edge_list.append((neighbor_id, cold_id))
                edge_weights.append(scaled_sim)

    logger.info(
        f"Built {len(edge_list)} pseudo-edges for {len(cold_node_ids)} cold nodes "
        f"(avg {len(edge_list) / max(len(cold_node_ids), 1):.1f} per node)"
    )

    return edge_list, edge_weights


def merge_edges(
    real_edges: List[Tuple[int, int]],
    real_weights: List[float],
    pseudo_edges: List[Tuple[int, int]],
    pseudo_weights: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge real and pseudo edges into PyG-compatible tensors.

    Returns:
        Tuple of:
        - edge_index: np.ndarray of shape (2, num_edges), int64
        - edge_weight: np.ndarray of shape (num_edges,), float32
        - is_pseudo: np.ndarray of shape (num_edges,), bool
            True for pseudo-edges, False for real edges.
            Useful for masking or differential message passing.
    """
    all_edges = real_edges + pseudo_edges
    all_weights = real_weights + pseudo_weights
    is_pseudo = [False] * len(real_edges) + [True] * len(pseudo_edges)

    if not all_edges:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
        )

    src = [e[0] for e in all_edges]
    dst = [e[1] for e in all_edges]

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_weight = np.array(all_weights, dtype=np.float32)
    is_pseudo_arr = np.array(is_pseudo, dtype=bool)

    return edge_index, edge_weight, is_pseudo_arr


def pseudo_edge_stats(
    edge_list: List[Tuple[int, int]],
    edge_weights: List[float],
    cold_node_ids: List[int],
) -> str:
    """Generate a report on pseudo-edge statistics.

    Returns the report as a string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Pseudo-Edge Statistics")
    lines.append("=" * 60)

    lines.append(f"\nTotal pseudo-edges: {len(edge_list)}")
    lines.append(f"Cold-start nodes: {len(cold_node_ids)}")

    if edge_weights:
        weights = np.array(edge_weights)
        lines.append(f"\nWeight distribution:")
        lines.append(f"  Min: {weights.min():.4f}")
        lines.append(f"  Max: {weights.max():.4f}")
        lines.append(f"  Mean: {weights.mean():.4f}")
        lines.append(f"  Std: {weights.std():.4f}")

        # Per cold node stats
        from collections import Counter
        edge_counts = Counter()
        cold_set = set(cold_node_ids)
        for src, dst in edge_list:
            if src in cold_set:
                edge_counts[src] += 1

        counts = list(edge_counts.values())
        if counts:
            lines.append(f"\nEdges per cold node:")
            lines.append(f"  Min: {min(counts)}")
            lines.append(f"  Max: {max(counts)}")
            lines.append(f"  Mean: {sum(counts)/len(counts):.1f}")

        # Cold nodes with zero pseudo-edges
        zero_nodes = len(cold_node_ids) - len(edge_counts)
        if zero_nodes > 0:
            lines.append(f"  Nodes with zero pseudo-edges: {zero_nodes}")

    report = "\n".join(lines)
    return report
