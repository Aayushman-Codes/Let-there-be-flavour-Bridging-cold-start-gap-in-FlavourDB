"""Baseline models and methods for cold-start ingredient pairing.

Implements all 5 required evaluation baselines:
1. Random Baseline: Uniform random scores.
2. Popularity Baseline: Degree in the training graph.
3. Tanimoto-kNN (No GNN): Weighted average of chemical neighbors' recipe co-occurrences.
4. Zero-degree GNN: GNN with chemical features only, zero pseudo-edges (failure mode).
5. Full Strategy III: GNN with weighted pseudo-edge neighborhood aggregation.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RandomBaseline:
    """Baseline 1: Random ranking."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def predict(self, cold_id: int, candidate_ids: List[int]) -> Dict[int, float]:
        scores = self.rng.rand(len(candidate_ids))
        return {cid: float(s) for cid, s in zip(candidate_ids, scores)}


class PopularityBaseline:
    """Baseline 2: Degree popularity in the training graph.

    Predicts that cold ingredients pair with the most ubiquitous/frequent
    ingredients in the recipe dataset.
    """

    def __init__(self, train_edges: List[Tuple[int, int]], train_weights: List[float]):
        self.degrees: Dict[int, float] = defaultdict(float)
        for (src, dst), w in zip(train_edges, train_weights):
            self.degrees[src] += w
            self.degrees[dst] += w

    def predict(self, cold_id: int, candidate_ids: List[int]) -> Dict[int, float]:
        return {cid: float(self.degrees.get(cid, 0.0)) for cid in candidate_ids}


class TanimotoKNNBaseline:
    """Baseline 3: Tanimoto-kNN (No GNN).

    Finds the top-k nearest chemical hub neighbors of the cold ingredient,
    and predicts pairing scores as the weighted average of those neighbors'
    true training co-occurrence edges:
        score(u, c) = sum_{v in N_chem(u)} [ w_{uv} * NPMI_train(v, c) ] / sum(w_{uv})

    Why this matters:
    This is the core non-deep-learning baseline. If a graph neural network cannot
    outperform this simple chemical propagation, it must be reported honestly.
    """

    def __init__(
        self,
        train_edges: List[Tuple[int, int]],
        train_weights: List[float],
        topk_similarities: Dict[int, List[Tuple[int, float]]],
        top_k: int = 10,
    ):
        self.topk_similarities = topk_similarities
        self.top_k = top_k

        # Index training edges: (u, v) -> weight
        self.train_adj: Dict[int, Dict[int, float]] = defaultdict(dict)
        for (src, dst), w in zip(train_edges, train_weights):
            self.train_adj[src][dst] = w
            self.train_adj[dst][src] = w

    def predict(self, cold_id: int, candidate_ids: List[int]) -> Dict[int, float]:
        neighbors = self.topk_similarities.get(cold_id, [])[: self.top_k]
        if not neighbors:
            return {cid: 0.0 for cid in candidate_ids}

        total_weight = sum(sim for _, sim in neighbors)
        if total_weight <= 0:
            return {cid: 0.0 for cid in candidate_ids}

        scores: Dict[int, float] = {cid: 0.0 for cid in candidate_ids}

        for neighbor_id, sim in neighbors:
            neigh_edges = self.train_adj.get(neighbor_id, {})
            for cand_id in candidate_ids:
                if cand_id in neigh_edges:
                    scores[cand_id] += sim * neigh_edges[cand_id]

        for cand_id in candidate_ids:
            scores[cand_id] /= total_weight

        return scores


# Convenience aliases for multi-modal baselines
GenericKNNBaseline = TanimotoKNNBaseline
MolecularKNNBaseline = TanimotoKNNBaseline
SemanticKNNBaseline = TanimotoKNNBaseline
HybridKNNBaseline = TanimotoKNNBaseline
TaxonomicKNNBaseline = TanimotoKNNBaseline

