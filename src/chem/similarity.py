"""Ingredient-level Tanimoto similarity computation.

Three strategies for comparing ingredients (which are sets of compounds):

1. **mean_best_match** (default): For each compound in ingredient A, find the
   most similar compound in ingredient B (max Tanimoto). Average these best-match
   scores across A's compounds, then symmetrize by averaging both directions.
   - Pros: Robust to different set sizes. Captures partial chemical overlap.
   - Cons: O(|A| * |B|) per pair. Sensitive to outlier compounds.

2. **max_pairwise**: Maximum Tanimoto across all compound pairs (A_i, B_j).
   - Pros: Fast, simple. If ANY two compounds are similar, the score is high.
   - Cons: Ignores the bulk of the chemical profile. One coincidental match
     inflates similarity. Good for "do these share at least one compound class?"

3. **union_fingerprint**: Compute Tanimoto between mean-pooled (or union) 
   fingerprints of each ingredient.
   - Pros: O(1) per pair after pooling. Memory-efficient for large-scale.
   - Cons: Union pooling loses specificity (see pool_union docstring).
     Mean-pooling produces non-binary vectors, requiring generalized Tanimoto.

Memory constraints:
- Never compute a dense N*N matrix over all ingredients.
- Compute similarity only against hub ingredients (configurable).
- Keep only top-k per query ingredient.
- Batch computation to stay under 16 GB.
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def tanimoto_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity between two binary fingerprint vectors.

    T(A,B) = |A ∩ B| / |A ∪ B| = sum(A & B) / (sum(A) + sum(B) - sum(A & B))

    Returns 0.0 if both vectors are all-zero (no structural information).
    """
    intersection = int(np.sum(np.bitwise_and(a, b)))
    union = int(np.sum(a)) + int(np.sum(b)) - intersection
    if union == 0:
        return 0.0
    return float(intersection / union)


def tanimoto_continuous(a: np.ndarray, b: np.ndarray) -> float:
    """Generalized Tanimoto for continuous (mean-pooled) fingerprint vectors.

    T(A,B) = sum(min(A,B)) / sum(max(A,B))

    This is the standard extension of Tanimoto to continuous vectors.
    Equivalent to binary Tanimoto when inputs are {0,1}.
    """
    numerator = float(np.sum(np.minimum(a, b)))
    denominator = float(np.sum(np.maximum(a, b)))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def pairwise_tanimoto_matrix(fps_a: List[np.ndarray], fps_b: List[np.ndarray]) -> np.ndarray:
    """Vectorized pairwise Tanimoto matrix of shape (|A|, |B|).

    Uses fast BLAS dot product for intersection calculation.
    """
    if not fps_a or not fps_b:
        return np.zeros((0, 0), dtype=np.float32)

    A = np.asarray(fps_a, dtype=np.float32)
    B = np.asarray(fps_b, dtype=np.float32)

    inter = np.dot(A, B.T)
    sum_a = A.sum(axis=1, keepdims=True)
    sum_b = B.sum(axis=1, keepdims=True).T
    union = sum_a + sum_b - inter

    # Avoid divide-by-zero where both compounds are all-zeros
    sim_matrix = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return sim_matrix


def _compound_fps_for_ingredient(
    ingredient_node_id: int,
    ingredient_compounds: Dict[int, Set[int]],
    node_id_to_cid: Dict[int, int],
    cid_to_fp: Dict[int, np.ndarray],
) -> List[np.ndarray]:
    """Get all valid compound fingerprints for an ingredient."""
    comp_nids = ingredient_compounds.get(ingredient_node_id, set())
    fps = []
    for nid in comp_nids:
        cid = node_id_to_cid.get(nid)
        if cid is not None and cid in cid_to_fp:
            fps.append(cid_to_fp[cid])
    return fps


def similarity_mean_best_match(
    fps_a: List[np.ndarray],
    fps_b: List[np.ndarray],
) -> float:
    """Mean best-match Tanimoto between two sets of compound fingerprints.

    For each compound in A, find the max Tanimoto to any compound in B.
    Do the same B->A. Return the average of both direction means.
    Symmetrized: sim(A,B) == sim(B,A).
    """
    if not fps_a or not fps_b:
        return 0.0

    mat = pairwise_tanimoto_matrix(fps_a, fps_b)
    best_a_to_b = mat.max(axis=1)  # max for each compound in A
    best_b_to_a = mat.max(axis=0)  # max for each compound in B

    ab = float(np.mean(best_a_to_b))
    ba = float(np.mean(best_b_to_a))
    return (ab + ba) / 2.0


def similarity_max_pairwise(
    fps_a: List[np.ndarray],
    fps_b: List[np.ndarray],
) -> float:
    """Maximum pairwise Tanimoto between two sets of compound fingerprints."""
    if not fps_a or not fps_b:
        return 0.0

    mat = pairwise_tanimoto_matrix(fps_a, fps_b)
    return float(mat.max())


def similarity_union_fingerprint(
    pooled_a: Optional[np.ndarray],
    pooled_b: Optional[np.ndarray],
    is_binary: bool = False,
) -> float:
    """Tanimoto between pre-pooled ingredient fingerprints.

    Uses binary or continuous Tanimoto depending on the pooling strategy.
    """
    if pooled_a is None or pooled_b is None:
        return 0.0
    if is_binary:
        return tanimoto_binary(pooled_a, pooled_b)
    return tanimoto_continuous(pooled_a, pooled_b)


def compute_sparse_topk_similarity(
    query_ids: List[int],
    candidate_ids: List[int],
    ingredient_compounds: Dict[int, Set[int]],
    node_id_to_cid: Dict[int, int],
    cid_to_fp: Dict[int, np.ndarray],
    strategy: str = "mean_best_match",
    top_k: int = 10,
    threshold: float = 0.0,
    batch_size: int = 256,
    pooled_fps: Optional[Dict[int, np.ndarray]] = None,
    cache_path: Optional[str] = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """Compute sparse top-k ingredient similarities.

    For each query ingredient, compute similarity to all candidates,
    keep only top-k above threshold.

    Memory guard: processes candidates with pre-stacked BLAS dot products.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Loaded cached top-k similarities from {cache_path}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")

    results: Dict[int, List[Tuple[int, float]]] = {qid: [] for qid in query_ids}

    # Case 1: Continuous or binary Tanimoto on pooled fingerprints (instantaneous)
    if strategy == "union_fingerprint":
        cand_fps = {cid: pooled_fps[cid] for cid in candidate_ids if pooled_fps and cid in pooled_fps}
        for qid in query_ids:
            if not pooled_fps or qid not in pooled_fps:
                continue
            q_fp = pooled_fps[qid]
            is_bin = (q_fp.dtype == np.uint8 and np.all((q_fp == 0) | (q_fp == 1)))
            cand_scores = []
            for cid, c_fp in cand_fps.items():
                if cid == qid:
                    continue
                sim = similarity_union_fingerprint(q_fp, c_fp, is_binary=is_bin)
                if sim >= threshold:
                    cand_scores.append((cid, sim))
            cand_scores.sort(key=lambda x: -x[1])
            results[qid] = cand_scores[:top_k]

    # Case 2: Set-based compound matching (mean_best_match or max_pairwise)
    else:
        # Pre-stack 2D float32 arrays and row sums for ingredients that have compounds
        relevant_ids = set(query_ids) | set(candidate_ids)
        stacked_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for nid in relevant_ids:
            fps = _compound_fps_for_ingredient(nid, ingredient_compounds, node_id_to_cid, cid_to_fp)
            if fps:
                arr = np.asarray(fps, dtype=np.float32)
                sums = arr.sum(axis=1, keepdims=True)
                stacked_data[nid] = (arr, sums)

        # Candidates and queries that have compound data
        active_cands = [cid for cid in candidate_ids if cid in stacked_data]
        active_queries = [qid for qid in query_ids if qid in stacked_data]

        logger.info(
            f"Computing sparse chemical top-{top_k} similarity ({strategy}): "
            f"{len(active_queries)} active queries x {len(active_cands)} candidates..."
        )
        t_start = time.time()

        for idx, qid in enumerate(active_queries):
            A, sum_a = stacked_data[qid]
            cand_scores = []

            for cid in active_cands:
                if cid == qid:
                    continue
                B, sum_b = stacked_data[cid]
                inter = np.dot(A, B.T)
                union = sum_a + sum_b.T - inter
                mat = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

                if strategy == "mean_best_match":
                    sim = float(0.5 * (mat.max(axis=1).mean() + mat.max(axis=0).mean()))
                elif strategy == "max_pairwise":
                    sim = float(mat.max())
                else:
                    raise ValueError(f"Unknown similarity strategy: {strategy}")

                if sim >= threshold:
                    cand_scores.append((cid, sim))

            cand_scores.sort(key=lambda x: -x[1])
            results[qid] = cand_scores[:top_k]

            if (idx + 1) % 100 == 0 or (idx + 1) == len(active_queries):
                elapsed = time.time() - t_start
                logger.info(
                    f"  Processed {idx + 1}/{len(active_queries)} ingredients "
                    f"({(idx + 1) / len(active_queries) * 100:.1f}%) in {elapsed:.1f}s..."
                )

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Saved computed top-k similarities to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache similarities to {cache_path}: {e}")

    return results
