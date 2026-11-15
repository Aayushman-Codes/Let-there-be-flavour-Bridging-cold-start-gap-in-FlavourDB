"""Taxonomic Ontology Similarity & Feature Construction for SavorLink.

Calculates Wu-Palmer (WUP) tree depth similarity across hierarchical lineages
and generates taxonomic one-hot / categorical feature vectors.
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def wu_palmer_similarity(path_a: str, path_b: str) -> float:
    """Calculate Wu-Palmer tree similarity between two hierarchical taxonomy paths.

    Sim(A, B) = 2 * depth(LCS) / (depth(A) + depth(B))
    Where LCS is the Lowest Common Subsumer in the ontology tree.
    """
    parts_a = path_a.strip("/").split("/")
    parts_b = path_b.strip("/").split("/")

    # Depth includes root (Root has depth 1)
    # Find length of common prefix
    common_depth = 1  # Root (Food)
    min_len = min(len(parts_a), len(parts_b))
    for i in range(min_len):
        if parts_a[i] == parts_b[i]:
            common_depth += 1
        else:
            break

    depth_a = len(parts_a) + 1
    depth_b = len(parts_b) + 1

    sim = (2.0 * common_depth) / (depth_a + depth_b)
    return float(np.clip(sim, 0.0, 1.0))


def build_taxonomic_feature_matrix(
    taxonomy_data: Dict[str, Dict[str, str]],
    ingredient_order: List[str],
    cache_npy_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Build multi-hot categorical feature matrix from hierarchical taxonomy paths."""
    if cache_npy_path and os.path.exists(cache_npy_path):
        try:
            arr = np.load(cache_npy_path)
            if arr.shape[0] == len(ingredient_order):
                logger.info(f"Loaded cached taxonomic features from {cache_npy_path} {arr.shape}")
                return arr
        except Exception as e:
            logger.warning(f"Failed to load taxonomic feature cache: {e}")

    # Gather unique classes across all levels
    all_classes: set[str] = set()
    for name in ingredient_order:
        tax = taxonomy_data.get(name.lower(), {})
        if tax.get("origin"):
            all_classes.add(f"orig:{tax['origin']}")
        if tax.get("group"):
            all_classes.add(f"grp:{tax['group']}")
        if tax.get("subgroup"):
            all_classes.add(f"sub:{tax['subgroup']}")

    sorted_classes = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    dim = len(sorted_classes)

    features = np.zeros((len(ingredient_order), dim), dtype=np.float32)
    for i, name in enumerate(ingredient_order):
        tax = taxonomy_data.get(name.lower(), {})
        if tax.get("origin"):
            features[i, class_to_idx[f"orig:{tax['origin']}"]] = 1.0
        if tax.get("group"):
            features[i, class_to_idx[f"grp:{tax['group']}"]] = 1.0
        if tax.get("subgroup"):
            features[i, class_to_idx[f"sub:{tax['subgroup']}"]] = 1.0

    logger.info(f"Built taxonomic multi-hot feature matrix shape {features.shape} ({dim} ontology classes)")

    if cache_npy_path:
        os.makedirs(os.path.dirname(cache_npy_path), exist_ok=True)
        np.save(cache_npy_path, features)
        logger.info(f"Saved taxonomic features to {cache_npy_path}")

    return features


def compute_sparse_taxonomic_similarity(
    query_ids: List[int],
    candidate_ids: List[int],
    id_to_name: Dict[int, str],
    taxonomy_data: Dict[str, Dict[str, str]],
    top_k: int = 10,
    threshold: float = 0.0,
    cache_path: Optional[str | Path] = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """Compute sparse top-k Wu-Palmer similarity over the taxonomy hierarchy."""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Loaded cached taxonomic top-k similarities from {cache_path}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load taxonomic similarity cache: {e}")

    results: Dict[int, List[Tuple[int, float]]] = {}
    cand_paths = {cid: taxonomy_data.get(id_to_name[cid].lower(), {}).get("path", "") for cid in candidate_ids}

    t0 = time.time()
    for qid in query_ids:
        q_name = id_to_name.get(qid, "").lower()
        q_path = taxonomy_data.get(q_name, {}).get("path", "")
        if not q_path:
            results[qid] = []
            continue

        cand_scores: List[Tuple[int, float]] = []
        for cid, c_path in cand_paths.items():
            if cid == qid or not c_path:
                continue
            sim = wu_palmer_similarity(q_path, c_path)
            if sim >= threshold:
                cand_scores.append((cid, float(sim)))

        cand_scores.sort(key=lambda x: -x[1])
        results[qid] = cand_scores[:top_k]

    logger.info(f"Computed taxonomic top-{top_k} similarities for {len(query_ids)} items in {time.time() - t0:.2f}s")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved taxonomic top-k similarities to {cache_path}")

    return results
