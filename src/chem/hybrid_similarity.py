"""Hybrid Molecular + Semantic Fusion for SavorLink Strategy III.

Combines biochemical aroma fingerprint similarity (Morgan/ECFP Tanimoto)
with encyclopedic culinary semantics (Sentence-Transformers MiniLM)
to provide a multi-modal representation for cold-start edge synthesis.
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


def build_hybrid_feature_matrix(
    mol_features: np.ndarray,
    sem_features: np.ndarray,
    cache_npy_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Concatenate L2-normalized molecular and semantic features."""
    if cache_npy_path and os.path.exists(cache_npy_path):
        try:
            arr = np.load(cache_npy_path)
            if arr.shape[0] == mol_features.shape[0]:
                logger.info(f"Loaded cached hybrid features from {cache_npy_path} {arr.shape}")
                return arr
        except Exception as e:
            logger.warning(f"Failed to load hybrid feature cache: {e}")

    # Normalize each feature block before concatenation
    mol_norm = mol_features / np.maximum(np.linalg.norm(mol_features, axis=1, keepdims=True), 1e-9)
    sem_norm = sem_features / np.maximum(np.linalg.norm(sem_features, axis=1, keepdims=True), 1e-9)

    fused = np.concatenate([mol_norm, sem_norm], axis=1).astype(np.float32)
    logger.info(f"Assembled hybrid feature matrix shape {fused.shape} (Mol: {mol_features.shape[1]} + Sem: {sem_features.shape[1]})")

    if cache_npy_path:
        os.makedirs(os.path.dirname(cache_npy_path), exist_ok=True)
        np.save(cache_npy_path, fused)
        logger.info(f"Saved hybrid features to {cache_npy_path}")

    return fused


def compute_hybrid_topk_similarity(
    query_ids: List[int],
    candidate_ids: List[int],
    mol_similarities: Dict[int, List[Tuple[int, float]]],
    sem_similarities: Dict[int, List[Tuple[int, float]]],
    alpha: float = 0.5,
    top_k: int = 10,
    threshold: float = 0.05,
    cache_path: Optional[str | Path] = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """Compute blended top-k similarity: alpha * Sim_mol + (1 - alpha) * Sim_sem."""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Loaded cached hybrid top-k similarities from {cache_path}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load hybrid similarity cache: {e}")

    results: Dict[int, List[Tuple[int, float]]] = {}
    t0 = time.time()

    for qid in query_ids:
        mol_map = dict(mol_similarities.get(qid, []))
        sem_map = dict(sem_similarities.get(qid, []))

        # Union of neighbors from both modalities
        all_neighbors = set(mol_map.keys()) | set(sem_map.keys())
        blended_scores: List[Tuple[int, float]] = []

        for cid in all_neighbors:
            if cid == qid or cid not in candidate_ids:
                continue
            s_mol = mol_map.get(cid, 0.0)
            s_sem = sem_map.get(cid, 0.0)
            score = float(alpha * s_mol + (1.0 - alpha) * s_sem)
            if score >= threshold:
                blended_scores.append((cid, score))

        blended_scores.sort(key=lambda x: -x[1])
        results[qid] = blended_scores[:top_k]

    logger.info(f"Computed hybrid top-{top_k} similarities (alpha={alpha:.2f}) in {time.time() - t0:.2f}s")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved hybrid top-k similarities to {cache_path}")

    return results
