"""Semantic Embedding & Text Cosine Similarity for SavorLink Ingredients.

Encodes natural language ingredient descriptions into dense 384-dimensional
semantic vectors via Sentence-Transformers (`all-MiniLM-L6-v2`), computes
bounded cosine similarities, and extracts sparse top-k nearest neighbors.
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


def compute_semantic_embeddings(
    descriptions: Dict[str, str],
    ingredient_order: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    cache_npy_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Encode ingredient text descriptions into dense vector matrix.

    Args:
        descriptions: Map ingredient_name -> natural language text summary.
        ingredient_order: Ordered list of ingredient names.
        model_name: HuggingFace Sentence-Transformer model name.
        cache_npy_path: Optional path to save/load cached embeddings.

    Returns:
        embeddings: Float32 array of shape [len(ingredient_order), embedding_dim] (default 384)
    """
    if cache_npy_path and os.path.exists(cache_npy_path):
        try:
            arr = np.load(cache_npy_path)
            if arr.shape[0] == len(ingredient_order):
                logger.info(f"Loaded cached semantic embeddings from {cache_npy_path} {arr.shape}")
                return arr
        except Exception as e:
            logger.warning(f"Failed to load semantic embedding cache: {e}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading Sentence-Transformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        # If local_files_only fails, try default
        model = SentenceTransformer(model_name)

    texts = [descriptions.get(name.lower(), f"{name.title()} is a culinary ingredient.") for name in ingredient_order]
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    logger.info(f"Encoded {len(texts)} ingredient descriptions into {embeddings.shape} in {time.time() - t0:.2f}s")

    if cache_npy_path:
        os.makedirs(os.path.dirname(cache_npy_path), exist_ok=True)
        np.save(cache_npy_path, embeddings)
        logger.info(f"Saved semantic embeddings to {cache_npy_path}")

    return embeddings


def compute_sparse_semantic_similarity(
    query_ids: List[int],
    candidate_ids: List[int],
    id_to_name: Dict[int, str],
    embeddings: np.ndarray,
    name_to_idx: Dict[str, int],
    top_k: int = 10,
    threshold: float = 0.0,
    cache_path: Optional[str | Path] = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """Compute sparse top-k cosine similarity from semantic embeddings.

    Args:
        query_ids: Ingredient IDs to find neighbors for.
        candidate_ids: Candidate ingredient IDs (e.g. hubs).
        id_to_name: Map raw ingredient ID -> name.
        embeddings: Precomputed semantic embedding matrix [N, D].
        name_to_idx: Map name -> row index in embeddings matrix.
        top_k: Maximum neighbors per query.
        threshold: Minimum cosine similarity threshold.
        cache_path: Optional path to pickle cache.

    Returns:
        results: Dict mapping query_id -> sorted list of (candidate_id, similarity)
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Loaded cached semantic top-k similarities from {cache_path}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load semantic similarity cache: {e}")

    # Normalize embeddings to unit L2 norm for cosine dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_emb = embeddings / norms

    results: Dict[int, List[Tuple[int, float]]] = {}
    valid_cands = [cid for cid in candidate_ids if id_to_name.get(cid, "").lower() in name_to_idx]
    cand_indices = [name_to_idx[id_to_name[cid].lower()] for cid in valid_cands]
    cand_matrix = norm_emb[cand_indices]  # [M, D]

    t0 = time.time()
    for qid in query_ids:
        q_name = id_to_name.get(qid, "").lower()
        if q_name not in name_to_idx:
            results[qid] = []
            continue

        q_idx = name_to_idx[q_name]
        q_vec = norm_emb[q_idx]  # [D]

        # Cosine similarity dot product: [M]
        sims = np.dot(cand_matrix, q_vec)
        # Bound in [0, 1]
        sims = np.clip(sims, 0.0, 1.0)

        cand_scores: List[Tuple[int, float]] = []
        for cid, sim in zip(valid_cands, sims):
            if cid == qid:
                continue
            if sim >= threshold:
                cand_scores.append((cid, float(sim)))

        cand_scores.sort(key=lambda x: -x[1])
        results[qid] = cand_scores[:top_k]

    logger.info(f"Computed semantic top-{top_k} similarities for {len(query_ids)} items in {time.time() - t0:.2f}s")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved semantic top-k similarities to {cache_path}")

    return results
