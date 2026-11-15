"""Graph construction utilities converting FlavorGraph data into PyTorch / PyG graph formats.

Ensures strict separation between node features (chemical only: pooled Morgan fingerprints)
and graph topology (ingredient-ingredient edges + pseudo-edges).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.data.loaders import FlavorGraphData
from src.data.splits import IngredientSplit
from src.graph.pseudo_edges import build_pseudo_edges, merge_edges

logger = logging.getLogger(__name__)


def build_node_id_mapping(node_ids: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Create contiguous 0-indexed integer IDs for a subset of FlavorGraph node_ids.

    Returns:
        (raw_to_idx, idx_to_raw) mappings.
    """
    sorted_ids = sorted(node_ids)
    raw_to_idx = {raw: idx for idx, raw in enumerate(sorted_ids)}
    idx_to_raw = {idx: raw for idx, raw in enumerate(sorted_ids)}
    return raw_to_idx, idx_to_raw


def assemble_feature_matrix(
    node_ids: List[int],
    raw_to_idx: Dict[int, int],
    pooled_fps: Dict[int, np.ndarray],
    n_bits: int = 2048,
) -> np.ndarray:
    """Build dense node feature matrix of shape (N, n_bits).

    For ingredients without valid compounds, an all-zero vector is assigned.
    Reports the count of valid vs all-zero feature rows.
    """
    num_nodes = len(node_ids)
    x = np.zeros((num_nodes, n_bits), dtype=np.float32)

    zero_count = 0
    for nid in node_ids:
        idx = raw_to_idx[nid]
        if nid in pooled_fps:
            x[idx] = pooled_fps[nid].astype(np.float32)
        else:
            zero_count += 1

    logger.info(
        f"Assembled feature matrix shape {x.shape}: "
        f"{num_nodes - zero_count:,} ingredients with valid chemical fingerprints, "
        f"{zero_count:,} ingredients with all-zero features ({zero_count / max(num_nodes, 1) * 100:.1f}%)"
    )
    return x
