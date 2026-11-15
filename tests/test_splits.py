"""Unit tests for ingredient splits with leakage detection.

Critical invariants:
1. Train/val/test are disjoint and cover all ingredients.
2. No training edge touches a val/test ingredient.
3. Deterministic given the same seed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.data.loaders import FlavorGraphData, NodeInfo, EdgeInfo
from src.data.splits import (
    create_ingredient_split,
    verify_no_leakage,
    cold_mask_training,
)


def _make_toy_graph() -> FlavorGraphData:
    """Create a small toy graph for testing.

    10 ingredients (IDs 0-9), 5 compounds (IDs 100-104).
    Ingredients 0-4 are hubs with compounds.
    Ingredients 5-9 are non-hubs without compounds.
    Edges: complete graph among 0-4, sparse among 5-9.
    """
    fg = FlavorGraphData()

    # Hub ingredients with compounds
    for i in range(5):
        fg.nodes[i] = NodeInfo(
            node_id=i, name=f"hub_ingr_{i}",
            node_type="ingredient", is_hub=True,
            compound_subtype="", pubchem_cid=None,
        )

    # Non-hub ingredients
    for i in range(5, 10):
        fg.nodes[i] = NodeInfo(
            node_id=i, name=f"nonhub_ingr_{i}",
            node_type="ingredient", is_hub=False,
            compound_subtype="", pubchem_cid=None,
        )

    # Compounds
    for i in range(100, 105):
        fg.nodes[i] = NodeInfo(
            node_id=i, name=f"compound_{i}",
            node_type="compound", is_hub=False,
            compound_subtype="food", pubchem_cid=1000 + i,
        )

    # Ingredient-compound edges (hubs only)
    for i in range(5):
        fg.edges.append(EdgeInfo(src=i, dst=100 + i, edge_type="ingr-fcomp", score=None))

    # Ingredient-ingredient edges: complete graph among 0-9
    for i in range(10):
        for j in range(i + 1, 10):
            fg.edges.append(EdgeInfo(
                src=i, dst=j, edge_type="ingr-ingr",
                score=0.1 * (i + j),  # Arbitrary NPMI
            ))

    fg.build_indices()
    return fg


class TestIngredientSplit:
    def test_disjoint_and_complete(self):
        fg = _make_toy_graph()
        split = create_ingredient_split(
            fg, test_frac=0.2, val_frac=0.2, min_degree=1, seed=42,
        )
        train = set(split.train_ids)
        val = set(split.val_ids)
        test = set(split.test_ids)

        assert train & val == set()
        assert train & test == set()
        assert val & test == set()
        assert train | val | test == set(fg.ingredients.keys())

    def test_no_leakage(self):
        fg = _make_toy_graph()
        split = create_ingredient_split(
            fg, test_frac=0.2, val_frac=0.2, min_degree=1, seed=42,
        )
        violations = verify_no_leakage(split, fg)
        leaks = [v for v in violations if v.startswith("LEAK")]
        assert len(leaks) == 0, f"Leakage detected: {leaks}"

    def test_no_train_edges_to_heldout(self):
        """The most critical test: no training edge should touch held-out nodes."""
        fg = _make_toy_graph()
        split = create_ingredient_split(
            fg, test_frac=0.2, val_frac=0.2, min_degree=1, seed=42,
        )
        heldout = set(split.val_ids) | set(split.test_ids)
        for src, dst in split.train_edges:
            assert src not in heldout, f"Train edge src {src} is held out"
            assert dst not in heldout, f"Train edge dst {dst} is held out"

    def test_deterministic(self):
        fg = _make_toy_graph()
        split1 = create_ingredient_split(fg, seed=42)
        split2 = create_ingredient_split(fg, seed=42)
        assert split1.train_ids == split2.train_ids
        assert split1.val_ids == split2.val_ids
        assert split1.test_ids == split2.test_ids

    def test_different_seeds(self):
        fg = _make_toy_graph()
        split1 = create_ingredient_split(fg, seed=42)
        split2 = create_ingredient_split(fg, seed=123)
        assert split1.test_ids != split2.test_ids or split1.val_ids != split2.val_ids

    def test_hub_only_mode(self):
        fg = _make_toy_graph()
        split = create_ingredient_split(
            fg, test_frac=0.2, val_frac=0.2,
            min_degree=1, hub_only=True, seed=42,
        )
        for nid in range(5, 10):
            assert nid in split.train_ids

    def test_ground_truth_exists(self):
        fg = _make_toy_graph()
        split = create_ingredient_split(
            fg, test_frac=0.2, val_frac=0.2, min_degree=1, seed=42,
        )
        if split.test_ids:
            assert len(split.test_ground_truth) > 0


class TestColdMaskTraining:
    def test_mask_removes_edges(self):
        fg = _make_toy_graph()
        split = create_ingredient_split(
            fg, test_frac=0.2, val_frac=0.2, min_degree=1, seed=42,
        )
        masked_ids, remaining, remaining_w, masked_gt = cold_mask_training(
            split, mask_frac=0.3, seed=42,
        )

        assert masked_ids.issubset(set(split.train_ids))

        for src, dst in remaining:
            assert src not in masked_ids
            assert dst not in masked_ids

        assert len(remaining) + len(masked_gt) == len(split.train_edges)

    def test_different_seeds_different_masks(self):
        fg = _make_toy_graph()
        split = create_ingredient_split(
            fg, test_frac=0.2, val_frac=0.2, min_degree=1, seed=42,
        )
        m1, _, _, _ = cold_mask_training(split, seed=1)
        m2, _, _, _ = cold_mask_training(split, seed=2)
        assert m1 != m2 or len(split.train_ids) <= 3
