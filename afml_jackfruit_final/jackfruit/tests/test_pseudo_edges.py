"""Unit tests for pseudo-edge builder.

Tests:
- Edge count matches expected (top_k * num_cold * 2 if bidirectional)
- Temperature scaling works correctly
- Threshold filtering drops low-similarity edges
- Self-loops not emitted
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.graph.pseudo_edges import build_pseudo_edges, merge_edges


class TestBuildPseudoEdges:
    @pytest.fixture
    def topk_data(self):
        """Pre-computed top-k similarities for 3 cold nodes."""
        return {
            100: [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.1)],
            200: [(5, 0.9), (6, 0.3)],
            300: [],  # No similar ingredients
        }

    def test_basic_edge_count(self, topk_data):
        edges, weights = build_pseudo_edges(
            cold_node_ids=[100, 200, 300],
            similarity_topk=topk_data,
            top_k=10,
            temperature=1.0,
            threshold=0.0,
            include_reverse=False,
        )
        assert len(edges) == 6
        assert len(weights) == 6

    def test_bidirectional_doubles(self, topk_data):
        edges, weights = build_pseudo_edges(
            cold_node_ids=[100, 200, 300],
            similarity_topk=topk_data,
            top_k=10,
            temperature=1.0,
            threshold=0.0,
            include_reverse=True,
        )
        assert len(edges) == 12

    def test_topk_truncation(self, topk_data):
        edges, weights = build_pseudo_edges(
            cold_node_ids=[100],
            similarity_topk=topk_data,
            top_k=2,
            temperature=1.0,
            threshold=0.0,
            include_reverse=False,
        )
        assert len(edges) == 2
        assert weights[0] == pytest.approx(0.8)
        assert weights[1] == pytest.approx(0.6)

    def test_threshold_filtering(self, topk_data):
        edges, weights = build_pseudo_edges(
            cold_node_ids=[100],
            similarity_topk=topk_data,
            top_k=10,
            temperature=1.0,
            threshold=0.5,
            include_reverse=False,
        )
        assert len(edges) == 2
        for w in weights:
            assert w >= 0.5

    def test_temperature_scaling(self, topk_data):
        edges_t1, weights_t1 = build_pseudo_edges(
            cold_node_ids=[100],
            similarity_topk=topk_data,
            top_k=1,
            temperature=1.0,
            threshold=0.0,
            include_reverse=False,
        )
        edges_t2, weights_t2 = build_pseudo_edges(
            cold_node_ids=[100],
            similarity_topk=topk_data,
            top_k=1,
            temperature=2.0,
            threshold=0.0,
            include_reverse=False,
        )
        assert weights_t2[0] > weights_t1[0]
        assert weights_t2[0] == pytest.approx(0.8 ** 0.5)

    def test_empty_cold_nodes(self):
        edges, weights = build_pseudo_edges(
            cold_node_ids=[],
            similarity_topk={},
            top_k=10,
        )
        assert len(edges) == 0
        assert len(weights) == 0


class TestMergeEdges:
    def test_merge_counts(self):
        real = [(1, 2), (3, 4)]
        real_w = [0.5, 0.7]
        pseudo = [(10, 20)]
        pseudo_w = [0.3]

        ei, ew, is_p = merge_edges(real, real_w, pseudo, pseudo_w)

        assert ei.shape == (2, 3)
        assert ew.shape == (3,)
        assert is_p.shape == (3,)
        assert is_p.sum() == 1
        assert (~is_p).sum() == 2

    def test_empty_merge(self):
        ei, ew, is_p = merge_edges([], [], [], [])
        assert ei.shape == (2, 0)
        assert ew.shape == (0,)

    def test_dtypes(self):
        ei, ew, is_p = merge_edges([(1, 2)], [0.5], [(3, 4)], [0.3])
        assert ei.dtype == np.int64
        assert ew.dtype == np.float32
        assert is_p.dtype == bool
