"""Unit tests for Tanimoto similarity strategies.

Tests the mathematical properties that must hold:
- Identity: sim(A, A) = 1.0
- Symmetry: sim(A, B) = sim(B, A)
- Disjoint: sim(all-zeros, nonzero) = 0.0
- Range: 0 <= sim <= 1
- Sparse top-k: correct ranking, correct count
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.chem.similarity import (
    tanimoto_binary,
    tanimoto_continuous,
    similarity_mean_best_match,
    similarity_max_pairwise,
    similarity_union_fingerprint,
    compute_sparse_topk_similarity,
)
from src.chem.fingerprints import pool_mean, pool_union


# --- Fixtures ---

@pytest.fixture
def fp_ones():
    """Fingerprint with all bits set."""
    return np.ones(2048, dtype=np.uint8)

@pytest.fixture
def fp_zeros():
    """Fingerprint with no bits set."""
    return np.zeros(2048, dtype=np.uint8)

@pytest.fixture
def fp_a():
    """Deterministic fingerprint A."""
    rng = np.random.RandomState(42)
    return (rng.rand(2048) > 0.7).astype(np.uint8)

@pytest.fixture
def fp_b():
    """Deterministic fingerprint B, partially overlapping with A."""
    rng = np.random.RandomState(123)
    return (rng.rand(2048) > 0.7).astype(np.uint8)

@pytest.fixture
def fp_c():
    """Deterministic fingerprint C, very different from A."""
    rng = np.random.RandomState(999)
    return (rng.rand(2048) > 0.95).astype(np.uint8)  # Sparse


# --- Binary Tanimoto tests ---

class TestTanimotoBinary:
    def test_identity(self, fp_a):
        assert tanimoto_binary(fp_a, fp_a) == pytest.approx(1.0)

    def test_symmetry(self, fp_a, fp_b):
        assert tanimoto_binary(fp_a, fp_b) == pytest.approx(
            tanimoto_binary(fp_b, fp_a)
        )

    def test_disjoint(self, fp_zeros, fp_a):
        assert tanimoto_binary(fp_zeros, fp_a) == pytest.approx(0.0)

    def test_both_zero(self, fp_zeros):
        assert tanimoto_binary(fp_zeros, fp_zeros) == pytest.approx(0.0)

    def test_range(self, fp_a, fp_b):
        sim = tanimoto_binary(fp_a, fp_b)
        assert 0.0 <= sim <= 1.0

    def test_identical_nonzero(self, fp_ones):
        assert tanimoto_binary(fp_ones, fp_ones) == pytest.approx(1.0)


# --- Continuous Tanimoto tests ---

class TestTanimotoContinuous:
    def test_identity(self):
        a = np.array([0.5, 0.3, 0.8], dtype=np.float64)
        assert tanimoto_continuous(a, a) == pytest.approx(1.0)

    def test_symmetry(self):
        a = np.array([0.1, 0.9, 0.5], dtype=np.float64)
        b = np.array([0.8, 0.2, 0.5], dtype=np.float64)
        assert tanimoto_continuous(a, b) == pytest.approx(
            tanimoto_continuous(b, a)
        )

    def test_zero_vector(self):
        a = np.zeros(10, dtype=np.float64)
        b = np.array([0.5] * 10, dtype=np.float64)
        assert tanimoto_continuous(a, b) == pytest.approx(0.0)

    def test_reduces_to_binary(self, fp_a, fp_b):
        """Continuous Tanimoto on binary vectors should match binary Tanimoto."""
        a_float = fp_a.astype(np.float64)
        b_float = fp_b.astype(np.float64)
        assert tanimoto_continuous(a_float, b_float) == pytest.approx(
            tanimoto_binary(fp_a, fp_b), abs=1e-10
        )


# --- Set-level similarity tests ---

class TestMeanBestMatch:
    def test_identical_sets(self, fp_a, fp_b):
        fps = [fp_a, fp_b]
        assert similarity_mean_best_match(fps, fps) == pytest.approx(1.0)

    def test_symmetry(self, fp_a, fp_b, fp_c):
        fps_x = [fp_a, fp_b]
        fps_y = [fp_c]
        assert similarity_mean_best_match(fps_x, fps_y) == pytest.approx(
            similarity_mean_best_match(fps_y, fps_x)
        )

    def test_empty_set(self, fp_a):
        assert similarity_mean_best_match([fp_a], []) == pytest.approx(0.0)
        assert similarity_mean_best_match([], [fp_a]) == pytest.approx(0.0)

    def test_range(self, fp_a, fp_b, fp_c):
        sim = similarity_mean_best_match([fp_a], [fp_b, fp_c])
        assert 0.0 <= sim <= 1.0


class TestMaxPairwise:
    def test_identical_sets(self, fp_a):
        assert similarity_max_pairwise([fp_a], [fp_a]) == pytest.approx(1.0)

    def test_symmetry(self, fp_a, fp_b, fp_c):
        assert similarity_max_pairwise([fp_a, fp_b], [fp_c]) == pytest.approx(
            similarity_max_pairwise([fp_c], [fp_a, fp_b])
        )

    def test_empty_set(self, fp_a):
        assert similarity_max_pairwise([fp_a], []) == pytest.approx(0.0)

    def test_dominates_mean(self, fp_a, fp_b, fp_c):
        """Max pairwise should be >= mean best match."""
        fps_x = [fp_a, fp_b]
        fps_y = [fp_c]
        max_sim = similarity_max_pairwise(fps_x, fps_y)
        mean_sim = similarity_mean_best_match(fps_x, fps_y)
        assert max_sim >= mean_sim - 1e-10


class TestUnionFingerprint:
    def test_identical(self, fp_a):
        assert similarity_union_fingerprint(fp_a, fp_a, is_binary=True) == pytest.approx(1.0)

    def test_symmetry(self, fp_a, fp_b):
        assert similarity_union_fingerprint(fp_a, fp_b, is_binary=True) == pytest.approx(
            similarity_union_fingerprint(fp_b, fp_a, is_binary=True)
        )

    def test_none_input(self, fp_a):
        assert similarity_union_fingerprint(fp_a, None) == pytest.approx(0.0)
        assert similarity_union_fingerprint(None, fp_a) == pytest.approx(0.0)

    def test_continuous(self):
        a = pool_mean([np.array([1, 0, 1, 0], dtype=np.uint8),
                       np.array([0, 1, 0, 1], dtype=np.uint8)])
        b = pool_mean([np.array([1, 1, 0, 0], dtype=np.uint8)])
        sim = similarity_union_fingerprint(a, b, is_binary=False)
        assert 0.0 <= sim <= 1.0


# --- Sparse top-k tests ---

class TestSparseTopK:
    def _make_data(self):
        """Create a small test dataset with known similarities."""
        rng = np.random.RandomState(42)

        # 5 ingredients, each with 1-3 compounds
        # Compounds have fingerprints
        n_bits = 64
        compound_fps = {}
        for cid in range(100, 120):
            compound_fps[cid] = (rng.rand(n_bits) > 0.5).astype(np.uint8)

        # ingredient -> compound node_ids
        ingredient_compounds = {
            1: {10, 11},      # compounds 10, 11
            2: {12, 13, 14},  # compounds 12, 13, 14
            3: {15},          # compound 15
            4: {16, 17},      # compounds 16, 17
            5: {18},          # compound 18
        }

        # node_id -> pubchem CID
        node_to_cid = {i: 100 + (i - 10) for i in range(10, 19)}

        # CID -> fingerprint
        cid_to_fp = {100 + i: compound_fps[100 + i] for i in range(9)}

        return ingredient_compounds, node_to_cid, cid_to_fp

    def test_topk_count(self):
        ic, n2c, c2fp = self._make_data()
        result = compute_sparse_topk_similarity(
            query_ids=[1],
            candidate_ids=[2, 3, 4, 5],
            ingredient_compounds=ic,
            node_id_to_cid=n2c,
            cid_to_fp=c2fp,
            strategy="mean_best_match",
            top_k=2,
            threshold=0.0,
        )
        assert len(result[1]) <= 2

    def test_topk_sorted_descending(self):
        ic, n2c, c2fp = self._make_data()
        result = compute_sparse_topk_similarity(
            query_ids=[1],
            candidate_ids=[2, 3, 4, 5],
            ingredient_compounds=ic,
            node_id_to_cid=n2c,
            cid_to_fp=c2fp,
            strategy="mean_best_match",
            top_k=4,
            threshold=0.0,
        )
        sims = [s for _, s in result[1]]
        assert sims == sorted(sims, reverse=True)

    def test_self_excluded(self):
        """Query should not appear in its own results."""
        ic, n2c, c2fp = self._make_data()
        result = compute_sparse_topk_similarity(
            query_ids=[1],
            candidate_ids=[1, 2, 3, 4, 5],
            ingredient_compounds=ic,
            node_id_to_cid=n2c,
            cid_to_fp=c2fp,
            strategy="mean_best_match",
            top_k=10,
            threshold=0.0,
        )
        neighbor_ids = {nid for nid, _ in result[1]}
        assert 1 not in neighbor_ids

    def test_threshold_filter(self):
        ic, n2c, c2fp = self._make_data()
        result = compute_sparse_topk_similarity(
            query_ids=[1],
            candidate_ids=[2, 3, 4, 5],
            ingredient_compounds=ic,
            node_id_to_cid=n2c,
            cid_to_fp=c2fp,
            strategy="mean_best_match",
            top_k=10,
            threshold=0.99,  # Very high threshold
        )
        for _, sim in result[1]:
            assert sim >= 0.99

    def test_symmetry_across_strategies(self):
        """All strategies should give symmetric results for pairs."""
        ic, n2c, c2fp = self._make_data()
        for strategy in ["mean_best_match", "max_pairwise"]:
            r1 = compute_sparse_topk_similarity(
                query_ids=[1], candidate_ids=[2],
                ingredient_compounds=ic, node_id_to_cid=n2c,
                cid_to_fp=c2fp, strategy=strategy, top_k=1, threshold=0.0,
            )
            r2 = compute_sparse_topk_similarity(
                query_ids=[2], candidate_ids=[1],
                ingredient_compounds=ic, node_id_to_cid=n2c,
                cid_to_fp=c2fp, strategy=strategy, top_k=1, threshold=0.0,
            )
            if r1[1] and r2[2]:
                assert r1[1][0][1] == pytest.approx(r2[2][0][1], abs=1e-10)

    def test_no_compounds_returns_empty(self):
        """Ingredient with no compounds should get no neighbors."""
        ic, n2c, c2fp = self._make_data()
        ic[99] = set()  # No compounds
        result = compute_sparse_topk_similarity(
            query_ids=[99],
            candidate_ids=[1, 2, 3],
            ingredient_compounds=ic,
            node_id_to_cid=n2c,
            cid_to_fp=c2fp,
            strategy="mean_best_match",
            top_k=10,
            threshold=0.0,
        )
        assert len(result[99]) == 0
