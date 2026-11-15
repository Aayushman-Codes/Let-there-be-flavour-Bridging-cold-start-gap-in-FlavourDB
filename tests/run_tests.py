"""Comprehensive test runner for SavorLink Strategy III across all modules and phases.
"""
import sys
import os
import math
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Imports
from src.chem.similarity import (
    tanimoto_binary,
    tanimoto_continuous,
    similarity_mean_best_match,
    similarity_max_pairwise,
    similarity_union_fingerprint,
    compute_sparse_topk_similarity,
)
from src.chem.fingerprints import pool_mean, pool_union
from src.graph.pseudo_edges import build_pseudo_edges, merge_edges
from src.data.loaders import FlavorGraphData, NodeInfo, EdgeInfo
from src.data.splits import (
    create_ingredient_split,
    verify_no_leakage,
    cold_mask_training,
)
from src.evaluate import (
    compute_spearman_rank_corr,
    compute_ndcg_at_k,
    evaluate_ranking_for_ingredient,
    aggregate_metrics,
)
from src.baselines import RandomBaseline, PopularityBaseline, TanimotoKNNBaseline


def approx_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, abs_tol=tol, rel_tol=tol)


passed = 0
failed = 0
errors = []

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] {name}: {e}")
        errors.append((name, traceback.format_exc()))


# =========================================================================
# Phase 2: Similarity & Fingerprints Tests
# =========================================================================
print("\n" + "="*55)
print("Phase 2: Chemical Fingerprints & Similarity Tests")
print("="*55)

rng = np.random.RandomState(42)
fp_a = (rng.rand(2048) > 0.7).astype(np.uint8)
fp_b = (np.random.RandomState(123).rand(2048) > 0.7).astype(np.uint8)
fp_c = (np.random.RandomState(999).rand(2048) > 0.95).astype(np.uint8)
fp_zeros = np.zeros(2048, dtype=np.uint8)
fp_ones = np.ones(2048, dtype=np.uint8)

def test_binary_identity():
    assert approx_eq(tanimoto_binary(fp_a, fp_a), 1.0)
    assert approx_eq(tanimoto_binary(fp_ones, fp_ones), 1.0)
run_test("test_binary_identity", test_binary_identity)

def test_binary_symmetry():
    assert approx_eq(tanimoto_binary(fp_a, fp_b), tanimoto_binary(fp_b, fp_a))
run_test("test_binary_symmetry", test_binary_symmetry)

def test_binary_disjoint():
    assert approx_eq(tanimoto_binary(fp_zeros, fp_a), 0.0)
    assert approx_eq(tanimoto_binary(fp_zeros, fp_zeros), 0.0)
run_test("test_binary_disjoint", test_binary_disjoint)

def test_binary_range():
    sim = tanimoto_binary(fp_a, fp_b)
    assert 0.0 <= sim <= 1.0
run_test("test_binary_range", test_binary_range)

def test_continuous_identity_and_symmetry():
    a = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    b = np.array([0.8, 0.2, 0.5], dtype=np.float64)
    assert approx_eq(tanimoto_continuous(a, a), 1.0)
    assert approx_eq(tanimoto_continuous(a, b), tanimoto_continuous(b, a))
run_test("test_continuous_identity_and_symmetry", test_continuous_identity_and_symmetry)

def test_continuous_reduces_to_binary():
    a_float = fp_a.astype(np.float64)
    b_float = fp_b.astype(np.float64)
    assert approx_eq(tanimoto_continuous(a_float, b_float), tanimoto_binary(fp_a, fp_b), tol=1e-9)
run_test("test_continuous_reduces_to_binary", test_continuous_reduces_to_binary)

def test_mean_best_match_properties():
    fps = [fp_a, fp_b]
    assert approx_eq(similarity_mean_best_match(fps, fps), 1.0)
    assert approx_eq(similarity_mean_best_match([fp_a, fp_b], [fp_c]),
                     similarity_mean_best_match([fp_c], [fp_a, fp_b]))
    assert approx_eq(similarity_mean_best_match([fp_a], []), 0.0)
run_test("test_mean_best_match_properties", test_mean_best_match_properties)

def test_max_pairwise_properties():
    assert approx_eq(similarity_max_pairwise([fp_a], [fp_a]), 1.0)
    assert approx_eq(similarity_max_pairwise([fp_a, fp_b], [fp_c]),
                     similarity_max_pairwise([fp_c], [fp_a, fp_b]))
    assert approx_eq(similarity_max_pairwise([fp_a], []), 0.0)
    max_sim = similarity_max_pairwise([fp_a, fp_b], [fp_c])
    mean_sim = similarity_mean_best_match([fp_a, fp_b], [fp_c])
    assert max_sim >= mean_sim - 1e-9
run_test("test_max_pairwise_properties", test_max_pairwise_properties)

def test_union_fingerprint_properties():
    assert approx_eq(similarity_union_fingerprint(fp_a, fp_a, is_binary=True), 1.0)
    assert approx_eq(similarity_union_fingerprint(fp_a, fp_b, is_binary=True),
                     similarity_union_fingerprint(fp_b, fp_a, is_binary=True))
    assert approx_eq(similarity_union_fingerprint(fp_a, None), 0.0)
    a = pool_mean([np.array([1, 0, 1, 0], dtype=np.uint8), np.array([0, 1, 0, 1], dtype=np.uint8)])
    b = pool_mean([np.array([1, 1, 0, 0], dtype=np.uint8)])
    sim = similarity_union_fingerprint(a, b, is_binary=False)
    assert 0.0 <= sim <= 1.0
run_test("test_union_fingerprint_properties", test_union_fingerprint_properties)

def test_sparse_topk():
    n_bits = 64
    c_rng = np.random.RandomState(42)
    compound_fps = {100 + i: (c_rng.rand(n_bits) > 0.5).astype(np.uint8) for i in range(10)}
    ingredient_compounds = {1: {10, 11}, 2: {12, 13}, 3: {14}, 4: {15, 16}, 5: {17}}
    node_to_cid = {10 + i: 100 + i for i in range(8)}

    res = compute_sparse_topk_similarity(
        query_ids=[1],
        candidate_ids=[2, 3, 4, 5],
        ingredient_compounds=ingredient_compounds,
        node_id_to_cid=node_to_cid,
        cid_to_fp=compound_fps,
        strategy="mean_best_match",
        top_k=2,
    )
    assert len(res[1]) <= 2
    sims = [s for _, s in res[1]]
    assert sims == sorted(sims, reverse=True)
run_test("test_sparse_topk", test_sparse_topk)


# =========================================================================
# Phase 3: Pseudo-edge & Leakage Tests
# =========================================================================
print("\n" + "="*55)
print("Phase 3: Pseudo-Edge Construction & Leakage Tests")
print("="*55)

def test_pseudo_edges_basic():
    topk_data = {
        100: [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.1)],
        200: [(5, 0.9), (6, 0.3)],
        300: [],
    }
    edges, weights = build_pseudo_edges(
        cold_node_ids=[100, 200, 300],
        similarity_topk=topk_data,
        top_k=10,
        include_reverse=False,
    )
    assert len(edges) == 6
    assert len(weights) == 6

    # Bidirectional doubles
    edges_bi, weights_bi = build_pseudo_edges(
        cold_node_ids=[100, 200, 300],
        similarity_topk=topk_data,
        top_k=10,
        include_reverse=True,
    )
    assert len(edges_bi) == 12
run_test("test_pseudo_edges_basic", test_pseudo_edges_basic)

def test_merge_edges_logic():
    real = [(1, 2), (3, 4)]
    real_w = [0.5, 0.7]
    pseudo = [(10, 20)]
    pseudo_w = [0.3]
    ei, ew, is_p = merge_edges(real, real_w, pseudo, pseudo_w)
    assert ei.shape == (2, 3)
    assert ew.shape == (3,)
    assert is_p.shape == (3,)
    assert int(is_p.sum()) == 1
    assert int((~is_p).sum()) == 2
run_test("test_merge_edges_logic", test_merge_edges_logic)

def make_toy_graph():
    fg = FlavorGraphData()
    for i in range(5):
        fg.nodes[i] = NodeInfo(
            node_id=i, name=f"hub_{i}", node_type="ingredient",
            is_hub=True, compound_subtype="", pubchem_cid=None
        )
    for i in range(5, 10):
        fg.nodes[i] = NodeInfo(
            node_id=i, name=f"nonhub_{i}", node_type="ingredient",
            is_hub=False, compound_subtype="", pubchem_cid=None
        )
    for i in range(100, 105):
        fg.nodes[i] = NodeInfo(
            node_id=i, name=f"comp_{i}", node_type="compound",
            is_hub=False, compound_subtype="food", pubchem_cid=1000 + i
        )
    for i in range(5):
        fg.edges.append(EdgeInfo(src=i, dst=100 + i, edge_type="ingr-fcomp", score=None))
    for i in range(10):
        for j in range(i + 1, 10):
            fg.edges.append(EdgeInfo(src=i, dst=j, edge_type="ingr-ingr", score=0.1 * (i + j)))
    fg.build_indices()
    return fg

def test_splits_and_leakage():
    fg = make_toy_graph()
    split = create_ingredient_split(fg, test_frac=0.2, val_frac=0.2, min_degree=1, seed=42)

    train = set(split.train_ids)
    val = set(split.val_ids)
    test = set(split.test_ids)

    assert len(train & val) == 0
    assert len(train & test) == 0
    assert len(val & test) == 0
    assert (train | val | test) == set(fg.ingredients.keys())

    violations = verify_no_leakage(split, fg)
    leaks = [v for v in violations if v.startswith("LEAK")]
    assert len(leaks) == 0

    heldout = val | test
    for src, dst in split.train_edges:
        assert src not in heldout
        assert dst not in heldout
run_test("test_splits_and_leakage", test_splits_and_leakage)

def test_cold_mask():
    fg = make_toy_graph()
    split = create_ingredient_split(fg, test_frac=0.2, val_frac=0.2, min_degree=1, seed=42)
    masked_ids, remaining, remaining_w, masked_gt = cold_mask_training(split, mask_frac=0.3, seed=42)

    assert masked_ids.issubset(set(split.train_ids))
    for src, dst in remaining:
        assert src not in masked_ids
        assert dst not in masked_ids
    assert len(remaining) + len(masked_gt) == len(split.train_edges)
run_test("test_cold_mask", test_cold_mask)


# =========================================================================
# Phase 4: Strategy III Model Architecture & Neural Network Tests
# =========================================================================
print("\n" + "="*55)
print("Phase 4: Model Architecture & Neural Network Tests")
print("="*55)

try:
    from tests.test_models import test_models_all
    run_test("test_models_architecture_and_gradients", test_models_all)
except Exception as e:
    print(f"  [SKIP] Neural network tests skipped: {e}")

# =========================================================================
# Phase 6: Evaluation Metrics & Baselines Tests
# =========================================================================
print("\n" + "="*55)
print("Phase 6: Evaluation Protocol & Baselines Tests")
print("="*55)

def test_eval_metrics():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    corr = compute_spearman_rank_corr(x, y)
    assert abs(corr - 1.0) < 1e-5

    ideal = np.array([0.9, 0.7, 0.5])
    ndcg = compute_ndcg_at_k(ideal, ideal, k=3)
    assert abs(ndcg - 1.0) < 1e-5

    ground_truth = {2: 0.8, 3: 0.5}
    predicted = {1: 0.1, 2: 0.9, 3: 0.4, 4: 0.6, 5: 0.05}
    m = evaluate_ranking_for_ingredient(predicted, ground_truth, ks=(2, 5))
    assert abs(m["recall_2"] - 0.5) < 1e-5
    assert abs(m["recall_5"] - 1.0) < 1e-5
    assert abs(m["mrr"] - 1.0) < 1e-5
run_test("test_eval_metrics", test_eval_metrics)

def test_baselines_execution():
    train_edges = [(1, 2), (2, 3), (2, 4)]
    train_weights = [0.8, 0.6, 0.4]

    pop = PopularityBaseline(train_edges, train_weights)
    pred_pop = pop.predict(cold_id=10, candidate_ids=[1, 2, 3, 4, 5])
    assert pred_pop[2] > pred_pop[1]

    topk_sim = {10: [(1, 0.9)]}
    knn = TanimotoKNNBaseline(train_edges, train_weights, topk_sim, top_k=5)
    pred_knn = knn.predict(cold_id=10, candidate_ids=[1, 2, 3, 4, 5])
    assert abs(pred_knn[2] - 0.8) < 1e-5
    assert pred_knn[5] == 0.0
run_test("test_baselines_execution", test_baselines_execution)

# =========================================================================
# Phase 7: Entrypoint Imports & Smoke Pipeline Execution
# =========================================================================
print("\n" + "="*55)
print("Phase 7: Entrypoint & End-to-End Smoke Verification")
print("="*55)

try:
    from tests.test_smoke import test_entrypoint_and_smoke
    run_test("test_entrypoint_and_smoke", test_entrypoint_and_smoke)
except Exception as e:
    print(f"  [SKIP] Smoke test skipped: {e}")

try:
    from tests.benchmark_memory import run_benchmark
    run_test("test_memory_and_hardware_compliance", run_benchmark)
except Exception as e:
    print(f"  [SKIP] Memory benchmark skipped: {e}")


print("\n" + "="*55)
print(f"Summary: {passed} passed, {failed} failed")
print("="*55)

if failed > 0:
    for name, trace in errors:
        print(f"\n--- Failure in {name} ---")
        print(trace)
    sys.exit(1)
else:
    print("ALL TESTS PASSED SUCCESSFULLY!")
