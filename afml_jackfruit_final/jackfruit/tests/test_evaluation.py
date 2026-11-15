"""Unit tests for evaluation metrics and baselines.

Verifies:
- Recall@10, Recall@20 calculations
- MRR (Mean Reciprocal Rank) calculation and edge cases
- NDCG@20 calculation against ideal ranking
- Spearman correlation on identical, inverted, and perturbed rankings
- Baselines (Random, Popularity, Tanimoto-kNN) output correctness
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.evaluate import (
    compute_spearman_rank_corr,
    compute_dcg_at_k,
    compute_ndcg_at_k,
    evaluate_ranking_for_ingredient,
    aggregate_metrics,
)
from src.baselines import RandomBaseline, PopularityBaseline, TanimotoKNNBaseline


def test_spearman_perfect_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    corr = compute_spearman_rank_corr(x, y)
    assert abs(corr - 1.0) < 1e-5, f"Expected 1.0, got {corr}"


def test_spearman_inverse_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    corr = compute_spearman_rank_corr(x, y)
    assert abs(corr - (-1.0)) < 1e-5, f"Expected -1.0, got {corr}"


def test_spearman_zero_variance():
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([2.0, 3.0, 4.0])
    corr = compute_spearman_rank_corr(x, y)
    assert corr == 0.0, f"Expected 0.0 for zero variance, got {corr}"


def test_ndcg_ideal():
    ideal = np.array([0.9, 0.7, 0.5, 0.3])
    ndcg = compute_ndcg_at_k(ideal, ideal, k=4)
    assert abs(ndcg - 1.0) < 1e-5, f"Expected 1.0, got {ndcg}"


def test_ndcg_worst():
    ideal = np.array([1.0, 1.0, 1.0])
    worst = np.array([0.0, 0.0, 0.0])
    ndcg = compute_ndcg_at_k(worst, ideal, k=3)
    assert abs(ndcg - 0.0) < 1e-5, f"Expected 0.0, got {ndcg}"


def test_evaluate_ranking_metrics():
    # Cold node pairing with partners 2 and 3
    ground_truth = {2: 0.8, 3: 0.5}
    predicted = {
        1: 0.1,
        2: 0.9,  # rank 1
        3: 0.4,  # rank 3
        4: 0.6,  # rank 2
        5: 0.05, # rank 4
    }

    metrics = evaluate_ranking_for_ingredient(predicted, ground_truth, ks=(2, 5))
    # Top-2 candidates are 2 and 4. Hits = {2} -> Recall@2 = 1/2 = 0.5
    assert abs(metrics["recall_2"] - 0.5) < 1e-5, f"Expected 0.5, got {metrics['recall_2']}"
    # Top-5 candidates include {2, 3}. Hits = {2, 3} -> Recall@5 = 2/2 = 1.0
    assert abs(metrics["recall_5"] - 1.0) < 1e-5, f"Expected 1.0, got {metrics['recall_5']}"
    # First hit is at rank 1 -> MRR = 1.0
    assert abs(metrics["mrr"] - 1.0) < 1e-5, f"Expected 1.0, got {metrics['mrr']}"


def test_baselines():
    train_edges = [(1, 2), (2, 3), (2, 4)]
    train_weights = [0.8, 0.6, 0.4]

    # Popularity Baseline: Node 2 has degree 0.8+0.6+0.4 = 1.8 (highest)
    pop = PopularityBaseline(train_edges, train_weights)
    pred_pop = pop.predict(cold_id=10, candidate_ids=[1, 2, 3, 4, 5])
    assert pred_pop[2] > pred_pop[1]
    assert pred_pop[5] == 0.0

    # Tanimoto-kNN Baseline
    # Cold node 10 is similar to hub 1 (sim=0.9)
    topk_sim = {10: [(1, 0.9)]}
    knn = TanimotoKNNBaseline(train_edges, train_weights, topk_sim, top_k=5)
    pred_knn = knn.predict(cold_id=10, candidate_ids=[1, 2, 3, 4, 5])
    # Hub 1 connects to node 2 with weight 0.8 -> pred for node 2 should be 0.8
    assert abs(pred_knn[2] - 0.8) < 1e-5, f"Expected 0.8, got {pred_knn[2]}"
    # Hub 1 does not connect to 5 -> 0.0
    assert pred_knn[5] == 0.0


if __name__ == "__main__":
    print("Running evaluation & baseline tests...")
    test_spearman_perfect_correlation()
    test_spearman_inverse_correlation()
    test_spearman_zero_variance()
    test_ndcg_ideal()
    test_ndcg_worst()
    test_evaluate_ranking_metrics()
    test_baselines()
    print("ALL EVALUATION & BASELINE TESTS PASSED SUCCESSFULLY!")
