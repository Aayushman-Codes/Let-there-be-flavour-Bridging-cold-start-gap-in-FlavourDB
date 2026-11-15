"""Evaluation protocol and ranking metrics for cold-start food pairing.

Metrics implemented per test ingredient:
- Recall@10, Recall@20
- Mean Reciprocal Rank (MRR)
- NDCG@20 (Normalized Discounted Cumulative Gain)
- Spearman rank correlation with ground-truth NPMI
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def compute_spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1D arrays.

    Pure-NumPy implementation to avoid hard dependencies.
    Handles ties cleanly via average rank.
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0

    def _rank(vals: np.ndarray) -> np.ndarray:
        order = np.argsort(vals)
        sorted_vals = vals[order]
        ranks = np.empty_like(vals, dtype=np.float64)
        n = len(vals)
        i = 0
        while i < n:
            j = i
            while j < n and sorted_vals[j] == sorted_vals[i]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    # Pearson correlation on ranks
    std_x = np.std(rx)
    std_y = np.std(ry)
    if std_x < 1e-9 or std_y < 1e-9:
        return 0.0

    cov = np.mean((rx - np.mean(rx)) * (ry - np.mean(ry)))
    corr = cov / (std_x * std_y)
    return float(np.clip(corr, -1.0, 1.0))


def compute_dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    k_eval = min(k, len(relevance))
    if k_eval == 0:
        return 0.0
    discounts = np.log2(np.arange(2, k_eval + 2))
    return float(np.sum(relevance[:k_eval] / discounts))


def compute_ndcg_at_k(true_relevance_ranked: np.ndarray, ideal_relevance: np.ndarray, k: int = 20) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg_at_k(true_relevance_ranked, k)
    idcg = compute_dcg_at_k(np.sort(ideal_relevance)[::-1], k)
    if idcg <= 0.0:
        return 0.0
    return float(min(1.0, dcg / idcg))


def evaluate_ranking_for_ingredient(
    predicted_scores: Dict[int, float],
    ground_truth: Dict[int, float],
    ks: Tuple[int, ...] = (10, 20),
) -> Dict[str, float]:
    """Evaluate ranking metrics for a single held-out cold ingredient.

    Args:
        predicted_scores: Map candidate_id -> predicted pairing score.
        ground_truth: Map partner_id -> true NPMI score (> 0).
        ks: Cutoff thresholds (default 10 and 20).

    Returns:
        Dictionary containing Recall@10, Recall@20, MRR, NDCG@20, Spearman.
    """
    if not ground_truth:
        return {
            "recall_10": 0.0,
            "recall_20": 0.0,
            "mrr": 0.0,
            "ndcg_20": 0.0,
            "spearman": 0.0,
        }

    # Sort all candidates by predicted score descending
    sorted_candidates = sorted(predicted_scores.items(), key=lambda x: -x[1])
    ranked_ids = [cid for cid, _ in sorted_candidates]

    positive_partners = set(ground_truth.keys())
    total_positives = len(positive_partners)

    # 1. Recall@K
    metrics: Dict[str, float] = {}
    for k in ks:
        top_k_set = set(ranked_ids[:k])
        hits = len(top_k_set & positive_partners)
        metrics[f"recall_{k}"] = hits / total_positives if total_positives > 0 else 0.0

    # 2. MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank_idx, cand_id in enumerate(ranked_ids, start=1):
        if cand_id in positive_partners:
            mrr = 1.0 / rank_idx
            break
    metrics["mrr"] = mrr

    # 3. NDCG@20
    # True relevance vector aligned with the predicted ranking
    relevance_ranked = np.array([ground_truth.get(cid, 0.0) for cid in ranked_ids[:20]], dtype=np.float64)
    all_ideal_relevance = np.array(list(ground_truth.values()), dtype=np.float64)
    metrics["ndcg_20"] = compute_ndcg_at_k(relevance_ranked, all_ideal_relevance, k=20)

    # 4. Spearman correlation on true partners
    # Compares the relative order of known true partners according to predictions vs ground truth NPMI
    if len(positive_partners) >= 3:
        partner_ids = list(positive_partners)
        pred_p = np.array([predicted_scores.get(pid, 0.0) for pid in partner_ids])
        true_p = np.array([ground_truth[pid] for pid in partner_ids])
        metrics["spearman"] = compute_spearman_rank_corr(pred_p, true_p)
    else:
        metrics["spearman"] = 0.0

    return metrics


def aggregate_metrics(all_ingredient_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Aggregate per-ingredient metrics into mean and std.

    Returns:
        Dict mapping metric_name -> (mean, std).
    """
    if not all_ingredient_metrics:
        return {}

    keys = all_ingredient_metrics[0].keys()
    summary: Dict[str, Tuple[float, float]] = {}

    for k in keys:
        vals = np.array([m[k] for m in all_ingredient_metrics], dtype=np.float64)
        summary[k] = (float(np.mean(vals)), float(np.std(vals)))

    return summary


def format_metrics_table(
    results_by_model: Dict[str, Dict[str, Tuple[float, float]]],
) -> str:
    """Format evaluation comparison table in markdown format.

    Args:
        results_by_model: Map model_name -> {metric_name: (mean, std)}

    Returns:
        Clean markdown table string.
    """
    metrics = ["recall_10", "recall_20", "mrr", "ndcg_20", "spearman"]
    headers = ["Model / Baseline", "Recall@10", "Recall@20", "MRR", "NDCG@20", "Spearman"]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for model_name, res in results_by_model.items():
        row = [f"**{model_name}**"]
        for m in metrics:
            if m in res:
                mean, std = res[m]
                row.append(f"{mean:.4f} ± {std:.4f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
