"""Ingredient-level train/val/test splits with cold-start simulation.

Critical design: We split INGREDIENTS, not edges. Test ingredients have
ALL their real edges removed, simulating true cold-start. This is the
only honest way to evaluate a cold-start method.

No-leakage invariants (enforced by tests):
1. Test/val ingredients have zero real edges in the training graph.
2. Test/val ingredients' FlavorGraph embeddings are not used as features.
3. Test/val ingredients' NPMI scores are not visible during training.
4. Node features come ONLY from chemistry (pooled fingerprints), never
   from graph structure or recipe co-occurrence.

Split eligibility:
- Only ingredients with compounds (i.e., hub ingredients) can be test/val,
  because they need chemical information for pseudo-edges.
- Only ingredients with sufficient ingr-ingr degree are useful for evaluation
  (we need ground-truth edges to compute ranking metrics).
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class IngredientSplit:
    """Container for train/val/test split of ingredients.

    Stores the node_ids in each split, plus the held-out edges for
    val/test evaluation.
    """
    train_ids: List[int] = field(default_factory=list)
    val_ids: List[int] = field(default_factory=list)
    test_ids: List[int] = field(default_factory=list)

    # Ground truth for evaluation: held-out edges and their NPMI scores
    # Maps (src, dst) -> npmi for val/test ingredients' edges
    val_ground_truth: Dict[Tuple[int, int], float] = field(default_factory=dict)
    test_ground_truth: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Training edges: only edges between train ingredients
    train_edges: List[Tuple[int, int]] = field(default_factory=list)
    train_edge_weights: List[float] = field(default_factory=list)


def create_ingredient_split(
    fg_data,  # FlavorGraphData, avoiding circular import
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    min_degree: int = 5,
    hub_only: bool = True,
    seed: int = 42,
) -> IngredientSplit:
    """Create train/val/test split of ingredients for cold-start evaluation.

    Algorithm:
    1. Identify eligible ingredients (hub, sufficient degree).
    2. Randomly split eligible ingredients into val and test.
    3. Everything else (including non-eligible) goes to train.
    4. Collect held-out edges (any edge touching val/test ingredients).
    5. Build training edge set (only edges between train ingredients).

    Args:
        fg_data: Loaded FlavorGraph data.
        test_frac: Fraction of eligible ingredients for test.
        val_frac: Fraction of eligible ingredients for val.
        min_degree: Minimum ingr-ingr degree to be eligible.
        hub_only: Only hub ingredients are eligible (they have compounds).
        seed: Random seed for reproducibility.

    Returns:
        IngredientSplit with all splits and ground truth.
    """
    rng = random.Random(seed)

    # Compute ingr-ingr degree
    degree: Dict[int, int] = defaultdict(int)
    for e in fg_data.ingr_ingr_edges:
        degree[e.src] += 1
        degree[e.dst] += 1

    # Identify eligible ingredients
    eligible = []
    for nid, info in fg_data.ingredients.items():
        if hub_only and not info.is_hub:
            continue
        if degree.get(nid, 0) < min_degree:
            continue
        eligible.append(nid)

    rng.shuffle(eligible)

    n_test = max(1, int(len(eligible) * test_frac))
    n_val = max(1, int(len(eligible) * val_frac))

    test_ids = sorted(eligible[:n_test])
    val_ids = sorted(eligible[n_test:n_test + n_val])
    # Train = everyone not in val/test (including non-eligible ingredients)
    heldout = set(test_ids) | set(val_ids)
    train_ids = sorted([nid for nid in fg_data.ingredients if nid not in heldout])

    logger.info(
        f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test "
        f"(from {len(eligible)} eligible, {len(fg_data.ingredients)} total)"
    )

    # Build ground truth and training edges
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    split = IngredientSplit(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    for e in fg_data.ingr_ingr_edges:
        score = e.score if e.score is not None else 0.0
        src_in_train = e.src in train_set
        dst_in_train = e.dst in train_set

        if src_in_train and dst_in_train:
            # Both endpoints in train -> training edge
            split.train_edges.append((e.src, e.dst))
            split.train_edge_weights.append(score)
        else:
            # At least one endpoint is held out -> ground truth
            if e.src in val_set or e.dst in val_set:
                split.val_ground_truth[(e.src, e.dst)] = score
            if e.src in test_set or e.dst in test_set:
                split.test_ground_truth[(e.src, e.dst)] = score

    logger.info(
        f"Edges: {len(split.train_edges)} train, "
        f"{len(split.val_ground_truth)} val ground-truth, "
        f"{len(split.test_ground_truth)} test ground-truth"
    )

    return split


def verify_no_leakage(
    split: IngredientSplit,
    fg_data,  # FlavorGraphData
) -> List[str]:
    """Verify that the split has no information leakage.

    Checks:
    1. No training edges touch val/test ingredients.
    2. Val/test sets are disjoint from each other and from train.
    3. All val/test ingredients have ground-truth edges.

    Returns:
        List of violation descriptions. Empty list = no leakage.
    """
    violations = []

    train_set = set(split.train_ids)
    val_set = set(split.val_ids)
    test_set = set(split.test_ids)

    # Check disjointness
    if train_set & val_set:
        violations.append(
            f"LEAK: {len(train_set & val_set)} ingredients in both train and val"
        )
    if train_set & test_set:
        violations.append(
            f"LEAK: {len(train_set & test_set)} ingredients in both train and test"
        )
    if val_set & test_set:
        violations.append(
            f"LEAK: {len(val_set & test_set)} ingredients in both val and test"
        )

    # Check no training edges touch held-out ingredients
    heldout = val_set | test_set
    for src, dst in split.train_edges:
        if src in heldout:
            violations.append(f"LEAK: train edge src {src} is in held-out set")
            break
        if dst in heldout:
            violations.append(f"LEAK: train edge dst {dst} is in held-out set")
            break

    # Check coverage: val/test ingredients should have ground-truth edges
    val_gt_nodes = set()
    for (s, d) in split.val_ground_truth:
        val_gt_nodes.add(s)
        val_gt_nodes.add(d)
    val_no_gt = val_set - val_gt_nodes
    if val_no_gt:
        violations.append(
            f"WARNING: {len(val_no_gt)} val ingredients have no ground-truth edges"
        )

    test_gt_nodes = set()
    for (s, d) in split.test_ground_truth:
        test_gt_nodes.add(s)
        test_gt_nodes.add(d)
    test_no_gt = test_set - test_gt_nodes
    if test_no_gt:
        violations.append(
            f"WARNING: {len(test_no_gt)} test ingredients have no ground-truth edges"
        )

    # Check completeness: all ingredients accounted for
    all_split = train_set | val_set | test_set
    all_ingr = set(fg_data.ingredients.keys())
    missing = all_ingr - all_split
    if missing:
        violations.append(
            f"WARNING: {len(missing)} ingredients not in any split"
        )
    extra = all_split - all_ingr
    if extra:
        violations.append(
            f"ERROR: {len(extra)} split IDs not in ingredient list"
        )

    if violations:
        for v in violations:
            logger.warning(v)
    else:
        logger.info("No leakage detected")

    return violations


def cold_mask_training(
    split: IngredientSplit,
    mask_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[Set[int], List[Tuple[int, int]], List[float], Dict[Tuple[int, int], float]]:
    """Simulate cold-start during training by masking a random subset.

    Each epoch, randomly select mask_frac of training ingredients, remove
    ALL their edges, and use those edges as additional training signal.
    This teaches the model to rely on pseudo-edges when real edges vanish.

    Args:
        split: The ingredient split.
        mask_frac: Fraction of train ingredients to mask.
        seed: Random seed (should vary per epoch).

    Returns:
        Tuple of:
        - masked_ids: Set of masked ingredient node_ids
        - remaining_edges: Training edges with masked ones removed
        - remaining_weights: Weights for remaining edges
        - masked_ground_truth: Edges of masked ingredients (for loss)
    """
    rng = random.Random(seed)
    n_mask = max(1, int(len(split.train_ids) * mask_frac))
    masked_ids = set(rng.sample(split.train_ids, n_mask))

    remaining_edges = []
    remaining_weights = []
    masked_ground_truth = {}

    for (src, dst), weight in zip(split.train_edges, split.train_edge_weights):
        if src in masked_ids or dst in masked_ids:
            masked_ground_truth[(src, dst)] = weight
        else:
            remaining_edges.append((src, dst))
            remaining_weights.append(weight)

    return masked_ids, remaining_edges, remaining_weights, masked_ground_truth
