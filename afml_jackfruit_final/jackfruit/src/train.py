"""Training and evaluation pipeline for SavorLink Strategy III.

Features:
- Real walk-derived edges from metapath corpus (CHNHC + NHCHN + RandWalk).
- Cold-start-aware training: random masking of training ingredients each epoch.
- Negative sampling: random non-edge pairs sampled as negatives per epoch.
- Validation loop + early stopping on val loss (patience-based).
- ReduceLROnPlateau scheduler tied to val loss.
- Best model checkpoint saved to disk and returned.
- Multi-seed execution reporting mean +- std across all 5 baselines:
    1. Random
    2. Popularity (degree)
    3. Tanimoto-kNN (non-GNN chemical kNN)
    4. Zero-degree GNN (failure mode)
    5. Full Strategy III
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# PyTorch import with informative error message
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    torch = None

from src.baselines import GenericKNNBaseline, PopularityBaseline, RandomBaseline, TanimotoKNNBaseline
from src.chem.fingerprints import compute_compound_fingerprints, pool_mean, pool_union
from src.chem.similarity import compute_sparse_topk_similarity
from src.data.loaders import FlavorGraphData, load_config, load_flavorgraph, resolve_path
from src.data.pubchem_lookup import PubChemCache, build_smiles_cache
from src.data.splits import IngredientSplit, cold_mask_training, create_ingredient_split, verify_no_leakage
from src.data.walk_loader import inject_walk_edges
from src.evaluate import aggregate_metrics, evaluate_ranking_for_ingredient, format_metrics_table
from src.graph.build_graph import assemble_feature_matrix, build_node_id_mapping
from src.graph.pseudo_edges import build_pseudo_edges, merge_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> "torch.device":
    """Resolve 'auto', 'cuda', 'cpu' to an actual torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed: int) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def sample_negatives(
    positive_pairs: List[Tuple[int, int]],
    all_node_indices: List[int],
    edge_set: Set[Tuple[int, int]],
    neg_ratio: int = 1,
    rng: random.Random = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Sample random non-edge pairs as negatives.

    For each positive pair we sample `neg_ratio` negatives: random (u, v)
    pairs that are NOT in `edge_set`.  Target score for all negatives is 0.0.

    Args:
        positive_pairs: List of positive (src_idx, dst_idx) pairs.
        all_node_indices: Pool of node indices to sample from.
        edge_set: Set of canonical (min, max) pairs that are real edges.
        neg_ratio: Negatives per positive.
        rng: Optional random.Random instance for reproducibility.

    Returns:
        (neg_pairs, neg_targets) — lists of the same length.
    """
    if rng is None:
        rng = random.Random()

    n_neg = len(positive_pairs) * neg_ratio
    neg_pairs: List[Tuple[int, int]] = []
    max_attempts = n_neg * 10
    attempts = 0

    while len(neg_pairs) < n_neg and attempts < max_attempts:
        u = rng.choice(all_node_indices)
        v = rng.choice(all_node_indices)
        if u == v:
            attempts += 1
            continue
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            neg_pairs.append((u, v))
        attempts += 1

    neg_targets = [0.0] * len(neg_pairs)
    return neg_pairs, neg_targets


# ---------------------------------------------------------------------------
# Feature + similarity pre-computation (shared across all seeds)
# ---------------------------------------------------------------------------

def prepare_features_and_similarities(
    fg: FlavorGraphData,
    config: dict,
    project_root: Path,
    smoke: bool = False,
    modality: str = "molecular",
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, float]]], Dict[int, int], Dict[int, int]]:
    """Compute or load node features and top-k similarities for specified modality."""
    all_ingr_ids = sorted(fg.ingredients.keys())
    raw_to_idx, idx_to_raw = build_node_id_mapping(all_ingr_ids)
    cache_dir = project_root / "data" / "cache"

    mod = modality.lower()
    if mod == "semantic":
        feat_path = cache_dir / "semantic_features.npy"
        sim_path = cache_dir / "semantic_similarities.pkl"
    elif mod == "taxonomic":
        feat_path = cache_dir / "taxonomic_features.npy"
        sim_path = cache_dir / "taxonomic_similarities.pkl"
    elif mod == "hybrid":
        feat_path = cache_dir / "hybrid_features.npy"
        sim_path = cache_dir / "hybrid_similarities.pkl"
    else:
        feat_path = cache_dir / "node_features.npy"
        sim_path = cache_dir / "topk_similarities.pkl"

    if feat_path.exists() and sim_path.exists():
        logger.info(f"Loading precomputed {modality} features from {feat_path} and similarities from {sim_path}...")
        x_matrix = np.load(feat_path)
        with open(sim_path, "rb") as f:
            topk_similarities = pickle.load(f)
        return x_matrix, topk_similarities, raw_to_idx, idx_to_raw

    # Fallback to computing molecular features if cache is missing
    cache_path = resolve_path(config["paths"]["pubchem_cache"], project_root)
    cache = PubChemCache(cache_path)

    # 1. Gather PubChem CIDs for all compounds
    comp_cids = {c.pubchem_cid for c in fg.compounds.values() if c.pubchem_cid is not None}
    logger.info(f"Checking SMILES cache for {len(comp_cids)} compound CIDs...")
    build_smiles_cache(comp_cids, cache, config["pubchem"])

    valid_smiles_count = cache.valid_count()
    if valid_smiles_count == 0 and not smoke:
        raise RuntimeError(
            f"PubChem SMILES cache at {cache_path} has 0 valid entries. "
            "Cannot compute fingerprints. Run with network access or populate the cache manually."
        )
    elif 0 < valid_smiles_count < len(comp_cids) * 0.5:
        logger.warning(
            f"Only {valid_smiles_count}/{len(comp_cids)} CIDs have valid SMILES — "
            "fingerprint coverage is <50%. Strategy III pseudo-edges may be sparse."
        )

    # If offline / smoke test without cache, populate representative flavor compound SMILES
    if smoke and cache.valid_count() == 0:
        logger.info("[Smoke Mode] Populating offline fallback SMILES for compound CIDs...")
        palette = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCOC(=O)C", "c1ccc(O)cc1", "CCCC(=O)OCC"]
        for i, cid in enumerate(comp_cids):
            cache.put(cid, palette[i % len(palette)])

    # 2. Compute fingerprints
    cid_to_smiles = {cid: cache.get(cid) for cid in comp_cids if cache.get(cid) is not None}
    cfg_fp = config.get("fingerprints", {})
    cid_to_fp = compute_compound_fingerprints(
        cid_to_smiles,
        fp_type=cfg_fp.get("type", "morgan"),
        radius=cfg_fp.get("radius", 2),
        n_bits=cfg_fp.get("n_bits", 2048),
    )

    node_id_to_cid = {
        nid: node.pubchem_cid for nid, node in fg.compounds.items() if node.pubchem_cid is not None
    }

    # 3. Pre-pool ingredient fingerprints
    strategy = config.get("similarity", {}).get("strategy", "mean_best_match")
    pooled_fps: Dict[int, np.ndarray] = {}
    for ingr_id, comp_ids in fg.ingredient_compounds.items():
        fps = [
            cid_to_fp[node_id_to_cid[cid]]
            for cid in comp_ids
            if cid in node_id_to_cid and node_id_to_cid[cid] in cid_to_fp
        ]
        if fps:
            pooled_fps[ingr_id] = pool_mean(fps)

    # 4. Assemble node ID mapping and feature matrix
    all_ingr_ids = sorted(fg.ingredients.keys())
    raw_to_idx, idx_to_raw = build_node_id_mapping(all_ingr_ids)
    x_matrix = assemble_feature_matrix(
        node_ids=all_ingr_ids,
        raw_to_idx=raw_to_idx,
        pooled_fps=pooled_fps,
        n_bits=cfg_fp.get("n_bits", 2048),
    )

    # 5. Pre-compute top-k chemical similarities against hub ingredients
    hub_ids = sorted(fg.hub_ingredients)
    cfg_sim = config.get("similarity", {})
    sim_cache_path = str(resolve_path(
        config.get("paths", {}).get("similarity_cache", "data/cache/topk_similarities.pkl"),
        project_root,
    ))
    logger.info(f"Computing sparse chemical top-k similarities against {len(hub_ids)} hub ingredients...")
    topk_similarities = compute_sparse_topk_similarity(
        query_ids=all_ingr_ids,
        candidate_ids=hub_ids,
        ingredient_compounds=fg.ingredient_compounds,
        node_id_to_cid=node_id_to_cid,
        cid_to_fp=cid_to_fp,
        strategy=strategy,
        top_k=cfg_sim.get("top_k", 10),
        threshold=cfg_sim.get("threshold", 0.05),
        batch_size=cfg_sim.get("batch_size", 256),
        pooled_fps=pooled_fps,
        cache_path=None if smoke else sim_cache_path,
    )

    return x_matrix, topk_similarities, raw_to_idx, idx_to_raw


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def train_single_seed(
    fg: FlavorGraphData,
    split: IngredientSplit,
    x_matrix: np.ndarray,
    topk_similarities: Dict[int, List[Tuple[int, float]]],
    raw_to_idx: Dict[int, int],
    config: dict,
    seed: int,
    device: str = "auto",
    checkpoint_dir: Optional[Union[Path, str]] = None,
    export_dir: Optional[Union[Path, str]] = None,
    modality: str = "molecular",
    loss_type: str = "bpr",
) -> Tuple[Dict[str, Dict[str, float]], object]:
    """Train Strategy III and evaluate all 5 baselines on the test split for one seed.

    Args:
        fg:                  Loaded FlavorGraphData.
        split:               Train/val/test ingredient split.
        x_matrix:            Node feature matrix (N, n_bits) float32.
        topk_similarities:   Pre-computed top-k chemical similarities.
        raw_to_idx:          Mapping from raw node ID to contiguous index.
        config:              Full config dict.
        seed:                Random seed for this run.
        device:              'auto', 'cuda', or 'cpu'.
        checkpoint_dir:      Directory to save best model checkpoint.
        export_dir:          Alias for checkpoint directory base.

    Returns:
        (seed_summary, best_model): Metric dict and best-epoch model instance.
    """
    from src.models.strategy3 import Strategy3Model

    if checkpoint_dir is None and export_dir is not None:
        checkpoint_dir = Path(export_dir) / "checkpoints"

    set_seed(seed)
    dev = resolve_device(device)
    logger.info(f"[Seed {seed}] Using device: {dev}")

    cfg_train = config.get("training", {})
    cfg_pseudo = config.get("pseudo_edges", {})
    epochs = cfg_train.get("epochs", 100)
    lr = float(cfg_train.get("lr", 0.001))
    weight_decay = float(cfg_train.get("weight_decay", 1e-5))
    patience = int(cfg_train.get("patience", 20))
    mask_frac = float(cfg_train.get("cold_mask_frac", 0.1))
    neg_ratio = int(cfg_train.get("neg_sample_ratio", 1))

    # Convert feature matrix to PyTorch tensor
    x_tensor = torch.from_numpy(x_matrix).to(dev)

    # Build a set of real edge pairs (canonical) for negative sampling
    all_ingr_indices = list(raw_to_idx.values())
    real_edge_set: Set[Tuple[int, int]] = set()
    for (s, d) in split.train_edges:
        si, di = raw_to_idx[s], raw_to_idx[d]
        real_edge_set.add((min(si, di), max(si, di)))

    # Determine pool for negative sampling
    neg_pool_mode = cfg_train.get("negative_pool", "hubs")
    if neg_pool_mode == "hubs":
        hub_raws = set(fg.hub_ingredients)
        neg_candidate_indices = [idx for raw, idx in raw_to_idx.items() if raw in hub_raws]
        if not neg_candidate_indices:
            neg_candidate_indices = all_ingr_indices
    else:
        neg_candidate_indices = all_ingr_indices

    # Pre-map all training edges into contiguous int64 NumPy arrays once (vectorization)
    t_prep = time.time()
    num_nodes = len(raw_to_idx)
    idx_to_raw = {idx: raw for raw, idx in raw_to_idx.items()}

    if split.train_edges:
        train_src_arr = np.array([raw_to_idx[s] for s, d in split.train_edges], dtype=np.int64)
        train_dst_arr = np.array([raw_to_idx[d] for s, d in split.train_edges], dtype=np.int64)
        train_w_arr = np.array(split.train_edge_weights, dtype=np.float32)
    else:
        train_src_arr = np.zeros(0, dtype=np.int64)
        train_dst_arr = np.zeros(0, dtype=np.int64)
        train_w_arr = np.zeros(0, dtype=np.float32)

    train_node_indices = [raw_to_idx[nid] for nid in split.train_ids]
    logger.info(
        f"[Seed {seed}] Pre-indexed {len(train_src_arr):,} training edges in {time.time() - t_prep:.2f}s."
    )

    # Instantiate model with dynamic input feature dimension
    in_dim = x_matrix.shape[1]
    model = Strategy3Model(config, in_dim=in_dim).to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 3, min_lr=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state_dict = None
    rng = random.Random(seed)

    logger.info(f"[Seed {seed}] Training [{modality.upper()} | {loss_type.upper()} Loss] for up to {epochs} epochs (patience={patience})...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        # ------------------------------------------------------------------ #
        # TRAIN STEP (Fully Vectorized)
        # ------------------------------------------------------------------ #
        model.train()
        optimizer.zero_grad()

        # 1. Vectorized cold-masking: sample a subset of training ingredient indices
        n_mask = max(1, int(len(train_node_indices) * mask_frac))
        masked_indices = rng.sample(train_node_indices, n_mask)
        masked_raw_ids = [idx_to_raw[idx] for idx in masked_indices]

        # Fast O(E) boolean mask using integer indexing
        mask_lut = np.zeros(num_nodes, dtype=bool)
        mask_lut[masked_indices] = True
        is_masked = mask_lut[train_src_arr] | mask_lut[train_dst_arr]

        rem_src = train_src_arr[~is_masked]
        rem_dst = train_dst_arr[~is_masked]
        rem_w = train_w_arr[~is_masked]

        masked_src = train_src_arr[is_masked]
        masked_dst = train_dst_arr[is_masked]
        masked_w = train_w_arr[is_masked]

        # 2. Build pseudo-edges for masked (cold-simulated) ingredients
        p_edges, p_weights = build_pseudo_edges(
            cold_node_ids=masked_raw_ids,
            similarity_topk=topk_similarities,
            top_k=cfg_pseudo.get("top_k", 10),
            temperature=cfg_pseudo.get("temperature", 1.0),
            threshold=cfg_pseudo.get("threshold", 0.05),
            include_reverse=cfg_pseudo.get("include_reverse", True),
        )

        if p_edges:
            p_src = np.array([raw_to_idx[s] for s, d in p_edges], dtype=np.int64)
            p_dst = np.array([raw_to_idx[d] for s, d in p_edges], dtype=np.int64)
            p_w = np.array(p_weights, dtype=np.float32)
            merged_src = np.concatenate([rem_src, p_src])
            merged_dst = np.concatenate([rem_dst, p_dst])
            merged_w = np.concatenate([rem_w, p_w])
        else:
            merged_src, merged_dst, merged_w = rem_src, rem_dst, rem_w

        edge_index_t = torch.from_numpy(np.stack([merged_src, merged_dst])).to(dev)
        edge_weight_t = torch.from_numpy(merged_w).to(dev)

        # 3. Build positive training targets: masked true edges + sample of remaining edges
        masked_cnt = len(masked_src)
        sample_size = min(len(rem_src), max(masked_cnt, cfg_train.get("batch_size", 512)))
        if sample_size > 0 and len(rem_src) > 0:
            samp_idx = rng.sample(range(len(rem_src)), sample_size)
            pos_src = np.concatenate([masked_src, rem_src[samp_idx]])
            pos_dst = np.concatenate([masked_dst, rem_dst[samp_idx]])
            pos_targets = np.concatenate([masked_w, rem_w[samp_idx]])
        else:
            pos_src = masked_src
            pos_dst = masked_dst
            pos_targets = masked_w

        if len(pos_src) == 0:
            continue

        # 4. Sample negatives
        pos_pairs = list(zip(pos_src, pos_dst))
        neg_pairs, neg_targets = sample_negatives(
            pos_pairs, neg_candidate_indices, real_edge_set, neg_ratio=neg_ratio, rng=rng
        )

        # 5. Forward pass + loss
        if loss_type == "bpr":
            pos_pairs_np = np.stack([pos_src, pos_dst])
            pos_pairs_t = torch.from_numpy(pos_pairs_np).to(dev)
            neg_src_np = np.array([p[0] for p in neg_pairs], dtype=np.int64)
            neg_dst_np = np.array([p[1] for p in neg_pairs], dtype=np.int64)
            neg_pairs_t = torch.from_numpy(np.stack([neg_src_np, neg_dst_np])).to(dev)

            pos_scores = model(
                x_tensor,
                edge_index_graph=edge_index_t,
                edge_weight_graph=edge_weight_t,
                eval_edges=pos_pairs_t,
            )
            neg_scores = model(
                x_tensor,
                edge_index_graph=edge_index_t,
                edge_weight_graph=edge_weight_t,
                eval_edges=neg_pairs_t,
            )
            train_loss = model.compute_loss(loss_type="bpr", pos_scores=pos_scores, neg_scores=neg_scores)
        else:
            all_src = np.concatenate([pos_src, np.array([p[0] for p in neg_pairs], dtype=np.int64)])
            all_dst = np.concatenate([pos_dst, np.array([p[1] for p in neg_pairs], dtype=np.int64)])
            all_targets = np.concatenate([pos_targets, np.array(neg_targets, dtype=np.float32)])

            pairs_t = torch.from_numpy(np.stack([all_src, all_dst])).to(dev)
            targets_t = torch.from_numpy(all_targets).to(dev)

            pred_scores = model(
                x_tensor,
                edge_index_graph=edge_index_t,
                edge_weight_graph=edge_weight_t,
                eval_edges=pairs_t,
            )
            train_loss = model.compute_loss(pred_scores, targets_t)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ------------------------------------------------------------------ #
        # VALIDATION STEP
        # ------------------------------------------------------------------ #
        model.eval()
        val_loss_val = float("nan")

        if split.val_ground_truth:
            val_pairs: List[Tuple[int, int]] = []
            val_targets_list: List[float] = []

            for (s, d), w in split.val_ground_truth.items():
                if s in raw_to_idx and d in raw_to_idx:
                    val_pairs.append((raw_to_idx[s], raw_to_idx[d]))
                    val_targets_list.append(float(w))

            # Val negatives
            val_neg_pairs, val_neg_targets = sample_negatives(
                val_pairs, neg_candidate_indices, real_edge_set, neg_ratio=1, rng=rng
            )

            if val_pairs:
                # Inference graph for validation: all train edges + pseudo-edges for val nodes
                val_pseudo_e, val_pseudo_w = build_pseudo_edges(
                    cold_node_ids=split.val_ids,
                    similarity_topk=topk_similarities,
                    top_k=cfg_pseudo.get("top_k", 10),
                    temperature=cfg_pseudo.get("temperature", 1.0),
                    threshold=cfg_pseudo.get("threshold", 0.05),
                    include_reverse=cfg_pseudo.get("include_reverse", True),
                )
                train_ei = [(raw_to_idx[s], raw_to_idx[d]) for s, d in split.train_edges]
                val_pe_idx = [(raw_to_idx[s], raw_to_idx[d]) for s, d in val_pseudo_e]
                val_ei_np, val_ew_np, _ = merge_edges(
                    train_ei, split.train_edge_weights, val_pe_idx, val_pseudo_w
                )
                val_ei_t = torch.from_numpy(val_ei_np).to(dev)
                val_ew_t = torch.from_numpy(val_ew_np).to(dev)

                with torch.no_grad():
                    if loss_type == "bpr":
                        val_pos_t = torch.tensor(val_pairs, dtype=torch.long, device=dev).t()
                        val_neg_t = torch.tensor(val_neg_pairs, dtype=torch.long, device=dev).t()
                        val_pos_scores = model(
                            x_tensor, edge_index_graph=val_ei_t, edge_weight_graph=val_ew_t, eval_edges=val_pos_t
                        )
                        val_neg_scores = model(
                            x_tensor, edge_index_graph=val_ei_t, edge_weight_graph=val_ew_t, eval_edges=val_neg_t
                        )
                        val_loss_t = model.compute_loss(loss_type="bpr", pos_scores=val_pos_scores, neg_scores=val_neg_scores)
                    else:
                        val_all_pairs = val_pairs + val_neg_pairs
                        val_all_targets = val_targets_list + val_neg_targets
                        val_p_t = torch.tensor(val_all_pairs, dtype=torch.long, device=dev).t()
                        val_tgt_t = torch.tensor(val_all_targets, dtype=torch.float32, device=dev)
                        val_pred = model(
                            x_tensor,
                            edge_index_graph=val_ei_t,
                            edge_weight_graph=val_ew_t,
                            eval_edges=val_p_t,
                        )
                        val_loss_t = model.compute_loss(val_pred, val_tgt_t)
                val_loss_val = val_loss_t.item()

            # Early stopping + LR scheduling
            scheduler.step(val_loss_val)
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_epoch = epoch
                epochs_no_improve = 0
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

        is_new_best = (val_loss_val == best_val_loss and not np.isnan(val_loss_val))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or is_new_best:
            elapsed = time.time() - t_start
            epoch_duration = time.time() - t_epoch_start
            lr_now = optimizer.param_groups[0]["lr"]
            star = " *" if is_new_best else ""
            msg = (
                f"  Epoch {epoch:03d}/{epochs:03d}{star} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss_val:.4f} | "
                f"Best Val: {best_val_loss:.4f} (ep {best_epoch}) | "
                f"LR: {lr_now:.2e} | "
                f"Epoch Time: {epoch_duration * 1000:.1f}ms | "
                f"Total: {elapsed:.0f}s"
            )
            logger.info(msg)
            print(f"[Seed {seed}] {msg}", flush=True)

        if epochs_no_improve >= patience:
            logger.info(f"  [Seed {seed}] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state_dict.items()})
        logger.info(f"[Seed {seed}] Restored best model from epoch {best_epoch} (val loss={best_val_loss:.4f})")

    # Save checkpoint
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"best_model_seed{seed}.pt"
        torch.save({
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": config,
            "seed": seed,
        }, ckpt_path)
        logger.info(f"[Seed {seed}] Checkpoint saved to {ckpt_path}")

    # ======================================================================
    # Evaluation on Test Split
    # ======================================================================
    logger.info(f"[Seed {seed}] Evaluating on {len(split.test_ids)} test ingredients...")
    model.eval()

    # Build inference graph: all training edges + pseudo-edges for test ingredients
    test_pseudo_edges, test_pseudo_weights = build_pseudo_edges(
        cold_node_ids=split.test_ids,
        similarity_topk=topk_similarities,
        top_k=cfg_pseudo.get("top_k", 10),
        temperature=cfg_pseudo.get("temperature", 1.0),
        threshold=cfg_pseudo.get("threshold", 0.05),
        include_reverse=cfg_pseudo.get("include_reverse", True),
    )

    train_edges_idx = [(raw_to_idx[s], raw_to_idx[d]) for s, d in split.train_edges]
    test_p_edges_idx = [(raw_to_idx[s], raw_to_idx[d]) for s, d in test_pseudo_edges]

    # Full Strategy III inference graph (with pseudo-edges)
    ei_s3_np, ew_s3_np, _ = merge_edges(
        train_edges_idx, split.train_edge_weights, test_p_edges_idx, test_pseudo_weights
    )
    ei_s3_t = torch.from_numpy(ei_s3_np).to(dev)
    ew_s3_t = torch.from_numpy(ew_s3_np).to(dev)

    # Zero-degree GNN inference graph (NO pseudo-edges — failure mode baseline)
    ei_zero_np, ew_zero_np, _ = merge_edges(train_edges_idx, split.train_edge_weights, [], [])
    ei_zero_t = torch.from_numpy(ei_zero_np).to(dev)
    ew_zero_t = torch.from_numpy(ew_zero_np).to(dev)

    with torch.no_grad():
        emb_s3 = model.encode(x_tensor, ei_s3_t, ew_s3_t)
        emb_zero = model.encode(x_tensor, ei_zero_t, ew_zero_t)

    # Initialize baselines
    random_bl = RandomBaseline(seed=seed)
    pop_bl = PopularityBaseline(split.train_edges, split.train_edge_weights)
    knn_bl = GenericKNNBaseline(
        split.train_edges, split.train_edge_weights, topk_similarities,
        top_k=cfg_pseudo.get("top_k", 10)
    )

    # Candidate ranking pool selection
    cand_pool_mode = config.get("evaluation", {}).get("candidate_pool", "hubs")
    if cand_pool_mode == "hubs":
        candidate_ids = sorted([cid for cid in fg.hub_ingredients if cid in raw_to_idx])
        if not candidate_ids:
            candidate_ids = sorted(fg.ingredients.keys())
    else:
        candidate_ids = sorted(fg.ingredients.keys())

    candidate_set = set(candidate_ids)
    # Group test ground truth by test ingredient, filtered to evaluated candidate pool
    test_gt_by_ingr: Dict[int, Dict[int, float]] = defaultdict(dict)
    for (s, d), w in split.test_ground_truth.items():
        if s in set(split.test_ids) and d in candidate_set:
            test_gt_by_ingr[s][d] = w
        if d in set(split.test_ids) and s in candidate_set:
            test_gt_by_ingr[d][s] = w

    cand_indices_t = torch.tensor(
        [raw_to_idx[cid] for cid in candidate_ids], dtype=torch.long, device=dev
    )

    knn_label = f"{modality.capitalize()}-kNN"
    s3_label = f"Strategy III ({modality.capitalize()})"

    model_metrics: Dict[str, List[Dict[str, float]]] = {
        "Random": [], "Popularity": [], knn_label: [],
        "Zero-degree GNN": [], s3_label: [],
    }

    with torch.no_grad():
        for cold_id in split.test_ids:
            gt = test_gt_by_ingr.get(cold_id, {})
            if not gt:
                continue

            cold_idx = raw_to_idx[cold_id]
            eval_cands = [cid for cid in candidate_ids if cid != cold_id]

            # 1. Random
            model_metrics["Random"].append(
                evaluate_ranking_for_ingredient(random_bl.predict(cold_id, eval_cands), gt)
            )
            # 2. Popularity
            model_metrics["Popularity"].append(
                evaluate_ranking_for_ingredient(pop_bl.predict(cold_id, eval_cands), gt)
            )
            # 3. Modality-specific kNN
            model_metrics[knn_label].append(
                evaluate_ranking_for_ingredient(knn_bl.predict(cold_id, eval_cands), gt)
            )
            # 4. Zero-degree GNN
            h_q_zero = emb_zero[cold_idx]
            scores_zero = model.head.score_all_pairs_for_node(h_q_zero, emb_zero[cand_indices_t])
            pred_zero = {cid: float(scores_zero[i].item()) for i, cid in enumerate(candidate_ids) if cid != cold_id}
            model_metrics["Zero-degree GNN"].append(evaluate_ranking_for_ingredient(pred_zero, gt))

            # 5. Full Strategy III
            h_q_s3 = emb_s3[cold_idx]
            scores_s3 = model.head.score_all_pairs_for_node(h_q_s3, emb_s3[cand_indices_t])
            pred_s3 = {cid: float(scores_s3[i].item()) for i, cid in enumerate(candidate_ids) if cid != cold_id}
            model_metrics[s3_label].append(evaluate_ranking_for_ingredient(pred_s3, gt))

    seed_summary = {name: aggregate_metrics(m_list) for name, m_list in model_metrics.items()}
    return seed_summary, model


# ---------------------------------------------------------------------------
# Export helpers (for Colab download / persistent storage)
# ---------------------------------------------------------------------------

def export_artifacts(
    model: "torch.nn.Module",
    fg: FlavorGraphData,
    split: IngredientSplit,
    topk_similarities: Dict[int, List[Tuple[int, float]]],
    raw_to_idx: Dict[int, int],
    idx_to_raw: Optional[Dict[int, int]] = None,
    config: dict = None,
    seed: int = 42,
    export_dir: Path = "data/exports",
    x_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    """Save all artifacts needed for downstream inference.

    Saves:
    - model weights as .pt (PyTorch checkpoint)
    - model + metadata as .pkl (full pickle for easy reload)
    - graph metadata (node mappings, ingredient names) as .pkl
    - topk_similarities as .pkl
    - split as .pkl
    - node_features.npy (if x_matrix provided)
    - inference_graph.npz (training graph edge_index & edge_weight)

    Returns dict of artifact_name -> Path.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    if idx_to_raw is None and raw_to_idx is not None:
        idx_to_raw = {idx: raw for raw, idx in raw_to_idx.items()}
    saved = {}

    # 1. PyTorch model weights (.pt)
    pt_path = export_dir / f"model_seed{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "seed": seed,
    }, pt_path)
    saved["model_pt"] = pt_path

    # 2. Full model pickle (.pkl) — includes model object for easy reload
    pkl_path = export_dir / f"model_seed{seed}.pkl"
    model_cpu = model.cpu()
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model": model_cpu,
            "config": config,
            "seed": seed,
        }, f)
    saved["model_pkl"] = pkl_path

    # 3. Graph metadata
    meta_path = export_dir / "graph_metadata.pkl"
    ingredient_names = {nid: info.name for nid, info in fg.ingredients.items()}
    with open(meta_path, "wb") as f:
        pickle.dump({
            "raw_to_idx": raw_to_idx,
            "idx_to_raw": idx_to_raw,
            "ingredient_names": ingredient_names,
            "hub_ingredients": list(fg.hub_ingredients),
            "num_ingredients": len(fg.ingredients),
        }, f)
    saved["graph_metadata"] = meta_path

    # 4. Chemical similarities
    sim_path = export_dir / "topk_similarities.pkl"
    with open(sim_path, "wb") as f:
        pickle.dump(topk_similarities, f)
    saved["topk_similarities"] = sim_path

    # 5. Split
    split_path = export_dir / f"split_seed{seed}.pkl"
    with open(split_path, "wb") as f:
        pickle.dump(split, f)
    saved["split"] = split_path

    # 6. Node features (for standalone inference)
    if x_matrix is not None:
        feat_path = export_dir / "node_features.npy"
        np.save(feat_path, x_matrix.astype(np.float32))
        saved["node_features"] = feat_path

    # 7. Training graph topology (for standalone GNN aggregation at inference time)
    train_edges_idx = [(raw_to_idx[s], raw_to_idx[d]) for s, d in split.train_edges]
    if train_edges_idx:
        src_list = [e[0] for e in train_edges_idx]
        dst_list = [e[1] for e in train_edges_idx]
        edge_index_np = np.array([src_list, dst_list], dtype=np.int64)
        edge_weight_np = np.array(split.train_edge_weights, dtype=np.float32)
    else:
        edge_index_np = np.zeros((2, 0), dtype=np.int64)
        edge_weight_np = np.zeros(0, dtype=np.float32)

    graph_path = export_dir / "inference_graph.npz"
    np.savez_compressed(graph_path, edge_index=edge_index_np, edge_weight=edge_weight_np)
    saved["inference_graph"] = graph_path

    logger.info(f"Exported {len(saved)} artifacts to {export_dir}:")
    for name, path in saved.items():
        size_mb = path.stat().st_size / 1e6
        logger.info(f"  {name}: {path.name} ({size_mb:.1f} MB)")

    return saved


def create_export_zip(
    export_dir: Path | str,
    zip_path: Optional[Path | str] = None,
) -> Path:
    """Bundle all files in export_dir into a single portable zip archive.

    Ideal for 1-click Google Colab download.
    """
    export_dir = Path(export_dir)
    if zip_path is None:
        zip_path = export_dir.parent / "savorlink_model_exports.zip"
    else:
        zip_path = Path(zip_path)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in export_dir.rglob("*"):
            if file.is_file() and file.name != zip_path.name:
                arcname = file.relative_to(export_dir)
                zf.write(file, arcname=arcname)

    size_mb = zip_path.stat().st_size / 1e6
    logger.info(f"Bundled export archive: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str,
    seeds: List[int] = (42, 123, 456),
    export_dir: Optional[str] = None,
    smoke: bool = False,
    hub_only: Optional[bool] = None,
    modality: str = "molecular",
    loss: str = "bpr",
) -> Dict:
    """Run full Strategy III pipeline across multiple seeds.

    Args:
        config_path: Path to YAML config file.
        seeds:       List of random seeds to run.
        export_dir:  If set, save model artifacts here after each seed.
        smoke:       If True, run a fast 2-epoch, 1-seed smoke verification.
        hub_only:    If True, restrict to 416 hub ingredients with chemistry.
        modality:    One of 'molecular', 'semantic', 'hybrid', 'taxonomic'.
        loss:        'bpr' (Pairwise BPR Ranking Loss) or 'huber' (Pointwise Huber Regression).

    Returns:
        Dict with 'final_table', 'export_paths', 'zip_archive', and 'wall_clock_seconds'.
    """
    pipeline_t0 = time.time()
    cfg = load_config(config_path)
    root = PROJECT_ROOT

    if hub_only is not None:
        cfg["data"]["hub_only"] = hub_only

    if smoke:
        logger.info("=" * 65)
        logger.info(f"=== SMOKE RUN MODE [{modality.upper()} | {loss.upper()}] (Fast 2-Epoch Verification) ===")
        logger.info("=" * 65)
        seeds = [seeds[0]] if seeds else [42]
        cfg["training"]["epochs"] = 2
        cfg["training"]["patience"] = 2

    logger.info("=" * 65)
    logger.info(f"SavorLink Strategy III — [{modality.upper()} | {loss.upper()}] Multi-Seed Benchmark")
    logger.info("=" * 65)

    # Load FlavorGraph
    fg = load_flavorgraph(cfg, root)

    # Inject walk-derived edges if configured
    walk_path_raw = cfg.get("paths", {}).get("walk_corpus")
    if walk_path_raw:
        walk_path = resolve_path(walk_path_raw, root)
        if walk_path.exists():
            cfg_walks = cfg.get("walks", {})
            max_lines = 5000 if smoke else None
            logger.info(f"Loading metapath walk corpus: {walk_path.name} (smoke={smoke})")
            fg = inject_walk_edges(
                fg=fg,
                walk_path=walk_path,
                window_size=cfg_walks.get("window_size", 5),
                min_count=cfg_walks.get("min_count", 2),
                score_mode=cfg_walks.get("score_mode", "pmi"),
                replace_csv_edges=cfg_walks.get("replace_csv_edges", True),
                max_lines=max_lines,
                cache_dir=cfg.get("paths", {}).get("cache_dir", "data/cache"),
            )
            logger.info(f"Walk edges loaded: {len(fg.ingr_ingr_edges):,} ingr-ingr edges in graph.")
        else:
            logger.warning(f"walk_corpus not found: {walk_path}. Falling back to CSV edges.")
    else:
        logger.info("No walk_corpus configured — using CSV ingr-ingr edges only.")

    # Pre-compute features and similarities (shared across seeds)
    x_matrix, topk_similarities, raw_to_idx, idx_to_raw = prepare_features_and_similarities(
        fg, cfg, root, smoke=smoke, modality=modality
    )

    models_all_seeds: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_export_paths: Dict[int, Dict[str, Path]] = {}

    export_dir_path = Path(export_dir) if export_dir else root / "data" / "exports"
    checkpoint_dir = export_dir_path / "checkpoints"

    for seed in seeds:
        logger.info(f"\n>>> Running Seed {seed} <<<")
        cfg_split = cfg.get("splits", {})
        split = create_ingredient_split(
            fg,
            test_frac=cfg_split.get("test_frac", 0.15),
            val_frac=cfg_split.get("val_frac", 0.10),
            min_degree=cfg_split.get("min_degree", 5),
            hub_only=cfg_split.get("hub_only", True),
            seed=seed,
        )
        verify_no_leakage(split, fg)

        seed_summary, trained_model = train_single_seed(
            fg=fg,
            split=split,
            x_matrix=x_matrix,
            topk_similarities=topk_similarities,
            raw_to_idx=raw_to_idx,
            config=cfg,
            seed=seed,
            device=cfg.get("device", "auto"),
            checkpoint_dir=checkpoint_dir,
            modality=modality,
            loss_type=loss,
        )

        # Export artifacts for this seed
        export_paths = export_artifacts(
            model=trained_model,
            fg=fg,
            split=split,
            topk_similarities=topk_similarities,
            raw_to_idx=raw_to_idx,
            idx_to_raw=idx_to_raw,
            config=cfg,
            seed=seed,
            export_dir=export_dir_path,
            x_matrix=x_matrix,
        )
        all_export_paths[seed] = export_paths

        for model_name, metrics in seed_summary.items():
            for m_name, (mean_val, _) in metrics.items():
                models_all_seeds[model_name][m_name].append(mean_val)

    # Bundle all exports into a single zip archive for 1-click Colab download
    zip_archive_path = create_export_zip(export_dir_path)

    # Compute overall mean +- std across seeds
    final_table: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for model_name, metrics_dict in models_all_seeds.items():
        final_table[model_name] = {}
        for m_name, values in metrics_dict.items():
            final_table[model_name][m_name] = (float(np.mean(values)), float(np.std(values)))

    table_md = format_metrics_table(final_table)
    total_elapsed = time.time() - pipeline_t0

    print("\n" + "=" * 65)
    print(f"FINAL MULTI-SEED EVALUATION RESULTS [{modality.upper()} | {loss.upper()}] (Mean ± Std)")
    print("=" * 65)
    print(table_md)
    print("=" * 65)
    print(f"Total Wall-Clock Time: {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)")
    print(f"Portable export archive ready at: {zip_archive_path}")
    print(f"Archive size: {zip_archive_path.stat().st_size / 1e6:.1f} MB")

    return {
        "final_table": final_table,
        "export_paths": all_export_paths,
        "zip_archive": zip_archive_path,
        "wall_clock_seconds": total_elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SavorLink Strategy III Training & Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--export_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Quick verification smoke run (2 epochs, 1 seed)")
    parser.add_argument("--hub_only", action="store_true", default=None, help="Restrict graph to 416 hub ingredients with chemistry")
    parser.add_argument("--all_nodes", action="store_true", help="Use all 6,653 ingredients in FlavorGraph")
    parser.add_argument(
        "--modality",
        type=str,
        default="molecular",
        choices=["molecular", "semantic", "hybrid", "taxonomic"],
        help="Cold-start similarity modality",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="bpr",
        choices=["bpr", "huber"],
        help="Loss function: 'bpr' (Pairwise BPR Ranking Loss) or 'huber' (Pointwise Huber Regression)",
    )
    args = parser.parse_args()

    hub_only = False if args.all_nodes else (True if args.hub_only else None)
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_pipeline(
        args.config,
        seed_list,
        export_dir=args.export_dir,
        smoke=args.smoke,
        hub_only=hub_only,
        modality=args.modality,
        loss=args.loss,
    )
