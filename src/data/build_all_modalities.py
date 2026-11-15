"""Batch Builder for all 4 Modalities in SavorLink Strategy III.

Generates and caches features and top-k similarities for:
1. Molecular: PubChem SMILES -> Morgan FPs -> Tanimoto Sim
2. Semantic: Wikipedia Descriptions -> Sentence-Transformers -> Cosine Sim
3. Taxonomic: FoodOn Hierarchy -> Wu-Palmer Tree Distance -> WUP Sim
4. Hybrid: Fused Mol + Sem Features -> Blended Sim (0.5 * Mol + 0.5 * Sem)

Runs locally on CPU in ~1-2 minutes total.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.loaders import load_config, load_flavorgraph
from src.data.pubchem_lookup import PubChemCache, build_smiles_cache
from src.chem.fingerprints import compute_compound_fingerprints, pool_mean
from src.chem.similarity import compute_sparse_topk_similarity
from src.graph.build_graph import assemble_feature_matrix, build_node_id_mapping

from src.data.semantic_lookup import WikipediaSummaryCache, build_semantic_cache
from src.chem.semantic_embeddings import compute_semantic_embeddings, compute_sparse_semantic_similarity

from src.data.taxonomic_lookup import FoodOnTaxonomyCache, build_taxonomy_cache
from src.chem.taxonomic_similarity import build_taxonomic_feature_matrix, compute_sparse_taxonomic_similarity

from src.chem.hybrid_similarity import build_hybrid_feature_matrix, compute_hybrid_topk_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_all_modalities(project_root: Path = PROJECT_ROOT) -> None:
    config_path = project_root / "configs" / "default.yaml"
    config = load_config(config_path)

    logger.info("=" * 68)
    logger.info("  SavorLink Strategy III: Building All 4 Modality Representations")
    logger.info("=" * 68)

    fg = load_flavorgraph(config, project_root, hub_only=True)
    all_ingr_ids = sorted(fg.ingredients.keys())
    hub_ids = sorted(fg.hub_ingredients)
    id_to_name = {nid: node.name for nid, node in fg.ingredients.items()}
    raw_to_idx, idx_to_raw = build_node_id_mapping(all_ingr_ids)
    ingredient_names = [id_to_name[nid] for nid in all_ingr_ids]

    cache_dir = project_root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. MOLECULAR MODALITY
    # -------------------------------------------------------------------------
    logger.info("\n--- [1/4] Processing Molecular Modality ---")
    pubchem_cache_path = cache_dir / "pubchem_smiles.json"
    cache = PubChemCache(pubchem_cache_path)
    comp_cids = {c.pubchem_cid for c in fg.compounds.values() if c.pubchem_cid is not None}
    build_smiles_cache(comp_cids, cache, config["pubchem"])

    cid_to_smiles = {cid: cache.get(cid) for cid in comp_cids if cache.get(cid) is not None}
    cid_to_fp = compute_compound_fingerprints(cid_to_smiles, n_bits=2048)

    node_id_to_cid = {
        nid: node.pubchem_cid for nid, node in fg.compounds.items() if node.pubchem_cid is not None
    }
    pooled_fps = {}
    for ingr_id, comp_ids in fg.ingredient_compounds.items():
        fps = [
            cid_to_fp[node_id_to_cid[cid]]
            for cid in comp_ids
            if cid in node_id_to_cid and node_id_to_cid[cid] in cid_to_fp
        ]
        if fps:
            pooled_fps[ingr_id] = pool_mean(fps)

    mol_features = assemble_feature_matrix(all_ingr_ids, raw_to_idx, pooled_fps, n_bits=2048)
    mol_features_path = cache_dir / "node_features.npy"
    np.save(mol_features_path, mol_features)
    logger.info(f"Saved molecular feature matrix: {mol_features.shape} to {mol_features_path}")

    mol_sim_path = cache_dir / "topk_similarities.pkl"
    mol_similarities = compute_sparse_topk_similarity(
        query_ids=all_ingr_ids,
        candidate_ids=hub_ids,
        ingredient_compounds=fg.ingredient_compounds,
        node_id_to_cid=node_id_to_cid,
        cid_to_fp=cid_to_fp,
        strategy=config.get("similarity", {}).get("strategy", "mean_best_match"),
        top_k=config.get("similarity", {}).get("top_k", 10),
        threshold=config.get("similarity", {}).get("threshold", 0.05),
        pooled_fps=pooled_fps,
        cache_path=str(mol_sim_path),
    )

    # -------------------------------------------------------------------------
    # 2. SEMANTIC MODALITY (Wikipedia + Sentence-Transformers)
    # -------------------------------------------------------------------------
    logger.info("\n--- [2/4] Processing Semantic Modality ---")
    sem_cache_path = cache_dir / "wikipedia_summaries.json"
    sem_cache = WikipediaSummaryCache(sem_cache_path)
    descriptions = build_semantic_cache(ingredient_names, sem_cache, fetch_online=True)

    sem_features_path = cache_dir / "semantic_features.npy"
    sem_features = compute_semantic_embeddings(
        descriptions=descriptions,
        ingredient_order=ingredient_names,
        model_name="all-MiniLM-L6-v2",
        cache_npy_path=sem_features_path,
    )

    name_to_idx = {name.lower(): idx for idx, name in enumerate(ingredient_names)}
    sem_sim_path = cache_dir / "semantic_similarities.pkl"
    sem_similarities = compute_sparse_semantic_similarity(
        query_ids=all_ingr_ids,
        candidate_ids=hub_ids,
        id_to_name=id_to_name,
        embeddings=sem_features,
        name_to_idx=name_to_idx,
        top_k=config.get("similarity", {}).get("top_k", 10),
        threshold=0.0,
        cache_path=sem_sim_path,
    )

    # -------------------------------------------------------------------------
    # 3. TAXONOMIC MODALITY (FoodOn Hierarchy + Wu-Palmer)
    # -------------------------------------------------------------------------
    logger.info("\n--- [3/4] Processing Taxonomic Modality ---")
    tax_cache_path = cache_dir / "foodon_taxonomy.json"
    tax_cache = FoodOnTaxonomyCache(tax_cache_path)
    taxonomy_data = build_taxonomy_cache(ingredient_names, tax_cache)

    tax_features_path = cache_dir / "taxonomic_features.npy"
    tax_features = build_taxonomic_feature_matrix(
        taxonomy_data=taxonomy_data,
        ingredient_order=ingredient_names,
        cache_npy_path=tax_features_path,
    )

    tax_sim_path = cache_dir / "taxonomic_similarities.pkl"
    tax_similarities = compute_sparse_taxonomic_similarity(
        query_ids=all_ingr_ids,
        candidate_ids=hub_ids,
        id_to_name=id_to_name,
        taxonomy_data=taxonomy_data,
        top_k=config.get("similarity", {}).get("top_k", 10),
        threshold=0.0,
        cache_path=tax_sim_path,
    )

    # -------------------------------------------------------------------------
    # 4. HYBRID MODALITY (Molecular + Semantic Fusion)
    # -------------------------------------------------------------------------
    logger.info("\n--- [4/4] Processing Hybrid Modality ---")
    hybrid_features_path = cache_dir / "hybrid_features.npy"
    hybrid_features = build_hybrid_feature_matrix(
        mol_features=mol_features,
        sem_features=sem_features,
        cache_npy_path=hybrid_features_path,
    )

    hybrid_sim_path = cache_dir / "hybrid_similarities.pkl"
    hybrid_similarities = compute_hybrid_topk_similarity(
        query_ids=all_ingr_ids,
        candidate_ids=hub_ids,
        mol_similarities=mol_similarities,
        sem_similarities=sem_similarities,
        alpha=0.5,
        top_k=config.get("similarity", {}).get("top_k", 10),
        threshold=0.05,
        cache_path=hybrid_sim_path,
    )

    logger.info("\n" + "=" * 68)
    logger.info("  ALL 4 MODALITIES SUCCESSFULLY BUILT AND VERIFIED!")
    logger.info(f"  Molecular:  Features {mol_features.shape} | Sim: {len(mol_similarities)} items")
    logger.info(f"  Semantic:   Features {sem_features.shape} | Sim: {len(sem_similarities)} items")
    logger.info(f"  Taxonomic:  Features {tax_features.shape} | Sim: {len(tax_similarities)} items")
    logger.info(f"  Hybrid:     Features {hybrid_features.shape} | Sim: {len(hybrid_similarities)} items")
    logger.info("=" * 68)


if __name__ == "__main__":
    build_all_modalities()
