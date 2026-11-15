"""Standalone inference engine for SavorLink Strategy III models.

Loads exported model artifacts (.pkl / .pt, graph metadata, embeddings, similarities)
from a directory or from a .zip archive (e.g. 'savorlink_model_exports.zip')
and provides fast food pairing recommendation and scoring.
"""
from __future__ import annotations

import logging
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class SavorLinkPredictor:
    """Inference predictor loaded from exported model checkpoints or zip bundles."""

    def __init__(
        self,
        model: "torch.nn.Module",
        config: dict,
        raw_to_idx: Dict[int, int],
        idx_to_raw: Dict[int, int],
        ingredient_names: Dict[int, str],
        hub_ingredients: List[int],
        topk_similarities: Dict[int, List[Tuple[int, float]]],
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        device: str = "auto",
    ):
        self.config = config
        self.raw_to_idx = raw_to_idx
        self.idx_to_raw = idx_to_raw
        self.ingredient_names = ingredient_names
        self.hub_ingredients = set(hub_ingredients)
        self.topk_similarities = topk_similarities
        self.node_features = node_features

        # Reverse name lookup (case-insensitive and whitespace-normalized)
        self.name_to_raw: Dict[str, int] = {}
        for nid, name in ingredient_names.items():
            norm = name.strip().lower()
            self.name_to_raw[norm] = nid

        # Resolve PyTorch device
        if torch is None:
            raise RuntimeError("PyTorch is required for SavorLinkPredictor inference.")

        if device == "auto":
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)

        self.model = model.to(self.dev)
        self.model.eval()

        # Cache PyTorch tensors
        self.x_tensor = torch.from_numpy(node_features).float().to(self.dev)
        self.base_edge_index = torch.from_numpy(edge_index).long().to(self.dev)
        self.base_edge_weight = torch.from_numpy(edge_weight).float().to(self.dev)

        # Pre-encoded embeddings cache
        self._cached_embeddings: Optional[torch.Tensor] = None

    @classmethod
    def from_bundle(
        cls,
        source_path: Union[str, Path],
        seed: int = 42,
        device: str = "auto",
    ) -> SavorLinkPredictor:
        """Load predictor directly from exported directory or zip bundle."""
        return load_inference_bundle(source_path, seed=seed, device=device)

    def predict(
        self,
        query: Union[str, int],
        top_n: int = 10,
        candidate_pool: Optional[List[Union[str, int]]] = None,
    ) -> List[Dict]:
        """Convenience alias for recommend_pairings."""
        return self.recommend_pairings(query, top_k=top_n, candidate_pool=candidate_pool)

    def predict_new_compounds(
        self,
        compound_cids: List[int],
        top_n: int = 10,
        cid_to_fp: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[Dict]:
        """Predict pairings for a novel ingredient defined purely by compound CIDs.

        Not implemented. The previous version ignored ``compound_cids`` and
        returned recommendations for an arbitrary hub ingredient, which looked
        like a working feature but was not. Implementing it needs RDKit to
        fingerprint the compounds, a similarity search against the hubs, and
        pseudo-edge injection for a brand-new node.
        """
        raise NotImplementedError(
            "predict_new_compounds() is not implemented: it requires computing "
            "fingerprints for new compounds (RDKit) and inserting a new node. "
            "Use predict() with an existing ingredient name instead."
        )

    def resolve_ingredient_id(self, query: Union[str, int]) -> int:
        """Resolve ingredient name or raw ID to integer node ID."""
        if isinstance(query, int):
            if query in self.ingredient_names:
                return query
            raise ValueError(f"Raw ingredient node_id {query} not found in graph.")

        q_norm = str(query).strip().lower()
        if q_norm in self.name_to_raw:
            return self.name_to_raw[q_norm]

        # Try prefix / substring match if exact not found
        matches = [name for name in self.name_to_raw if q_norm in name]
        if len(matches) == 1:
            return self.name_to_raw[matches[0]]
        elif len(matches) > 1:
            return self.name_to_raw[matches[0]]

        raise ValueError(f"Ingredient '{query}' not recognized in knowledge base.")

    def get_ingredient_name(self, node_id: int) -> str:
        """Get friendly name for node ID."""
        return self.ingredient_names.get(node_id, f"ingredient_{node_id}")

    def compute_all_embeddings(self, force_refresh: bool = False) -> torch.Tensor:
        """Compute and cache latent embeddings for all ingredients in the base graph."""
        if self._cached_embeddings is not None and not force_refresh:
            return self._cached_embeddings

        with torch.no_grad():
            emb = self.model.encode(
                self.x_tensor,
                self.base_edge_index,
                self.base_edge_weight,
            )
        self._cached_embeddings = emb
        return emb

    def recommend_pairings(
        self,
        query: Union[str, int],
        top_k: int = 15,
        candidate_pool: Optional[List[Union[str, int]]] = None,
        use_pseudo_edges_if_cold: bool = True,
    ) -> List[Dict]:
        """Recommend best pairing partners for an ingredient.

        Args:
            query: Ingredient name (e.g. 'strawberry', 'garlic') or node ID.
            top_k: Number of ranked recommendations to return.
            candidate_pool: Optional subset of candidate ingredients to score against.
            use_pseudo_edges_if_cold: If True and the ingredient has pseudo-edges,
                                      dynamically injects them during inference.

        Returns:
            List of dicts with keys: 'rank', 'node_id', 'name', 'score', 'is_hub'.
        """
        query_id = self.resolve_ingredient_id(query)
        query_idx = self.raw_to_idx[query_id]

        # Determine candidates
        if candidate_pool is not None:
            cand_ids = [self.resolve_ingredient_id(c) for c in candidate_pool]
            cand_ids = [cid for cid in cand_ids if cid != query_id]
        else:
            cand_ids = [cid for cid in self.raw_to_idx.keys() if cid != query_id]

        cand_indices = [self.raw_to_idx[cid] for cid in cand_ids]
        cand_indices_t = torch.tensor(cand_indices, dtype=torch.long, device=self.dev)

        # Check if pseudo-edges are needed for cold node
        p_edges = self.topk_similarities.get(query_id, [])
        needs_dynamic_pseudo = use_pseudo_edges_if_cold and len(p_edges) > 0

        with torch.no_grad():
            if needs_dynamic_pseudo:
                # Build dynamic pseudo-edges for query node
                src_list = []
                dst_list = []
                w_list = []
                for neighbor_id, sim in p_edges[:10]:
                    if neighbor_id in self.raw_to_idx:
                        n_idx = self.raw_to_idx[neighbor_id]
                        src_list.extend([query_idx, n_idx])
                        dst_list.extend([n_idx, query_idx])
                        w_list.extend([float(sim), float(sim)])

                if src_list:
                    p_ei = torch.tensor([src_list, dst_list], dtype=torch.long, device=self.dev)
                    p_ew = torch.tensor(w_list, dtype=torch.float32, device=self.dev)
                    combined_ei = torch.cat([self.base_edge_index, p_ei], dim=1)
                    combined_ew = torch.cat([self.base_edge_weight, p_ew], dim=0)
                    emb = self.model.encode(self.x_tensor, combined_ei, combined_ew)
                else:
                    emb = self.compute_all_embeddings()
            else:
                emb = self.compute_all_embeddings()

            h_query = emb[query_idx]
            h_cands = emb[cand_indices_t]

            scores_t = self.model.head.score_all_pairs_for_node(h_query, h_cands)
            scores_np = scores_t.cpu().numpy()

        ranked_order = np.argsort(-scores_np)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_order, start=1):
            cid = cand_ids[idx]
            results.append({
                "rank": rank,
                "node_id": cid,
                "name": self.get_ingredient_name(cid),
                "score": float(scores_np[idx]),
                "is_hub": cid in self.hub_ingredients,
            })

        return results

    def predict_pair_score(
        self,
        ingr_a: Union[str, int],
        ingr_b: Union[str, int],
    ) -> float:
        """Predict the continuous food pairing score (NPMI) between two ingredients."""
        id_a = self.resolve_ingredient_id(ingr_a)
        id_b = self.resolve_ingredient_id(ingr_b)

        idx_a = self.raw_to_idx[id_a]
        idx_b = self.raw_to_idx[id_b]

        emb = self.compute_all_embeddings()
        pair_t = torch.tensor([[idx_a], [idx_b]], dtype=torch.long, device=self.dev)

        with torch.no_grad():
            score = self.model.head(emb, pair_t)
            return float(score.item())


def load_inference_bundle(
    source_path: Union[str, Path],
    seed: int = 42,
    device: str = "auto",
) -> SavorLinkPredictor:
    """Load a SavorLinkPredictor from an export folder or a .zip archive.

    Args:
        source_path: Path to the export directory or to 'savorlink_model_exports.zip'.
        seed: Random seed corresponding to the saved model file.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        Configured SavorLinkPredictor ready for immediate inference.
    """
    path = Path(source_path)

    temp_dir_obj = None
    if path.is_file() and path.suffix == ".zip":
        temp_dir_obj = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(temp_dir_obj.name)
        export_dir = Path(temp_dir_obj.name)
    elif path.is_dir():
        export_dir = path
    else:
        raise FileNotFoundError(f"Export source path does not exist: {source_path}")

    # 1. Load Model
    pkl_model_path = export_dir / f"model_seed{seed}.pkl"
    pt_model_path = export_dir / f"model_seed{seed}.pt"

    if pkl_model_path.exists():
        with open(pkl_model_path, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        config = saved["config"]
    elif pt_model_path.exists():
        from src.models.strategy3 import Strategy3Model
        ckpt = torch.load(pt_model_path, map_location="cpu")
        config = ckpt["config"]
        model = Strategy3Model(config)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise FileNotFoundError(
            f"Neither {pkl_model_path.name} nor {pt_model_path.name} found in {export_dir}"
        )

    # 2. Graph Metadata
    with open(export_dir / "graph_metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    # 3. Similarities
    with open(export_dir / "topk_similarities.pkl", "rb") as f:
        sims = pickle.load(f)

    # 4. Node features
    feat_path = export_dir / "node_features.npy"
    if feat_path.exists():
        node_features = np.load(feat_path)
    else:
        node_features = np.zeros((meta["num_ingredients"], config["fingerprints"]["n_bits"]), dtype=np.float32)

    # 5. Graph topology
    graph_path = export_dir / "inference_graph.npz"
    if graph_path.exists():
        # Copy out and close the file so a temporary extraction folder can be
        # deleted on Windows (an open .npz handle blocks the cleanup).
        with np.load(graph_path) as g_data:
            edge_index = g_data["edge_index"].copy()
            edge_weight = g_data["edge_weight"].copy()
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros(0, dtype=np.float32)

    predictor = SavorLinkPredictor(
        model=model,
        config=config,
        raw_to_idx=meta["raw_to_idx"],
        idx_to_raw=meta["idx_to_raw"],
        ingredient_names=meta["ingredient_names"],
        hub_ingredients=meta["hub_ingredients"],
        topk_similarities=sims,
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
    )
    return predictor

