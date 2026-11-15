"""Full SavorLink Strategy III Model.

Integrates:
- Chemical node features (pooled Morgan fingerprints)
- Real + pseudo-edge message passing with Tanimoto scaling (Branch A)
- Optional cuisine conditioning (Branch B)
- Gated multi-branch fusion
- Symmetric link-prediction head for NPMI regression
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.branches import BranchA, BranchB
from src.models.heads import GatedFusion, LinkPredictionHead

logger = logging.getLogger(__name__)


class Strategy3Model(nn.Module):
    """SavorLink Strategy III Architecture."""

    def __init__(self, config: dict, in_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        cfg_m = config.get("model", {})
        cfg_fp = config.get("fingerprints", {})

        if in_dim is None:
            in_dim = cfg_fp.get("n_bits", 2048)
        hidden_dim = cfg_m.get("hidden_dim", 128)
        out_dim = hidden_dim
        num_layers = cfg_m.get("num_layers", 2)
        dropout = cfg_m.get("dropout", 0.3)
        use_gat = cfg_m.get("use_gat", False)
        heads = cfg_m.get("heads", 4)
        fusion_mode = cfg_m.get("fusion", "gated")
        task = cfg_m.get("task", "regression")

        # Branch A: Chemical & Graph Topology Encoder
        self.branch_a = BranchA(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_gat=use_gat,
            heads=heads,
        )

        # Branch B: Cuisine-Conditioned Modulation (Optional)
        cfg_c = cfg_m.get("cuisine_branch", {})
        cuisine_enabled = cfg_c.get("enabled", False)
        num_cuisines = cfg_c.get("num_cuisines", 0)

        self.branch_b = BranchB(
            num_cuisines=num_cuisines,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            enabled=cuisine_enabled,
        )

        # Multi-Branch Fusion (zero overhead when cuisine branch is disabled)
        self.fusion = GatedFusion(dim=out_dim, mode=fusion_mode, enabled=cuisine_enabled)

        # Link Prediction Head
        self.head = LinkPredictionHead(
            node_dim=out_dim,
            hidden_dim=hidden_dim // 2,
            task=task,
            dropout=dropout,
        )

        self.task = task

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        cuisine_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode all graph nodes into latent representation space.

        Args:
            x: Node feature matrix [num_nodes, in_dim]
            edge_index: Graph connectivity (real + pseudo edges) [2, num_edges]
            edge_weight: Edge weights (1.0 for real, Tanimoto for pseudo) [num_edges]
            cuisine_ids: Optional cuisine annotations

        Returns:
            h: Node embeddings [num_nodes, out_dim]
        """
        h_a = self.branch_a(x, edge_index, edge_weight)
        h_b = self.branch_b(h_a, cuisine_ids)
        h = self.fusion(h_a, h_b)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index_graph: torch.Tensor,
        edge_weight_graph: Optional[torch.Tensor],
        eval_edges: torch.Tensor,
        cuisine_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute edge prediction scores for target evaluation pairs."""
        h = self.encode(x, edge_index_graph, edge_weight_graph, cuisine_ids=cuisine_ids)
        return self.head(h, eval_edges)

    def compute_loss(
        self,
        pred_scores: Optional[torch.Tensor] = None,
        true_scores: Optional[torch.Tensor] = None,
        loss_type: Optional[str] = None,
        pos_scores: Optional[torch.Tensor] = None,
        neg_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate loss based on task type (regression, BCE, or BPR pairwise ranking)."""
        mode = (loss_type or self.task).lower()
        if mode == "bpr" and pos_scores is not None and neg_scores is not None:
            # Pairwise Bayesian Personalized Ranking Loss
            pos = pos_scores.view(-1, 1)
            num_pos = pos.size(0)
            if neg_scores.size(0) % num_pos == 0 and num_pos > 0:
                neg = neg_scores.view(num_pos, -1)
                diff = pos - neg
            else:
                diff = pos - neg_scores.view(1, -1)
            return -torch.mean(F.logsigmoid(diff))
        elif mode == "regression":
            # Smooth L1 / Huber loss provides robustness to outliers in NPMI distribution
            return F.smooth_l1_loss(pred_scores, true_scores)
        else:
            return F.binary_cross_entropy(pred_scores, true_scores)
