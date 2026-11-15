"""Fusion mechanisms and link prediction heads for SavorLink Strategy III.

GatedFusion:
    Combines Branch A (chemical + graph topology) with optional Branch B
    (cuisine conditioned). Defaults to passthrough when Branch B is disabled.

LinkPredictionHead:
    Predicts food pairing intensity between ingredient pairs (u, v).
    Crucially, pairing is symmetric: pair(u, v) must equal pair(v, u).
    We enforce this symmetry structurally in the representation:
        phi(u, v) = [h_u * h_v, h_u + h_v, |h_u - h_v|]
    followed by an MLP predicting continuous NPMI (regression) or edge
    probability (classification).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GatedFusion(nn.Module):
    """Gated fusion mechanism for multi-branch node representations."""

    def __init__(self, dim: int, mode: str = "gated", enabled: bool = True):
        super().__init__()
        self.dim = dim
        self.mode = mode.lower()
        self.enabled = enabled

        if not self.enabled:
            self.gate = None
            self.proj = None
        elif self.mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.Sigmoid(),
            )
        elif self.mode == "concat":
            self.proj = nn.Linear(2 * dim, dim)
        elif self.mode == "add":
            pass
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled or h_b is None:
            return h_a

        if self.mode == "gated":
            z = self.gate(torch.cat([h_a, h_b], dim=-1))
            return z * h_a + (1.0 - z) * h_b
        elif self.mode == "concat":
            return F.relu(self.proj(torch.cat([h_a, h_b], dim=-1)))
        elif self.mode == "add":
            return h_a + h_b
        return h_a


class LinkPredictionHead(nn.Module):
    """Link prediction head for ingredient-ingredient interaction.

    Supports:
        task='regression' (default): Predict continuous NPMI score in [0, 1].
        task='classification': Predict edge existence probability.
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        task: str = "regression",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.task = task.lower()
        self.dropout = dropout

        # Feature size for symmetric combination:
        # [h_u * h_v, h_u + h_v, |h_u - h_v|] -> 3 * node_dim
        in_feats = 3 * node_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def pair_features(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        """Construct symmetric interaction representation between pairs."""
        prod = h_u * h_v
        summ = h_u + h_v
        diff = torch.abs(h_u - h_v)
        return torch.cat([prod, summ, diff], dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scores for given edge indices.

        Args:
            h: Node embeddings [num_nodes, node_dim]
            edge_index: Pairs to score [2, num_pairs]

        Returns:
            scores: Tensor of shape [num_pairs]
        """
        src = edge_index[0]
        dst = edge_index[1]

        h_u = h[src]
        h_v = h[dst]

        feats = self.pair_features(h_u, h_v)
        out = self.mlp(feats).squeeze(-1)

        if self.task == "classification":
            return torch.sigmoid(out)
        else:
            # Regression on NPMI in [0, 1] — Sigmoid ensures predictions stay in (0, 1) without dying at 0
            return torch.sigmoid(out)

    def score_all_pairs_for_node(
        self,
        h_query: torch.Tensor,
        h_all: torch.Tensor,
    ) -> torch.Tensor:
        """Score a single query node against ALL candidate nodes in parallel.

        Args:
            h_query: Query node embedding [1, node_dim] or [node_dim]
            h_all: All candidate node embeddings [N, node_dim]

        Returns:
            scores: [N] prediction scores
        """
        h_q = h_query.view(1, -1).expand(h_all.size(0), -1)
        feats = self.pair_features(h_q, h_all)
        out = self.mlp(feats).squeeze(-1)

        if self.task == "classification":
            return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)
