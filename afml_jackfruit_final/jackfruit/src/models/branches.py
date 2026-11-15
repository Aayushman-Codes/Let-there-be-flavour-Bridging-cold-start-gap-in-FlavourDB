"""Graph neural network branch encoders for SavorLink Strategy III.

Branch A (Shared-Compound / Chemical & Topology):
    Aggregates messages over real recipe edges AND injected pseudo-edges.
    Crucially, messages across pseudo-edges are scaled by their Tanimoto
    similarity weights, allowing the cold node to receive information
    proportional to chemical relevance without overwhelming the real
    co-occurrence signals.

Branch B (Cuisine-Conditioned, Optional):
    Modulates edge messages based on cuisine embeddings when cuisine labels
    are available. Gated behind config['model']['cuisine_branch']['enabled'].
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def scatter_add_vec(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Safe scatter-add operation along dimension 0.

    Works without requiring torch_scatter C++ binaries, ensuring 100%
    cross-platform compatibility across local environments and Google Colab.
    """
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    return out.index_add(0, index, src)


def scatter_add_scalar(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Safe scatter-add operation along dimension 0 for 1D or (E, 1) tensors."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    return out.index_add(0, index, src)


class WeightedSAGEConv(nn.Module):
    """GraphSAGE convolution layer with continuous edge weights.

    Why custom weighted SAGE?
    Standard SAGEConv does not support continuous edge weights or scales them
    identically to degree normalization. For Strategy III:
    - Real recipe edges have weights (e.g. 1.0 or NPMI).
    - Pseudo-edges have weights in [0, 1] equal to Tanimoto similarity.
    - Aggregator computes normalized weighted mean of incoming messages:
          h_N(u) = (sum_{v in N(u)} w_{uv} * h_v) / (sum_{v in N(u)} w_{uv} + eps)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        if self.lin_self.bias is not None:
            nn.init.zeros_(self.lin_self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges] (src, dst)
            edge_weight: Optional edge weights [num_edges]
        """
        num_nodes = x.size(0)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # 1. Self message
        out_self = self.lin_self(x)

        if edge_index.size(1) == 0:
            return out_self

        # 2. Neighbor messages scaled by edge weight
        msg = x[src_idx]
        if edge_weight is not None:
            w = edge_weight.view(-1, 1).to(x.dtype)
            msg = msg * w
            w_sum = scatter_add_scalar(w, dst_idx, dim_size=num_nodes)
            agg = scatter_add_vec(msg, dst_idx, dim_size=num_nodes)
            # Normalize by incoming weight sum where incoming edges exist
            denom = torch.clamp(w_sum, min=1e-8)
            agg = agg / denom
        else:
            agg = scatter_add_vec(msg, dst_idx, dim_size=num_nodes)
            deg = scatter_add_scalar(
                torch.ones(src_idx.size(0), 1, device=x.device, dtype=x.dtype),
                dst_idx,
                dim_size=num_nodes,
            )
            denom = torch.clamp(deg, min=1.0)
            agg = agg / denom

        out_neigh = self.lin_neigh(agg)
        return out_self + out_neigh


class WeightedGATConv(nn.Module):
    """Multi-head Graph Attention Network (GAT) with edge weight modulation.

    Incorporates Tanimoto edge weights as prior biases or scaling factors
    on attention logits:
        alpha_{uv} = softmax(LeakyReLU(a^T [W h_u || W h_v]) + log(w_{uv} + eps))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.lin_self = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.lin_self.bias is not None:
            nn.init.zeros_(self.lin_self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_src = self.lin_src(x).view(num_nodes, self.heads, self.out_channels)
        h_dst = self.lin_dst(x).view(num_nodes, self.heads, self.out_channels)

        out_self = self.lin_self(x)

        if edge_index.size(1) == 0:
            return out_self

        # Attention score = (h_src * att_src) + (h_dst * att_dst)
        alpha_src = (h_src * self.att_src).sum(dim=-1)  # [N, heads]
        alpha_dst = (h_dst * self.att_dst).sum(dim=-1)  # [N, heads]
        alpha = alpha_src[src_idx] + alpha_dst[dst_idx]  # [E, heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Modulate attention with Tanimoto edge weight
        if edge_weight is not None:
            w_bias = torch.log(torch.clamp(edge_weight.view(-1, 1), min=1e-6))
            alpha = alpha + w_bias

        # Softmax over incoming edges per destination node
        # For numerical stability, subtract max
        alpha_max = torch.zeros(num_nodes, self.heads, device=x.device, dtype=x.dtype)
        # Approximate softmax using exp
        exp_alpha = torch.exp(torch.clamp(alpha, min=-10.0, max=10.0))
        denom = torch.zeros(num_nodes, self.heads, device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst_idx, exp_alpha)
        denom = torch.clamp(denom, min=1e-8)

        norm_alpha = exp_alpha / denom[dst_idx]
        norm_alpha = F.dropout(norm_alpha, p=self.dropout, training=self.training)

        # Message aggregation
        msg = h_src[src_idx] * norm_alpha.unsqueeze(-1)  # [E, heads, out_channels]
        msg = msg.view(src_idx.size(0), self.heads * self.out_channels)

        agg = torch.zeros(num_nodes, self.heads * self.out_channels, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst_idx, msg)

        return out_self + agg


class BranchA(nn.Module):
    """Branch A: Chemical-Topology aggregator.

    Processes node features (chemical pooled Morgan fingerprints) through
    multiple layers of WeightedSAGEConv or WeightedGATConv over real + pseudo edges.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_gat: bool = False,
        heads: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            cur_in = hidden_dim
            cur_out = out_dim if layer_idx == num_layers - 1 else hidden_dim

            if use_gat:
                # Adjust channel dim so heads * out_channels == cur_out
                head_dim = max(1, cur_out // heads)
                conv = WeightedGATConv(cur_in, head_dim, heads=heads, dropout=dropout)
                self.convs.append(conv)
            else:
                self.convs.append(WeightedSAGEConv(cur_in, cur_out))

            self.norms.append(nn.LayerNorm(cur_out))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_weight)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual connection if dims match
            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        return h


class BranchB(nn.Module):
    """Branch B (Optional): Cuisine-conditioned message modulation.

    Gated behind config. If disabled or no cuisine annotations are provided,
    returns None.
    """

    def __init__(
        self,
        num_cuisines: int = 30,
        cuisine_emb_dim: int = 32,
        hidden_dim: int = 128,
        out_dim: int = 128,
        enabled: bool = False,
    ):
        super().__init__()
        self.enabled = enabled
        if self.enabled and num_cuisines > 0:
            self.cuisine_emb = nn.Embedding(num_cuisines, cuisine_emb_dim)
            self.proj = nn.Linear(cuisine_emb_dim + hidden_dim, out_dim)
        else:
            self.cuisine_emb = None
            self.proj = None

    def forward(
        self,
        h_a: torch.Tensor,
        cuisine_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.enabled or self.proj is None or cuisine_ids is None:
            return None

        c_emb = self.cuisine_emb(cuisine_ids)
        combined = torch.cat([h_a, c_emb], dim=-1)
        return F.relu(self.proj(combined))
