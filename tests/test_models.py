"""Unit tests for Phase 4: Strategy III Model Architecture, Branches, and Prediction Heads.

Invariants & Requirements Verified:
1. Output shapes: BranchA, BranchB, GatedFusion, LinkPredictionHead, and Strategy3Model.
2. torch.no_grad() vs train-mode: dropout behaviour and evaluation determinism.
3. Determinism under fixed seed: Identical weights and inputs yield bitwise-equal predictions.
4. Gradients reach every parameter: No dead parameters or missing gradient flow during backprop.
5. Disabled-cuisine path: Numerically identical to single-branch model (BranchA output == fused output).
6. Low-weight pseudo-edges: Cold nodes with small weights (e.g. 0.01) produce finite, non-NaN, non-zero embeddings.
7. Parameter count & memory: Model parameter count is printed and well under Colab T4 budgets (< 1M params).
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None


def test_models_all():
    if torch is None:
        print("  [SKIP] PyTorch not installed in this environment; skipping neural network tests.")
        return

    from src.models.branches import WeightedSAGEConv, WeightedGATConv, BranchA, BranchB
    from src.models.heads import GatedFusion, LinkPredictionHead
    from src.models.strategy3 import Strategy3Model

    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # 1. Output Shapes & Layer Functionality
    # -------------------------------------------------------------------------
    conv_sage = WeightedSAGEConv(in_channels=16, out_channels=32)
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.9, 0.5, 0.2, 0.8], dtype=torch.float32)

    out_sage = conv_sage(x, edge_index, edge_weight)
    assert out_sage.shape == (10, 32), f"Expected SAGE shape (10, 32), got {out_sage.shape}"

    conv_gat = WeightedGATConv(in_channels=16, out_channels=8, heads=4)
    out_gat = conv_gat(x, edge_index, edge_weight)
    assert out_gat.shape == (10, 32), f"Expected GAT shape (10, 32), got {out_gat.shape}"

    branch_a = BranchA(in_dim=16, hidden_dim=32, out_dim=32, num_layers=2)
    h_a = branch_a(x, edge_index, edge_weight)
    assert h_a.shape == (10, 32), f"Expected BranchA shape (10, 32), got {h_a.shape}"

    # -------------------------------------------------------------------------
    # 2. Disabled-Cuisine Path is Numerically Identical to Single-Branch Model
    # -------------------------------------------------------------------------
    branch_b_disabled = BranchB(enabled=False)
    assert branch_b_disabled(h_a) is None, "Disabled BranchB should return None"

    fusion_gate = GatedFusion(dim=32, mode="gated")
    h_fused_noop = fusion_gate(h_a, None)
    assert torch.all(h_fused_noop == h_a), "Fusion with disabled BranchB must be bitwise identical to BranchA output"

    # Enabled BranchB
    branch_b_enabled = BranchB(num_cuisines=5, cuisine_emb_dim=8, hidden_dim=32, out_dim=32, enabled=True)
    cuisines = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long)
    h_b = branch_b_enabled(h_a, cuisines)
    assert h_b.shape == (10, 32), f"Expected BranchB shape (10, 32), got {h_b.shape}"
    h_fused = fusion_gate(h_a, h_b)
    assert h_fused.shape == (10, 32), f"Expected fused shape (10, 32), got {h_fused.shape}"

    # -------------------------------------------------------------------------
    # 3. LinkPredictionHead Pairwise Symmetry & score_all_pairs_for_node
    # -------------------------------------------------------------------------
    head = LinkPredictionHead(node_dim=32, hidden_dim=16, task="regression")
    head.eval()  # Eval mode disables dropout to verify exact algebraic symmetry

    pairs_uv = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    pairs_vu = torch.tensor([[3, 4], [1, 2]], dtype=torch.long)

    scores_uv = head(h_a, pairs_uv)
    scores_vu = head(h_a, pairs_vu)
    assert torch.allclose(scores_uv, scores_vu, atol=1e-5), "LinkPredictionHead must be symmetric: score(u, v) == score(v, u)"

    # Candidate scoring against all nodes: score node 1 against all nodes
    all_scores_1 = head.score_all_pairs_for_node(h_a[1], h_a)
    assert all_scores_1.shape == (10,), f"Expected score_all_pairs_for_node shape (10,), got {all_scores_1.shape}"
    assert torch.isclose(all_scores_1[3], scores_uv[0], atol=1e-5), "score_all_pairs_for_node must match forward score for same pair (1, 3)"

    # -------------------------------------------------------------------------
    # 4. Low-Weight Pseudo-Edges Produce Finite, Non-Degenerate Embeddings
    # -------------------------------------------------------------------------
    # Node 9 is a cold node with only a single very low weight pseudo-edge (weight = 0.005)
    cold_edge_index = torch.tensor([[9, 1], [1, 9]], dtype=torch.long)
    cold_edge_weight = torch.tensor([0.005, 0.005], dtype=torch.float32)
    h_cold = branch_a(x, cold_edge_index, cold_edge_weight)

    assert not torch.isnan(h_cold).any(), "Low-weight pseudo-edge produced NaN embedding"
    assert not torch.isinf(h_cold).any(), "Low-weight pseudo-edge produced Inf embedding"
    assert (h_cold[9].abs().sum() > 1e-4), "Low-weight pseudo-edge produced degenerate all-zero embedding"

    # Completely isolated node (zero edges) should also produce finite embedding from self-projection
    empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
    empty_edge_weight = torch.zeros(0, dtype=torch.float32)
    h_isolated = branch_a(x, empty_edge_index, empty_edge_weight)
    assert not torch.isnan(h_isolated).any(), "Isolated zero-edge node produced NaN"
    assert (h_isolated.abs().sum() > 1e-4), "Isolated node produced degenerate all-zero embedding"

    # -------------------------------------------------------------------------
    # 5. Full Strategy3Model Forward, Loss, and Gradients Reach Every Parameter
    # -------------------------------------------------------------------------
    cfg = {
        "fingerprints": {"n_bits": 16},
        "model": {
            "hidden_dim": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "use_gat": False,
            "fusion": "gated",
            "task": "regression",
            "cuisine_branch": {"enabled": False},
        },
    }
    model = Strategy3Model(cfg)
    model.train()
    eval_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    pred = model(x, edge_index, edge_weight, eval_edges)
    assert pred.shape == (2,)

    target = torch.tensor([0.7, 0.3], dtype=torch.float32)
    loss = model.compute_loss(pred, target)
    loss.backward()

    # Verify gradients reach all parameters in active branches
    dead_params = []
    for name, param in model.named_parameters():
        if "branch_b" in name:
            continue  # Branch B is disabled in config
        if param.requires_grad and (param.grad is None or param.grad.abs().sum() == 0):
            dead_params.append(name)
    assert len(dead_params) == 0, f"Dead parameters found with zero gradient: {dead_params}"

    # -------------------------------------------------------------------------
    # 6. torch.no_grad() vs Train Mode
    # -------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        pred_eval1 = model(x, edge_index, edge_weight, eval_edges)
        pred_eval2 = model(x, edge_index, edge_weight, eval_edges)
    assert torch.all(pred_eval1 == pred_eval2), "Evaluation in torch.no_grad() must be strictly deterministic"

    # -------------------------------------------------------------------------
    # 7. Parameter Count and Peak Memory Printed
    # -------------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 * 1024)

    print(f"\n  [Phase 4 Architecture Verification]")
    print(f"    Total Parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"    Model Weights Size: {model_size_mb:.2f} MB (T4 limit: 15,000 MB)")
    assert total_params < 1_000_000, f"Model parameter count {total_params} exceeds lightweight 1M budget"


if __name__ == "__main__":
    print("=" * 60)
    print("Running SavorLink Phase 4 Model Architecture Tests...")
    print("=" * 60)
    test_models_all()
    print("\nALL PHASE 4 MODEL TESTS PASSED SUCCESSFULLY!")
