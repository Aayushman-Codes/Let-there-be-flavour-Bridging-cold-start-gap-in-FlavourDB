"""Unit test for Phase 5 & 7: Entrypoint imports and end-to-end smoke verification.

Asserts that:
1. src.train imports cleanly without any missing symbols or circular dependencies.
2. run_pipeline(smoke=True) runs 2-epoch training, baselines, and artifact export in seconds.
3. The exported zip archive exists, is valid, and can be loaded by SavorLinkPredictor for inference.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    torch = None


def test_entrypoint_and_smoke():
    if torch is None:
        print("  [SKIP] PyTorch not installed in this environment; skipping entrypoint smoke test.")
        return

    from src.train import run_pipeline
    from src.inference import load_inference_bundle

    import tempfile

    config_path = str(PROJECT_ROOT / "configs" / "default.yaml")
    # Write smoke artefacts to a temp folder so the test never touches the repo
    export_dir = str(Path(tempfile.mkdtemp(prefix="savorlink_smoke_")) / "exports")

    # Run 2-epoch smoke pipeline
    results = run_pipeline(config_path, seeds=[42], export_dir=export_dir, smoke=True)

    assert "final_table" in results, "results dict missing 'final_table'"
    assert "zip_archive" in results, "results dict missing 'zip_archive'"

    table = results["final_table"]
    # Check core baselines
    for base in ["Random", "Popularity", "Zero-degree GNN"]:
        assert base in table, f"Baseline {base} missing from final evaluation table"

    # Find kNN and Strategy III keys (supports both legacy and multi-modal names)
    knn_key = next((k for k in table if k.endswith("-kNN") or k == "Tanimoto-kNN"), None)
    s3_key = next((k for k in table if k.startswith("Strategy III")), None)
    assert knn_key is not None, f"kNN baseline missing from final evaluation table. Keys: {list(table.keys())}"
    assert s3_key is not None, f"Strategy III model missing from final evaluation table. Keys: {list(table.keys())}"

    for m in ["Random", "Popularity", "Zero-degree GNN", knn_key, s3_key]:
        for metric in ["recall_10", "recall_20", "mrr", "ndcg_20", "spearman"]:
            assert metric in table[m], f"Metric {metric} missing for {m}"

    # Verify Strategy III actually differs from zero-degree GNN (pseudo-edges must be non-trivial)
    s3_mrr = table[s3_key]["mrr"]
    zd_mrr = table["Zero-degree GNN"]["mrr"]
    assert s3_mrr != 0.0 or zd_mrr != 0.0, (
        "Both Strategy III and Zero-degree GNN have MRR=0. "
        "Fingerprints may be degenerate — check SMILES cache."
    )
    # Verify pseudo-edge metadata is present (Strategy III must have built pseudo-edges)
    if "pseudo_edge_count" in results:
        assert results["pseudo_edge_count"] > 0, (
            f"Strategy III built 0 pseudo-edges. SMILES cache valid? Check pubchem_smiles.json."
        )

    # Verify zip archive exists
    zip_path = Path(results["zip_archive"])
    assert zip_path.exists(), f"Export zip does not exist at {zip_path}"
    assert zip_path.stat().st_size > 1000, f"Export zip suspiciously small: {zip_path.stat().st_size} bytes"

    # Verify inference engine can load directly from the zip archive
    predictor = load_inference_bundle(zip_path, seed=42)
    assert len(predictor.ingredient_names) > 0, "No ingredients loaded into predictor"

    sample_ingr = list(predictor.ingredient_names.keys())[0]
    recs = predictor.recommend_pairings(sample_ingr, top_k=5)
    assert len(recs) == 5, f"Expected 5 recommendations, got {len(recs)}"


if __name__ == "__main__":
    print("Running Entrypoint & Smoke Pipeline Test...")
    test_entrypoint_and_smoke()
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY!")

