"""Show cold-start pairing recommendations from a trained model.

Usage:
    python -m src.demo                                  # taxonomic model, a held-out ingredient
    python -m src.demo --modality semantic --ingredient coconut

Loads a model trained by ``python -m src.run_benchmark`` from
results/models/<modality>/, picks a held-out (cold-start) test ingredient, and
prints side by side:
  - the kNN baseline's top pairings (the best method in the benchmark),
  - Strategy III's (GNN) top pairings,
  - the ingredient's true recipe partners, so the hits can be checked.
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import GenericKNNBaseline  # noqa: E402
from src.inference import load_inference_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="SavorLink cold-start pairing demo")
    parser.add_argument("--modality", default="taxonomic",
                        choices=["molecular", "semantic", "hybrid", "taxonomic"])
    parser.add_argument("--models", default=str(PROJECT_ROOT / "results" / "models"))
    parser.add_argument("--ingredient", default=None,
                        help="Name of a held-out test ingredient (default: a random one)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    export_dir = Path(args.models) / args.modality
    if not (export_dir / f"model_seed{args.seed}.pkl").exists():
        sys.exit(f"No trained model in {export_dir}. Run `python -m src.run_benchmark` first.")

    predictor = load_inference_bundle(export_dir, seed=args.seed, device="cpu")
    with open(export_dir / f"split_seed{args.seed}.pkl", "rb") as fh:
        split = pickle.load(fh)
    names = predictor.ingredient_names
    name_to_id = {n: i for i, n in names.items()}

    if args.ingredient:
        cold_id = name_to_id.get(args.ingredient)
        if cold_id not in split.test_ids:
            options = ", ".join(sorted(names[i] for i in split.test_ids)[:15])
            sys.exit(f"'{args.ingredient}' is not a held-out test ingredient. Try one of: {options}, ...")
    else:
        cold_id = random.Random(args.seed).choice(sorted(split.test_ids))

    # True partners: held-out recipe edges with positive co-occurrence strength
    truth = {}
    for (s, d), w in split.test_ground_truth.items():
        if w and w > 0 and cold_id in (s, d) and s != d:
            truth[d if s == cold_id else s] = w

    candidates = [c for c in sorted(predictor.hub_ingredients) if c != cold_id]
    knn = GenericKNNBaseline(split.train_edges, split.train_edge_weights, predictor.topk_similarities)
    knn_scores = knn.predict(cold_id, candidates)
    knn_top = sorted(knn_scores, key=lambda c: -knn_scores[c])[: args.top]
    gnn_top = predictor.recommend_pairings(cold_id, top_k=args.top)

    def mark(cid: int) -> str:
        return f"{names[cid]} *" if cid in truth else names[cid]

    print(f"\nCold-start ingredient: {names[cold_id]}  (modality: {args.modality}; * = true recipe partner)\n")
    print(f"{'#':<4}{'kNN baseline':<34}{'Strategy III (GNN)':<34}")
    print("-" * 72)
    gnn_ids = [r["node_id"] for r in gnn_top]
    for i in range(args.top):
        k = mark(knn_top[i]) if i < len(knn_top) else ""
        g = mark(gnn_ids[i]) if i < len(gnn_ids) else ""
        print(f"{i + 1:<4}{k:<34}{g:<34}")
    hits_k = sum(c in truth for c in knn_top)
    hits_g = sum(c in truth for c in gnn_ids)
    print("-" * 72)
    print(f"Hits in top {args.top}: kNN {hits_k}, GNN {hits_g}  (of {len(truth)} true partners)")
    top_truth = sorted(truth, key=lambda c: -truth[c])[:10]
    print("Strongest true partners: " + ", ".join(names[c] for c in top_truth))


if __name__ == "__main__":
    main()
