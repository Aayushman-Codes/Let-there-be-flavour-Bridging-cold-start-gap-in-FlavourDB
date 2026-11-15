# SavorLink Architecture

This document describes how the code is organised and how data flows from the raw FlavorGraph files to the results table. For the project overview and results, see [README.md](README.md).

## 1. Pipeline

```
data/raw/*.csv ──► loaders.load_flavorgraph (hub-only: 416 ingredients, 1,897 recipe edges)
                          │
data/cache/*  ──► train.prepare_features_and_similarities(modality)
                          │   node features X  +  top-k similar hubs per ingredient
                          ▼
              splits.create_ingredient_split(seed)
                          │   train / val / test ingredients; held-out edges become ground truth
                          ▼
              train.train_single_seed
                ├─ each epoch: cold-mask 10% of train ingredients, inject their pseudo-edges,
                │              BPR loss on masked + sampled warm edges (4 anchored negatives each)
                ├─ early stopping on validation BPR loss
                └─ test: score every hub candidate for each cold test ingredient
                          │
                          ▼
              evaluate.evaluate_ranking_for_ingredient  (Random, Popularity, kNN,
                          │                               zero-degree GNN, Strategy III)
                          ▼
              run_benchmark → results/RESULTS.md, results/results.json, results/models/<modality>/
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
        src.compare              app/streamlit_app.py
   (terminal comparison)   (frontend: reads data/ + results/ live,
                            reruns kNN and GNN inference per ingredient)
```

The feature caches in `data/cache/` are precomputed and committed, so training needs only PyTorch and NumPy. `src/data/build_all_modalities.py` rebuilds them (requires RDKit, sentence-transformers and network access).

## 2. Modules

| Path | Responsibility |
|---|---|
| `src/data/loaders.py` | Parse FlavorGraph CSVs into typed node/edge dataclasses; optional hub-only filtering. |
| `src/data/splits.py` | Ingredient-level train/val/test split; collects held-out edges as ground truth; `verify_no_leakage`. |
| `src/data/walk_loader.py` | Optional: turn the FlavorGraph metapath walk corpus into ingredient co-occurrence edges (disabled by default, see §6). |
| `src/data/pubchem_lookup.py` | Resumable, rate-limited PubChem SMILES cache. |
| `src/data/semantic_lookup.py` | Ingredient text descriptions (Wikipedia lookup with template fallback). |
| `src/data/taxonomic_lookup.py` | Rule-based 3-level food taxonomy (`TAXONOMIC_RULES`). |
| `src/data/build_all_modalities.py` | Rebuilds every file in `data/cache/`. |
| `src/chem/fingerprints.py` | RDKit Morgan fingerprints and pooling (mean / union). |
| `src/chem/similarity.py` | Tanimoto and set-level ingredient similarity; sparse top-k against hubs. |
| `src/chem/semantic_embeddings.py` | Sentence-transformer embeddings and cosine top-k. |
| `src/chem/taxonomic_similarity.py` | Wu–Palmer similarity and multi-hot taxonomy features. |
| `src/chem/hybrid_similarity.py` | Concatenated features and blended top-k (α = 0.5). |
| `src/graph/pseudo_edges.py` | Build weighted pseudo-edges for cold nodes; merge with real edges. |
| `src/graph/build_graph.py` | Node-ID mapping and feature-matrix assembly. |
| `src/models/branches.py` | `WeightedSAGEConv`, `WeightedGATConv`, `BranchA` (graph encoder), `BranchB` (optional cuisine branch). |
| `src/models/heads.py` | `GatedFusion`; symmetric `LinkPredictionHead`. |
| `src/models/strategy3.py` | `Strategy3Model`: encoder + head + losses (BPR, Huber, BCE). |
| `src/baselines.py` | Random, Popularity, and the modality-agnostic kNN baseline. |
| `src/evaluate.py` | Recall@K, MRR, NDCG@20, Spearman; aggregation and table formatting. |
| `src/train.py` | Feature loading, training loop, test evaluation, artefact export, `run_pipeline`, CLI. |
| `src/run_benchmark.py` | Runs all modalities × seeds and writes `results/RESULTS.md` and `results/results.json`. |
| `src/compare.py` | Terminal comparison of the 4 methods from `results/results.json` (tables, head-to-head, leaderboard, verdict). |
| `app/streamlit_app.py` | Frontend: benchmark charts, before/after cold-start lab, semantic context with optional live re-encoding. |
| `src/demo.py` | kNN vs Strategy III recommendations for one held-out ingredient. |
| `src/inference.py` | `SavorLinkPredictor`: load an exported model and recommend pairings. |
| `src/export_colab_bundle.py` | Packs config, data and source into `data/colab_training_bundle.zip` for Colab. |

## 3. Model (Strategy III)

- **Input projection:** `Linear(in_dim → 128) → LayerNorm → ReLU → Dropout(0.3)`, where `in_dim` is 2048 / 384 / 141 / 2432 by modality.
- **Encoder:** 2 × `WeightedSAGEConv` with weighted mean aggregation, `h_N(u) = Σ w_uv·h_v / (Σ w_uv + ε)`; real edges carry their NPMI weight and pseudo-edges their similarity. LayerNorm between layers. GAT is available via `model.use_gat`.
- **Cuisine branch and fusion:** implemented but disabled (`cuisine_branch.enabled: false`); FlavorGraph has no cuisine labels, so the fusion is a no-op.
- **Head:** symmetric pair features `[h_u ⊙ h_v, h_u + h_v, |h_u − h_v|]` → `MLP(384 → 64 → 1)`, so `s(u, v) = s(v, u)`.
- **Loss:** BPR on raw logits, `−log σ(s(u, v⁺) − s(u, v⁻))`, with 4 negatives per positive that share the anchor `u`. Huber regression on NPMI is available with `--loss huber`.
- **Cold-mask training:** each epoch hides all edges of 10% of training ingredients, injects their pseudo-edges, and trains on the hidden edges plus a sample of warm edges.

No `nn.Embedding(num_nodes)` table is used anywhere, so unseen ingredients can be embedded from features alone.

## 4. Evaluation protocol

- **Graph:** the 416 hub ingredients (416 of the 417 ingredients with chemical annotations) and their 1,897 Recipe1M co-occurrence edges (NPMI).
- **Split:** hubs with ≥ 5 edges are eligible; 15% test, 10% validation, per seed (42, 123, 456). Held-out ingredients lose every edge from the training graph.
- **Ground truth:** a held-out ingredient's partners are its removed edges with NPMI > 0 (no self-pairs).
- **Candidates:** all other 415 hubs. Pseudo-edges and kNN neighbours never point at other held-out ingredients.
- **Metrics:** Recall@10, Recall@20, MRR, NDCG@20 (graded by NPMI), Spearman ρ over true partners; mean ± std across seeds.

## 5. Configuration

Everything is in `configs/default.yaml`:

| Key | Meaning |
|---|---|
| `data.hub_only` | Restrict the graph to the 416 hubs (default `true`). |
| `paths.walk_corpus` | Optional walk corpus; `null` uses the CSV recipe edges (default). |
| `similarity.*`, `pseudo_edges.*` | Top-k, threshold and temperature for neighbours and pseudo-edges. |
| `model.*` | Hidden size, layers, dropout, SAGE vs GAT, task. |
| `training.*` | Epochs, learning rate, patience, cold-mask fraction, `neg_sample_ratio`. |
| `evaluation.*` | Cut-offs, seeds, candidate pool. |

## 6. Design decisions

- **CSV edges, not walk edges, as ground truth.** Among the hubs, walk-derived co-occurrence connects every pair (≈417 "partners" per ingredient among 415 candidates) and mostly reflects shared compounds. That made Recall and MRR meaningless in V2. See [report 3](reports/versioned_report_3_corrected.md) §1.
- **Anchored negatives.** Each BPR negative shares the positive's anchor ingredient, so the model must learn ingredient-specific pairing rather than general popularity.
- **Precomputed caches.** Network lookups (PubChem, Wikipedia) and heavy feature extraction run once; training stays fast and offline.
- **Pure PyTorch.** Aggregation uses `index_add_`, so there is no `torch_scatter` / PyTorch Geometric build dependency.
