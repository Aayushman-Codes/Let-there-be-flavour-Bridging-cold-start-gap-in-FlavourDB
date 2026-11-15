# SavorLink Strategy III — Version 3 Corrected Results

**Date:** 2026-11-15
**Supersedes:** the results tables in `versioned_reports.md` §9 and `versioned_report_2.md` §6
**Reproduce:** the fixes below are part of the code from this version on. Run
`python -m src.run_benchmark` (all 4 modalities × seeds 42, 123, 456; hub-only mode, CPU, ≈10 s per modality).

---

## 1. Why the earlier tables were wrong

| # | Problem | Evidence | Effect |
|---|---------|----------|--------|
| 1 | **Ground truth was a near-complete graph.** In hub-only mode the walk corpus replaced the CSV edges. Among 416 hubs it produced 86,680 edges for 86,320 possible pairs. | Every test ingredient had 417 "true partners" among 415 candidates. | Recall@K and MRR are pinned at their ceiling for *every* method (Random MRR = 1.000). The "Strategy III best Recall@20 / MRR" claim was an artefact. |
| 2 | **Self-pairs** — the walk window counted an ingredient revisited within 5 steps as pairing with itself. | 360 more edges than possible pairs. | Inflated ground truth. |
| 3 | **Zero-score "partners"** — PPMI is clipped at 0 but the edges were kept, and evaluation treated every edge as a positive. | Median walk edge score = 0.0. | Non-pairings counted as hits. |
| 4 | **Target was mostly chemistry, not recipes.** Walks go hub → compound → hub; 84% of hub pairs share ≥1 compound. | 72,452 / 86,320 hub pairs share a compound. | Molecular similarity was partly predicting its own target. |
| 5 | **BPR loss computed on sigmoid outputs.** Score differences were squashed into (−1, 1). | Val loss stalled at 0.59–0.62 vs 0.69 chance. | GNN barely trained. |
| 6 | **Negatives were 1:1 and unanchored.** `neg_sample_ratio` was missing from the config; negatives were random pairs unrelated to the positive's node. | `cfg_train.get("neg_sample_ratio", 1)`. | Not BPR as described; report claimed 4:1. |
| 7 | **Pseudo-edges / kNN neighbours could point at other held-out ingredients**, which have no edges at test time. | 138 of 580 test pseudo-edges (earlier export). | Cold nodes leaned on other cold nodes. |
| 8 | **Colab notebook passed `--epochs`**, which `train.py` did not accept. | argparse: unrecognized arguments. | The 4-modality GPU loop could not run; the notebook printed the hand-written report table instead. |
| 9 | **Exported model was a 2-epoch smoke run.** | `data/exports/model_seed42.pt` config `epochs: 2`. | No fully trained model was ever exported. |

The tables in report 2 also could not be reproduced. A clean rerun of the *unfixed* code (4 modalities × 3 seeds) gave Hybrid-kNN NDCG@20 = 0.166 (reported 0.216), Popularity 0.086 (reported 0.183), and Random/Popularity rows identical to the old zero-SMILES smoke run.

## 2. Fixes applied

1. Hub-only mode uses FlavorGraph's CSV Recipe1M NPMI edges (walk corpus disabled in `configs/default.yaml`).
2. Walk loader skips self-pairs (kept for anyone re-enabling walks).
3. Evaluation counts a pair as a true partner only if its score is > 0 and it is not a self-pair.
4. BPR is computed on raw logits (`LinkPredictionHead(..., logits=True)`); ranking at test time is unchanged (sigmoid is monotone).
5. BPR negatives are anchored: each positive (u, v) is compared with 4 sampled (u, w), in both directions; `neg_sample_ratio: 4` added to the config.
6. Pseudo-edges and kNN neighbours exclude all val/test ingredients.
7. `--epochs` added to the `train.py` CLI.

All 19 tests in `tests/run_tests.py` pass on the patched code.

## 3. Corrected setup

- 416 hub ingredients; 311 eligible (CSV degree ≥ 5).
- Split per seed: 339 train / 31 val / 46 test ingredients; 1,043 training edges, 562 test ground-truth pairs.
- Candidates: all 415 other hubs. Random MRR ≈ 0.10, so metrics now discriminate.
- Seeds 42, 123, 456; values are mean ± std across seeds.

## 4. Corrected results

Baselines that do not depend on modality:

| Model | Recall@10 | Recall@20 | MRR | NDCG@20 | Spearman |
|---|---|---|---|---|---|
| Random | 0.020 ± 0.003 | 0.041 ± 0.003 | 0.097 ± 0.018 | 0.029 ± 0.003 | −0.022 ± 0.025 |
| Popularity | 0.058 ± 0.015 | 0.114 ± 0.018 | 0.150 ± 0.042 | 0.083 ± 0.021 | 0.101 ± 0.057 |

Per modality:

| Modality | Method | Recall@10 | Recall@20 | MRR | NDCG@20 | Spearman |
|---|---|---|---|---|---|---|
| Molecular | kNN | **0.088** ± 0.007 | **0.148** ± 0.011 | **0.239** ± 0.011 | **0.127** ± 0.010 | 0.047 ± 0.069 |
| | Zero-degree GNN | 0.032 ± 0.013 | 0.064 ± 0.025 | 0.130 ± 0.055 | 0.050 ± 0.022 | 0.091 ± 0.063 |
| | Strategy III | 0.032 ± 0.008 | 0.060 ± 0.007 | 0.127 ± 0.019 | 0.045 ± 0.008 | 0.067 ± 0.038 |
| Semantic | kNN | **0.111** ± 0.003 | **0.187** ± 0.005 | **0.293** ± 0.041 | **0.165** ± 0.014 | **0.161** ± 0.092 |
| | Zero-degree GNN | 0.033 ± 0.013 | 0.063 ± 0.014 | 0.137 ± 0.034 | 0.045 ± 0.014 | −0.044 ± 0.045 |
| | Strategy III | 0.030 ± 0.007 | 0.057 ± 0.017 | 0.126 ± 0.026 | 0.041 ± 0.014 | −0.057 ± 0.018 |
| Hybrid | kNN | **0.099** ± 0.004 | **0.161** ± 0.017 | **0.274** ± 0.014 | **0.142** ± 0.011 | 0.055 ± 0.074 |
| | Zero-degree GNN | 0.036 ± 0.014 | 0.065 ± 0.019 | 0.128 ± 0.029 | 0.047 ± 0.013 | 0.057 ± 0.017 |
| | Strategy III | 0.021 ± 0.006 | 0.046 ± 0.013 | 0.101 ± 0.012 | 0.036 ± 0.009 | 0.011 ± 0.024 |
| Taxonomic | kNN | **0.127** ± 0.012 | **0.189** ± 0.009 | **0.340** ± 0.055 | **0.184** ± 0.029 | **0.142** ± 0.062 |
| | Zero-degree GNN | 0.024 ± 0.011 | 0.056 ± 0.025 | 0.089 ± 0.018 | 0.039 ± 0.015 | 0.065 ± 0.057 |
| | Strategy III | 0.035 ± 0.007 | 0.062 ± 0.020 | 0.131 ± 0.016 | 0.048 ± 0.009 | 0.032 ± 0.050 |

## 5. Findings

1. **Every kNN baseline beats every GNN**, in every modality, on Recall, MRR and NDCG — by roughly 2.5–4× on Recall@10.
2. **The GNN does not beat Popularity.** With V3 training, Strategy III Recall@10 is 0.021–0.035 vs Popularity 0.058. With the original V2 training it reaches 0.021–0.060 — at best equal to Popularity on Recall@10 and still below it on NDCG@20 (see §6).
3. **Pseudo-edges do not reliably help.** Strategy III vs zero-degree GNN is within noise for molecular, semantic and taxonomic, and worse for hybrid. Under V2 training the zero-degree GNN matches or beats Strategy III in every modality.
4. **With V3 training the GNN overfits to warm nodes immediately.** Training loss falls (0.84 → 0.60) while validation loss on cold nodes rises (0.69 → 0.82); the best checkpoint is epoch 1–3 in all 12 runs. Running 300 epochs with no early stopping gives the same best epoch and identical test metrics. V2 training selects later checkpoints (epochs 6–33) but, as §6.3 explains, most likely by learning popularity.
5. **Among kNN modalities, taxonomic > semantic > hybrid > molecular** on NDCG@20. The previous claim that hybrid is best and that molecular ≈ semantic does not hold under the corrected ground truth.

**Conclusion:** on FlavorGraph hub ingredients, transferring known recipe neighbours from similar ingredients (kNN) is a strong cold-start method, and Strategy III's GNN adds nothing over it under either the V2 or the V3 training setup. This answers the TA's question directly: with this formulation, the GNN is not justified.

## 6. Comparison with the Version 2 architecture (`afml_jackfruit_final.zip`)

The original Version 2 code (the source of `versioned_report_2.md`) was extracted from `afml_jackfruit_final.zip` and benchmarked separately. Its source files are identical to the project before `fixes.patch`. All 19 tests pass on both versions, so the test suite does not distinguish them.

### 6.1 Version 2 as shipped (walk-graph ground truth)

| Method (molecular shown; other modalities identical in pattern) | Recall@10 | Recall@20 | MRR | NDCG@20 |
|---|---|---|---|---|
| Random | 0.024 | 0.048 | **1.000** | 0.020 |
| Popularity | 0.024 | 0.048 | 0.993 | 0.086 |
| Molecular-kNN | 0.024 | 0.048 | 1.000 | 0.155 |
| Strategy III | 0.024 | 0.048 | 1.000 | 0.020 |

Every method, including Random, sits at the Recall/MRR ceiling. This protocol cannot rank methods, and its results should not be reported.

### 6.2 Both versions on the corrected ground truth

The Version 2 code was rerun with only the data source changed (`walk_corpus: null`), so both versions face the same test: same split (339/31/46), same 562 test pairs, same candidates. The kNN, Random and Popularity rows are identical across versions; only the GNN training differs.

| Modality | kNN | GNN — V2 training | GNN — V3 training |
|---|---|---|---|
| | Recall@10 / NDCG@20 | Recall@10 / NDCG@20 | Recall@10 / NDCG@20 |
| Taxonomic | **0.127 / 0.184** | 0.060 / 0.076 | 0.035 / 0.048 |
| Semantic | **0.111 / 0.165** | 0.059 / 0.074 | 0.030 / 0.041 |
| Hybrid | **0.099 / 0.142** | 0.033 / 0.045 | 0.021 / 0.036 |
| Molecular | **0.088 / 0.127** | 0.021 / 0.039 | 0.032 / 0.045 |
| *Popularity (reference)* | *0.058 / 0.083* | | |
| *Random (reference)* | *0.020 / 0.029* | | |

V2 zero-degree GNN on the same test: taxonomic 0.067 / 0.088, semantic 0.073 / 0.084, hybrid 0.048 / 0.057, molecular 0.024 / 0.038 — i.e. without pseudo-edges it does as well as or better than V2 Strategy III.

### 6.3 Why V2's GNN training scores higher

V2 training differs from V3 in three ways: sigmoid outputs inside the BPR loss, 1:1 negatives, and **unanchored** negatives (random hub pairs unrelated to the positive's node). An ablation on V3 code (3 seeds × 4 modalities each) isolates the first two:

| V3 variant | Strategy III Recall@10 range across modalities |
|---|---|
| 4:1 anchored, raw logits (V3 default) | 0.021 – 0.035 |
| 1:1 anchored, raw logits | 0.020 – 0.046 |
| 4:1 anchored, sigmoid | 0.030 – 0.045 |
| 1:1 anchored, sigmoid | 0.025 – 0.041 |
| *V2 (1:1 unanchored, sigmoid)* | *0.021 – 0.060* |

Neither the negative ratio nor the sigmoid explains the gap; the remaining difference is anchoring. With unanchored negatives, a positive pair (which tends to involve frequently co-occurring hubs) is compared with an arbitrary pair, so the model can lower its loss by learning which ingredients are popular. V2's semantic and taxonomic GNN land almost exactly on the Popularity baseline (Recall@10 0.059 / 0.060 vs 0.058), consistent with it having learned popularity rather than ingredient-specific pairing. This is an interpretation of the numbers; it was not tested directly (e.g., by correlating GNN scores with node degree).

### 6.4 Which version is better

| Aspect | Better | Reason |
|---|---|---|
| Evaluation protocol | **V3** | V2's ground truth makes every candidate a partner (Random MRR = 1.000); V3's discriminates (Random MRR = 0.097). |
| Leakage hygiene | **V3** | V3 removes self-pairs and zero-score partners, and stops pseudo-edges/kNN from pointing at other held-out ingredients. |
| Reproducibility | **V3** | V3's `--epochs` flag makes the Colab notebook runnable; V2's reported tables do not reproduce. |
| GNN scores on the fair test | V2 (modestly) | Most likely by learning popularity; it still does not beat Popularity on NDCG@20 or any kNN. |
| Best method overall | Same in both | **Taxonomic-kNN** (Recall@10 0.127, NDCG@20 0.184), with all four kNN variants ahead of every GNN. |

**Recommendation:** report results under the V3 evaluation. Present kNN as the method that works, and report both GNN trainings, noting that the higher V2 numbers coincide with the Popularity baseline.

## 7. Caveats that still apply

- **Semantic modality is not Wikipedia text.** 289 of 416 descriptions are the identical template "X is a culinary food ingredient utilized across global cuisines…"; the rest are short hand-written sentences. Semantic-kNN mostly measures name similarity (e.g., *cheddar_cheese* ↔ *cheese*).
- **Taxonomic modality is not FoodOn.** It is a hand-written rules table (`TAXONOMIC_RULES` in `src/data/taxonomic_lookup.py`) plus keyword fallback; e.g., jackfruit is filed under vegetable. Its strong result is for this hand-built hierarchy.
- **Molecular fingerprints are degenerate.** The 416 hubs have only 300 distinct pooled-fingerprint rows; 33% of molecular top-k weights are ≥ 0.99, linking unrelated foods (e.g., agave = abalone = acorn). Fixing this needs RDKit to recompute fingerprints, which was not available for this rerun.
- **Small test set.** 46 test ingredients and 562 ground-truth pairs per seed; standard deviations are large for Spearman.
- **`SavorLinkPredictor.predict_new_compounds()` is still a stub** that ignores its compound IDs. It should not be demonstrated as a working feature.
- **Hybrid similarity** treats a neighbour missing from one modality's top-10 as 0, which biases against neighbours found by only one modality.
