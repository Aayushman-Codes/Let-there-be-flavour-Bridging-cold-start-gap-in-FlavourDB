# SavorLink Evaluation Protocol & Methodology Reference

This document provides a detailed specification of the evaluation protocol, ranking metrics, candidate pool formulations, and the critical methodological transition between Version 2 and Version 3.

---

## 1. Zero-Degree Cold-Start Evaluation Setup

In inductive graph learning, the zero-degree cold-start problem represents the challenge of predicting links for nodes that have **zero edges** in the training graph.

```
       Training Graph (339 Nodes)              Held-out Test Node (Cold)
        [Hub A] ─── [Hub B]                                [Cold Hub u]
           │          │                                   (Zero edges!)
        [Hub C] ─── [Hub D]                                     │
                 ▲                                              ▼
                 │                                   Predict affinity scores
                 └──────── Top-415 Hub Candidates ──────────────┘
```

### Partitioning Scheme
- Evaluated on the 416 hub ingredients of FlavorGraph (ingredients annotated with chemical compounds).
- Nodes with degree $\ge 5$ in Recipe1M are eligible for evaluation (311 eligible hubs).
- Split per seed ($42, 123, 456$):
  - **Training Set:** 339 ingredients, 1,043 internal Recipe1M edges
  - **Validation Set:** 31 ingredients (used for early stopping)
  - **Test Set:** 46 ingredients (held out exclusively for evaluation)
- Every edge touching any validation or test ingredient is severed from the graph prior to training.

### Candidate Ranking Pool
- When evaluating a held-out test ingredient $u$, the model must rank all other **415 hub ingredients** as potential pairing candidates.
- Candidate set size: $N = 415$.
- Pseudo-edges and nearest-neighbor lookups for test nodes are strictly prevented from pointing to other held-out test or validation nodes.

---

## 2. Metric Formulations

For each cold-start test ingredient $u$, let:
- $\mathcal{T}_u$ denote the set of true ground-truth recipe partners (NPMI $> 0$ in Recipe1M).
- $\hat{\mathbf{r}}_u = [r_1, r_2, \dots, r_N]$ denote the candidates ranked in descending order of predicted score $\hat{s}(u, c)$.
- $\text{rank}(c)$ be the 1-based index of candidate $c$ in $\hat{\mathbf{r}}_u$.

### 1. Recall@K
Fraction of true partners retrieved in the top $K$ recommendations:
$$\text{Recall}@K = \frac{|\{c \in \mathcal{T}_u : \text{rank}(c) \le K\}|}{|\mathcal{T}_u|}$$
Evaluated at $K \in \{10, 20\}$.

### 2. Mean Reciprocal Rank (MRR)
Measures how quickly the system finds the *first* correct partner:
$$\text{MRR} = \frac{1}{|U_{\text{test}}|} \sum_{u \in U_{\text{test}}} \frac{1}{\min_{c \in \mathcal{T}_u} \text{rank}(c)}$$

### 3. Normalized Discounted Cumulative Gain (NDCG@K)
Evaluates graded relevance using the true Recipe1M NPMI scores $y_{uc} > 0$ as relevance weights:
$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{y_{u, r_i}} - 1}{\log_2(i + 1)}$$
$$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
where $\text{IDCG}@K$ is the ideal DCG obtained by sorting true partners by their ground-truth NPMI score descending.

### 4. Spearman Rank Correlation ($\rho$)
Calculates Spearman's rank correlation coefficient between predicted scores $\hat{s}(u, c)$ and ground-truth NPMI scores $y_{uc}$ over all true partners $c \in \mathcal{T}_u$:
$$\rho = 1 - \frac{6 \sum_{c \in \mathcal{T}_u} d_c^2}{|\mathcal{T}_u|(|\mathcal{T}_u|^2 - 1)}$$
where $d_c = \text{rank}_{\text{pred}}(c) - \text{rank}_{\text{true}}(c)$.

---

## 3. Protocol Transition: Version 2 vs Version 3

In earlier iterations of this project (documented in `reports/versioned_reports.md` and `reports/versioned_report_2.md`), results indicated that Strategy III achieved high Recall and MRR. A subsequent methodological audit revealed several critical flaws that were corrected in Version 3.

### Flaws Identified in Version 2

| Issue | Technical Cause | Consequence |
|---|---|---|
| **Graph Saturation** | Metapath random-walk corpus generated 86,680 co-occurrence edges across 416 hubs (out of 86,320 possible pairs). | Every test ingredient had $\approx 417$ "true partners" among 415 candidates. Recall and MRR hit their mathematical ceiling for *every* method (Random MRR = 1.000). |
| **Self-Pair Ingestion** | Walk context window of size 5 counted ingredients revisited within 5 steps as pairing with themselves. | Ground-truth was polluted with self-edges. |
| **Zero-Score Edges** | Pairs with PPMI $\le 0$ were retained in the edge list and treated as positive hits during evaluation. | Non-pairings were scored as successful retrievals. |
| **Target Leakage** | Walks followed `hub -> compound -> hub` paths; 84% of hub pairs share at least one chemical compound. | Molecular features were predicting a walk graph that was fundamentally constructed from molecular overlap. |
| **BPR Sigmoid Squashing** | BPR loss was computed on sigmoid-squashed outputs $\sigma(s_1) - \sigma(s_2) \in (-1, 1)$ rather than unconstrained logits. | Logit gradients vanished; validation loss stalled near random guess (0.69). |
| **Unanchored Negatives** | Negatives were drawn as independent random pairs $(a, b)$ instead of pairs sharing the anchor node $(u, w)$. | Model learned global ingredient popularity rather than personalized pairing affinity. |
| **Test Set Leakage in Pseudo-Edges** | Cold node pseudo-edge construction included other test/validation nodes as nearest neighbors. | Information leaked between cold test nodes. |

### Rigorous Corrections Applied in Version 3

1. **Unpolluted Ground Truth:** Replaced walk corpus with FlavorGraph's official Recipe1M NPMI co-occurrence CSV edges (`edges_191120.csv`).
2. **Leak-Free Ground Truth Filter:** Only edges with $\text{NPMI} > 0$ and $u \ne v$ are retained as true pairing partners.
3. **Strict Candidate Exclusion:** Pseudo-edges and kNN nearest neighbors strictly exclude any node belonging to the validation or test splits.
4. **Logit-Space Anchored BPR:** BPR ranking loss is computed directly on raw logits with 4 anchored negative candidates $(u, w^-)$ per positive $(u, v^+)$:
   $$\mathcal{L}_{\text{BPR}} = -\sum_{(u, v^+)} \sum_{k=1}^4 \ln \sigma(s(u, v^+) - s(u, w^-_k))$$
5. **Multi-Seed Determinism:** Benchmark evaluates all models across 3 seeds ($42, 123, 456$) with explicit variance reporting.

---

## 4. Fair Comparison: V2 vs V3 on Unpolluted Ground Truth

When both architectures are benchmarked against the unpolluted Recipe1M ground truth under identical test partitions:

| Modality | kNN Baseline | GNN (V2 Training) | GNN (V3 Training) |
|---|---|---|---|
| | Recall@10 / NDCG@20 | Recall@10 / NDCG@20 | Recall@10 / NDCG@20 |
| **Taxonomic** | **0.127 / 0.184** | 0.060 / 0.076 | 0.035 / 0.048 |
| **Semantic** | **0.111 / 0.165** | 0.059 / 0.074 | 0.030 / 0.041 |
| **Hybrid** | **0.099 / 0.142** | 0.033 / 0.045 | 0.021 / 0.036 |
| **Molecular** | **0.088 / 0.127** | 0.021 / 0.039 | 0.032 / 0.045 |
| *Popularity (Reference)* | *0.058 / 0.083* | — | — |
| *Random (Reference)* | *0.020 / 0.029* | — | — |

### Takeaway
Under clean, discriminating ground truth, **nearest-neighbor baselines (kNN) decisively outperform the GNN across all modalities by 2.7x to 4.6x on Recall@10**. The higher numbers produced by V2 training coincide almost exactly with the Popularity baseline (0.060 vs 0.058), confirming that unanchored negative sampling caused the GNN to learn general ingredient popularity rather than specific pairing affinities.
