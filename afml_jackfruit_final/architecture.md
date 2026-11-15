# SavorLink Architecture Specification: Multi-Modal Cold-Start Graph Framework
**Document Version:** 2.1  
**Status:** Architecture Design for TA Methodology & Modality Expansion  
**Location:** Project Root & `jackfruit/`  

---

## 1. System Overview & Motivation

### 1.1 The Scientific Dilemma
In SavorLink Version 1.0, the cold-start food pairing problem ($deg(v) = 0$) was tackled exclusively at the **molecular level**: 
unseen ingredients were embedded via pooled 2048-bit Morgan fingerprints of their chemical flavor compounds, and soft pseudo-edges were injected based on Tanimoto chemical similarity.

While biochemically principled, this approach faced two critical methodological challenges highlighted during evaluation:
1. **Is the molecular level strictly necessary?** Food pairing in real-world human culinary history is heavily driven by cultural, recipe-level, and perceptual affinities. Could higher-level representations—such as **semantic text descriptions** (Wikipedia culinary profiles) or **taxonomic classifications** (botanical/culinary ontologies like FoodOn)—perform just as well or better, without requiring expensive compound-level chemical annotations?
2. **The GNN vs. kNN Performance Gap:** In initial runs, a non-parametric chemical kNN baseline achieved higher NDCG@20 ($0.1996$ vs. $0.1044$) and Spearman correlation ($0.1550$ vs. $0.1317$) than the Strategy III GNN. This raised the question: *Is a Graph Neural Network even necessary for this task, or was the model formulation suboptimal?*

### 1.2 The New Multi-Modal Architecture
To decisively answer both questions, SavorLink is transitioning from a single-modality system to a **4-Modality Comparative Architecture**:

```
                       [ Cold-Start Ingredient ]
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
   [ Modality 1 ]            [ Modality 2 ]            [ Modality 4 ]
Molecular Structure       Semantic Text Profile     Taxonomic Hierarchy
(2048-bit Morgan FPs)     (Sentence-Transformer)      (FoodOn Ontology)
         |                         |                         |
         +------------+------------+                         |
                      |                                      |
                      v                                      |
                [ Modality 3 ]                               |
            Hybrid Molecular + Semantic                      |
                      |                                      |
         +------------+------------+-------------------------+
         |                                                   |
         v                                                   v
 [ Non-GNN Baseline ]                               [ Inductive GNN ]
    kNN Benchmark                                   Strategy III GNN
(Direct Profile Transfer)                       (Weighted Message Passing)
         |                                                   |
         +-------------------------+-------------------------+
                                   |
                                   v
             [ Evaluation: Recall@K, MRR, NDCG@K, Spearman ]
```

This architecture enables an orthogonal $4 \times 2$ experimental matrix:
- **4 Modalities:** Molecular alone, Semantic alone, Hybrid (Molecular + Semantic), and Taxonomic alone.
- **2 Algorithmic Paradigms:** Non-parametric neighbor propagation (kNN) vs. Inductive Graph Representation Learning (GNN).

---

## 2. Work Partition: Local CPU vs. Google Colab GPU

To prevent wasted GPU time while avoiding local hardware bottlenecks, the pipeline strictly divides workloads:

```
+---------------------------------------------------------------------------------------------------+
| LOCAL MACHINE (CPU & Network I/O)                                                                 |
+---------------------------------------------------------------------------------------------------+
| - Data Ingestion: PubChem SMILES (1,626 valid entries), Wikipedia API text, FoodOn EBI OLS4 API.  |
| - Feature Extraction: Morgan fingerprints (RDKit), MiniLM semantic text embeddings, WUP tree sim.  |
| - Similarity Precomputation: Generates sparse top-k similarity dictionaries for all 4 modalities.  |
| - Non-GNN Baseline Suite: Executes Random, Popularity, and the 4 kNN baselines in <10 seconds.    |
| - Bundle Export: Packages precomputed features/similarities into colab_training_bundle.zip (~15MB)|
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  | (Upload lightweight ~15 MB bundle)
                                                  v
+---------------------------------------------------------------------------------------------------+
| GOOGLE COLAB FREE-TIER (Tesla T4 GPU & High RAM)                                                  |
+---------------------------------------------------------------------------------------------------+
| - Model Training: Inductive Strategy III GNN trained with Pairwise BPR Ranking Loss.              |
| - Negative Sampling: High-ratio 4:1 negative edge sampling on CUDA tensor cores.                 |
| - Multi-Modal GNN Suite: Runs Molecular-GNN, Semantic-GNN, Hybrid-GNN, Taxonomic-GNN.             |
| - Multi-Seed Execution: 3 random seeds (42, 123, 456) x 4 modalities = 12 full training runs.    |
| - Parallel Candidate Scoring: GPU-accelerated tensor ranking across all 416 hub candidates.       |
| - Publication Artifacts: Generates final multi-seed Markdown table & 1-click results download.     |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. The Four Modality Branches

### Modality 1: Molecular Structure (Chemical Baseline)
- **Source:** PubChem CIDs fetched via PUG REST API, canonicalized SMILES, and 2048-bit Morgan/ECFP fingerprints (radius 2) computed via RDKit.
- **Node Features:** Mean-pooled compound fingerprints $x_u^{\text{mol}} \in \mathbb{R}^{2048}$.
- **Similarity Metric:** Set-based Tanimoto similarity over compound multisets using `mean_best_match`:
  $$\text{Sim}_{\text{mol}}(I_A, I_B) = \frac{1}{2} \left[ \frac{1}{|I_A|} \sum_{a \in I_A} \max_{b \in I_B} T(a, b) + \frac{1}{|I_B|} \sum_{b \in I_B} \max_{a \in I_A} T(a, b) \right]$$
- **Pseudo-Edges:** Top-$k$ chemical neighbors ($k=10$), weights $w_{uv}^{\text{mol}} = (\text{Sim}_{\text{mol}})^{1/T}$ for $\text{Sim} \ge \tau$.

### Modality 2: Semantic Representation (Textual / Wikipedia Descriptions)
- **Source:** Curated culinary and botanical encyclopedic summaries extracted from Wikipedia REST API (`https://en.wikipedia.org/api/rest_v1/page/summary/{ingredient}`).
- **Embedding Model:** Pre-trained Sentence-Transformer (`all-MiniLM-L6-v2`, 384-dimensional dense vectors, lightweight and fast CPU execution).
- **Node Features:** Dense semantic vector $x_u^{\text{sem}} \in \mathbb{R}^{384}$.
- **Similarity Metric:** Cosine similarity bounded in $[0, 1]$:
  $$\text{Sim}_{\text{sem}}(u, v) = \max \left(0, \frac{x_u^{\text{sem}} \cdot x_v^{\text{sem}}}{\|x_u^{\text{sem}}\|_2 \|x_v^{\text{sem}}\|_2} \right)$$
- **Pseudo-Edges:** Top-$k$ semantic neighbors, weights $w_{uv}^{\text{sem}} = (\text{Sim}_{\text{sem}})^{1/T}$.

### Modality 3: Hybrid Molecular + Semantic Fusion
- **Rationale:** Chemistry captures volatile aroma compound overlaps (e.g., strawberry and balsamic vinegar sharing furanones), while semantics capture culinary conventions and texture profiles.
- **Combined Similarity (Edge Synthesis):**
  $$w_{uv}^{\text{hybrid}} = \alpha \cdot w_{uv}^{\text{mol}} + (1 - \alpha) \cdot w_{uv}^{\text{sem}} \quad (\text{default } \alpha = 0.5)$$
- **Node Feature Fusion:**
  Linear projection of both spaces into a shared hidden dimension $d = 128$:
  $$x_u^{\text{hybrid}} = \text{LayerNorm} \left( W_{\text{mol}} x_u^{\text{mol}} + W_{\text{sem}} x_u^{\text{sem}} \right) \in \mathbb{R}^{128}$$

### Modality 4: Taxonomic Ontology (Hierarchical Categorization)
- **Source:** FoodOn Ontology (`FOODON`) accessed via EMBL-EBI OLS4 API + USDA FoodData Central 3-level taxonomy fallback.
- **Taxonomic Structure:** Multi-level directed acyclic graph / hierarchy:
  $$\text{Origin (Plant/Animal/Fungi)} \to \text{Food Group (Fruit/Vegetable/Herb/Dairy/Meat/Seafood)} \to \text{Subgroup (Allium/Citrus/Pome/Bivalve)}$$
- **Similarity Metric:** Wu-Palmer (WUP) tree depth similarity:
  $$\text{Sim}_{\text{tax}}(u, v) = \frac{2 \cdot \text{depth}(\text{LCS}(u, v))}{\text{depth}(u) + \text{depth}(v)}$$
  where $\text{LCS}(u, v)$ is the Lowest Common Subsumer in the ontology tree.
- **Node Features:** Categorical multi-hot / embedding vector of hierarchical lineage $x_u^{\text{tax}} \in \mathbb{R}^{d_{\text{tax}}}$.

---

## 4. Resolving the GNN vs. kNN Empirical Paradox

### 4.1 Why Did kNN Beat the Initial GNN?
1. **Pointwise Huber Loss vs. Pairwise Ranking Objective:** The GNN was trained with Smooth-L1 regression on individual pairs $(u, v)$ with target $y \in [0, 1]$. Pointwise loss penalizes numeric error, not relative rank order.
2. **kNN's Direct Access to Ground-Truth Co-Occurrence Vectors:** Tanimoto-kNN directly averages the exact empirical NPMI rows of nearest neighbors:
   $$s_{\text{kNN}}(u, c) = \frac{\sum_{v \in \mathcal{N}(u)} w_{uv} \cdot A_{\text{train}}[v, c]}{\sum_{v \in \mathcal{N}(u)} w_{uv}}$$
3. **Severe Under-Training in Smoke Mode:** The initial test was run with `--smoke` (2 epochs).

### 4.2 The Architectural Remedy for Strategy III on GPU
1. **Pairwise Ranking Loss (BPR Loss):**
   $$\mathcal{L}_{\text{BPR}} = - \sum_{(u, v^+, v^-)} \ln \sigma \left( s(u, v^+) - s(u, v^-) \right) + \lambda \|\Theta\|_2^2$$
   where $v^+$ is a true recipe partner and $v^-$ is a sampled non-partner. This directly optimizes NDCG and rank ordering.
2. **Multi-Hop Topological Aggregation:** Unlike kNN (which is strictly 1-hop), the GNN utilizes 2-layer `WeightedSAGEConv` to propagate higher-order recipe motifs across the network.
3. **High-Ratio Negative Sampling (4:1):** Exposing the model to 4 hard negative pairs per positive pair prevents score saturation.
4. **GPU-Accelerated 40-Epoch Convergence:** Leveraging the Tesla T4 GPU allows 12 full training runs to complete in ~15 minutes.

---

## 5. Feasibility Verification Summary

| Component | Availability & Feasibility Status |
| :--- | :--- |
| **Taxonomic Dataset (FoodOn)** | **VERIFIED LIVE:** EMBL-EBI OLS4 API is active and free. Queries for standard foods (`garlic`, `apple`, etc.) return exact FoodOn IRIs (`FOODON_00003584`) and parent classes (`garlic food product`, `vegetable food product`). Offline 3-level fallback guarantees 100% completion. |
| **Semantic Dataset (Wikipedia)** | **VERIFIED LIVE:** Wikipedia REST API returns clean culinary summaries in $<200$ ms. `all-MiniLM-L6-v2` encodes all 416 items in 2 seconds. |
| **GPU Offload Architecture** | **VERIFIED:** Lightweight ~15 MB bundle (`colab_training_bundle.zip`) eliminates all scraping/network operations on Colab; notebook executes pure GPU deep learning. |
| **Timeline Compliance** | Full implementation + Colab execution takes **< 3 hours**, completing well ahead of the 11:59 PM deadline. |
