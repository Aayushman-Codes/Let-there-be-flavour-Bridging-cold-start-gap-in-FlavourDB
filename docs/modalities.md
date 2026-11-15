# SavorLink Multi-Modal Representations Reference

This document details the four representation modalities implemented in SavorLink for cold-start ingredient pairing on FlavorGraph.

---

## Modality Overview

| Modality | Dimension | Primary Feature Source | Similarity Metric | Aggregation / Computation |
|---|---|---|---|---|
| **Molecular** | 2048-d | PubChem SMILES + 2048-bit Morgan Fingerprints | Set-based Tanimoto | Mean Best Match across compound sets |
| **Semantic** | 384-d | Wikipedia / Curated Text Descriptions | Cosine Similarity | `all-MiniLM-L6-v2` dense embeddings |
| **Taxonomic** | 141-d | 3-tier Categorical Food Hierarchy | Wu-Palmer Similarity | Multi-hot tree encoding + depth LCS |
| **Hybrid** | 2432-d | Concatenated Molecular + Semantic | Blended Metric | $0.5 \times \text{Tanimoto} + 0.5 \times \text{Cosine}$ |

---

## 1. Molecular Modality (2048-d)

### Concept
Ingredients possess flavor by virtue of their chemical compound composition. When two ingredients share key volatile aroma compounds or compounds with high structural similarity, they often exhibit complementary flavor profiles according to the food pairing hypothesis.

### Implementation
- **Source Files:** `src/chem/fingerprints.py`, `src/chem/similarity.py`, `src/data/pubchem_lookup.py`
- **Feature Extraction:**
  1. For each ingredient, retrieve the PubChem CIDs of all connected food flavor compounds (`ingr-fcomp`) and drug-like compounds (`ingr-dcomp`).
  2. Query PubChem REST APIs to fetch canonical SMILES strings.
  3. Use RDKit to generate 2048-bit Morgan circular fingerprints with radius $r = 2$ (equivalent to ECFP4):
     $$\mathbf{f}_c \in \{0, 1\}^{2048}$$
  4. Aggregate compound fingerprints for ingredient $u$ using continuous mean pooling:
     $$\mathbf{x}_u^{\text{chem}} = \frac{1}{|C_u|} \sum_{c \in C_u} \mathbf{f}_c \in [0, 1]^{2048}$$
- **Pairwise Similarity:**
  To preserve compound-level specificity and avoid pooling wash-out, similarity between ingredients $A$ and $B$ is computed via **Mean Best Match (MBM)** Tanimoto across their raw compound sets $C_A$ and $C_B$:
  $$\text{Sim}_{\text{MBM}}(A, B) = \frac{1}{2|C_A|} \sum_{c_i \in C_A} \max_{c_j \in C_B} T(c_i, c_j) + \frac{1}{2|C_B|} \sum_{c_j \in C_B} \max_{c_i \in C_A} T(c_i, c_j)$$
  where $T(a, b) = \frac{|a \cap b|}{|a \cup b|}$ is standard binary Tanimoto coefficient.

---

## 2. Semantic Modality (384-d)

### Concept
Culinary ingredients are described by culinary roles, origins, cooking preparations, and sensory descriptors in natural language. Semantic embeddings capture contextual and linguistic affinities between culinary items.

### Implementation
- **Source Files:** `src/chem/semantic_embeddings.py`, `src/data/semantic_lookup.py`
- **Text Retrieval:**
  - Retrieves Wikipedia summaries and curated food descriptions for all 416 hub ingredients (cached in `data/cache/wikipedia_summaries.json`).
- **Embedding Model:**
  - Uses `sentence-transformers/all-MiniLM-L6-v2` (a 6-layer MiniLM model fine-tuned for semantic sentence matching).
  - Produces a dense 384-dimensional embedding vector $\mathbf{e}_u \in \mathbb{R}^{384}$.
  - Embeddings are strictly $L_2$-normalized: $\|\mathbf{e}_u\|_2 = 1.0$.
- **Pairwise Similarity:**
  - Standard cosine similarity between unit vectors:
    $$\text{Sim}_{\text{semantic}}(u, v) = \mathbf{e}_u^\top \mathbf{e}_v$$

---

## 3. Taxonomic Modality (141-d)

### Concept
Ingredients that share close culinary, botanical, or agricultural lineages (e.g., citrus fruits, cruciferous vegetables, dairy products) share culinary pairings and recipe functions.

### Implementation
- **Source Files:** `src/chem/taxonomic_similarity.py`, `src/data/taxonomic_lookup.py`
- **Ontology Hierarchy:**
  - Implements a 3-tier culinary food classification tree:
    1. **Origin:** Plant, Animal, Fungi, Mineral, Synthetic, Processed
    2. **Group:** Fruit, Vegetable, Herb, Spice, Meat, Seafood, Dairy, Grain, Legume, etc.
    3. **Subgroup:** Citrus, Berry, Allium, Cruciferous, Bovine, Poultry, etc.
- **Feature Representation:**
  - Converts hierarchical paths into 141-dimensional multi-hot vectors representing category memberships at all tree levels.
- **Pairwise Similarity:**
  - Evaluates Wu-Palmer ontological similarity:
    $$\text{Sim}_{\text{Wu-Palmer}}(u, v) = \frac{2 \cdot \text{depth}(\text{LCS}(u, v))}{\text{depth}(u) + \text{depth}(v)}$$
    where $\text{LCS}(u, v)$ is the Least Common Subsumer in the ontology tree, and $\text{depth}(\text{root}) = 1$.

---

## 4. Hybrid Modality (2432-d)

### Concept
Combines microscopic biochemical features (molecular fingerprints) with macro-level linguistic and culinary descriptors (semantic text embeddings).

### Implementation
- **Source Files:** `src/chem/hybrid_similarity.py`
- **Feature Concatenation:**
  $$\mathbf{x}_u^{\text{hybrid}} = [\mathbf{x}_u^{\text{chem}} \,\|\, \mathbf{x}_u^{\text{semantic}}] \in \mathbb{R}^{2432}$$
  Both components are $L_2$-normalized prior to concatenation so that neither sub-vector dominates gradients or distances.
- **Similarity Blending:**
  $$\text{Sim}_{\text{hybrid}}(u, v) = 0.5 \cdot \text{Sim}_{\text{MBM}}(u, v) + 0.5 \cdot \text{Sim}_{\text{semantic}}(u, v)$$

---

## Modality Performance Summary in Benchmark

Across 3 seeds (42, 123, 456) evaluating on 46 held-out cold-start hub ingredients:

| Modality | kNN Recall@10 | kNN NDCG@20 | GNN Recall@10 | GNN NDCG@20 | Best Overall Method |
|---|---|---|---|---|---|
| **Taxonomic** | **0.127 ± 0.012** | **0.184 ± 0.029** | 0.035 ± 0.007 | 0.048 ± 0.009 | **Taxonomic-kNN** |
| **Semantic** | 0.111 ± 0.003 | 0.165 ± 0.014 | 0.030 ± 0.007 | 0.041 ± 0.014 | Semantic-kNN |
| **Hybrid** | 0.099 ± 0.004 | 0.142 ± 0.011 | 0.021 ± 0.006 | 0.036 ± 0.009 | Hybrid-kNN |
| **Molecular** | 0.088 ± 0.007 | 0.127 ± 0.010 | 0.032 ± 0.008 | 0.045 ± 0.008 | Molecular-kNN |

### Key Observations:
1. **Taxonomy and Semantics Outperform Chemistry:**
   Taxonomic and Semantic representations significantly outperform Molecular fingerprints (0.184 and 0.165 NDCG@20 vs 0.127 for molecular kNN). Recipes are governed by cultural, functional, and category compatibility as much as volatile molecular overlap.
2. **Hybrid Modality Bottleneck:**
   Hybrid similarity treats a neighbor missing from one modality's top-10 list as 0, which introduces a penalty against neighbors strongly identified by only one representation. Consequently, pure taxonomic or pure semantic representations achieve superior retrieval.
