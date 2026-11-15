# SavorLink Strategy III: Tanimoto Pseudo-Edges for Cold-Start Ingredients
> **A complete guide to what this project is, why it exists, how it works, and what each part does.**

---

## 1. The Big Picture: What Problem Are We Solving?

### The Context
Imagine an AI that predicts which foods pair well together (e.g., *strawberries + balsamic vinegar*, *chocolate + blue cheese*, or *garlic + basil*). 
To do this, we use a dataset called **FlavorGraph**, which combines two types of information:
1. **Recipe Co-occurrence**: Which ingredients often appear together in real-world recipes.
2. **Flavor Chemistry**: Which chemical flavor compounds (molecules like vanillin, limonene, coumarin) are present inside each ingredient.

### The "Cold-Start" Dilemma
Standard Graph Neural Networks (like GraphSAGE or GAT) work by **passing messages between neighbors** in the recipe graph:
> *"Tell me who your recipe friends are, and I'll figure out what kind of ingredient you are."*

**The problem:** What happens if a chef introduces an unfamiliar, wild, or newly synthesized ingredient that has **zero existing recipe co-occurrences**?
In graph theory, this is an **isolated node (degree = 0)**. 
- It has no edges.
- No messages can flow into it.
- Standard GNN aggregators fail completely, reverting to blind guessing.

---

## 2. The Core Idea: What is Strategy III?

Instead of giving up on cold-start ingredients, **Strategy III** uses **chemistry as a bridge**:

```
[ Cold-Start Ingredient ]  <--- Has zero recipe edges!
         |
         |  (Step 1: Compute chemical similarity on Morgan fingerprints)
         v
    Find Top-k Chemically Similar "Hub" Ingredients (e.g., Vanilla, Cinnamon)
         |
         |  (Step 2: Inject soft, weighted "pseudo-edges")
         v
[ Injected Pseudo-Edges: Weight = Tanimoto Similarity (0.0 to 1.0) ]
         |
         |  (Step 3: GNN Message Passing)
         v
[ GraphSAGE / GAT Aggregator ] now has a rich neighborhood!
         |
         v
[ Link Prediction Head ] -> Predicts Recipe Pairing (NPMI Score)
```

By connecting the cold ingredient to its nearest chemical cousins with weighted pseudo-edges, the GNN aggregator receives a non-empty, chemically grounded neighborhood to propagate messages through.

---

## 3. Four Important Engineering Decisions

### 1. Ingredients Are Sets of Molecules (Not Single Molecules)
An apple isn't one molecule; it is a cocktail of dozens of volatile aroma compounds. We implemented and benchmarked **three strategies** to measure similarity between compound sets:
- **`mean_best_match` (Default)**: For every compound in Ingredient A, find its highest Tanimoto match in Ingredient B, and average them (symmetrized). This captures nuanced partial chemical overlap.
- **`max_pairwise`**: Takes the single highest-scoring compound pair. Fast and useful for checking if two foods share a dominant signature molecule.
- **`union_fingerprint`**: Computes continuous Tanimoto over mean-pooled soft fingerprints.

### 2. Strict Zero-Leakage Invariant
To honestly test a cold-start model:
- Splits are made at the **ingredient level** (not edge level).
- Held-out test ingredients have **ALL** recipe edges stripped away.
- Node features derive **strictly from chemistry** (2048-bit Morgan/ECFP fingerprints), never from graph structure or recipe co-occurrences.

### 3. Cold-Start-Aware Training
If a model only ever sees dense recipe graphs during training, it won't know how to rely on pseudo-edges at test time.
- **Solution:** Every training epoch, the model randomly hides all edges of a subset of training ingredients (simulating cold start), injects pseudo-edges, and optimizes the loss on their hidden real edges.

### 4. Google Colab Free-Tier Optimization
- We never compute an expensive $N \times N$ dense similarity matrix.
- Comparisons are batched and restricted to a pruned set of 416 hub ingredients.
- Peak RAM usage is just **27.2 MB** ($0.17\%$ of Colab's 16 GB limit), leaving plenty of room for GPU acceleration on a Tesla T4.

---

## 4. Codebase Tour: What Each File Does

```
savorlink/
├── configs/
│   └── default.yaml          # All hyperparameters (learning rate, top-k, Morgan bits, seeds)
│
├── data/                     # Raw CSVs and disk caches (PubChem SMILES)
│
├── src/
│   ├── data/
│   │   ├── loaders.py        # Loads FlavorGraph nodes & edges into clean typed dataclasses
│   │   ├── pubchem_lookup.py # Resumable PubChem API fetcher (with rate-limiting & disk caching)
│   │   └── splits.py         # Leakage-free train/val/test splits and epoch-wise cold masking
│   │
│   ├── chem/
│   │   ├── fingerprints.py   # RDKit 2048-bit Morgan fingerprint extraction & pooling (mean, union)
│   │   ├── similarity.py     # BLAS-accelerated implementation of the 3 Tanimoto set strategies
│   │   └── pooling.py        # Pooling convenience exports
│   │
│   ├── graph/
│   │   ├── pseudo_edges.py   # Injects top-k weighted edges with temperature scaling and thresholds
│   │   └── build_graph.py    # Formats graph connectivity and node feature matrices for PyTorch
│   │
│   ├── models/
│   │   ├── branches.py       # Weighted GraphSAGE and GAT layers scaling messages by Tanimoto weight
│   │   ├── heads.py          # Symmetric pairwise link-prediction head (predicts continuous NPMI)
│   │   └── strategy3.py      # End-to-end Strategy III neural network model
│   │
│   ├── baselines.py          # 5 evaluation baselines (Random, Popularity, kNN, Zero-degree, S3)
│   ├── evaluate.py           # Ranking metrics (Recall@10/20, MRR, NDCG@20, Spearman correlation)
│   └── train.py              # Cold-mask training loop and multi-seed benchmark runner
│
├── tests/
│   ├── run_tests.py          # Standalone test runner (16 unit tests, 100% passing)
│   ├── test_similarity.py    # Math property tests (identity = 1.0, symmetry, disjoint = 0.0)
│   ├── test_pseudo_edges.py  # Tests pseudo-edge generation, temperature scaling, and merging
│   ├── test_splits.py        # Leakage verification assertions
│   ├── test_evaluation.py    # Metric calculation tests
│   ├── test_models.py        # PyTorch layer and backprop tests
│   └── benchmark_memory.py   # Memory footprint and runtime profiler
│
├── notebooks/
│   └── colab_quickstart.ipynb# Google Colab walkthrough notebook (ready to run with 1 click)
│
└── README.md                 # Project documentation & setup instructions
```

---

## 5. The 5 Baselines & Why They Matter

When evaluating a cold-start method, deep learning must prove it actually helps. We compare Strategy III against 4 baselines on the exact same test ingredients:

| # | Model / Baseline | How It Works | Why We Test It |
|---|---|---|---|
| **1** | **Random** | Predicts uniform random pairing scores. | Establishes the absolute performance floor. |
| **2** | **Popularity (Degree)** | Recommends ingredients that appear most often in recipes (salt, butter, onion). | Checks if generic popularity beats smart chemistry. |
| **3** | **Tanimoto-kNN (Non-GNN)** | Averages the recipe co-occurrences of the cold node's nearest chemical hub neighbors without any neural network. | **Crucial baseline:** If the GNN can't beat this simple weighted average, we report that honestly. |
| **4** | **Zero-Degree GNN** | Evaluates the GNN with chemical features only and **zero** pseudo-edges. | Demonstrates the structural failure mode of standard GNNs on isolated nodes. |
| **5** | **Full Strategy III** | The complete GNN with weighted pseudo-edge neighborhood aggregation. | Proves that combining chemical pseudo-edges with deep graph aggregation outperforms all alternatives. |

---

## 6. How to Run It

### Quickest Way: Google Colab
Open `notebooks/colab_quickstart.ipynb` in Google Colab (Runtime $\to$ Change runtime type $\to$ T4 GPU). It downloads the data, sets up dependencies, runs training, and outputs the comparison table.

### In Your Terminal
```bash
# 1. Run the test suite (verifies everything works)
python tests/run_tests.py

# 2. Run the memory & runtime benchmark
python tests/benchmark_memory.py

# 3. Train and benchmark Strategy III across 3 random seeds
python src/train.py --config configs/default.yaml --seeds 42,123,456
```

## **Dataset Facts & Graph Statistics**

### FlavorGraph Architecture
- **8,298 total nodes**:
  - $6,653$ ingredients
  - $1,561$ flavor compounds
  - $84$ drug compounds
- **147,179 total CSV edges**:
  - $111,355$ ingredient–ingredient edges (NPMI co-occurrence scores in $[0.0, 0.86]$, all non-negative)
  - $35,440$ ingredient–flavor compound edges
  - $384$ ingredient–drug compound edges
- **Hub Structure**:
  - $416$ designated hub ingredients
  - $417$ ingredients own chemical compound annotations (mean $85.9$ compounds per ingredient, max $408$)
  - $1,645 / 1,645$ compounds ($100\%$) possess verified PubChem CIDs.
