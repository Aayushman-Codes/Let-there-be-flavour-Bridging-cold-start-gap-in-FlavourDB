# SavorLink: Cold-Start Food Pairing on FlavorGraph

SavorLink asks a simple question: **when a new ingredient arrives with no recipe history, can we predict what it pairs well with?** It compares an inductive graph neural network ("Strategy III") against simple similarity baselines on [FlavorGraph](https://github.com/lamypark/FlavorGraph), using four ways of describing an ingredient: chemistry, text, taxonomy, and a chemistry + text hybrid.

**Main result:** a simple nearest-neighbour method (find similar known ingredients, borrow their recipe partners) beats the GNN in every setting. The GNN does not beat a "recommend popular ingredients" baseline either. This is reported as an honest negative result — see [What we achieved](#what-we-achieved).

Course project for *Advanced Foundations of Machine Learning*, PES University.

---

## Quick start (Windows)

```bat
setup.bat
```

This creates a virtual environment, installs dependencies, runs the test suite, trains and evaluates all four modalities (about 1 minute on a CPU), prints a comparison of the 4 methods in the terminal, and opens the **frontend** at <http://localhost:8501>. Use `setup.bat results` to reprint the comparison, or `setup.bat app` to relaunch the frontend. If it fails, follow [setup.md](setup.md).

On macOS/Linux, or to run steps by hand:

```bash
pip install -r requirements.txt
python tests/run_tests.py                       # 19 tests
python -m src.run_benchmark                     # 4 modalities x 3 seeds -> results/
python -m src.compare                           # terminal comparison of the 4 methods
python -m streamlit run app/streamlit_app.py    # frontend
```

### The frontend

| Tab | Shows |
|---|---|
| Overview | The problem, dataset numbers, and the headline result |
| Benchmark | All 4 methods compared: bar chart per metric with error bars, baselines, leaderboard, head-to-head |
| Cold-start lab | One held-out ingredient: its real recipe neighbourhood (before), the isolated cold node, then each modality's pseudo-edges with kNN vs GNN recommendations and true partners marked (after) |
| Semantic context | The sentence-transformer description attached to each node, its embedding and semantic neighbours. You can type a new description and re-rank live (optional; needs `requirements-data.txt`) |

Everything is computed live from `data/` and `results/`.

---

## The problem

Graph neural networks learn about a node from its neighbours. A brand-new ingredient has **no edges** in the recipe graph, so a standard GNN has nothing to aggregate (the *zero-degree cold-start* problem).

**Strategy III** connects the cold ingredient to its most similar known "hub" ingredients with weighted **pseudo-edges**, so message passing has a neighbourhood to work with. Similarity comes from one of four modalities:

| Modality | How an ingredient is described | Similarity |
|---|---|---|
| Molecular | Mean-pooled 2048-bit Morgan fingerprints of its flavour compounds (PubChem SMILES) | Set-based Tanimoto (mean best match) |
| Semantic | A short text description embedded with `all-MiniLM-L6-v2` (384-d) | Cosine |
| Taxonomic | Position in a 3-level food hierarchy (origin → group → subgroup), 141-d multi-hot | Wu–Palmer |
| Hybrid | Molecular + semantic features concatenated (2432-d) | 0.5 × molecular + 0.5 × semantic |

Every method is evaluated the same way: held-out ingredients have **all** their recipe edges removed, and each method ranks the other 415 hub ingredients as candidate partners.

## What we achieved

1. **A leakage-free cold-start benchmark** on FlavorGraph's 416 hub ingredients: ingredient-level splits (339 train / 31 val / 46 test per seed), Recipe1M co-occurrence edges as ground truth, 3 seeds, five ranking metrics.
2. **Four ingredient representations** and a modality-agnostic kNN baseline and GNN, all driven by one config.
3. **A fully inductive GNN** (weighted GraphSAGE, symmetric link-prediction head, BPR ranking loss, cold-mask training), with no per-node embedding table.
4. **An honest finding.** Mean of 3 seeds (from [report 3](reports/versioned_report_3_corrected.md)):

| Modality | kNN Recall@10 | kNN NDCG@20 | Strategy III Recall@10 | Strategy III NDCG@20 |
|---|---|---|---|---|
| Taxonomic | **0.127** | **0.184** | 0.035 | 0.048 |
| Semantic | 0.111 | 0.165 | 0.030 | 0.041 |
| Hybrid | 0.099 | 0.142 | 0.021 | 0.036 |
| Molecular | 0.088 | 0.127 | 0.032 | 0.045 |
| *Popularity* | *0.058* | *0.083* | | |
| *Random* | *0.020* | *0.029* | | |

- kNN beats the GNN in every modality, by roughly 2.5–4× on Recall@10.
- Pseudo-edges do not reliably help: Strategy III is within noise of a GNN that gets no pseudo-edges at all.
- The GNN overfits the warm training graph within a few epochs; validation loss on cold ingredients never improves.
- Among kNN variants, taxonomic > semantic > hybrid > molecular.

So, for this data, the GNN is not justified: transferring known neighbours' recipe partners is the better cold-start method.

## Limitations

- **Semantic text is not Wikipedia.** 289 of 416 descriptions are the same template sentence; the rest are short hand-written ones. Semantic similarity therefore mostly reflects ingredient names.
- **The taxonomy is hand-built**, not taken from the FoodOn ontology (`TAXONOMIC_RULES` in `src/data/taxonomic_lookup.py`).
- **Molecular fingerprints are partly degenerate:** the 416 hubs have only 300 distinct pooled fingerprints, so some unrelated foods look chemically identical.
- **Small test set:** 46 ingredients and 562 true pairs per seed; Spearman varies a lot between seeds.
- `SavorLinkPredictor.predict_new_compounds()` is not implemented (it raises `NotImplementedError`).

## Repository layout

```
├── README.md                  this file
├── ARCHITECTURE.md            pipeline, modules, model and evaluation protocol
├── setup.bat / setup.md       one-click setup, and manual instructions
├── requirements.txt           core dependencies
├── requirements-data.txt      optional: rebuild data/cache, live text encoding (RDKit, sentence-transformers)
├── app/streamlit_app.py       the frontend
├── .streamlit/config.toml     frontend settings (port, theme)
├── configs/default.yaml       every path and hyperparameter
├── data/raw/                  FlavorGraph nodes and edges CSVs
├── data/cache/                precomputed features and top-k similarities for all 4 modalities
├── notebooks/colab_gpu_training.ipynb   the same benchmark on a Colab GPU
├── reports/                   versioned reports (the project changelog)
├── src/                       source code (see ARCHITECTURE.md)
└── tests/                     test suite (python tests/run_tests.py)
```

## Changelog (versioned reports)

| Report | Version | Summary |
|---|---|---|
| [versioned_reports.md](reports/versioned_reports.md) | V1 → V3 addendum | First audit: the PubChem cache was empty, so pseudo-edges were never built. Addendum adds the 4-modality design. |
| [versioned_report_2.md](reports/versioned_report_2.md) | V2 | 4 modalities, BPR loss, Colab bundle. Its results table was later found to be unreliable. |
| [versioned_report_3_corrected.md](reports/versioned_report_3_corrected.md) | V3 (current) | Fixed ground truth and training, corrected results for all modalities, V2 vs V3 comparison. |

## Data and acknowledgements

- Graph data: **FlavorGraph** — Park et al., *"FlavorGraph: a large-scale food-chemical graph for generating food representations and recommending food pairings"*, Scientific Reports 11, 931 (2021). Repository: [lamypark/FlavorGraph](https://github.com/lamypark/FlavorGraph) (Apache-2.0).
- Compound structures: **PubChem** (SMILES cached in `data/cache/pubchem_smiles.json`).
- Text embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
