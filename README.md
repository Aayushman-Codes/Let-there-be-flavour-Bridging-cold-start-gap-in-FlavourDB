# Let There Be Flavour

**Cold-start food pairing on FlavorGraph.** Say a new ingredient shows up and nobody has cooked with it yet. Can you guess what it goes well with?

That was our question. We tested a graph neural network built for this case ("Strategy III") against a plain nearest-neighbour method on [FlavorGraph](https://github.com/lamypark/FlavorGraph), describing each ingredient in four ways: chemistry, text, a food taxonomy, and a chemistry + text hybrid.

The nearest-neighbour method won every time. It finds known ingredients that look like the new one and borrows their recipe partners, and it beat the GNN by 2.7 to 4.6 times on Recall@10. The GNN also scored below a baseline that just recommends popular ingredients. The full write-up is in **[our IEEE-format report (PDF)](SavorLink_IEEE_Report.pdf)**.

Course project for *Advanced Foundations of Machine Learning* (UE24AM342AA1), PES University.

| Team | SRN |
|---|---|
| Akshit Singhal | PES1UG24AM027 |
| Ahan A Mysore | PES1UG24AM019 |
| Aayushman Singh | PES1UG24AM006 |
| Abhay TS | PES1UG24AM008 |

---

## Run it

On Windows, one command does everything:

```bat
setup.bat
```

It makes a virtual environment, installs the dependencies, runs the 19 tests, then trains and scores all four modalities. That takes about a minute on a laptop CPU. Then it prints the method comparison in the terminal and opens the app at <http://localhost:8501>.

Two shortcuts for later: `setup.bat results` reprints the comparison and `setup.bat app` reopens the app. If the script breaks on your machine, [setup.md](setup.md) walks through the same steps by hand.

On macOS or Linux:

```bash
pip install -r requirements.txt
python tests/run_tests.py                       # 19 tests
python -m src.run_benchmark                     # 4 modalities x 3 seeds -> results/
python -m src.compare                           # terminal comparison of the 4 methods
python -m streamlit run app/streamlit_app.py    # frontend
```

### What's in the app

| Tab | What you see |
|---|---|
| Overview | The problem, the dataset in numbers, and the main result |
| Benchmark | Every method side by side, one bar chart per metric with error bars and baselines, plus a leaderboard |
| Cold-start lab | Pick a held-out ingredient. You see its real recipe partners, then the same node with every edge cut, then each modality's pseudo-edges with the kNN and GNN top ten (real partners ticked) |
| Semantic context | The text description stored on each node and its nearest neighbours by meaning. Type your own description and re-rank on the spot (needs `requirements-data.txt`) |

The app reads straight from `data/` and `results/`, so what it shows is whatever the last benchmark run produced.

---

## How it works

A GNN learns about a node by looking at its neighbours. A brand-new ingredient has no edges in the recipe graph, so there's nothing to look at. This is the zero-degree cold-start problem.

Strategy III patches the gap. It links the new ingredient to its most similar known "hub" ingredients with weighted **pseudo-edges**, which gives message passing something to work with. "Similar" can mean one of four things:

| Modality | How an ingredient is described | Similarity |
|---|---|---|
| Molecular | Mean-pooled 2048-bit Morgan fingerprints of its flavour compounds (PubChem SMILES) | Set-based Tanimoto (mean best match) |
| Semantic | A short text description embedded with `all-MiniLM-L6-v2` (384-d) | Cosine |
| Taxonomic | Position in a 3-level food hierarchy (origin, group, subgroup), 141-d multi-hot | Wu-Palmer |
| Hybrid | Molecular and semantic features side by side (2432-d) | 0.5 x molecular + 0.5 x semantic |

Testing works the same for every method. We hide some ingredients, delete all of their recipe edges, and ask each method to rank the other 415 hubs as possible partners. A good method puts the real partners near the top.

For the module-by-module picture, see [ARCHITECTURE.md](ARCHITECTURE.md).

## What we built

- **A cold-start benchmark without leakage** on FlavorGraph's 416 hub ingredients. Splits are by ingredient (339 train, 31 validation, 46 test per seed), ground truth is Recipe1M co-occurrence, and every number is averaged over seeds 42, 123 and 456 across five ranking metrics.
- **Four ingredient representations**, with the kNN baseline and the GNN both running on any of them from one config file.
- **A fully inductive GNN** built from weighted GraphSAGE, a symmetric link-prediction head, BPR ranking loss and cold-mask training. It has no per-node embedding table, so it can score ingredients it never saw.
- **A frontend** for watching a single ingredient go through the whole process.

## Results

Mean of three seeds. Full tables, standard deviations and the V2 vs V3 comparison are in [report 3](reports/versioned_report_3_corrected.md) and the [PDF report](SavorLink_IEEE_Report.pdf).

| Modality | kNN Recall@10 | kNN NDCG@20 | Strategy III Recall@10 | Strategy III NDCG@20 |
|---|---|---|---|---|
| Taxonomic | **0.127** | **0.184** | 0.035 | 0.048 |
| Semantic | 0.111 | 0.165 | 0.030 | 0.041 |
| Hybrid | 0.099 | 0.142 | 0.021 | 0.036 |
| Molecular | 0.088 | 0.127 | 0.032 | 0.045 |
| *Popularity* | *0.058* | *0.083* | | |
| *Random* | *0.020* | *0.029* | | |

kNN came out ahead in all four modalities. Taxonomic kNN was the best single setup, then semantic, hybrid and molecular.

The pseudo-edges didn't do much for the GNN. Strategy III lands within noise of the same model trained with no pseudo-edges at all. It also overfits fast. The best validation epoch was always one of the first three, and after that it only gets better at the warm training graph.

On this data, borrowing neighbours' recipe partners worked better for new ingredients than the GNN did.

## Known gaps

- **The text descriptions are thin.** 289 of the 416 are the same template sentence, and the rest are short and hand-written. So semantic similarity is mostly picking up on ingredient names.
- **We wrote the taxonomy ourselves** instead of pulling it from FoodOn. It lives in `TAXONOMIC_RULES` in `src/data/taxonomic_lookup.py`.
- **Some fingerprints collide.** The 416 hubs share only 300 distinct pooled fingerprints, which makes a few unrelated foods look chemically identical.
- **The test set is small**, at 46 ingredients and 562 true pairs per seed. Spearman in particular jumps around between seeds.
- `SavorLinkPredictor.predict_new_compounds()` was never finished and raises `NotImplementedError`.

## Repository layout

```
├── README.md                  this file
├── SavorLink_IEEE_Report.pdf  the project report (IEEE conference format)
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

## Changelog

Each version of the project got its own report. They're kept as the changelog.

| Report | Version | What changed |
|---|---|---|
| [versioned_reports.md](reports/versioned_reports.md) | V1, plus V3 addendum | First audit. The PubChem cache was empty, so pseudo-edges were never actually built. The addendum sketches the four-modality design. |
| [versioned_report_2.md](reports/versioned_report_2.md) | V2 | Four modalities, BPR loss and the Colab bundle. Its results table turned out to be unreliable. |
| [versioned_report_3_corrected.md](reports/versioned_report_3_corrected.md) | V3 (current) | Fixed the ground truth and training, reran every modality, and compared V2 with V3. |
| [SavorLink_IEEE_Report.pdf](SavorLink_IEEE_Report.pdf) | Final | The V3 results written up as a 3-page IEEE conference paper. |

## Credits

- Graph data is from **FlavorGraph**: Park et al., *"FlavorGraph: a large-scale food-chemical graph for generating food representations and recommending food pairings"*, Scientific Reports 11, 931 (2021). Code at [lamypark/FlavorGraph](https://github.com/lamypark/FlavorGraph) (Apache-2.0).
- Compound structures are from **PubChem**, cached in `data/cache/pubchem_smiles.json`.
- Text embeddings use `sentence-transformers/all-MiniLM-L6-v2`.
- We used Claude and Antigravity as coding assistants during the project.
