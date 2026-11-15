# Setup

## setup.bat (Windows)

| Command | What it does |
|---|---|
| `setup.bat` | Full run: create `.venv`, install, run tests, train + evaluate all 4 modalities, print the comparison, launch the frontend |
| `setup.bat results` | Print the terminal comparison of the 4 methods from existing results |
| `setup.bat app` | Launch the frontend only, using existing results (quickest for recording a demo) |

The frontend opens at <http://localhost:8501>. Press `Ctrl+C` in the terminal to stop it. Verbose logs go to `results/logs/`.

## Manual setup

Use these steps if `setup.bat` fails, or if you are on macOS/Linux. Every command is run from the repository root.

## 1. Requirements

- Python **3.10 or newer** (`python --version`)
- About 1.5 GB of disk space for PyTorch and Streamlit
- No GPU and no internet access are needed after installation; all features are precomputed in `data/cache/`.

## 2. Create a virtual environment

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the PyTorch install fails or is very slow, install the CPU build directly, then rerun the line above:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. Run the tests

```bash
python tests/run_tests.py
```

Expected ending: `Summary: 19 passed, 0 failed`. The smoke test writes only to a temporary folder.

## 5. Train and evaluate

All four modalities, seeds 42/123/456 (about 1 minute on a CPU):

```bash
python -m src.run_benchmark
```

Useful options:

```bash
python -m src.run_benchmark --modalities taxonomic --seeds 42     # one modality, one seed
python -m src.run_benchmark --epochs 50                            # override the epoch count
python src/train.py --modality semantic --seeds 42 --loss huber    # single-modality CLI
```

Output:

- `results/RESULTS.md`: combined results table
- `results/results.json`: the same results, structured (read by the comparison and the frontend)
- `results/models/<modality>/`: trained model (`model_seed*.pkl/.pt`), split, graph and features for inference

## 6. Compare the 4 methods in the terminal

```bash
python -m src.compare
```

Prints the baselines, one table per modality, a kNN-vs-GNN head-to-head, a leaderboard and a verdict.

## 7. Launch the frontend

```bash
python -m streamlit run app/streamlit_app.py
```

Then open <http://localhost:8501>. Four tabs:

1. **Overview**: the problem, dataset numbers and the headline result.
2. **Benchmark**: the 4 methods compared (bar chart per metric with error bars, leaderboard, head-to-head).
3. **Cold-start lab**: pick a held-out ingredient and see its real neighbourhood (before), the isolated cold node, and each modality's pseudo-edges with kNN and GNN recommendations, true partners marked (after).
4. **Semantic context**: the sentence-transformer description attached to each node, its embedding, and its semantic neighbours. You can type a new description and re-rank live.

The app reads `data/` and `results/` directly; after rerunning the benchmark, click **Reload data from disk** in the sidebar.

Live re-encoding in tab 4 is optional and needs sentence-transformers (`python -m pip install -r requirements-data.txt`; the model, about 90 MB, downloads on first use). Without it, everything else still works and the tab shows a message.

## 8. Terminal demo

```bash
python -m src.demo                                   # taxonomic model, random held-out ingredient
python -m src.demo --modality semantic --ingredient sugar
```

`--ingredient` must be a held-out test ingredient; the script lists valid names if it is not.

## 9. Google Colab (optional)

1. Locally: `python -m src.export_colab_bundle` → creates `data/colab_training_bundle.zip`.
2. Open `notebooks/colab_gpu_training.ipynb` in Colab, choose a GPU runtime, upload the zip, and run all cells.
3. The last cell downloads `results/` (table and trained models).

## 10. Rebuilding the feature caches (optional)

Only needed if you change the data or the feature extraction. Requires RDKit, sentence-transformers and internet access (PubChem, Wikipedia):

```bash
python -m pip install -r requirements-data.txt
python -m src.data.build_all_modalities
```

## 11. Using the metapath walk corpus (optional, not recommended)

FlavorGraph's walk corpus (`M17-metapath_CHNHC+NHCHN+RandWalk-whichmeta_100-num_walks_50-len_metapath.txt`, about 241 MB) is not in this repository because it exceeds GitHub's 100 MB file limit. It is linked from the [FlavorGraph README](https://github.com/lamypark/FlavorGraph) ("Generated pairing paths"). To use it, place it in the repository root and set `paths.walk_corpus` in `configs/default.yaml` to its file name.

It is disabled by default: in hub-only mode it connects nearly every pair of hubs, which makes the evaluation meaningless (see `reports/versioned_report_3_corrected.md`, §1).

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Install Python 3.10+ and tick "Add python.exe to PATH", or use `py -3` instead of `python`. |
| `running scripts is disabled` when activating in PowerShell | Use `cmd`, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the repository root, using `python -m src.<module>`. |
| `No trained model in results/models/...` from the demo | Run `python -m src.run_benchmark` first. |
| Frontend says "No benchmark results found" | Run `setup.bat` (or `python -m src.run_benchmark`), then click **Reload data from disk**. |
| Port 8501 already in use | Stop the other Streamlit window, or run `python -m streamlit run app/streamlit_app.py --server.port 8502`. |
| Browser didn't open automatically | Open <http://localhost:8501> yourself. |
