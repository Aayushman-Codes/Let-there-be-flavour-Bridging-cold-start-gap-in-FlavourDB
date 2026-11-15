# SavorLink Environment Setup & Operational Guide

This document provides step-by-step instructions for installing dependencies, running the test suite, executing multi-seed benchmarks, and serving the interactive Streamlit application.

For architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/pipelines.md](docs/pipelines.md). For benchmark results and paper links, see [README.md](README.md).

---

## Table of Contents

1. [System Prerequisites](#1-system-prerequisites)
2. [Quickstart (Windows One-Click)](#2-quickstart-windows-one-click)
3. [Manual Installation (Windows, macOS, Linux)](#3-manual-installation-windows-macos-linux)
4. [Running the Test Suite](#4-running-the-test-suite)
5. [Training & Multi-Seed Benchmarking](#5-training--multi-seed-benchmarking)
6. [Interactive Frontend & Demos](#6-interactive-frontend--demos)
7. [Optional Feature Regeneration](#7-optional-feature-regeneration)
8. [Google Colab Cloud Training](#8-google-colab-cloud-training)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. System Prerequisites

- **Python Version:** Python **3.10 or newer** (tested on 3.10, 3.11, 3.12). Check via:
  ```bash
  python --version
  ```
- **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, macOS 12+ (Apple Silicon or Intel).
- **Disk Space:** Approximately $1.5\text{ GB}$ (primarily for PyTorch and Streamlit).
- **GPU Requirements:** None. All models and benchmarks run under pure CPU in under 1 minute. CUDA acceleration is auto-detected if available.
- **Offline Self-Containment:** All required feature matrices and sparse similarities are precomputed and versioned in `data/cache/`. Zero network access is required after `pip install`.

---

## 2. Quickstart (Windows One-Click)

On Windows, the root script `setup.bat` handles the entire environment lifecycle:

```bat
setup.bat
```

### What `setup.bat` executes:
1. Verifies that Python 3.10+ is accessible on the system `PATH`.
2. Creates an isolated virtual environment at `.venv\`.
3. Upgrades `pip` and installs all dependencies from `requirements.txt`.
4. Runs all 19 automated tests in `tests/run_tests.py` (logging to `results/logs/tests.log`).
5. Executes the full multi-seed benchmark across all 4 modalities ($\approx 35\text{ seconds}$ on CPU).
6. Outputs the formatted comparison table to the terminal.
7. Automatically launches the Streamlit frontend at <http://localhost:8501>.

### Quick CLI Shortcuts:
- `setup.bat results`: Reprints the terminal benchmark comparison from existing `results/results.json`.
- `setup.bat app`: Immediately opens the Streamlit frontend using existing results.

---

## 3. Manual Installation (Windows, macOS, Linux)

If you prefer manual setup or are running on macOS/Linux:

### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Core Requirements
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*Note for CPU-only systems:* If PyTorch wheels take too long to download, install the dedicated CPU wheel:
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. Running the Test Suite

Execute the standalone verification runner to ensure that mathematical invariants, graph indexing, pseudo-edge construction, and leakage checks pass:

```bash
python tests/run_tests.py
```

### Expected Output:
```
=======================================================
Summary: 19 passed, 0 failed
=======================================================
ALL TESTS PASSED SUCCESSFULLY!
```

The test suite covers:
- Chemical Tanimoto metrics (binary, continuous, mean best match, max pairwise)
- Dynamic pseudo-edge construction and symmetric merging
- Disjoint ingredient-level graph splitting and strict leakage audits
- Inductive neural network forward/backward gradient flows (zero node-ID embeddings)
- Metric calculations (Recall@K, MRR, NDCG@K, Spearman rank correlation)
- Non-GNN baselines (Random, Popularity, Tanimoto-kNN)
- End-to-end 2-epoch training pipeline smoke test
- Hardware and memory footprint compliance (Google Colab free-tier budgets)

---

## 5. Training & Multi-Seed Benchmarking

Run the full benchmark across all 4 representation modalities ($\text{molecular}$, $\text{semantic}$, $\text{hybrid}$, $\text{taxonomic}$) and 3 fixed seeds ($42, 123, 456$):

```bash
python -m src.run_benchmark
```

### Useful CLI Options:
```bash
# Benchmark a single modality with one seed
python -m src.run_benchmark --modalities taxonomic --seeds 42

# Override training epochs
python -m src.run_benchmark --epochs 50

# Single-modality training entrypoint with Huber loss
python src/train.py --modality semantic --seeds 42 --loss huber
```

### Generated Artifacts:
- `results/RESULTS.md`: Markdown summary table of all modalities and baselines.
- `results/results.json`: Structured results consumed by the CLI comparison tool and Streamlit app.
- `results/models/<modality>/`: Checkpoints (`model_seed*.pt`), graph splits, and serialized inference bundles.

---

## 6. Interactive Frontend & Demos

### 1. Streamlit Web Dashboard
Launch the browser application:
```bash
python -m streamlit run app/streamlit_app.py
```
Open <http://localhost:8501> in your browser. The dashboard contains four tabs:
1. **Overview:** High-level project background, dataset statistics, and core findings.
2. **Benchmark:** Interactive comparison charts across all 5 metrics with standard deviation error bars, head-to-head comparisons, and leaderboard.
3. **Cold-Start Lab:** Single-ingredient inspection. Select any held-out ingredient (e.g., `cheddar_cheese`, `sugar`, `jackfruit`) to observe its ground-truth recipe partners, complete edge disconnection, pseudo-edge reconnection, and predictions.
4. **Semantic Context:** Natural language descriptors and cosine nearest neighbors for each node, with an optional real-time sentence encoder.

### 2. Terminal Comparison Tool
To inspect the latest benchmark summary without launching a web browser:
```bash
python -m src.compare
```

### 3. Single-Ingredient CLI Demo
Demonstrate pairing predictions on a specific held-out test ingredient:
```bash
# Random held-out test ingredient under taxonomic modality
python -m src.demo

# Specific ingredient under semantic modality
python -m src.demo --modality semantic --ingredient sugar
```

---

## 7. Optional Feature Regeneration

To regenerate the precomputed caches in `data/cache/` directly from PubChem, Wikipedia, and FoodOn:

```bash
# Install optional chemistry and sentence-transformer libraries
python -m pip install -r requirements-data.txt

# Run full feature extraction pipeline
python -m src.data.build_all_modalities
```

*Requirements:* Internet connection (for PubChem REST API and Wikipedia lookup), RDKit, and sentence-transformers.

---

## 8. Google Colab Cloud Training

To run the full multi-seed benchmark on a cloud GPU (e.g., Tesla T4):

1. **Build Cloud Bundle Locally:**
   ```bash
   python -m src.export_colab_bundle
   ```
   This generates `data/colab_training_bundle.zip` ($\approx 4\text{ MB}$).
2. **Run Colab Notebook:**
   - Open `notebooks/colab_gpu_training.ipynb` in [Google Colab](https://colab.research.google.com/).
   - Select **Runtime -> Change runtime type -> T4 GPU**.
   - Upload `colab_training_bundle.zip` to the file panel.
   - Run all cells. The final cell downloads `savorlink_colab_results.zip` containing all trained weights and logs.

---

## 9. Troubleshooting & FAQ

| Issue | Root Cause | Solution |
|---|---|---|
| `'python' is not recognized` | Python is missing from system `PATH`. | Reinstall Python 3.10+ and check **"Add python.exe to PATH"**, or use `py -3`. |
| `running scripts is disabled` (PowerShell) | Windows execution policy restricts script activation. | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` or use `cmd.exe`. |
| `ModuleNotFoundError: No module named 'src'` | Command executed from inside a subdirectory. | Run all commands from the project root using `python -m src.<module>`. |
| `No trained model in results/models/...` | Demo script executed before running benchmark. | Run `python -m src.run_benchmark` first. |
| `Frontend says 'No benchmark results found'` | Streamlit opened before running benchmark. | Run `setup.bat` (or `python -m src.run_benchmark`), then click **"Reload data from disk"** in the sidebar. |
| `Port 8501 is already in use` | Another Streamlit process is active. | Terminate existing window or run `python -m streamlit run app/streamlit_app.py --server.port 8502`. |
