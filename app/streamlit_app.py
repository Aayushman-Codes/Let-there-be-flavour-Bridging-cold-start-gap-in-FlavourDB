"""SavorLink demo frontend.

Run:  streamlit run app/streamlit_app.py      (setup.bat does this for you)

Everything is read live from the project files: the FlavorGraph data and
feature caches in data/, and the trained models and results in results/
(written by `python -m src.run_benchmark`). Recommendations are computed on
the fly for whichever ingredient you pick.
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import GenericKNNBaseline  # noqa: E402
from src.inference import load_inference_bundle  # noqa: E402

MODALITIES = ["molecular", "semantic", "hybrid", "taxonomic"]
SIM_FILES = {"molecular": "topk_similarities.pkl", "semantic": "semantic_similarities.pkl",
             "hybrid": "hybrid_similarities.pkl", "taxonomic": "taxonomic_similarities.pkl"}
MODALITY_BLURB = {
    "molecular": "Morgan fingerprints of the ingredient's flavour compounds; set-based Tanimoto similarity.",
    "semantic": "A short text description embedded with all-MiniLM-L6-v2; cosine similarity.",
    "hybrid": "0.5 x molecular + 0.5 x semantic similarity.",
    "taxonomic": "Position in a 3-level food hierarchy (origin / group / subgroup); Wu-Palmer similarity.",
}
METRICS = {"recall_10": "Recall@10", "recall_20": "Recall@20", "mrr": "MRR",
           "ndcg_20": "NDCG@20", "spearman": "Spearman"}
TEMPLATE_MARK = "culinary food ingredient utilized"
COLORS = {"kNN": "#2E7D32", "Strategy III": "#C62828", "Zero-degree GNN": "#EF6C00",
          "Popularity": "#546E7A", "Random": "#9E9E9E"}
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "cache"

st.set_page_config(page_title="SavorLink", page_icon="🍽️", layout="wide")


# --------------------------------------------------------------------------- #
# Live data access (cached per file modification time, so new runs show up)
# --------------------------------------------------------------------------- #
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_results(mtime: float):
    path = RESULTS / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_export(modality: str, seed: int, mtime: float):
    d = RESULTS / "models" / modality
    with open(d / f"split_seed{seed}.pkl", "rb") as fh:
        split = pickle.load(fh)
    with open(d / "graph_metadata.pkl", "rb") as fh:
        meta = pickle.load(fh)
    return split, meta


@st.cache_data(show_spinner=False)
def load_similarities(modality: str, mtime: float):
    with open(CACHE / SIM_FILES[modality], "rb") as fh:
        return pickle.load(fh)


@st.cache_data(show_spinner=False)
def load_text_and_taxonomy(mtime: float):
    texts = json.loads((CACHE / "wikipedia_summaries.json").read_text(encoding="utf-8"))
    taxonomy = json.loads((CACHE / "foodon_taxonomy.json").read_text(encoding="utf-8"))
    return texts, taxonomy


@st.cache_data(show_spinner=False)
def load_semantic_matrix(mtime: float):
    return np.load(CACHE / "semantic_features.npy")


@st.cache_resource(show_spinner="Loading trained GNN...")
def load_predictor(modality: str, seed: int, mtime: float):
    return load_inference_bundle(RESULTS / "models" / modality, seed=seed, device="cpu")


@st.cache_resource(show_spinner="Loading sentence-transformer (first time downloads ~90 MB)...")
def load_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2"), None
    except Exception as exc:  # optional feature: fall back gracefully
        return None, f"{type(exc).__name__}: {exc}"


def models_ready(seed: int) -> bool:
    return all((RESULTS / "models" / m / f"model_seed{seed}.pkl").exists() for m in MODALITIES)


# --------------------------------------------------------------------------- #
# Live computation
# --------------------------------------------------------------------------- #
def heldout(split) -> set:
    return set(split.val_ids) | set(split.test_ids)


def true_partners(split, cold_id: int) -> dict:
    out = {}
    for (s, d), w in split.test_ground_truth.items():
        if w and w > 0 and s != d and cold_id in (s, d):
            out[d if s == cold_id else s] = float(w)
    return out


def neighbours(sims: dict, cold_id: int, exclude: set, k: int = 10):
    return [(c, s) for c, s in sims.get(cold_id, []) if c not in exclude][:k]


def knn_rank(split, cold_id: int, nbrs, candidates, top: int = 10):
    knn = GenericKNNBaseline(split.train_edges, split.train_edge_weights, {cold_id: nbrs})
    scores = knn.predict(cold_id, candidates)
    return sorted(candidates, key=lambda c: -scores[c])[:top]


def gnn_rank(modality: str, seed: int, cold_id: int, exclude: set, top: int = 10):
    pred = load_predictor(modality, seed, _mtime(RESULTS / "models" / modality / f"model_seed{seed}.pkl"))
    # Same rule as evaluation: pseudo-edges never point at other held-out ingredients
    pred.topk_similarities = {q: [(c, s) for c, s in v if c not in exclude]
                              for q, v in pred.topk_similarities.items()}
    return [r["node_id"] for r in pred.recommend_pairings(cold_id, top_k=top)]


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def dot_graph(center: str, edges, truth_names=frozenset(), color="#1565C0", dashed=False, isolated=False):
    """edges: list of (name, weight). Returns a Graphviz DOT string."""
    lines = ['graph G {', 'layout=neato; overlap=false; splines=true; bgcolor="transparent";',
             'node [shape=ellipse, style=filled, fontname="Helvetica", fontsize=11, color="#90A4AE", fillcolor="#ECEFF1"];',
             'edge [color="#78909C", fontcolor="#78909C", fontname="Helvetica"];',
             f'"{center}" [shape=doublecircle, fillcolor="{color}", fontcolor="white", fontsize=13];']
    if isolated:
        lines.append('size="3,1.6";')
        lines.append('"no edges" [shape=plaintext, style="", fontcolor="#9E9E9E"];')
        lines.append(f'"{center}" -- "no edges" [style=invis, len=1.2];')
    for name, w in edges:
        fill = "#C8E6C9" if name in truth_names else "#ECEFF1"
        lines.append(f'"{name}" [fillcolor="{fill}"];')
        style = "dashed" if dashed else "solid"
        lines.append(f'"{center}" -- "{name}" [label="{w:.2f}", fontsize=9, style={style}, '
                     f'penwidth={0.8 + 3 * min(max(w, 0), 1):.2f}, len={2.4 - min(max(w, 0), 1):.2f}];')
    lines.append("}")
    return "\n".join(lines)


def ranked_list(ids, names, truth):
    return pd.DataFrame({"#": range(1, len(ids) + 1),
                         "Ingredient": [names[i] for i in ids],
                         "True partner": ["✅" if i in truth else "" for i in ids]})


def results_frame(results: dict) -> pd.DataFrame:
    rows = []
    for modality, table in results["modalities"].items():
        for model, metrics in table.items():
            for key, (mean, std) in metrics.items():
                rows.append({"Modality": modality.capitalize(), "Model": model, "Metric": METRICS.get(key, key),
                             "key": key, "Mean": mean, "Std": std})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
results = load_results(_mtime(RESULTS / "results.json"))

with st.sidebar:
    st.title("🍽️ SavorLink")
    st.caption("Cold-start food pairing on FlavorGraph")
    seed = st.selectbox("Data split (seed)", [42, 123, 456], index=0,
                        help="Each seed holds out a different set of 46 cold-start ingredients.")
    st.divider()
    st.markdown("**Live data sources**")
    for label, path in [("Results", RESULTS / "results.json"),
                        ("Trained models", RESULTS / "models" / "taxonomic" / f"model_seed{seed}.pkl"),
                        ("Feature caches", CACHE / "semantic_features.npy"),
                        ("Raw graph", ROOT / "data" / "raw" / "edges_191120.csv")]:
        ok = path.exists()
        stamp = datetime.fromtimestamp(_mtime(path)).strftime("%Y-%m-%d %H:%M") if ok else "missing"
        st.markdown(f"{'🟢' if ok else '🔴'} {label} · `{stamp}`")
    if st.button("🔄 Reload data from disk", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

if results is None or not models_ready(seed):
    st.error("No benchmark results found. Run `python -m src.run_benchmark` (or `setup.bat`) first, then reload.")
    st.stop()

split, meta = load_export("taxonomic", seed, _mtime(RESULTS / "models" / "taxonomic" / f"split_seed{seed}.pkl"))
names = meta["ingredient_names"]
name_to_id = {n: i for i, n in names.items()}
hubs = sorted(meta["hub_ingredients"])
exclude = heldout(split)
texts, taxonomy = load_text_and_taxonomy(_mtime(CACHE / "wikipedia_summaries.json"))

tab_overview, tab_results, tab_lab, tab_semantic = st.tabs(
    ["① Overview", "② Benchmark: 4 methods compared", "③ Cold-start lab (before → after)", "④ Semantic context"])

# ---------------------------------------------------------------- Overview --
with tab_overview:
    st.header("Can we recommend pairings for an ingredient with no recipe history?")
    st.markdown(
        "A brand-new ingredient has **no edges** in the recipe graph, so a graph neural network has nothing to "
        "aggregate. We connect the cold ingredient to similar known ingredients using one of **four ways of "
        "describing food**, then compare two ways of using those neighbours:\n"
        "- **kNN** — borrow the known recipe partners of the most similar ingredients.\n"
        "- **Strategy III (GNN)** — add weighted *pseudo-edges* to the similar ingredients and let a GraphSAGE "
        "model score every candidate pair.")
    n_edges = len(split.train_edges) + len({k for k in split.val_ground_truth} | {k for k in split.test_ground_truth})
    c = st.columns(5)
    c[0].metric("Hub ingredients", f"{len(hubs)}")
    c[1].metric("Recipe edges (Recipe1M)", f"{n_edges:,}")
    c[2].metric("Held-out test ingredients", f"{len(split.test_ids)}")
    c[3].metric("Description modalities", "4")
    c[4].metric("Seeds per experiment", f"{len(results['seeds'])}")

    best_mod = max(results["modalities"], key=lambda m: results["modalities"][m]["kNN"]["ndcg_20"][0])
    best = results["modalities"][best_mod]["kNN"]
    gnn_best = max(results["modalities"], key=lambda m: results["modalities"][m]["Strategy III"]["ndcg_20"][0])
    gnn = results["modalities"][gnn_best]["Strategy III"]
    pop = results["modalities"][best_mod]["Popularity"]
    st.subheader("What we found")
    c = st.columns(3)
    c[0].metric(f"Best: {best_mod.capitalize()}-kNN", f"NDCG@20 {best['ndcg_20'][0]:.3f}",
                f"{best['ndcg_20'][0] / pop['ndcg_20'][0]:.1f}x Popularity")
    c[1].metric(f"Best GNN: Strategy III ({gnn_best})", f"NDCG@20 {gnn['ndcg_20'][0]:.3f}",
                f"{gnn['ndcg_20'][0] - pop['ndcg_20'][0]:+.3f} vs Popularity", delta_color="normal")
    c[2].metric("Popularity baseline", f"NDCG@20 {pop['ndcg_20'][0]:.3f}")
    st.info("**kNN beats the GNN in all four modalities, and the GNN does not beat simply recommending popular "
            "ingredients.** Taxonomic and semantic descriptions work best. This is an honest negative result for "
            "the GNN: transferring known neighbours' recipe partners is the better cold-start method here.")
    with st.expander("How the evaluation works"):
        st.markdown(
            "- Graph: FlavorGraph's 416 hub ingredients and their Recipe1M co-occurrence edges (NPMI).\n"
            "- Each seed holds out 15% of eligible ingredients as **test** and removes *all* their edges.\n"
            "- Every method ranks the other 415 hubs; a hit is a held-out true recipe partner.\n"
            "- Metrics: Recall@10/20, MRR, NDCG@20 (graded by NPMI), Spearman; mean ± std over 3 seeds.")

# ---------------------------------------------------------------- Results ---
with tab_results:
    df = results_frame(results)
    st.header("Benchmark: the 4 methods compared")
    st.caption(f"Generated {results['generated']} · seeds {', '.join(map(str, results['seeds']))} · "
               "mean ± std across seeds · higher is better")
    metric_label = st.radio("Metric", list(METRICS.values()), index=3, horizontal=True)
    sub = df[(df.Metric == metric_label) & df.Model.isin(["kNN", "Strategy III", "Zero-degree GNN"])]
    ref = df[(df.Metric == metric_label) & df.Model.isin(["Popularity", "Random"])].drop_duplicates("Model")
    order = [m.capitalize() for m in MODALITIES]
    bars = alt.Chart(sub).mark_bar().encode(
        x=alt.X("Modality:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        xOffset=alt.XOffset("Model:N", sort=["kNN", "Strategy III", "Zero-degree GNN"]),
        y=alt.Y("Mean:Q", title=metric_label),
        color=alt.Color("Model:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Modality", "Model", alt.Tooltip("Mean:Q", format=".3f"), alt.Tooltip("Std:Q", format=".3f")])
    errs = alt.Chart(sub).mark_errorbar(color="#37474F").encode(
        x=alt.X("Modality:N", sort=order), xOffset=alt.XOffset("Model:N", sort=["kNN", "Strategy III", "Zero-degree GNN"]),
        y=alt.Y("lo:Q", title=metric_label), y2="hi:Q").transform_calculate(lo="datum.Mean - datum.Std", hi="datum.Mean + datum.Std")
    rules = alt.Chart(ref).mark_rule(strokeDash=[6, 4], size=2).encode(
        y="Mean:Q", color=alt.Color("Model:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values()))),
        tooltip=["Model", alt.Tooltip("Mean:Q", format=".3f")])
    st.altair_chart((bars + errs + rules).properties(height=380), width="stretch")
    st.caption("Dashed lines: Popularity and Random baselines (same for every modality). Error bars: ± 1 std across seeds.")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Leaderboard (NDCG@20)")
        board = []
        first = next(iter(results["modalities"].values()))
        board.append(("Popularity", "—", first["Popularity"]))
        board.append(("Random", "—", first["Random"]))
        for m, t in results["modalities"].items():
            board.append((f"{m.capitalize()}-kNN", m, t["kNN"]))
            board.append((f"Strategy III", m, t["Strategy III"]))
        board.sort(key=lambda r: -r[2]["ndcg_20"][0])
        st.dataframe(pd.DataFrame([{"Rank": i + 1, "Method": n, "Modality": m,
                                    "NDCG@20": v["ndcg_20"][0], "Recall@10": v["recall_10"][0],
                                    "MRR": v["mrr"][0]} for i, (n, m, v) in enumerate(board)]),
                     hide_index=True, width="stretch",
                     column_config={k: st.column_config.ProgressColumn(k, format="%.3f", min_value=0,
                                                                       max_value=max(r[2]["ndcg_20"][0] for r in board) if k == "NDCG@20" else 0.4)
                                    for k in ["NDCG@20", "Recall@10", "MRR"]})
    with right:
        st.subheader("Head-to-head: kNN vs GNN")
        h2h = []
        for m, t in results["modalities"].items():
            k, g = t["kNN"]["recall_10"][0], t["Strategy III"]["recall_10"][0]
            h2h.append({"Modality": m.capitalize(), "kNN R@10": k, "GNN R@10": g,
                        "kNN advantage": f"{k / g:.1f}x" if g else "∞",
                        "Pseudo-edges help?": "yes" if t["Strategy III"]["ndcg_20"][0] > t["Zero-degree GNN"]["ndcg_20"][0] else "no"})
        st.dataframe(pd.DataFrame(h2h), hide_index=True, width="stretch",
                     column_config={"kNN R@10": st.column_config.NumberColumn(format="%.3f"),
                                    "GNN R@10": st.column_config.NumberColumn(format="%.3f")})
    with st.expander("Full results table"):
        full = df.pivot_table(index=["Modality", "Model"], columns="Metric", values="Mean").reset_index()
        st.dataframe(full, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Lab -------
with tab_lab:
    st.header("Cold-start lab: watch each method rebuild a cold ingredient's neighbourhood")
    test_names = sorted(names[i] for i in split.test_ids)
    # Default to the held-out ingredient with the richest recipe neighbourhood (clearest demo)
    richest = max(split.test_ids, key=lambda i: len(true_partners(split, i)))
    default = test_names.index(names[richest])
    pick = st.selectbox("Held-out ingredient", test_names, index=default, key="lab_pick",
                        help="These ingredients had all their recipe edges removed before training.")
    cold_id = name_to_id[pick]
    truth = true_partners(split, cold_id)
    truth_names = frozenset(names[i] for i in truth)
    candidates = [h for h in hubs if h != cold_id]

    st.subheader("Step 1 · Before: the real recipe neighbourhood")
    c1, c2 = st.columns([1.3, 1])
    top_truth = sorted(truth.items(), key=lambda x: -x[1])[:12]
    with c1:
        st.graphviz_chart(dot_graph(pick, [(names[i], w) for i, w in top_truth], truth_names, color="#2E7D32"),
                          width="stretch")
    with c2:
        st.markdown(f"**{pick}** co-occurs with **{len(truth)}** hub ingredients in Recipe1M recipes "
                    "(edge label = NPMI strength; strongest 12 shown). These are the answers we hide.")
        st.dataframe(pd.DataFrame([{"Partner": names[i], "NPMI": round(w, 3)} for i, w in top_truth]),
                     hide_index=True, width="stretch")

    st.subheader("Step 2 · Cold start: every recipe edge removed")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.graphviz_chart(dot_graph(pick, [], color="#757575", isolated=True), width="content")
    with c2:
        st.markdown("The ingredient is now **isolated** (degree 0). A plain GNN has no neighbours to aggregate — "
                    "the *zero-degree GNN* baseline shows what happens then. Each method below has to recover the "
                    "partners from its description of the ingredient alone.")

    st.subheader("Step 3 · After: apply each method")
    mod = st.radio("Description modality", MODALITIES, horizontal=True, format_func=str.capitalize, key="lab_mod")
    st.caption(MODALITY_BLURB[mod])
    sims = load_similarities(mod, _mtime(CACHE / SIM_FILES[mod]))
    nbrs = neighbours(sims, cold_id, exclude)
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        st.markdown("**Pseudo-edges to the most similar known hubs** (label = similarity)")
        st.graphviz_chart(dot_graph(pick, [(names[c], s) for c, s in nbrs], color="#1565C0", dashed=True),
                          width="stretch")
    knn_ids = knn_rank(split, cold_id, nbrs, candidates)
    gnn_ids = gnn_rank(mod, seed, cold_id, exclude)
    with c2:
        hits = sum(i in truth for i in knn_ids)
        st.markdown(f"**kNN top 10** · {hits} hit{'s' if hits != 1 else ''}")
        st.dataframe(ranked_list(knn_ids, names, truth), hide_index=True, width="stretch", height=388)
    with c3:
        hits = sum(i in truth for i in gnn_ids)
        st.markdown(f"**Strategy III (GNN) top 10** · {hits} hit{'s' if hits != 1 else ''}")
        st.dataframe(ranked_list(gnn_ids, names, truth), hide_index=True, width="stretch", height=388)

    st.subheader(f"All 4 modalities for “{pick}”")
    rows = []
    for m in MODALITIES:
        n = neighbours(load_similarities(m, _mtime(CACHE / SIM_FILES[m])), cold_id, exclude)
        k_ids = knn_rank(split, cold_id, n, candidates)
        g_ids = gnn_rank(m, seed, cold_id, exclude)
        rows += [{"Modality": m.capitalize(), "Method": "kNN", "Hits in top 10": sum(i in truth for i in k_ids)},
                 {"Modality": m.capitalize(), "Method": "Strategy III", "Hits in top 10": sum(i in truth for i in g_ids)}]
    hit_df = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(hit_df).mark_bar().encode(
        x=alt.X("Modality:N", sort=[m.capitalize() for m in MODALITIES], title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="Method:N", y=alt.Y("Hits in top 10:Q", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("Method:N", scale=alt.Scale(domain=["kNN", "Strategy III"],
                                                    range=[COLORS["kNN"], COLORS["Strategy III"]]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Modality", "Method", "Hits in top 10"]).properties(height=260), width="stretch")
    st.caption("One ingredient is a single example; the Benchmark tab averages all 46 held-out ingredients × 3 seeds.")

# ---------------------------------------------------------------- Semantic --
with tab_semantic:
    st.header("Semantic context: a sentence-transformer description for every node")
    st.markdown("Each ingredient node gets a short text description, encoded by **all-MiniLM-L6-v2** into a "
                "384-dimensional vector. Similar vectors → similar ingredients → semantic pseudo-edges.")
    test_names = sorted(names[i] for i in split.test_ids)
    pick_s = st.selectbox("Held-out ingredient", test_names, key="sem_pick",
                          index=test_names.index(st.session_state.get("lab_pick", test_names[0])))
    sid = name_to_id[pick_s]
    truth = true_partners(split, sid)
    candidates = [h for h in hubs if h != sid]
    text = texts.get(pick_s, "")
    is_template = TEMPLATE_MARK in text

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Stored description (attached to the node)**")
        st.markdown(f"> {text}")
        if is_template:
            st.warning("This is the generic template sentence shared by 289 of the 416 ingredients, so its "
                       "embedding mostly encodes the ingredient's name. Try writing a better one below.")
        else:
            st.success("Hand-written description.")
        tax = taxonomy.get(pick_s, {})
        st.markdown(f"**Taxonomy path:** `{tax.get('path', 'n/a')}`")
    with c2:
        X = load_semantic_matrix(_mtime(CACHE / "semantic_features.npy"))
        vec = X[meta["raw_to_idx"][sid]]
        st.markdown("**Its 384-d embedding (first 64 dimensions)**")
        st.altair_chart(alt.Chart(pd.DataFrame({"dim": range(64), "value": vec[:64]})).mark_bar().encode(
            x=alt.X("dim:O", axis=None), y=alt.Y("value:Q", title=None),
            color=alt.condition("datum.value > 0", alt.value("#1565C0"), alt.value("#C62828"))
        ).properties(height=170), width="stretch")

    def semantic_block(vector, title):
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        v = vector / max(np.linalg.norm(vector), 1e-9)
        idx_to_raw = meta["idx_to_raw"]
        order = np.argsort(-(Xn @ v))
        nb = []
        for j in order:
            rid = idx_to_raw[int(j)]
            if rid != sid and rid not in exclude:
                nb.append((rid, float(max(0.0, Xn[j] @ v))))
            if len(nb) == 10:
                break
        recs = knn_rank(split, sid, nb, candidates)
        hits = sum(i in truth for i in recs)
        st.markdown(f"**{title}** · kNN hits in top 10: **{hits}** of {len(truth)} true partners")
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame({"Nearest hubs": [names[r] for r, _ in nb],
                                  "Cosine": [round(s, 3) for _, s in nb]}),
                    hide_index=True, width="stretch")
        b.dataframe(ranked_list(recs, names, truth).rename(columns={"Ingredient": "Recommended partner"}),
                    hide_index=True, width="stretch")
        return hits

    st.subheader("Before: using the stored description")
    base_hits = semantic_block(vec, "Stored description")

    st.subheader("After: add your own context to the node (live)")
    st.caption("Optional feature. Needs sentence-transformers (`pip install -r requirements-data.txt`); the first use "
               "downloads the model (~90 MB).")
    custom = st.text_area("Describe the ingredient", height=90,
                          placeholder=f"e.g. {pick_s.replace('_', ' ').capitalize()} is ... (taste, texture, origin, how it is cooked)")
    if st.button("Encode and re-rank", type="primary", disabled=not custom.strip()):
        encoder, err = load_encoder()
        if encoder is None:
            st.error(f"Live encoding is unavailable ({err}). The stored embeddings above still work.")
        else:
            new_vec = encoder.encode([custom.strip()], convert_to_numpy=True)[0]
            new_hits = semantic_block(new_vec, "Your description")
            delta = new_hits - base_hits
            (st.success if delta > 0 else st.info if delta == 0 else st.warning)(
                f"Hits changed from {base_hits} to {new_hits} ({delta:+d}) by changing only the node's text.")
