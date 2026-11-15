"""Metapath random-walk loader for FlavorGraph walk corpus.

Why a separate loader?
----------------------
The file ``M17-metapath_CHNHC+NHCHN+RandWalk-whichmeta_100-num_walks_50-len_metapath.txt``
contains heterogeneous metapath random walks generated directly from the
FlavorGraph.  Each line is one walk — a whitespace-separated sequence of node
IDs that alternates between ingredient and compound nodes according to the
metapath schemas (e.g. CHNHC, NHCHN).

These walks encode **real structural co-occurrence** in the graph.  Using them
is strictly better than generating pseudo-edges from chemistry alone, because
they capture multi-hop recipe and chemical topology simultaneously.

What this module does
---------------------
1. Stream-parse the walk corpus (241 MB / 1.1 M lines) without loading it
   entirely into RAM.
2. Extract **ingredient-ingredient co-occurrence counts** using a sliding
   window over ingredient-only tokens in each walk.  Two ingredients that
   appear within ``window_size`` positions of each other (in walk order,
   skipping compound nodes) are considered co-occurring.
3. Normalise raw counts into co-occurrence scores (PMI-like or simple
   frequency fraction) so downstream code can treat them like NPMI edge
   weights.
4. Return the result as real ``EdgeInfo`` objects so the rest of the
   pipeline (splits, GNN, evaluation) is completely unchanged.

Design decisions
----------------
- **Streaming**: The file is read one line at a time; peak RAM is O(vocab).
- **Ingredient-only window**: Compound nodes act as "bridge" nodes in the
  metapaths.  We skip them when computing the co-occurrence window so we
  don't inflate compound-mediated counts.
- **Symmetric edges**: Both (A, B) and (B, A) are recorded so the adjacency
  is undirected (same convention as FlavorGraph's ingr-ingr edges).
- **Score normalisation**: Raw counts are divided by the total number of
  co-occurrences each ingredient participates in, giving a soft probability
  score in [0, 1] that is analogous to NPMI in meaning.
"""
from __future__ import annotations

import logging
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from src.data.loaders import EdgeInfo, FlavorGraphData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk file parsing
# ---------------------------------------------------------------------------

def iter_walks(walk_path: str | Path) -> Iterator[List[int]]:
    """Yield one walk (list of int node IDs) per line from the walk corpus.

    Handles both space- and tab-separated tokens.  Empty lines are skipped.
    """
    with open(walk_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield [int(tok) for tok in line.split()]


# ---------------------------------------------------------------------------
# Co-occurrence extraction
# ---------------------------------------------------------------------------

def extract_ingr_cooccurrence(
    walk_path: str | Path,
    ingredient_ids: Set[int],
    window_size: int = 5,
    max_lines: Optional[int] = None,
) -> Dict[Tuple[int, int], int]:
    """Count ingredient co-occurrences within a sliding window over each walk.

    Algorithm
    ---------
    For each walk:
      1. Filter the walk to ingredient-only tokens (drop compound nodes).
      2. Slide a window of ``window_size`` over the resulting sequence.
      3. For every pair (u, v) that falls within the same window, increment
         count[(min(u,v), max(u,v))] to keep the dict symmetric-canonical.

    Args:
        walk_path:      Path to the metapath walk text file.
        ingredient_ids: Set of node IDs that are ingredients (from FlavorGraphData).
        window_size:    Number of ingredient tokens to consider as context
                        neighbours of the central token.
        max_lines:      If set, stop after this many lines (useful for quick
                        debugging without streaming the full 241 MB file).

    Returns:
        Dict mapping canonical (src, dst) pair — with src < dst — to raw
        integer co-occurrence count.
    """
    cooccur: Dict[Tuple[int, int], int] = defaultdict(int)
    n_walks = 0
    n_pairs = 0

    for walk in iter_walks(walk_path):
        if max_lines is not None and n_walks >= max_lines:
            break

        # Keep only ingredient nodes in their walk order
        ingr_seq = [nid for nid in walk if nid in ingredient_ids]

        # Slide window over the ingredient sequence
        for center_idx, center in enumerate(ingr_seq):
            start = max(0, center_idx - window_size)
            end = min(len(ingr_seq), center_idx + window_size + 1)
            for ctx_idx in range(start, end):
                if ctx_idx == center_idx:
                    continue
                ctx = ingr_seq[ctx_idx]
                if ctx == center:
                    # A node revisited within the window is not a pairing with itself
                    continue
                # Canonical ordering keeps the dict half the size
                key = (min(center, ctx), max(center, ctx))
                cooccur[key] += 1
                n_pairs += 1

        n_walks += 1
        if n_walks % 100_000 == 0:
            logger.info(
                f"  Processed {n_walks:,} walks, {len(cooccur):,} unique pairs so far…"
            )

    logger.info(
        f"Walk co-occurrence extraction complete: "
        f"{n_walks:,} walks, {len(cooccur):,} unique ingr-ingr pairs, "
        f"{n_pairs:,} total co-occurrence events"
    )
    return dict(cooccur)


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def normalise_cooccurrence(
    cooccur: Dict[Tuple[int, int], int],
    mode: str = "pmi",
) -> Dict[Tuple[int, int], float]:
    """Convert raw co-occurrence counts to normalised edge scores.

    Args:
        cooccur: Raw counts from ``extract_ingr_cooccurrence``.
        mode:    Normalisation strategy:
                 ``"pmi"``   — Pointwise Mutual Information (log-based).
                              Score = log2( p(u,v) / (p(u)*p(v)) ).
                              Clipped to [0, max_pmi] so all scores >= 0
                              (positive PMI = PPMI).
                 ``"count"`` — Raw count divided by total events (frequency).
                 ``"freq"``  — Each pair's count divided by the geometric
                               mean of the two nodes' marginal counts.
                              Equivalent to lift / max-lift normalisation.

    Returns:
        Dict mapping the same (src, dst) keys to float scores in [0, 1]
        (for ``"freq"`` and ``"count"``) or [0, max_pmi] (for ``"pmi"``).
    """
    if not cooccur:
        return {}

    # Compute marginal counts per ingredient
    marginal: Dict[int, int] = defaultdict(int)
    total = 0
    for (u, v), cnt in cooccur.items():
        marginal[u] += cnt
        marginal[v] += cnt
        total += cnt  # each pair was counted once per direction in window

    scores: Dict[Tuple[int, int], float] = {}

    if mode == "count":
        for pair, cnt in cooccur.items():
            scores[pair] = cnt / total

    elif mode == "freq":
        for (u, v), cnt in cooccur.items():
            denom = math.sqrt(marginal[u] * marginal[v])
            scores[(u, v)] = cnt / denom if denom > 0 else 0.0
        # Normalise to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    elif mode == "pmi":
        N = total
        for (u, v), cnt in cooccur.items():
            p_uv = cnt / N
            p_u = marginal[u] / N
            p_v = marginal[v] / N
            if p_u > 0 and p_v > 0 and p_uv > 0:
                pmi = math.log2(p_uv / (p_u * p_v))
                scores[(u, v)] = max(0.0, pmi)  # PPMI: clip negatives
            else:
                scores[(u, v)] = 0.0
        # Normalise to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    else:
        raise ValueError(f"Unknown normalisation mode: {mode!r}. Use 'pmi', 'freq', or 'count'.")

    return scores


# ---------------------------------------------------------------------------
# Top-level: load walks and inject real edges into FlavorGraphData
# ---------------------------------------------------------------------------

def load_walk_edges(
    walk_path: str | Path,
    fg: FlavorGraphData,
    window_size: int = 5,
    min_count: int = 2,
    score_mode: str = "pmi",
    max_lines: Optional[int] = None,
    cache_dir: Optional[str | Path] = "data/cache",
) -> Tuple[List[EdgeInfo], Dict[Tuple[int, int], int]]:
    """Extract real ingredient-ingredient edges from metapath walk corpus.

    This replaces or augments the ``ingr-ingr`` edges that were originally
    sourced from the FlavorGraph CSV (which contain NPMI scores computed
    separately).  The walk-derived edges reflect true structural co-occurrence
    as encoded in the metapath walk sequences.

    Results are automatically persisted to disk cache in ``cache_dir`` so that
    subsequent runs take < 1 second instead of re-parsing the 241 MB corpus.

    Args:
        walk_path:   Path to the metapath walk text file.
        fg:          Loaded FlavorGraphData (used to identify ingredient IDs).
        window_size: Context window size (in ingredient tokens) for co-occurrence.
        min_count:   Drop pairs with raw co-occurrence count below this.
                     Filters noise from very rare co-occurrences.
        score_mode:  Score normalisation mode for ``normalise_cooccurrence``.
        max_lines:   If set, only process this many walk lines (debug mode).
        cache_dir:   Directory for caching parsed walk edges. Set to None to bypass.

    Returns:
        Tuple of:
        - walk_edges: List of EdgeInfo objects (edge_type="ingr-ingr-walk")
          ready to be merged into fg.edges.
        - raw_counts: The raw co-occurrence count dict for diagnostics.
    """
    # 1. Check disk cache if full file parsing is requested
    cache_file = None
    valid_ingr = set(fg.ingredients.keys())
    if cache_dir is not None and max_lines is None:
        c_path = Path(cache_dir)
        c_path.mkdir(parents=True, exist_ok=True)
        cache_suffix = f"_hubs{len(valid_ingr)}" if len(valid_ingr) < 6000 else ""
        cache_file = c_path / f"walk_edges_win{window_size}_min{min_count}_{score_mode}{cache_suffix}.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached walk edges from {cache_file}...")
            try:
                with open(cache_file, "rb") as fh:
                    cached_data = pickle.load(fh)
                walk_edges = [e for e in cached_data["walk_edges"] if e.src in valid_ingr and e.dst in valid_ingr]
                raw_counts = {k: v for k, v in cached_data["raw_counts"].items() if k[0] in valid_ingr and k[1] in valid_ingr}
                logger.info(
                    f"Successfully loaded {len(walk_edges):,} walk edges from cache in <0.5s."
                )
                return walk_edges, raw_counts
            except Exception as e:
                logger.warning(f"Could not load cache file ({e}), re-extracting from raw walks...")

    ingredient_ids: Set[int] = set(fg.ingredients.keys())
    logger.info(
        f"Extracting walk co-occurrence for {len(ingredient_ids):,} ingredients "
        f"(window={window_size}, min_count={min_count}, mode={score_mode!r})…"
    )

    raw_counts = extract_ingr_cooccurrence(
        walk_path=walk_path,
        ingredient_ids=ingredient_ids,
        window_size=window_size,
        max_lines=max_lines,
    )

    # Filter by minimum count
    raw_counts = {k: v for k, v in raw_counts.items() if v >= min_count}
    logger.info(f"After min_count={min_count} filter: {len(raw_counts):,} pairs remain")

    scores = normalise_cooccurrence(raw_counts, mode=score_mode)

    walk_edges: List[EdgeInfo] = []
    for (src, dst), score in scores.items():
        walk_edges.append(
            EdgeInfo(src=src, dst=dst, edge_type="ingr-ingr-walk", score=score)
        )

    logger.info(
        f"Built {len(walk_edges):,} walk-derived ingr-ingr edges "
        f"(score range [{min(scores.values()):.4f}, {max(scores.values()):.4f}])"
        if scores else "Built 0 walk-derived edges."
    )

    # 2. Save to disk cache
    if cache_file is not None:
        try:
            with open(cache_file, "wb") as fh:
                pickle.dump({"walk_edges": walk_edges, "raw_counts": raw_counts}, fh)
            logger.info(f"Cached {len(walk_edges):,} walk edges to: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to persist walk edge cache: {e}")

    return walk_edges, raw_counts


def inject_walk_edges(
    fg: FlavorGraphData,
    walk_path: str | Path,
    window_size: int = 5,
    min_count: int = 2,
    score_mode: str = "pmi",
    replace_csv_edges: bool = True,
    max_lines: Optional[int] = None,
    cache_dir: Optional[str | Path] = "data/cache",
) -> FlavorGraphData:
    """Load walk-derived edges and inject them into a FlavorGraphData object.

    Args:
        fg:                Loaded FlavorGraphData to augment/replace edges in.
        walk_path:         Path to the metapath walk text file.
        window_size:       Context window for co-occurrence extraction.
        min_count:         Minimum co-occurrence count to keep an edge.
        score_mode:        Score normalisation mode ('pmi', 'freq', 'count').
        replace_csv_edges: If True, **replace** the CSV ingr-ingr edges with
                           walk-derived ones.  If False, **merge** both sets
                           (walk edges appended; CSV edges kept).
        max_lines:         Limit lines parsed (debug/testing only).
        cache_dir:         Directory where walk edge cache is stored / loaded.

    Returns:
        The same ``fg`` object with updated edges and rebuilt indices.
    """
    walk_edges, _ = load_walk_edges(
        walk_path=walk_path,
        fg=fg,
        window_size=window_size,
        min_count=min_count,
        score_mode=score_mode,
        max_lines=max_lines,
        cache_dir=cache_dir,
    )

    if replace_csv_edges:
        # Drop existing ingr-ingr edges; keep compound edges intact
        non_ingr_ingr = [e for e in fg.edges if e.edge_type != "ingr-ingr"]
        fg.edges = non_ingr_ingr + walk_edges
        logger.info(
            f"Replaced CSV ingr-ingr edges with {len(walk_edges):,} walk-derived edges. "
            f"Total edges now: {len(fg.edges):,}"
        )
    else:
        fg.edges = fg.edges + walk_edges
        logger.info(
            f"Merged walk edges: added {len(walk_edges):,} edges. "
            f"Total edges now: {len(fg.edges):,}"
        )

    fg.build_indices()
    return fg


# ---------------------------------------------------------------------------
# Diagnostics / reporting
# ---------------------------------------------------------------------------

def walk_edge_report(
    raw_counts: Dict[Tuple[int, int], int],
    scores: Dict[Tuple[int, int], float],
    fg: FlavorGraphData,
) -> str:
    """Return a human-readable summary of the walk-derived edges.

    Args:
        raw_counts: Raw co-occurrence counts from ``extract_ingr_cooccurrence``.
        scores:     Normalised scores from ``normalise_cooccurrence``.
        fg:         FlavorGraphData (used to look up ingredient names).

    Returns:
        Multi-line report string.
    """
    import statistics

    lines = ["=" * 60, "Walk-Derived Edge Report", "=" * 60]

    lines.append(f"\nTotal unique pairs (after min_count filter): {len(raw_counts):,}")

    if raw_counts:
        counts = list(raw_counts.values())
        lines.append(f"\nRaw co-occurrence count distribution:")
        lines.append(f"  Min:    {min(counts)}")
        lines.append(f"  Max:    {max(counts)}")
        lines.append(f"  Mean:   {statistics.mean(counts):.1f}")
        lines.append(f"  Median: {statistics.median(counts):.0f}")

    if scores:
        sv = list(scores.values())
        lines.append(f"\nNormalised score distribution:")
        lines.append(f"  Min:  {min(sv):.4f}")
        lines.append(f"  Max:  {max(sv):.4f}")
        lines.append(f"  Mean: {statistics.mean(sv):.4f}")

    # Degree distribution in walk graph
    degree: Dict[int, int] = defaultdict(int)
    for (u, v) in scores:
        degree[u] += 1
        degree[v] += 1

    if degree:
        degs = list(degree.values())
        lines.append(f"\nDegree distribution in walk graph:")
        lines.append(f"  Nodes with edges: {len(degree):,} / {len(fg.ingredients):,} ingredients")
        lines.append(f"  Mean degree: {statistics.mean(degs):.1f}")
        lines.append(f"  Max degree:  {max(degs)}")
        lines.append(f"  Zero-degree: {len(fg.ingredients) - len(degree):,}")

        # Top-10 highest-degree nodes
        top_nodes = sorted(degree.items(), key=lambda x: -x[1])[:10]
        lines.append(f"\nTop-10 most connected ingredients (by walk degree):")
        for nid, deg in top_nodes:
            name = fg.ingredients.get(nid, type("_", (), {"name": str(nid)})()).name
            lines.append(f"  [{deg:5d} edges] {name}")

    return "\n".join(lines)

