"""Print a clean terminal comparison of the 4 modalities from results/results.json.

Usage:
    python -m src.compare                 # after python -m src.run_benchmark
    python -m src.compare --results path/to/results.json

Plain ASCII output so it renders in any Windows console.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS = [("recall_10", "Recall@10"), ("recall_20", "Recall@20"), ("mrr", "MRR"),
           ("ndcg_20", "NDCG@20"), ("spearman", "Spearman")]
MODELS = ["kNN", "Strategy III", "Zero-degree GNN"]
WIDTH = 92


def _cell(pair) -> str:
    return f"{pair[0]:.3f} +/- {pair[1]:.3f}"


def _table(title: str, rows) -> str:
    """rows: list of (label, {metric: [mean, std]})."""
    head = f"  {'Model':<22}" + "".join(f"{label:>14}" for _, label in METRICS)
    out = [f"\n{title}", "  " + "-" * (WIDTH - 2), head, "  " + "-" * (WIDTH - 2)]
    for label, values in rows:
        out.append(f"  {label:<22}" + "".join(f"{values[k][0]:>14.3f}" for k, _ in METRICS))
    out.append("  " + "-" * (WIDTH - 2))
    return "\n".join(out)


def render(results: dict) -> str:
    mods = results["modalities"]
    first = next(iter(mods.values()))
    lines = ["=" * WIDTH,
             "  SAVORLINK - COLD-START FOOD PAIRING: COMPARISON OF THE 4 METHODS".center(WIDTH),
             "=" * WIDTH,
             f"  Generated {results.get('generated', '?')} | seeds {', '.join(map(str, results.get('seeds', [])))}"
             f" | values are means across seeds (higher is better)"]

    lines.append(_table("REFERENCE BASELINES (same for every modality)",
                        [(n, first[n]) for n in ("Random", "Popularity") if n in first]))

    for modality, table in mods.items():
        rows = [(m if m != "kNN" else f"{modality.capitalize()}-kNN", table[m]) for m in MODELS if m in table]
        lines.append(_table(f"{modality.upper()} MODALITY", rows))

    # Head-to-head: kNN vs Strategy III per modality
    lines += ["", "HEAD-TO-HEAD: kNN vs Strategy III (GNN)", "  " + "-" * (WIDTH - 2),
              f"  {'Modality':<12}{'kNN R@10':>11}{'GNN R@10':>11}{'kNN NDCG':>11}{'GNN NDCG':>11}   Winner",
              "  " + "-" * (WIDTH - 2)]
    knn_wins = 0
    for modality, t in mods.items():
        k, g = t["kNN"], t["Strategy III"]
        winner = "kNN" if k["ndcg_20"][0] > g["ndcg_20"][0] else "Strategy III"
        knn_wins += winner == "kNN"
        ratio = k["recall_10"][0] / g["recall_10"][0] if g["recall_10"][0] > 0 else float("inf")
        lines.append(f"  {modality.capitalize():<12}{k['recall_10'][0]:>11.3f}{g['recall_10'][0]:>11.3f}"
                     f"{k['ndcg_20'][0]:>11.3f}{g['ndcg_20'][0]:>11.3f}   {winner} ({ratio:.1f}x on Recall@10)")

    # Leaderboard of all methods by NDCG@20
    entries = [("Random", first["Random"]), ("Popularity", first["Popularity"])]
    for modality, t in mods.items():
        entries.append((f"{modality.capitalize()}-kNN", t["kNN"]))
        entries.append((f"Strategy III ({modality})", t["Strategy III"]))
    entries.sort(key=lambda e: -e[1]["ndcg_20"][0])
    lines += ["", "LEADERBOARD (ranked by NDCG@20)", "  " + "-" * (WIDTH - 2)]
    for rank, (name, v) in enumerate(entries, 1):
        bar = "#" * int(round(v["ndcg_20"][0] * 150))
        lines.append(f"  {rank:>2}. {name:<28} NDCG@20 {v['ndcg_20'][0]:.3f}  R@10 {v['recall_10'][0]:.3f}  {bar}")

    # Verdict
    best_name, best = entries[0]
    pop = first["Popularity"]["ndcg_20"][0]
    gnn_beats_pop = sum(t["Strategy III"]["ndcg_20"][0] > pop for t in mods.values())
    pseudo_helps = sum(t["Strategy III"]["ndcg_20"][0] > t["Zero-degree GNN"]["ndcg_20"][0] for t in mods.values())
    lines += ["", "VERDICT", "  " + "-" * (WIDTH - 2),
              f"  * Best method overall: {best_name} (NDCG@20 {best['ndcg_20'][0]:.3f}, Recall@10 {best['recall_10'][0]:.3f})",
              f"  * kNN beats Strategy III in {knn_wins} of {len(mods)} modalities",
              f"  * Strategy III beats the Popularity baseline in {gnn_beats_pop} of {len(mods)} modalities",
              f"  * Pseudo-edges beat the zero-degree GNN in {pseudo_helps} of {len(mods)} modalities",
              "=" * WIDTH]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the 4 modalities in the terminal")
    parser.add_argument("--results", default=str(PROJECT_ROOT / "results" / "results.json"))
    args = parser.parse_args()
    path = Path(args.results)
    if not path.exists():
        sys.exit(f"{path} not found. Run `python -m src.run_benchmark` first.")
    print(render(json.loads(path.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
