"""
FlavorGraph data loaders.

Why separate loaders? The FlavorGraph CSV schema is quirky:
- node_ids have gaps (0..8747 but only 8298 rows)
- The 'id' column is PubChem CID but only for compounds, empty for ingredients
- The 'is_hub' column overloads hub status for ingredients ('hub'/'no_hub')
  and compound subtype for compounds ('food'/'drug')
- ingr-compound edge scores are ALL empty; only ingr-ingr edges have NPMI scores

The loaders normalize all of this into clean typed dataclasses so downstream
code never touches raw CSV quirks.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


@dataclass(frozen=True)
class NodeInfo:
    """A single node from FlavorGraph."""
    node_id: int
    name: str
    node_type: str          # "ingredient" or "compound"
    is_hub: bool            # For ingredients: True if hub. For compounds: always False.
    compound_subtype: str   # For compounds: "food" or "drug". For ingredients: "".
    pubchem_cid: Optional[int]  # PubChem CID, only for compounds.


@dataclass(frozen=True)
class EdgeInfo:
    """A single edge from FlavorGraph."""
    src: int        # node_id
    dst: int        # node_id
    edge_type: str  # "ingr-ingr", "ingr-fcomp", "ingr-dcomp"
    score: Optional[float]  # NPMI for ingr-ingr; None for compound edges


@dataclass
class FlavorGraphData:
    """Container for the full FlavorGraph dataset, pre-indexed for fast access."""
    nodes: Dict[int, NodeInfo] = field(default_factory=dict)
    edges: List[EdgeInfo] = field(default_factory=list)

    # Derived indices (populated by build_indices)
    ingredients: Dict[int, NodeInfo] = field(default_factory=dict)
    compounds: Dict[int, NodeInfo] = field(default_factory=dict)
    hub_ingredients: Set[int] = field(default_factory=set)

    # ingredient -> set of compound node_ids
    ingredient_compounds: Dict[int, Set[int]] = field(default_factory=dict)

    # ingredient-ingredient edges only (for NPMI prediction)
    ingr_ingr_edges: List[EdgeInfo] = field(default_factory=list)

    # Adjacency: ingredient -> set of (neighbor_ingredient, npmi_score)
    ingr_adj: Dict[int, Set[Tuple[int, float]]] = field(default_factory=dict)

    def build_indices(self) -> None:
        """Build all derived indices from nodes and edges.

        Why post-init? Because we want to be able to load incrementally
        (e.g., add synthetic cold-start nodes) and rebuild.
        """
        self.ingredients = {
            nid: n for nid, n in self.nodes.items()
            if n.node_type == "ingredient"
        }
        self.compounds = {
            nid: n for nid, n in self.nodes.items()
            if n.node_type == "compound"
        }
        self.hub_ingredients = {
            nid for nid, n in self.ingredients.items() if n.is_hub
        }

        # Build ingredient -> compounds mapping from compound edges
        self.ingredient_compounds = {}
        for e in self.edges:
            if e.edge_type in ("ingr-fcomp", "ingr-dcomp"):
                src_type = self.nodes[e.src].node_type if e.src in self.nodes else None
                dst_type = self.nodes[e.dst].node_type if e.dst in self.nodes else None
                if src_type == "ingredient":
                    self.ingredient_compounds.setdefault(e.src, set()).add(e.dst)
                if dst_type == "ingredient":
                    self.ingredient_compounds.setdefault(e.dst, set()).add(e.src)

        # Build ingr-ingr edge list and adjacency.
        # Recognises both the original CSV type ("ingr-ingr") and the
        # walk-derived type ("ingr-ingr-walk") so downstream code is unaffected
        # regardless of which data source is active.
        _ingr_ingr_types = {"ingr-ingr", "ingr-ingr-walk"}
        self.ingr_ingr_edges = [e for e in self.edges if e.edge_type in _ingr_ingr_types]
        self.ingr_adj = {}
        for e in self.ingr_ingr_edges:
            score = e.score if e.score is not None else 0.0
            self.ingr_adj.setdefault(e.src, set()).add((e.dst, score))
            self.ingr_adj.setdefault(e.dst, set()).add((e.src, score))

    def filter_to_hubs(self) -> "FlavorGraphData":
        """Return a new FlavorGraphData restricted to hub ingredients and their connections.

        Keeps all 416 hub ingredients (which possess chemical compound annotations)
        and all compound nodes, while dropping the 6,236 featureless ingredients and
        any edges touching them.
        """
        hubs = set(self.hub_ingredients)
        filtered = FlavorGraphData()
        filtered.nodes = {
            nid: n for nid, n in self.nodes.items()
            if n.node_type == "compound" or (n.node_type == "ingredient" and n.is_hub)
        }
        _ingr_ingr_types = {"ingr-ingr", "ingr-ingr-walk"}
        filtered.edges = [
            e for e in self.edges
            if (e.edge_type in _ingr_ingr_types and e.src in hubs and e.dst in hubs)
            or (e.edge_type in ("ingr-fcomp", "ingr-dcomp") and (e.src in hubs or e.dst in hubs))
        ]
        filtered.build_indices()
        return filtered


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file.

    Why a helper? So we can layer overrides later (CLI args, Colab widgets)
    without changing downstream code.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path: str, project_root: str | Path) -> Path:
    """Resolve a potentially relative path against project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(project_root) / p


def load_flavorgraph(
    config: dict,
    project_root: str | Path,
    hub_only: Optional[bool] = None,
) -> FlavorGraphData:
    """Load FlavorGraph nodes and edges CSVs into a FlavorGraphData container.

    Args:
        config: Full configuration dictionary.
        project_root: Path to project root.
        hub_only: If True, restrict to 416 hub ingredients with chemistry.
                  If None, defaults to config['data'].get('hub_only', False).

    Why not just pd.read_csv? We want:
    1. Typed dataclasses with semantic field names
    2. Normalization of the quirky is_hub/compound_subtype overload
    3. No pandas dependency in the data layer (lighter for Colab)
    4. Explicit handling of empty scores and float CIDs
    """
    cfg_data = config["data"]
    root = Path(project_root)

    if hub_only is None:
        hub_only = cfg_data.get("hub_only", False)

    nodes_path = resolve_path(config["paths"]["raw_nodes"], root)
    edges_path = resolve_path(config["paths"]["raw_edges"], root)

    fg = FlavorGraphData()

    # --- Load nodes ---
    with open(nodes_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row[cfg_data["node_id_col"]])
            name = row[cfg_data["name_col"]]
            ntype = row[cfg_data["node_type_col"]]
            hub_raw = row[cfg_data["hub_col"]]

            # Parse the overloaded is_hub column
            if ntype == cfg_data["ingredient_type"]:
                is_hub = hub_raw in cfg_data["hub_values"]
                compound_subtype = ""
                pubchem_cid = None
            else:
                # Compound node
                is_hub = False
                compound_subtype = hub_raw  # "food" or "drug"
                # Parse PubChem CID (stored as float string like "439246.0")
                raw_id = row[cfg_data["pubchem_id_col"]].strip()
                pubchem_cid = int(float(raw_id)) if raw_id else None

            fg.nodes[nid] = NodeInfo(
                node_id=nid,
                name=name,
                node_type=ntype if ntype == cfg_data["ingredient_type"] else "compound",
                is_hub=is_hub,
                compound_subtype=compound_subtype,
                pubchem_cid=pubchem_cid,
            )

    # --- Load edges ---
    with open(edges_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row[cfg_data["edge_src_col"]])
            dst = int(row[cfg_data["edge_dst_col"]])
            etype = row[cfg_data["edge_type_col"]]
            raw_score = row[cfg_data["edge_score_col"]].strip()
            score = float(raw_score) if raw_score else None

            fg.edges.append(EdgeInfo(src=src, dst=dst, edge_type=etype, score=score))

    fg.build_indices()
    if hub_only:
        fg = fg.filter_to_hubs()
    return fg


def print_data_report(fg: FlavorGraphData) -> str:
    """Print a concise summary of the loaded FlavorGraph data.

    Returns the report as a string for logging/testing.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("FlavorGraph Data Report")
    lines.append("=" * 60)

    lines.append(f"\nTotal nodes: {len(fg.nodes)}")
    lines.append(f"  Ingredients: {len(fg.ingredients)}")
    lines.append(f"    Hub ingredients: {len(fg.hub_ingredients)}")
    lines.append(f"  Compounds: {len(fg.compounds)}")

    food_comp = sum(1 for c in fg.compounds.values() if c.compound_subtype == "food")
    drug_comp = sum(1 for c in fg.compounds.values() if c.compound_subtype == "drug")
    lines.append(f"    Food compounds: {food_comp}")
    lines.append(f"    Drug compounds: {drug_comp}")

    has_cid = sum(1 for c in fg.compounds.values() if c.pubchem_cid is not None)
    lines.append(f"    With PubChem CID: {has_cid}/{len(fg.compounds)}")

    lines.append(f"\nTotal edges: {len(fg.edges)}")
    from collections import Counter
    edge_types = Counter(e.edge_type for e in fg.edges)
    for et, count in edge_types.most_common():
        lines.append(f"  {et}: {count}")

    # Ingredient-compound coverage
    ingr_with_comp = len(fg.ingredient_compounds)
    lines.append(f"\nIngredients with compounds: {ingr_with_comp}/{len(fg.ingredients)}")
    if fg.ingredient_compounds:
        comp_counts = [len(cs) for cs in fg.ingredient_compounds.values()]
        lines.append(f"  Compounds per ingredient: mean={sum(comp_counts)/len(comp_counts):.1f}, "
                     f"max={max(comp_counts)}, median={sorted(comp_counts)[len(comp_counts)//2]}")

    # NPMI distribution for ingr-ingr
    scores = [e.score for e in fg.ingr_ingr_edges if e.score is not None]
    if scores:
        lines.append(f"\nIngr-ingr NPMI scores ({len(scores)} edges):")
        lines.append(f"  Range: [{min(scores):.4f}, {max(scores):.4f}]")
        lines.append(f"  Mean: {sum(scores)/len(scores):.4f}")
        sorted_s = sorted(scores)
        lines.append(f"  Median: {sorted_s[len(sorted_s)//2]:.4f}")
        neg = sum(1 for s in scores if s < 0)
        lines.append(f"  Negative: {neg}")

    # Degree distribution
    degrees = {}
    for e in fg.ingr_ingr_edges:
        degrees[e.src] = degrees.get(e.src, 0) + 1
        degrees[e.dst] = degrees.get(e.dst, 0) + 1
    if degrees:
        degs = list(degrees.values())
        lines.append(f"\nIngr-ingr degree distribution:")
        lines.append(f"  Mean: {sum(degs)/len(degs):.1f}, Max: {max(degs)}")
        zero_deg = len(fg.ingredients) - len(degrees)
        lines.append(f"  Zero-degree ingredients: {zero_deg}")

    report = "\n".join(lines)
    print(report)
    return report
