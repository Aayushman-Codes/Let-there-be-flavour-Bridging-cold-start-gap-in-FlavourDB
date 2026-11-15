"""PubChem SMILES lookup with disk caching, rate limiting, and resumability.

Why this module?
- Ingredients map to compounds via FlavorGraph edges, and compounds have PubChem CIDs.
- To compute Tanimoto similarity we need SMILES -> RDKit Mol -> fingerprint.
- FlavorGraph compounds have CIDs but not SMILES, so we fetch from PubChem.
- PubChem has rate limits (~5 req/s), so we batch, cache, and make it resumable.
- The cache is a simple JSON file: {cid_str: smiles_or_null}.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PubChemCache:
    """Disk-backed cache for PubChem CID -> SMILES mappings."""

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Optional[str]] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk if it exists."""
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info(f"Loaded {len(self._data)} entries from PubChem cache")

    def save(self) -> None:
        """Persist cache to disk."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def get(self, cid: int) -> Optional[str]:
        """Get cached SMILES for a CID, or None if not cached."""
        return self._data.get(str(cid))

    def has(self, cid: int) -> bool:
        """Check if a CID has a valid non-null SMILES in the cache."""
        return str(cid) in self._data and self._data[str(cid)] is not None

    def put(self, cid: int, smiles: Optional[str]) -> None:
        """Store a CID -> SMILES mapping. None means lookup failed/invalid."""
        self._data[str(cid)] = smiles

    def missing(self, cids: Set[int]) -> Set[int]:
        """Return CIDs not yet in the cache."""
        return {c for c in cids if not self.has(c)}

    @property
    def size(self) -> int:
        return len(self._data)

    def valid_count(self) -> int:
        """Count entries with non-null SMILES."""
        return sum(1 for v in self._data.values() if v is not None)


def _fetch_pug_rest_batch(cids: List[int], timeout: float = 10.0) -> Dict[int, Optional[str]]:
    """Fetch SMILES using PubChem PUG REST API directly via JSON.

    URL format requests CanonicalSMILES, ConnectivitySMILES, and SMILES.
    """
    cid_str = ",".join(str(c) for c in cids)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/CanonicalSMILES,ConnectivitySMILES,IsomericSMILES,SMILES/JSON"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SavorLink/1.0 (academic course project)"}
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results: Dict[int, Optional[str]] = {}
    properties = payload.get("PropertyTable", {}).get("Properties", [])
    for prop in properties:
        cid = prop.get("CID")
        smiles = (
            prop.get("CanonicalSMILES")
            or prop.get("ConnectivitySMILES")
            or prop.get("SMILES")
            or prop.get("IsomericSMILES")
        )
        if cid is not None:
            results[int(cid)] = smiles

    # Set missing to None
    for c in cids:
        if c not in results:
            results[c] = None

    return results


def fetch_smiles_batch(
    cids: List[int],
    batch_size: int = 100,
    max_rps: float = 5.0,
    retry_max: int = 3,
    retry_backoff: float = 2.0,
    timeout: float = 10.0,
) -> Dict[int, Optional[str]]:
    """Fetch SMILES from PubChem for a batch of CIDs.

    Uses PubChem PUG REST API with automatic rate limiting and exponential backoff.
    """
    results: Dict[int, Optional[str]] = {}
    min_interval = 1.0 / max_rps

    for batch_start in range(0, len(cids), batch_size):
        batch = cids[batch_start:batch_start + batch_size]
        logger.info(
            f"Fetching PubChem SMILES for CIDs {batch_start+1}-"
            f"{min(batch_start+batch_size, len(cids))}/{len(cids)}"
        )

        success = False
        network_unreachable = False
        for attempt in range(retry_max):
            try:
                start_time = time.time()
                batch_res = _fetch_pug_rest_batch(batch, timeout=timeout)
                results.update(batch_res)
                success = True

                elapsed = time.time() - start_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                break

            except Exception as e:
                wait = retry_backoff ** attempt
                logger.warning(
                    f"PubChem batch failed (attempt {attempt+1}/{retry_max}): {e}."
                )
                # If network is completely unreachable (DNS / sandbox / AVG proxy), fail fast
                if ("nodename nor servname provided" in str(e)
                        or "Network is unreachable" in str(e)
                        or isinstance(e, PermissionError)
                        or "Permission denied" in str(e)
                        or "WinError 10013" in str(e)):
                    logger.warning("Network unreachable (offline or sandboxed). Skipping further PubChem retries.")
                    network_unreachable = True
                    break
                time.sleep(wait)

        if not success:
            for cid in batch:
                if cid not in results:
                    results[cid] = None

        if network_unreachable:
            # Mark all remaining CIDs as None and exit batch loop
            for c in cids:
                if c not in results:
                    results[c] = None
            break

    return results


def validate_and_canonicalize_smiles(smiles: str) -> Optional[str]:
    """Validate a SMILES string with RDKit if available, else strip and validate basic format.

    Why canonicalize? Canonical SMILES ensures consistent fingerprinting.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except ImportError:
        # If RDKit not yet available in current sub-environment, return stripped smiles
        return smiles.strip()
    except Exception:
        return None


def build_smiles_cache(
    cids: Set[int],
    cache: PubChemCache,
    config: dict,
) -> PubChemCache:
    """Fetch and cache SMILES for all CIDs not already cached.

    Resumable: if interrupted, rerun and it continues where it left off.
    """
    to_fetch = sorted(cache.missing(cids))
    if not to_fetch:
        logger.info("All CIDs already cached, nothing to fetch")
        return cache

    logger.info(
        f"Need to fetch {len(to_fetch)} CIDs from PubChem "
        f"({cache.size} already cached)"
    )

    results = fetch_smiles_batch(
        cids=to_fetch,
        batch_size=config.get("batch_size", 100),
        max_rps=config.get("max_requests_per_second", 5),
        retry_max=config.get("retry_max", 3),
        retry_backoff=config.get("retry_backoff", 2.0),
        timeout=config.get("timeout", 10),
    )

    valid_count = 0
    for cid, smiles in results.items():
        if smiles is not None:
            canonical = validate_and_canonicalize_smiles(smiles)
            cache.put(cid, canonical)
            if canonical is not None:
                valid_count += 1
        else:
            cache.put(cid, None)

    cache.save()
    logger.info(
        f"Fetched {len(results)} CIDs: {valid_count} valid SMILES, "
        f"{len(results) - valid_count} failed/invalid"
    )
    return cache


def coverage_report(
    fg_data,
    cache: PubChemCache,
) -> str:
    """Report how many ingredients end up with valid compound SMILES.

    Ingredients with zero valid SMILES cannot participate in Tanimoto
    similarity and thus cannot receive pseudo-edges.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PubChem SMILES Coverage Report")
    lines.append("=" * 60)

    # Compound-level coverage
    total_compounds = len(fg_data.compounds)
    compounds_with_cid = sum(1 for c in fg_data.compounds.values() if c.pubchem_cid is not None)
    compounds_with_smiles = sum(
        1 for c in fg_data.compounds.values()
        if c.pubchem_cid is not None and cache.get(c.pubchem_cid) is not None
    )
    lines.append(f"\nCompound-level:")
    lines.append(f"  Total compounds: {total_compounds}")
    lines.append(f"  With PubChem CID: {compounds_with_cid} ({compounds_with_cid/max(total_compounds,1)*100:.1f}%)")
    lines.append(f"  With valid SMILES in cache: {compounds_with_smiles} ({compounds_with_smiles/max(total_compounds,1)*100:.1f}%)")

    # Ingredient-level coverage
    ingr_compound_counts = []
    ingr_valid_smiles_counts = []
    zero_smiles_ingredients = []

    for nid, info in fg_data.ingredients.items():
        comp_ids = fg_data.ingredient_compounds.get(nid, set())
        n_compounds = len(comp_ids)
        ingr_compound_counts.append(n_compounds)

        n_valid = 0
        for cid in comp_ids:
            comp_node = fg_data.nodes.get(cid)
            if comp_node and comp_node.pubchem_cid is not None:
                smiles = cache.get(comp_node.pubchem_cid)
                if smiles is not None:
                    n_valid += 1
        ingr_valid_smiles_counts.append(n_valid)

        if n_compounds > 0 and n_valid == 0:
            zero_smiles_ingredients.append(info.name)

    ingr_with_any = sum(1 for c in ingr_compound_counts if c > 0)
    ingr_with_smiles = sum(1 for c in ingr_valid_smiles_counts if c > 0)

    lines.append(f"\nIngredient-level:")
    lines.append(f"  Total ingredients: {len(fg_data.ingredients)}")
    lines.append(f"  With any compound links: {ingr_with_any}")
    lines.append(f"  With valid SMILES in cache: {ingr_with_smiles}")
    lines.append(f"  Without compound links (recipe-only): {len(fg_data.ingredients) - ingr_with_any}")

    if ingr_valid_smiles_counts:
        valid_nonzero = [c for c in ingr_valid_smiles_counts if c > 0]
        if valid_nonzero:
            lines.append(f"  Valid SMILES per ingredient (where >0): "
                         f"mean={sum(valid_nonzero)/len(valid_nonzero):.1f}, "
                         f"max={max(valid_nonzero)}")

    if zero_smiles_ingredients:
        lines.append(f"\n  Ingredients with compounds but zero valid SMILES ({len(zero_smiles_ingredients)}):")
        for name in sorted(zero_smiles_ingredients)[:20]:
            lines.append(f"    - {name}")
        if len(zero_smiles_ingredients) > 20:
            lines.append(f"    ... and {len(zero_smiles_ingredients) - 20} more")

    report = "\n".join(lines)
    return report
