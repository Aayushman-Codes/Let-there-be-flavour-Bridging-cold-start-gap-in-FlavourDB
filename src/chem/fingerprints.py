"""
Molecular fingerprint computation and ingredient-level pooling.

Why fingerprints?
- Tanimoto similarity is defined on binary (or count) fingerprints.
- Morgan/ECFP fingerprints capture local chemical neighborhoods, making them
  ideal for measuring structural similarity between flavor compounds.
- For ingredient-level comparison, we need to aggregate per-compound fingerprints
  into a single representation per ingredient. We support three strategies.

Memory note:
- A 2048-bit fingerprint is 256 bytes as a dense vector.
- 1645 compounds × 2048 bits = ~400 KB — trivially fits in memory.
- We store them as numpy uint8 arrays (bit-packed would save 8x but complicates
  Tanimoto computation with negligible benefit at this scale).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_rdkit_warned = False


def smiles_to_fingerprint(
    smiles: str,
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Convert a SMILES string to a binary fingerprint vector.

    Args:
        smiles: Canonical SMILES string.
        fp_type: "morgan" (ECFP-like) or "rdkit" (path-based).
        radius: Morgan fingerprint radius (ignored for rdkit type).
        n_bits: Length of the fingerprint bit vector.

    Returns:
        numpy array of shape (n_bits,) with dtype uint8, or None if invalid.

    Why uint8 instead of bool?
        numpy bitwise operations are faster on uint8 than bool arrays,
        and the memory difference is negligible at our scale.
    """
    global _rdkit_warned
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, RDKFingerprint
        from rdkit import DataStructs

        RDLogger.DisableLog('rdApp.*')

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if fp_type == "morgan":
            try:
                from rdkit.Chem import rdFingerprintGenerator
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
                bitvect = gen.GetFingerprint(mol)
            except (ImportError, AttributeError):
                bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == "rdkit":
            bitvect = RDKFingerprint(mol, fpSize=n_bits)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

        arr = np.zeros(n_bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        return arr

    except ImportError as e:
        if not _rdkit_warned:
            logger.warning(
                f"RDKit is not installed in the active environment ({e}). "
                "Cannot compute real chemical Morgan fingerprints. "
                "In Colab / Jupyter, run: %pip install rdkit"
            )
            _rdkit_warned = True
        return None
    except Exception as e:
        logger.warning(f"Failed to compute fingerprint for {smiles}: {e}")
        return None


def compute_compound_fingerprints(
    cid_to_smiles: Dict[int, str],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> Dict[int, np.ndarray]:
    """Compute fingerprints for all compounds with valid SMILES.

    Args:
        cid_to_smiles: Mapping from PubChem CID to canonical SMILES.
        fp_type: Fingerprint type.
        radius: Morgan radius.
        n_bits: Bit vector length.

    Returns:
        Dict mapping CID to fingerprint array. Only includes valid entries.
    """
    try:
        import rdkit
    except ImportError:
        logger.error(
            "RDKit is not installed in the active environment!\n"
            "Cannot generate Morgan chemical fingerprints from SMILES.\n"
            "To fix in Colab/Jupyter, run:\n"
            "    %pip install rdkit"
        )
        return {}

    results = {}
    failed = 0
    for cid, smiles in cid_to_smiles.items():
        fp = smiles_to_fingerprint(smiles, fp_type, radius, n_bits)
        if fp is not None:
            results[cid] = fp
        else:
            failed += 1

    logger.info(f"Computed fingerprints: {len(results)} valid, {failed} failed "
                f"out of {len(cid_to_smiles)}")
    return results


# ---------------------------------------------------------------------------
# Ingredient-level fingerprint pooling
# ---------------------------------------------------------------------------

def pool_mean(fingerprints: List[np.ndarray]) -> np.ndarray:
    """Mean-pool multiple fingerprints into one continuous vector.

    Why mean-pooling?
    - Produces a "soft" fingerprint that captures the average chemical profile.
    - Used with "union_fingerprint" similarity strategy: Tanimoto on mean-pooled
      vectors approximates the overlap of compound sets.
    - Trade-off: loses information about individual compounds; a single outlier
      compound with unique bits gets diluted.

    Returns: float64 array of shape (n_bits,), values in [0, 1].
    """
    return np.mean(fingerprints, axis=0).astype(np.float64)


def pool_union(fingerprints: List[np.ndarray]) -> np.ndarray:
    """Union (bitwise OR) of multiple fingerprints.

    Why union?
    - A bit is set if ANY compound in the ingredient has that substructure.
    - Captures the full chemical diversity of the ingredient.
    - Trade-off: as more compounds are added, more bits are set, and Tanimoto
      becomes less discriminative (everything looks similar). This is the
      "curse of the union" — ingredients with many compounds tend to have
      high similarity to everything.

    Returns: uint8 array of shape (n_bits,), values in {0, 1}.
    """
    result = np.zeros_like(fingerprints[0], dtype=np.uint8)
    for fp in fingerprints:
        result = np.bitwise_or(result, fp)
    return result


def pool_ingredient_fingerprints(
    compound_node_ids: Set[int],
    node_id_to_pubchem_cid: Dict[int, int],
    cid_to_fingerprint: Dict[int, np.ndarray],
    strategy: str = "mean",
) -> Optional[np.ndarray]:
    """Pool compound fingerprints into a single ingredient-level fingerprint.

    This is used by the "union_fingerprint" similarity strategy, where we
    first pool each ingredient's compounds into one fingerprint, then compute
    Tanimoto between the pooled fingerprints.

    Args:
        compound_node_ids: FlavorGraph node_ids of compounds for this ingredient.
        node_id_to_pubchem_cid: Mapping from node_id to PubChem CID.
        cid_to_fingerprint: Mapping from PubChem CID to fingerprint array.
        strategy: "mean" or "union" pooling.

    Returns:
        Pooled fingerprint, or None if no valid fingerprints found.
    """
    fps = []
    for nid in compound_node_ids:
        cid = node_id_to_pubchem_cid.get(nid)
        if cid is not None and cid in cid_to_fingerprint:
            fps.append(cid_to_fingerprint[cid])

    if not fps:
        return None

    if strategy == "mean":
        return pool_mean(fps)
    elif strategy == "union":
        return pool_union(fps)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")
