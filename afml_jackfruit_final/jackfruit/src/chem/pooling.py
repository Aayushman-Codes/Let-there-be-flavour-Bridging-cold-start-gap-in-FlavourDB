"""Convenience module that re-exports pooling functions.

Why a separate module? The pooling functions are defined in fingerprints.py
(they're tightly coupled to the fingerprint representation), but downstream
code (pseudo_edges, models) may want to import just pooling without pulling
in all of fingerprints.py. This module provides a clean interface.
"""
from src.chem.fingerprints import (
    pool_ingredient_fingerprints,
    pool_mean,
    pool_union,
)

__all__ = ["pool_ingredient_fingerprints", "pool_mean", "pool_union"]
