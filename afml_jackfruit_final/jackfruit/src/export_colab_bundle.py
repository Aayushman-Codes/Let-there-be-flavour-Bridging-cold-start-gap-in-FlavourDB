"""Packager for Google Colab GPU Multi-Modal Training Bundle.

Packages all precomputed multi-modal feature matrices, top-k similarities,
graph structure files, and model source code into a compact ~15 MB archive:
`data/colab_training_bundle.zip`.

This bundle allows Google Colab (Tesla T4 GPU) to run all 12 deep learning
experiments (4 modalities x 3 seeds) with zero scraping or API delays.
"""
from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_colab_bundle(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    bundle_path = project_root / "data" / "colab_training_bundle.zip"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    required_cache_files = [
        "node_features.npy",
        "topk_similarities.pkl",
        "semantic_features.npy",
        "semantic_similarities.pkl",
        "taxonomic_features.npy",
        "taxonomic_similarities.pkl",
        "hybrid_features.npy",
        "hybrid_similarities.pkl",
    ]

    logger.info("=" * 68)
    logger.info("  Packaging SavorLink Colab Training Bundle")
    logger.info("=" * 68)

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Configs
        cfg_file = project_root / "configs" / "default.yaml"
        if cfg_file.exists():
            z.write(cfg_file, "configs/default.yaml")
            logger.info("  Added configs/default.yaml")

        # 2. Raw Graph CSVs
        for raw_name in ["nodes_191120.csv", "edges_191120.csv"]:
            p = project_root / "data" / "raw" / raw_name
            if p.exists():
                z.write(p, f"data/raw/{raw_name}")
                logger.info(f"  Added data/raw/{raw_name} ({p.stat().st_size / 1e6:.2f} MB)")

        # 3. Splits
        splits_dir = project_root / "data" / "splits"
        if splits_dir.exists():
            for sp in splits_dir.glob("*.pkl"):
                z.write(sp, f"data/splits/{sp.name}")
                logger.info(f"  Added data/splits/{sp.name}")

        # 4. Multi-Modal Cache Files
        cache_dir = project_root / "data" / "cache"
        for cf in required_cache_files:
            p = cache_dir / cf
            if p.exists():
                z.write(p, f"data/cache/{cf}")
                logger.info(f"  Added data/cache/{cf} ({p.stat().st_size / 1e3:.1f} KB)")
            else:
                logger.warning(f"  [MISSING] data/cache/{cf} not found!")

        # 5. Source code (clean, no pycache)
        src_dir = project_root / "src"
        for root, dirs, files in os.walk(src_dir):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    full_p = Path(root) / f
                    rel_p = full_p.relative_to(project_root)
                    z.write(full_p, str(rel_p))

        logger.info("  Added all src/ Python modules.")

    total_mb = bundle_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 68)
    logger.info(f"  BUNDLE READY: {bundle_path} ({total_mb:.2f} MB)")
    logger.info("=" * 68)
    return bundle_path


if __name__ == "__main__":
    create_colab_bundle()
