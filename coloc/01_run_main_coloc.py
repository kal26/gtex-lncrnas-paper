#!/usr/bin/env python3
"""Main multi-tissue joint-GWAS ColocBoost (fit + per-tissue combine).

Requires --coloc_items_dir with items.txt (refs/main/). Tissues for combine
are taken from the ``Tissue__…`` prefix on each item.

Writes output tables under ``--output_dir``::

  {Tissue}.ColocBoost.parquet

Cross-tissue locus clustering is a separate step: ``02_cluster_cross_tissue.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coloc import add_coloc_path_args, coloc, load_items


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--coloc_items_dir", required=True)
    add_coloc_path_args(p)
    args = p.parse_args()

    items = load_items(args.coloc_items_dir, required=("items",))
    output_dir = Path(args.output_dir)
    coloc(
        items_list=items.items_list,
        coloc_base_dir=output_dir / "colocboost_tmp",
        tissue_output_dir=output_dir,
        full_dist_output_dir=output_dir / "tissue_full_dist",
        paths=args,
        label="main",
    )
    print(f"[main] done → {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
