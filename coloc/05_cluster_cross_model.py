#!/usr/bin/env python3
"""Cluster WB discovery modes into shared GWAS loci (mode-as-tissue merge).

Reads flat discovery ColocBoost tables from ``04`` and writes::

  cross_phenotype_set_loci.parquet

Symlinks each mode file as ``{mode}.ColocBoost.parquet`` so clustering treats
mode as the tissue label (matching ``cos_id`` prefix ``{mode}_{N}``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cluster_loci import cluster_loci


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--discovery_dir",
        required=True,
        help="Root from 04 with Whole_Blood.{mode}.ColocBoost.parquet",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output loci parquet (e.g. …/cross_phenotype_set_loci.parquet)",
    )
    p.add_argument("--ld_meta", required=True)
    args = p.parse_args()

    discovery_dir = Path(args.discovery_dir)
    modes = ("joint", "protein_coding", "single_phenotype")
    link_dir = discovery_dir / "_mode_cluster_links"
    link_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        src = discovery_dir / f"Whole_Blood.{mode}.ColocBoost.parquet"
        if not src.is_file():
            raise FileNotFoundError(f"Missing {mode} ColocBoost from 04: {src}")
        dst = link_dir / f"{mode}.ColocBoost.parquet"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src.resolve(), dst)
        print(f"linked {dst.name} → {src.name}")

    print(f"[wb/mode_merge] → {args.output}")
    cluster_loci(
        coloc_dir=link_dir,
        output_path=args.output,
        ld_meta=args.ld_meta,
        event_filter="gwas_any",
        locus_prefix="cross_phenotype_set",
    )
    print(f"[wb/mode_merge] done → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
