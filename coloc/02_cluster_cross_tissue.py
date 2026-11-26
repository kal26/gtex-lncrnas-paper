#!/usr/bin/env python3
"""Cluster main per-tissue ColocBoost tables into cross-tissue loci.

Reads ``*.ColocBoost.parquet`` from ``--coloc_dir`` (output of ``01``) and
writes::

  cross_tissue_loci.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cluster_loci import cluster_loci


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coloc_dir",
        required=True,
        help="Directory of {Tissue}.ColocBoost.parquet from 01",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output loci parquet path (e.g. …/cross_tissue_loci.parquet)",
    )
    p.add_argument("--ld_meta", required=True)
    args = p.parse_args()

    coloc_dir = Path(args.coloc_dir)
    if not coloc_dir.is_dir():
        raise FileNotFoundError(f"Missing --coloc_dir: {coloc_dir}")

    print(f"[main/cluster] → {args.output}")
    cluster_loci(
        coloc_dir=coloc_dir,
        output_path=args.output,
        ld_meta=args.ld_meta,
        event_filter="gwas_qtl_cos",
        locus_prefix="cross_tissue",
    )
    print(f"[main/cluster] done → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
