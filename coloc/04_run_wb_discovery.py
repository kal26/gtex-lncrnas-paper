#!/usr/bin/env python3
"""Whole Blood discovery: joint, protein-coding-only, and single-phenotype fits.

Runs all three discovery designs under ``--output_dir``. Mode clustering is a
separate step: ``05_cluster_cross_model.py``.

Requires three ``--coloc_items_dir_*`` folders (items.txt; single also needs
jobs.tsv or jobs.tsv.gz). Writes flat output tables::

  Whole_Blood.joint.ColocBoost.parquet
  Whole_Blood.protein_coding.ColocBoost.parquet
  Whole_Blood.single_phenotype.ColocBoost.parquet

Public ``cos_id`` values are ``{mode}_{N}`` so mode-merge loci members are unique.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coloc import (
    add_coloc_path_args,
    coloc,
    load_items,
    write_keep_gene_ids,
)


def _run_mode(
    *,
    label: str,
    mode: str,
    output_dir: Path,
    items_dir: str,
    paths: argparse.Namespace,
    required: tuple[str, ...] = ("items",),
    keep_gene_mode: str | None = None,
    use_jobs: bool = False,
) -> Path:
    req = list(required)
    if use_jobs and "jobs" not in req:
        req.append("jobs")
    items = load_items(items_dir, required=req)
    mode_tmp = output_dir / f"_tmp_{mode}"
    keep_path = None
    if keep_gene_mode is not None:
        keep_path = mode_tmp / f"keep_gene_ids.{keep_gene_mode}.txt"
        print(f"[{label}] keep genes from {paths.gencode_classification}")
        write_keep_gene_ids(
            gencode_classification=paths.gencode_classification,
            mode=keep_gene_mode,
            output=keep_path,
        )
    coloc(
        items_list=items.items_list,
        coloc_base_dir=mode_tmp / "colocboost_tmp",
        tissue_output_dir=output_dir,
        full_dist_output_dir=mode_tmp / "tissue_full_dist",
        paths=paths,
        keep_gene_id_path=keep_path,
        jobs_tsv=items.jobs_tsv if use_jobs else None,
        label=label,
        id_label=mode,
        file_label="Whole_Blood",
        mode_tag=mode,
    )
    return output_dir / f"Whole_Blood.{mode}.ColocBoost.parquet"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--coloc_items_dir_joint",
        required=True,
        help="refs/ for joint all-biotype (items.txt)",
    )
    p.add_argument(
        "--coloc_items_dir_protein_coding",
        required=True,
        help="refs/ for protein-coding-only (items.txt)",
    )
    p.add_argument(
        "--coloc_items_dir_single_phenotype",
        required=True,
        help="refs/ for single-phenotype (items.txt + jobs.tsv[.gz])",
    )
    add_coloc_path_args(p)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_mode(
        label="wb/joint",
        mode="joint",
        output_dir=output_dir,
        items_dir=args.coloc_items_dir_joint,
        paths=args,
    )
    _run_mode(
        label="wb/protein_coding",
        mode="protein_coding",
        output_dir=output_dir,
        items_dir=args.coloc_items_dir_protein_coding,
        paths=args,
        keep_gene_mode="only_protein_coding",
    )
    _run_mode(
        label="wb/single_phenotype",
        mode="single_phenotype",
        output_dir=output_dir,
        items_dir=args.coloc_items_dir_single_phenotype,
        paths=args,
        use_jobs=True,
    )
    print(f"[discovery] done → {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
