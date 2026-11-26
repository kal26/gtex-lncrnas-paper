#!/usr/bin/env python3
"""VEP × fine-mapped QTL Firth enrichment.

Writes one combined TSV under --output_dir:
  qtl_vep_odds.txt

Rows are keyed by gene_type (novel_lncRNA / existing_lncRNA), susie_type
(eqtl / sqtl), pip_cutoff, and annotation. existing_lncRNA uses PIP > 0.9;
novel_lncRNA uses a PIP sweep plus CS-max lead.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
QTL_ODDS = HERE / "qtl_odds.py"

SUSIE_TYPES = ("eqtl", "sqtl")
COMBINED_NAME = "qtl_vep_odds.txt"


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--combined_annotations",
        type=Path,
        required=True,
        help="odds_annotations.parquet from 02_combine_odds_annotations.py",
    )
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--esusie_dir", type=Path, required=True)
    p.add_argument("--ssusie_dir", type=Path, required=True)
    p.add_argument("--gencode_class", type=Path, required=True)
    p.add_argument("--lncrna_category_file", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    part_files: list[Path] = []
    for susie_type in SUSIE_TYPES:
        for gene_type in ("existing_lncRNA", "novel_lncRNA"):
            cmd = [
                sys.executable,
                str(QTL_ODDS),
                "--combined_annotations",
                str(args.combined_annotations),
                "--gene_type",
                gene_type,
                "--susie_type",
                susie_type,
                "--output_dir",
                str(args.output_dir),
            ]
            if gene_type == "novel_lncRNA":
                cmd.extend(
                    [
                        "--esusie_dir",
                        str(args.esusie_dir),
                        "--ssusie_dir",
                        str(args.ssusie_dir),
                        "--gencode_class",
                        str(args.gencode_class),
                        "--lncrna_category_file",
                        str(args.lncrna_category_file),
                    ]
                )
            _run(cmd)
            part_files.append(
                args.output_dir / f"{gene_type}.{susie_type}_odds.txt"
            )

    combined = pd.concat([pd.read_table(f) for f in part_files], ignore_index=True)
    out_path = args.output_dir / COMBINED_NAME
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote combined odds results to {out_path} ({len(combined):,} rows)")
    for f in part_files:
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
