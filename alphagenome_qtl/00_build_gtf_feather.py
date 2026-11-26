#!/usr/bin/env python3
"""Build AlphaGenome-compatible GTF + splice-site feathers from a local GENCODE GTF.

Matches ``alphagenome/scripts/process_gtf.py`` (pyranges GTF parse + all-biotype
splice junctions). Default outputs sit next to the input GTF:

  gencode.v47.annotation.gtf
  → gencode.v47.annotation.gtf.feather
  → gencode.v47.splice_sites_{starts,ends}.feather

Example::

  python 00_build_gtf_feather.py --gtf /path/to/gencode.v47.annotation.gtf
  python 00_build_gtf_feather.py --gtf /path/to/gencode.v39.annotation.gtf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyranges
from alphagenome.data import transcript as transcript_utils


def _prefix_from_gtf(gtf_path: Path) -> str:
    stem = gtf_path.name
    if stem.endswith(".annotation.gtf"):
        return stem[: -len(".annotation.gtf")]
    if stem.endswith(".gtf"):
        return stem[: -len(".gtf")]
    return gtf_path.stem


def build_gtf_feathers(gtf_path: Path, output_path: Path) -> None:
    output_path = Path(output_path)
    if not str(output_path).endswith(".feather"):
        raise ValueError(f"output path must end with .feather: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gtf = pyranges.read_gtf(str(gtf_path), as_df=True, duplicate_attr=True)
    gtf["gene_id_nopatch"] = gtf["gene_id"].str.split(".", expand=True)[0]
    gtf.to_feather(output_path)
    print(f"Wrote {len(gtf)} annotation rows to {output_path}")

    transcripts: dict = {}
    n_failed = 0
    for transcript_id, df in gtf.dropna(subset=["transcript_id"]).groupby("transcript_id"):
        try:
            transcripts[transcript_id] = transcript_utils.Transcript.from_gtf_df(df)
        except Exception:
            n_failed += 1

    junctions = []
    for transcript in transcripts.values():
        junctions.extend(
            [intron.chromosome, intron.start, intron.end, intron.strand]
            for intron in transcript.introns
        )
    splice_df = (
        pd.DataFrame(junctions, columns=["Chromosome", "Start", "End", "Strand"])
        .sort_values(by=["Chromosome", "Start"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    prefix = _prefix_from_gtf(Path(gtf_path))
    starts_path = output_path.parent / f"{prefix}.splice_sites_starts.feather"
    ends_path = output_path.parent / f"{prefix}.splice_sites_ends.feather"
    splice_df[["Chromosome", "Start", "Strand"]].to_feather(starts_path)
    splice_df[["Chromosome", "End", "Strand"]].to_feather(ends_path)
    print(
        f"Wrote {len(splice_df)} unique splice junctions "
        f"({len(transcripts)} transcripts, {n_failed} skipped) to"
    )
    print(f"  {starts_path}")
    print(f"  {ends_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build AlphaGenome GTF + splice-site feathers from a GENCODE GTF."
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        required=True,
        help="Local GENCODE annotation GTF (e.g. gencode.v47.annotation.gtf).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="GTF feather path (default: {gtf}.feather).",
    )
    args = parser.parse_args()
    gtf_path = args.gtf
    if not gtf_path.is_file():
        raise SystemExit(f"GTF not found: {gtf_path}")
    output_path = args.output if args.output is not None else Path(f"{gtf_path}.feather")
    build_gtf_feathers(gtf_path, output_path)


if __name__ == "__main__":
    main()
