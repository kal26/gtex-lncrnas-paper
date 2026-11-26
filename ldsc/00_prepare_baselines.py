#!/usr/bin/env python3
"""Step 0 — build paper baseline_dir layout from public baseline-LD v2.2.

Source: https://zenodo.org/records/10515792 (multi-column ``baselineLD.{chr}.*``).

Writes under ``--baseline_dir``:

- ``baselineLD_v2.2_noGTExEQTL/baselineLD.{chr}.*`` — drop ``GTEx_eQTL_MaxCPP``
- ``baselineLD_v2.2_baseOnly/baselineLD.{chr}.*`` — keep ``base`` only
- ``single_tracks/<track>/<track>.{chr}.*`` — optional attenuation tracks
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

CHROMOSOMES = [str(c) for c in range(1, 23)]
ANNOT_META = ("CHR", "BP", "SNP", "CM")
L2_META = ("CHR", "SNP", "BP")
DROP_ANNOT = "GTEx_eQTL_MaxCPP"
DROP_L2 = "GTEx_eQTL_MaxCPPL2"


def _normalize_prefix(prefix: str) -> Path:
    p = Path(prefix)
    # Accept .../baselineLD or .../baselineLD.
    name = p.name if p.name.endswith(".") else p.name + "."
    return p.with_name(name)


def _read_header(path: Path) -> list[str]:
    with gzip.open(path, "rt") as f:
        return f.readline().rstrip("\n").split("\t")


def _subset_tsv_gz(src: Path, dest: Path, keep_cols: list[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = _read_header(src)
    missing = [c for c in keep_cols if c not in header]
    if missing:
        raise KeyError(f"{src}: missing columns {missing}")
    idx = [header.index(c) for c in keep_cols]
    with gzip.open(src, "rt") as fin, gzip.open(dest, "wt") as fout:
        next(fin)  # skip header already parsed
        fout.write("\t".join(keep_cols) + "\n")
        for line in fin:
            fields = line.rstrip("\n").split("\t")
            fout.write("\t".join(fields[i] for i in idx) + "\n")


def _subset_M(src: Path, dest: Path, keep_indices: list[int]) -> None:
    """Keep 0-based annotation indices from a whitespace-separated .M file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    vals = src.read_text().split()
    dest.write_text(" ".join(vals[i] for i in keep_indices) + "\n")


def _annot_m_index(header: list[str], annot_name: str) -> int:
    """0-based .M index for an annotation column (after CHR/BP/SNP/CM)."""
    return header.index(annot_name) - len(ANNOT_META)


def prepare_baselines(
    public_baseline_prefix: Path,
    baseline_dir: Path,
    tracks: list[str],
) -> None:
    prefix = _normalize_prefix(str(public_baseline_prefix))
    annot1 = Path(f"{prefix}1.annot.gz")
    l2_1 = Path(f"{prefix}1.l2.ldscore.gz")
    if not annot1.is_file():
        raise FileNotFoundError(f"missing {annot1}")

    annot_header = _read_header(annot1)
    l2_header = _read_header(l2_1)
    if DROP_ANNOT not in annot_header:
        raise KeyError(f"{DROP_ANNOT} not in {annot1}")
    if "base" not in annot_header:
        raise KeyError(f"base not in {annot1}")

    keep_nogtex_annot = [c for c in annot_header if c != DROP_ANNOT]
    keep_nogtex_l2 = [c for c in l2_header if c != DROP_L2]
    keep_nogtex_m = [
        i
        for i in range(len(annot_header) - len(ANNOT_META))
        if annot_header[i + len(ANNOT_META)] != DROP_ANNOT
    ]
    base_m_i = _annot_m_index(annot_header, "base")

    print(f"[00_prepare_baselines] prefix={prefix}")
    print(f"[00_prepare_baselines] drop {DROP_ANNOT}; write under {baseline_dir}")

    for chrom in CHROMOSOMES:
        src_a = Path(f"{prefix}{chrom}.annot.gz")
        src_l = Path(f"{prefix}{chrom}.l2.ldscore.gz")
        src_m = Path(f"{prefix}{chrom}.l2.M")
        src_m5 = Path(f"{prefix}{chrom}.l2.M_5_50")

        d = baseline_dir / "baselineLD_v2.2_noGTExEQTL"
        _subset_tsv_gz(src_a, d / f"baselineLD.{chrom}.annot.gz", keep_nogtex_annot)
        _subset_tsv_gz(src_l, d / f"baselineLD.{chrom}.l2.ldscore.gz", keep_nogtex_l2)
        _subset_M(src_m, d / f"baselineLD.{chrom}.l2.M", keep_nogtex_m)
        _subset_M(src_m5, d / f"baselineLD.{chrom}.l2.M_5_50", keep_nogtex_m)

        d = baseline_dir / "baselineLD_v2.2_baseOnly"
        _subset_tsv_gz(
            src_a,
            d / f"baselineLD.{chrom}.annot.gz",
            list(ANNOT_META) + ["base"],
        )
        _subset_tsv_gz(
            src_l,
            d / f"baselineLD.{chrom}.l2.ldscore.gz",
            list(L2_META) + ["baseL2"],
        )
        _subset_M(src_m, d / f"baselineLD.{chrom}.l2.M", [base_m_i])
        _subset_M(src_m5, d / f"baselineLD.{chrom}.l2.M_5_50", [base_m_i])

    for track in tracks:
        if track not in annot_header:
            raise KeyError(f"track {track!r} not in annot header")
        l2_name = f"{track}L2"
        if l2_name not in l2_header:
            raise KeyError(f"{l2_name} not in ldscore header")
        m_i = _annot_m_index(annot_header, track)
        print(f"[00_prepare_baselines] track={track}")
        for chrom in CHROMOSOMES:
            d = baseline_dir / "single_tracks" / track
            _subset_tsv_gz(
                Path(f"{prefix}{chrom}.annot.gz"),
                d / f"{track}.{chrom}.annot.gz",
                list(ANNOT_META) + [track],
            )
            _subset_tsv_gz(
                Path(f"{prefix}{chrom}.l2.ldscore.gz"),
                d / f"{track}.{chrom}.l2.ldscore.gz",
                list(L2_META) + [l2_name],
            )
            _subset_M(Path(f"{prefix}{chrom}.l2.M"), d / f"{track}.{chrom}.l2.M", [m_i])
            _subset_M(
                Path(f"{prefix}{chrom}.l2.M_5_50"),
                d / f"{track}.{chrom}.l2.M_5_50",
                [m_i],
            )

    print(f"[00_prepare_baselines] done baseline_dir={baseline_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--public_baseline_prefix",
        required=True,
        help="Public baselineLD prefix, e.g. /path/to/baselineLD. (trailing dot optional)",
    )
    p.add_argument(
        "--baseline_dir",
        required=True,
        type=Path,
        help="Output baselines folder passed later as --baseline_dir to 04_run_ldsc.sh",
    )
    p.add_argument(
        "--tracks",
        default="",
        help="Optional comma-separated attenuation track names (e.g. Coding_UCSC,H3K27ac_Hnisz)",
    )
    args = p.parse_args()
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    prepare_baselines(
        public_baseline_prefix=Path(args.public_baseline_prefix),
        baseline_dir=args.baseline_dir,
        tracks=tracks,
    )


if __name__ == "__main__":
    main()
