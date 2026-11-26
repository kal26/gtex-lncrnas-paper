#!/usr/bin/env python3
"""Build EUR LD-block panels + ``ld_metadata.tsv`` from GTEx WGS and LD-block BEDs.

Genotype-derived LD matrices are not included on Zenodo. Rebuild locally from protected
GTEx WGS PLINK beds and public EUR LD-block boundaries (paper: pyrho EUR, tiled
to cover each autosome, MHC excluded).

For each block, runs PLINK (``--keep`` EUR samples → QC → ``--r square gz``),
re-compresses the square LD matrix as xz (extension remains ``.ld.gz``; readers
detect xz magic), then writes::

  {output_dir}/LD_chr{chrom}_{start}_{end}.{bed,bim,fam,ld.gz}
  {output_dir}/ld_metadata.tsv

``ld_metadata.tsv`` columns (paper / pecotmr layout)::

  chrom  start  end  path
  1      16104  1170341  …/LD_chr1_16104_1170341.ld.gz,…/LD_chr1_16104_1170341.bim

``start``/``end`` are **1-based inclusive** coordinates matching the LD file
stem (BED start + 1). Empty post-QC blocks are skipped in the metadata.

Example::

  python 00_build_ld_meta.py \\
    --genotype_stem /path/to/GTEx_WGS.SHAPEIT2_phased.MAF01 \\
    --sample_keep /path/to/european_subject_ids.keep.txt \\
    --ld_blocks_bed /path/to/pyrho_EUR_LD_blocks.bed \\
    --chrom_sizes /path/to/hg38.chrom.sizes \\
    --output_dir /path/to/ld_blocks_gtex_eur_tiled

Always tiles gaps to chrom ends and excludes the MHC window. Uses ``plink``
from the pixi ``coloc`` env (bioconda ``plink`` 1.9).
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Paper MHC window (0-based half-open, chr6): trim/drop blocks overlapping this.
MHC_CHROM = "chr6"
MHC_START0 = 25_000_000
MHC_END0 = 34_000_000

Block = Tuple[str, int, int]  # chrom (with chr prefix), start0, end0


def _chrom_key(chrom: str) -> Tuple[int, str]:
    c = chrom[3:] if chrom.startswith("chr") else chrom
    try:
        return (int(c), chrom)
    except ValueError:
        return (99, chrom)


def _norm_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            sizes[_norm_chrom(parts[0])] = int(parts[1])
    return sizes


def read_ld_blocks_bed(path: Path) -> List[Block]:
    """Read ``chr start end`` BED (optional header). Autosomes only."""
    blocks: List[Block] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            if i == 0 and parts[0].lower() in ("chr", "chrom", "#chrom"):
                continue
            chrom = _norm_chrom(parts[0])
            stem = chrom[3:]
            if not stem.isdigit() or not (1 <= int(stem) <= 22):
                continue
            blocks.append((chrom, int(parts[1]), int(parts[2])))
    blocks.sort(key=lambda b: (_chrom_key(b[0]), b[1], b[2]))
    return blocks


def tile_blocks(blocks: Sequence[Block], chrom_sizes: dict[str, int]) -> List[Block]:
    """Fill [0, first), gaps, and (last, chrom_end] per autosome (paper tiling)."""
    by_chrom: dict[str, List[Tuple[int, int]]] = {}
    for chrom, start, end in blocks:
        by_chrom.setdefault(chrom, []).append((start, end))
    out: List[Block] = []
    for chrom in sorted(by_chrom, key=_chrom_key):
        chr_size = chrom_sizes.get(chrom)
        if chr_size is None:
            raise KeyError(f"{chrom} missing from --chrom_sizes")
        regions = sorted(by_chrom[chrom])
        tiled: List[Tuple[int, int]] = []
        if regions[0][0] > 0:
            tiled.append((0, regions[0][0]))
        for i, (start, end) in enumerate(regions):
            tiled.append((start, end))
            if i + 1 < len(regions) and end < regions[i + 1][0]:
                tiled.append((end, regions[i + 1][0]))
        if regions[-1][1] < chr_size:
            tiled.append((regions[-1][1], chr_size))
        out.extend((chrom, s, e) for s, e in tiled if e > s)
    return out


def subtract_interval(
    start: int, end: int, cut_start: int, cut_end: int
) -> List[Tuple[int, int]]:
    """Return pieces of [start, end) after removing [cut_start, cut_end)."""
    if end <= cut_start or start >= cut_end:
        return [(start, end)] if end > start else []
    pieces: List[Tuple[int, int]] = []
    if start < cut_start:
        pieces.append((start, min(end, cut_start)))
    if end > cut_end:
        pieces.append((max(start, cut_end), end))
    return [(s, e) for s, e in pieces if e > s]


def exclude_mhc(
    blocks: Sequence[Block],
    *,
    chrom: str = MHC_CHROM,
    start0: int = MHC_START0,
    end0: int = MHC_END0,
) -> List[Block]:
    """Drop/trim blocks overlapping the MHC window (paper default chr6:25–34 Mb)."""
    out: List[Block] = []
    for c, s, e in blocks:
        if _norm_chrom(c) != _norm_chrom(chrom):
            out.append((c, s, e))
            continue
        for ns, ne in subtract_interval(s, e, start0, end0):
            out.append((c, ns, ne))
    out.sort(key=lambda b: (_chrom_key(b[0]), b[1], b[2]))
    return out


def write_blocks_bed(path: Path, blocks: Sequence[Block]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("chr\tstart\tend\n")
        for chrom, start, end in blocks:
            fh.write(f"{chrom}\t{start}\t{end}\n")


def write_eur_keep_from_phenotypes(phenotypes: Path, output: Path) -> Path:
    """PLINK ``--keep`` file (FID IID) for GTEx subjects with RACE==3."""
    import pandas as pd

    df = pd.read_csv(phenotypes, sep="\t", dtype=str)
    if "SUBJID" not in df.columns or "RACE" not in df.columns:
        raise ValueError(
            f"{phenotypes} must have SUBJID and RACE columns (GTEx SubjectPhenotypesDS)"
        )
    ids = df.loc[df["RACE"].astype(str) == "3", "SUBJID"].astype(str).tolist()
    if not ids:
        raise RuntimeError(f"No RACE==3 subjects in {phenotypes}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        for sid in ids:
            fh.write(f"{sid}\t{sid}\n")
    print(f"Wrote {output} ({len(ids)} EUR subjects)")
    return output


def run_plink_ld_block(
    *,
    genotype_stem: Path,
    sample_keep: Path,
    chrom: str,
    start0: int,
    end0: int,
    output_dir: Path,
    plink_bin: str = "plink",
) -> Optional[Tuple[Path, Path]]:
    """Compute one block LD panel. Returns (ld_path, bim_path) or None if empty."""
    chrom_num = chrom[3:] if chrom.startswith("chr") else chrom
    start1 = start0 + 1
    end = end0
    stem = output_dir / f"LD_chr{chrom_num}_{start1}_{end}"
    output_dir.mkdir(parents=True, exist_ok=True)
    threads = "1"

    cmd_bed = [
        plink_bin,
        "--bfile",
        str(genotype_stem),
        "--keep",
        str(sample_keep),
        "--chr",
        chrom_num,
        "--from-bp",
        str(start1),
        "--to-bp",
        str(end),
        "--maf",
        "0.009",
        "--geno",
        "0.05",
        "--hwe",
        "1e-6",
        "--make-bed",
        "--threads",
        threads,
        "--out",
        str(stem),
    ]
    proc = subprocess.run(cmd_bed, capture_output=True, text=True)
    log_path = Path(str(stem) + ".log")
    if proc.returncode != 0:
        log_txt = log_path.read_text() if log_path.is_file() else (proc.stderr or "")
        if "All variants removed" in log_txt or "All variants excluded" in log_txt:
            Path(str(stem) + ".bim").write_text("")
            # Empty xz payload named .ld.gz (readers sniff magic).
            with open(str(stem) + ".ld.gz", "wb") as fh:
                subprocess.run(["xz", "-c"], input=b"", stdout=fh, check=True)
            return None
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        raise RuntimeError(
            f"plink --make-bed failed for chr{chrom_num}:{start1}-{end} (rc={proc.returncode})"
        )

    cmd_ld = [
        plink_bin,
        "--bfile",
        str(stem),
        "--r",
        "square",
        "gz",
        "--threads",
        threads,
        "--out",
        str(stem),
    ]
    proc = subprocess.run(cmd_ld, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        raise RuntimeError(
            f"plink --r square failed for chr{chrom_num}:{start1}-{end} (rc={proc.returncode})"
        )

    ld_gz = Path(str(stem) + ".ld.gz")
    if not ld_gz.is_file() or ld_gz.stat().st_size == 0:
        return None
    # Re-compress gzip → xz in place (keep .ld.gz name; cluster_loci / pecotmr sniff xz).
    tmp = Path(str(ld_gz) + ".xztmp")
    with gzip.open(ld_gz, "rb") as gin, open(tmp, "wb") as gout:
        subprocess.run(["xz", "-c"], stdin=gin, stdout=gout, check=True)
    tmp.replace(ld_gz)

    bim = Path(str(stem) + ".bim")
    if not bim.is_file() or bim.stat().st_size == 0:
        return None
    return ld_gz, bim


def write_ld_metadata(path: Path, rows: Iterable[Tuple[str, int, int, Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("chrom\tstart\tend\tpath\n")
        for chrom, start1, end, ld_path, bim_path in rows:
            chrom_num = chrom[3:] if chrom.startswith("chr") else chrom
            fh.write(
                f"{chrom_num}\t{start1}\t{end}\t{ld_path.resolve()},{bim_path.resolve()}\n"
            )


def build_ld_meta(
    *,
    genotype_stem: Path,
    sample_keep: Path,
    ld_blocks_bed: Path,
    output_dir: Path,
    chrom_sizes: Path,
) -> Path:
    plink_bin = "plink"
    if shutil.which(plink_bin) is None and not Path(plink_bin).is_file():
        raise FileNotFoundError(
            f"plink not found ({plink_bin!r}); use pixi run -e coloc … "
            f"(bioconda plink)"
        )
    if shutil.which("xz") is None:
        raise FileNotFoundError("xz not found on PATH (needed to compress LD matrices)")

    for p in (Path(str(genotype_stem) + ".bed"), Path(str(genotype_stem) + ".bim"), Path(str(genotype_stem) + ".fam")):
        if not p.is_file():
            raise FileNotFoundError(f"Missing genotype file: {p}")
    if not sample_keep.is_file():
        raise FileNotFoundError(f"Missing --sample_keep: {sample_keep}")
    if not ld_blocks_bed.is_file():
        raise FileNotFoundError(f"Missing --ld_blocks_bed: {ld_blocks_bed}")
    if not chrom_sizes.is_file():
        raise FileNotFoundError(f"Missing --chrom_sizes: {chrom_sizes}")

    blocks = read_ld_blocks_bed(ld_blocks_bed)
    if not blocks:
        raise RuntimeError(f"No autosomal blocks in {ld_blocks_bed}")
    print(f"[00] loaded {len(blocks):,} LD blocks from {ld_blocks_bed}")

    sizes = read_chrom_sizes(chrom_sizes)
    blocks = tile_blocks(blocks, sizes)
    print(f"[00] tiled → {len(blocks):,} blocks")

    n0 = len(blocks)
    blocks = exclude_mhc(blocks)
    print(
        f"[00] MHC exclude {MHC_CHROM}:{MHC_START0:,}-{MHC_END0:,} "
        f"→ {len(blocks):,} blocks (was {n0:,})"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    bed_out = output_dir / "ld_blocks.used.bed"
    write_blocks_bed(bed_out, blocks)
    print(f"[00] wrote block list → {bed_out}")

    meta_rows: List[Tuple[str, int, int, Path, Path]] = []
    n = len(blocks)
    for i, block in enumerate(blocks, 1):
        if i == 1 or i % 25 == 0 or i == n:
            print(f"[00] PLINK LD {i}/{n}: {block[0]}:{block[1]}-{block[2]}", flush=True)
        chrom, start0, end0 = block
        result = run_plink_ld_block(
            genotype_stem=genotype_stem,
            sample_keep=sample_keep,
            chrom=chrom,
            start0=start0,
            end0=end0,
            output_dir=output_dir,
            plink_bin=plink_bin,
        )
        if result is None:
            print(
                f"[00] skip empty "
                f"chr{chrom[3:] if chrom.startswith('chr') else chrom}:{start0 + 1}-{end0}"
            )
            continue
        ld_path, bim_path = result
        meta_rows.append((chrom, start0 + 1, end0, ld_path, bim_path))

    meta_rows.sort(key=lambda r: (_chrom_key(_norm_chrom(str(r[0]))), r[1], r[2]))
    meta_path = output_dir / "ld_metadata.tsv"
    write_ld_metadata(meta_path, meta_rows)
    print(f"[00] wrote {meta_path} ({len(meta_rows):,} blocks with LD)")
    return meta_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--genotype_stem",
        required=True,
        type=Path,
        help="GTEx WGS PLINK stem (…SHAPEIT2_phased.MAF01)",
    )
    p.add_argument(
        "--sample_keep",
        type=Path,
        default=None,
        help="PLINK --keep file (FID IID). EUR subjects used for LD.",
    )
    p.add_argument(
        "--subject_phenotypes",
        type=Path,
        default=None,
        help="GTEx SubjectPhenotypesDS.txt; if set with no --sample_keep, write RACE==3 keep file under --output_dir",
    )
    p.add_argument(
        "--ld_blocks_bed",
        required=True,
        type=Path,
        help="EUR LD-block BED (chr/start/end; pyrho EUR); gaps are tiled to chrom ends",
    )
    p.add_argument(
        "--chrom_sizes",
        required=True,
        type=Path,
        help="chrom.sizes for tiling (hg38 primary autosomes)",
    )
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_keep = args.sample_keep
    if sample_keep is None:
        if args.subject_phenotypes is None:
            p.error("pass --sample_keep or --subject_phenotypes")
        sample_keep = write_eur_keep_from_phenotypes(
            args.subject_phenotypes, output_dir / "european_subject_ids.keep.txt"
        )

    build_ld_meta(
        genotype_stem=args.genotype_stem,
        sample_keep=sample_keep,
        ld_blocks_bed=args.ld_blocks_bed,
        output_dir=output_dir,
        chrom_sizes=args.chrom_sizes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
