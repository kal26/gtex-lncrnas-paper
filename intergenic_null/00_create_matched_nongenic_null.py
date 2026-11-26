#!/usr/bin/env python3
"""Build a matched intergenic-null GTF from a collapsed gene GTF + full annotation GTF.

Point this at:
  --collapsed_gtf  collapsed gene model used for quantification (exons to match)
  --full_gtf       full annotation GTF (all exons / gene spans for masking)
  --chrom_sizes    chrom.sizes for bedtools slop
  --output         output null GTF

Recommended --output names (strandedness of the collapsed input):
  gencode.v47.matched_null.unstranded.gtf
  gencode.v47.matched_null.stranded.gtf
  Macaque_Mmul10.matched_null.stranded.gtf
  Marmoset_mCalJa1.2.matched_null.stranded.gtf

Same algorithm as the paper (bedtools preprocess + greedy placement). All paths
are explicit CLI arguments; there are no defaults.
"""
from __future__ import annotations

import argparse
import bisect
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Unplaced scaffolds often have no usable intergenic interval after gene masking; chrM and
# patch/alternate contigs are usually omitted from primary-assembly analyses.
DEFAULT_EXCLUDE_CHROM_REGEX = r"^(chrUn_.*|chrM|.+_random|.+_alt)$"


@dataclass(frozen=True)
class NullConfig:
    collapsed_gtf: Path
    full_gtf: Path
    chrom_sizes: Path
    output_gtf: Path
    gtf_header_lines: tuple[str, ...]


_ATTR_PAT = {
    "gene_id": re.compile(r'gene_id "([^"]*)"'),
    "transcript_id": re.compile(r'transcript_id "([^"]*)"'),
    "exon_id": re.compile(r'exon_id "([^"]*)"'),
}


def _attr_lookup(attr: str, name: str):
    pat = _ATTR_PAT.get(name)
    if pat is None:
        return pd.NA
    m = pat.search(attr)
    return m.group(1) if m else pd.NA


def read_collapsed_gtf(path: Path, exclude_chrom_re: re.Pattern | None) -> pd.DataFrame:
    cols = [
        "Chromosome",
        "Source",
        "Feature",
        "Start",
        "End",
        "Score",
        "Strand",
        "Frame",
        "attribute",
    ]
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=cols,
        dtype={"Chromosome": str},
        keep_default_na=False,
    )
    if df.empty:
        raise ValueError(f"No GTF rows in {path}")
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    for k in ("gene_id", "transcript_id", "exon_id"):
        df[k] = df["attribute"].map(lambda a: _attr_lookup(a, k))
    if exclude_chrom_re is not None:
        n0 = len(df)
        mask = ~df["Chromosome"].astype(str).map(lambda c: bool(exclude_chrom_re.fullmatch(c)))
        df = df.loc[mask].copy()
        dropped = n0 - len(df)
        if dropped:
            print(f"[read_collapsed_gtf] dropped {dropped} rows matching exclude-chrom regex")
    if df.empty:
        raise ValueError(f"No GTF rows left after chromosome exclusion in {path}")
    return df


def normalize_exon_id(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).split(";")[0].strip()


def _awk_exon_bed(src: Path) -> subprocess.Popen:
    prog = (
        r'$3=="exon" { '
        r'match($0, /exon_id "([^"]+)"/, eid); '
        r'match($0, /gene_id "([^"]+)"/, gid); '
        r"print $1, $4-1, $5, eid[1], gid[1] }"
    )
    return subprocess.Popen(
        ["awk", "-v", "OFS=\t", prog, str(src)],
        stdout=subprocess.PIPE,
    )


def _run_sort_bed(inp: subprocess.Popen, out_path: Path) -> None:
    assert inp.stdout is not None
    with open(out_path, "wb") as out_f:
        subprocess.run(["bedtools", "sort", "-i", "-"], stdin=inp.stdout, stdout=out_f, check=True)
    inp.stdout.close()
    inp.wait()
    if inp.returncode != 0:
        raise subprocess.CalledProcessError(inp.returncode, "awk")


def bedtools_prepare(tmp_dir: Path, cfg: NullConfig) -> None:
    """Match shell pipeline: exon BEDs, merged genes, slop 1000 on exons and genes, intergenic BED."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    all_exons = tmp_dir / "all_exons.bed"
    exons_match = tmp_dir / "exons_to_match.bed"
    merged_genes = tmp_dir / "merged_genes.bed"
    exons_buf = tmp_dir / "exons_buffer.bed"
    intronic = tmp_dir / "intronic_regions.bed"
    genes_buf = tmp_dir / "genes_buffer.bed"
    intergenic = tmp_dir / "intergenic_regions.bed"
    bg = tmp_dir / "background_regions.bed"

    _run_sort_bed(_awk_exon_bed(cfg.full_gtf), all_exons)
    _run_sort_bed(_awk_exon_bed(cfg.collapsed_gtf), exons_match)

    merge_cmd = (
        f"awk -v OFS='\\t' '$3==\"gene\" {{print $1, $4-1, $5}}' '{cfg.full_gtf}' "
        f"| bedtools sort -i - | bedtools merge -i - > '{merged_genes}'"
    )
    subprocess.run(merge_cmd, shell=True, check=True)

    with open(exons_buf, "wb") as exo:
        subprocess.run(
            [
                "bedtools",
                "slop",
                "-i",
                str(all_exons),
                "-g",
                str(cfg.chrom_sizes),
                "-b",
                "1000",
            ],
            stdout=exo,
            check=True,
        )
    with open(intronic, "wb") as intr:
        subprocess.run(
            ["bedtools", "subtract", "-a", str(merged_genes), "-b", str(exons_buf)],
            stdout=intr,
            check=True,
        )
    with open(genes_buf, "wb") as gb:
        subprocess.run(
            [
                "bedtools",
                "slop",
                "-i",
                str(merged_genes),
                "-g",
                str(cfg.chrom_sizes),
                "-b",
                "1000",
            ],
            stdout=gb,
            check=True,
        )

    whole_genome = tmp_dir / "whole_genome.bed"
    subprocess.run(
        f"awk -v OFS='\\t' '{{print $1, \"0\", $2}}' '{cfg.chrom_sizes}' > '{whole_genome}'",
        shell=True,
        check=True,
    )
    with open(intergenic, "wb") as ig:
        subprocess.run(
            ["bedtools", "subtract", "-a", str(whole_genome), "-b", str(genes_buf)],
            stdout=ig,
            check=True,
        )

    with open(bg, "wb") as bgf:
        subprocess.run(
            f"cat '{intergenic}' '{intronic}' | bedtools sort -i -",
            shell=True,
            stdout=bgf,
            check=True,
        )


def get_closest_background_no_overlap(
    exon: pd.Series,
    background_starts: np.ndarray,
    background_ends: np.ndarray,
) -> tuple[int, int] | None:
    exon_length = int(exon["end"]) - int(exon["start"])
    idx_right = bisect.bisect_left(background_starts, exon["start"])
    while idx_right < len(background_starts) and background_starts[idx_right] < exon["start"]:
        idx_right += 1
    idx_left = idx_right - 1
    max_iter = len(background_starts)
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        if idx_right < len(background_starts):
            rs, re = background_starts[idx_right], background_ends[idx_right]
            if re - rs >= exon_length:
                background_starts[idx_right] = background_starts[idx_right] + exon_length
                return int(rs), int(rs + exon_length)
        if idx_left >= 0:
            ls, le = background_starts[idx_left], background_ends[idx_left]
            if le - ls >= exon_length:
                background_ends[idx_left] = background_ends[idx_left] - exon_length
                return int(le - exon_length), int(le)
        idx_right += 1
        idx_left -= 1
    return None


def place_null_exons(tmp_dir: Path, exclude_chrom_re: re.Pattern | None) -> pd.DataFrame:
    exon_path = tmp_dir / "exons_to_match.bed"
    intergenic_path = tmp_dir / "intergenic_regions.bed"
    exons = pd.read_csv(
        exon_path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "exon_id", "gene_id"],
    )
    if exclude_chrom_re is not None:
        n0 = len(exons)
        keep = ~exons["chr"].astype(str).map(lambda c: bool(exclude_chrom_re.fullmatch(c)))
        exons = exons.loc[keep].copy()
        dropped = n0 - len(exons)
        if dropped:
            print(f"[place_null_exons] skipped {dropped} exon BED rows matching exclude-chrom regex")
        if exons.empty:
            raise RuntimeError("No exons left after chromosome exclusion")

    background = pd.read_csv(
        intergenic_path,
        sep="\t",
        header=None,
        names=["chr", "start", "end"],
    )
    background = background[background["chr"].isin(exons["chr"].unique())]
    background = background.sort_values(["chr", "start"])

    empty_bg = sorted(set(exons["chr"].unique()) - set(background.groupby("chr").groups.keys()))
    if empty_bg:
        n_hit = int(exons["chr"].isin(empty_bg).sum())
        raise RuntimeError(
            f"{len(empty_bg)} chromosome(s) have exons but no intergenic intervals in BED "
            f"(affects {n_hit} exons). Examples: {empty_bg[:8]}. "
            "Primary-assembly analyses exclude chrUn_*/chrM/*_random/*_alt by default."
        )

    out_chunks = []
    failures: list[str] = []
    for chrom, bg_chr in background.groupby("chr"):
        starts = bg_chr["start"].to_numpy(dtype=np.int64).copy()
        ends = bg_chr["end"].to_numpy(dtype=np.int64).copy()
        ex_chr = exons[exons["chr"] == chrom].sort_values(["start", "end", "exon_id"])
        rows = []
        for _, ex in ex_chr.iterrows():
            got = get_closest_background_no_overlap(ex, starts, ends)
            if got is None:
                failures.append(f"{chrom}:{ex['exon_id']} (gene {ex['gene_id']})")
            else:
                rows.append((got[0], got[1], ex["exon_id"], ex["gene_id"], chrom))
        if rows:
            out_chunks.append(pd.DataFrame(rows, columns=["start", "end", "exon_id", "gene_id", "chr"]))
    if failures:
        n = len(failures)
        sample = "; ".join(failures[:5])
        raise RuntimeError(
            f"Could not place {n} exons into intergenic background (showing up to 5): {sample}"
        )
    return pd.concat(out_chunks, ignore_index=True)


def apply_null_to_gtf(null_bed: pd.DataFrame, collapsed_df: pd.DataFrame) -> pd.DataFrame:
    gtf = collapsed_df.copy().reset_index(drop=False)
    if gtf.columns[0] != "index":
        gtf = gtf.rename(columns={gtf.columns[0]: "index"})
    nb = null_bed.copy()
    nb["exon_id"] = nb["exon_id"].map(normalize_exon_id)

    exon_only = gtf[gtf["Feature"] == "exon"].copy()
    exon_only["exon_id"] = exon_only["exon_id"].map(normalize_exon_id)
    merged_ex = exon_only.merge(
        nb[["exon_id", "start", "end"]],
        on="exon_id",
        how="inner",
    )
    if len(merged_ex) != len(exon_only):
        raise RuntimeError(
            f"exon merge: expected {len(exon_only)} rows, got {len(merged_ex)} (check exon_id keys)"
        )

    coord = merged_ex[["index", "start", "end"]].copy()
    full = gtf.merge(coord, on="index", how="left", suffixes=("", "_null"))
    gene_bounds = full.groupby("gene_id", dropna=False).agg({"start": "min", "end": "max"})
    gene_bounds = gene_bounds.rename(columns={"start": "start_gene", "end": "end_gene"})
    full = full.merge(gene_bounds, on="gene_id", how="left")

    full["Start"] = np.where(full["Feature"] == "exon", full["start"] + 1, full["start_gene"] + 1)
    full["End"] = np.where(full["Feature"] == "exon", full["end"], full["end_gene"])
    return full


def write_minimal_gtf(out_df: pd.DataFrame, output_file: Path, header_lines: tuple[str, ...]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        for _, row in out_df.iterrows():
            eid = row["exon_id"]
            if row["Feature"] == "exon" and not pd.isna(eid):
                eid_out = normalize_exon_id(eid)
            else:
                eid_out = ""
            attrs = (
                f'gene_id "{row["gene_id"]}"; transcript_id "{row["transcript_id"]}"; '
                f'exon_id "{eid_out}";'
            )
            line = (
                f'{row["Chromosome"]}\t{row["Source"]}\t{row["Feature"]}\t'
                f'{int(row["Start"])}\t{int(row["End"])}\t{row["Score"]}\t'
                f'{row["Strand"]}\t{row["Frame"]}\t{attrs}\n'
            )
            f.write(line)


def run_null(
    cfg: NullConfig,
    tmp_dir: Path,
    exclude_chrom_re: re.Pattern | None,
) -> None:
    for p in (cfg.collapsed_gtf, cfg.full_gtf, cfg.chrom_sizes):
        if not p.exists():
            raise FileNotFoundError(p)
    bedtools_prepare(tmp_dir, cfg)
    null_bed = place_null_exons(tmp_dir, exclude_chrom_re)
    collapsed = read_collapsed_gtf(cfg.collapsed_gtf, exclude_chrom_re)
    out_tab = apply_null_to_gtf(null_bed, collapsed)
    write_minimal_gtf(out_tab, cfg.output_gtf, cfg.gtf_header_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collapsed_gtf",
        type=Path,
        required=True,
        help="Collapsed gene GTF (exons to place nulls for; e.g. gencode genes GTF)",
    )
    parser.add_argument(
        "--full_gtf",
        type=Path,
        required=True,
        help="Full annotation GTF (all exons + gene spans used for masking)",
    )
    parser.add_argument(
        "--chrom_sizes",
        type=Path,
        required=True,
        help="Chromosome sizes file for bedtools slop (two columns: chrom, size)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output matched intergenic-null GTF "
            "(e.g. gencode.v47.matched_null.unstranded.gtf or "
            "…matched_null.stranded.gtf)"
        ),
    )
    args = parser.parse_args()
    exclude_chrom_re = re.compile(DEFAULT_EXCLUDE_CHROM_REGEX)
    cfg = NullConfig(
        collapsed_gtf=args.collapsed_gtf,
        full_gtf=args.full_gtf,
        chrom_sizes=args.chrom_sizes,
        output_gtf=args.output,
        gtf_header_lines=(
            "##description: matched intergenic null exons",
            "##format: gtf",
        ),
    )
    tmp_dir = cfg.output_gtf.parent / f"{cfg.output_gtf.name}.tmpdir"
    print(f"[00_create_matched_nongenic_null] -> {cfg.output_gtf}")
    try:
        run_null(
            cfg,
            tmp_dir,
            exclude_chrom_re=exclude_chrom_re,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
