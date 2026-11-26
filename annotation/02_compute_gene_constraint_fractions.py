#!/usr/bin/env python3
"""Gene-level Zoonomia primate constraint fractions (Sullivan et al. Science 2023).

For each gene (novel_lncRNA, existing_lncRNA, protein_coding, pseudogene), score the
non-overlapping (*_unique) union of:
  - exons (bases not overlapping any other gene's exons, either strand)
  - splice-junction GT-AG dinucleotides (same exclusivity)

Primate constraint: Zoonomia primate phastCons >= 0.961
  (threshold matched to mammalian genome constrained fraction)

Gene score = fraction of scored bases meeting the threshold. Output columns used
by 03_classify_lncrna.py: frac_cons_exon_unique_primate, frac_cons_sj_unique_primate.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


PATH_GTF: Path
PATH_BW_PRIMATE: Path
PATH_CLASS: Path
OUT_DIR: Path
OUT_PATH: Path

PRIMATE_PHASTCONS_MIN = 0.961
INCLUDE_GENE_TYPES = {"novel_lncRNA", "existing_lncRNA", "protein_coding", "pseudogene"}
FEATURE_KEYS = ["exon_unique", "sj_unique"]
WINDOW_SIZE = 10_000_000


def strip_version(eid: str) -> str:
    return eid.split(".")[0]


def parse_gtf_attr(attr_str: str) -> dict:
    return dict(re.findall(r'(\w+) "([^"]+)"', attr_str))


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def subtract_intervals(
    a: list[tuple[int, int]], b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return parts of *a* not covered by *b*. Both must be sorted, non-overlapping."""
    if not a:
        return []
    if not b:
        return list(a)
    out: list[tuple[int, int]] = []
    j = 0
    nb = len(b)
    for as_, ae in a:
        start = as_
        while j < nb and b[j][1] <= start:
            j += 1
        k = j
        cur = start
        while k < nb and b[k][0] < ae:
            bs, be = b[k]
            if bs > cur:
                out.append((cur, min(bs, ae)))
            cur = max(cur, be)
            if cur >= ae:
                break
            k += 1
        if cur < ae:
            out.append((cur, ae))
    return out


def get_introns(exons: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (exons[i][1], exons[i + 1][0])
        for i in range(len(exons) - 1)
        if exons[i + 1][0] > exons[i][1]
    ]


def compute_exclusive_exons(
    genes: dict[str, dict],
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]]]:
    """Exonic bases unique to each gene (no other gene exon, either strand)."""
    by_chrom: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for gid, g in genes.items():
        chrom = g["chrom"]
        for s, e in g["exons"]:
            if e > s:
                by_chrom[chrom].append((s, e, gid))

    exclusive: dict[str, list[tuple[int, int]]] = {gid: [] for gid in genes}
    any_exon: dict[str, list[tuple[int, int]]] = {}

    for chrom, ivals in by_chrom.items():
        any_exon[chrom] = merge_intervals([(s, e) for s, e, _ in ivals])

        events: list[tuple[int, int, str]] = []
        for s, e, gid in ivals:
            events.append((s, 1, gid))
            events.append((e, -1, gid))
        events.sort(key=lambda x: (x[0], 0 if x[1] < 0 else 1, x[1], x[2]))

        active: dict[str, int] = {}
        prev: int | None = None
        for pos, delta, gid in events:
            if prev is not None and pos > prev and len(active) == 1:
                only_gid = next(iter(active))
                exclusive[only_gid].append((prev, pos))
            if delta == 1:
                active[gid] = active.get(gid, 0) + 1
            else:
                active[gid] -= 1
                if active[gid] == 0:
                    del active[gid]
            prev = pos

    for gid in exclusive:
        exclusive[gid] = merge_intervals(exclusive[gid])

    n_excl = sum(1 for iv in exclusive.values() if iv)
    log(f"  Exclusive (non-overlapping) exon genes: {n_excl:,} / {len(genes):,}")
    return exclusive, any_exon


def load_gene_meta() -> tuple[set[str], pd.DataFrame]:
    class_df = pd.read_table(
        PATH_CLASS,
        usecols=["gene_id_stable", "gene_type_grouped"],
    )
    class_df = class_df[class_df["gene_type_grouped"].isin(INCLUDE_GENE_TYPES)].copy()
    return set(class_df["gene_id_stable"]), class_df


def parse_gtf_features(target_gene_ids: set[str]) -> dict[str, dict]:
    """Build gene-level exon and splice-junction intervals (0-based half-open)."""
    log(f"Parsing GTF exons: {PATH_GTF}")
    tx: dict[str, dict] = {}
    gene_chrom: dict[str, str] = {}
    gene_strand: dict[str, str] = {}
    n_lines = 0

    with open(PATH_GTF) as fh:
        for line in fh:
            n_lines += 1
            if n_lines % 2_000_000 == 0:
                log(f"  ...{n_lines // 1_000_000}M lines")
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue

            attrs = parse_gtf_attr(fields[8])
            gene_id_stable = strip_version(attrs.get("gene_id", ""))
            if gene_id_stable not in target_gene_ids:
                continue

            chrom = fields[0]
            start = int(fields[3]) - 1
            end = int(fields[4])
            strand = fields[6]
            gene_chrom[gene_id_stable] = chrom
            gene_strand[gene_id_stable] = strand

            tx_id = attrs.get("transcript_id", "")
            if not tx_id:
                continue
            if tx_id not in tx:
                tx[tx_id] = {
                    "gene_id_stable": gene_id_stable,
                    "chrom": chrom,
                    "strand": strand,
                    "exons": [],
                }
            tx[tx_id]["exons"].append((start, end))

    for t in tx.values():
        t["exons"].sort()

    genes: dict[str, dict] = {}
    for gid in target_gene_ids:
        if gid not in gene_chrom:
            continue
        genes[gid] = {
            "chrom": gene_chrom[gid],
            "strand": gene_strand[gid],
            "exons": [],
            "sj": [],
        }

    for t in tx.values():
        gid = t["gene_id_stable"]
        if gid not in genes:
            continue
        exons = t["exons"]
        if not exons:
            continue
        genes[gid]["exons"].extend(exons)
        for i_start, i_end in get_introns(exons):
            if i_end - i_start < 4:
                continue
            genes[gid]["sj"].append((i_start, i_start + 2))
            genes[gid]["sj"].append((i_end - 2, i_end))

    for gid, g in genes.items():
        g["exons"] = merge_intervals(g["exons"])
        g["sj"] = merge_intervals(g["sj"])

    log(f"  Genes with exons: {len(genes):,} / {len(target_gene_ids):,}")
    return genes


def _resolve_chrom(bw, chrom: str) -> str | None:
    chroms = bw.chroms() or {}
    if chrom in chroms:
        return chrom
    bare = chrom.replace("chr", "")
    if bare in chroms:
        return bare
    if f"chr{bare}" in chroms:
        return f"chr{bare}"
    return None


def count_track(
    bw,
    cname: str | None,
    chrom_len: int,
    threshold: float,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval (n_scored, n_constrained) from one contiguous bigWig read."""
    n = len(starts)
    n_scored = np.zeros(n, dtype=np.int64)
    n_cons = np.zeros(n, dtype=np.int64)
    if cname is None or n == 0:
        return n_scored, n_cons

    s = np.clip(starts, 0, chrom_len)
    e = np.maximum(np.clip(ends, 0, chrom_len), s)
    lo, hi = int(s.min()), int(e.max())
    if hi <= lo:
        return n_scored, n_cons

    v = np.asarray(bw.values(cname, lo, hi, numpy=True), dtype=np.float64)
    ok = np.isfinite(v)
    cons = ok & (np.where(ok, v, -np.inf) >= threshold)
    cs_ok = np.concatenate(([0], np.cumsum(ok, dtype=np.int64)))
    cs_cons = np.concatenate(([0], np.cumsum(cons, dtype=np.int64)))
    a = s - lo
    b = e - lo
    return cs_ok[b] - cs_ok[a], cs_cons[b] - cs_cons[a]


def build_feature_intervals(genes: dict[str, dict]) -> pd.DataFrame:
    """Flat table of scored unique-exon / unique-SJ intervals."""
    log("Computing exons unique to each gene (no other-gene exon overlap)")
    exclusive_by_gene, any_exon_by_chrom = compute_exclusive_exons(genes)
    any_exon_starts = {c: np.array([s for s, _ in iv]) for c, iv in any_exon_by_chrom.items()}

    chroms, starts, ends, gidx, fidx = [], [], [], [], []
    for gi, (gid, g) in enumerate(genes.items()):
        chrom = g["chrom"]
        exclusive = exclusive_by_gene.get(gid, [])
        all_iv = g["exons"] + g["sj"]
        if all_iv:
            span_lo = min(s for s, _ in all_iv)
            span_hi = max(e for _, e in all_iv)
            chrom_exons = any_exon_by_chrom.get(chrom, [])
            i0 = max(0, int(np.searchsorted(any_exon_starts[chrom], span_lo, "right")) - 1)
            i1 = int(np.searchsorted(any_exon_starts[chrom], span_hi, "left"))
            other_exons = subtract_intervals(chrom_exons[i0:i1], exclusive)
        else:
            other_exons = []
        feats = {
            "exon_unique": exclusive,
            "sj_unique": subtract_intervals(g["sj"], other_exons),
        }
        for fi, key in enumerate(FEATURE_KEYS):
            for s, e in feats[key]:
                if e > s:
                    chroms.append(chrom)
                    starts.append(s)
                    ends.append(e)
                    gidx.append(gi)
                    fidx.append(fi)

    return pd.DataFrame(
        {
            "chrom": chroms,
            "start": np.asarray(starts, dtype=np.int64),
            "end": np.asarray(ends, dtype=np.int64),
            "gene_idx": np.asarray(gidx, dtype=np.int64),
            "feat_idx": np.asarray(fidx, dtype=np.int64),
        }
    )


def compute_table(
    genes: dict[str, dict],
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    meta = meta_df.drop_duplicates("gene_id_stable").set_index("gene_id_stable")
    gene_ids = list(genes)
    meta = meta.reindex(gene_ids)
    gene_types = [None if pd.isna(t) else t for t in meta["gene_type_grouped"]]

    iv = build_feature_intervals(genes)
    n_genes, n_feat = len(gene_ids), len(FEATURE_KEYS)
    log(f"  Intervals to score: {len(iv):,}")

    bw_p = pyBigWig.open(str(PATH_BW_PRIMATE))
    chrom_info = {}
    for chrom in iv["chrom"].unique():
        cp = _resolve_chrom(bw_p, chrom)
        chrom_info[chrom] = (
            cp,
            int(bw_p.chroms()[cp]) if cp is not None else 0,
        )

    iv["win"] = iv["start"] // WINDOW_SIZE
    tasks = []
    for (chrom, _), idx in iv.groupby(["chrom", "win"], sort=False).indices.items():
        tasks.append(
            (idx, iv["start"].values[idx], iv["end"].values[idx], *chrom_info[chrom])
        )

    log(f"Scoring {len(tasks):,} windows\n  bigWig: {PATH_BW_PRIMATE}")

    n_iv = len(iv)
    sp = np.zeros(n_iv, dtype=np.int64)
    cp_ = np.zeros(n_iv, dtype=np.int64)
    log_every = max(1, len(tasks) // 20)
    for done, (idx, starts, ends, cname_p, len_p) in enumerate(tasks, 1):
        a, b = count_track(bw_p, cname_p, len_p, PRIMATE_PHASTCONS_MIN, starts, ends)
        sp[idx], cp_[idx] = a, b
        if done % log_every == 0 or done == len(tasks):
            log(f"  Scored window {done:,}/{len(tasks):,}")
    bw_p.close()

    g, f = iv["gene_idx"].values, iv["feat_idx"].values
    flat = g * n_feat + f

    def per_gene(x: np.ndarray) -> np.ndarray:
        return np.bincount(flat, weights=x, minlength=n_genes * n_feat).reshape(
            n_genes, n_feat
        )

    n_sp, n_cp = per_gene(sp), per_gene(cp_)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_p = np.where(n_sp > 0, n_cp / n_sp, np.nan)

    out = pd.DataFrame(
        {
            "gene_id_stable": gene_ids,
            "gene_type_grouped": gene_types,
            "frac_cons_exon_unique_primate": frac_p[:, FEATURE_KEYS.index("exon_unique")],
            "frac_cons_sj_unique_primate": frac_p[:, FEATURE_KEYS.index("sj_unique")],
        }
    )
    return out


def main(argv: list[str] | None = None) -> None:
    global PATH_GTF, PATH_BW_PRIMATE, PATH_CLASS, OUT_DIR, OUT_PATH

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full_gtf", type=Path, required=True)
    p.add_argument("--primate-bw", type=Path, required=True)
    p.add_argument("--gencode-classification", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args(argv)

    PATH_GTF = args.full_gtf
    PATH_BW_PRIMATE = args.primate_bw
    PATH_CLASS = args.gencode_classification
    OUT_DIR = args.output_dir
    OUT_PATH = OUT_DIR / "gene_constraint_fractions.tsv.gz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [PATH_GTF, PATH_BW_PRIMATE, PATH_CLASS]:
        if not path.exists():
            raise FileNotFoundError(path)

    target_genes, meta_df = load_gene_meta()
    genes = parse_gtf_features(target_genes)
    df = compute_table(genes, meta_df)
    df.to_csv(OUT_PATH, sep="\t", index=False, compression="gzip")
    log(f"Wrote {len(df):,} genes → {OUT_PATH}")

    for col in ("frac_cons_exon_unique_primate", "frac_cons_sj_unique_primate"):
        s = df[col]
        log(f"  {col}: scored={s.notna().sum():,}; median={s.median(skipna=True):.4f}")


if __name__ == "__main__":
    main()
