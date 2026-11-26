#!/usr/bin/env python3
"""Locus clustering for LD-block coloc (import-only; cross-tissue or mode-as-tissue).

Expects ColocBoost tables from ``coloc.combine_tissue``::

  {coloc_dir}/{Tissue}.ColocBoost.parquet
  # or discovery mode-as-tissue:
  {coloc_dir}/{mode}.ColocBoost.parquet   (cos_id = {mode}_{N})

``cluster_loci(..., event_filter=...)`` (used by ``02`` / ``05``):

- ``gwas_qtl_cos`` (default): only **GWAS CoS with ≥1 QTL gene**; drop
  QTL-only / GWAS-only CoS and all uCoS.
- ``gwas_or_qtl``: keep CoS/uCoS with GWAS and/or QTL (composition flags).
- ``gwas_any``: any CoS/uCoS with a GWAS trait (QTL-only dropped); mode-merge
  path that must retain GWAS uCoS.

Loci are clustered when sets share a lead, or share a credible-set variant and
clear min cross-set |r| > 0.5 or median cross-set |r| > 0.8.

Output (single file)::

  {output_path}   e.g. cross_tissue_loci.parquet
  columns: locus_id, member_cos_ids, gene_ids_robust, gene_types_robust,
           gwas_traits, gwas_gws_traits
"""

from __future__ import annotations

import glob
import gzip
import lzma
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


ENSG_RE = re.compile(r"ENSG\d+")
ENSG_CAPTURE = r"(ENSG\d+)"
OVERLAP_MIN_ABS_R = 0.5
OVERLAP_MEDIAN_ABS_R = 0.8
GWAS_P_TIER_GWS = float(-np.log10(5.7e-9))
GWAS_P_TIER_SUB = float(-np.log10(1e-5))
GWAS_P_TIER_RANK = {"gw_sig": 0, "sub_sig": 1, "not_sig": 2, "no_gwas": 3}

LOCI_COLS = [
    "locus_id",
    "member_cos_ids",
    "gene_ids_robust",
    "gene_types_robust",
    "gwas_traits",
    "gwas_gws_traits",
]

MODES_GWAS = ("joint_gwas",)
MODES_WITH_XQTL = ("joint_gwas",)  # xQTL-only tables are not included; joint covers GWAS+QTL
EVENT_FILTERS = ("gwas_qtl_cos", "gwas_or_qtl", "gwas_any")

ROBUST_QTL_TYPE_COL = "gene_types_robust"
RELAXED_QTL_TYPE_COL = "gene_types_relaxed"
ROBUST_GENE_ID_COL = "gene_ids_robust"
RELAXED_GENE_ID_COL = "gene_ids_relaxed"


def _open_text(path: str):
    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic[:6] == b"\xfd7zXZ\x00":
        return lzma.open(path, "rt")
    if magic[:2] == b"\x1f\x8b":
        return gzip.open(path, "rt")
    return open(path, "rt")


def normalize_bim_id(s: str) -> str:
    s = s.replace("_", ":")
    s = re.sub(r":b[0-9]+$", "", s)
    return ":".join(s.split(":")[:4])


def parse_chrom_pos(vid: str) -> Optional[Tuple[str, int]]:
    parts = str(vid).split(":")
    if len(parts) < 2:
        return None
    chrom = parts[0][3:] if parts[0].startswith("chr") else parts[0]
    try:
        return chrom, int(parts[1])
    except (ValueError, TypeError):
        return None


def lead_chrom(vid) -> str:
    cp = parse_chrom_pos(str(vid) if vid is not None else "")
    return cp[0] if cp is not None else "?"


def as_str_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    if isinstance(val, np.ndarray):
        return [str(v) for v in val.tolist()]
    if isinstance(val, (list, tuple, set)):
        return [str(v) for v in val]
    s = str(val).strip()
    if not s or s in ("nan", "None", "<NA>"):
        return []
    return [s]


def _union_lists(series: pd.Series) -> list:
    out: list = []
    seen = set()
    for val in series:
        for x in as_str_list(val):
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def is_ucos_row(row: pd.Series) -> bool:
    """Heuristic: uCoS if no GWAS+QTL together (phenotype sides incomplete)."""
    # Public cos_id is Tissue_N / mode_N / Tissue.mode_N — ucos vs cos is not encoded.
    # Use phenotype_ids: both GWAS and QTL → CoS; else uCoS-like.
    ph = str(row.get("phenotype_ids", "") or "")
    has_gwas = False
    has_qtl = False
    for part in ph.split(";"):
        s = part.strip().lower()
        if not s:
            continue
        if "expression" in s or "splicing" in s:
            has_qtl = True
        else:
            has_gwas = True
    return not (has_gwas and has_qtl)


def series_has_gwas(df: pd.DataFrame) -> pd.Series:
    if "gwas_traits" in df.columns:
        return df["gwas_traits"].map(lambda x: len(as_str_list(x)) > 0)
    if "phenotype_ids" in df.columns:
        return df["phenotype_ids"].map(
            lambda p: any(
                "expression" not in s.lower() and "splicing" not in s.lower()
                for s in str(p or "").split(";")
                if s.strip()
            )
        )
    return pd.Series([False] * len(df), index=df.index)


def series_has_robust_qtl(df: pd.DataFrame) -> pd.Series:
    if ROBUST_GENE_ID_COL in df.columns:
        return df[ROBUST_GENE_ID_COL].map(lambda x: len(as_str_list(x)) > 0)
    if ROBUST_QTL_TYPE_COL in df.columns:
        return df[ROBUST_QTL_TYPE_COL].map(lambda x: len(as_str_list(x)) > 0)
    return pd.Series([False] * len(df), index=df.index)


def filter_gwas_cos(df: pd.DataFrame, *, quiet: bool = False, label: str = "") -> pd.DataFrame:
    """Keep GWAS+QTL CoS rows only."""
    if df.empty:
        return df
    has_g = series_has_gwas(df)
    has_q = series_has_robust_qtl(df)
    # Drop uCoS-like (missing one side)
    keep = has_g & has_q
    out = df.loc[keep].copy()
    if not quiet:
        print(f"[{label}] gwas_qtl_cos: {len(out):,}/{len(df):,} kept")
    return out


def filter_gwas_any_sets(
    df: pd.DataFrame, *, quiet: bool = False, label: str = ""
) -> pd.DataFrame:
    if df.empty:
        return df
    keep = series_has_gwas(df)
    out = df.loc[keep].copy()
    if not quiet:
        print(f"[{label}] gwas_any: {len(out):,}/{len(df):,} kept")
    return out


def filter_gwas_or_qtl_sets(
    df: pd.DataFrame, *, quiet: bool = False, label: str = ""
) -> pd.DataFrame:
    if df.empty:
        return df
    keep = series_has_gwas(df) | series_has_robust_qtl(df)
    out = df.loc[keep].copy()
    if not quiet:
        print(f"[{label}] gwas_or_qtl: {len(out):,}/{len(df):,} kept")
    return out


def _apply_event_filter(
    df: pd.DataFrame, event_filter: str, *, quiet: bool, label: str
) -> pd.DataFrame:
    if event_filter == "gwas_qtl_cos":
        return filter_gwas_cos(df, quiet=quiet, label=label)
    if event_filter == "gwas_any":
        return filter_gwas_any_sets(df, quiet=quiet, label=label)
    if event_filter == "gwas_or_qtl":
        return filter_gwas_or_qtl_sets(df, quiet=quiet, label=label)
    raise ValueError(f"Unknown event_filter={event_filter!r}; expected one of {EVENT_FILTERS}")


def _lead_from_lists(variants, vcp) -> str:
    vs = as_str_list(variants)
    if not vs:
        return ""
    if vcp is None:
        return vs[0]
    try:
        ps = list(vcp) if not isinstance(vcp, list) else vcp
        if len(ps) != len(vs):
            return vs[0]
        arr = pd.to_numeric(pd.Series(ps), errors="coerce")
        if arr.notna().any():
            return vs[int(arr.values.argmax())]
    except Exception:  # noqa: BLE001
        pass
    return vs[0]


def read_ld_meta(ld_meta_path: str) -> pd.DataFrame:
    """Load LD-block metadata; expand ``path`` → ``ld_path``, ``bim_path``.

    ``00_build_ld_meta.py`` writes ``chrom start end path`` where ``path`` is
    ``{ld.gz},{bim}`` (comma-separated).
    """
    df = pd.read_csv(ld_meta_path, sep="\t")
    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    if "ld_path" not in df.columns:
        if "path" not in df.columns:
            raise KeyError(
                f"ld_meta missing ld_path/path columns: {list(df.columns)}"
            )
        parts = df["path"].astype(str).str.split(",", n=1, expand=True)
        df["ld_path"] = parts[0]
        df["bim_path"] = parts[1] if parts.shape[1] > 1 else ""
    return df


def _load_bim_maps(bim_path: str) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    id2idx: Dict[str, int] = {}
    pos2idx: Dict[str, List[int]] = defaultdict(list)
    with _open_text(bim_path) as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            vid = normalize_bim_id(parts[1])
            id2idx[vid] = i
            try:
                pos2idx[f"{parts[0]}:{int(parts[3])}"].append(i)
            except (ValueError, TypeError):
                pass
    return id2idx, dict(pos2idx)


def _variant_to_idx(
    vid: str, id2idx: Dict[str, int], pos2idx: Dict[str, List[int]]
) -> Optional[int]:
    key = normalize_bim_id(vid)
    if key in id2idx:
        return id2idx[key]
    cp = parse_chrom_pos(vid)
    if cp is None:
        return None
    chrom, pos = cp
    hits = pos2idx.get(f"chr{chrom}:{pos}") or pos2idx.get(f"{chrom}:{pos}") or []
    return hits[0] if len(hits) == 1 else None


def _load_ld_block(
    ld_path: str, bim_path: str
) -> Tuple[Dict[str, int], Dict[str, List[int]], List[np.ndarray]]:
    id2idx, pos2idx = _load_bim_maps(bim_path)
    rows: List[np.ndarray] = []
    with _open_text(ld_path) as fh:
        for line in fh:
            rows.append(np.fromstring(line.strip(), sep="\t", dtype=np.float32))
    return id2idx, pos2idx, rows


def _abs_r_matrix(rows: List[np.ndarray], ia: int, ib: int) -> float:
    if ia == ib:
        return 1.0
    lo, hi = (ia, ib) if ia < ib else (ib, ia)
    if hi >= len(rows) or lo >= len(rows[hi]):
        return 0.0
    return float(abs(rows[hi][lo]))


def _variants_in_block(
    variants: Set[str], id2idx: Dict[str, int], pos2idx: Dict[str, List[int]]
) -> List[int]:
    out = []
    for v in variants:
        i = _variant_to_idx(v, id2idx, pos2idx)
        if i is not None:
            out.append(i)
    return out


def _precompute_event_ld_indices(
    events: List[dict],
    chrom: str,
    ld_meta_chr: pd.DataFrame,
) -> List[Optional[Tuple]]:
    """Per-event: which LD tile + indices, or None."""
    cache: Dict[str, Tuple] = {}
    out: List[Optional[Tuple]] = []
    for ev in events:
        variants = ev.get("variants") or set()
        lead = ev.get("lead") or ""
        positions = []
        for v in set(variants) | ({lead} if lead else set()):
            cp = parse_chrom_pos(v)
            if cp is not None:
                positions.append(cp[1])
        if not positions:
            out.append(None)
            continue
        pos_min, pos_max = min(positions), max(positions)
        tiles = ld_meta_chr[
            (ld_meta_chr["end"].astype(int) >= pos_min)
            & (ld_meta_chr["start"].astype(int) <= pos_max)
        ]
        if tiles.empty:
            out.append(None)
            continue
        # Prefer tile containing the lead
        chosen = tiles.iloc[0]
        for _, row in tiles.iterrows():
            if lead:
                cp = parse_chrom_pos(lead)
                if cp and int(row["start"]) <= cp[1] <= int(row["end"]):
                    chosen = row
                    break
        ld_path = str(chosen["ld_path"])
        if ld_path not in cache:
            bim = chosen.get("bim_path")
            bim_file = bim if isinstance(bim, str) and bim else ld_path + ".bim"
            try:
                cache[ld_path] = _load_ld_block(ld_path, bim_file)
            except Exception as e:  # noqa: BLE001
                print(f"[cluster_loci] WARN LD {ld_path}: {e}", file=sys.stderr)
                cache[ld_path] = None  # type: ignore
        blk = cache[ld_path]
        if blk is None:
            out.append(None)
            continue
        id2idx, pos2idx, rows = blk
        idxs = _variants_in_block(set(variants), id2idx, pos2idx)
        out.append((id2idx, pos2idx, rows, idxs, lead))
    return out


def _cross_set_abs_r(
    a: Optional[Tuple], b: Optional[Tuple]
) -> Optional[np.ndarray]:
    if a is None or b is None:
        return None
    id2idx_a, pos2idx_a, rows_a, idxs_a, _ = a
    id2idx_b, pos2idx_b, rows_b, idxs_b, _ = b
    # Require same block object
    if rows_a is not rows_b:
        return None
    if not idxs_a or not idxs_b:
        return None
    vals = [_abs_r_matrix(rows_a, ia, ib) for ia in idxs_a for ib in idxs_b]
    return np.asarray(vals, dtype=np.float64) if vals else None


def _overlap_r_merges(arr: Optional[np.ndarray]) -> bool:
    if arr is None or arr.size == 0:
        return False
    return float(arr.min()) > OVERLAP_MIN_ABS_R or float(np.median(arr)) > OVERLAP_MEDIAN_ABS_R


def cluster_events(
    events: List[dict],
    *,
    chrom: str,
    ld_meta_chr: pd.DataFrame,
) -> List[int]:
    """Union-Find clustering of CoS events on one chromosome."""
    n = len(events)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Same lead merges
    lead_to = defaultdict(list)
    for i, ev in enumerate(events):
        lead = str(ev.get("lead") or "")
        if lead and lead not in {"", "nan", "None", "<NA>"}:
            lead_to[lead].append(i)
    for idxs in lead_to.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    pre = _precompute_event_ld_indices(events, chrom, ld_meta_chr)
    var_sets = [set(ev.get("variants") or []) for ev in events]
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if not (var_sets[i] & var_sets[j]):
                continue
            if _overlap_r_merges(_cross_set_abs_r(pre[i], pre[j])):
                union(i, j)

    root_map: Dict[int, int] = {}
    clusters: List[int] = []
    next_id = 0
    for i in range(n):
        r = find(i)
        if r not in root_map:
            root_map[r] = next_id
            next_id += 1
        clusters.append(root_map[r])
    return clusters


def load_colocboost_events(
    coloc_dir: str,
    tissues: Optional[Set[str]],
    quiet: bool,
    event_filter: str = "gwas_qtl_cos",
    pattern: str = "*.ColocBoost.parquet",
) -> pd.DataFrame:
    """Load ColocBoost tables as clustering events."""
    paths = sorted(glob.glob(os.path.join(coloc_dir, pattern)))
    rows: List[pd.DataFrame] = []
    for lp in paths:
        base = os.path.basename(lp)
        # {Tissue}.ColocBoost.parquet OR {Tissue}.{mode}.ColocBoost.parquet
        # OR {mode}.ColocBoost.parquet (mode-as-tissue for discovery merge)
        name = base.replace(".ColocBoost.parquet", "")
        # For mode-as-tissue links, name is the mode (joint / protein_coding / …)
        tissue = name.split(".")[0] if "." in name else name
        label = name  # full stem for logging
        if tissues is not None and tissue not in tissues and name not in tissues:
            continue
        df = pd.read_parquet(lp)
        if df.empty:
            continue
        df = df.copy()
        df = _apply_event_filter(
            df, event_filter, quiet=quiet, label=label
        )
        if df.empty:
            continue
        if "variants" not in df.columns:
            raise KeyError(f"{lp} missing variants list column")
        df["tissue"] = name  # for mode-merge, tissue==mode name
        df["cos_variants"] = df["variants"].map(
            lambda v: sorted(set(as_str_list(v)))
        )
        df["lead_variant_id"] = [
            _lead_from_lists(v, p)
            for v, p in zip(df["variants"], df["vcp"] if "vcp" in df.columns else [None] * len(df))
        ]
        if "gwas_traits" not in df.columns:
            df["gwas_traits"] = [[] for _ in range(len(df))]
        else:
            df["gwas_traits"] = df["gwas_traits"].map(as_str_list)
        if "gwas_gws_traits" not in df.columns:
            df["gwas_gws_traits"] = [[] for _ in range(len(df))]
        else:
            df["gwas_gws_traits"] = df["gwas_gws_traits"].map(as_str_list)
        if ROBUST_GENE_ID_COL not in df.columns:
            df[ROBUST_GENE_ID_COL] = [[] for _ in range(len(df))]
        if ROBUST_QTL_TYPE_COL not in df.columns:
            df[ROBUST_QTL_TYPE_COL] = [[] for _ in range(len(df))]
        rows.append(df)
        if not quiet:
            print(f"[{label}] {len(df)} sets (filter={event_filter})")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_loci_table(
    members: pd.DataFrame,
    *,
    locus_prefix: str,
) -> pd.DataFrame:
    """Collapse clustered member rows → loci table."""
    if members.empty or "locus_id" not in members.columns:
        return pd.DataFrame(columns=LOCI_COLS)
    rows = []
    for old_lid, grp in members.groupby(members["locus_id"].astype(str), sort=False):
        rows.append(
            {
                "_old_locus": old_lid,
                "member_cos_ids": sorted(set(grp["cos_id"].astype(str))),
                "gene_ids_robust": _union_lists(grp[ROBUST_GENE_ID_COL])
                if ROBUST_GENE_ID_COL in grp.columns
                else [],
                "gene_types_robust": _union_lists(grp[ROBUST_QTL_TYPE_COL])
                if ROBUST_QTL_TYPE_COL in grp.columns
                else [],
                "gwas_traits": _union_lists(grp["gwas_traits"])
                if "gwas_traits" in grp.columns
                else [],
                "gwas_gws_traits": _union_lists(grp["gwas_gws_traits"])
                if "gwas_gws_traits" in grp.columns
                else [],
                "_n_members": len(grp),
                "_n_tissues": grp["tissue"].nunique()
                if "tissue" in grp.columns
                else len(grp),
            }
        )
    loc = pd.DataFrame(rows)
    if loc.empty:
        return pd.DataFrame(columns=LOCI_COLS)
    loc = loc.sort_values(
        ["_n_tissues", "_n_members"], ascending=[False, False]
    ).reset_index(drop=True)
    loc["locus_id"] = [f"{locus_prefix}_{i}" for i in range(1, len(loc) + 1)]
    return loc[LOCI_COLS]


def cluster_loci(
    *,
    coloc_dir: Path | str,
    output_path: Path | str,
    ld_meta: str,
    event_filter: str,
    locus_prefix: str = "cross_tissue",
    tissues: Optional[List[str]] = None,
    quiet: bool = False,
    pattern: str = "*.ColocBoost.parquet",
) -> None:
    """Cluster CoS across tissues/modes into loci; write one loci parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        coloc_dir=str(coloc_dir),
        output_path=str(output_path),
        ld_meta=ld_meta,
        tissues=tissues,
        event_filter=event_filter,
        quiet=quiet,
        locus_prefix=locus_prefix,
        pattern=pattern,
    )
    rc = run_merge(args)
    if rc:
        raise RuntimeError(f"cluster_loci failed (rc={rc})")


def run_merge(args) -> int:
    """Load ColocBoost tables, cluster per chromosome, write loci parquet."""
    wanted = set(args.tissues) if args.tissues else None
    long_df = load_colocboost_events(
        args.coloc_dir,
        wanted,
        args.quiet,
        event_filter=args.event_filter,
        pattern=getattr(args, "pattern", "*.ColocBoost.parquet"),
    )
    if long_df.empty:
        print("[cluster_loci] No CoS to merge.", file=sys.stderr)
        return 1

    ld_meta = read_ld_meta(args.ld_meta)
    long_df = long_df.copy()
    long_df["lead_chrom"] = long_df["lead_variant_id"].map(lead_chrom)

    long_parts: list[pd.DataFrame] = []
    for chrom, grp in long_df.groupby("lead_chrom", sort=False):
        chrom = str(chrom)
        ld_meta_chr = (
            ld_meta[ld_meta["chrom"] == chrom] if "chrom" in ld_meta.columns else ld_meta
        )
        events = [
            {
                "variants": set(v) if v is not None else set(),
                "lead": ""
                if lead is None or (isinstance(lead, float) and np.isnan(lead))
                else str(lead),
            }
            for v, lead in zip(grp["cos_variants"], grp["lead_variant_id"])
        ]
        clusters = cluster_events(events, chrom=chrom, ld_meta_chr=ld_meta_chr)
        chrom_df = grp.copy()
        chrom_df["locus_id"] = [f"_tmp_chr{chrom}_{cl}" for cl in clusters]
        long_parts.append(chrom_df)
        if not args.quiet:
            print(
                f"[cluster_loci] chr{chrom}: {len(chrom_df)} events, "
                f"{chrom_df['locus_id'].nunique()} raw loci"
            )

    members = pd.concat(long_parts, ignore_index=True)
    loci = build_loci_table(members, locus_prefix=args.locus_prefix)
    loci.to_parquet(args.output_path, index=False)
    print(f"Wrote {args.output_path} ({len(loci):,} loci)")
    return 0
