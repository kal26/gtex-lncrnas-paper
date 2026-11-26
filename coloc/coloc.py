#!/usr/bin/env python3
"""Shared coloc engine: fit ColocBoost per LD-block item, then combine per tissue.

Numbered entrypoints (``01``/``03``/``04``) call ``coloc()``, which shells out to
``coloc_ld_block.R`` then stitches block ``.txt`` tables into per-tissue
``{Tissue}.ColocBoost.parquet`` (cis + weak-GWAS filter). Locus clustering
lives in ``cluster_loci.py`` (``02`` / ``05``).

Paper analysis settings (PIP, GWAS *p*, cis window, joint-GWAS flags) are fixed
here; entrypoints pass data paths and mode-specific extras only.

Block inputs (no ``.focal.txt``)::

  {tissue}.{chrN_start_end}.colocboost.joint_gwas[.robust].txt
  {tissue}.{chrN_start_end}.jNNNNNNN.colocboost.joint_gwas[.robust].txt  (single-pheno; jobs.tsv index)
  {tissue}.{chrN_start_end}.colocboost.xqtl_coloc[.robust].txt

Single-pheno ``job_tag`` is ``j`` + zero-padded 1-based row index in ``jobs.tsv``.
Internal block ``coloc_id`` is ``{tissue}__{region}[__{job_tag}]__{cosN}``; the
public id is ``{Tissue}_{N}`` (or ``{mode}_{N}`` for discovery).

Per-tissue output (full-distance archive under ``tissue_full_dist/``)::

  {Tissue}.ColocBoost.parquet   # one row per CoS; variants+vcp lists
  # discovery: Whole_Blood.{mode}.ColocBoost.parquet
  # ablation:  {Tissue}.{mode}.ColocBoost.parquet

Cross-tissue / cross-phenotype (``cluster_loci``)::

  cross_tissue_loci.parquet / cross_phenotype_set_loci.parquet

After combining: (1) drop QTL genes with no set variant within TSS ± cis window;
(2) drop GWAS with CS-max *p* ≥ 1e-5; (3) reclassify GWAS–QTL CoS→uCoS when one side is
lost. Relaxed gene types matched by variant overlap (dual-assemble renumbers ``cosN``).
"""

from __future__ import annotations

import argparse
import ast
import gzip
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# --- leads helpers (from former leads_colocboost.py) ---

GENOME_WIDE_SIG = -np.log10(5.7e-9)
# Dual-assemble (NPC 0.2 vs 0.75) renumbers cosN; match relaxed CoS by variants.
# 0.75 re-clustering can *grow* a CoS (e.g. gene uCoS at 0.2 becomes gene+GWAS CoS
# at 0.75), so also allow reverse subset and mid Jaccard when outcome tokens overlap.
NONROBUST_JACCARD_MIN = 0.8
NONROBUST_SUBSET_JACCARD_MIN = 0.5
NONROBUST_OUTCOME_JACCARD_MIN = 0.5


def cs_outcome_tokens(cs_id) -> Set[str]:
    """Outcome tokens from a ColocBoost cs_id (``cos13:y18_y29_y134`` → y18, y29, y134)."""
    s = str(cs_id)
    if ":" not in s:
        return set()
    return {t for t in s.split(":", 1)[1].split("_") if t}


def match_nonrobust_by_variants(
    robust_variants: Set[str],
    robust_cs_id: str,
    candidates: List[Tuple[str, Set[str], str]],
    jaccard_min: float = NONROBUST_JACCARD_MIN,
    subset_jaccard_min: float = NONROBUST_SUBSET_JACCARD_MIN,
    outcome_jaccard_min: float = NONROBUST_OUTCOME_JACCARD_MIN,
) -> Optional[str]:
    """Match a robust CoS to a standard/non-robust CoS by variant overlap.

    Do not use ``cosN``: the two assemble passes number CoS independently, so
    ``cos13`` in robust is not ``cos13`` in the NPC-0.2 table. Preference order:
    exact variant set, then either direction of variant-set subset with Jaccard
    >= ``subset_jaccard_min``, then Jaccard >= ``outcome_jaccard_min`` with a
    shared outcome token (``y16`` etc.), then Jaccard >= ``jaccard_min``.
    Ties: more outcomes, then outcome-set superset.
    Each candidate is ``(coloc_id, variant_set, cs_id)``. None if nothing qualifies.
    """
    if not robust_variants or not candidates:
        return None
    rob_out = cs_outcome_tokens(robust_cs_id)
    exact: List[Tuple[float, int, int, str]] = []
    subset: List[Tuple[float, int, int, str]] = []
    outcome_overlap: List[Tuple[float, int, int, str]] = []
    scored: List[Tuple[float, int, int, str]] = []
    for cid, varset, cs_id in candidates:
        if not varset:
            continue
        inter = len(robust_variants & varset)
        if inter == 0:
            continue
        union = len(robust_variants | varset)
        jacc = inter / union if union else 0.0
        cand_out = cs_outcome_tokens(cs_id)
        n_out = len(cand_out)
        y_super = int(bool(rob_out) and rob_out <= cand_out)
        key = (jacc, n_out, y_super, str(cid))
        if varset == robust_variants:
            exact.append(key)
        elif (
            (robust_variants <= varset or varset <= robust_variants)
            and jacc >= subset_jaccard_min
        ):
            subset.append(key)
        if jacc >= outcome_jaccard_min and rob_out and (rob_out & cand_out):
            outcome_overlap.append(key)
        if jacc >= jaccard_min:
            scored.append(key)

    def _pick(cands: List[Tuple[float, int, int, str]]) -> str:
        cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return cands[0][3]

    if exact:
        return _pick(exact)
    if subset:
        return _pick(subset)
    if outcome_overlap:
        return _pick(outcome_overlap)
    if scored:
        return _pick(scored)
    return None


def _variant_sets_by_coloc_id(df: pd.DataFrame) -> Dict[str, Set[str]]:
    if df is None or df.empty or "coloc_id" not in df.columns or "variant_id" not in df.columns:
        return {}
    return (
        df.groupby(df["coloc_id"].astype(str))["variant_id"]
        .apply(lambda s: {str(v) for v in s if v is not None and str(v) not in ("", "nan")})
        .to_dict()
    )


def unstringify_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode object cells that are Python literal strings (e.g. after parquet round-trip)."""
    out = df.copy()

    def _cell(x):
        if x is None:
            return x
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return x
        if isinstance(x, (list, tuple, dict, set, np.ndarray)):
            return x
        if not isinstance(x, str):
            try:
                if pd.isna(x):
                    return x
            except (TypeError, ValueError):
                pass
            return x
        s = x.strip()
        if not s or s in ("nan", "None", "<NA>"):
            return x
        if s[0] not in "[({":
            return x
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return x

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_cell)
    return out


def _split_phenotype_field(val):
    if isinstance(val, list):
        return val
    return str(val).split(";")


def _split_pval_field(val):
    if isinstance(val, list):
        return val
    return re.split(r",\s*", str(val))


def _parse_pval_nominal(val) -> List[float] | None:
    """Parse a ``pval_nominal_phenotypes`` cell into floats (NaN for bad entries).

    Cells are comma-separated -log10(nominal p) values aligned with ``phenotype_ids``
    order (multi-outcome CoS) or a single value (uCoS). Returns None if empty/missing.
    """
    if isinstance(val, (list, tuple, np.ndarray)):
        seq = list(val)
    else:
        s = str(val).strip()
        if not s or s in ("nan", "None", "<NA>"):
            return None
        seq = re.split(r",\s*", s)
    out: List[float] = []
    for x in seq:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def max_pval_nominal_phenotypes(robust_cos: pd.DataFrame) -> pd.Series:
    """Per coloc_id, elementwise max ``pval_nominal_phenotypes`` across CoS variants.

    Each robust row is one variant; ``pval_nominal_phenotypes`` holds per-phenotype
    -log10(nominal p) at that variant (order matching ``phenotype_ids``). The CoS-level
    value is the max over variants. Returns a Series indexed by ``coloc_id`` of
    comma-separated strings aligned with ``phenotype_ids``.
    """
    results: Dict[str, str] = {}
    for coloc_id, grp in robust_cos.groupby("coloc_id"):
        parsed = [
            p
            for p in (_parse_pval_nominal(v) for v in grp["pval_nominal_phenotypes"])
            if p is not None
        ]
        if not parsed:
            results[coloc_id] = ""
            continue
        max_len = max(len(p) for p in parsed)
        mat = np.full((len(parsed), max_len), np.nan)
        for i, p in enumerate(parsed):
            mat[i, : len(p)] = p
        all_nan_cols = np.all(np.isnan(mat), axis=0)
        max_vals = np.full(max_len, np.nan)
        if not np.all(all_nan_cols):
            max_vals[~all_nan_cols] = np.nanmax(mat[:, ~all_nan_cols], axis=0)
        results[coloc_id] = ", ".join("nan" if np.isnan(m) else f"{m:.6g}" for m in max_vals)
    return pd.Series(results)


def gwas_gws_traits_from_cs_max(
    phenotype_id,
    cs_max_pval_nominal,
    gws_cut: float = GENOME_WIDE_SIG,
) -> List[str]:
    """GWAS trait names with CoS-max -log10(p) above the genome-wide cut.

    Uses the max over *any* variant in the CoS (``cos_max_pval_nominal_phenotypes``),
    not the lead alone. Phenotype order matches ``phenotype_ids`` (``;``-separated).
    Skips eQTL / sQTL phenotype parts (names containing expression / splicing).
    """
    if phenotype_id is None:
        return []
    try:
        if isinstance(phenotype_id, float) and np.isnan(phenotype_id):
            return []
    except (TypeError, ValueError):
        pass
    parts = [p.strip() for p in str(phenotype_id).split(";") if p.strip()]
    parsed = _parse_pval_nominal(cs_max_pval_nominal)
    if not parsed:
        return []
    out: List[str] = []
    for i, part in enumerate(parts):
        if i >= len(parsed):
            break
        pl = part.lower()
        if "expression" in pl or "splicing" in pl:
            continue
        val = parsed[i]
        if val is not None and np.isfinite(val) and float(val) > gws_cut:
            out.append(part)
    return out


def write_leads_parquet(df: pd.DataFrame, path: str) -> None:
    """Write parquet; normalize object columns for pyarrow.

    List/array columns (e.g. gene_ids) stay as lists. Scalar object columns that
    mix str/float/int/NaN across block tables (e.g. npc_outcomes) are coerced
    to string so pyarrow does not fail with ArrowTypeError.
    """
    out = df.copy()

    def _cell(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (set, tuple)):
            return list(x)
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            return x
        if x is None:
            return None
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return None
        try:
            if pd.isna(x):
                return None
        except (TypeError, ValueError):
            pass
        # Remaining scalars (int/float mixed into object cols) → str for parquet.
        return str(x)

    for col in out.columns:
        if out[col].dtype != object:
            continue
        sample = out[col].dropna().head(50)
        if len(sample) and any(isinstance(v, (list, dict, np.ndarray)) for v in sample):
            out[col] = out[col].map(_cell)
            continue
        na_mask = out[col].isna()
        coerced = out[col].map(_cell)
        # Empty string for NA (same convention as write_parquet below).
        if na_mask.any():
            coerced = coerced.mask(na_mask, None)
        out[col] = coerced
    out.to_parquet(path, index=False)


def _load_gencode(gencode_path: str) -> dict:
    gencode_genes = pd.read_table(gencode_path, low_memory=False)
    gencode_genes = gencode_genes[gencode_genes["gene_type_grouped"] != "artifact"]
    gene_id_to_type: dict = {}
    for _, r in gencode_genes.iterrows():
        gid = str(r["gene_id_stable"])
        gt = r["gene_type_grouped"]
        gene_id_to_type[gid] = gt
        stem = gid.split(".")[0]
        gene_id_to_type.setdefault(stem, gt)
    return gene_id_to_type


ENSG_RE = re.compile(r"(ENSG\d+(?:\.\d+)?)")
# Tissue.chrN_start_end[.jN][.all_genes|.xqtl_only].colocboost.(joint_gwas|xqtl_coloc)[.robust].txt
# Single-pheno disk tags are j + 1-based jobs.tsv row index (see format_job_tag).
BLOCK_TXT_RE = re.compile(
    r"^(?P<tissue>.+)\.(?P<region>chr[^_]+_\d+_\d+)"
    r"(?:\.(?P<tag>j\d+))?"
    r"\.(?:(?:all_genes|xqtl_only)\.)?colocboost\.(?P<mode>joint_gwas|xqtl_coloc)"
    r"(?P<robust>\.robust)?\.txt$"
)

JOB_TAG_WIDTH = 7


def format_job_tag(job_index: int) -> str:
    """1-based ``jobs.tsv`` row index → disk / ``coloc_id`` tag ``j0000001``."""
    idx = int(job_index)
    if idx < 1:
        raise ValueError(f"job_index must be >= 1, got {job_index!r}")
    return f"j{idx:0{JOB_TAG_WIDTH}d}"

# Keep GWAS traits with CS-max -log10(p) strictly above this (p < 1e-5).
GWAS_SUB_SIG_NEGLOG10 = float(-np.log10(1e-5))  # 5.0

_GTF_ATTR_RE = re.compile(r'(\w+)\s+"([^"]+)"')


def _parse_collapsed_gtf_arg(raw: str) -> List[Path]:
    """Split comma-separated ``--collapsed_gtf`` paths (order preserved; later override)."""
    return [Path(p.strip()) for p in str(raw).split(",") if p.strip()]


def load_tss_map_from_gtf(gtf_paths: Iterable[Path]) -> Dict[str, Tuple[str, int]]:
    """Map gene_id / gene stem → (chrom, tss0) from GENCODE ``gene`` features.

    TSS is the strand-aware gene-boundary transcription start in **0-based**
    coordinates (``start-1`` on ``+``, ``end-1`` on ``-``), matching GTEx
    expression BED 1 bp intervals. When multiple GTFs are given, later files
    overwrite earlier entries for the same stem / versioned id.
    """
    out: Dict[str, Tuple[str, int]] = {}
    paths = list(gtf_paths)
    if not paths:
        raise ValueError("load_tss_map_from_gtf requires at least one GTF path")
    for gtf_path in paths:
        opener = gzip.open if str(gtf_path).endswith(".gz") else open
        n_genes = 0
        with opener(gtf_path, "rt") as fh:
            for line in fh:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9 or fields[2] != "gene":
                    continue
                chrom, start1, end1, strand = (
                    fields[0],
                    int(fields[3]),
                    int(fields[4]),
                    fields[6],
                )
                attrs = dict(_GTF_ATTR_RE.findall(fields[8]))
                gid = attrs.get("gene_id", "")
                if not gid:
                    continue
                if strand == "+":
                    tss0 = start1 - 1
                elif strand == "-":
                    tss0 = end1 - 1
                else:
                    raise ValueError(
                        f"Unexpected strand {strand!r} for {gid} in {gtf_path}"
                    )
                out[gid] = (chrom, tss0)
                out[gid.split(".")[0]] = (chrom, tss0)
                n_genes += 1
        print(f"[combine] TSS GTF {gtf_path}: {n_genes:,} gene features")
    return out


def parse_variant_pos(variant_id: str) -> Tuple[str, int]:
    s = str(variant_id)
    if ":" in s:
        parts = s.split(":")
        return parts[0], int(parts[1])
    parts = s.split("_")
    return parts[0], int(parts[1])


def chrom_equal(a: str, b: str) -> bool:
    return str(a).replace("chr", "") == str(b).replace("chr", "")


def qtl_gene_ids_from_phenotype(val) -> List[str]:
    """ENSG ids from expression/splicing phenotype parts only."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    genes: List[str] = []
    for part in str(val).split(";"):
        part = part.strip()
        if not part:
            continue
        pl = part.lower()
        if "expression" not in pl and "splicing" not in pl:
            continue
        genes.extend(ENSG_RE.findall(part))
    return genes


def resolve_tss(
    gene_id: str,
    tss_map: Dict[str, Tuple[str, int]],
) -> Tuple[str, int]:
    """Return (chrom, tss0) for ``gene_id``; raise if the gene is absent from ``tss_map``."""
    key = gene_id if gene_id in tss_map else str(gene_id).split(".")[0]
    if key not in tss_map:
        raise KeyError(
            f"No TSS for gene {gene_id!r} (stem {key!r}) in TSS map — "
            "refusing silent skip; extend --collapsed_gtf coverage"
        )
    return tss_map[key]


def gene_lacks_cis_cos_variant(
    gene_id: str,
    variant_ids: Iterable[str],
    tss_map: Dict[str, Tuple[str, int]],
    window_bp: int,
) -> bool:
    """True if no CoS variant lies within TSS±window (same chrom).

    A gene is kept when **at least one** CoS variant is within the window, even if
    other CoS variants are farther away. Missing TSS is a hard error.
    """
    chrom, tss = resolve_tss(gene_id, tss_map)
    for vid in variant_ids:
        vchrom, vpos = parse_variant_pos(vid)
        if chrom_equal(vchrom, chrom) and abs(vpos - tss) <= window_bp:
            return False
    return True


def gene_stems_outside_cis(
    gene_ids: Iterable[str],
    variant_ids: Iterable[str],
    tss_map: Dict[str, Tuple[str, int]],
    window_bp: int,
) -> Set[str]:
    """ENSG stems for QTL genes with no CoS variant within TSS±window.

    Raises ``KeyError`` if any gene lacks a TSS (no silent keep).
    """
    outside: Set[str] = set()
    missing: List[str] = []
    variants = list(variant_ids)
    for g in gene_ids:
        g_str = str(g)
        try:
            if gene_lacks_cis_cos_variant(g_str, variants, tss_map, window_bp):
                outside.add(g_str.split(".")[0])
        except KeyError:
            missing.append(g_str)
    if missing:
        uniq = sorted(set(missing))
        preview = ", ".join(uniq[:20])
        more = f" (+{len(uniq) - 20} more)" if len(uniq) > 20 else ""
        raise KeyError(
            f"No TSS for {len(uniq)} QTL gene(s) in cis trim: {preview}{more}. "
            "Refusing silent skip; extend --collapsed_gtf coverage."
        )
    return outside


def _pheno_gene_stem(pheno: str) -> Optional[str]:
    m = ENSG_RE.search(pheno)
    return m.group(1).split(".")[0] if m else None


def _is_qtl_pheno(pheno: str) -> bool:
    pl = pheno.lower()
    return "expression" in pl or "splicing" in pl


def _parse_aligned_neglog10(raw) -> List[Optional[float]]:
    """Parse a comma-separated -log10(p) string into floats (nan → None)."""
    if raw is None:
        return []
    try:
        if isinstance(raw, float) and np.isnan(raw):
            return []
    except (TypeError, ValueError):
        pass
    out: List[Optional[float]] = []
    for tok in re.split(r",\s*", str(raw).strip()):
        tok = tok.strip()
        if not tok or tok.lower() == "nan":
            out.append(None)
            continue
        try:
            val = float(tok)
        except ValueError:
            out.append(None)
            continue
        out.append(val if np.isfinite(val) else None)
    return out


def _min_npc_from_str(s) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    vals = []
    for x in str(s).split(";"):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(float(x))
        except ValueError:
            pass
    return float(min(vals)) if vals else np.nan


def phenotype_keep_mask(
    phenos: List[str],
    *,
    drop_qtl_stems: Set[str],
    cs_max_neglog10,
    gwas_min_neglog10: float = GWAS_SUB_SIG_NEGLOG10,
) -> List[bool]:
    """Per-phenotype keep flags: drop far QTLs and weak GWAS (CS-max ≤ cut)."""
    cs_vals = _parse_aligned_neglog10(cs_max_neglog10)
    keep: List[bool] = []
    for i, p in enumerate(phenos):
        if _is_qtl_pheno(p):
            stem = _pheno_gene_stem(p)
            keep.append(not (stem is not None and stem in drop_qtl_stems))
        else:
            val = cs_vals[i] if i < len(cs_vals) else None
            keep.append(val is not None and float(val) > gwas_min_neglog10)
    return keep


def apply_phenotype_keep_mask(
    phenotype_id,
    keep_mask: List[bool],
    npc_outcomes=None,
    pval_nominal_phenotypes=None,
) -> Tuple[str, Optional[str], Optional[str], bool, bool]:
    """Filter parallel phenotype fields by ``keep_mask``.

    Returns (new_phenotype_id, new_npc_outcomes, new_pval_nominal, has_qtl, has_gwas).
    """
    if phenotype_id is None or (isinstance(phenotype_id, float) and np.isnan(phenotype_id)):
        return "", None, None, False, False
    phenos = [p.strip() for p in str(phenotype_id).split(";") if p.strip()]
    if len(keep_mask) != len(phenos):
        raise ValueError(
            f"keep_mask length {len(keep_mask)} != n phenotypes {len(phenos)}"
        )

    kept_phenos = [p for p, k in zip(phenos, keep_mask) if k]
    new_pid = ";".join(kept_phenos)
    has_qtl = any(_is_qtl_pheno(p) for p in kept_phenos)
    has_gwas = any(not _is_qtl_pheno(p) for p in kept_phenos)

    new_npc: Optional[str] = None
    if npc_outcomes is not None and not (
        isinstance(npc_outcomes, float) and np.isnan(npc_outcomes)
    ):
        npc_parts = [x.strip() for x in str(npc_outcomes).split(";")]
        if len(npc_parts) == len(phenos):
            new_npc = ";".join(x for x, k in zip(npc_parts, keep_mask) if k)
        else:
            new_npc = str(npc_outcomes)

    new_pval: Optional[str] = None
    if pval_nominal_phenotypes is not None and not (
        isinstance(pval_nominal_phenotypes, float) and np.isnan(pval_nominal_phenotypes)
    ):
        pval_parts = re.split(r",\s*", str(pval_nominal_phenotypes).strip())
        if len(pval_parts) == len(phenos):
            new_pval = ", ".join(x.strip() for x, k in zip(pval_parts, keep_mask) if k)
        else:
            new_pval = str(pval_nominal_phenotypes)

    return new_pid, new_npc, new_pval, has_qtl, has_gwas


def cs_id_to_ucos(cs_id: str) -> str:
    """Rename a CoS id to uCoS (``cos1`` → ``ucos1``); leave uCoS unchanged."""
    s = public_cos_id(cs_id)
    if s.lower().startswith("ucos"):
        return s
    if s.lower().startswith("cos"):
        return "u" + s
    return s


def public_cos_id(cs_id) -> str:
    """Strip ColocBoost ``:y…`` outcome tokens; keep ``cosN`` / ``ucosN``."""
    s = str(cs_id).strip()
    if not s or s in ("nan", "None", "<NA>"):
        return ""
    return s.split(":", 1)[0]


def replace_cs_id_in_coloc_id(coloc_id: str, old_cs: str, new_cs: str) -> str:
    """Update trailing ``__cos_id`` after a cos_id rename (CoS→uCoS)."""
    cid = str(coloc_id)
    old = public_cos_id(old_cs)
    new = public_cos_id(new_cs)
    suffix = "__" + old
    if cid.endswith(suffix):
        return cid[: -len(suffix)] + "__" + new
    return cid.replace("__" + old, "__" + new, 1)


def make_coloc_id(
    tissue_id: str,
    region_id: str,
    cs_id,
    *,
    job_tag: Optional[str] = None,
) -> str:
    """Build ``coloc_id``: ``{tissue}__{region}[__{job_tag}]__{cosN}``."""
    parts = [str(tissue_id), str(region_id)]
    if job_tag:
        parts.append(str(job_tag))
    cos = public_cos_id(cs_id)
    if cos:
        parts.append(cos)
    return "__".join(parts)


def _phenotype_sides(phenotype_id) -> Tuple[bool, bool]:
    """Return (has_gwas, has_qtl) from a phenotype_id string."""
    if phenotype_id is None or (isinstance(phenotype_id, float) and np.isnan(phenotype_id)):
        return False, False
    has_gwas = False
    has_qtl = False
    for part in str(phenotype_id).split(";"):
        part = part.strip()
        if not part:
            continue
        if _is_qtl_pheno(part):
            has_qtl = True
        else:
            has_gwas = True
    return has_gwas, has_qtl


def filter_sets_posthoc_trim(
    df: pd.DataFrame,
    tss_map: Dict[str, Tuple[str, int]],
    window_bp: int,
    gwas_min_neglog10: float = GWAS_SUB_SIG_NEGLOG10,
) -> Tuple[pd.DataFrame, dict]:
    """Strip far QTL genes + weak GWAS; reclassify GWAS–QTL CoS→uCoS when one side remains.

    - QTL gene kept if ≥1 set variant within TSS±window.
    - GWAS trait kept if CS-max -log10(p) > ``gwas_min_neglog10`` (default: p < 1e-5).
    - Former GWAS–QTL CoS that lose all QTL or all passing GWAS become uCoS.
    - Sets with no phenotypes left are dropped.
    """
    empty_stats = {
        "n_cos_in": 0,
        "n_cos_out": 0,
        "n_cos_unchanged": 0,
        "n_cos_partial": 0,
        "n_cos_dropped_empty": 0,
        "n_cos_reclassified_ucos": 0,
        "n_gene_cos_pairs": 0,
        "n_gene_cos_outside": 0,
        "n_weak_gwas_stripped": 0,
    }
    if df is None or df.empty or "coloc_id" not in df.columns:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame(), empty_stats

    work = df.copy()
    work["_cid"] = work["coloc_id"].astype(str)
    vars_by = (
        work.groupby("_cid", sort=False)["variant_id"]
        .apply(lambda s: list(dict.fromkeys(s.astype(str).tolist())))
        .to_dict()
    )
    if "phenotype_ids" not in work.columns:
        raise KeyError("expected phenotype_ids on CoS table")
    pid_by = work.groupby("_cid", sort=False)["phenotype_ids"].first().to_dict()
    cs_by = (
        work.groupby("_cid", sort=False)["cos_id"].first().astype(str).map(public_cos_id).to_dict()
        if "cos_id" in work.columns
        else {cid: "" for cid in pid_by}
    )
    has_npc = "npc_outcomes" in work.columns
    has_pval = "pval_nominal_phenotypes" in work.columns
    npc_by = (
        work.groupby("_cid", sort=False)["npc_outcomes"].first().to_dict()
        if has_npc
        else {}
    )
    # CoS-max -log10(p) across variants (aligned to phenotype_ids), for weak-GWAS strip.
    csmax_by: Dict[str, str] = {}
    if has_pval:
        csmax_by = {
            str(k): v for k, v in max_pval_nominal_phenotypes(work).items()
        }

    n_gene_pairs = 0
    n_outside_pairs = 0
    n_weak_gwas = 0
    # old_cid → (pid2, npc2, keep_mask, new_cs_id or None)
    rewrite: Dict[str, Tuple[str, Optional[str], List[bool], Optional[str]]] = {}
    keep_cids: Set[str] = set()
    n_unchanged = n_partial = n_dropped = n_reclass = 0

    for cid, pid in pid_by.items():
        genes = list(dict.fromkeys(qtl_gene_ids_from_phenotype(pid)))
        n_gene_pairs += len(genes)
        outside = gene_stems_outside_cis(
            genes, vars_by.get(cid, []), tss_map, window_bp
        )
        n_outside_pairs += len(outside)

        phenos = (
            []
            if pid is None or (isinstance(pid, float) and np.isnan(pid))
            else [p.strip() for p in str(pid).split(";") if p.strip()]
        )
        has_gwas0, has_qtl0 = _phenotype_sides(pid)
        keep = phenotype_keep_mask(
            phenos,
            drop_qtl_stems=outside,
            cs_max_neglog10=csmax_by.get(cid),
            gwas_min_neglog10=gwas_min_neglog10,
        )
        n_weak_gwas += sum(
            1 for p, k in zip(phenos, keep) if (not _is_qtl_pheno(p)) and (not k)
        )

        if phenos and all(keep):
            keep_cids.add(cid)
            n_unchanged += 1
            continue

        if not phenos or not any(keep):
            n_dropped += 1
            continue

        pid2, npc2, _nlp_unused, has_qtl1, has_gwas1 = apply_phenotype_keep_mask(
            pid,
            keep,
            npc_by.get(cid) if has_npc else None,
            None,
        )
        old_cs = cs_by.get(cid, "")
        new_cs: Optional[str] = None
        is_cos = str(old_cs).lower().startswith("cos") and not str(old_cs).lower().startswith(
            "ucos"
        )
        if (
            is_cos
            and has_gwas0
            and has_qtl0
            and (has_gwas1 or has_qtl1)
            and not (has_gwas1 and has_qtl1)
        ):
            new_cs = cs_id_to_ucos(old_cs)
            n_reclass += 1

        rewrite[cid] = (pid2, npc2, keep, new_cs)
        keep_cids.add(cid)
        n_partial += 1

    out = work[work["_cid"].isin(keep_cids)].copy()
    if not out.empty and rewrite:
        for old_cid, (pid2, npc2, keep, new_cs) in rewrite.items():
            mask = out["_cid"] == old_cid
            out.loc[mask, pheno_col] = pid2
            if has_npc and npc2 is not None:
                out.loc[mask, "npc_outcomes"] = npc2
            if has_pval:

                def _strip_pval(raw, _keep=keep):
                    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                        return raw
                    parts = re.split(r",\s*", str(raw).strip())
                    if len(parts) != len(_keep):
                        return raw
                    return ", ".join(x.strip() for x, k in zip(parts, _keep) if k)

                out.loc[mask, "pval_nominal_phenotypes"] = out.loc[
                    mask, "pval_nominal_phenotypes"
                ].map(_strip_pval)
            if new_cs is not None and "cos_id" in out.columns:
                old_cs = cs_by.get(old_cid, "")
                out.loc[mask, "cos_id"] = public_cos_id(new_cs)
                out.loc[mask, "coloc_id"] = [
                    replace_cs_id_in_coloc_id(c, old_cs, new_cs)
                    for c in out.loc[mask, "coloc_id"].astype(str)
                ]
            # npc_outcome (min of npc_outcomes) is not stored; derive if needed.

    out = out.drop(columns=["_cid"])
    stats = {
        "n_cos_in": len(pid_by),
        "n_cos_out": int(out["coloc_id"].nunique()) if not out.empty else 0,
        "n_cos_unchanged": n_unchanged,
        "n_cos_partial": n_partial,
        "n_cos_dropped_empty": n_dropped,
        "n_cos_reclassified_ucos": n_reclass,
        "n_gene_cos_pairs": n_gene_pairs,
        "n_gene_cos_outside": n_outside_pairs,
        "n_weak_gwas_stripped": n_weak_gwas,
    }
    return out.reset_index(drop=True), stats


def parse_block_filename(name: str) -> Optional[Tuple[str, str, str, bool, Optional[str]]]:
    """Return (tissue, region_id, mode, is_robust, job_tag) or None."""
    if ".focal." in name:
        return None
    m = BLOCK_TXT_RE.match(name)
    if m is None:
        return None
    return (
        m.group("tissue"),
        m.group("region"),
        m.group("mode"),
        bool(m.group("robust")),
        m.group("tag"),
    )


_BLOCK_SUFFIXES = (
    "colocboost.joint_gwas.txt",
    "colocboost.joint_gwas.robust.txt",
    "colocboost.xqtl_coloc.txt",
    "colocboost.xqtl_coloc.robust.txt",
)


def _list_block_files_one_chrom(
    coloc_dir: Path, tissue_id: str, chrom: str
) -> Tuple[str, List[Tuple[Path, str, str, bool, Optional[str]]]]:
    """Glob+stat one chromosome's block tables."""
    rows: List[Tuple[Path, str, str, bool, Optional[str]]] = []
    for suf in _BLOCK_SUFFIXES:
        # Untagged + .jN tags match chrN_*.
        pattern = f"{tissue_id}.{chrom}_*.{suf}"
        for path in coloc_dir.glob(pattern):
            parsed = parse_block_filename(path.name)
            if parsed is None:
                continue
            tissue, region, mode, is_robust, tag = parsed
            if tissue != tissue_id:
                continue
            try:
                # Skip header-only placeholders (~89B) to avoid useless NFS opens later.
                if path.stat().st_size < 100:
                    continue
            except OSError:
                continue
            rows.append((path, region, mode, is_robust, tag))
    return chrom, rows


def list_block_files(
    coloc_dir: Path, tissue_id: str
) -> List[Tuple[Path, str, str, bool, Optional[str]]]:
    """List block .txt tables via per-chromosome globs (avoids full-dir NFS scandir)."""
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
    print(f"[combine] {tissue_id}: listing block tables by chrom", flush=True)
    out: List[Tuple[Path, str, str, bool, Optional[str]]] = []
    for chrom in chroms:
        chrom, rows = _list_block_files_one_chrom(coloc_dir, tissue_id, chrom)
        if rows:
            print(
                f"[combine] {tissue_id}/{chrom}: {len(rows):,} nonempty block tables",
                flush=True,
            )
            out.extend(rows)
    print(
        f"[combine] {tissue_id}: listed {len(out):,} nonempty block tables total",
        flush=True,
    )
    return out


def read_block_table(
    path: Path,
    region_id: str,
    mode: str,
    *,
    tissue_id: str,
    job_tag: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, sep="\t")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, ValueError) as e:
        print(f"[combine] skip {path.name}: {e}", file=sys.stderr)
        return None
    if df.empty:
        return None
    df["region_id"] = region_id
    # ColocBoost writes cs_id; rename to cos_id for CoS (colocalization set).
    if "cs_id" in df.columns and "cos_id" not in df.columns:
        df = df.rename(columns={"cs_id": "cos_id"})
    # Drop ColocBoost :y… tokens from the public cos_id.
    if "cos_id" in df.columns:
        df["cos_id"] = df["cos_id"].map(public_cos_id)
    # ColocBoost writes outcomes_npc_str / npc_outcome; keep only npc_outcomes.
    if "outcomes_npc_str" in df.columns and "npc_outcomes" not in df.columns:
        df = df.rename(columns={"outcomes_npc_str": "npc_outcomes"})
    if "npc_outcome" in df.columns:
        df = df.drop(columns=["npc_outcome"])
    # ColocBoost writes neg_log10_p_value (-log10 nominal p, phenotype-aligned).
    if "neg_log10_p_value" in df.columns and "pval_nominal_phenotypes" not in df.columns:
        df = df.rename(columns={"neg_log10_p_value": "pval_nominal_phenotypes"})
    elif "neg_log10_p_value" in df.columns:
        df = df.drop(columns=["neg_log10_p_value"])
    # ColocBoost tables use phenotype_id; write phenotype_ids.
    if "phenotype_id" in df.columns and "phenotype_ids" not in df.columns:
        df = df.rename(columns={"phenotype_id": "phenotype_ids"})
    df["coloc_id"] = [
        make_coloc_id(tissue_id, region_id, cos, job_tag=job_tag)
        for cos in df["cos_id"].astype(str)
    ]
    # Mode lives in the filename ({tissue}.{mode}.all…); do not store coloc_mode.
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Object columns mix str / float / int / NaN across block TSVs (e.g. npc_outcomes).
    # Coerce those to string for pyarrow; leave list/dict columns (leads gene_ids, etc.).
    out = df
    for col in out.select_dtypes(include=["object"]).columns:
        sample = out[col].dropna().head(50)
        if len(sample) and any(isinstance(v, (list, dict, np.ndarray)) for v in sample):
            continue
        if out is df:
            out = df.copy()
        na_mask = out[col].isna()
        coerced = out[col].astype(str)
        if na_mask.any():
            coerced = coerced.mask(na_mask, "")
        out[col] = coerced
    out.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")


def _as_plain_list(x) -> List:
    """Convert pandas/numpy/Arrow unique() results to a plain Python list."""
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, np.ndarray):
        return [v for v in x.tolist()]
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if isinstance(x, pd.Series):
        return x.tolist()
    # ArrowStringArray and similar array-like objects
    try:
        return [v for v in list(x)]
    except TypeError:
        return [x]


def _typeset(x) -> Set[str]:
    out: Set[str] = set()
    for v in _as_plain_list(x):
        if v is None:
            continue
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            continue
        s = str(v).strip()
        if not s or s in ("nan", "None", "<NA>"):
            continue
        out.add(s)
    return out


def _specificity(types: Set[str]) -> Tuple[bool, bool]:
    """Coloc CS specificity for gene_type_grouped.

    - lncRNA-specific: only lncRNAs (new and/or existing); non-lncRNAs disqualify
    - new-specific: only novel lncRNAs (existing lncRNA also disqualifies)
    Mixed new+existing with no non-lncRNA → lncRNA-specific but not new-specific.
    """
    has_new = "novel_lncRNA" in types
    has_existing = "existing_lncRNA" in types
    has_any_lncrna = has_new or has_existing
    non_lncrna = types - {"novel_lncRNA", "existing_lncRNA"}
    disqualifying_non_lncrna = bool(non_lncrna)

    is_lncrna_specific = has_any_lncrna and not disqualifying_non_lncrna
    is_new_specific = has_new and not has_existing and not disqualifying_non_lncrna
    return is_new_specific, is_lncrna_specific


def _explode_phenotype_rows(df: pd.DataFrame, gene_id_to_type: dict) -> pd.DataFrame:
    """One row per phenotype part of each input row (lead or CoS)."""
    if "phenotype_ids" not in df.columns:
        raise KeyError("expected phenotype_ids on CoS table")
    work = df[["coloc_id", "phenotype_ids"]].copy()
    # During combine, per-variant rows still use pval_nominal_phenotypes;
    # leads already have cos_max_pval_nominal_phenotypes.
    if "pval_nominal_phenotypes" in df.columns:
        work["pval_nominal_phenotypes"] = df["pval_nominal_phenotypes"].values
    elif "cos_max_pval_nominal_phenotypes" in df.columns:
        work["pval_nominal_phenotypes"] = df["cos_max_pval_nominal_phenotypes"].values
    else:
        work["pval_nominal_phenotypes"] = np.nan
    work["individual_phenotype_id"] = work["phenotype_ids"].map(_split_phenotype_field)
    work["individual_pval"] = work["pval_nominal_phenotypes"].map(_split_pval_field)
    exploded = work.explode(
        ["individual_phenotype_id", "individual_pval"], ignore_index=False
    )
    exploded["individual_pval"] = pd.to_numeric(exploded["individual_pval"], errors="coerce")
    ip = exploded["individual_phenotype_id"].astype(str)
    exploded["phenotype_type"] = np.where(
        ip.str.contains("expression", na=False),
        "eqtl",
        np.where(ip.str.contains("splicing", na=False), "sqtl", "gwas"),
    )
    exploded["individual_gene_id"] = ip.str.extract(r"(ENSG\d+(?:\.\d+)?)", expand=False)
    stem = exploded["individual_gene_id"].str.split(".").str[0]
    exploded["gene_type"] = stem.map(gene_id_to_type)
    # QTL phenotypes with an ENSG must resolve in the classification table.
    qtl = exploded["phenotype_type"].isin(["eqtl", "sqtl"])
    has_gene = exploded["individual_gene_id"].notna()
    untyped = qtl & has_gene & exploded["gene_type"].isna()
    if untyped.any():
        missing = sorted({str(s) for s in stem.loc[untyped].dropna().unique()})
        preview = ", ".join(missing[:20])
        more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        raise KeyError(
            f"No gene_type for {len(missing)} QTL gene stem(s) in gencode "
            f"classification: {preview}{more}. Refusing silent NaN gene_type."
        )
    return exploded


def _nonrobust_annotation_maps(
    nonrobust_df: pd.DataFrame, gene_id_to_type: dict
) -> Tuple[Set[str], Dict[str, List], Dict[str, List], Dict[str, List[Tuple[str, Set[str], str]]]]:
    """Build coloc_id → gene types/ids and per-region variant-set candidates."""
    if nonrobust_df is None or nonrobust_df.empty:
        return set(), {}, {}, {}
    nr = nonrobust_df.copy()
    if "vcp" in nr.columns:
        nr["vcp"] = pd.to_numeric(nr["vcp"], errors="coerce")
        nr = nr.sort_values("vcp", ascending=False)
    nr_leads = nr.drop_duplicates(subset=["coloc_id"], keep="first")
    nr_cids = set(nr_leads["coloc_id"].astype(str))
    exploded = _explode_phenotype_rows(nr_leads, gene_id_to_type)
    types_by = (
        exploded[exploded["gene_type"].notna()]
        .groupby("coloc_id")["gene_type"]
        .agg(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )
    ids_by = (
        exploded[exploded["individual_gene_id"].notna()]
        .groupby("coloc_id")["individual_gene_id"]
        .agg(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )
    types_by = {str(k): v for k, v in types_by.items()}
    ids_by = {str(k): v for k, v in ids_by.items()}
    var_by_cid = _variant_sets_by_coloc_id(nr)
    cs_by_cid = nr_leads.set_index(nr_leads["coloc_id"].astype(str))["cos_id"].astype(str).to_dict()
    region_by_cid = (
        nr_leads.set_index(nr_leads["coloc_id"].astype(str))["region_id"].astype(str).to_dict()
        if "region_id" in nr_leads.columns
        else {}
    )
    nr_by_region: Dict[str, List[Tuple[str, Set[str], str]]] = defaultdict(list)
    for cid, varset in var_by_cid.items():
        nr_by_region[str(region_by_cid.get(cid, ""))].append(
            (str(cid), varset, str(cs_by_cid.get(cid, "")))
        )
    return nr_cids, types_by, ids_by, dict(nr_by_region)


def _match_nonrobust_cid(
    row: pd.Series,
    rob_vars: Dict[str, Set[str]],
    nr_cids: Set[str],
    nr_by_region: Dict[str, List[Tuple[str, Set[str], str]]],
) -> Optional[str]:
    cid = str(row["coloc_id"])
    if cid in nr_cids:
        return cid
    return match_nonrobust_by_variants(
        rob_vars.get(cid, set()),
        str(row.get("cos_id", "")),
        nr_by_region.get(str(row.get("region_id", "")), []),
    )


TISSUE_COS_COLS = [
    "cos_id",
    "variants",
    "vcp",
    "cos_npc",
    "npc_outcomes",
    "phenotype_ids",
    "cos_max_pval_nominal_phenotypes",
    "gene_ids_robust",
    "gene_types_robust",
    "gwas_traits",
    "gwas_gws_traits",
    "gene_types_relaxed",
    "gene_ids_relaxed",
]


def _variant_vcp_lists_by_coloc(
    robust_df: pd.DataFrame,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """coloc_id → variants / vcp lists (VCP-descending, unique variants)."""
    if robust_df is None or robust_df.empty:
        return {}, {}
    work = robust_df.copy()
    if "vcp" in work.columns:
        work["vcp"] = pd.to_numeric(work["vcp"], errors="coerce")
        work = work.sort_values("vcp", ascending=False)
    vars_map: Dict[str, list] = {}
    vcp_map: Dict[str, list] = {}
    has_vcp = "vcp" in work.columns
    for cid, grp in work.groupby(work["coloc_id"].astype(str), sort=False):
        seen: set = set()
        vs: list = []
        ps: list = []
        vcps = grp["vcp"] if has_vcp else [np.nan] * len(grp)
        for v, p in zip(grp["variant_id"].astype(str), vcps):
            if v in seen:
                continue
            seen.add(v)
            vs.append(v)
            ps.append(float(p) if pd.notna(p) else float("nan"))
        vars_map[str(cid)] = vs
        vcp_map[str(cid)] = ps
    return vars_map, vcp_map


def build_leads(
    robust_df: pd.DataFrame,
    gene_id_to_type: dict,
    nonrobust_df: Optional[pd.DataFrame] = None,
    *,
    tissue: str,
    id_label: Optional[str] = None,
) -> pd.DataFrame:
    """One CoS row per robust coloc_id (``{id_label}_{N}`` public ids)."""
    if robust_df.empty:
        return pd.DataFrame(columns=TISSUE_COS_COLS)
    label = id_label or tissue
    df = robust_df.copy()
    if "vcp" in df.columns:
        df["vcp"] = pd.to_numeric(df["vcp"], errors="coerce")
        df = df.sort_values("vcp", ascending=False)
    leads = df.drop_duplicates(subset=["coloc_id"], keep="first").copy()
    leads["_old_cid"] = leads["coloc_id"].astype(str)

    # CoS-max GWAS/xQTL -log10(nominal p) across any variant (not just the lead).
    if "pval_nominal_phenotypes" in df.columns:
        leads["cos_max_pval_nominal_phenotypes"] = leads["_old_cid"].map(
            max_pval_nominal_phenotypes(df)
        )
    else:
        leads["cos_max_pval_nominal_phenotypes"] = ""
    if "phenotype_ids" not in leads.columns:
        raise KeyError("expected phenotype_ids on CoS leads")

    # Public cos numbering: sort by region, then by per-block CoS id.
    sort_cols = [c for c in ("region_id", "_old_cid") if c in leads.columns]
    leads = leads.sort_values(sort_cols).reset_index(drop=True)

    vars_map, vcp_map = _variant_vcp_lists_by_coloc(df)
    leads["variants"] = leads["_old_cid"].map(lambda c: vars_map.get(c, []))
    leads["vcp"] = leads["_old_cid"].map(lambda c: vcp_map.get(c, []))

    # Working coloc_id for gene-annotation groupbys (unique within the combine).
    leads["coloc_id"] = leads["_old_cid"]

    exploded = _explode_phenotype_rows(leads, gene_id_to_type)
    gene_type_agg = (
        exploded[exploded["gene_type"].notna()]
        .groupby("coloc_id")["gene_type"]
        .agg(lambda s: sorted(set(s.dropna().astype(str))))
    )
    leads["gene_types_robust"] = leads["coloc_id"].map(gene_type_agg).map(
        lambda x: x if isinstance(x, list) else []
    )

    gene_id_list = (
        exploded[exploded["individual_gene_id"].notna()]
        .groupby("coloc_id")["individual_gene_id"]
        .agg(lambda s: sorted(set(s.dropna().astype(str))))
    )
    leads["gene_ids_robust"] = leads["coloc_id"].map(gene_id_list).map(
        lambda x: x if isinstance(x, list) else []
    )

    gwas_unique_vals = (
        exploded[exploded["phenotype_type"] == "gwas"]
        .groupby("coloc_id")["individual_phenotype_id"]
        .agg(lambda s: sorted(set(s.dropna().astype(str))))
    )
    leads["gwas_traits"] = leads["coloc_id"].map(gwas_unique_vals).map(
        lambda x: x if isinstance(x, list) else []
    )

    leads["gwas_gws_traits"] = [
        gwas_gws_traits_from_cs_max(pid, csmax)
        for pid, csmax in zip(
            leads["phenotype_ids"], leads["cos_max_pval_nominal_phenotypes"]
        )
    ]

    # Relaxed gene composition via variant-overlap match to non-robust assemble.
    rob_vars = _variant_sets_by_coloc_id(df)
    nr_cids, types_by, ids_by, nr_by_region = _nonrobust_annotation_maps(
        nonrobust_df if nonrobust_df is not None else pd.DataFrame(),
        gene_id_to_type,
    )
    nr_types: List = []
    nr_ids: List = []
    for _, row in leads.iterrows():
        match_cid = _match_nonrobust_cid(row, rob_vars, nr_cids, nr_by_region)
        if match_cid is None:
            nr_types.append([])
            nr_ids.append([])
        else:
            nr_types.append(list(types_by.get(match_cid, [])))
            nr_ids.append(list(ids_by.get(match_cid, [])))
    leads["gene_types_relaxed"] = nr_types
    leads["gene_ids_relaxed"] = nr_ids

    leads["cos_id"] = [f"{label}_{i}" for i in range(1, len(leads) + 1)]
    # Keep region_id for ablation item construction (omitted from Zenodo tables by
    # rebuild; pipeline writers may drop via output column lists).
    for c in TISSUE_COS_COLS:
        if c not in leads.columns:
            if c in (
                "variants",
                "vcp",
                "gene_ids_robust",
                "gene_types_robust",
                "gwas_traits",
                "gwas_gws_traits",
                "gene_types_relaxed",
                "gene_ids_relaxed",
            ):
                leads[c] = [[] for _ in range(len(leads))]
            else:
                leads[c] = ""
    out_cols = list(TISSUE_COS_COLS)
    if "region_id" in leads.columns:
        out_cols = out_cols + ["region_id"]
    return leads[out_cols].reset_index(drop=True)


def _colocboost_basename(
    tissue_id: str,
    *,
    file_label: Optional[str] = None,
    mode_tag: Optional[str] = None,
) -> str:
    """Basename for ColocBoost parquet.

    - main: ``{Tissue}.ColocBoost.parquet``
    - discovery: ``Whole_Blood.{mode}.ColocBoost.parquet`` (file_label=Whole_Blood, mode_tag=mode)
    - ablation: ``{Tissue}.{mode}.ColocBoost.parquet`` (mode_tag=mode)
    """
    base = file_label or tissue_id
    if mode_tag:
        return f"{base}.{mode_tag}.ColocBoost.parquet"
    return f"{base}.ColocBoost.parquet"


def combine_tissue(
    coloc_dir: Path,
    output_dir: Path,
    tissue_id: str,
    gene_id_to_type: dict,
    *,
    tss_map: Dict[str, Tuple[str, int]],
    full_dist_output_dir: Path,
    cis_window_bp: int,
    id_label: Optional[str] = None,
    file_label: Optional[str] = None,
    mode_tag: Optional[str] = None,
    drop_region_id: bool = True,
) -> None:
    """Stitch block .txt → ``*.ColocBoost.parquet`` (cis/GWAS filter).

    Variant-level tables are archived under ``full_dist_output_dir`` only.
    """
    files = list_block_files(coloc_dir, tissue_id)
    if not files:
        raise FileNotFoundError(
            f"[combine] No block-level .txt for tissue {tissue_id} in {coloc_dir}"
        )

    full_dist_output_dir.mkdir(parents=True, exist_ok=True)
    # Ablation/recovery: public cos_id = ``{Tissue}.{mode}_{N}`` when mode_tag is set
    # and id_label is not overridden (discovery passes id_label=mode explicitly).
    if id_label is None and mode_tag:
        id_label = f"{tissue_id}.{mode_tag}"
    print(
        f"[combine] {tissue_id}: ColocBoost output filtered "
        f"(strip QTL genes with no set variant within TSS ± {cis_window_bp:,} bp; "
        f"strip GWAS with CS-max p ≥ 1e-5; reclassify GWAS–QTL CoS→uCoS when one "
        f"side remains); full-distance archive → {full_dist_output_dir}"
    )

    buckets = {
        ("joint_gwas", False): [],
        ("joint_gwas", True): [],
        ("xqtl_coloc", False): [],
        ("xqtl_coloc", True): [],
    }
    print(
        f"[combine] {tissue_id}: reading {len(files):,} block tables",
        flush=True,
    )
    for n_read, (path, region, mode, is_robust, tag) in enumerate(files, 1):
        if n_read % 5000 == 0:
            print(
                f"[combine] {tissue_id}: read {n_read:,}/{len(files):,} tables",
                flush=True,
            )
        df = read_block_table(
            path, region, mode, tissue_id=tissue_id, job_tag=tag
        )
        if df is None:
            continue
        buckets[(mode, is_robust)].append(df)

    cis_frames: Dict[Tuple[str, bool], pd.DataFrame] = {}
    n_written = 0
    for mode, is_robust in (
        ("joint_gwas", False),
        ("joint_gwas", True),
        ("xqtl_coloc", False),
        ("xqtl_coloc", True),
    ):
        parts = buckets[(mode, is_robust)]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        tag = ".long" if is_robust else ""
        fname = f"{tissue_id}.{mode}.all{tag}.parquet"

        # Unfiltered + cis-trimmed variant-level archive (diagnostics / rebuild).
        write_parquet(df, full_dist_output_dir / fname)
        n_written += 1
        try:
            df_cis, stats = filter_sets_posthoc_trim(df, tss_map, cis_window_bp)
        except KeyError as e:
            raise KeyError(f"[combine] FATAL cis trim: {e}") from e
        cis_frames[(mode, is_robust)] = df_cis
        write_parquet(df_cis, full_dist_output_dir / f"cis_{fname}")
        print(
            f"[combine] {tissue_id}/{mode}{tag}: posthoc trim "
            f"sets {stats['n_cos_out']:,}/{stats['n_cos_in']:,} kept "
            f"(unchanged={stats['n_cos_unchanged']:,}, "
            f"partial={stats['n_cos_partial']:,}, "
            f"dropped_empty={stats['n_cos_dropped_empty']:,}, "
            f"reclassified_ucos={stats['n_cos_reclassified_ucos']:,}; "
            f"gene×set outside={stats['n_gene_cos_outside']:,}/"
            f"{stats['n_gene_cos_pairs']:,}; "
            f"weak_gwas_stripped={stats['n_weak_gwas_stripped']:,})"
        )
        n_written += 1

        # Write only joint_gwas robust as ColocBoost (one row per CoS).
        if is_robust and mode == "joint_gwas":
            nr_full = (
                pd.concat(buckets[(mode, False)], ignore_index=True)
                if buckets[(mode, False)]
                else pd.DataFrame()
            )
            cos_full = build_leads(
                df,
                gene_id_to_type,
                nonrobust_df=nr_full,
                tissue=tissue_id,
                id_label=id_label,
            )
            if drop_region_id and "region_id" in cos_full.columns:
                cos_full = cos_full.drop(columns=["region_id"])
            full_path = full_dist_output_dir / _colocboost_basename(
                tissue_id, file_label=file_label, mode_tag=mode_tag
            )
            write_leads_parquet(cos_full, str(full_path))
            print(f"Wrote {full_path} ({len(cos_full)} rows)")
            n_written += 1

            nr_main = cis_frames.get((mode, False), pd.DataFrame())
            cos = build_leads(
                df_cis,
                gene_id_to_type,
                nonrobust_df=nr_main,
                tissue=tissue_id,
                id_label=id_label,
            )
            # Keep region_id in the output when drop_region_id=False (ablation
            # select); default output omits it.
            if drop_region_id and "region_id" in cos.columns:
                cos = cos.drop(columns=["region_id"])
            out_name = _colocboost_basename(
                tissue_id, file_label=file_label, mode_tag=mode_tag
            )
            out_path = output_dir / out_name
            write_leads_parquet(cos, str(out_path))
            print(f"Wrote {out_path} ({len(cos)} rows)")
            n_written += 1

    print(f"[combine] {tissue_id}: wrote {n_written} parquet(s) from {len(files)} block tables")


# ---------------------------------------------------------------------------
# Entrypoint helpers + fit/combine orchestration
# ---------------------------------------------------------------------------

_ENSG_RE = re.compile(r"(ENSG\d+)")

# --- paper analysis settings (Methods) ---
CIS_WINDOW_BP = 1000000

COLOC_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ColocItems:
    coloc_items_dir: Path
    items_list: Path
    jobs_tsv: Path


def add_coloc_path_args(parser: argparse.ArgumentParser) -> None:
    """Data-path flags shared by main / discovery / ablation entrypoints."""
    parser.add_argument("--ld_meta", required=True)
    parser.add_argument("--genotype_stem", required=True)
    parser.add_argument("--expression_pheno_dir", required=True)
    parser.add_argument("--splicing_pheno_dir", required=True)
    parser.add_argument("--expression_covar_dir", required=True)
    parser.add_argument("--splicing_covar_dir", required=True)
    parser.add_argument("--gwas_id_list", required=True)
    parser.add_argument("--gwas_dir", required=True)
    parser.add_argument("--gwas_meta", required=True)
    parser.add_argument("--gwas_column_matching", required=True)
    parser.add_argument("--gencode_classification", required=True)
    parser.add_argument("--collapsed_gtf", required=True)


def resolve_jobs_tsv(root: Path) -> Path:
    """Prefer ``jobs.tsv``, else gzip/xz siblings used for large discovery tables."""
    for name in ("jobs.tsv", "jobs.tsv.gz", "jobs.tsv.xz"):
        path = root / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return root / "jobs.tsv"


def resolve_coloc_items_dir(coloc_items_dir: str | Path) -> ColocItems:
    root = Path(coloc_items_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing --coloc_items_dir: {root}")
    return ColocItems(
        coloc_items_dir=root,
        items_list=root / "items.txt",
        jobs_tsv=resolve_jobs_tsv(root),
    )


def require_coloc_items_files(items: ColocItems, required: Sequence[str]) -> None:
    mapping = {
        "items": items.items_list,
        "jobs": items.jobs_tsv,
    }
    for key in required:
        if key not in mapping:
            raise ValueError(f"Unknown coloc items file key: {key}")
        path = mapping[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty {key} file in --coloc_items_dir: {path}"
            )


def read_nonempty_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def tissues_from_region_items(region_items: Sequence[str]) -> list[str]:
    """Unique tissue ids from ``Tissue__chr…`` items (order preserved)."""
    out: list[str] = []
    seen: set[str] = set()
    for item in region_items:
        tissue = str(item).split("__", 1)[0].strip()
        if not tissue or tissue in seen:
            continue
        seen.add(tissue)
        out.append(tissue)
    return out


def load_items(
    coloc_items_dir: str | Path, required: Sequence[str] = ("items",)
) -> ColocItems:
    items = resolve_coloc_items_dir(coloc_items_dir)
    require_coloc_items_files(items, required)
    return items


def coloc_ld_block(
    *,
    region_item: str,
    output_dir: Path,
    paths: argparse.Namespace,
    keep_gene_id_path: Optional[Path] = None,
    jobs_tsv: Optional[Path] = None,
) -> None:
    """Fit one tissue × LD-region item via coloc_ld_block.R."""
    cmd = [
        "Rscript",
        str(COLOC_DIR / "coloc_ld_block.R"),
        "--region_item",
        region_item,
        "--genotype_stem",
        paths.genotype_stem,
        "--expression_pheno_dir",
        paths.expression_pheno_dir,
        "--splicing_pheno_dir",
        paths.splicing_pheno_dir,
        "--expression_covar_dir",
        paths.expression_covar_dir,
        "--splicing_covar_dir",
        paths.splicing_covar_dir,
        "--output_dir",
        str(output_dir),
        "--gwas_id_list",
        paths.gwas_id_list,
        "--gwas_dir",
        paths.gwas_dir,
        "--gwas_meta",
        paths.gwas_meta,
        "--ld_meta",
        paths.ld_meta,
        "--gwas_column_matching",
        paths.gwas_column_matching,
    ]
    if keep_gene_id_path is not None:
        cmd.extend(["--keep_gene_id_path", str(keep_gene_id_path)])
    if jobs_tsv is not None:
        cmd.extend(["--jobs_tsv", str(jobs_tsv)])
    subprocess.run(cmd, check=True)


def combine_one_tissue(
    *,
    tissue_id: str,
    coloc_base_dir: Path,
    tissue_output_dir: Path,
    full_dist_output_dir: Path,
    gencode_classification: str,
    collapsed_gtf: str,
    id_label: Optional[str] = None,
    file_label: Optional[str] = None,
    mode_tag: Optional[str] = None,
    drop_region_id: bool = True,
) -> None:
    """Combine one tissue's block tables into ColocBoost parquet."""
    gene_id_to_type = _load_gencode(gencode_classification)
    collapsed_paths = _parse_collapsed_gtf_arg(collapsed_gtf)
    missing = [str(p) for p in collapsed_paths if not p.is_file()]
    if not collapsed_paths or missing:
        raise FileNotFoundError(
            f"Missing collapsed GTF for cis filter: {missing or '(empty)'}"
        )
    tss_map = load_tss_map_from_gtf(collapsed_paths)
    tissue_output_dir.mkdir(parents=True, exist_ok=True)
    full_dist_output_dir.mkdir(parents=True, exist_ok=True)
    combine_tissue(
        coloc_base_dir,
        tissue_output_dir,
        tissue_id,
        gene_id_to_type,
        tss_map=tss_map,
        full_dist_output_dir=full_dist_output_dir,
        cis_window_bp=CIS_WINDOW_BP,
        id_label=id_label,
        file_label=file_label,
        mode_tag=mode_tag,
        drop_region_id=drop_region_id,
    )


def coloc(
    *,
    items_list: Path,
    coloc_base_dir: Path,
    tissue_output_dir: Path,
    full_dist_output_dir: Path,
    paths: argparse.Namespace,
    keep_gene_id_path: Optional[Path] = None,
    jobs_tsv: Optional[Path] = None,
    label: str = "coloc",
    id_label: Optional[str] = None,
    file_label: Optional[str] = None,
    mode_tag: Optional[str] = None,
    drop_region_id: bool = True,
) -> None:
    """Fit every item, then combine every tissue into ColocBoost parquets.

    Tissues for combine are derived from ``items_list`` (``Tissue__…`` prefix).
    """
    coloc_base_dir.mkdir(parents=True, exist_ok=True)
    tissue_output_dir.mkdir(parents=True, exist_ok=True)
    full_dist_output_dir.mkdir(parents=True, exist_ok=True)

    region_items = read_nonempty_lines(items_list)
    print(f"[{label}] fit {len(region_items)} items from {items_list}")
    for region_item in region_items:
        coloc_ld_block(
            region_item=region_item,
            output_dir=coloc_base_dir,
            paths=paths,
            keep_gene_id_path=keep_gene_id_path,
            jobs_tsv=jobs_tsv,
        )

    tissues = tissues_from_region_items(region_items)
    if not tissues:
        raise ValueError(f"No tissues parsed from items in {items_list}")
    print(f"[{label}] combine {len(tissues)} tissues → {tissue_output_dir}")
    for tissue_id in tissues:
        combine_one_tissue(
            tissue_id=tissue_id,
            coloc_base_dir=coloc_base_dir,
            tissue_output_dir=tissue_output_dir,
            full_dist_output_dir=full_dist_output_dir,
            gencode_classification=paths.gencode_classification,
            collapsed_gtf=paths.collapsed_gtf,
            id_label=id_label,
            file_label=file_label,
            mode_tag=mode_tag,
            drop_region_id=drop_region_id,
        )


def _gene_core(gid: str) -> str:
    m = _ENSG_RE.search(str(gid))
    return m.group(1) if m else str(gid).split(".")[0]


def keep_gene_ids_for_mode(gene_id_to_type: dict, mode: str) -> list[str]:
    """ENSG-core keep list for PC-only or non-novel_lncRNA (v39) restricted fits."""
    if mode == "only_protein_coding":
        return sorted(
            {
                _gene_core(gid)
                for gid, gt in gene_id_to_type.items()
                if gt == "protein_coding" and _gene_core(gid)
            }
        )
    if mode == "only_v39":
        return sorted(
            {
                _gene_core(gid)
                for gid, gt in gene_id_to_type.items()
                if gt != "novel_lncRNA" and _gene_core(gid)
            }
        )
    raise ValueError(f"keep gene list not defined for mode={mode!r}")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: list[str] = []
    seen: set[str] = set()
    for line in lines:
        s = str(line).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        unique.append(s)
    path.write_text("\n".join(unique) + ("\n" if unique else ""))
    print(f"Wrote {path} ({len(unique)} lines)")


def write_keep_gene_ids(*, gencode_classification: str, mode: str, output: Path) -> Path:
    """Write ENSG-core keep list from gencode classification (PC or non-novel_lncRNA)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    gene_id_to_type = _load_gencode(gencode_classification)
    ids = keep_gene_ids_for_mode(gene_id_to_type, mode)
    write_lines(output, ids)
    return output
