#!/usr/bin/env python3
"""Build lncrna_category.txt (figure / Data Availability column set).

Assembles genomic context, conservation, E2G TSS overlap, expression-vs-null,
tissue specificity, and cross-species concordance flags for GENCODE v47 lncRNA/TEC genes.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
from statsmodels.stats.multitest import multipletests

MIN_N_TISSUE = 10
GENOME_CONS_FRAC = 0.0326  # Sullivan et al. genome constrained fraction
TAU_TISSUE_SPECIFIC_MIN = 0.85
E2G_PROMOTER_FLANK = 500
E2G_ELEM_GROUPS = ["promoter", "genic", "intergenic"]
E2G_COLS = ["#chr", "start", "end", "class"]

HOSTED_NCRNA_TYPES = [
    "miRNA",
    "snoRNA",
    "snRNA",
    "scaRNA",
    "scRNA",
    "sRNA",
    "misc_RNA",
    "ribozyme",
    "vault_RNA",
    "vaultRNA",
    "rRNA",
    "rRNA_pseudogene",
]

# Histology subtypes excluded from expressed + τ; cell lines kept for expressed, dropped for τ
_SUBTYPE_TISSUES = {
    "Colon - Transverse - Muscularis",
    "Stomach - Mixed Cell",
    "Pancreas - Mixed Cell",
    "Small Intestine - Terminal Ileum - Mixed Cell",
    "Stomach - Mucosa",
    "Stomach - Muscularis",
    "Pancreas - Acini",
    "Small Intestine - Terminal Ileum - Lymphoid Aggregate",
    "Colon - Transverse - Mixed Cell",
    "Colon - Transverse - Mucosa",
    "Liver - Mixed Cell",
    "Pancreas - Islets",
    "Liver - Hepatocyte",
    "Liver - Portal Tract",
}
_CELL_LINE_TISSUES = {
    "Cells - EBV-transformed lymphocytes",
    "Cells - Cultured fibroblasts",
}
_EXCLUDE_TISSUES = _SUBTYPE_TISSUES | _CELL_LINE_TISSUES

OUT_COLS = [
    "gene_id",
    "gene_id_stable",
    "gene_name",
    "gene_type",
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "genomic_context",
    "is_mirna_host",
    "is_ncORF",
    "primate_cons_exon_unique",
    "primate_cons_sj_unique",
    "tss_e2g_enh",
    "macaque_ortholog",
    "marmoset_ortholog",
    "tau",
    "tau_no_testis",
    "is_tissue_specific",
    "peak_tissue",
    "rescued_from_testis",
    "macaque_tissue_concordant",
    "marmoset_tissue_concordant",
    "expressed_gtex",
    "gtex_min_q",
    "expressed_dgtex",
    "dgtex_min_q",
    "expressed_macaque",
    "macaque_min_q",
    "expressed_marmoset",
    "marmoset_min_q",
]


def _tissue_file_stem(tissue_id: str) -> str:
    return (
        tissue_id.replace("(", "")
        .replace(")", "")
        .replace(" - ", "_")
        .replace(" ", "_")
    )


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise NotADirectoryError(f"{label} not found: {path}")
    return path


def load_tissue_name_map(path: Path) -> dict[str, str]:
    """Load tissue_stem → tissue_name from refs/tissue_name.txt."""
    df = pd.read_csv(path, sep="\t")
    missing = {"tissue_name", "tissue_stem"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No tissue names in {path}")
    return dict(zip(df["tissue_stem"].astype(str), df["tissue_name"].astype(str)))


def load_gtex_official_median_tpm(path: Path, tissue_names_path: Path) -> pd.DataFrame:
    """Load GTEx v11 official gene median TPM GCT; columns = pretty SMTSD names."""
    with gzip.open(path, "rt") as f:
        f.readline()
        f.readline()
        raw = pd.read_csv(f, sep="\t")
    med = raw.set_index("Name").drop(columns=["Description"], errors="ignore")
    stem_to_pretty = load_tissue_name_map(tissue_names_path)
    renamed = {}
    for c in med.columns:
        pretty = stem_to_pretty.get(c) or stem_to_pretty.get(c.replace("-", "_"))
        if pretty is None:
            raise KeyError(f"Cannot map official median tissue column {c!r} to SMTSD")
        renamed[c] = pretty
    return med.rename(columns=renamed)


def strand_overlap(lnc_pr, target_pr, same_strand: bool) -> set:
    """Same/opposite-strand overlap; rebuild PyRanges from dfs to avoid pyranges bugs."""
    ids: set = set()
    lnc_df = lnc_pr.df
    tgt_df = target_pr.df
    for strand in ["+", "-"]:
        opp = "-" if strand == "+" else "+"
        tstrand = strand if same_strand else opp
        lnc_s = lnc_df.loc[lnc_df["Strand"] == strand].reset_index(drop=True)
        tgt_s = tgt_df.loc[tgt_df["Strand"] == tstrand].reset_index(drop=True)
        if lnc_s.empty or tgt_s.empty:
            continue
        ovlp = pr.PyRanges(lnc_s).overlap(pr.PyRanges(tgt_s))
        if len(ovlp) > 0:
            ids.update(ovlp.df["gene_id"])
    return ids


def near_opp_tss_ids(lnc_tss_pr, ref_transcripts, strand_col: str = "ref_strand") -> set:
    ref_tss_win = ref_transcripts[["Chromosome", "Strand", "gene_id", "TSS"]].copy()
    ref_tss_win = ref_tss_win.rename(columns={"Strand": strand_col, "gene_id": "ref_gene_id"})
    ref_tss_win["Start"] = (ref_tss_win["TSS"] - 1000).clip(lower=0)
    ref_tss_win["End"] = ref_tss_win["TSS"] + 1000
    ref_pr = pr.PyRanges(ref_tss_win[["Chromosome", "Start", "End", "ref_gene_id", strand_col]])
    near_df = lnc_tss_pr.join(ref_pr).df
    if len(near_df) == 0:
        return set()
    return set(near_df.loc[near_df["Strand"] != near_df[strand_col], "gene_id"])


def positional_sets(lnc_pr, gene_pr, exon_pr, lnc_tss_pr, transcripts) -> dict:
    sense_exon = strand_overlap(lnc_pr, exon_pr, same_strand=True)
    sense_body = strand_overlap(lnc_pr, gene_pr, same_strand=True)
    antisense_exon = strand_overlap(lnc_pr, exon_pr, same_strand=False)
    antisense_body = strand_overlap(lnc_pr, gene_pr, same_strand=False)
    return {
        "sense_exon": sense_exon,
        "sense_intronic": sense_body - sense_exon,
        "antisense_exon": antisense_exon,
        "antisense_intronic": antisense_body - antisense_exon,
        "antisense_body": antisense_body,
        "near_promoter": near_opp_tss_ids(lnc_tss_pr, transcripts),
    }


def assign_positional(df, sets, gid: str, only_intergenic: bool = False):
    out = df.copy()
    mask = (
        out["genomic_context"].eq("intergenic")
        if only_intergenic
        else pd.Series(True, index=out.index)
    )

    sel = mask & out[gid].isin(sets["sense_intronic"])
    out.loc[sel, "genomic_context"] = "sense_intronic"
    sel = mask & out[gid].isin(sets["sense_exon"])
    out.loc[sel, "genomic_context"] = "sense_overlapping"

    sel = mask & out[gid].isin(sets["antisense_intronic"]) & ~out[gid].isin(sets["near_promoter"])
    out.loc[sel, "genomic_context"] = "antisense_intronic"
    sel = mask & out[gid].isin(sets["antisense_exon"]) & ~out[gid].isin(sets["near_promoter"])
    out.loc[sel, "genomic_context"] = "antisense_overlapping"

    sel = mask & out[gid].isin(sets["near_promoter"])
    out.loc[sel, "genomic_context"] = "bidirectional"
    return out


def _resolve_gtex_tissue_null(gtex_by_tissue_dir: Path, smtsd: str) -> Path | None:
    stem = _tissue_file_stem(smtsd)
    path = gtex_by_tissue_dir / f"{stem}.null_quantifications.txt"
    if path.is_file():
        return path
    stem2 = (
        smtsd.replace(" - ", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    if stem2 != stem:
        path2 = gtex_by_tissue_dir / f"{stem2}.null_quantifications.txt"
        if path2.is_file():
            return path2
    return None


def build_gtex_tissue_fdr_table(
    gtex_null: pd.DataFrame,
    gtex_sample_meta: pd.DataFrame,
    gtex_by_tissue_dir: Path,
    min_n: int = MIN_N_TISSUE,
) -> tuple[pd.DataFrame, pd.Series]:
    """All-gene × tissue FDR 5% calls for bulk sites with n≥min_n (no histology subtypes)."""
    n_by = gtex_sample_meta.loc[gtex_sample_meta["SMAFRZE"] == "RNASEQ"].groupby("SMTSD").size()
    parts = []
    q_parts = []
    missing = []
    for smtsd, n in n_by.items():
        if n < min_n or smtsd in _SUBTYPE_TISSUES:
            continue
        path = _resolve_gtex_tissue_null(gtex_by_tissue_dir, smtsd)
        if path is None:
            missing.append(smtsd)
            continue
        df = pd.read_table(path)
        part = (
            df[["gene_id", "fdr05_quantified"]]
            .rename(columns={"fdr05_quantified": smtsd})
            .set_index("gene_id")
        )
        parts.append(part.astype(bool))
        tested = df["mean_counts"] > 0
        q = pd.Series(np.nan, index=df["gene_id"].astype(str))
        if tested.any():
            _, qv, _, _ = multipletests(df.loc[tested, "p_value"].to_numpy(), method="fdr_bh")
            q.loc[df.loc[tested, "gene_id"].astype(str)] = qv
        q_parts.append(q.rename(smtsd))
    if missing:
        print(f"  WARNING: missing null files for {len(missing)} tissues: {missing[:5]}")
    if not parts:
        raise RuntimeError("No per-tissue null FDR tables found")
    tissue_fdr = parts[0]
    for part in parts[1:]:
        tissue_fdr = tissue_fdr.join(part, how="outer")
    tissue_fdr = tissue_fdr.fillna(False).astype(bool)
    print(
        f"  GTEx tissue FDR matrix: {tissue_fdr.shape[0]:,} genes × {tissue_fdr.shape[1]} tissues "
        f"(n≥{min_n}; no histology subtypes; not written — use expression/by_tissue/GTEx/)"
    )

    pooled_tested = gtex_null["mean_counts"] > 0
    pooled_q = pd.Series(np.nan, index=gtex_null["gene_id"].astype(str))
    if pooled_tested.any():
        _, qv, _, _ = multipletests(
            gtex_null.loc[pooled_tested, "p_value"].to_numpy(), method="fdr_bh"
        )
        pooled_q.loc[gtex_null.loc[pooled_tested, "gene_id"].astype(str)] = qv
    q_mat = pd.concat(q_parts + [pooled_q.rename("pooled")], axis=1)
    min_q = q_mat.min(axis=1, skipna=True)
    return tissue_fdr, min_q


def gtex_expressed_union_from_tissue_fdr(
    gtex_null: pd.DataFrame, tissue_fdr: pd.DataFrame
) -> set[str]:
    pooled = set(gtex_null.loc[gtex_null["fdr05_quantified"], "gene_id"].astype(str))
    any_tissue = set(tissue_fdr.index[tissue_fdr.any(axis=1)].astype(str))
    print(
        f"  GTEx tissues in FDR table: {tissue_fdr.shape[1]}; "
        f"pooled={len(pooled):,}; any_tissue={len(any_tissue):,}; "
        f"union={len(pooled | any_tissue):,}"
    )
    return pooled | any_tissue


def cohort_expressed_union_and_min_q(
    pooled_df: pd.DataFrame,
    tissue_dir: Path,
    gene_col: str = "gene_id",
    min_n: int = MIN_N_TISSUE,
) -> tuple[set[str], pd.Series]:
    """Pooled FDR5 ∪ any by_tissue null table with n_samples≥min_n; plus min BH q."""
    pooled = set(pooled_df.loc[pooled_df["fdr05_quantified"], gene_col].astype(str))
    any_tissue: set[str] = set()
    q_parts = []
    n_used = 0
    pooled_tested = pooled_df["mean_counts"] > 0
    pooled_q = pd.Series(np.nan, index=pooled_df[gene_col].astype(str))
    if pooled_tested.any():
        _, qv, _, _ = multipletests(
            pooled_df.loc[pooled_tested, "p_value"].to_numpy(), method="fdr_bh"
        )
        pooled_q.loc[pooled_df.loc[pooled_tested, gene_col].astype(str)] = qv
    q_parts.append(pooled_q.rename("pooled"))
    for path in sorted(tissue_dir.glob("*.null_quantifications.txt")):
        df = pd.read_table(path)
        n_samp = int(df["n_samples"].iloc[0])
        if n_samp < min_n:
            continue
        any_tissue |= set(df.loc[df["fdr05_quantified"], gene_col].astype(str))
        tested = df["mean_counts"] > 0
        q = pd.Series(np.nan, index=df[gene_col].astype(str))
        if tested.any():
            _, qv, _, _ = multipletests(df.loc[tested, "p_value"].to_numpy(), method="fdr_bh")
            q.loc[df.loc[tested, gene_col].astype(str)] = qv
        q_parts.append(q.rename(path.stem))
        n_used += 1
    print(
        f"  {tissue_dir.name}: tissues={n_used}; pooled={len(pooled):,}; "
        f"any_tissue={len(any_tissue):,}; union={len(pooled | any_tissue):,}"
    )
    min_q = pd.concat(q_parts, axis=1).min(axis=1, skipna=True)
    return pooled | any_tissue, min_q


def compute_tau(tpm_df: pd.DataFrame) -> pd.Series:
    log_tpm = np.log10(tpm_df + 1)
    max_expr = log_tpm.max(axis=1)
    normed = log_tpm.div(max_expr, axis=0)
    n_tissues = log_tpm.shape[1]
    tau = (1 - normed).sum(axis=1) / (n_tissues - 1)
    tau[max_expr == 0] = np.nan
    return tau


def to_uberon_key(name) -> str:
    s = str(name).replace(" - ", "-")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def agg_to_nhp_site(agg) -> str:
    s = str(agg).replace("_", " ")
    return {"Adipose": "Adipose Tissue"}.get(s, s)


def top_two_organs_table(medians: pd.DataFrame) -> pd.DataFrame:
    vals = medians.to_numpy(dtype=float)
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    order = np.argsort(-vals, axis=1)
    cols = np.asarray(medians.columns)
    top_idx = order[:, 0]
    second_idx = order[:, 1] if order.shape[1] > 1 else top_idx
    top = cols[top_idx].astype(object)
    second = cols[second_idx].astype(object)
    top_val = vals[np.arange(len(vals)), top_idx]
    missing = ~np.isfinite(top_val) | (top_val <= 0)
    top[missing] = None
    second[missing] = None
    return pd.DataFrame({"top": top, "second": second}, index=medians.index)


def _merge_interval_arrays(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, sub in df.groupby("chrom"):
        s = sub["start"].to_numpy()
        e = sub["end"].to_numpy()
        order = np.argsort(s, kind="mergesort")
        s, e = s[order], e[order]
        ms, me = [], []
        cs, ce = int(s[0]), int(e[0])
        for i in range(1, len(s)):
            if s[i] <= ce:
                ce = max(ce, int(e[i]))
            else:
                ms.append(cs)
                me.append(ce)
                cs, ce = int(s[i]), int(e[i])
        ms.append(cs)
        me.append(ce)
        merged[str(chrom)] = (np.asarray(ms), np.asarray(me))
    return merged


def _overlaps_any_vec(
    merged: dict[str, tuple[np.ndarray, np.ndarray]],
    chroms: np.ndarray,
    qstart: np.ndarray,
    qend: np.ndarray,
) -> np.ndarray:
    res = np.zeros(len(qstart), dtype=bool)
    for chrom in np.unique(chroms):
        if chrom not in merged:
            continue
        idx = np.where(chroms == chrom)[0]
        starts, ends = merged[chrom]
        pos = np.searchsorted(starts, qend[idx], side="right") - 1
        ok = np.zeros(len(idx), dtype=bool)
        valid = pos >= 0
        v = pos[valid]
        ok[valid] = ends[v] > qstart[idx][valid]
        res[idx] = ok
    return res


def _load_e2g_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Bad e2g manifest line {i + 1}: {line!r}")
            if i == 0 and parts[0].lower() in {"tissue", "gtex_tissue"}:
                continue
            rows.append((parts[0], parts[1]))
    if not rows:
        raise ValueError(f"Empty e2g manifest: {path}")
    return rows


def compute_e2g_tss_overlaps(
    gtf_df: pd.DataFrame,
    gene_ids: set[str],
    e2g_dir: Path,
    e2g_manifest: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-tissue TSS±500bp overlaps with ENCODE-rE2G elements.

    Returns
    -------
    by_tissue
        gene_stable × tissue flags for promoter / genic / intergenic
    tss_e2g_any
        Series indexed by gene_stable: True if any class in any tissue
    """
    manifest = _load_e2g_manifest(e2g_manifest)
    parts = []
    for tissue, file_token in manifest:
        bed = Path(file_token) if os.path.isabs(file_token) else e2g_dir / Path(file_token).name
        if not bed.is_file():
            continue
        df = pd.read_table(bed, usecols=E2G_COLS).rename(
            columns={"#chr": "chrom", "class": "elem_class"}
        )
        elems = df[["chrom", "start", "end", "elem_class"]].drop_duplicates()
        elems["tissue"] = tissue
        parts.append(elems)
    if not parts:
        raise FileNotFoundError(f"No E2G bed files found under {e2g_dir}")
    elements = pd.concat(parts, ignore_index=True)

    tx = gtf_df[gtf_df["Feature"] == "transcript"].copy()
    tx["gene_stable"] = tx["gene_id"].astype(str).str.split(".").str[0]
    tx = tx[tx["gene_stable"].isin(gene_ids)].copy()
    tx["tss"] = np.where(tx["Strand"].to_numpy() == "+", tx["Start"].to_numpy(), tx["End"].to_numpy())
    windows = tx.rename(columns={"Chromosome": "chrom"})[
        ["gene_stable", "chrom", "tss"]
    ].drop_duplicates()
    windows["win_start"] = windows["tss"] - E2G_PROMOTER_FLANK
    windows["win_end"] = windows["tss"] + E2G_PROMOTER_FLANK

    rows = []
    for tissue, sub in elements.groupby("tissue", sort=True):
        t = windows.copy()
        for grp in E2G_ELEM_GROUPS:
            grp_elems = sub[sub["elem_class"] == grp]
            merged = _merge_interval_arrays(grp_elems) if len(grp_elems) else {}
            t[grp] = _overlaps_any_vec(
                merged, t["chrom"].to_numpy(), t["win_start"].to_numpy(), t["win_end"].to_numpy()
            )
        pg = t.groupby("gene_stable", as_index=False)[E2G_ELEM_GROUPS].any()
        pg["tissue"] = tissue
        rows.append(pg)
    by_tissue = pd.concat(rows, ignore_index=True)
    any_tissue = by_tissue.groupby("gene_stable")[E2G_ELEM_GROUPS].any()
    tss_e2g_any = any_tissue.any(axis=1)
    tss_e2g_any.name = "tss_e2g_enh"
    return by_tissue, tss_e2g_any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gencode-v47", type=Path, required=True, help="GENCODE v47 full annotation GTF")
    p.add_argument(
        "--gencode-classification",
        type=Path,
        required=True,
        help="gencode_v39_v47_gene_classification.txt",
    )
    p.add_argument("--riboseq-orf", type=Path, required=True, help="CLS riboSeqMin ORF TSV")
    p.add_argument(
        "--gtex-null",
        type=Path,
        required=True,
        help="Pooled GTEx null quantifications TSV",
    )
    p.add_argument(
        "--dgtex-null",
        type=Path,
        required=True,
        help="Pooled dGTEx null quantifications TSV",
    )
    p.add_argument(
        "--macaque-null",
        type=Path,
        required=True,
        help="Pooled macaque null quantifications TSV",
    )
    p.add_argument(
        "--marmoset-null",
        type=Path,
        required=True,
        help="Pooled marmoset null quantifications TSV",
    )
    p.add_argument(
        "--gtex-by-tissue-dir",
        type=Path,
        required=True,
        help="Directory of per-tissue GTEx *.null_quantifications.txt",
    )
    p.add_argument(
        "--dgtex-by-tissue-dir",
        type=Path,
        required=True,
        help="Directory of per-tissue dGTEx *.null_quantifications.txt",
    )
    p.add_argument(
        "--macaque-by-tissue-dir",
        type=Path,
        required=True,
        help="Directory of per-tissue macaque *.null_quantifications.txt",
    )
    p.add_argument(
        "--marmoset-by-tissue-dir",
        type=Path,
        required=True,
        help="Directory of per-tissue marmoset *.null_quantifications.txt",
    )
    p.add_argument("--orthologs", type=Path, required=True, help="Human–NHP lifted lncRNA TSV")
    p.add_argument("--uberon-match", type=Path, required=True, help="Uberon ID match table")
    p.add_argument(
        "--gtex-sample-meta",
        type=Path,
        required=True,
        help="GTEx sample attributes (SampleAttributesDS)",
    )
    p.add_argument(
        "--tissue-names",
        type=Path,
        required=True,
        help="GTEx tissue_name/tissue_stem table (refs/tissue_name.txt)",
    )
    p.add_argument(
        "--gtex-median-tpm",
        type=Path,
        required=True,
        help="Official GTEx gene median TPM GCT.gz",
    )
    p.add_argument(
        "--macaque-site-medians",
        type=Path,
        required=True,
        help="Macaque tissue-site median expression table",
    )
    p.add_argument(
        "--marmoset-site-medians",
        type=Path,
        required=True,
        help="Marmoset tissue-site median expression table",
    )
    p.add_argument(
        "--gene-cons-frac",
        type=Path,
        required=True,
        help="gene_constraint_fractions.tsv.gz from 02_compute_gene_constraint_fractions.py",
    )
    p.add_argument(
        "--e2g-dir",
        type=Path,
        required=True,
        help="Directory of ENCODE-rE2G prediction beds",
    )
    p.add_argument(
        "--e2g-manifest",
        type=Path,
        required=True,
        help="TSV: tissue\\tbed_file (basename or path)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output lncrna_category.txt",
    )
    p.add_argument(
        "--e2g-by-tissue-out",
        type=Path,
        required=True,
        help="Output gene_tss_e2g_by_tissue.parquet",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gencode_v47 = _require_file(args.gencode_v47, "GENCODE v47 GTF")
    gencode_class = _require_file(args.gencode_classification, "GENCODE classification")
    riboseq_orf_path = _require_file(args.riboseq_orf, "Ribo-seq ORF table")
    gtex_null_path = _require_file(args.gtex_null, "GTEx null")
    dgtex_null_path = _require_file(args.dgtex_null, "dGTEx null")
    macaque_null_path = _require_file(args.macaque_null, "macaque null")
    marmoset_null_path = _require_file(args.marmoset_null, "marmoset null")
    gtex_by_tissue_dir = _require_dir(args.gtex_by_tissue_dir, "GTEx by-tissue dir")
    dgtex_by_tissue_dir = _require_dir(args.dgtex_by_tissue_dir, "dGTEx by-tissue dir")
    macaque_by_tissue_dir = _require_dir(args.macaque_by_tissue_dir, "macaque by-tissue dir")
    marmoset_by_tissue_dir = _require_dir(args.marmoset_by_tissue_dir, "marmoset by-tissue dir")
    orthologs_path = _require_file(args.orthologs, "orthologs")
    uberon_match_path = _require_file(args.uberon_match, "Uberon match")
    gtex_sample_meta_path = _require_file(args.gtex_sample_meta, "GTEx sample meta")
    tissue_names_path = _require_file(args.tissue_names, "GTEx tissue names")
    gtex_median_tpm = _require_file(args.gtex_median_tpm, "GTEx median TPM")
    macaque_site_medians_path = _require_file(args.macaque_site_medians, "macaque site medians")
    marmoset_site_medians_path = _require_file(
        args.marmoset_site_medians, "marmoset site medians"
    )
    gene_cons_frac = _require_file(args.gene_cons_frac, "gene constraint fractions")
    e2g_dir = _require_dir(args.e2g_dir, "E2G beds directory")
    e2g_manifest = _require_file(args.e2g_manifest, "E2G manifest")

    full_gencode_v47 = pr.read_gtf(str(gencode_v47)).as_df()
    gencode_cls = pd.read_table(
        gencode_class, usecols=["gene_id_stable", "gene_type_grouped"]
    )
    riboseq_orf = pd.read_table(riboseq_orf_path, usecols=["gene", "translation"])
    riboseq_orf = riboseq_orf[riboseq_orf["translation"] == "Translated"]

    gtex_null = pd.read_table(gtex_null_path)
    dgtex_null = pd.read_table(dgtex_null_path)
    macaque_null = pd.read_table(macaque_null_path)
    marmoset_null = pd.read_table(marmoset_null_path)

    orth = pd.read_csv(orthologs_path, sep="\t")
    uberon_match = pd.read_csv(uberon_match_path)
    gtex_sample_meta = pd.read_csv(
        gtex_sample_meta_path, sep="\t", usecols=["SAMPID", "SMTSD", "SMTS", "SMAFRZE"]
    )
    macaque_site_medians = pd.read_csv(macaque_site_medians_path, sep="\t", index_col=0)
    marmoset_site_medians = pd.read_csv(marmoset_site_medians_path, sep="\t", index_col=0)

    median_tpm_all = load_gtex_official_median_tpm(gtex_median_tpm, tissue_names_path)
    _before = median_tpm_all.shape[1]
    median_tpm = median_tpm_all[[c for c in median_tpm_all.columns if c not in _EXCLUDE_TISSUES]]
    print(
        f"τ tissues: {median_tpm.shape[1]} "
        f"(expressed uses these + {len(_CELL_LINE_TISSUES)} cell lines = "
        f"{median_tpm.shape[1] + len(_CELL_LINE_TISSUES)}; "
        f"dropped {_before - median_tpm.shape[1]} subtypes/cell lines from {_before} official columns)"
    )
    print(
        f"GTEx medians (official): {median_tpm.shape}; "
        f"macaque sites: {macaque_site_medians.shape}; marmoset sites: {marmoset_site_medians.shape}"
    )

    # --- Feature extraction ---
    gencode_genes = full_gencode_v47[full_gencode_v47["Feature"] == "gene"][
        ["Chromosome", "Start", "End", "Strand", "gene_id", "gene_type", "gene_name"]
    ].copy()

    pc_genes = gencode_genes[gencode_genes["gene_type"] == "protein_coding"].copy()
    lnc_genes = gencode_genes[gencode_genes["gene_type"].isin(["lncRNA", "TEC"])].copy()
    pseudogene_genes = gencode_genes[
        gencode_genes["gene_type"].str.contains("pseudogene", na=False)
    ].copy()

    pc_exons = full_gencode_v47[
        (full_gencode_v47["Feature"] == "exon")
        & (full_gencode_v47["gene_type"] == "protein_coding")
    ][["Chromosome", "Start", "End", "Strand", "gene_id"]].copy()

    pseudogene_exons = full_gencode_v47[
        (full_gencode_v47["Feature"] == "exon")
        & (full_gencode_v47["gene_type"].str.contains("pseudogene", na=False))
    ][["Chromosome", "Start", "End", "Strand", "gene_id"]].copy()

    small_ncrna = gencode_genes[gencode_genes["gene_type"].isin(HOSTED_NCRNA_TYPES)].copy()

    cls_by_stable = gencode_cls.set_index("gene_id_stable")["gene_type_grouped"]
    lnc_genes["gene_id_stable"] = lnc_genes["gene_id"].str.split(".").str[0]
    lnc_genes["gene_type_grouped"] = lnc_genes["gene_id_stable"].map(cls_by_stable)

    n_existing_lnc = int(
        (gencode_cls["gene_type_grouped"] == "existing_lncRNA").sum()
    )

    pc_transcripts = full_gencode_v47[
        (full_gencode_v47["Feature"] == "transcript")
        & (full_gencode_v47["gene_type"] == "protein_coding")
    ][["Chromosome", "Start", "End", "Strand", "gene_id", "transcript_id"]].copy()

    lnc_transcripts = full_gencode_v47[
        (full_gencode_v47["Feature"] == "transcript")
        & (full_gencode_v47["gene_type"].isin(["lncRNA", "TEC"]))
    ][["Chromosome", "Start", "End", "Strand", "gene_id", "transcript_id"]].copy()

    pseudogene_transcripts = full_gencode_v47[
        (full_gencode_v47["Feature"] == "transcript")
        & (full_gencode_v47["gene_type"].str.contains("pseudogene", na=False))
    ][["Chromosome", "Start", "End", "Strand", "gene_id", "transcript_id"]].copy()

    for _df in (pc_transcripts, lnc_transcripts, pseudogene_transcripts):
        _df["TSS"] = np.where(_df["Strand"] == "+", _df["Start"], _df["End"])

    def _strand_ok(df):
        return df[df["Strand"].isin(["+", "-"])].copy()

    pc_genes = _strand_ok(pc_genes)
    lnc_genes = _strand_ok(lnc_genes)
    pseudogene_genes = _strand_ok(pseudogene_genes)
    pc_exons = _strand_ok(pc_exons)
    pseudogene_exons = _strand_ok(pseudogene_exons)
    small_ncrna = _strand_ok(small_ncrna)
    pc_transcripts = _strand_ok(pc_transcripts)
    lnc_transcripts = _strand_ok(lnc_transcripts)
    pseudogene_transcripts = _strand_ok(pseudogene_transcripts)

    print(
        f"PC genes: {len(pc_genes):,} | Pseudogene: {len(pseudogene_genes):,} | "
        f"lncRNA+TEC: {len(lnc_genes):,}"
    )
    print(
        f"  novel: {lnc_genes['gene_type_grouped'].eq('novel_lncRNA').sum():,} | "
        f"existing lncRNA genes: {n_existing_lnc:,}"
    )

    # --- Overlap and proximity ---
    lnc_pr = pr.PyRanges(lnc_genes[["Chromosome", "Start", "End", "Strand", "gene_id"]])
    ncrna_pr = pr.PyRanges(small_ncrna[["Chromosome", "Start", "End", "Strand", "gene_id"]])
    pc_pr = pr.PyRanges(pc_genes[["Chromosome", "Start", "End", "Strand", "gene_id"]])
    pc_exon_pr = pr.PyRanges(pc_exons[["Chromosome", "Start", "End", "Strand", "gene_id"]])
    ps_pr = pr.PyRanges(pseudogene_genes[["Chromosome", "Start", "End", "Strand", "gene_id"]])
    ps_exon_pr = pr.PyRanges(
        pseudogene_exons[["Chromosome", "Start", "End", "Strand", "gene_id"]]
    )
    lnc_tss = lnc_transcripts[["Chromosome", "Strand", "gene_id", "transcript_id", "TSS"]].copy()
    lnc_tss["Start"] = lnc_tss["TSS"]
    lnc_tss["End"] = lnc_tss["TSS"] + 1
    lnc_tss_pr = pr.PyRanges(
        lnc_tss[["Chromosome", "Start", "End", "gene_id", "transcript_id", "Strand"]]
    )

    host_gene_ids = strand_overlap(lnc_pr, ncrna_pr, same_strand=True)
    mirna_df = small_ncrna.loc[
        small_ncrna["gene_type"].eq("miRNA"),
        ["Chromosome", "Start", "End", "Strand", "gene_id"],
    ].reset_index(drop=True)
    mirna_pr = pr.PyRanges(mirna_df)
    mirna_host_ids = strand_overlap(lnc_pr, mirna_pr, same_strand=True)
    pc_sets = positional_sets(lnc_pr, pc_pr, pc_exon_pr, lnc_tss_pr, pc_transcripts)
    ps_sets = positional_sets(lnc_pr, ps_pr, ps_exon_pr, lnc_tss_pr, pseudogene_transcripts)

    print(f"is_host_gene (any small ncRNA): {len(host_gene_ids):,}")
    print(
        f"is_mirna_host (miRNA only): {len(mirna_host_ids):,} "
        f"(subset of host: {len(mirna_host_ids & host_gene_ids):,})"
    )
    print(
        f"PC near promoter: {len(pc_sets['near_promoter']):,} | "
        f"sense exon: {len(pc_sets['sense_exon']):,}"
    )
    print(
        f"Pseudogene near promoter: {len(ps_sets['near_promoter']):,} | "
        f"sense exon: {len(ps_sets['sense_exon']):,}"
    )

    # --- Genomic position assignment ---
    lncrnas = lnc_genes.copy()
    gid = "gene_id"
    lncrnas["genomic_context"] = "intergenic"
    lncrnas = assign_positional(lncrnas, pc_sets, gid, only_intergenic=False)
    n_pc = (~lncrnas["genomic_context"].eq("intergenic")).sum()
    lncrnas = assign_positional(lncrnas, ps_sets, gid, only_intergenic=True)
    n_ps = (~lncrnas["genomic_context"].eq("intergenic")).sum() - n_pc

    lncrnas["is_host_gene"] = lncrnas[gid].isin(host_gene_ids)
    lncrnas["is_mirna_host"] = lncrnas[gid].isin(mirna_host_ids)

    riboseq_tx_stable = set(riboseq_orf["gene"].astype(str).str.split(".").str[0])
    lnc_tx_for_orf = lnc_transcripts[["gene_id", "transcript_id"]].dropna().copy()
    lnc_tx_for_orf["transcript_id_stable"] = (
        lnc_tx_for_orf["transcript_id"].astype(str).str.split(".").str[0]
    )
    ncorf_gene_ids = set(
        lnc_tx_for_orf.loc[
            lnc_tx_for_orf["transcript_id_stable"].isin(riboseq_tx_stable),
            "gene_id",
        ]
    )
    lncrnas["is_ncORF"] = lncrnas[gid].isin(ncorf_gene_ids)

    assert set(lncrnas["genomic_context"]) <= {
        "intergenic",
        "sense_intronic",
        "sense_overlapping",
        "antisense_intronic",
        "antisense_overlapping",
        "bidirectional",
    }
    print(lncrnas["genomic_context"].value_counts())
    print(f"PC-assigned: {n_pc:,} | Pseudogene-assigned: {n_ps:,}")
    print(
        f"is_host_gene: {lncrnas['is_host_gene'].sum():,} | "
        f"is_mirna_host: {lncrnas['is_mirna_host'].sum():,} | "
        f"is_ncORF: {lncrnas['is_ncORF'].sum():,}"
    )

    # --- Ortholog and cross-species expression flags ---
    for col in ["gene_id.macaque", "gene_id.marmoset"]:
        orth[col] = orth[col].replace(["NA", "-"], np.nan)

    orth_by_human = orth.drop_duplicates("gene_id.human").set_index("gene_id.human")
    mac_id = lncrnas["gene_id"].map(orth_by_human["gene_id.macaque"])
    mar_id = lncrnas["gene_id"].map(orth_by_human["gene_id.marmoset"])
    lncrnas["macaque_ortholog_id"] = mac_id
    lncrnas["marmoset_ortholog_id"] = mar_id
    lncrnas["macaque_ortholog"] = mac_id.notna()
    lncrnas["marmoset_ortholog"] = mar_id.notna()

    print(f"Building expressed unions (min_n={MIN_N_TISSUE})…")
    gtex_tissue_fdr, gtex_min_q = build_gtex_tissue_fdr_table(
        gtex_null,
        gtex_sample_meta,
        gtex_by_tissue_dir,
        MIN_N_TISSUE,
    )
    gtex_expr_ids = gtex_expressed_union_from_tissue_fdr(gtex_null, gtex_tissue_fdr)
    dgtex_expr_ids, dgtex_min_q = cohort_expressed_union_and_min_q(
        dgtex_null, dgtex_by_tissue_dir
    )
    macaque_expr_ids, macaque_min_q = cohort_expressed_union_and_min_q(
        macaque_null, macaque_by_tissue_dir
    )
    marmoset_expr_ids, marmoset_min_q = cohort_expressed_union_and_min_q(
        marmoset_null, marmoset_by_tissue_dir
    )

    lncrnas["expressed_gtex"] = lncrnas["gene_id"].isin(gtex_expr_ids)
    lncrnas["expressed_dgtex"] = lncrnas["gene_id"].isin(dgtex_expr_ids)
    lncrnas["expressed_macaque"] = lncrnas["macaque_ortholog_id"].isin(macaque_expr_ids)
    lncrnas["expressed_marmoset"] = lncrnas["marmoset_ortholog_id"].isin(marmoset_expr_ids)

    lncrnas["gtex_min_q"] = lncrnas["gene_id"].map(gtex_min_q)
    lncrnas["dgtex_min_q"] = lncrnas["gene_id"].map(dgtex_min_q)
    lncrnas["macaque_min_q"] = lncrnas["macaque_ortholog_id"].map(macaque_min_q)
    lncrnas["marmoset_min_q"] = lncrnas["marmoset_ortholog_id"].map(marmoset_min_q)

    print(
        lncrnas[
            [
                "macaque_ortholog",
                "marmoset_ortholog",
                "expressed_gtex",
                "expressed_dgtex",
                "expressed_macaque",
                "expressed_marmoset",
            ]
        ]
        .sum()
        .to_string()
    )
    print(
        "min_q coverage:",
        {
            c: int(lncrnas[c].notna().sum())
            for c in ["gtex_min_q", "dgtex_min_q", "macaque_min_q", "marmoset_min_q"]
        },
    )

    # --- Sequence conservation (primate unique-to-gene only) ---
    cons = pd.read_table(gene_cons_frac)
    cons = cons[cons["gene_type_grouped"].isin(["novel_lncRNA", "existing_lncRNA"])].copy()
    frac_cols = ["frac_cons_exon_unique_primate", "frac_cons_sj_unique_primate"]

    lncrnas["gene_id_stable"] = lncrnas["gene_id"].str.split(".").str[0]
    lncrnas = lncrnas.merge(
        cons[["gene_id_stable"] + frac_cols],
        on="gene_id_stable",
        how="left",
    )
    lncrnas["primate_cons_exon_unique"] = (
        lncrnas["frac_cons_exon_unique_primate"] > GENOME_CONS_FRAC
    ).fillna(False)
    lncrnas["primate_cons_sj_unique"] = (
        lncrnas["frac_cons_sj_unique_primate"] > GENOME_CONS_FRAC
    ).fillna(False)
    print(
        f"Primate unique Zoonomia fractions: "
        f"{lncrnas['frac_cons_exon_unique_primate'].notna().sum():,}/{len(lncrnas):,}; "
        f"primate_cons_exon_unique: {int(lncrnas['primate_cons_exon_unique'].sum()):,}; "
        f"primate_cons_sj_unique: {int(lncrnas['primate_cons_sj_unique'].sum()):,}"
    )

    # --- Tissue specificity, Testis rescue, NHP concordance ---
    expressed_gtex_ids = set(lncrnas.loc[lncrnas["expressed_gtex"], "gene_id"])
    tissue_to_organ = (
        gtex_sample_meta.drop_duplicates("SMTSD").set_index("SMTSD")["SMTS"].to_dict()
    )
    tissue_to_organ_map = {t: tissue_to_organ.get(t, t) for t in median_tpm.columns}
    non_testis_cols = [c for c in median_tpm.columns if tissue_to_organ_map.get(c, c) != "Testis"]

    median_tpm_filt = median_tpm.loc[median_tpm.index.isin(expressed_gtex_ids)]
    tau_all = compute_tau(median_tpm_filt)
    tau_nt = compute_tau(median_tpm_filt[non_testis_cols])

    tau_df = pd.DataFrame(
        {
            "gene_id": tau_all.index,
            "tau": tau_all.values,
            "tau_no_testis": tau_nt.reindex(tau_all.index).values,
        }
    )
    tau_df["is_tissue_specific"] = tau_df["tau"] > TAU_TISSUE_SPECIFIC_MIN
    tau_df["is_ts_no_testis"] = tau_df["tau_no_testis"] > TAU_TISSUE_SPECIFIC_MIN
    tau_df["max_tissue_all"] = median_tpm.reindex(tau_df["gene_id"]).idxmax(axis=1).values
    tau_df["max_tissue_no_testis"] = (
        median_tpm.reindex(tau_df["gene_id"])[non_testis_cols].idxmax(axis=1).values
    )
    tau_df["max_organ_all"] = tau_df["max_tissue_all"].map(
        lambda t: tissue_to_organ_map.get(t, t) if pd.notna(t) else np.nan
    )

    is_testis_peak = tau_df["max_organ_all"].eq("Testis")
    tau_df["rescued_from_testis"] = (
        tau_df["is_tissue_specific"].fillna(False)
        & is_testis_peak
        & tau_df["is_ts_no_testis"].fillna(False)
    )
    tau_df["peak_tissue"] = np.where(
        tau_df["rescued_from_testis"],
        tau_df["max_tissue_no_testis"],
        tau_df["max_tissue_all"],
    )
    tau_df["peak_organ"] = tau_df["peak_tissue"].map(
        lambda t: tissue_to_organ_map.get(t, t) if pd.notna(t) else np.nan
    )

    lncrnas = lncrnas.merge(
        tau_df[
            [
                "gene_id",
                "tau",
                "tau_no_testis",
                "is_tissue_specific",
                "peak_tissue",
                "peak_organ",
                "rescued_from_testis",
            ]
        ],
        on="gene_id",
        how="left",
    )
    lncrnas["is_tissue_specific"] = lncrnas["is_tissue_specific"].fillna(False)
    lncrnas["rescued_from_testis"] = lncrnas["rescued_from_testis"].fillna(False)

    uberon_strict = uberon_match.loc[uberon_match["UberonID_match"] == 1].copy()
    gtex_key_to_name = {to_uberon_key(t): t for t in median_tpm.columns}
    gtex_tissue_to_nhp_sites: dict[str, set] = {}
    for _, row in uberon_strict.iterrows():
        gtex_name = gtex_key_to_name.get(row["GTEx_detail_tissue"])
        if gtex_name is None:
            continue
        site = agg_to_nhp_site(row["dGTEx_aggregate_tissue"])
        gtex_tissue_to_nhp_sites.setdefault(gtex_name, set()).add(site)

    def nhp_peak_matches(gtex_tissue, gtex_organ, top, second):
        if top is None or pd.isna(top):
            return False
        allowed = set()
        if gtex_organ is not None and not pd.isna(gtex_organ):
            allowed.add(gtex_organ)
        if gtex_tissue is not None and not pd.isna(gtex_tissue):
            allowed |= gtex_tissue_to_nhp_sites.get(gtex_tissue, set())
        if not allowed:
            return False
        if top in allowed:
            return True
        return top == "Testis" and (second in allowed)

    mac_peaks = top_two_organs_table(macaque_site_medians)
    mar_peaks = top_two_organs_table(marmoset_site_medians)

    mac_top = lncrnas["macaque_ortholog_id"].map(mac_peaks["top"])
    mac_second = lncrnas["macaque_ortholog_id"].map(mac_peaks["second"])
    mar_top = lncrnas["marmoset_ortholog_id"].map(mar_peaks["top"])
    mar_second = lncrnas["marmoset_ortholog_id"].map(mar_peaks["second"])

    mac_match_raw = np.array(
        [
            nhp_peak_matches(g, o, t, s)
            for g, o, t, s in zip(
                lncrnas["peak_tissue"], lncrnas["peak_organ"], mac_top, mac_second
            )
        ]
    )
    mar_match_raw = np.array(
        [
            nhp_peak_matches(g, o, t, s)
            for g, o, t, s in zip(
                lncrnas["peak_tissue"], lncrnas["peak_organ"], mar_top, mar_second
            )
        ]
    )

    lncrnas["macaque_tissue_concordant"] = (
        lncrnas["is_tissue_specific"]
        & lncrnas["macaque_ortholog"]
        & mac_match_raw
        & lncrnas["expressed_macaque"]
    )
    lncrnas["marmoset_tissue_concordant"] = (
        lncrnas["is_tissue_specific"]
        & lncrnas["marmoset_ortholog"]
        & mar_match_raw
        & lncrnas["expressed_marmoset"]
    )

    print(
        f"tissue-specific: {lncrnas['is_tissue_specific'].sum():,}; "
        f"rescued_from_testis: {lncrnas['rescued_from_testis'].sum():,}"
    )
    print(
        f"macaque_tissue_concordant: {lncrnas['macaque_tissue_concordant'].sum():,}; "
        f"marmoset_tissue_concordant: {lncrnas['marmoset_tissue_concordant'].sum():,}"
    )
    print("peak_tissue (TS genes) top 10:")
    print(
        lncrnas.loc[lncrnas["is_tissue_specific"], "peak_tissue"].value_counts().head(10).to_string()
    )

    # --- E2G TSS overlaps ---
    print("Computing ENCODE-rE2G TSS overlaps…")
    e2g_gene_ids = set(
        gencode_cls.loc[
            gencode_cls["gene_type_grouped"].isin(
                ["novel_lncRNA", "existing_lncRNA", "protein_coding"]
            ),
            "gene_id_stable",
        ]
    )
    e2g_by_tissue, tss_e2g_any = compute_e2g_tss_overlaps(
        full_gencode_v47, e2g_gene_ids, e2g_dir, e2g_manifest
    )
    args.e2g_by_tissue_out.parent.mkdir(parents=True, exist_ok=True)
    e2g_by_tissue.to_parquet(args.e2g_by_tissue_out, index=False)
    print(f"Wrote {args.e2g_by_tissue_out} ({len(e2g_by_tissue):,} rows)")
    lncrnas["tss_e2g_enh"] = lncrnas["gene_id_stable"].map(tss_e2g_any).fillna(False).astype(bool)
    print(f"tss_e2g_enh: {int(lncrnas['tss_e2g_enh'].sum()):,}")

    # --- Write outputs ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lncrnas[OUT_COLS].to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(lncrnas):,} lncRNA genes to {args.output}")
    print("Columns:", ", ".join(OUT_COLS))
    print(f"E2G by-tissue table: {args.e2g_by_tissue_out}")


if __name__ == "__main__":
    main()
