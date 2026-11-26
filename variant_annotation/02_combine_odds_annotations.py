#!/usr/bin/env python3
"""Join VEP annotations with max-PIP / cis-loci columns from GTEx v11 SuSiE.

Takes a single VEP TSV from ``01_vep_annotate_variants.py`` (binary ``*_d`` columns
plus ``af_c``), renames annotation columns with a ``_v11`` suffix for
downstream Firth helpers, and writes ``odds_annotations.parquet``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os

import numpy as np
import pandas as pd

from qtl_odds import (
    LNCRNA_SUBTYPES,
    PIP_LOCI_GENE_TYPES,
    build_group_gene_sets,
    load_susie_all_tissues,
)

# Column bases must match 01_vep_annotate_variants.BASE_* keys.
_vep = importlib.import_module("01_vep_annotate_variants")
REGULATORY_CATEGORIES = list(_vep.BASE_REGULATORY_COLUMNS)
CONSEQUENCE_CATEGORIES = list(_vep.BASE_CONSEQUENCE_COLUMNS)

SUSIE_TYPES = ["eqtl", "sqtl"]
ANNOT_SUFFIX = "_v11"


def parse_snp(snp_series: pd.Series):
    parts = snp_series.str.split("_", n=2, expand=True)
    chr_ = parts[0]
    pos = pd.to_numeric(parts[1], errors="coerce")
    return chr_, pos


def build_loci_windows(genes_with_cs, gene_span):
    span_sub = gene_span[gene_span["gene_id"].str.split(".").str[0].isin(genes_with_cs)]
    if span_sub.empty:
        return []
    return list(span_sub[["chr", "window_start", "window_end"]].itertuples(index=False, name=None))


def compute_loci_columns_fast(merged, loci_windows):
    out = merged.copy()
    for (gt, st), windows in loci_windows.items():
        col = f"{gt}_{st}_loci"
        if not windows:
            out[col] = False
            continue
        by_chr = {}
        for wchr, wstart, wend in windows:
            by_chr.setdefault(wchr, []).append((wstart, wend))
        in_loci = np.zeros(len(out), dtype=bool)
        for c in out["chr"].unique():
            mask = (out["chr"] == c).values
            if c not in by_chr:
                continue
            intervals = by_chr[c]
            starts = np.array([i[0] for i in intervals], dtype=np.int64)
            ends = np.array([i[1] for i in intervals], dtype=np.int64)
            order = np.argsort(starts)
            starts = starts[order]
            ends = ends[order]
            max_ends = np.maximum.accumulate(ends)
            positions = (
                out.loc[mask, "pos"].astype("Int64").fillna(-1).astype(np.int64).values
            )
            valid_pos = positions >= 0
            idx = np.searchsorted(starts, positions[valid_pos], side="right") - 1
            idx = np.clip(idx, 0, len(starts) - 1)
            overlaps = (starts[idx] <= positions[valid_pos]) & (
                max_ends[idx] >= positions[valid_pos]
            )
            in_loci[np.where(mask)[0][valid_pos]] = overlaps
        out[col] = in_loci
    return out


def compute_pip_columns(merged, susie, susie_type, expanded_gene_types, group_gene_sets):
    out = merged.copy()
    for gt in expanded_gene_types:
        col = f"{gt}_{susie_type}_pip"
        genes = group_gene_sets.get(gt, set())
        if not genes:
            out[col] = 0
            continue
        susie_df = susie[susie["gene_id_stable"].isin(genes)]
        pip_max = susie_df.groupby("variant_id")["pip"].max()
        out[col] = out["SNP"].map(pip_max).fillna(0)
    return out


def suffix_vep_annot_columns(vep: pd.DataFrame, gencode_grouped_types: list[str]) -> pd.DataFrame:
    """Rename bare ``*_d`` VEP columns to ``*_d_v11`` for Firth helpers / notebooks."""
    out = vep.copy()
    rename = {}
    for cat in REGULATORY_CATEGORIES:
        src = f"{cat}_d"
        if src in out.columns:
            rename[src] = f"{src}{ANNOT_SUFFIX}"
    for cat in CONSEQUENCE_CATEGORIES:
        for gt in gencode_grouped_types:
            src = f"{cat}_{gt}_d"
            if src in out.columns:
                rename[src] = f"{src}{ANNOT_SUFFIX}"
    return out.rename(columns=rename)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vep_annotations",
        required=True,
        help="VEP TSV/TSV.gz from 01_vep_annotate_variants.py (with af_c)",
    )
    p.add_argument(
        "--gene_loci_pos",
        required=True,
        help="4-col TSV (#chr, start, end, gene_id); Zenodo: variant_annotation/gene_positions.txt",
    )
    p.add_argument("--esusie_dir", required=True)
    p.add_argument("--ssusie_dir", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--lncrna_category_file", required=True)
    p.add_argument("--gencode_class", required=True)
    args = p.parse_args()

    vep = pd.read_table(args.vep_annotations)
    if "SNP" not in vep.columns:
        raise KeyError("--vep_annotations must have a SNP column")
    if "af_c" not in vep.columns:
        raise KeyError("--vep_annotations missing af_c (join AF in 01_ via --af_table)")

    with open(args.gencode_class, newline="") as gf:
        r = csv.DictReader(gf, delimiter="\t")
        gencode_grouped_types = sorted(
            {
                row["gene_type_grouped"].strip()
                for row in r
                if row.get("gene_type_grouped", "").strip()
            }
        )

    merged = suffix_vep_annot_columns(vep, gencode_grouped_types)
    merged["chr"], merged["pos"] = parse_snp(merged["SNP"])
    merged["pos"] = merged["pos"].astype("Int64")

    gene_pos = pd.read_table(
        args.gene_loci_pos,
        comment="#",
        header=None,
        names=["chr", "start", "end", "gene_id"],
    )
    gene_pos = gene_pos.drop_duplicates(subset=["gene_id"])
    gene_span = (
        gene_pos.groupby("gene_id").agg({"chr": "first", "start": "min", "end": "max"}).reset_index()
    )
    gene_span["window_start"] = (gene_span["start"] - 1_000_000).clip(lower=0)
    gene_span["window_end"] = gene_span["end"] + 1_000_000

    gencode = pd.read_table(args.gencode_class, low_memory=False)
    gene_type_map = gencode.set_index("gene_id_stable")["gene_type_grouped"].to_dict()
    lncrna_category = pd.read_table(args.lncrna_category_file)[
        ["gene_id_stable", "genomic_context"]
    ].drop_duplicates()
    lncrna_category_map = lncrna_category.set_index("gene_id_stable")[
        "genomic_context"
    ].to_dict()

    expanded_gene_types = PIP_LOCI_GENE_TYPES + [
        f"{scope}_{subtype}"
        for scope in ["all_lncRNA", "novel_lncRNA", "existing_lncRNA"]
        for subtype in LNCRNA_SUBTYPES
    ]

    esusie = load_susie_all_tissues(
        args.esusie_dir, "v11.eQTLs", gene_type_map, lncrna_category_map
    )
    ssusie = load_susie_all_tissues(
        args.ssusie_dir, "v11.sQTLs", gene_type_map, lncrna_category_map
    )
    esusie_group_sets = build_group_gene_sets(esusie)
    ssusie_group_sets = build_group_gene_sets(ssusie)

    genes_with_cs = {}
    for gt in expanded_gene_types:
        for st in SUSIE_TYPES:
            sets = esusie_group_sets if st == "eqtl" else ssusie_group_sets
            genes_with_cs[(gt, st)] = sets.get(gt, set())

    loci_windows = {
        (gt, st): build_loci_windows(genes, gene_span)
        for (gt, st), genes in genes_with_cs.items()
    }
    merged_with_loci = compute_loci_columns_fast(merged, loci_windows)
    merged_with_pip = compute_pip_columns(
        merged_with_loci, esusie, "eqtl", expanded_gene_types, esusie_group_sets
    )
    merged_with_pip = compute_pip_columns(
        merged_with_pip, ssusie, "sqtl", expanded_gene_types, ssusie_group_sets
    )

    reg_cols = [f"{cat}_d{ANNOT_SUFFIX}" for cat in REGULATORY_CATEGORIES]
    cons_cols = [
        f"{cat}_{gt}_d{ANNOT_SUFFIX}"
        for cat in CONSEQUENCE_CATEGORIES
        for gt in gencode_grouped_types
    ]
    expected_cols = (
        reg_cols
        + cons_cols
        + ["af_c", "chr", "pos"]
        + [f"{gt}_{st}_pip" for gt in expanded_gene_types for st in SUSIE_TYPES]
        + [f"{gt}_{st}_loci" for gt in expanded_gene_types for st in SUSIE_TYPES]
        + ["SNP"]
    )
    combined = merged_with_pip[[c for c in expected_cols if c in merged_with_pip.columns]]
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    combined.set_index("SNP").to_parquet(args.output_path)
    print(f"Wrote {args.output_path} with shape {combined.shape}")


if __name__ == "__main__":
    main()
