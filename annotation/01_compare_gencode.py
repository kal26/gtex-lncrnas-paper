#!/usr/bin/env python3
"""Compare GENCODE v39 vs v47 and write gene classification.

Gene-, transcript-, and exon-level change classes, biotype grouping, and exonic
ambiguity between the annotations used by GTEx v10 and v11.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr

GENE_COLS = [
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "gene_id",
    "gene_id_stable",
    "gene_type",
    "gene_type_grouped",
]
EXON_COLS = ["Chromosome", "Start", "End", "Strand", "gene_id_stable"]
TX_COLS = [
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "gene_id_stable",
    "transcript_id_stable",
]
EX_COLS = [
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "gene_id_stable",
    "exon_id_stable",
]


def group_gene_type(df: pd.DataFrame) -> pd.DataFrame:
    g = df["gene_type"]
    df["gene_type_grouped"] = np.where(
        g.isin(
            [
                "Mt_rRNA",
                "Mt_tRNA",
                "miRNA",
                "misc_RNA",
                "rRNA",
                "rRNA_pseudogene",
                "scRNA",
                "snRNA",
                "snoRNA",
                "ribozyme",
                "sRNA",
                "scaRNA",
                "vault_RNA",
                "vaultRNA",
            ]
        ),
        "sRNA",
        "None",
    )
    df["gene_type_grouped"] = np.where(
        g.isin(
            [
                "Mt_tRNA_pseudogene",
                "tRNA_pseudogene",
                "snoRNA_pseudogene",
                "snRNA_pseudogene",
                "scRNA_pseudogene",
                "noncoding_RNA_pseudogene",
                "misc_RNA_pseudogene",
                "miRNA_pseudogene",
                "transcribed_processed_pseudogene",
                "transcribed_unprocessed_pseudogene",
                "transcribed_unitary_pseudogene",
                "translated_processed_pseudogene",
                "translated_unprocessed_pseudogene",
                "unitary_pseudogene",
                "unprocessed_pseudogene",
                "pseudogene",
                "polymorphic_pseudogene",
                "processed_pseudogene",
                "IG_C_pseudogene",
                "IG_J_pseudogene",
                "IG_V_pseudogene",
                "IG_pseudogene",
                "TR_V_pseudogene",
                "TR_J_pseudogene",
            ]
        ),
        "pseudogene",
        df["gene_type_grouped"],
    )
    df["gene_type_grouped"] = np.where(
        g.isin(
            [
                "protein_coding",
                "IG_C_gene",
                "IG_D_gene",
                "IG_J_gene",
                "IG_V_gene",
                "TR_C_gene",
                "TR_J_gene",
                "TR_V_gene",
                "TR_D_gene",
            ]
        ),
        "protein_coding",
        df["gene_type_grouped"],
    )
    df["gene_type_grouped"] = np.where(
        g.isin(
            [
                "3prime_overlapping_ncRNA",
                "antisense",
                "bidirectional_promoter_lncRNA",
                "lincRNA",
                "macro_lncRNA",
                "non_coding",
                "processed_transcript",
                "sense_intronic",
                "sense_overlapping",
                "TEC",
                "lncRNA",
            ]
        ),
        "lncRNA",
        df["gene_type_grouped"],
    )
    df["gene_type_grouped"] = np.where(
        g.isin(["artifact"]), "artifact", df["gene_type_grouped"]
    )
    return df


def stranded_interval_set(
    df: pd.DataFrame,
    chr_col: str = "Chromosome",
    start_col: str = "Start",
    end_col: str = "End",
    strand_col: str = "Strand",
) -> set:
    return set(zip(df[chr_col], df[start_col], df[end_col], df[strand_col]))


def full_exon_bp_per_gene(gtf_df: pd.DataFrame) -> pd.Series:
    exons = gtf_df[gtf_df["Feature"] == "exon"][EXON_COLS].dropna(subset=["gene_id_stable"])
    merged = pr.PyRanges(exons).merge(by=["gene_id_stable"]).as_df()
    merged["bp"] = merged["End"] - merged["Start"]
    return merged.groupby("gene_id_stable")["bp"].sum()


def coll_exon_bp_per_gene(gtf_df: pd.DataFrame) -> pd.Series:
    exons = (
        gtf_df[gtf_df["Feature"] == "exon"][EXON_COLS]
        .dropna(subset=["gene_id_stable"])
        .copy()
    )
    exons["bp"] = exons["End"] - exons["Start"]
    return exons.groupby("gene_id_stable")["bp"].sum()


def merged_exon_intervals(gtf_df: pd.DataFrame) -> pd.DataFrame:
    exons = (
        gtf_df[gtf_df["Feature"] == "exon"][EXON_COLS]
        .dropna(subset=["gene_id_stable"])
        .copy()
    )
    return pr.PyRanges(exons).merge(by=["gene_id_stable"]).as_df()


def exon_turnover_v39_v47(gtf_v39: pd.DataFrame, gtf_v47: pd.DataFrame) -> pd.DataFrame:
    """Per-gene exon bp and symmetric difference between GENCODE releases."""
    m39 = merged_exon_intervals(gtf_v39).copy()
    m47 = merged_exon_intervals(gtf_v47).copy()
    m39["bp"] = m39["End"] - m39["Start"]
    m47["bp"] = m47["End"] - m47["Start"]
    bp39 = m39.groupby("gene_id_stable")["bp"].sum().rename("exon_bp_v39")
    bp47 = m47.groupby("gene_id_stable")["bp"].sum().rename("exon_bp_v47")

    joined = (
        pr.PyRanges(m39)
        .join(pr.PyRanges(m47), strandedness=False, suffix="_v47")
        .df
    )
    joined = joined[joined["gene_id_stable"] == joined["gene_id_stable_v47"]].copy()
    joined["overlap_bp"] = (
        joined[["End", "End_v47"]].min(axis=1) - joined[["Start", "Start_v47"]].max(axis=1)
    ).clip(lower=0)
    intersect = joined.groupby("gene_id_stable")["overlap_bp"].sum().rename("intersect_exon_bp")

    out = pd.concat([bp39, bp47, intersect], axis=1)
    out["exon_bp_v39"] = out["exon_bp_v39"].fillna(0).astype(np.int64)
    out["exon_bp_v47"] = out["exon_bp_v47"].fillna(0).astype(np.int64)
    out["intersect_exon_bp"] = out["intersect_exon_bp"].fillna(0).astype(np.int64)
    out["delta_exon_bp"] = (
        out["exon_bp_v39"] + out["exon_bp_v47"] - 2 * out["intersect_exon_bp"]
    ).astype(np.int64)
    out["pct_exon_turnover"] = np.where(
        out["exon_bp_v47"] > 0,
        100.0 * out["delta_exon_bp"] / out["exon_bp_v47"],
        np.nan,
    )
    return out.reset_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-v39", type=Path, required=True, help="GENCODE v39 full annotation GTF")
    p.add_argument("--full-v47", type=Path, required=True, help="GENCODE v47 full annotation GTF")
    p.add_argument(
        "--collapsed-v39", type=Path, required=True, help="GENCODE v39 collapsed genes GTF"
    )
    p.add_argument(
        "--collapsed-v47", type=Path, required=True, help="GENCODE v47 collapsed genes GTF"
    )
    p.add_argument(
        "--output-classification",
        type=Path,
        required=True,
        help="Output TSV: gencode_v39_v47_gene_classification.txt",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    full_v39 = pr.read_gtf(str(args.full_v39)).as_df()
    full_v47 = pr.read_gtf(str(args.full_v47)).as_df()
    for df in [full_v39, full_v47]:
        df["gene_id_stable"] = df["gene_id"].str.split(".").str[0]
        df["transcript_id_stable"] = df["transcript_id"].str.split(".").str[0]
        df["exon_id_stable"] = df["exon_id"].str.split(".").str[0]

    coll_v39 = pr.read_gtf(str(args.collapsed_v39)).as_df()
    coll_v47 = pr.read_gtf(str(args.collapsed_v47)).as_df()
    coll_v39["gene_id_stable"] = coll_v39["gene_id"].str.split(".").str[0]
    coll_v47["gene_id_stable"] = coll_v47["gene_id"].str.split(".").str[0]

    group_gene_type(full_v39)
    group_gene_type(full_v47)
    group_gene_type(coll_v39)
    group_gene_type(coll_v47)

    # --- Gene-level classification (full annotation) ---
    full_v39_gene = full_v39[full_v39["Feature"] == "gene"][GENE_COLS].drop_duplicates(
        "gene_id_stable"
    )
    full_v47_gene = full_v47[full_v47["Feature"] == "gene"][GENE_COLS].drop_duplicates(
        "gene_id_stable"
    )

    full_gene_merged = pd.merge(
        full_v47_gene.rename(
            columns={
                "gene_id": "gene_id_v47",
                "gene_type": "gene_type_v47",
                "gene_type_grouped": "gene_type_grouped_v47",
                "Chromosome": "chr_v47",
                "Start": "start_v47",
                "End": "end_v47",
                "Strand": "strand_v47",
            }
        ),
        full_v39_gene.rename(
            columns={
                "gene_id": "gene_id_v39",
                "gene_type": "gene_type_v39",
                "gene_type_grouped": "gene_type_grouped_v39",
                "Chromosome": "chr_v39",
                "Start": "start_v39",
                "End": "end_v39",
                "Strand": "strand_v39",
            }
        ),
        on="gene_id_stable",
        how="outer",
    )

    full_gene_merged["in_v47_full"] = ~full_gene_merged["gene_id_v47"].isna()
    full_gene_merged["in_v39_full"] = ~full_gene_merged["gene_id_v39"].isna()
    full_gene_merged["gene_type"] = np.where(
        full_gene_merged["gene_type_v47"].isna(),
        full_gene_merged["gene_type_v39"],
        full_gene_merged["gene_type_v47"],
    )
    full_gene_merged["gene_type_grouped"] = np.where(
        full_gene_merged["gene_type_grouped_v47"].isna(),
        full_gene_merged["gene_type_grouped_v39"],
        full_gene_merged["gene_type_grouped_v47"],
    )

    loc_changed = (
        (full_gene_merged["chr_v47"] != full_gene_merged["chr_v39"])
        | (full_gene_merged["start_v47"] != full_gene_merged["start_v39"])
        | (full_gene_merged["end_v47"] != full_gene_merged["end_v39"])
        | (full_gene_merged["strand_v47"] != full_gene_merged["strand_v39"])
    )
    full_gene_merged["full_gene_status"] = np.where(
        ~full_gene_merged["in_v47_full"],
        "removed_gene",
        np.where(
            ~full_gene_merged["in_v39_full"],
            "new_gene",
            np.where(loc_changed, "updated_gene", "no_change"),
        ),
    )

    v39_gene_coords = stranded_interval_set(full_v39_gene)
    v47_coord_cols = ["chr_v47", "start_v47", "end_v47", "strand_v47"]
    full_gene_merged["_coord_v47"] = list(zip(*[full_gene_merged[c] for c in v47_coord_cols]))
    reassigned_gene_mask = (full_gene_merged["full_gene_status"] == "new_gene") & full_gene_merged[
        "_coord_v47"
    ].isin(v39_gene_coords)
    full_gene_merged.loc[reassigned_gene_mask, "full_gene_status"] = "reassigned_gene"
    full_gene_merged = full_gene_merged.drop(columns="_coord_v47")
    print(full_gene_merged["full_gene_status"].value_counts())

    # --- Exon-level ambiguity (full vs collapsed exon bp) ---
    print("Computing exon bp totals …")
    full_bp_v47 = full_exon_bp_per_gene(full_v47).rename("full_exon_bp_v47")
    full_bp_v39 = full_exon_bp_per_gene(full_v39).rename("full_exon_bp_v39")
    coll_bp_v47 = coll_exon_bp_per_gene(coll_v47).rename("coll_exon_bp_v47")
    coll_bp_v39 = coll_exon_bp_per_gene(coll_v39).rename("coll_exon_bp_v39")

    ambiguity_df = pd.concat(
        [full_bp_v47, coll_bp_v47, full_bp_v39, coll_bp_v39],
        axis=1,
    ).reset_index()

    ambiguity_df["pct_ambiguous_v47"] = (
        (ambiguity_df["full_exon_bp_v47"] - ambiguity_df["coll_exon_bp_v47"].fillna(0))
        .div(ambiguity_df["full_exon_bp_v47"])
        .clip(0, 1)
    )
    ambiguity_df["pct_ambiguous_v39"] = (
        (ambiguity_df["full_exon_bp_v39"] - ambiguity_df["coll_exon_bp_v39"].fillna(0))
        .div(ambiguity_df["full_exon_bp_v39"])
        .clip(0, 1)
    )
    print(f"v47 any ambiguity: {(ambiguity_df['pct_ambiguous_v47'] > 0).sum():,}")
    print(f"v39 any ambiguity: {(ambiguity_df['pct_ambiguous_v39'] > 0).sum():,}")

    # --- Exon turnover ---
    print("Computing exon turnover (v39 vs v47) …")
    exon_turnover_df = exon_turnover_v39_v47(full_v39, full_v47)
    print(
        f"exon turnover genes: {len(exon_turnover_df):,}; "
        f"median pct_exon_turnover={np.nanmedian(exon_turnover_df['pct_exon_turnover']):.2f}"
    )

    # --- Transcript-level classification ---
    v39_tx = (
        full_v39[full_v39["Feature"] == "transcript"][TX_COLS]
        .dropna(subset=["transcript_id_stable"])
        .drop_duplicates(["gene_id_stable", "transcript_id_stable"])
    )
    v47_tx = (
        full_v47[full_v47["Feature"] == "transcript"][TX_COLS]
        .dropna(subset=["transcript_id_stable"])
        .drop_duplicates(["gene_id_stable", "transcript_id_stable"])
    )
    v39_tx_coords = stranded_interval_set(v39_tx)

    tx_merged = v47_tx.merge(
        v39_tx.rename(
            columns={
                "Chromosome": "chr_v39",
                "Start": "start_v39",
                "End": "end_v39",
                "Strand": "strand_v39",
            }
        )[["gene_id_stable", "transcript_id_stable", "chr_v39", "start_v39", "end_v39", "strand_v39"]],
        on=["gene_id_stable", "transcript_id_stable"],
        how="left",
    )
    tx_merged["tx_id_absent_from_v39"] = tx_merged["chr_v39"].isna()
    tx_merged["_coord"] = list(
        zip(tx_merged["Chromosome"], tx_merged["Start"], tx_merged["End"], tx_merged["Strand"])
    )
    tx_merged["tx_is_reassigned"] = (
        tx_merged["tx_id_absent_from_v39"] & tx_merged["_coord"].isin(v39_tx_coords)
    )
    tx_merged["tx_is_new"] = tx_merged["tx_id_absent_from_v39"] & ~tx_merged["tx_is_reassigned"]
    tx_merged["tx_location_changed"] = ~tx_merged["tx_id_absent_from_v39"] & (
        (tx_merged["Chromosome"] != tx_merged["chr_v39"])
        | (tx_merged["Start"] != tx_merged["start_v39"])
        | (tx_merged["End"] != tx_merged["end_v39"])
        | (tx_merged["Strand"] != tx_merged["strand_v39"])
    )
    tx_merged = tx_merged.drop(columns="_coord")

    gene_tx_stats = (
        tx_merged.groupby("gene_id_stable")
        .agg(
            has_new_tx=("tx_is_new", "any"),
            has_reassigned_tx=("tx_is_reassigned", "any"),
            all_tx_ids_new=("tx_id_absent_from_v39", "all"),
            has_updated_tx=("tx_location_changed", "any"),
        )
        .reset_index()
    )
    removed_tx_flags = (
        v39_tx[["gene_id_stable", "transcript_id_stable"]]
        .merge(
            v47_tx[["gene_id_stable", "transcript_id_stable"]],
            on=["gene_id_stable", "transcript_id_stable"],
            how="left",
            indicator=True,
        )
        .query('_merge == "left_only"')
        .groupby("gene_id_stable")
        .size()
        .reset_index(name="n_removed_tx")
        .assign(has_removed_tx=True)[["gene_id_stable", "has_removed_tx"]]
    )
    gene_tx_stats = gene_tx_stats.merge(removed_tx_flags, on="gene_id_stable", how="left")
    gene_tx_stats["has_removed_tx"] = gene_tx_stats["has_removed_tx"].fillna(False).astype(bool)
    gene_tx_stats["transcript_status"] = np.select(
        [
            gene_tx_stats["has_new_tx"],
            gene_tx_stats["has_reassigned_tx"],
            gene_tx_stats["has_updated_tx"],
            gene_tx_stats["has_removed_tx"],
        ],
        ["new_transcript", "reassigned_transcript", "updated_transcript", "removed_transcript"],
        default="no_change",
    )
    print(gene_tx_stats["transcript_status"].value_counts())

    # --- Exon-level classification ---
    v39_ex = (
        full_v39[full_v39["Feature"] == "exon"][EX_COLS]
        .dropna(subset=["exon_id_stable"])
        .drop_duplicates(["gene_id_stable", "exon_id_stable"])
    )
    v47_ex = (
        full_v47[full_v47["Feature"] == "exon"][EX_COLS]
        .dropna(subset=["exon_id_stable"])
        .drop_duplicates(["gene_id_stable", "exon_id_stable"])
    )
    v39_ex_coords = stranded_interval_set(v39_ex)

    ex_merged = v47_ex.merge(
        v39_ex.rename(
            columns={
                "Chromosome": "chr_v39",
                "Start": "start_v39",
                "End": "end_v39",
                "Strand": "strand_v39",
            }
        )[["gene_id_stable", "exon_id_stable", "chr_v39", "start_v39", "end_v39", "strand_v39"]],
        on=["gene_id_stable", "exon_id_stable"],
        how="left",
    )
    ex_merged["ex_id_absent_from_v39"] = ex_merged["chr_v39"].isna()
    ex_merged["_coord"] = list(
        zip(ex_merged["Chromosome"], ex_merged["Start"], ex_merged["End"], ex_merged["Strand"])
    )
    ex_merged["ex_is_reassigned"] = (
        ex_merged["ex_id_absent_from_v39"] & ex_merged["_coord"].isin(v39_ex_coords)
    )
    ex_merged["ex_is_new"] = ex_merged["ex_id_absent_from_v39"] & ~ex_merged["ex_is_reassigned"]
    ex_merged["ex_location_changed"] = ~ex_merged["ex_id_absent_from_v39"] & (
        (ex_merged["Chromosome"] != ex_merged["chr_v39"])
        | (ex_merged["Start"] != ex_merged["start_v39"])
        | (ex_merged["End"] != ex_merged["end_v39"])
        | (ex_merged["Strand"] != ex_merged["strand_v39"])
    )
    ex_merged = ex_merged.drop(columns="_coord")

    gene_ex_stats = (
        ex_merged.groupby("gene_id_stable")
        .agg(
            has_new_ex=("ex_is_new", "any"),
            has_reassigned_ex=("ex_is_reassigned", "any"),
            has_updated_ex=("ex_location_changed", "any"),
        )
        .reset_index()
    )
    removed_ex_flags = (
        v39_ex[["gene_id_stable", "exon_id_stable"]]
        .merge(
            v47_ex[["gene_id_stable", "exon_id_stable"]],
            on=["gene_id_stable", "exon_id_stable"],
            how="left",
            indicator=True,
        )
        .query('_merge == "left_only"')
        .groupby("gene_id_stable")
        .size()
        .reset_index(name="n_removed_ex")
        .assign(has_removed_ex=True)[["gene_id_stable", "has_removed_ex"]]
    )
    gene_ex_stats = gene_ex_stats.merge(removed_ex_flags, on="gene_id_stable", how="left")
    gene_ex_stats["has_removed_ex"] = gene_ex_stats["has_removed_ex"].fillna(False).astype(bool)
    gene_ex_stats["exon_status"] = np.select(
        [
            gene_ex_stats["has_new_ex"],
            gene_ex_stats["has_reassigned_ex"],
            gene_ex_stats["has_updated_ex"],
            gene_ex_stats["has_removed_ex"],
        ],
        ["new_exon", "reassigned_exon", "updated_exon", "removed_exon"],
        default="no_change",
    )
    print(gene_ex_stats["exon_status"].value_counts())

    # --- Merge classifications and assign change_level ---
    result = (
        full_gene_merged[
            [
                "gene_id_stable",
                "gene_id_v47",
                "gene_id_v39",
                "gene_type",
                "gene_type_grouped",
                "gene_type_v47",
                "in_v47_full",
                "in_v39_full",
                "full_gene_status",  # intermediate for change_level; dropped before write
            ]
        ]
        .merge(
            ambiguity_df[
                [
                    "gene_id_stable",
                    "pct_ambiguous_v47",
                    "pct_ambiguous_v39",
                ]
            ],
            on="gene_id_stable",
            how="left",
        )
        .merge(
            exon_turnover_df[["gene_id_stable", "pct_exon_turnover"]],
            on="gene_id_stable",
            how="left",
        )
        .merge(
            gene_tx_stats[
                [
                    "gene_id_stable",
                    "transcript_status",
                    "has_new_tx",
                    "has_reassigned_tx",
                    "all_tx_ids_new",
                ]
            ],
            on="gene_id_stable",
            how="left",
        )
        .merge(
            gene_ex_stats[
                [
                    "gene_id_stable",
                    "exon_status",
                    "has_new_ex",
                    "has_reassigned_ex",
                ]
            ],
            on="gene_id_stable",
            how="left",
        )
    )

    TX_CHANGE = {"updated_transcript", "removed_transcript", "reassigned_transcript"}
    EX_CHANGE = {"new_exon", "updated_exon", "removed_exon", "reassigned_exon"}

    for col in ["has_reassigned_tx", "has_reassigned_ex", "all_tx_ids_new"]:
        result[col] = result[col].fillna(True if col == "all_tx_ids_new" else False).astype(bool)

    def is_new_gene_id(row):
        return bool(row["in_v47_full"]) and not bool(row["in_v39_full"])

    def qualifies_reassigned_gene(row):
        if row["full_gene_status"] == "reassigned_gene":
            return True
        if not is_new_gene_id(row):
            return False
        tx_reassigned_case = bool(row.get("all_tx_ids_new", False)) and bool(
            row.get("has_reassigned_tx", False)
        )
        ex_reassigned_case = (row["full_gene_status"] == "new_gene") and bool(
            row.get("has_reassigned_ex", False)
        )
        return tx_reassigned_case or ex_reassigned_case

    def assign_change_level(row):
        if row["full_gene_status"] == "removed_gene":
            return "removed_gene"
        if qualifies_reassigned_gene(row):
            return "reassigned_gene"
        if is_new_gene_id(row) and row["full_gene_status"] == "new_gene":
            if not row.get("has_reassigned_tx", False) and not row.get("has_reassigned_ex", False):
                return "new_gene"
        # Methods five categories: merge transcript-level edits into modified_gene
        # (new transcript models on an existing stable gene ID, or any other tx/exon change).
        if not is_new_gene_id(row) and row.get("transcript_status") == "new_transcript":
            if not row.get("has_reassigned_ex", False):
                return "modified_gene"
        if row.get("transcript_status") in TX_CHANGE or row.get("exon_status") in EX_CHANGE:
            return "modified_gene"
        return "no_change"

    def assign_reassignment_subcase(row):
        if row["change_level"] != "reassigned_gene":
            return pd.NA
        if row["full_gene_status"] == "reassigned_gene":
            return "gene_coord_match"
        if bool(row.get("all_tx_ids_new", False)) and bool(row.get("has_reassigned_tx", False)):
            return "all_new_tx_some_reassigned_tx"
        if row["full_gene_status"] == "new_gene" and bool(row.get("has_reassigned_ex", False)):
            return "new_gene_new_coords_some_reassigned_ex"
        return "other_reassigned"

    result["change_level"] = result.apply(assign_change_level, axis=1)
    result["reassignment_subcase"] = result.apply(assign_reassignment_subcase, axis=1)

    # Public gene_type_grouped = coarse type with lncRNA split into novel/existing.
    # (Internal coarse column used ``lncRNA`` before this overwrite.)
    coarse = result["gene_type_grouped"]
    result["gene_type_grouped"] = np.where(
        coarse != "lncRNA",
        coarse,
        np.where(result["change_level"].eq("new_gene"), "novel_lncRNA", "existing_lncRNA"),
    )
    result = result.rename(columns={"gene_type": "gene_type_detail"})
    print(result["change_level"].value_counts())
    print(result.loc[result["change_level"] == "reassigned_gene", "reassignment_subcase"].value_counts())
    print(result["gene_type_grouped"].value_counts())

    # Drop intermediates used only to assign change_level / reassignment_subcase.
    result = result.drop(
        columns=[
            "has_new_tx",
            "has_reassigned_tx",
            "all_tx_ids_new",
            "has_new_ex",
            "has_reassigned_ex",
            "full_gene_status",
        ],
        errors="ignore",
    )

    # Stable column order.
    out_cols = [
        "gene_id_stable",
        "gene_id_v47",
        "gene_id_v39",
        "gene_type_detail",
        "gene_type_grouped",
        "gene_type_v47",
        "in_v47_full",
        "in_v39_full",
        "pct_ambiguous_v47",
        "pct_ambiguous_v39",
        "pct_exon_turnover",
        "transcript_status",
        "exon_status",
        "change_level",
        "reassignment_subcase",
    ]
    result = result[out_cols]

    # --- Write output ---
    args.output_classification.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_classification, sep="\t", index=False)
    print(f"Wrote {len(result):,} rows to {args.output_classification}")


if __name__ == "__main__":
    main()
