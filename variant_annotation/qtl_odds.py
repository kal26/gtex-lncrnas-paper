#!/usr/bin/env python

import argparse
import os
import re

import numpy as np
import pandas as pd

# Sentinel stored in output pip_cutoff column for CS-lead variants (highest PIP per credible set).
CS_MAX_PIP_CUTOFF = -1.0
ANNOT_VERSION = "v11"
# Paper PIP grids (Fig 3e / Supp 9).
EXISTING_LNCRNA_PIP_CUTOFFS = [0.9]
NEW_LNCRNA_PIP_CUTOFFS = [0.01, 0.5, 0.9, 0.95, 0.99, CS_MAX_PIP_CUTOFF]
DEFAULT_PIP_CUTOFFS = [0.01, 0.5, 0.9, 0.95]

_SPLICE_ACCEPTOR_V11 = re.compile(r"^splice_acceptor_variant_(.+)_d_v11$")

LNCRNA_SUBTYPES = [
    "bidirectional",
    "antisense_overlapping",
    "antisense_intronic",
    "sense_overlapping",
    "sense_intronic",
    "intergenic",
]

# Primary analysis biotypes for Firth gene_type CLI / EXPANDED_GENE_TYPES.
BASE_GENE_TYPES = ["novel_lncRNA", "existing_lncRNA", "protein_coding"]
# Biotypes with pip/loci columns (includes non-lncRNA sharers for specificity filters).
PIP_LOCI_GENE_TYPES = [
    "novel_lncRNA",
    "existing_lncRNA",
    "protein_coding",
    "pseudogene",
    "sRNA",
]
EXPANDED_GENE_TYPES = BASE_GENE_TYPES + [
    f"{scope}_{subtype}"
    for scope in ["all_lncRNA", "novel_lncRNA", "existing_lncRNA"]
    for subtype in LNCRNA_SUBTYPES
]

# Compound gene types derived from combinations of BASE_GENE_TYPES PIP columns.
# These do not have their own pip/loci columns in the parquet; they are computed
# at runtime from the base types' columns.
SPECIFICITY_GENE_TYPES = [
    "novel_lncRNA_specific",
    "existing_lncRNA_specific",
    "protein_coding_specific",
]
SHARED_GENE_TYPES = [
    "novel_lncRNA_shared_protein_coding",
    "existing_lncRNA_shared_protein_coding",
]
COMPOUND_GENE_TYPES = SPECIFICITY_GENE_TYPES + SHARED_GENE_TYPES

# All gene types accepted by the CLI.
ALL_GENE_TYPES = EXPANDED_GENE_TYPES + COMPOUND_GENE_TYPES

# Maps each compound gene type to its computation recipe.
# kind="specific": outcome = primary_pip > cutoff AND all(exclude_pip <= min_cutoff)
# kind="shared":   outcome = primary_pip > cutoff AND required_pip > cutoff
#
# Specificity for lncRNA types means "not shared with any non-lncRNA biotype":
# sharing with another lncRNA (new or existing) is acceptable, but sharing with
# protein_coding, pseudogene, or sRNA disqualifies the variant.
_COMPOUND_CONFIG: dict[str, dict] = {
    "novel_lncRNA_specific": {
        "kind": "specific",
        "primary": "novel_lncRNA",
        "exclude": ["protein_coding", "pseudogene", "sRNA"],
    },
    "existing_lncRNA_specific": {
        "kind": "specific",
        "primary": "existing_lncRNA",
        "exclude": ["protein_coding", "pseudogene", "sRNA"],
    },
    "protein_coding_specific": {
        "kind": "specific",
        "primary": "protein_coding",
        "exclude": ["novel_lncRNA", "existing_lncRNA", "pseudogene", "sRNA"],
    },
    "novel_lncRNA_shared_protein_coding": {
        "kind": "shared",
        "primary": "novel_lncRNA",
        "require": "protein_coding",
    },
    "existing_lncRNA_shared_protein_coding": {
        "kind": "shared",
        "primary": "existing_lncRNA",
        "require": "protein_coding",
    },
}


def load_susie_all_tissues(susie_dir, file_pattern, gene_type_map, lncrna_category_map):
    """Load every ``{tissue}.{file_pattern}.SuSiE_summary.parquet`` under ``susie_dir``.

    Tissue stems are discovered from filenames (portal SuSiE tarball has all 50).
    """
    suffix = f".{file_pattern}.SuSiE_summary.parquet"
    paths = sorted(
        os.path.join(susie_dir, name)
        for name in os.listdir(susie_dir)
        if name.endswith(suffix)
    )
    dfs = []
    for path in paths:
        tissue = os.path.basename(path)[: -len(suffix)]
        df = pd.read_parquet(path)
        df["tissue_id"] = tissue
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(
            f"No *{suffix} files under {susie_dir}"
        )
    out = pd.concat(dfs, ignore_index=True)
    out["gene_id_stable"] = out["phenotype_id"].str.split(".").str[0].str.split(":").str[-1]
    out["gene_type"] = out["gene_id_stable"].map(gene_type_map)
    out["lncrna_category"] = out["gene_id_stable"].map(lncrna_category_map)
    return out


def build_group_gene_sets(susie_df: pd.DataFrame):
    sets = {}
    for gt in PIP_LOCI_GENE_TYPES:
        sets[gt] = set(
            susie_df.loc[susie_df["gene_type"] == gt, "gene_id_stable"].dropna().unique()
        )
    for subtype in LNCRNA_SUBTYPES:
        subtype_mask = susie_df["lncrna_category"] == subtype
        sets[f"all_lncRNA_{subtype}"] = set(
            susie_df.loc[
                subtype_mask
                & susie_df["gene_type"].isin(["novel_lncRNA", "existing_lncRNA"]),
                "gene_id_stable",
            ]
            .dropna()
            .unique()
        )
        sets[f"novel_lncRNA_{subtype}"] = set(
            susie_df.loc[
                subtype_mask & (susie_df["gene_type"] == "novel_lncRNA"), "gene_id_stable"
            ]
            .dropna()
            .unique()
        )
        sets[f"existing_lncRNA_{subtype}"] = set(
            susie_df.loc[
                subtype_mask & (susie_df["gene_type"] == "existing_lncRNA"),
                "gene_id_stable",
            ]
            .dropna()
            .unique()
        )
    return sets


def cs_max_lead_variants(susie_df: pd.DataFrame, gene_ids: set) -> set:
    """Variant IDs with the highest PIP within each credible set (phenotype_id × cs_id)."""
    if susie_df.empty or not gene_ids:
        return set()
    sub = susie_df[susie_df["gene_id_stable"].isin(gene_ids)]
    if sub.empty:
        return set()
    idx = sub.groupby(["phenotype_id", "cs_id"], sort=False)["pip"].idxmax()
    return set(sub.loc[idx, "variant_id"].dropna().unique())


def discover_gene_type_grouped_suffixes(columns) -> list[str]:
    """Infer gene_type_grouped labels from consequence column names (v11 schema)."""
    groups = sorted(
        {m.group(1) for c in columns if (m := _SPLICE_ACCEPTOR_V11.match(str(c)))}
    )
    if not groups:
        raise ValueError(
            "Could not find columns matching splice_acceptor_variant_<group>_d_v11 "
            "(expected gene-split combined_annotations from 02_combine_odds_annotations.py)."
        )
    return groups


def add_merged_annotation_columns(combined_annotations: pd.DataFrame):
    # Discover annotation base names from columns ending in _d_v11
    # (regulatory: enhancer_d_v11; consequence: intron_variant_protein_coding_d_v11).
    annot_cols = [
        c.removesuffix("_v11")
        for c in combined_annotations.columns
        if c.endswith("_d_v11")
    ]

    grouped = discover_gene_type_grouped_suffixes(combined_annotations.columns)

    merged_cols = []
    for gt in grouped:
        splicing_col = f"splicing_merged_{gt}_d_v11"
        stop_col = f"stop_merged_{gt}_d_v11"

        splice_cols = [
            f"splice_acceptor_variant_{gt}_d_v11",
            f"splice_donor_variant_{gt}_d_v11",
            f"splice_region_variant_{gt}_d_v11",
        ]
        stop_cols = [
            f"frameshift_variant_{gt}_d_v11",
            f"stop_gained_{gt}_d_v11",
        ]

        available_splice = [c for c in splice_cols if c in combined_annotations.columns]
        available_stop = [c for c in stop_cols if c in combined_annotations.columns]

        if available_splice:
            combined_annotations[splicing_col] = (
                combined_annotations[available_splice].eq(1).any(axis=1)
            )
        if available_stop:
            combined_annotations[stop_col] = (
                combined_annotations[available_stop].eq(1).any(axis=1)
            )

        merged_cols.append(f"splicing_merged_{gt}_d")
        merged_cols.append(f"stop_merged_{gt}_d")

    all_annot_cols = annot_cols + merged_cols
    return combined_annotations, all_annot_cols


def fit_firth_annot_odds(
    df: pd.DataFrame,
    y,
    feature_cols,
    covars=None,
) -> pd.DataFrame:
    """Firth logistic OR for each binary annotation column: ``y ~ feature + covars``.

    ``feature_cols`` are full column names present in ``df`` (e.g. ``promoter_d_v11``).
    Fits on the full row set (no control subsampling). Returns one row per feature with
    ``annotation``, ``odds_ratio``, ``ci_lower``, ``ci_upper``, ``p_value``,
    ``n_case``, ``n_case_annot``, ``n_total``.
    """
    from firthmodels import FirthLogisticRegression

    if covars is None:
        covars = []
    y = np.asarray(y, dtype=float)
    if len(y) != len(df):
        raise ValueError(f"y length {len(y)} != df rows {len(df)}")

    rows = []
    n_case = int(y.sum())
    n_total = int(len(y))
    for feat in feature_cols:
        if feat not in df.columns:
            raise KeyError(f"Missing annotation column: {feat}")
        n_unique = int(df[feat].nunique(dropna=False))
        n_case_annot = int(((y == 1) & (df[feat].to_numpy() == 1)).sum())
        print(
            f"{feat}: {int(df[feat].sum())} "
            f"({100.0 * float(df[feat].mean()):.2f}%) annotated; "
            f"n_case_annot={n_case_annot}"
        )
        if n_unique < 2:
            rows.append(
                {
                    "annotation": feat,
                    "odds_ratio": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "p_value": float("nan"),
                    "n_case": n_case,
                    "n_case_annot": n_case_annot,
                    "n_total": n_total,
                }
            )
            print(f"{feat}: only one unique annotation value, skipping fit")
            continue

        X = df[[feat, *covars]].astype(float)
        model = FirthLogisticRegression(backend="numpy").fit(X, y)
        coef = float(model.coef_[0])
        se = float(model.bse_[0])
        odds_ratio = float(np.exp(coef))
        ci_low = float(np.exp(coef - 1.96 * se))
        ci_high = float(np.exp(coef + 1.96 * se))
        p_value = float(model.pvalues_[0])
        rows.append(
            {
                "annotation": feat,
                "odds_ratio": odds_ratio,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "p_value": p_value,
                "n_case": n_case,
                "n_case_annot": n_case_annot,
                "n_total": n_total,
            }
        )
        print(f"\t{feat}: {odds_ratio} ({ci_low}, {ci_high}), p={p_value}")
    return pd.DataFrame(rows)


def get_firth_odds(
    annotated_vars: pd.DataFrame,
    loci_col: str,
    annot_cols,
    annot_version: str = "v11",
    covars=None,
    pip_cutoffs=None,
    pip_col: str = None,
    y_builder=None,
    cs_max_lead_variants_set: set | None = None,
):
    """Compute Firth logistic odds ratios.

    Exactly one of ``pip_col`` or ``y_builder`` must be supplied.

    - ``pip_col``: standard path — outcome is ``loci_vars[pip_col] > pip_cutoff``.
    - ``y_builder``: compound path — callable ``(loci_vars, pip_cutoff) -> array``
      that returns a float array of 0/1 outcomes for each pip_cutoff.
    """
    if (pip_col is None) == (y_builder is None):
        raise ValueError("Exactly one of pip_col or y_builder must be provided.")

    if covars is None:
        covars = []
    if pip_cutoffs is None:
        pip_cutoffs = list(DEFAULT_PIP_CUTOFFS)

    loci_vars = annotated_vars[annotated_vars[loci_col]]
    feature_cols = [f"{annot}_{annot_version}" for annot in annot_cols]

    odds_results = []
    for pip_cutoff in pip_cutoffs:
        if pip_cutoff == CS_MAX_PIP_CUTOFF:
            if y_builder is not None:
                raise ValueError("CS-max lead outcome requires pip_col, not y_builder")
            if cs_max_lead_variants_set is None:
                raise ValueError("cs_max_lead_variants_set required for CS-max PIP cutoff")
            snp_ids = loci_vars.index if loci_vars.index.name == "SNP" else loci_vars["SNP"]
            y = np.asarray(snp_ids.isin(cs_max_lead_variants_set)).astype(float)
            label = "CS lead (max PIP per credible set)"
        elif y_builder is not None:
            y = y_builder(loci_vars, pip_cutoff)
            label = f"pip > {pip_cutoff}"
        else:
            y = np.asarray(loci_vars[pip_col] > pip_cutoff).astype(float)
            label = f"pip > {pip_cutoff}"
        print(f"At {label}, {y.sum():.0f} ({y.mean()*100:.2f}%) variants pass")
        fit_df = fit_firth_annot_odds(loci_vars, y, feature_cols, covars=covars)
        fit_df["pip_cutoff"] = pip_cutoff
        odds_results.append(fit_df)

    return pd.concat(odds_results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Firth logistic regression odds ratios for variant annotations "
            "using odds_annotations.parquet and PIP/loci columns."
        )
    )

    parser.add_argument(
        "--combined_annotations",
        required=True,
        help="Path to QTL+VEP parquet (odds_annotations.parquet from 02_combine_odds_annotations.py)",
    )
    parser.add_argument(
        "--gene_type",
        required=True,
        choices=ALL_GENE_TYPES,
        help="Gene type for this task.",
    )
    parser.add_argument(
        "--susie_type",
        required=True,
        choices=["eqtl", "sqtl"],
        help="SuSiE type for this task.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write odds results into",
    )
    parser.add_argument(
        "--esusie_dir",
        default=None,
        help="GTEx v11 eQTL SuSiE summary directory (required for novel_lncRNA).",
    )
    parser.add_argument(
        "--ssusie_dir",
        default=None,
        help="GTEx v11 sQTL SuSiE summary directory (required for novel_lncRNA).",
    )
    parser.add_argument(
        "--gencode_class",
        default=None,
        help="gencode_v39_v47_gene_classification.txt (required for novel_lncRNA).",
    )
    parser.add_argument(
        "--lncrna_category_file",
        default=None,
        help="lncrna_category.txt (required for novel_lncRNA).",
    )

    args = parser.parse_args()

    gene_type = args.gene_type
    susie_type = args.susie_type
    annot_version = ANNOT_VERSION

    print(
        f"Running odds for gene_type={gene_type}, susie_type={susie_type}, "
        f"annot_version={annot_version}"
    )

    if gene_type == "existing_lncRNA":
        pip_cutoffs = list(EXISTING_LNCRNA_PIP_CUTOFFS)
        include_cs_max_lead = False
    elif gene_type == "novel_lncRNA":
        pip_cutoffs = list(NEW_LNCRNA_PIP_CUTOFFS)
        include_cs_max_lead = True
    else:
        pip_cutoffs = list(DEFAULT_PIP_CUTOFFS)
        include_cs_max_lead = False

    if include_cs_max_lead:
        for req in ("esusie_dir", "ssusie_dir", "gencode_class", "lncrna_category_file"):
            if getattr(args, req) is None:
                raise ValueError(f"--{req} is required for gene_type={gene_type}")

    annotated = pd.read_parquet(args.combined_annotations)
    annotated, all_annot_cols = add_merged_annotation_columns(annotated)
    af_subset = annotated.dropna(subset=["af_c"])

    cs_max_lead_variants_set = None
    if include_cs_max_lead and gene_type not in _COMPOUND_CONFIG:
        gencode = pd.read_table(args.gencode_class, low_memory=False)
        gene_type_map = gencode.set_index("gene_id_stable")["gene_type_grouped"].to_dict()
        lncrna_category = pd.read_table(args.lncrna_category_file)[
            ["gene_id_stable", "genomic_context"]
        ].drop_duplicates()
        lncrna_category_map = lncrna_category.set_index("gene_id_stable")[
            "genomic_context"
        ].to_dict()
        susie_pattern = "v11.eQTLs" if susie_type == "eqtl" else "v11.sQTLs"
        susie_dir = args.esusie_dir if susie_type == "eqtl" else args.ssusie_dir
        susie_df = load_susie_all_tissues(
            susie_dir, susie_pattern, gene_type_map, lncrna_category_map
        )
        group_sets = build_group_gene_sets(susie_df)
        gene_ids = group_sets.get(gene_type, set())
        cs_max_lead_variants_set = cs_max_lead_variants(susie_df, gene_ids)

    output_file = os.path.join(
        args.output_dir,
        f"{gene_type}.{susie_type}_odds.txt",
    )

    # Build loci filter and outcome (y_builder) depending on gene type.
    if gene_type in _COMPOUND_CONFIG:
        cfg = _COMPOUND_CONFIG[gene_type]
        primary = cfg["primary"]
        loci_col = f"{primary}_{susie_type}_loci"

        min_pip_cutoff = min(pip_cutoffs)

        if cfg["kind"] == "specific":
            exclude_pip_cols = [f"{g}_{susie_type}_pip" for g in cfg["exclude"]]
            primary_pip_col = f"{primary}_{susie_type}_pip"

            def y_builder(loci_vars, pip_cutoff, _pcol=primary_pip_col, _ecols=exclude_pip_cols, _mincutoff=min_pip_cutoff):
                primary_passes = loci_vars[_pcol] > pip_cutoff
                others_excluded = pd.Series(True, index=loci_vars.index)
                for col in _ecols:
                    if col in loci_vars.columns:
                        others_excluded = others_excluded & (loci_vars[col] <= _mincutoff)
                return np.asarray(primary_passes & others_excluded).astype(float)

        elif cfg["kind"] == "shared":
            primary_pip_col = f"{primary}_{susie_type}_pip"
            required_pip_col = f"{cfg['require']}_{susie_type}_pip"

            def y_builder(loci_vars, pip_cutoff, _pcol=primary_pip_col, _rcol=required_pip_col):
                primary_passes = loci_vars[_pcol] > pip_cutoff
                required_passes = loci_vars[_rcol] > pip_cutoff
                return np.asarray(primary_passes & required_passes).astype(float)

        odds_df = get_firth_odds(
            af_subset,
            loci_col=loci_col,
            annot_cols=all_annot_cols,
            annot_version=annot_version,
            covars=["af_c"],
            pip_cutoffs=pip_cutoffs,
            y_builder=y_builder,
            cs_max_lead_variants_set=cs_max_lead_variants_set,
        )
    else:
        pip_col = f"{gene_type}_{susie_type}_pip"
        loci_col = f"{gene_type}_{susie_type}_loci"

        odds_df = get_firth_odds(
            af_subset,
            loci_col=loci_col,
            annot_cols=all_annot_cols,
            annot_version=annot_version,
            covars=["af_c"],
            pip_cutoffs=pip_cutoffs,
            pip_col=pip_col,
            cs_max_lead_variants_set=cs_max_lead_variants_set,
        )
    odds_df["gene_type"] = gene_type
    odds_df["susie_type"] = susie_type
    odds_df["annot_version"] = annot_version

    os.makedirs(args.output_dir, exist_ok=True)
    odds_df.to_csv(output_file, sep="\t", index=False)
    print(f"Wrote odds results to {output_file}")


if __name__ == "__main__":
    main()

