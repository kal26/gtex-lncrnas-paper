#!/usr/bin/env python3
"""
Gene vs matched-intergenic-null expression calls (Wilcoxon + multiple-testing flags).

One command writes the pooled cohort table and every per-tissue table for tissues
with n ≥ 10, using sample attributes to map samples → tissues.

Inputs:
  --sample_meta_file, --gene_matrix, --null_rnaseqc_dir
When SMAFRZE is present, only RNASEQ-freeze rows are kept.

Outputs (TSV):
  ``{output_dir}/{cohort}.null_quantifications.txt`` (pooled)
  ``{output_dir}/by_tissue/{cohort}/{Tissue}.null_quantifications.txt`` (per tissue)

Columns: ``gene_id``, ``p_value``, ``median_counts``, ``mean_counts``,
``median_null_counts``, ``mean_null_counts``, ``n_samples``,
``fdr05_quantified``, ``q_value``.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

FLAT_NULL_COLS = [
    "gene_id",
    "p_value",
    "median_counts",
    "mean_counts",
    "median_null_counts",
    "mean_null_counts",
    "n_samples",
    "fdr05_quantified",
    "q_value",
]


def null_quantifications_path(output_dir: Path, cohort_label: str) -> Path:
    """Pooled ``{cohort}.null_quantifications.txt``."""
    return Path(output_dir) / f"{cohort_label}.null_quantifications.txt"


def write_null_quantifications_table(df: pd.DataFrame, path: Path) -> None:
    """Write a flat-schema null-quantifications TSV."""
    cols = [c for c in FLAT_NULL_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[cols + extra].to_csv(path, sep="\t", index=False)
    print(f"Wrote {path} ({len(df)} genes)")


def _tissue_file_name(s: str) -> str:
    """Filename stem for per-tissue output tables."""
    return (
        s.replace("(", "")
        .replace(")", "")
        .replace(" - ", "_")
        .replace(" ", "_")
    )


def load_sample_meta(path: Path) -> pd.DataFrame:
    """Load sample attributes from TSV/TXT or Excel."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def sample_to_tissue_from_meta(
    meta: pd.DataFrame,
    sample_id_col: str,
    tissue_col: str,
) -> pd.Series:
    """Unique sample_id → tissue series from filtered metadata."""
    for col, label in ((sample_id_col, "--sample_id_col"), (tissue_col, "--tissue_col")):
        if col not in meta.columns:
            raise KeyError(f"{label} {col!r} not in sample metadata")
    sub = meta[[sample_id_col, tissue_col]].dropna()
    sub[sample_id_col] = sub[sample_id_col].astype(str)
    sub[tissue_col] = sub[tissue_col].astype(str)
    return sub.drop_duplicates(sample_id_col).set_index(sample_id_col)[tissue_col]


def fdr_bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (q), monotone in rank order."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    if n == 0:
        return np.zeros(0, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    # enforce monotonicity from the largest rank downward
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q_ranked
    return out


def add_fdr_quantified(
    df: pd.DataFrame,
    pval_col: str,
    count_col: str,
    new_col: str = "fdr05_quantified",
    q_col: str = "q_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """BH q-values + FDR flag on rows with count_col > 0."""
    df = df.copy()
    tested_mask = df[count_col] > 0
    df[new_col] = False
    df[q_col] = np.nan
    if tested_mask.any():
        pvals = df.loc[tested_mask, pval_col].to_numpy(dtype=float)
        qvals = fdr_bh_qvalues(pvals)
        df.loc[tested_mask, q_col] = qvals
        df.loc[tested_mask, new_col] = qvals <= alpha
    return df


def _read_sample_gct(file_path: Path) -> tuple[pd.DataFrame, str]:
    with gzip.open(file_path, "rt") as g:
        g.readline()
        g.readline()
        header = g.readline().strip().split("\t")
        sample = header[2]
        rows = []
        for line in g:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                rows.append({"Name": parts[0], "Description": parts[1], sample: parts[2]})
    return pd.DataFrame(rows), sample


def concat_sample_gcts(
    input_dir: Path,
    sample_ids: list[str],
    output_gct: Path,
) -> Path:
    """Merge per-sample *.gene_reads.gct.gz under input_dir/rnaseqc_null/."""
    rnaseqc_null = input_dir / "rnaseqc_null"
    gct_files = []
    for sid in sample_ids:
        p = rnaseqc_null / f"{sid}.gene_reads.gct.gz"
        if p.is_file():
            gct_files.append(p)
        else:
            print(f"Warning: missing {p}", file=sys.stderr)
    if not gct_files:
        raise FileNotFoundError(f"No GCT files under {rnaseqc_null}")

    print(f"Concatenating {len(gct_files)} null GCTs -> {output_gct}")
    dfs, samples = zip(*[_read_sample_gct(f) for f in sorted(gct_files)])
    merged = reduce(
        lambda x, y: x.merge(y, on=["Name", "Description"], how="outer"),
        dfs,
    ).fillna(0)
    merged = merged[["Name", "Description"] + list(samples)].sort_values("Name")

    output_gct.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_gct, "wt") as out:
        out.write(f"#1.2\n{len(merged)}\t{len(samples)}\n")
        out.write("Name\tDescription\t" + "\t".join(samples) + "\n")
        for _, row in merged.iterrows():
            out.write(
                "\t".join(
                    [str(row["Name"]), str(row["Description"])]
                    + [str(row[s]) for s in samples]
                )
                + "\n"
            )
    return output_gct


def load_expression_matrix(path: Path) -> pd.DataFrame:
    """Load gene x sample GCT (skip 2 header lines); index Name, columns sample IDs."""
    path = Path(path)
    df = pd.read_csv(path, sep="\t", skiprows=2)
    if "Name" not in df.columns:
        raise ValueError(f"{path} must have a Name column")
    df = df.set_index("Name")
    drop = [c for c in ("Description",) if c in df.columns]
    return df.drop(columns=drop)


def _wilcoxon_kwargs() -> dict:
    """Prefer asymptotic/approx Wilcoxon for speed on modest sample sizes."""
    try:
        stats.wilcoxon([1.0, 2.0], [0.0, 1.0], alternative="greater", method="asymptotic")
        return {"method": "asymptotic"}
    except TypeError:
        try:
            stats.wilcoxon([1.0, 2.0], [0.0, 1.0], alternative="greater", mode="approx")
            return {"mode": "approx"}
        except TypeError:
            return {}


def run_cohort_wilcoxon_on_matrices(
    gene_df: pd.DataFrame,
    null_df: pd.DataFrame,
    sample_cols: list[str],
    desc: str = "null quant",
) -> pd.DataFrame:
    """Wilcoxon on preloaded gene/null matrices for a sample subset."""
    common = gene_df.index.intersection(null_df.index)
    sample_cols = [c for c in sample_cols if c in gene_df.columns and c in null_df.columns]
    if not sample_cols:
        raise ValueError("No overlapping sample columns between gene and null matrices")
    gene_sub = gene_df.loc[common, sample_cols]
    null_sub = null_df.loc[common, sample_cols]
    kw = _wilcoxon_kwargs()

    results = []
    for gene_id in tqdm(common, desc=desc):
        gene_counts = pd.to_numeric(gene_sub.loc[gene_id], errors="coerce").to_numpy()
        null_counts = pd.to_numeric(null_sub.loc[gene_id], errors="coerce").to_numpy()
        mask = ~np.isnan(gene_counts) & ~np.isnan(null_counts)
        gene_counts = gene_counts[mask]
        null_counts = null_counts[mask]
        if len(gene_counts) == 0 or np.all(gene_counts == 0) or np.all(gene_counts == null_counts):
            p_value = 1.0
        else:
            try:
                _, p_value = stats.wilcoxon(
                    gene_counts, null_counts, alternative="greater", **kw
                )
            except ValueError:
                p_value = 1.0
        results.append(
            (
                gene_id,
                float(p_value),
                float(np.median(gene_counts)) if len(gene_counts) else 0.0,
                float(np.mean(gene_counts)) if len(gene_counts) else 0.0,
                float(np.median(null_counts)) if len(null_counts) else 0.0,
                float(np.mean(null_counts)) if len(null_counts) else 0.0,
                int(len(gene_counts)),
            )
        )

    out = pd.DataFrame(
        results,
        columns=[
            "gene_id",
            "p_value",
            "median_counts",
            "mean_counts",
            "median_null_counts",
            "mean_null_counts",
            "n_samples",
        ],
    )
    out = add_fdr_quantified(out, "p_value", "mean_counts", "fdr05_quantified")
    return out


def run_cohort_tissue_null_quantifications(
    gene_df: pd.DataFrame,
    null_df: pd.DataFrame,
    sample_tissue: pd.Series,
    output_dir: Path,
    cohort_label: str,
    min_n: int = 10,
) -> pd.DataFrame:
    """Per–Tissue-Site-Detail null quantifications for details with ≥ min_n samples."""
    tissue_root = Path(output_dir) / "by_tissue" / cohort_label
    tissue_root.mkdir(parents=True, exist_ok=True)
    n_by = sample_tissue.value_counts()
    eligible = n_by[n_by >= min_n].sort_index()
    print(f"Tissue details with n≥{min_n}: {len(eligible)} / {len(n_by)}")
    rows = []
    for tissue, n in eligible.items():
        stem = _tissue_file_name(str(tissue))
        out_path = tissue_root / f"{stem}.null_quantifications.txt"
        samples = sample_tissue.index[sample_tissue == tissue].tolist()
        samples = [s for s in samples if s in gene_df.columns and s in null_df.columns]
        if len(samples) < min_n:
            print(f"  skip {tissue}: only {len(samples)} overlapping samples")
            continue
        print(f"  {tissue}: n={len(samples)}")
        out = run_cohort_wilcoxon_on_matrices(
            gene_df,
            null_df,
            samples,
            desc=f"null quant {stem}",
        )
        write_null_quantifications_table(out, out_path)
        n_fdr = int(out["fdr05_quantified"].sum()) if "fdr05_quantified" in out.columns else 0
        rows.append(
            {
                "tissue_detail": tissue,
                "file_stem": stem,
                "n_samples": len(samples),
                "n_fdr05": n_fdr,
                "path": str(out_path),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = tissue_root / "tissues_used.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"Wrote tissue summary {summary_path}")
    return summary



def load_filtered_sample_to_tissue(
    sample_meta_file: Path,
    sample_id_col: str,
    tissue_col: str,
) -> pd.Series:
    meta = load_sample_meta(sample_meta_file)
    if "SMAFRZE" in meta.columns:
        meta = meta.loc[meta["SMAFRZE"].astype(str) == "RNASEQ"]
    sample_to_tissue = sample_to_tissue_from_meta(meta, sample_id_col, tissue_col)
    print(
        f"Sample metadata: {len(sample_to_tissue)} samples across "
        f"{sample_to_tissue.nunique()} tissues "
        f"(id={sample_id_col}, tissue={tissue_col})"
    )
    return sample_to_tissue


def run_null_expression(args) -> None:
    """Pooled call + per-tissue calls for every tissue with n ≥ min_n."""
    sample_to_tissue = load_filtered_sample_to_tissue(
        Path(args.sample_meta_file), args.sample_id_col, args.tissue_col
    )
    sample_ids = sample_to_tissue.index.astype(str).tolist()
    output_dir = Path(args.output_dir)

    combined = (
        Path(args.null_rnaseqc_dir)
        / "analysis"
        / f"{args.cohort_label}_null_gene_reads.gct.gz"
    )
    null_matrix = concat_sample_gcts(
        Path(args.null_rnaseqc_dir),
        sample_ids,
        combined,
    )

    gene_df = load_expression_matrix(Path(args.gene_matrix))
    null_df = load_expression_matrix(null_matrix)
    common_genes = gene_df.index.intersection(null_df.index)
    print(f"Genes: {len(gene_df)} gene, {len(null_df)} null, {len(common_genes)} matched")
    gene_df = gene_df.loc[common_genes]
    null_df = null_df.loc[common_genes]
    keep = [
        s
        for s in sample_to_tissue.index
        if s in gene_df.columns and s in null_df.columns
    ]
    sample_to_tissue = sample_to_tissue.loc[keep]
    print(f"Sample↔tissue map: {len(sample_to_tissue)} samples overlapping matrices")

    pooled_path = null_quantifications_path(output_dir, args.cohort_label)
    pooled = run_cohort_wilcoxon_on_matrices(
        gene_df,
        null_df,
        keep,
        desc="null quant pooled",
    )
    write_null_quantifications_table(pooled, pooled_path)

    run_cohort_tissue_null_quantifications(
        gene_df,
        null_df,
        sample_to_tissue,
        output_dir,
        cohort_label=args.cohort_label,
        min_n=10,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort_label",
        required=True,
        help="Cohort name used in output paths (GTEx, dGTEx, macaque, marmoset).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory. Writes {cohort}.null_quantifications.txt and "
        "by_tissue/{cohort}/{Tissue}.null_quantifications.txt.",
    )
    p.add_argument(
        "--sample_meta_file",
        required=True,
        help="Sample attributes table (TSV/TXT or Excel).",
    )
    p.add_argument(
        "--sample_id_col",
        default="SAMPID",
        help="Sample ID column (default: SAMPID; dGTEx/NHP portal tables use SAMPLE_ID).",
    )
    p.add_argument(
        "--tissue_col",
        default="SMTSD",
        help="Tissue / tissue-detail column (default: SMTSD).",
    )
    p.add_argument("--gene_matrix", required=True, help="Gene expression GCT (portal RNA-SeQC gene_reads).")
    p.add_argument(
        "--null_rnaseqc_dir",
        required=True,
        help="Directory with rnaseqc_null/{sample}.gene_reads.gct.gz from 01.",
    )
    args = p.parse_args()
    run_null_expression(args)


if __name__ == "__main__":
    main()
