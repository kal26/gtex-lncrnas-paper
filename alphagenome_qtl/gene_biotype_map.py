"""Map QTL phenotype IDs across GTEx v10/v11 gene builds and causality biotypes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def strip_ensembl_version(gene_id: str) -> str:
    return str(gene_id).split(".")[0]


def load_gene_classification(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def build_version_to_stable_maps(classification: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    v39_to_stable = {}
    v47_to_stable = {}
    for row in classification.itertuples(index=False):
        stable = str(row.gene_id_stable)
        if pd.notna(row.gene_id_v39):
            v39_to_stable[str(row.gene_id_v39)] = stable
            v39_to_stable[strip_ensembl_version(str(row.gene_id_v39))] = stable
        if pd.notna(row.gene_id_v47):
            v47_to_stable[str(row.gene_id_v47)] = stable
            v47_to_stable[strip_ensembl_version(str(row.gene_id_v47))] = stable
    return v39_to_stable, v47_to_stable


def phenotype_to_gene_stable(
    phenotype_id: str,
    *,
    version: str,
    v39_to_stable: dict[str, str],
    v47_to_stable: dict[str, str],
    gene_id: str | None = None,
) -> str | None:
    version = version.lower()
    pid = str(phenotype_id)
    if version == "v10":
        return v39_to_stable.get(pid) or v39_to_stable.get(strip_ensembl_version(pid))
    if version == "v11":
        mapped = v47_to_stable.get(pid) or v47_to_stable.get(strip_ensembl_version(pid))
        if mapped is not None:
            return mapped
        if gene_id is not None and pd.notna(gene_id):
            gid = str(gene_id)
            return v47_to_stable.get(gid) or v47_to_stable.get(strip_ensembl_version(gid))
    return None


def annotate_gene_stable(
    df: pd.DataFrame,
    *,
    version: str,
    gene_classification: Path,
) -> pd.DataFrame:
    classification = load_gene_classification(gene_classification)
    v39_to_stable, v47_to_stable = build_version_to_stable_maps(classification)
    out = df.copy()
    out["gene_id_stable"] = [
        phenotype_to_gene_stable(
            row["gene_key"],
            version=version,
            v39_to_stable=v39_to_stable,
            v47_to_stable=v47_to_stable,
            gene_id=row["gene_id"] if "gene_id" in out.columns else None,
        )
        for _, row in out.iterrows()
    ]
    return out


CAUSALITY_EVAL_BIOTYPES = ("lncRNA", "protein_coding", "pseudogene")
# Named multi-biotype filters for train/eval (exclude pseudogene for panel 3g).
CAUSALITY_BIOTYPE_SUBSETS: dict[str, tuple[str, ...]] = {
    "lncRNA_protein_coding": ("lncRNA", "protein_coding"),
}
CAUSALITY_BIOTYPE_FILTERS = CAUSALITY_EVAL_BIOTYPES + tuple(CAUSALITY_BIOTYPE_SUBSETS.keys())
_LNCRNA_SPLIT_TYPES = frozenset({"novel_lncRNA", "existing_lncRNA"})
# Matched-N transfer keeps the GENCODE split (novel vs existing lncRNA).
CAUSALITY_SPLIT_BIOTYPES = ("novel_lncRNA", "existing_lncRNA", "protein_coding")


def collapse_to_causality_biotype(gene_type_grouped: str | None) -> str | None:
    """Map GENCODE split biotypes to causality evaluation groups."""
    if gene_type_grouped is None or pd.isna(gene_type_grouped):
        return None
    value = str(gene_type_grouped)
    if value in _LNCRNA_SPLIT_TYPES:
        return "lncRNA"
    if value in CAUSALITY_EVAL_BIOTYPES:
        return value
    return None


def resolve_causality_biotype_filter(biotype_filter: str) -> tuple[str, ...]:
    """Expand a train/eval biotype filter name to one or more causality biotypes."""
    if biotype_filter in CAUSALITY_BIOTYPE_SUBSETS:
        return CAUSALITY_BIOTYPE_SUBSETS[biotype_filter]
    if biotype_filter in CAUSALITY_EVAL_BIOTYPES:
        return (biotype_filter,)
    raise ValueError(
        f"Unsupported biotype filter {biotype_filter!r}; "
        f"expected one of {CAUSALITY_BIOTYPE_FILTERS}."
    )


def causality_biotype_lookup(
    classification: pd.DataFrame | None = None,
    *,
    gene_classification: Path | None = None,
) -> dict[str, str]:
    """Stable gene ID -> pooled causality biotype (lncRNA / protein_coding / pseudogene)."""
    if classification is None:
        if gene_classification is None:
            raise ValueError("gene_classification path is required")
        classification = load_gene_classification(gene_classification)
    lookup: dict[str, str] = {}
    for row in classification.itertuples(index=False):
        stable = str(row.gene_id_stable)
        if pd.isna(row.gene_id_stable):
            continue
        biotype = collapse_to_causality_biotype(getattr(row, "gene_type_grouped", None))
        if biotype is not None:
            lookup[stable] = biotype
    return lookup


def annotate_causality_biotype(
    df: pd.DataFrame,
    *,
    gene_classification: Path,
    version: str = "v11",
) -> pd.DataFrame:
    """Add ``gene_biotype`` using ``gene_id_stable`` and GENCODE split-lncRNA labels."""
    out = df.copy()
    if "gene_id_stable" not in out.columns:
        out = annotate_gene_stable(out, version=version, gene_classification=gene_classification)
    lookup = causality_biotype_lookup(gene_classification=gene_classification)
    out["gene_biotype"] = out["gene_id_stable"].map(lookup)
    return out


def filter_to_causality_biotype(
    df: pd.DataFrame,
    biotype: str,
    *,
    gene_classification: Path,
    version: str = "v11",
) -> pd.DataFrame:
    """Filter examples to a single biotype or named multi-biotype subset."""
    allowed = resolve_causality_biotype_filter(biotype)
    annotated = annotate_causality_biotype(
        df, gene_classification=gene_classification, version=version
    )
    out = annotated[annotated["gene_biotype"].isin(allowed)].copy()
    return out.reset_index(drop=True)


def split_biotype_lookup(
    classification: pd.DataFrame | None = None,
    *,
    gene_classification: Path | None = None,
) -> dict[str, str]:
    """Stable gene ID -> novel_lncRNA / existing_lncRNA / protein_coding / pseudogene."""
    if classification is None:
        if gene_classification is None:
            raise ValueError("gene_classification path is required")
        classification = load_gene_classification(gene_classification)
    allowed = _LNCRNA_SPLIT_TYPES | set(CAUSALITY_EVAL_BIOTYPES)
    lookup: dict[str, str] = {}
    for row in classification.itertuples(index=False):
        if pd.isna(row.gene_id_stable):
            continue
        value = getattr(row, "gene_type_grouped", None)
        if value is None or pd.isna(value):
            continue
        value = str(value)
        if value in allowed:
            lookup[str(row.gene_id_stable)] = value
    return lookup


def annotate_split_biotype(
    df: pd.DataFrame,
    *,
    gene_classification: Path,
    version: str = "v11",
) -> pd.DataFrame:
    """Add ``split_biotype`` without collapsing novel/existing lncRNA."""
    out = df.copy()
    if "gene_id_stable" not in out.columns:
        out = annotate_gene_stable(out, version=version, gene_classification=gene_classification)
    lookup = split_biotype_lookup(gene_classification=gene_classification)
    out["split_biotype"] = out["gene_id_stable"].map(lookup)
    return out


def filter_to_split_biotype(
    df: pd.DataFrame,
    biotype: str,
    *,
    gene_classification: Path,
    version: str = "v11",
) -> pd.DataFrame:
    """Filter a cohort to one split biotype (novel/existing lncRNA or protein_coding)."""
    if biotype not in CAUSALITY_SPLIT_BIOTYPES:
        raise ValueError(
            f"Unsupported split biotype {biotype!r}; expected one of {CAUSALITY_SPLIT_BIOTYPES}."
        )
    annotated = annotate_split_biotype(
        df, gene_classification=gene_classification, version=version
    )
    out = annotated[annotated["split_biotype"] == biotype].copy()
    return out.reset_index(drop=True)


def transfer_source_rf_biotype(biotype: str) -> str:
    """Map a split transfer biotype to the existing per-biotype RF directory name."""
    if biotype in _LNCRNA_SPLIT_TYPES:
        return "lncRNA"
    if biotype in CAUSALITY_EVAL_BIOTYPES:
        return biotype
    raise ValueError(f"Unsupported transfer biotype {biotype!r}.")


def filter_cohort_to_causality_biotype(
    cohort: pd.DataFrame,
    biotype: str,
    *,
    gene_classification: Path,
    version: str = "v11",
) -> pd.DataFrame:
    """Filter a distance-balanced causality cohort to one evaluation biotype."""
    if cohort.empty:
        return cohort.copy()
    if biotype not in CAUSALITY_EVAL_BIOTYPES:
        raise ValueError(
            f"Unsupported biotype {biotype!r}; expected one of {CAUSALITY_EVAL_BIOTYPES}."
        )
    annotated = annotate_causality_biotype(
        cohort, gene_classification=gene_classification, version=version
    )
    out = annotated[annotated["gene_biotype"] == biotype].copy()
    return out.reset_index(drop=True)
