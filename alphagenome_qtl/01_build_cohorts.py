#!/usr/bin/env python3
"""Step 1 — build lead + distance-matched background cohorts (unscored).

Pass the data folders explicitly (see README). Writes under ``--output_dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def _default_metadata_name(dataset_version: str, cache_type: str) -> str:
    if cache_type == "allpairs_background":
        return f"{dataset_version}_allpairs_background_metadata.parquet"
    return f"{dataset_version}_susie_lead_metadata.parquet"

# --- step implementation ---

from pathlib import Path

import numpy as np
import pandas as pd

def _load_gtf_dataframe(gtf_path: Path) -> pd.DataFrame:
    """Load the AlphaGenome GENCODE feather built by ``00_build_gtf_feather.py``.

    Expects ``{gtf}.feather`` next to ``--full_gtf`` (0-based half-open
    ``Start``/``End`` plus ``Chromosome``/``Strand``/``Feature``/``gene_id``).
    """
    gtf_path = Path(gtf_path)
    feather = Path(f"{gtf_path}.feather")
    if not feather.is_file():
        raise FileNotFoundError(
            f"Missing AlphaGenome GTF feather {feather}; "
            "run 00_build_gtf_feather.py on the GENCODE GTF first."
        )
    return pd.read_feather(feather)

def _gene_key_series(frame: pd.DataFrame) -> pd.Series:
    gene_id = frame["gene_id"].astype(str).str.split(".").str[0]
    return frame["Chromosome"].astype(str) + ":" + gene_id

def load_median_tss_by_gene(gtf_path: Path) -> dict[str, int]:
    """Return gene_id (no version) -> median TSS across all transcripts of the gene.

    Per the paper methods, the per-gene TSS is the median of the transcription
    start sites of *all transcripts* of that gene (strand-aware), not the gene
    body boundary. AlphaGenome ``gene_annotation.extract_tss`` returns 0-based
    point intervals; we take the median over transcript features. Genes with no
    transcript rows use the gene-feature TSS (strand-aware gene start/end) so
    every gene still has a defined distance anchor.
    """
    from alphagenome.data import gene_annotation

    gtf = _load_gtf_dataframe(gtf_path)

    median_tss: dict[str, int] = {}
    gene_tss = gene_annotation.extract_tss(gtf, feature="gene")
    if not gene_tss.empty:
        for key, start in zip(_gene_key_series(gene_tss), gene_tss["Start"].astype(int)):
            median_tss[key] = int(start)

    transcript_tss = gene_annotation.extract_tss(gtf, feature="transcript")
    if not transcript_tss.empty:
        medians = (
            transcript_tss.assign(_gene_key=_gene_key_series(transcript_tss).values)
            .groupby("_gene_key")["Start"]
            .median()
        )
        for key, value in medians.items():
            median_tss[key] = int(value)
    return median_tss

def load_gene_bounds_by_gene(gtf_path: Path) -> dict[str, tuple[int, int]]:
    """Return chrom:gene_id -> (start, end) in AlphaGenome 0-based half-open coords.

    Uses the pyranges/feather ``gene`` rows directly, whose ``Start``/``End`` are
    already 0-based half-open (matching ``alphagenome.data.genome.Interval``).
    """
    gtf = _load_gtf_dataframe(gtf_path)
    genes = gtf[gtf["Feature"] == "gene"]
    if genes.empty:
        return {}
    grouped = (
        pd.DataFrame(
            {
                "_gene_key": _gene_key_series(genes).values,
                "start": genes["Start"].astype(int).values,
                "end": genes["End"].astype(int).values,
            }
        )
        .groupby("_gene_key")
        .agg(start=("start", "min"), end=("end", "max"))
    )
    return {key: (int(row.start), int(row.end)) for key, row in grouped.iterrows()}

def add_tss_distance(
    df: pd.DataFrame,
    *,
    gtf_path: Path,
    gene_key_column: str = "gene_key",
) -> pd.DataFrame:
    """Add absolute TSS distance; missing gene TSS is a hard error (no NaN keep).

    Uses median transcript TSS from the GENCODE annotation GTF (v39 for v10 /
    v47 for v11), not ``genes.gtf``.
    """
    from alphagenome.data import genome

    tss_map = load_median_tss_by_gene(gtf_path)
    distances: list[float] = []
    missing: list[str] = []
    for row in df.itertuples(index=False):
        variant = genome.Variant.from_str(str(row.variant_id), genome.VariantFormat.GTEX)
        gene_base = str(getattr(row, gene_key_column)).split(".")[0]
        tss_key = f"{variant.chromosome}:{gene_base}"
        tss = tss_map.get(tss_key)
        if tss is None:
            missing.append(tss_key)
            continue
        distances.append(float(abs(variant.position - tss)))
    if missing:
        uniq = sorted(set(missing))
        preview = ", ".join(uniq[:20])
        more = f" (+{len(uniq) - 20} more)" if len(uniq) > 20 else ""
        raise KeyError(
            f"No TSS for {len(uniq)} gene×chrom key(s) in {gtf_path}: {preview}{more}. "
            "Refusing silent NaN keep; extend GTF coverage or fix gene_key/chrom."
        )
    out = df.copy()
    out["tss_distance"] = distances
    out["log_tss_distance"] = np.log1p(out["tss_distance"].astype(float))
    return out

def assign_equal_width_bins(
    distances: np.ndarray,
    *,
    n_bins: int = 10,
) -> np.ndarray:
    """Assign equal-width bins on the distance scale (paper: 10 equal-width bins)."""
    valid = np.isfinite(distances)
    bins = np.full(len(distances), -1, dtype=int)
    if valid.sum() == 0:
        return bins

    values = distances[valid]
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if min_val == max_val:
        bins[valid] = 0
        return bins

    edges = np.linspace(min_val, max_val, int(n_bins) + 1)
    bins[valid] = np.digitize(values, edges[1:-1], right=False)
    return bins

def build_distance_balanced_cohort_from_pools(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    *,
    n_distance_bins: int = 10,
    distance_column: str = "log_tss_distance",
    seed: int = 42,
) -> pd.DataFrame:
    """Exact 50/50 distance balancing using separate positive and negative pools.

    Assigns 10 equal-width bins over the pooled log-TSS-distance distribution,
    then within each bin keeps ``min(n_pos, n_neg)`` of each class so the cohort
    is exactly class-balanced. Bins with no negatives (or no positives) are
    dropped.
    """
    if distance_column not in positives.columns or distance_column not in negatives.columns:
        raise ValueError(f"{distance_column} column required in both pools.")

    pos_work = positives[positives[distance_column].notna()].copy()
    neg_work = negatives[negatives[distance_column].notna()].copy()
    if pos_work.empty or neg_work.empty:
        return pd.DataFrame()

    pos_dist = pos_work[distance_column].to_numpy(dtype=float)
    neg_dist = neg_work[distance_column].to_numpy(dtype=float)
    combined_dist = np.concatenate([pos_dist, neg_dist])
    combined_bins = assign_equal_width_bins(combined_dist, n_bins=n_distance_bins)
    pos_work["distance_bin"] = combined_bins[: len(pos_work)]
    neg_work["distance_bin"] = combined_bins[len(pos_work) :]
    pos_work = pos_work[pos_work["distance_bin"] >= 0].copy()
    neg_work = neg_work[neg_work["distance_bin"] >= 0].copy()
    if pos_work.empty or neg_work.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for bin_id, pos_bin in pos_work.groupby("distance_bin"):
        neg_bin = neg_work[neg_work["distance_bin"] == bin_id]
        if neg_bin.empty:
            continue
        n_matched = min(len(pos_bin), len(neg_bin))
        if n_matched == 0:
            continue
        pos_sample = pos_bin.sample(
            n=n_matched,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).copy()
        pos_sample["causality_label"] = 1
        neg_sample = neg_bin.sample(
            n=n_matched,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).copy()
        neg_sample["causality_label"] = 0
        selected.append(pos_sample)
        selected.append(neg_sample)

    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)

def rebalance_cohort_exact(
    df: pd.DataFrame,
    *,
    label_column: str = "causality_label",
    bin_column: str = "distance_bin",
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample the majority class within each distance bin to exact 50/50."""
    if df.empty:
        return df
    work = df.reset_index(drop=True)
    labels = work[label_column].astype(int)
    if labels.nunique() < 2:
        return work
    if bin_column not in work.columns:
        work = work.copy()
        work[bin_column] = 0

    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for _, grp in work.groupby(bin_column):
        pos = grp[grp[label_column].astype(int) == 1]
        neg = grp[grp[label_column].astype(int) == 0]
        n_matched = min(len(pos), len(neg))
        if n_matched == 0:
            continue
        pos_sel = pos.sample(
            n=n_matched,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        neg_sel = neg.sample(
            n=n_matched,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        selected.extend([pos_sel, neg_sel])
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)

from util import (
    multimodal_feature_columns,
    normalize_chromosome,
)
from alphagenome.data import genome

from util import is_snv_variant_id

def collect_distance_matched_negative_ids(
    susie_examples: pd.DataFrame,
    background_pool: pd.DataFrame,
    *,
    gtf_path: Path,
    high_pip_threshold: float = 0.9,
    n_distance_bins: int = 10,
    seed: int = 42,
) -> set[str]:
    """Return example_ids of negatives to GPU-score for the background cache.

    One genome-wide distance-matched negative per high-PIP positive per tissue.
    Train/eval re-match from this scored pool at runtime, so split/CV statistics
    are unchanged while GPU work stays ~1:1 with leads.
    """
    cohort = build_causality_cohort(
        susie_examples,
        gtf_path=gtf_path,
        background_examples=background_pool,
        split=None,
        chromosomes=None,
        high_pip_threshold=high_pip_threshold,
        n_distance_bins=n_distance_bins,
        seed=seed,
        require_multimodal_features=False,
    )
    if cohort.empty:
        return set()
    return set(
        cohort.loc[cohort["causality_label"] == 0, "example_id"].astype(str).tolist()
    )

def _filter_examples(
    examples: pd.DataFrame,
    *,
    split: str | None = None,
    chromosomes: set[str] | None = None,
) -> pd.DataFrame:
    df = examples.copy()
    if split is not None:
        df = df[df["split"] == split].copy()
    if chromosomes is not None:
        chromosomes = {normalize_chromosome(c) for c in chromosomes}
        variant_chrom = df["variant_id"].map(
            lambda v: genome.Variant.from_str(v, genome.VariantFormat.GTEX).chromosome
        )
        df = df[variant_chrom.map(normalize_chromosome).isin(chromosomes)]
    return df

def _apply_multimodal_filter(df: pd.DataFrame, *, require_multimodal_features: bool) -> pd.DataFrame:
    if not require_multimodal_features:
        return df
    feature_cols = multimodal_feature_columns(df)
    if not feature_cols:
        raise ValueError("No multimodal feature columns (mm_*) found.")
    filtered = df[df[feature_cols].notna().any(axis=1)].copy()
    if filtered.empty:
        return filtered
    return filtered

def _ensure_tss_distance(df: pd.DataFrame, *, gtf_path: Path) -> pd.DataFrame:
    # Positives carry no precomputed distance; background metadata already stores
    # GTF median-transcript TSS distance (see build_background_cohort). Both paths
    # therefore use the same definition, so trust stored values and only compute
    # when absent.
    if "log_tss_distance" in df.columns and df["log_tss_distance"].notna().any():
        out = df.copy()
        if "tss_distance" not in out.columns:
            out["tss_distance"] = np.expm1(out["log_tss_distance"].astype(float))
        return out
    return add_tss_distance(df, gtf_path=gtf_path)

def can_build_causality_train_cohort(
    examples: pd.DataFrame,
    background_examples: pd.DataFrame,
    *,
    split: str = "train",
    high_pip_threshold: float = 0.9,
    require_multimodal_features: bool = True,
) -> bool:
    """Fast pre-check for train split without loading GTF or distance balancing."""
    if background_examples is None or background_examples.empty:
        return False

    susie_df = _filter_examples(examples, split=split)
    if susie_df.empty:
        return False
    susie_df = susie_df[susie_df["variant_id"].map(is_snv_variant_id)].copy()
    if susie_df.empty:
        return False

    positives = susie_df[susie_df["pip"] >= high_pip_threshold].copy()
    if positives.empty:
        return False

    negatives = _filter_examples(background_examples, split=split)
    if not negatives.empty:
        negatives = negatives[negatives["variant_id"].map(is_snv_variant_id)].copy()
    if negatives.empty:
        return False

    if require_multimodal_features:
        positives = _apply_multimodal_filter(
            positives,
            require_multimodal_features=True,
        )
        negatives = _apply_multimodal_filter(
            negatives,
            require_multimodal_features=True,
        )
    return not positives.empty and not negatives.empty

def build_causality_cohort(
    examples: pd.DataFrame,
    *,
    gtf_path: Path,
    background_examples: pd.DataFrame,
    split: str | None = None,
    chromosomes: set[str] | None = None,
    high_pip_threshold: float = 0.9,
    n_distance_bins: int = 10,
    seed: int = 42,
    require_multimodal_features: bool = False,
    score_column: str | None = None,
) -> pd.DataFrame:
    """Return a distance-balanced causal/non-causal cohort with causality_label.

    Positives: high-PIP SuSiE variants (default PIP >= 0.9).
    Negatives: cis-tested allpairs SNVs excluding SuSiE 95% credible-set hits.
    Distance matching uses a single global pass over 10 equal-width log-distance
    bins, keeping ``min(n_pos, n_neg)`` of each class within each bin so the
    cohort is exactly class-balanced. Bins with no negatives are dropped. Tissue
    is used only for the downstream tissue-weighted metric, not for balancing.
    """
    if background_examples is None or background_examples.empty:
        raise ValueError("background_examples is required for causality cohort construction.")

    susie_df = _filter_examples(examples, split=split, chromosomes=chromosomes)
    if susie_df.empty:
        return susie_df

    susie_df = susie_df[susie_df["variant_id"].map(is_snv_variant_id)].copy()
    if susie_df.empty:
        return susie_df

    positives = susie_df[susie_df["pip"] >= high_pip_threshold].copy()
    if positives.empty:
        return positives

    negatives = _filter_examples(background_examples, split=split, chromosomes=chromosomes)
    if not negatives.empty:
        negatives = negatives[negatives["variant_id"].map(is_snv_variant_id)].copy()

    if score_column:
        if (
            score_column not in negatives.columns
            and score_column == "predicted_beta"
            and "zero_shot_score" in negatives.columns
        ):
            negatives = negatives.copy()
            negatives[score_column] = negatives["zero_shot_score"].astype(float)
        if score_column in negatives.columns:
            negatives = negatives[negatives[score_column].notna()].copy()
        if score_column in positives.columns:
            positives = positives[positives[score_column].notna()].copy()

    if positives.empty or negatives.empty:
        return pd.DataFrame()

    positives = _ensure_tss_distance(positives, gtf_path=gtf_path)
    negatives = _ensure_tss_distance(negatives, gtf_path=gtf_path)

    positives = _apply_multimodal_filter(
        positives,
        require_multimodal_features=require_multimodal_features,
    )
    negatives = _apply_multimodal_filter(
        negatives,
        require_multimodal_features=require_multimodal_features,
    )
    if positives.empty or negatives.empty:
        return pd.DataFrame()

    cohort = build_distance_balanced_cohort_from_pools(
        positives,
        negatives,
        n_distance_bins=n_distance_bins,
        seed=seed,
    )
    # SpliceJunctionScorer uses gene masks for all GTF biotypes, so
    # lncRNA/pseudogene junction scores are kept for RF training.
    return cohort.reset_index(drop=True)

def summarize_causality_pairs(pairs: pd.DataFrame) -> dict[str, int | float]:
    if pairs.empty:
        return {"n_rows": 0, "n_positives": 0, "n_negatives": 0, "class_balance": float("nan")}
    labels = pairs["causality_label"].astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = n_pos + n_neg
    return {
        "n_rows": int(len(pairs)),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "class_balance": float(n_pos / total) if total else float("nan"),
    }

import pyarrow.parquet as pq

from gene_biotype_map import annotate_gene_stable

from util import DEFAULT_CHR_SPLIT, ChromosomeSplit, normalize_chromosome

def _load_susie_summary(path: Path, tissue: str, qtl_type: str) -> pd.DataFrame:
    table = pq.read_table(path)
    df = table.to_pandas()
    required = {"phenotype_id", "variant_id", "pip", "cs_id", "cs_size", "af"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SuSiE summary missing columns {sorted(missing)} in {path}")

    keep = [
        "phenotype_id",
        "variant_id",
        "pip",
        "cs_id",
        "cs_size",
        "af",
        "gene_name",
        "biotype",
    ]
    if "gene_id" in df.columns:
        keep.append("gene_id")
    if "afc" in df.columns:
        keep.extend(["afc", "afc_se"])

    df = df[[c for c in keep if c in df.columns]].copy()
    df["tissue"] = tissue
    df["qtl_type"] = qtl_type.lower()
    df["gene_key"] = df["phenotype_id"]
    df["chromosome"] = df["variant_id"].map(
        lambda v: genome.Variant.from_str(v, genome.VariantFormat.GTEX).chromosome
    )
    df["locus_id"] = (
        df["tissue"]
        + ":"
        + df["qtl_type"]
        + ":"
        + df["phenotype_id"]
        + ":cs"
        + df["cs_id"].astype(str)
    )
    df["pip"] = df["pip"].astype(float).clip(0.0, 1.0)
    if "afc" not in df.columns:
        raise ValueError(
            f"SuSiE summary {path} missing 'afc' column required for susie_beta target."
        )
    df["susie_beta"] = df["afc"].astype(float)
    return df

def filter_snv_variants(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["variant_id"].map(is_snv_variant_id)].copy()

def assign_locus_splits(
    df: pd.DataFrame,
    split: ChromosomeSplit = DEFAULT_CHR_SPLIT,
    *,
    drop_unassigned: bool = True,
) -> pd.DataFrame:
    lead_rows = df.loc[df.groupby("locus_id")["pip"].idxmax()]
    locus_chrom = lead_rows.set_index("locus_id")["chromosome"].map(normalize_chromosome)
    locus_split = locus_chrom.map(split.assign)
    out = df.copy()
    out["split"] = out["locus_id"].map(locus_split)
    if drop_unassigned:
        return out[out["split"].notna()].reset_index(drop=True)
    return out.reset_index(drop=True)

def filter_chromosomes(df: pd.DataFrame, chromosomes: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
    if not chromosomes or df.empty:
        return df
    allowed = {normalize_chromosome(c) for c in chromosomes}
    chrom = df["chromosome"].map(normalize_chromosome)
    return df[chrom.isin(allowed)].copy()

def build_examples_for_tissue_qtl(
    *,
    version: str,
    susie_dir: Path,
    tissue: str,
    qtl_type: str,
    split: ChromosomeSplit = DEFAULT_CHR_SPLIT,
    min_variants_per_locus: int = 1,
    min_pip: float | None = None,
    drop_unassigned_splits: bool = True,
    chromosomes: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    qtl_label = "eQTL" if qtl_type.lower() == "eqtl" else "sQTL"
    susie_path = Path(susie_dir) / f"{tissue}.{version}.{qtl_label}s.SuSiE_summary.parquet"
    if not susie_path.exists():
        raise FileNotFoundError(susie_path)

    df = _load_susie_summary(susie_path, tissue=tissue, qtl_type=qtl_type)
    if chromosomes is not None:
        df = filter_chromosomes(df, chromosomes)
        if df.empty:
            return df
    df = filter_snv_variants(df)
    locus_sizes = df.groupby("locus_id").size()
    keep_loci = locus_sizes[locus_sizes >= min_variants_per_locus].index
    df = df[df["locus_id"].isin(keep_loci)].copy()

    df = assign_locus_splits(
        df,
        split=split,
        drop_unassigned=drop_unassigned_splits,
    )
    if df.empty:
        return df

    if min_pip is not None:
        df = df[df["pip"] >= float(min_pip)].copy()
        if df.empty:
            return df

    df["example_id"] = df["locus_id"] + ":" + df["variant_id"]
    return df.drop_duplicates(subset=["example_id"]).reset_index(drop=True)

def add_interval_columns(
    df: pd.DataFrame,
    window_size: int,
    *,
    gtf_path: Path | None = None,
) -> pd.DataFrame:
    """Build fixed-width intervals centered on the variant (and target gene when possible).

    AlphaGenome variant scoring expects a **fixed 1 Mb** context (``window_size``).
    Expanding beyond that to fit large gene bodies causes ref/alt tensor shape
    mismatches inside the model. GeneMask only scores genes fully contained in the
    interval; distal QTLs may miss GeneMask RNA but still receive CenterMask scores.
    """
    gene_bounds = load_gene_bounds_by_gene(gtf_path) if gtf_path is not None else {}
    length = int(window_size)
    starts: list[int] = []
    ends: list[int] = []
    for row in df.itertuples(index=False):
        variant = genome.Variant.from_str(str(row.variant_id), genome.VariantFormat.GTEX)
        # AlphaGenome Interval/Variant: 0-based half-open; GTEx variant IDs are 1-based.
        var_pos = variant.position - 1
        left = var_pos
        right = var_pos + max(
            len(variant.reference_bases), len(variant.alternate_bases), 1
        )
        gene_base = None
        if hasattr(row, "gene_id_stable") and pd.notna(row.gene_id_stable):
            gene_base = str(row.gene_id_stable)
        elif hasattr(row, "gene_key"):
            gene_base = str(row.gene_key).split(".")[0]
        if gene_base and gene_bounds:
            bounds = gene_bounds.get(f"{row.chromosome}:{gene_base}")
            if bounds is not None:
                left = min(left, bounds[0])
                right = max(right, bounds[1])
        center = (left + right) // 2
        start = max(0, center - length // 2)
        end = start + length
        starts.append(int(start))
        ends.append(int(end))
    out = df.copy()
    out["window_start"] = starts
    out["window_end"] = ends
    return out

def build_examples(
    version: str,
    tissues: list[str],
    qtl_types: list[str],
    *,
    susie_dir: Path,
    gene_classification: Path,
    gtf_path: Path | None = None,
    window_size: int = 1_048_576,
    seed: int = 42,
    min_variants_per_locus: int = 1,
    min_pip: float | None = None,
    drop_unassigned_splits: bool = True,
    chromosomes: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    _ = seed
    frames = []
    for tissue in tissues:
        for qtl_type in qtl_types:
            frame = build_examples_for_tissue_qtl(
                version=version,
                susie_dir=susie_dir,
                tissue=tissue,
                qtl_type=qtl_type,
                min_variants_per_locus=min_variants_per_locus,
                min_pip=min_pip,
                drop_unassigned_splits=drop_unassigned_splits,
                chromosomes=chromosomes,
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise RuntimeError("No fine-mapping examples were built; check SuSiE paths.")
    merged = pd.concat(frames, ignore_index=True)
    merged = annotate_gene_stable(
        merged, version=version, gene_classification=gene_classification
    )
    merged = add_interval_columns(merged, window_size=window_size, gtf_path=gtf_path)
    merged["dataset_version"] = version
    return merged

# --- Metadata I/O (shared with background cohort builder) ---

from util import load_examples_parquet, save_examples_parquet

def _chromosomes_in_metadata(examples: pd.DataFrame) -> list[str]:
    if "chromosome" in examples.columns:
        chroms = examples["chromosome"].map(normalize_chromosome)
    else:
        chroms = (
            examples["variant_id"]
            .map(lambda v: genome.Variant.from_str(v, genome.VariantFormat.GTEX).chromosome)
            .map(normalize_chromosome)
        )
    return sorted({str(c) for c in chroms.dropna().unique()})

def summarize_metadata(
    examples: pd.DataFrame,
    *,
    dataset_version: str,
    cache_type: str,
    metadata_path: Path,
) -> dict:
    summary = {
        "dataset_version": dataset_version,
        "cache_type": cache_type,
        "metadata_path": str(metadata_path),
        "num_examples": int(len(examples)),
        "num_loci": int(examples["locus_id"].nunique()) if "locus_id" in examples.columns else None,
        "chromosomes_in_metadata": _chromosomes_in_metadata(examples),
    }
    if "tissue" in examples.columns:
        summary["num_tissues"] = int(examples["tissue"].nunique())
        summary["tissues"] = sorted(examples["tissue"].astype(str).unique())
    return summary

def build_lead_variant_examples(
    version: str,
    tissues: list[str],
    qtl_types: list[str],
    *,
    susie_dir: Path,
    gene_classification: Path,
    gtf_path: Path,
    window_size: int = 1_048_576,
    seed: int = 42,
    min_variants_per_locus: int = 1,
    min_pip: float = 0.9,
    chromosomes: list[str] | None = None,
) -> pd.DataFrame:
    return build_examples(
        version=version,
        tissues=tissues,
        qtl_types=qtl_types,
        susie_dir=susie_dir,
        gene_classification=gene_classification,
        gtf_path=gtf_path,
        window_size=window_size,
        seed=seed,
        min_variants_per_locus=min_variants_per_locus,
        min_pip=min_pip,
        drop_unassigned_splits=False,
        chromosomes=chromosomes,
    )

def build_susie_lead_metadata(
    *,
    version: str,
    metadata_path: Path,
    tissues: list[str],
    qtl_types: list[str],
    susie_dir: Path,
    gene_classification: Path,
    gtf_path: Path,
    window_size: int = 1_048_576,
    seed: int = 42,
    min_variants_per_locus: int = 1,
    min_pip: float = 0.9,
) -> dict:
    metadata_path = Path(metadata_path)
    examples = build_lead_variant_examples(
        version,
        tissues,
        qtl_types,
        susie_dir=susie_dir,
        gene_classification=gene_classification,
        gtf_path=gtf_path,
        window_size=window_size,
        seed=seed,
        min_variants_per_locus=min_variants_per_locus,
        min_pip=min_pip,
    )
    save_examples_parquet(examples, metadata_path)

    return summarize_metadata(
        examples,
        dataset_version=version,
        cache_type="susie_lead_variant",
        metadata_path=metadata_path,
    )

import json

from util import DEFAULT_CHR_SPLIT, add_score_keys, normalize_chromosome, save_examples_parquet

CAUSALITY_CHROMOSOMES = tuple(f"chr{i}" for i in range(1, 23))
WINDOW_SIZE = 1_048_576
SEED = 42
MIN_VARIANTS_PER_LOCUS = 1
MIN_PIP = 0.9
MAX_VARIANTS_PER_GENE = 50
HIGH_PIP_THRESHOLD = 0.9
N_DISTANCE_BINS = 10
QTL_TYPES = ["eqtl"]

def load_susie_credible_set_variant_keys(susie_path: Path) -> set[tuple[str, str]]:
    """Return all (phenotype_id, variant_id) pairs in SuSiE 95% credible sets."""
    table = pq.read_table(susie_path, columns=["phenotype_id", "variant_id"])
    df = table.to_pandas()
    return {(str(g), str(v)) for g, v in zip(df["phenotype_id"], df["variant_id"])}

def _cap_variants_per_gene(
    df: pd.DataFrame,
    *,
    max_variants_per_gene: int,
    seed: int,
    group_column: str = "gene_key",
) -> pd.DataFrame:
    if max_variants_per_gene <= 0:
        return df.iloc[0:0].copy()
    parts: list[pd.DataFrame] = []
    for gene_id, sub in df.groupby(group_column, sort=False):
        if len(sub) <= max_variants_per_gene:
            parts.append(sub)
            continue
        parts.append(
            sub.sample(n=int(max_variants_per_gene), random_state=seed + hash(gene_id) % 10_000)
        )
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)

def _load_allpairs_for_chromosome(
    allpairs_path: Path,
    *,
    exclude_keys: set[tuple[str, str]],
    max_variants_per_gene: int,
    seed: int,
) -> pd.DataFrame:
    from util import is_snv_variant_id

    df = pq.read_table(allpairs_path).to_pandas()
    # The allpairs file is already exactly the set of cis-tested variant-gene
    # pairs, so negatives are simply those minus SuSiE credible-set members. Cap
    # per gene first to bound the per-row SNV/credible-set work and peak memory
    # (genes are chromosome-specific, so a per-chromosome cap == a global cap).
    df = _cap_variants_per_gene(
        df,
        max_variants_per_gene=max_variants_per_gene,
        seed=seed,
        group_column="gene_id",
    )
    if df.empty:
        return df
    df = df[df["variant_id"].map(is_snv_variant_id)].copy()
    if df.empty:
        return df
    keys = list(zip(df["gene_id"].astype(str), df["variant_id"].astype(str)))
    keep = [key not in exclude_keys for key in keys]
    return df.loc[keep].copy()

def build_allpairs_background_for_tissue_qtl(
    version: str,
    tissue: str,
    qtl_type: str,
    *,
    susie_dir: Path,
    allpairs_dir: Path,
    gene_classification: Path,
    gtf_path: Path,
    window_size: int = 1_048_576,
    seed: int = 42,
    max_variants_per_gene: int = 50,
    chromosomes: tuple[str, ...] = CAUSALITY_CHROMOSOMES,
) -> pd.DataFrame:
    """Cis-tested SNV background pool for one tissue × QTL type.

    Negatives are every cis-tested allpairs SNV that is not a member of a SuSiE
    95% credible set. Credible-set exclusion is the low-PIP proxy; the pool spans
    all cis-tested genes and is *not* restricted to genes that carry a credible
    set. ``max_variants_per_gene`` bounds pool size/memory only.
    """
    qtl_key = qtl_type.lower()
    qtl_label = "eQTL" if qtl_key == "eqtl" else "sQTL"
    susie_path = Path(susie_dir) / f"{tissue}.{version}.{qtl_label}s.SuSiE_summary.parquet"
    exclude_keys = load_susie_credible_set_variant_keys(susie_path)
    frames: list[pd.DataFrame] = []

    for chrom in chromosomes:
        chr_num = str(chrom).replace("chr", "")
        if qtl_key == "eqtl":
            allpairs_path = Path(allpairs_dir) / f"{tissue}.{version}.allpairs.chr{chr_num}.parquet"
        elif qtl_key == "sqtl":
            allpairs_path = (
                Path(allpairs_dir)
                / f"{tissue}.{version}.cis_sqtl.allpairs.chr{chr_num}.parquet"
            )
        else:
            raise ValueError(f"Unsupported qtl_type: {qtl_type}")
        if not allpairs_path.exists():
            raise FileNotFoundError(allpairs_path)
        sub = _load_allpairs_for_chromosome(
            allpairs_path,
            exclude_keys=exclude_keys,
            max_variants_per_gene=max_variants_per_gene,
            seed=seed,
        )
        if sub.empty:
            continue
        sub = sub.rename(columns={"gene_id": "phenotype_id"})
        sub["gene_key"] = sub["phenotype_id"]
        sub["tissue"] = tissue
        sub["qtl_type"] = qtl_key
        sub["chromosome"] = sub["variant_id"].map(
            lambda v: genome.Variant.from_str(v, genome.VariantFormat.GTEX).chromosome
        )
        sub["pip"] = 0.0
        sub["cs_id"] = -1
        sub["cs_size"] = 0
        sub["susie_beta"] = 0.0
        sub["background_source"] = "allpairs"
        sub["locus_id"] = (
            sub["tissue"]
            + ":"
            + sub["qtl_type"]
            + ":"
            + sub["phenotype_id"].astype(str)
            + ":allpairs"
        )
        sub["example_id"] = sub["locus_id"] + ":" + sub["variant_id"]
        sub["split"] = sub["chromosome"].map(normalize_chromosome).map(DEFAULT_CHR_SPLIT.assign)
        sub = sub[sub["split"].notna()].copy()
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["example_id"]).reset_index(drop=True)
    # Per-gene capping already applied per-chromosome in _load_allpairs_for_chromosome.

    df = annotate_gene_stable(
        df, version=version, gene_classification=gene_classification
    )

    # TSS distance from the GTF (median transcript TSS), matching how positives are
    # measured, so distance balancing compares like with like. Overwrites the GTEx
    # allpairs `tss_distance` (gene-boundary based) in the source files.
    df = add_tss_distance(df, gtf_path=gtf_path)
    return add_interval_columns(df, window_size, gtf_path=gtf_path)

def build_distance_matched_background_examples(
    version: str,
    tissues: list[str],
    qtl_types: list[str],
    *,
    susie_dir: Path,
    allpairs_dir: Path,
    gene_classification: Path,
    gtf_path: Path,
    window_size: int = 1_048_576,
    seed: int = 42,
    max_variants_per_gene: int = 50,
    chromosomes: tuple[str, ...] = CAUSALITY_CHROMOSOMES,
    high_pip_threshold: float = 0.9,
    n_distance_bins: int = 10,
) -> pd.DataFrame:
    """Distance-matched negative pool for GPU scoring (~1 negative per high-PIP positive).

    Builds and matches one tissue × QTL type at a time to limit peak memory.
    Train/eval re-match from the scored pool at runtime, so split/CV statistics
    are unchanged with far fewer GPU score keys than positives × matching configs.
    Distance matching is tissue-stratified: negatives are drawn from the same
    tissue's allpairs pool on the same chromosome as the high-PIP positives
    they balance.
    """
    gtf_path = Path(gtf_path)

    matched_frames: list[pd.DataFrame] = []
    num_full_pool_examples = 0
    tissue_stats: list[dict] = []

    for tissue in tissues:
        for qtl_type in qtl_types:
            print(
                f"[background] distance-match tissue={tissue} qtl_type={qtl_type}",
                flush=True,
            )
            tissue_pool = build_allpairs_background_for_tissue_qtl(
                version,
                tissue,
                qtl_type,
                susie_dir=susie_dir,
                allpairs_dir=allpairs_dir,
                gene_classification=gene_classification,
                gtf_path=gtf_path,
                window_size=window_size,
                seed=seed,
                max_variants_per_gene=max_variants_per_gene,
                chromosomes=chromosomes,
            )
            num_full_pool_examples += int(len(tissue_pool))
            if tissue_pool.empty:
                tissue_stats.append(
                    {
                        "tissue": tissue,
                        "qtl_type": qtl_type,
                        "num_pool_examples": 0,
                        "num_matched_negatives": 0,
                    }
                )
                continue

            tissue_susie = build_examples(
                version=version,
                tissues=[tissue],
                qtl_types=[qtl_type],
                susie_dir=susie_dir,
                gene_classification=gene_classification,
                gtf_path=gtf_path,
                window_size=window_size,
                seed=seed,
                min_variants_per_locus=1,
                min_pip=high_pip_threshold,
                drop_unassigned_splits=False,
                chromosomes=list(chromosomes),
            )
            matched_ids = collect_distance_matched_negative_ids(
                tissue_susie,
                tissue_pool,
                gtf_path=gtf_path,
                high_pip_threshold=high_pip_threshold,
                n_distance_bins=n_distance_bins,
                seed=seed,
            )
            if matched_ids:
                matched_sub = tissue_pool[
                    tissue_pool["example_id"].astype(str).isin(matched_ids)
                ].copy()
                matched_frames.append(matched_sub)
            tissue_stats.append(
                {
                    "tissue": tissue,
                    "qtl_type": qtl_type,
                    "num_pool_examples": int(len(tissue_pool)),
                    "num_matched_negatives": int(len(matched_ids)),
                }
            )

    if not matched_frames:
        raise RuntimeError("No distance-matched background negatives could be selected.")

    matched = pd.concat(matched_frames, ignore_index=True)
    matched = matched.drop_duplicates(subset=["example_id"]).reset_index(drop=True)
    matched = add_score_keys(matched).drop_duplicates(subset=["score_key"]).reset_index(drop=True)
    build_distance_matched_background_examples.last_build_stats = {
        "num_full_pool_examples": num_full_pool_examples,
        "num_matched_negative_examples": int(len(matched)),
        "num_unique_score_keys": int(matched["score_key"].nunique()) if "score_key" in matched.columns else int(len(matched)),
        "high_pip_threshold": high_pip_threshold,
        "n_distance_bins": n_distance_bins,
        "tissue_stratified": True,
        "per_tissue_stats": tissue_stats,
    }
    print(
        json.dumps(build_distance_matched_background_examples.last_build_stats, indent=2),
        flush=True,
    )
    return matched

build_distance_matched_background_examples.last_build_stats: dict = {}

def build_allpairs_background_metadata(
    *,
    version: str,
    metadata_path: Path,
    tissues: list[str],
    qtl_types: list[str],
    susie_dir: Path,
    allpairs_dir: Path,
    gene_classification: Path,
    gtf_path: Path,
    window_size: int = 1_048_576,
    seed: int = 42,
    max_variants_per_gene: int = 50,
    high_pip_threshold: float = 0.9,
    n_distance_bins: int = 10,
) -> dict:
    metadata_path = Path(metadata_path)
    chromosomes = CAUSALITY_CHROMOSOMES

    examples = build_distance_matched_background_examples(
        version,
        tissues,
        qtl_types,
        susie_dir=susie_dir,
        allpairs_dir=allpairs_dir,
        gene_classification=gene_classification,
        gtf_path=gtf_path,
        window_size=window_size,
        seed=seed,
        max_variants_per_gene=max_variants_per_gene,
        chromosomes=chromosomes,
        high_pip_threshold=high_pip_threshold,
        n_distance_bins=n_distance_bins,
    )
    save_examples_parquet(examples, metadata_path)

    summary = summarize_metadata(
        examples,
        dataset_version=version,
        cache_type="allpairs_background",
        metadata_path=metadata_path,
    )
    stats = getattr(build_distance_matched_background_examples, "last_build_stats", {})
    if stats:
        summary.update(stats)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SuSiE lead or distance-matched background cohorts (unscored)."
    )
    parser.add_argument(
        "--cache-type",
        choices=["susie_lead", "allpairs_background"],
        default="susie_lead",
    )
    parser.add_argument("--dataset-version", choices=["v10", "v11"], required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for metadata parquet.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Filename under --output_dir "
            "(default: {version}_susie_lead_metadata.parquet or "
            "{version}_allpairs_background_metadata.parquet)."
        ),
    )
    parser.add_argument(
        "--full_gtf",
        type=Path,
        required=True,
        help="GENCODE annotation GTF (v39 for v10 QTLs, v47 for v11).",
    )
    parser.add_argument(
        "--susie-dir",
        type=Path,
        required=True,
        help="Folder of *.{version}.{e,s}QTLs.SuSiE_summary.parquet files.",
    )
    parser.add_argument(
        "--allpairs-dir",
        type=Path,
        default=None,
        help="Folder of per-chromosome cis-allpairs parquet files (required for allpairs_background).",
    )
    parser.add_argument(
        "--gene-classification",
        type=Path,
        required=True,
        help="GENCODE v39/v47 gene classification TSV (stable IDs + biotypes).",
    )
    parser.add_argument("--tissues", nargs="+", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or _default_metadata_name(
        args.dataset_version, args.cache_type
    )
    output = args.output_dir / output_name

    common = dict(
        version=args.dataset_version,
        metadata_path=output,
        tissues=args.tissues,
        qtl_types=list(QTL_TYPES),
        susie_dir=args.susie_dir,
        gene_classification=args.gene_classification,
        gtf_path=args.full_gtf,
        window_size=WINDOW_SIZE,
        seed=SEED,
    )
    if args.cache_type == "allpairs_background":
        if args.allpairs_dir is None:
            raise SystemExit("--allpairs-dir is required for --cache-type allpairs_background")
        summary = build_allpairs_background_metadata(
            **common,
            allpairs_dir=args.allpairs_dir,
            max_variants_per_gene=MAX_VARIANTS_PER_GENE,
            high_pip_threshold=HIGH_PIP_THRESHOLD,
            n_distance_bins=N_DISTANCE_BINS,
        )
    else:
        summary = build_susie_lead_metadata(
            **common,
            min_variants_per_locus=MIN_VARIANTS_PER_LOCUS,
            min_pip=MIN_PIP,
        )

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
