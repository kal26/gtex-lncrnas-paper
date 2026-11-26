"""Step 2 — GPU-score lead + background cohorts (v10 and v11 in one process)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def run_chr_score_caches(
    *,
    score_chromosomes: list[str],
    reference_fasta: Path,
    base_model_version: str,
    kaggle_model_version: int = 1,
    gtf_feather_path_v10: str,
    gtf_feather_path_v11: str,
    progress_every: int,
    output_dir: Path,
    v10_lead_metadata: Path,
    v11_lead_metadata: Path,
    v10_background_metadata: Path,
    v11_background_metadata: Path,
) -> list[dict]:
    scores_dir = output_dir
    scores_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    version_specs: list[tuple[str, str]] = [
        ("v10", gtf_feather_path_v10),
        ("v11", gtf_feather_path_v11),
    ]

    for version, gtf_feather_path in version_specs:
        print(f"[score_chr_caches] loading AlphaGenome model for {version}", flush=True)
        splice_starts, splice_ends = default_splice_site_feather_paths(gtf_feather_path)
        print(f"[score_chr_caches] {version} gtf_feather={gtf_feather_path}", flush=True)
        print(f"[score_chr_caches] {version} splice_site_starts={splice_starts}", flush=True)
        print(f"[score_chr_caches] {version} splice_site_ends={splice_ends}", flush=True)
        model = load_native_alphagenome_model(
            base_model_version=base_model_version,
            kaggle_model_version=kaggle_model_version,
            reference_fasta=reference_fasta,
            gtf_feather_path=gtf_feather_path,
            splice_site_starts_feather_path=splice_starts,
            splice_site_ends_feather_path=splice_ends,
        )
        lead_path = v10_lead_metadata if version == "v10" else v11_lead_metadata
        score_kwargs = dict(
            version=version,
            scores_dir=scores_dir,
            score_chromosomes=score_chromosomes,
            reference_fasta=reference_fasta,
            base_model_version=base_model_version,
            gtf_feather_path=gtf_feather_path,
            splice_site_starts_feather_path=splice_starts,
            splice_site_ends_feather_path=splice_ends,
            progress_every=progress_every,
            model=model,
        )

        print(
            f"[score_chr_caches] {version} SuSiE lead-variant on {', '.join(score_chromosomes)}",
            flush=True,
        )
        lead_summary = score_metadata_tidy_cache(
            **score_kwargs,
            cache_type="susie_lead_variant",
            metadata_path=lead_path,
        )
        lead_summary["tidy_root"] = str(tidy_scores_root(scores_dir, version))
        summaries.append(lead_summary)
        print(json.dumps(lead_summary, indent=2))

        bg_output = (
            v10_background_metadata if version == "v10" else v11_background_metadata
        )
        print(
            f"[score_chr_caches] {version} distance-matched background on "
            f"{', '.join(score_chromosomes)}",
            flush=True,
        )
        bg_summary = score_metadata_tidy_cache(
            **score_kwargs,
            cache_type="allpairs_background",
            metadata_path=bg_output,
        )
        bg_summary["tidy_root"] = str(tidy_scores_root(scores_dir, version))
        summaries.append(bg_summary)
        print(json.dumps(bg_summary, indent=2))

        del model

    return summaries

# --- step implementation ---

import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd

def _strip_gene_version(gene_key: str) -> str:
    return str(gene_key).split(".")[0]

def _tissue_display_name(tissue: str) -> str:
    return str(tissue).replace("_", " ")

def _tissue_feature_token(tissue: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", _tissue_display_name(tissue).strip())
    return token.strip("_") or "unknown"

def _tissues_match(left: str, right: str) -> bool:
    return _tissue_feature_token(left).lower() == _tissue_feature_token(right).lower()

def variant_to_genome_objects(row: pd.Series):
    from alphagenome.data import genome

    variant = genome.Variant.from_str(
        str(row["variant_id"]), genome.VariantFormat.GTEX
    )
    interval = genome.Interval(
        chromosome=str(row["chromosome"]),
        start=int(row["window_start"]),
        end=int(row["window_end"]),
    )
    return interval, variant

def native_scorer_for_qtl_type(qtl_type: str):
    from alphagenome.models import dna_output, variant_scorers

    qtl_type = qtl_type.lower()
    if qtl_type == "eqtl":
        return variant_scorers.GeneMaskLFCScorer(
            requested_output=dna_output.OutputType.RNA_SEQ
        )
    if qtl_type == "sqtl":
        return variant_scorers.GeneMaskSplicingScorer(
            requested_output=dna_output.OutputType.SPLICE_SITE_USAGE,
            width=None,
        )
    raise ValueError(f"Unsupported qtl_type for native scoring: {qtl_type}")

def default_splice_site_feather_paths(gtf_feather_path: str | Path | None) -> tuple[str | None, str | None]:
    """Infer GENCODE splice-site start/end feathers next to a GTF feather.

    ``gencode.v47.annotation.gtf.feather`` →
    ``gencode.v47.splice_sites_{starts,ends}.feather``.
    """
    if not gtf_feather_path:
        return None, None
    path = Path(gtf_feather_path)
    name = path.name
    for suffix in (".annotation.gtf.feather", ".gtf.feather", ".feather"):
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            break
    else:
        prefix = path.stem
    starts = path.parent / f"{prefix}.splice_sites_starts.feather"
    ends = path.parent / f"{prefix}.splice_sites_ends.feather"
    if starts.is_file() and ends.is_file():
        return str(starts), str(ends)
    return None, None

def apply_splice_junction_all_biotypes(model, gtf_feather_path: str | Path) -> None:
    """Re-bind SpliceJunctionScorer gene masks to every GTF gene biotype.

    AlphaGenome's ``SpliceJunctionVariantScorer`` hardcodes
    ``filter_protein_coding=True``. GeneMask splice-site / splice-site-usage
    scorers already use all genes in the GTF. Reload the unfiltered GTF so
    junction scores are also assigned to lncRNA / pseudogene / other genes.
    """
    import pandas as pd
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome.models import variant_scorers as variant_scorers_lib
    from alphagenome_research.model.variant_scoring import gene_mask_extractor

    gtf = pd.read_feather(str(gtf_feather_path))
    organism = ag_dna_model.Organism.HOMO_SAPIENS
    scorer = model._variant_scorers[organism][variant_scorers_lib.BaseVariantScorer.SPLICE_JUNCTION]
    scorer._gene_mask_extractor = gene_mask_extractor.GeneMaskExtractor(
        gtf,
        gene_mask_extractor.GeneMaskType.BODY,
        gene_query_type=gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING,
        filter_protein_coding=False,
    )

def load_native_alphagenome_model(
    *,
    base_model_version: str = "all_folds",
    kaggle_model_version: int = 1,
    reference_fasta: Path,
    gtf_feather_path: str | None = None,
    splice_site_starts_feather_path: str | None = None,
    splice_site_ends_feather_path: str | None = None,
):
    """Load AlphaGenome weights pinned to a Kaggle model instance version.

    Default pin is ``google/alphagenome/jax/all_folds/1``, matching the
    checkpoint used for GTEx v10/v11 scoring (downloaded 18 May 2026).
    """
    import kagglehub
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model
    from kagglehub import auth as kaggle_auth

    settings = dict(research_dna_model.default_organism_settings())
    human = settings[ag_dna_model.Organism.HOMO_SAPIENS]
    if not gtf_feather_path:
        raise ValueError("gtf_feather_path is required (paper v10/v11 GENCODE feathers)")
    settings[ag_dna_model.Organism.HOMO_SAPIENS] = dataclasses.replace(
        human,
        fasta_path=str(reference_fasta),
        gtf_feather_path=gtf_feather_path,
        splice_site_starts_feather_path=(
            splice_site_starts_feather_path or human.splice_site_starts_feather_path
        ),
        splice_site_ends_feather_path=(
            splice_site_ends_feather_path or human.splice_site_ends_feather_path
        ),
        calibration_path=None,
    )
    if kaggle_auth.get_username() is None:
        kagglehub.login()
    # Pin numeric Kaggle version; create_from_kaggle() downloads unversioned/latest.
    handle = (
        f"google/alphagenome/jax/{str(base_model_version).lower()}/{int(kaggle_model_version)}"
    )
    print(f"[alphagenome] loading pinned Kaggle model: {handle}", flush=True)
    checkpoint_path = kagglehub.model_download(handle)
    model = research_dna_model.create(
        checkpoint_path,
        organism_settings=settings,
    )
    if gtf_feather_path:
        apply_splice_junction_all_biotypes(model, gtf_feather_path)
    return model

def _extract_gene_tissue_score(
    tidy_df: pd.DataFrame,
    *,
    gene_key: str,
    tissue: str,
    signed: bool,
) -> tuple[float, str | None]:
    if tidy_df.empty:
        return float("nan"), "AlphaGenome returned no score rows for this variant interval"
    gene_base = _strip_gene_version(gene_key)
    sub = tidy_df[tidy_df["gene_id"].astype(str) == gene_base].copy()
    if sub.empty:
        available = sorted(tidy_df["gene_id"].astype(str).unique())[:5]
        hint = ", ".join(available)
        if len(tidy_df["gene_id"].unique()) > 5:
            hint += ", ..."
        return (
            float("nan"),
            f"gene_id {gene_base} not present in AlphaGenome GeneMask output "
            f"(sample available genes: {hint})",
        )
    tissue_name = _tissue_display_name(tissue)
    if "gtex_tissue" in sub.columns:
        tissue_match = sub["gtex_tissue"].fillna("").apply(lambda t: _tissues_match(str(t), tissue))
        if tissue_match.any():
            sub = sub[tissue_match]
        elif not sub.empty:
            available_tissues = sorted(sub["gtex_tissue"].fillna("").unique())[:5]
            return (
                float("nan"),
                f"tissue {tissue_name!r} not present for gene {gene_base} "
                f"(available: {', '.join(available_tissues)})",
            )
    if sub.empty:
        return float("nan"), f"no score rows after filtering to gene {gene_base}"
    values = sub["raw_score"].astype(float)
    if signed:
        return float(values.iloc[np.argmax(np.abs(values))]), None
    return float(np.abs(values).max()), None

def _sanitize_feature_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    return token.strip("_") or "unknown"

def feature_column_name(*parts: str) -> str:
    return "mm_" + "__".join(_sanitize_feature_token(p) for p in parts if p)

def build_recommended_variant_scorers():
    """AlphaGenome's 19 recommended variant scoring configurations.

    Returns the library-provided ``RECOMMENDED_VARIANT_SCORERS`` filtered to human
    support. This matches the paper's "set of 19 recommended variant scoring
    configurations covering diverse genomic modalities" (Methods: "Variant
    scoring") used for the supervised (RF) causality features. Using the library
    set guarantees the correct per-modality configs — e.g. CHIP_HISTONE width=2001,
    the ``*_ACTIVE`` aggregations, PROCAP, polyadenylation, splice sites/junctions,
    and contact maps — rather than a hand-picked subset.
    """
    from alphagenome.models import variant_scorers
    from alphagenome.protos import dna_model_pb2

    return list(
        variant_scorers.get_recommended_scorers(
            dna_model_pb2.Organism.ORGANISM_HOMO_SAPIENS
        )
    )

def _max_abs_raw_score(rows: pd.DataFrame) -> float | None:
    if rows.empty:
        return None
    values = rows["raw_score"].astype(float)
    return float(values.iloc[int(np.argmax(np.abs(values.to_numpy())))])

def _matched_gtex_tissue_rows(gene_rows: pd.DataFrame, tissue: str) -> pd.DataFrame:
    if gene_rows.empty or "gtex_tissue" not in gene_rows.columns:
        return gene_rows.iloc[0:0]
    mask = gene_rows["gtex_tissue"].fillna("").astype(str).apply(
        lambda gtex_tissue: _tissues_match(gtex_tissue, tissue)
    )
    return gene_rows.loc[mask]

def _gene_mask_scorer_score(
    gene_rows: pd.DataFrame,
    *,
    scorer_name: str,
    tissue: str,
) -> float | None:
    """One scalar per GeneMask scorer config for the eQTL target gene."""
    if gene_rows.empty:
        return None
    if "RNA_SEQ" in scorer_name:
        return _max_abs_raw_score(_matched_gtex_tissue_rows(gene_rows, tissue))
    return _max_abs_raw_score(gene_rows)

def extract_multimodal_features(
    tidy_df: pd.DataFrame,
    *,
    gene_key: str,
    tissue: str,
    scorer_names: set[str] | None = None,
) -> dict[str, float]:
    """Convert tidy AlphaGenome scores to one RF feature per recommended scorer.

    Matches the paper causality RF (Ext. Data Fig. 5e): each recommended variant
    scorer configuration contributes a single aggregated scalar. Gene-centric
    scorers use the eQTL target gene; GeneMask RNA uses the matched GTEx tissue;
    all other scorers use the maximum absolute raw score across output tracks.
    """
    if tidy_df is None or tidy_df.empty:
        return {}

    features: dict[str, float] = {}
    gene_base = _strip_gene_version(gene_key)

    if scorer_names is None:
        scorer_names = set(tidy_df.get("variant_scorer", pd.Series(dtype=str)).astype(str))

    for scorer_name in sorted(scorer_names):
        sub = tidy_df[tidy_df["variant_scorer"].astype(str) == scorer_name]
        if sub.empty:
            continue

        score: float | None
        if "GeneMask" in scorer_name:
            gene_rows = sub[sub["gene_id"].astype(str) == gene_base]
            score = _gene_mask_scorer_score(gene_rows, scorer_name=scorer_name, tissue=tissue)
        elif "gene_id" in sub.columns and sub["gene_id"].notna().any():
            gene_rows = sub[sub["gene_id"].astype(str) == gene_base]
            score = _max_abs_raw_score(gene_rows)
        else:
            score = _max_abs_raw_score(sub)

        if score is not None:
            features[feature_column_name(scorer_name)] = score

    return features

from util import multimodal_feature_columns  # shared with RF / cohort builders

def scorer_feature_column(scorer_name: str) -> str:
    """The single ``mm_*`` column name for a recommended scorer configuration."""
    return feature_column_name(scorer_name)

def scorer_feature_coverage(columns, scorer_names) -> dict[str, int]:
    """Return 1 when a scorer's single ``mm_*`` column is present, else 0."""
    mm_cols = {str(c) for c in columns if str(c).startswith("mm_")}
    return {
        str(name): int(scorer_feature_column(name) in mm_cols)
        for name in scorer_names
    }

def assert_scorer_feature_coverage(
    columns,
    scorer_names,
    *,
    context: str = "",
) -> dict[str, int]:
    """Fail loudly when a configured scorer produced zero ``mm_*`` feature columns."""
    coverage = scorer_feature_coverage(columns, scorer_names)
    where = f" in {context}" if context else ""

    summary = ", ".join(f"{name}={coverage[name]}" for name in sorted(coverage))
    print(f"[multimodal] scorer feature coverage{where}: {summary}", flush=True)

    missing = sorted(name for name, n in coverage.items() if n == 0)
    if missing:
        raise ValueError(
            f"[multimodal] {len(missing)} scorer(s) produced no mm_* feature columns{where}: "
            + "; ".join(missing)
            + ". Check extract_multimodal_features handling for these scorers."
        )
    return coverage

def prune_null_mm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop mm_* columns that are all-null (other genes on the same chr part)."""
    mm_cols = multimodal_feature_columns(df)
    if not mm_cols:
        return df
    keep_mm = [col for col in mm_cols if df[col].notna().any()]
    if len(keep_mm) == len(mm_cols):
        return df
    drop_mm = [col for col in mm_cols if col not in keep_mm]
    return df.drop(columns=drop_mm)

def _gene_mask_lfc_feature_column(features: dict[str, float]) -> str | None:
    for col in features:
        if "GeneMaskLFCScorer" in col and "RNA_SEQ" in col:
            return col
    return None

def _primary_gene_mask_score(features: dict[str, float]) -> float | None:
    col = _gene_mask_lfc_feature_column(features)
    if col is None:
        return None
    return float(features[col])

def resolve_scorers():
    scorers = build_recommended_variant_scorers()
    scorer_names = {str(s.name) for s in scorers}
    return scorers, scorer_names

def _annotate_tidy_with_score_key(tidy: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if tidy is None or tidy.empty:
        return pd.DataFrame()
    out = tidy.copy()
    for col in ("score_key", "variant_id", "gene_key", "gene_key_stable", "window_start", "window_end", "chromosome"):
        if col in row.index:
            out[col] = row[col]
    return out

def enrich_row_from_tidy(
    tidy_df: pd.DataFrame,
    *,
    gene_key: str,
    tissue: str,
    qtl_type: str = "eqtl",
    scorer_names: set[str] | None = None,
) -> tuple[dict[str, float], float | None, str | None, str | None]:
    """Return (mm_features, zero_shot_score, native_scorer, error)."""

    if tidy_df is None or tidy_df.empty:
        return {}, None, None, "empty tidy scores"

    features = extract_multimodal_features(
        tidy_df,
        gene_key=gene_key,
        tissue=tissue,
        scorer_names=scorer_names,
    )
    zero_shot = _primary_gene_mask_score(features)
    native_scorer = None
    err = None
    if zero_shot is None:
        try:
            scorer = native_scorer_for_qtl_type(qtl_type)
            signed = qtl_type.lower() == "eqtl"
            zero_shot, err = _extract_gene_tissue_score(
                tidy_df,
                gene_key=gene_key,
                tissue=tissue,
                signed=signed,
            )
            if zero_shot is not None and not pd.isna(zero_shot):
                native_scorer = str(scorer)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
    else:
        native_scorer = "GeneMaskLFCScorer(RNA_SEQ)"
    return features, zero_shot, native_scorer, err

def score_queue_to_tidy(
    queue: pd.DataFrame,
    *,
    scorers,
    scorer_names: set[str],
    reference_fasta: Path,
    base_model_version: str = "all_folds",
    kaggle_model_version: int = 1,
    gtf_feather_path: str | None = None,
    splice_site_starts_feather_path: str | None = None,
    splice_site_ends_feather_path: str | None = None,
    model=None,
    progress_every: int = 10,
) -> pd.DataFrame:
    """Score unique score_key rows once each; return concatenated tidy output."""
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome.models import variant_scorers

    if queue.empty:
        return pd.DataFrame()

    if "score_key" not in queue.columns:

        keyed = add_score_keys(queue)
    else:
        keyed = queue.copy()
    if model is None:
        model = load_native_alphagenome_model(
            base_model_version=base_model_version,
            kaggle_model_version=kaggle_model_version,
            reference_fasta=reference_fasta,
            gtf_feather_path=gtf_feather_path,
            splice_site_starts_feather_path=splice_site_starts_feather_path,
            splice_site_ends_feather_path=splice_site_ends_feather_path,
        )

    log_every = max(1, int(progress_every))
    frames: list[pd.DataFrame] = []
    for idx, (_, row) in enumerate(keyed.iterrows(), start=1):
        row_series = pd.Series(row)
        try:
            interval, variant = variant_to_genome_objects(row_series)
            score_outputs = model.score_variant(
                interval,
                variant,
                variant_scorers=scorers,
                organism=ag_dna_model.Organism.HOMO_SAPIENS,
            )
            tidy = variant_scorers.tidy_scores(score_outputs)
            annotated = _annotate_tidy_with_score_key(
                tidy if tidy is not None else pd.DataFrame(),
                row_series,
            )
            if not annotated.empty:
                frames.append(annotated)
        except Exception as exc:  # noqa: BLE001
            print(f"[multimodal_scoring] tidy score failed {row['score_key']}: {exc}", flush=True)
        if idx % log_every == 0 or idx == len(keyed):
            print(f"[multimodal_scoring] tidy scored {idx}/{len(keyed)} score_keys", flush=True)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

import gc
import json
import os
import shutil
import uuid
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from util import (
    load_examples_parquet,
    save_examples_parquet,
)
from util import normalize_chromosome

SCORE_KEY_COLUMNS = ("variant_id", "gene_key", "window_start", "window_end")

from util import add_score_keys, strip_gene_version  # shared with cohort builders

def unique_scoring_queue(examples: pd.DataFrame) -> pd.DataFrame:
    """One row per unique (variant, gene, window) for GPU scoring."""
    return add_score_keys(examples).drop_duplicates(subset=["score_key"]).reset_index(drop=True)

# Columns required by enrich_row_from_tidy / extract_multimodal_features (plus
# output_type/track_name so shard merges keep one row per track).
_ENRICH_TIDY_COLUMNS = (
    "score_key",
    "variant_scorer",
    "output_type",
    "track_name",
    "gene_id",
    "gtex_tissue",
    "raw_score",
)

# Default score_key batch size for enrich. chr1 allpairs background alone is
# ~55k metadata keys × ~55k tidy rows/key ≈ 200GB if loaded at once.
_DEFAULT_ENRICH_KEY_BATCH_SIZE = 50

# AlphaGenome tidy_scores() attaches genome Interval/Variant objects; drop for parquet.
_TIDY_OBJECT_COLUMNS_TO_DROP = frozenset({"scored_interval", "scored_variant"})

def _coerce_tidy_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Make AlphaGenome tidy output parquet-safe (Interval/Variant → str or drop)."""
    if df.empty:
        return df
    out = df.copy()
    for col in _TIDY_OBJECT_COLUMNS_TO_DROP:
        if col in out.columns:
            out = out.drop(columns=[col])
    for col in out.columns:
        if out[col].dtype != object:
            continue
        out[col] = out[col].map(
            lambda v: None
            if v is None or (isinstance(v, float) and pd.isna(v))
            else v
            if isinstance(v, (str, bytes))
            else str(v)
        )
    return out

def tidy_scores_root(scores_dir: Path, dataset_version: str) -> Path:
    return Path(scores_dir) / "tidy" / dataset_version

def tidy_partition_dir(root: Path, chromosome: str) -> Path:
    return Path(root) / normalize_chromosome(chromosome)

# Lead and background cohorts write separate files under the same chr dir so a
# full 02 run never overwrites the other cohort's scores.
_TIDY_SCORES_BASENAME = {
    "susie_lead": "leads_scores.parquet",
    "susie_lead_variant": "leads_scores.parquet",
    "allpairs_background": "background_scores.parquet",
}

def tidy_scores_basename(cache_type: str) -> str:
    key = str(cache_type).strip()
    if key not in _TIDY_SCORES_BASENAME:
        raise ValueError(
            f"Unknown cache_type {cache_type!r}; expected one of "
            f"{sorted(set(_TIDY_SCORES_BASENAME))}."
        )
    return _TIDY_SCORES_BASENAME[key]

def tidy_partition_path(root: Path, chromosome: str, cache_type: str) -> Path:
    return tidy_partition_dir(root, chromosome) / tidy_scores_basename(cache_type)

def cache_type_from_metadata_path(metadata_path: Path) -> str:
    """Infer lead vs background tidy file from step-01 metadata filename."""
    name = Path(metadata_path).name
    if "allpairs_background" in name:
        return "allpairs_background"
    if "susie_lead" in name:
        return "susie_lead_variant"
    raise ValueError(
        f"Cannot infer cache_type from metadata name {name!r}; "
        "expected *susie_lead* or *allpairs_background*."
    )

def _read_score_keys_from_parquet(path: Path) -> set[str]:
    """Collect unique score_keys without materializing every tidy row as Python strings."""
    if not path.exists():
        return set()
    pf = pq.ParquetFile(path)
    if pf.metadata is None or pf.metadata.num_rows == 0:
        return set()
    keys: set[str] = set()
    for rg_idx in range(pf.num_row_groups):
        col = pf.read_row_group(rg_idx, columns=["score_key"]).column("score_key")
        if len(col) == 0:
            continue
        keys.update(pc.unique(col).to_pylist())
    return keys

def _read_score_keys(partition_path: Path) -> set[str]:
    return _read_score_keys_from_parquet(Path(partition_path))

def _parquet_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pq.ParquetFile(path).metadata.num_rows)

def _count_tidy_rows(partition_path: Path) -> int:
    return _parquet_row_count(Path(partition_path))

def load_tidy_for_score_keys(
    partition_path: Path,
    score_keys: set[str],
    *,
    columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Load only rows for the given score_keys (used by derive/enrich; avoids full-chr RAM)."""
    if not score_keys:
        return pd.DataFrame()
    partition_path = Path(partition_path)
    if not partition_path.exists():
        return pd.DataFrame()
    key_list = sorted(score_keys)
    read_columns = list(columns) if columns is not None else None
    table = pq.read_table(
        partition_path,
        filters=[("score_key", "in", key_list)],
        columns=read_columns,
    )
    if table.num_rows == 0:
        return pd.DataFrame()
    return table.to_pandas()

def write_tidy_partition(path: Path, df: pd.DataFrame) -> None:
    """Overwrite ``path`` with ``df`` (full rewrite; no append/resume)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if df.empty:
        pq.write_table(pa.table({}), path)
        return
    _coerce_tidy_for_parquet(df).to_parquet(path, index=False)

def _enriched_scalar_part_path(parts_dir: Path, chromosome: str) -> Path:
    return Path(parts_dir) / f"{normalize_chromosome(chromosome)}.scalar.parquet"

def write_enriched_chromosome_parts(
    chrom_df: pd.DataFrame,
    parts_dir: Path,
    chromosome: str,
) -> int:
    """Persist one enriched chromosome part (metadata, zero-shot, ~19 ``mm_*`` columns)."""
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    final_path = _enriched_scalar_part_path(parts_dir, chromosome)
    tmp_path = parts_dir / f".{final_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        save_examples_parquet(chrom_df, tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return int(len(chrom_df))

def _enrich_one_chromosome(
    chrom_examples: pd.DataFrame,
    *,
    chromosome: str,
    tidy_root: Path,
    cache_type: str,
    parts_dir: Path | None,
    batch_size: int,
    progress_every: int,
    scorer_names: set[str],
) -> dict:
    """Enrich one chromosome; write scalar part when ``parts_dir`` is set."""
    chrom = normalize_chromosome(chromosome)
    n_rows = int(len(chrom_examples))
    if n_rows == 0:
        return {
            "chromosome": chrom,
            "n_rows": 0,
            "missing_keys": 0,
            "wrote_part": False,
            "frame": None,
        }

    chrom_keys = sorted(set(chrom_examples["score_key"]))
    partition_path = tidy_partition_path(tidy_root, chrom, cache_type)
    chrom_rows: list[dict] = []
    chrom_missing = 0
    n_batches = (len(chrom_keys) + batch_size - 1) // batch_size
    print(
        f"[enrich] {chrom}: {n_rows} rows, {len(chrom_keys)} keys "
        f"({tidy_scores_basename(cache_type)}) in {n_batches} batches of <={batch_size}",
        flush=True,
    )

    for batch_idx in range(0, len(chrom_keys), batch_size):
        batch_keys = set(chrom_keys[batch_idx : batch_idx + batch_size])
        batch_examples = chrom_examples.loc[chrom_examples["score_key"].isin(batch_keys)]
        part = load_tidy_for_score_keys(
            partition_path,
            batch_keys,
            columns=_ENRICH_TIDY_COLUMNS,
        )
        tidy_by_key = (
            {key: sub for key, sub in part.groupby("score_key", sort=False)}
            if not part.empty
            else {}
        )
        batch_rows, batch_missing = _enrich_rows_for_chromosome(
            batch_examples,
            tidy_by_key=tidy_by_key,
            scorer_names=scorer_names,
            progress_label=str(chrom),
            progress_every=progress_every,
            row_offset=len(chrom_rows),
            row_total=n_rows,
        )
        chrom_missing += batch_missing
        chrom_rows.extend(batch_rows)
        del part, tidy_by_key, batch_rows, batch_examples
        gc.collect()

    frame = None
    wrote_part = False
    if chrom_rows:
        chrom_df = pd.DataFrame(chrom_rows)
        if parts_dir is not None:
            n_written = write_enriched_chromosome_parts(chrom_df, parts_dir, chrom)
            wrote_part = True
            print(f"[enrich] wrote parts for {chrom} ({n_written} rows)", flush=True)
        else:
            frame = chrom_df
        del chrom_df
    del chrom_rows
    gc.collect()

    return {
        "chromosome": chrom,
        "n_rows": n_rows,
        "missing_keys": int(chrom_missing),
        "wrote_part": wrote_part,
        "frame": frame,
    }

def _enrich_one_chromosome_job(payload: dict) -> dict:
    """ProcessPool entrypoint: enrich one chromosome from a picklable payload."""
    chrom_examples = payload["chrom_examples"]
    return _enrich_one_chromosome(
        chrom_examples,
        chromosome=payload["chromosome"],
        tidy_root=Path(payload["tidy_root"]),
        cache_type=str(payload["cache_type"]),
        parts_dir=Path(payload["parts_dir"]) if payload["parts_dir"] is not None else None,
        batch_size=int(payload["batch_size"]),
        progress_every=int(payload["progress_every"]),
        scorer_names=set(payload["scorer_names"]),
    )

def score_tidy_partition(
    *,
    examples: pd.DataFrame,
    output_root: Path,
    chromosome: str,
    reference_fasta: Path,
    base_model_version: str = "all_folds",
    gtf_feather_path: str | None = None,
    splice_site_starts_feather_path: str | None = None,
    splice_site_ends_feather_path: str | None = None,
    progress_every: int = 100,
    dataset_version: str,
    cache_type: str,
    model=None,
) -> dict:
    """Score unique score_keys on one chromosome; overwrite that cohort's parquet.

    Leads → ``leads_scores.parquet``; backgrounds → ``background_scores.parquet``
    under ``tidy/{version}/chr1/`` (etc.). Re-run the command if a job dies mid-chromosome.
    """
    chrom = normalize_chromosome(chromosome)
    partition_path = tidy_partition_path(output_root, chrom, cache_type)

    queue = unique_scoring_queue(examples)
    queue = queue[queue["chromosome"].map(normalize_chromosome) == chrom].copy()
    if queue.empty:
        write_tidy_partition(partition_path, pd.DataFrame())
        print(f"[tidy_score_cache] {chrom}: no examples on chromosome; wrote empty", flush=True)
        return {
            "dataset_version": dataset_version,
            "cache_type": cache_type,
            "chromosome": chrom,
            "partition_path": str(partition_path),
            "num_score_keys_on_chr": 0,
            "num_score_keys_scored": 0,
            "num_tidy_rows": 0,
            "skipped": True,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

    print(
        f"[tidy_score_cache] {chrom}: scoring {len(queue)} unique score_keys "
        f"→ {partition_path.name}",
        flush=True,
    )
    scored_at = datetime.now(timezone.utc).isoformat()
    scorers, scorer_names = resolve_scorers()

    tidy_batch = score_queue_to_tidy(
        queue,
        scorers=scorers,
        scorer_names=scorer_names,
        reference_fasta=reference_fasta,
        base_model_version=base_model_version,
        gtf_feather_path=gtf_feather_path,
        splice_site_starts_feather_path=splice_site_starts_feather_path,
        splice_site_ends_feather_path=splice_site_ends_feather_path,
        model=model,
        progress_every=max(1, int(progress_every)),
    )
    if not tidy_batch.empty:
        tidy_batch = tidy_batch.copy()
        tidy_batch["dataset_version"] = dataset_version
        tidy_batch["cache_type"] = cache_type
        tidy_batch["scored_at"] = scored_at
    write_tidy_partition(partition_path, tidy_batch)

    keys_after = len(_read_score_keys(partition_path))
    num_tidy_rows = _count_tidy_rows(partition_path)
    print(
        f"[tidy_score_cache] {chrom}: wrote {keys_after} score_keys "
        f"({num_tidy_rows} tidy rows) → {partition_path}",
        flush=True,
    )
    return {
        "dataset_version": dataset_version,
        "cache_type": cache_type,
        "chromosome": chrom,
        "partition_path": str(partition_path),
        "num_score_keys_on_chr": int(queue["score_key"].nunique()),
        "num_score_keys_scored": int(keys_after),
        "num_tidy_rows": int(num_tidy_rows),
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "scoring_params": {
            "reference_fasta": str(reference_fasta),
            "base_model_version": base_model_version,
            "gtf_feather_path": gtf_feather_path,
            "splice_site_starts_feather_path": splice_site_starts_feather_path,
            "splice_site_ends_feather_path": splice_site_ends_feather_path,
            "multimodal_profile": "paper",
        },
    }

def _enrich_rows_for_chromosome(
    chrom_examples: pd.DataFrame,
    *,
    tidy_by_key: dict[str, pd.DataFrame],
    scorer_names: set[str],
    progress_label: str,
    progress_every: int,
    row_offset: int,
    row_total: int,
) -> tuple[list[dict], int]:
    """Enrich example rows for one chromosome from cached tidy scores."""
    enriched_rows: list[dict] = []
    missing_keys = 0
    log_every = max(1, int(progress_every))

    for local_idx, row in enumerate(chrom_examples.itertuples(index=False), start=1):
        row_dict = row._asdict()
        score_key = row_dict["score_key"]
        tidy_sub = tidy_by_key.get(score_key)
        if tidy_sub is None or tidy_sub.empty:
            missing_keys += 1
            enriched_rows.append(row_dict)
            continue

        features, zero_shot, native_scorer, err = enrich_row_from_tidy(
            tidy_sub,
            gene_key=str(row_dict["gene_key"]),
            tissue=str(row_dict["tissue"]),
            qtl_type=str(row_dict.get("qtl_type", "eqtl")),
            scorer_names=scorer_names,
        )
        row_dict.update(features)
        if zero_shot is not None:
            row_dict["zero_shot_score"] = zero_shot
            row_dict["native_scorer"] = native_scorer
        if err is not None:
            row_dict["multimodal_score_error"] = err
        enriched_rows.append(row_dict)

        global_idx = row_offset + local_idx
        if global_idx % log_every == 0 or global_idx == row_total:
            print(
                f"[enrich] {progress_label} assembled {global_idx}/{row_total} example rows",
                flush=True,
            )

    return enriched_rows, missing_keys

def enrich_examples_from_tidy(
    examples: pd.DataFrame,
    *,
    tidy_root: Path,
    cache_type: str,
    progress_every: int = 5000,
    parts_dir: Path | None = None,
    key_batch_size: int | None = None,
) -> pd.DataFrame:
    """Join QTL metadata rows to tidy scores; extract paper-style RF features.

    Within each chromosome, score_keys are processed in batches so filtered tidy
    loads stay bounded.
    """
    if examples.empty:
        return examples.copy()

    _, scorer_names = resolve_scorers()
    keyed = add_score_keys(examples)
    chromosomes = sorted(keyed["chromosome"].map(normalize_chromosome).unique())
    batch_size = max(
        1,
        int(key_batch_size if key_batch_size is not None else _DEFAULT_ENRICH_KEY_BATCH_SIZE),
    )

    if parts_dir is not None:
        parts_dir = Path(parts_dir)
        parts_dir.mkdir(parents=True, exist_ok=True)

    chrom_payloads: list[dict] = []
    for chrom in chromosomes:
        chrom_mask = keyed["chromosome"].map(normalize_chromosome) == chrom
        chrom_examples = keyed.loc[chrom_mask].copy()
        if chrom_examples.empty:
            continue
        chrom_payloads.append(
            {
                "chromosome": chrom,
                "chrom_examples": chrom_examples,
                "tidy_root": str(tidy_root),
                "cache_type": cache_type,
                "parts_dir": str(parts_dir) if parts_dir is not None else None,
                "batch_size": batch_size,
                "progress_every": progress_every,
                "scorer_names": sorted(scorer_names),
            }
        )

    print(f"[enrich] chromosomes={len(chrom_payloads)}", flush=True)
    results = [_enrich_one_chromosome_job(payload) for payload in chrom_payloads]

    missing_keys = sum(int(r["missing_keys"]) for r in results)
    wrote_parts = sum(1 for r in results if r["wrote_part"])
    in_memory_frames = [r["frame"] for r in results if r["frame"] is not None]

    if wrote_parts == 0 and not in_memory_frames:
        raise ValueError(f"No tidy score partitions found under {tidy_root}")

    if missing_keys:
        print(
            f"[enrich] warning: {missing_keys}/{len(keyed)} rows missing tidy score_key "
            f"under {tidy_root} (will lack mm_* / zero_shot_score)",
            flush=True,
        )

    if parts_dir is not None:
        print(f"[enrich] finished {wrote_parts} chromosome parts under {parts_dir}", flush=True)
        _assert_enriched_scorer_coverage(parts_dir, scorer_names)
        return pd.DataFrame()

    return pd.concat(in_memory_frames, ignore_index=True)

def _assert_enriched_scorer_coverage(parts_dir: Path, scorer_names: set[str]) -> None:
    """Fail loudly if any configured scorer produced zero mm_* features."""

    produced: set[str] = set()
    for scalar_part in sorted(Path(parts_dir).glob("*.scalar.parquet")):
        produced.update(
            col for col in load_examples_parquet(scalar_part).columns if str(col).startswith("mm_")
        )

    if not produced:
        print(
            f"[enrich] warning: no mm_* features found under {parts_dir}; "
            "skipping scorer coverage check",
            flush=True,
        )
        return
    assert_scorer_feature_coverage(produced, scorer_names, context=Path(parts_dir).name)

def score_metadata_tidy_cache(
    *,
    version: str,
    cache_type: str,
    metadata_path: Path,
    scores_dir: Path,
    score_chromosomes: list[str],
    reference_fasta: Path,
    base_model_version: str = "all_folds",
    gtf_feather_path: str | None = None,
    splice_site_starts_feather_path: str | None = None,
    splice_site_ends_feather_path: str | None = None,
    progress_every: int = 100,
    model=None,
) -> dict:
    """GPU-score an existing metadata parquet into tidy chr partitions."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found: {metadata_path}")

    examples = load_examples_parquet(metadata_path)
    tidy_root = tidy_scores_root(scores_dir, version)

    summaries: list[dict] = []
    for chrom in score_chromosomes:
        summary = score_tidy_partition(
            examples=examples,
            output_root=tidy_root,
            chromosome=chrom,
            reference_fasta=reference_fasta,
            base_model_version=base_model_version,
            gtf_feather_path=gtf_feather_path,
            splice_site_starts_feather_path=splice_site_starts_feather_path,
            splice_site_ends_feather_path=splice_site_ends_feather_path,
            progress_every=progress_every,
            dataset_version=version,
            cache_type=cache_type,
            model=model,
        )
        summaries.append(summary)

    feature_probe = examples.head(min(5, len(examples)))
    feature_cols: list[str] = []
    num_with_zero_shot = 0
    if summaries and not all(s.get("skipped") for s in summaries):
        feature_probe = enrich_examples_from_tidy(
            feature_probe,
            tidy_root=tidy_root,
            cache_type=cache_type,
        )
        feature_cols = multimodal_feature_columns(feature_probe)
        num_with_zero_shot = int(
            feature_probe.get("zero_shot_score", pd.Series(dtype=float)).notna().sum()
        )

    if "chromosome" in examples.columns:
        chroms_in_metadata = sorted(
            {normalize_chromosome(c) for c in examples["chromosome"].dropna().unique()}
        )
    else:
        from alphagenome.data import genome

        chroms_in_metadata = sorted(
            {
                normalize_chromosome(c)
                for c in examples["variant_id"]
                .map(lambda v: genome.Variant.from_str(v, genome.VariantFormat.GTEX).chromosome)
                .dropna()
                .unique()
            }
        )

    return {
        "dataset_version": version,
        "cache_type": cache_type,
        "metadata_path": str(metadata_path),
        "tidy_root": str(tidy_root),
        "num_examples": int(len(examples)),
        "num_loci": int(examples["locus_id"].nunique()) if "locus_id" in examples.columns else None,
        "chromosomes_in_metadata": chroms_in_metadata,
        "scoring_chromosomes": [normalize_chromosome(c) for c in score_chromosomes],
        "chr_partitions": summaries,
        "num_with_zero_shot_score": num_with_zero_shot,
        "sample_mm_columns": len(feature_cols),
    }

from dataclasses import dataclass, field

from util import (
    load_examples_parquet,
    save_examples_parquet,
    write_merged_examples_parquet,
)
from util import is_snv_variant_id

@dataclass
class DeriveSpec:
    label: str
    cache_path: Path
    output_path: Path
    tidy_root: Path
    dataset_version: str
    filters: dict = field(default_factory=dict)
    cache_type: str | None = None

def _enrich_key(spec: DeriveSpec) -> tuple[str, str, str]:
    cache_type = spec.cache_type or cache_type_from_metadata_path(spec.cache_path)
    return (
        str(spec.cache_path.resolve()),
        str(spec.tidy_root.resolve()),
        cache_type,
    )

def run_derive_pipeline(
    specs: list[DeriveSpec],
    *,
    enriched_dir: Path,
) -> list[dict]:
    """Enrich each unique (metadata, tidy_root, cache_type) once; write filtered tables."""
    enriched_by_key: dict[tuple[str, str, str], Path] = {}
    summaries: list[dict] = []

    for spec in specs:
        key = _enrich_key(spec)
        if key not in enriched_by_key:
            enriched_by_key[key] = load_or_enrich_cache(
                spec.cache_path,
                tidy_scores_root_path=spec.tidy_root,
                enriched_dir=enriched_dir,
                cache_type=key[2],
            )

        print(f"[derive] filter+merge {spec.label} -> {spec.output_path}", flush=True)
        summary = derive_from_enriched_parts(
            enriched_by_key[key],
            spec.output_path,
            label=spec.label,
            cache_path=spec.cache_path,
            **spec.filters,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    for parts_path in enriched_by_key.values():
        if parts_path.exists():
            shutil.rmtree(parts_path)
    if enriched_dir.is_dir():
        try:
            next(enriched_dir.iterdir())
        except StopIteration:
            enriched_dir.rmdir()

    return summaries

def enriched_parts_dir(enriched_dir: Path, metadata_path: Path) -> Path:
    return enriched_dir / f"{Path(metadata_path).stem}.enriched.parts"

def list_enriched_scalar_parts(parts_dir: Path) -> list[Path]:
    return sorted(Path(parts_dir).glob("*.scalar.parquet"))

def _part_chromosome(part_path: Path) -> str:
    """Chromosome name for an enriched part (``chr1.scalar.parquet`` -> ``chr1``)."""
    return Path(part_path).name.split(".")[0]

def load_or_enrich_cache(
    metadata_path: Path,
    *,
    tidy_scores_root_path: Path,
    enriched_dir: Path,
    cache_type: str | None = None,
) -> Path:
    """Enrich metadata from tidy scores; write per-chromosome parquet parts."""
    parts_dir = enriched_parts_dir(enriched_dir, metadata_path)
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    resolved_cache_type = cache_type or cache_type_from_metadata_path(metadata_path)
    print(
        f"[derive] enrich {metadata_path.name} "
        f"({tidy_scores_basename(resolved_cache_type)}) -> {parts_dir.name}/",
        flush=True,
    )
    metadata = load_examples_parquet(metadata_path)
    enrich_examples_from_tidy(
        metadata,
        tidy_root=Path(tidy_scores_root_path),
        cache_type=resolved_cache_type,
        parts_dir=parts_dir,
    )
    return parts_dir

def join_scored_examples(
    cache: pd.DataFrame,
    *,
    high_pip_threshold: float | None = None,
    splits: list[str] | None = None,
    require_multimodal_features: bool = False,
) -> pd.DataFrame:
    df = cache.copy()

    df = df[df["variant_id"].map(is_snv_variant_id)].copy()

    if high_pip_threshold is not None:
        df = df[df["pip"] >= float(high_pip_threshold)].copy()

    if splits is not None:
        df = df[df["split"].isin(splits)].copy()

    if require_multimodal_features:
        feature_cols = multimodal_feature_columns(df)
        if not feature_cols:
            raise ValueError("No multimodal feature columns (mm_*) found in cache.")
        df = df[df[feature_cols].notna().any(axis=1)].copy()

    return df.reset_index(drop=True)

def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value

def derive_parts_dir_for(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.parent / f"{output_path.stem}.derive.parts"

def _parquet_num_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    from util import _PARQUET_THRIFT_KWARGS

    return int(pq.ParquetFile(path, **_PARQUET_THRIFT_KWARGS).metadata.num_rows)

def _build_merge_summary(
    output_path: Path,
    written_parts: list[Path],
    *,
    label: str | None,
    cache_path: Path | None,
    parts_dir: Path,
    filters: dict,
    merge_only: bool,
) -> dict:
    if output_path.exists():
        num_examples_out = _parquet_num_rows(output_path)
    else:
        num_examples_out = sum(_parquet_num_rows(part) for part in written_parts)
    return {
        "label": label or output_path.stem,
        "cache": str(cache_path or output_path),
        "output": str(output_path),
        "derive_parts": str(parts_dir),
        "num_examples_out": int(num_examples_out),
        "num_parts_merged": int(len(written_parts)),
        "filters": filters,
        "merge_only": merge_only,
    }

def derive_from_enriched_parts(
    parts_dir: Path,
    output_path: Path,
    *,
    label: str | None = None,
    cache_path: Path | None = None,
    **kwargs,
) -> dict:
    """Filter enriched chr parts and merge to one parquet."""
    parts = list_enriched_scalar_parts(parts_dir)
    if not parts:
        raise FileNotFoundError(f"No enriched scalar parts under {parts_dir}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    derive_parts_dir = derive_parts_dir_for(output_path)

    if derive_parts_dir.exists():
        shutil.rmtree(derive_parts_dir)
    derive_parts_dir.mkdir(parents=True)

    written_parts: list[Path] = []
    num_examples_in = 0
    num_examples_out = 0
    loci_out: set = set()
    split_counts: dict = {}

    for scalar_part in parts:
        scalar_df = load_examples_parquet(scalar_part)
        num_examples_in += len(scalar_df)
        derived = prune_null_mm_columns(join_scored_examples(scalar_df, **kwargs))
        del scalar_df

        if derived.empty:
            del derived
            continue

        out_part = derive_parts_dir / f"{_part_chromosome(scalar_part)}.parquet"
        save_examples_parquet(derived, out_part)
        written_parts.append(out_part)

        num_examples_out += len(derived)
        if "locus_id" in derived.columns:
            loci_out.update(derived["locus_id"].unique())
        if "split" in derived.columns:
            for split_val, count in derived["split"].value_counts(dropna=False).items():
                split_counts[split_val] = split_counts.get(split_val, 0) + int(count)
        del derived

    write_merged_examples_parquet(written_parts, output_path)

    filters = _json_safe({k: v for k, v in kwargs.items() if v is not None and v is not False})
    summary = _build_merge_summary(
        output_path,
        written_parts,
        label=label,
        cache_path=cache_path,
        parts_dir=derive_parts_dir_for(output_path),
        filters=filters,
        merge_only=False,
    )
    summary["enriched_parts"] = str(parts_dir)
    summary["num_examples_in"] = int(num_examples_in)
    summary["num_loci_out"] = int(len(loci_out))
    if split_counts:
        summary["splits"] = split_counts
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))

    if derive_parts_dir.exists():
        shutil.rmtree(derive_parts_dir)
    return summary

SCORING_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]
BASE_MODEL_VERSION = "all_folds"
KAGGLE_MODEL_VERSION = 1
PROGRESS_EVERY = 100


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-score v10/v11 lead + background tidy caches for autosomes chr1–22."
    )
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument(
        "--gtf-feather-path-v10",
        type=str,
        required=True,
        help="GENCODE v39 AlphaGenome GTF feather (v10).",
    )
    parser.add_argument(
        "--gtf-feather-path-v11",
        type=str,
        required=True,
        help="GENCODE v47 AlphaGenome GTF feather (v11).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for tidy AlphaGenome score caches (tidy/v10, tidy/v11, …).",
    )
    parser.add_argument(
        "--v10-lead-metadata",
        type=Path,
        required=True,
        help="Lead metadata parquet from step 01.",
    )
    parser.add_argument(
        "--v11-lead-metadata",
        type=Path,
        required=True,
        help="Lead metadata parquet from step 01.",
    )
    parser.add_argument(
        "--v10-background-metadata",
        type=Path,
        required=True,
        help="Background metadata parquet from step 01.",
    )
    parser.add_argument(
        "--v11-background-metadata",
        type=Path,
        required=True,
        help="Background metadata parquet from step 01.",
    )
    args = parser.parse_args()

    run_chr_score_caches(
        score_chromosomes=list(SCORING_CHROMOSOMES),
        reference_fasta=args.reference_fasta,
        base_model_version=BASE_MODEL_VERSION,
        kaggle_model_version=KAGGLE_MODEL_VERSION,
        gtf_feather_path_v10=args.gtf_feather_path_v10,
        gtf_feather_path_v11=args.gtf_feather_path_v11,
        progress_every=PROGRESS_EVERY,
        output_dir=args.output_dir,
        v10_lead_metadata=args.v10_lead_metadata,
        v11_lead_metadata=args.v11_lead_metadata,
        v10_background_metadata=args.v10_background_metadata,
        v11_background_metadata=args.v11_background_metadata,
    )

if __name__ == "__main__":
    main()
