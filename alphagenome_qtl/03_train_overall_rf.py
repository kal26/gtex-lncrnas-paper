#!/usr/bin/env python3
"""Step 3 — train_overall_rf: join scored caches into RF tables, then train.

All written artifacts land under ``--output_dir``. Requires step-01/02 inputs and
always rebuilds RF example tables before training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gene_biotype_map import CAUSALITY_BIOTYPE_FILTERS
from rf import TrainCausalityRFConfig, load_cv_evaluation_report, train_causality_rf_from_config
from util import load_sibling_script
_mod = load_sibling_script('02_score_cohorts.py')
DeriveSpec = _mod.DeriveSpec
run_derive_pipeline = _mod.run_derive_pipeline

# Paper RF / cohort defaults (Fig 3f / Supp 10).
HIGH_PIP_THRESHOLD = 0.9
N_DISTANCE_BINS = 10
SEED = 42
N_ESTIMATORS = 100
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 1


def _build_rf_tables(
    *,
    lead_metadata: Path,
    background_metadata: Path,
    tidy_root: Path,
    enriched_dir: Path,
    multimodal_examples: Path,
    allpairs_background: Path,
    dataset_version: str,
    high_pip_threshold: float,
) -> None:
    specs = [
        DeriveSpec(
            label=f"{dataset_version}_rf_lead",
            cache_path=lead_metadata,
            output_path=multimodal_examples,
            tidy_root=tidy_root,
            dataset_version=dataset_version,
            filters={
                "require_multimodal_features": True,
                "high_pip_threshold": high_pip_threshold,
            },
        ),
        DeriveSpec(
            label=f"{dataset_version}_rf_background",
            cache_path=background_metadata,
            output_path=allpairs_background,
            tidy_root=tidy_root,
            dataset_version=dataset_version,
            filters={"require_multimodal_features": True},
        ),
    ]
    summaries = run_derive_pipeline(specs, enriched_dir=enriched_dir)
    print(json.dumps({"num_derived": len(summaries)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join scored lead/background caches into RF example parquets under "
            "--output_dir, then train/evaluate the overall causality random forest."
        )
    )
    parser.add_argument("--dataset-version", required=True, choices=["v10", "v11"])
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help=(
            "Root for RF tables, enriched intermediates, and "
            "{version}_causality_rf[_{biotype}]/ train artifacts."
        ),
    )
    parser.add_argument(
        "--lead-metadata",
        type=Path,
        required=True,
        help="Lead metadata parquet from steps 01–02.",
    )
    parser.add_argument(
        "--background-metadata",
        type=Path,
        required=True,
        help="Background metadata parquet from steps 01–02.",
    )
    parser.add_argument(
        "--tidy-root",
        type=Path,
        required=True,
        help="Tidy AlphaGenome score cache root from step 02.",
    )
    parser.add_argument("--full_gtf", type=Path, required=True)
    parser.add_argument(
        "--biotype-filter",
        default=None,
        choices=list(CAUSALITY_BIOTYPE_FILTERS),
        help=(
            "Restrict train/eval examples to one causality biotype or the "
            "lncRNA_protein_coding subset (excludes pseudogene)."
        ),
    )
    parser.add_argument(
        "--gene-classification",
        type=Path,
        default=None,
        help="GENCODE v39/v47 classification TSV (required with --biotype-filter).",
    )
    args = parser.parse_args()

    if args.biotype_filter is not None and args.gene_classification is None:
        raise SystemExit("--gene-classification is required with --biotype-filter")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    multimodal_examples = (
        args.output_dir / f"{args.dataset_version}_qtl_variant_features.parquet"
    )
    allpairs_background = (
        args.output_dir
        / f"{args.dataset_version}_allpairs_background.multimodal.scored.parquet"
    )
    enriched_dir = args.output_dir / "enriched"
    train_dirname = (
        f"{args.dataset_version}_causality_rf_{args.biotype_filter}"
        if args.biotype_filter
        else f"{args.dataset_version}_causality_rf"
    )
    train_dir = args.output_dir / train_dirname

    _build_rf_tables(
        lead_metadata=args.lead_metadata,
        background_metadata=args.background_metadata,
        tidy_root=args.tidy_root,
        enriched_dir=enriched_dir,
        multimodal_examples=multimodal_examples,
        allpairs_background=allpairs_background,
        dataset_version=args.dataset_version,
        high_pip_threshold=HIGH_PIP_THRESHOLD,
    )

    cfg = TrainCausalityRFConfig(
        dataset_version=args.dataset_version,
        multimodal_examples_parquet=str(multimodal_examples),
        allpairs_background_parquet=str(allpairs_background),
        output_dir=str(train_dir),
        gtf_path=str(args.full_gtf),
        biotype_filter=args.biotype_filter,
        gene_classification=(
            str(args.gene_classification) if args.gene_classification else None
        ),
        high_pip_threshold=HIGH_PIP_THRESHOLD,
        n_distance_bins=N_DISTANCE_BINS,
        seed=SEED,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )
    cv_report_path = train_causality_rf_from_config(cfg)

    biotype = args.biotype_filter or "all_genes"
    model_label = f"{args.dataset_version}_{biotype}"

    report = load_cv_evaluation_report(cv_report_path, model_label=model_label)
    out_report = train_dir / "evaluation_report.json"
    out_report.write_text(json.dumps(report, indent=2))

    # Explicit zero-shot summary (full CV holdouts; Fig 3f uses step-04 matched-N).
    zero_shot_path = train_dir / "zero_shot_auroc.tsv"
    fold_csv = train_dir / "cv_fold_results.csv"
    if not fold_csv.exists():
        raise FileNotFoundError(f"Missing {fold_csv} after causality RF training.")
    fold_tbl = pd.read_csv(fold_csv)
    if "matched_track_zero_shot_auroc" not in fold_tbl.columns:
        raise ValueError(
            f"{fold_csv} missing matched_track_zero_shot_auroc column."
        )
    vals = fold_tbl["matched_track_zero_shot_auroc"].astype(float)
    pd.DataFrame(
        [
            {
                "eval_biotype": args.biotype_filter or "all_genes",
                "mean_auroc": float(vals.mean()),
                "std_auroc": float(vals.std(ddof=0)),
                "n_folds": int(vals.notna().sum()),
            }
        ]
    ).to_csv(zero_shot_path, sep="\t", index=False)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "train_dir": str(train_dir),
                "multimodal_examples": str(multimodal_examples),
                "allpairs_background": str(allpairs_background),
                "cv_report": str(cv_report_path),
                "evaluation_report": str(out_report),
                "zero_shot_tsv": str(zero_shot_path),
                "model_label": model_label,
                "overall": report.get("overall", {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
