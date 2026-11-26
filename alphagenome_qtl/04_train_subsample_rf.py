#!/usr/bin/env python3
"""Step 4 — learning_curves_rf: shared train-N label-subsample RFs (Fig 3f).

Trains across eval biotypes at subsampled label counts, then writes
``qtl_causality_auroc.tsv`` under ``--output_dir`` with matched-N RF curves,
zero-shot baseline rows, and fold-aggregated v10/v11 catalog-holdout auROCs
(Supp 10c).

Shared train-N grid across biotypes; evaluation always uses the full
rebalanced chromosome-fold holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rf import (
    CAUSALITY_CV_FOLDS,
    DEFAULT_EVAL_BIOTYPES,
    DEFAULT_GRID,
    DEFAULT_N_REPLICATES,
    DEFAULT_TRAIN_BIOTYPES,
    append_catalog_metrics_to_learning_curves,
    build_catalog_comparison_metrics,
    export_combined_label_subsample_curves,
    parse_train_biotypes,
    run_label_subsample,
)

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train v11 label-subsample causality RFs for each eval biotype, "
            "then export qtl_causality_auroc.tsv (matched-N curves, zero-shot "
            "baseline, and fold-aggregated v10/v11 catalog-holdout auROCs)."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help=(
            "Shared AlphaGenome output root (step 03 train dirs + this step's "
            "qtl_causality_auroc.tsv)."
        ),
    )
    parser.add_argument(
        "--gene-classification",
        type=Path,
        default=None,
        help=(
            "GENCODE v39/v47 gene classification TSV (novel/existing lncRNA labels). "
            "Required unless --catalog-metrics-only."
        ),
    )
    parser.add_argument(
        "--catalog-metrics-only",
        action="store_true",
        help=(
            "Skip label-subsample training/export; only rescore catalog holdouts "
            "and append fold-aggregated auROCs into the existing "
            "qtl_causality_auroc.tsv."
        ),
    )
    args = parser.parse_args()

    if args.gene_classification is None and not args.catalog_metrics_only:
        raise SystemExit("--gene-classification is required unless --catalog-metrics-only")

    if args.catalog_metrics_only:
        catalog_metrics = build_catalog_comparison_metrics(
            args.output_dir, n_folds=CAUSALITY_CV_FOLDS
        )
        curves_path = append_catalog_metrics_to_learning_curves(
            args.output_dir, catalog_metrics
        )
        print(
            json.dumps(
                {
                    "mode": "catalog_metrics_only",
                    "catalog_metrics_rows": int(len(catalog_metrics)),
                    "qtl_causality_auroc_tsv": str(curves_path),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    eval_biotypes = list(DEFAULT_EVAL_BIOTYPES)
    train_biotypes = parse_train_biotypes(",".join(DEFAULT_TRAIN_BIOTYPES))
    grid = list(DEFAULT_GRID)
    grid_arg = ",".join(str(n) for n in DEFAULT_GRID)

    for eval_biotype in eval_biotypes:
        run_label_subsample(
            output_root=args.output_dir,
            output_dir=None,
            eval_biotype=eval_biotype,
            grid_arg=grid_arg,
            train_biotypes=train_biotypes,
            gene_classification=args.gene_classification,
            n_folds=CAUSALITY_CV_FOLDS,
            n_replicates=DEFAULT_N_REPLICATES,
            seed=SEED,
        )

    export_combined_label_subsample_curves(
        args.output_dir,
        eval_biotypes=eval_biotypes,
        grid=grid,
        train_biotypes=train_biotypes,
        n_replicates=DEFAULT_N_REPLICATES,
        n_folds=CAUSALITY_CV_FOLDS,
        gene_classification=args.gene_classification,
    )

    catalog_metrics = build_catalog_comparison_metrics(
        args.output_dir, n_folds=CAUSALITY_CV_FOLDS
    )
    curves_path = append_catalog_metrics_to_learning_curves(
        args.output_dir, catalog_metrics
    )
    print(
        json.dumps(
            {
                "catalog_metrics_rows": int(len(catalog_metrics)),
                "qtl_causality_auroc_tsv": str(curves_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
