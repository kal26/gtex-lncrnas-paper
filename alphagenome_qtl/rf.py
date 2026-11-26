"""Causality random forests: model, CV, overall train/eval, matched-N learning curves."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from util import multimodal_feature_columns

def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return float("nan")
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return float("nan")
    return float(sum(value * weight for value, weight in pairs) / total_weight)

def paper_causality_metrics(
    df: pd.DataFrame,
    *,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    labels_column: str = "causality_label",
    score_column: str | None = None,
) -> dict[str, float]:
    """Tissue-weighted auROC / average precision for causality labels."""
    
    work = df.reset_index(drop=True)
    if labels is None:
        labels = work[labels_column].astype(int).to_numpy()
    if scores is None:
        if score_column is None:
            raise ValueError("score_column required when scores are not provided.")
        scores = work[score_column].to_numpy(dtype=float)

    if len(np.unique(labels)) < 2:
        return {
            "causality_auroc": float("nan"),
            "causality_avg_precision": float("nan"),
        }

    if "tissue" not in work.columns or work["tissue"].nunique() <= 1:
        return {
            "causality_auroc": float(roc_auc_score(labels, scores)),
            "causality_avg_precision": float(average_precision_score(labels, scores)),
        }

    auroc_pairs: list[tuple[float, float]] = []
    ap_pairs: list[tuple[float, float]] = []
    for _, sub in work.groupby("tissue"):
        idx = sub.index.to_numpy()
        tissue_labels = labels[idx]
        tissue_scores = scores[idx]
        if len(np.unique(tissue_labels)) < 2:
            continue
        weight = float(len(sub))
        auroc_pairs.append((float(roc_auc_score(tissue_labels, tissue_scores)), weight))
        ap_pairs.append((float(average_precision_score(tissue_labels, tissue_scores)), weight))

    return {
        "causality_auroc": _weighted_mean(auroc_pairs),
        "causality_avg_precision": _weighted_mean(ap_pairs),
    }

@dataclass
class CausalityRFConfig:
    """Paper defaults: RandomForestClassifier max_depth=5, other hyperparameters default."""

    n_estimators: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 1
    class_weight: str | None = None
    random_state: int = 42

class CausalityRandomForest:
    def __init__(self, cfg: CausalityRFConfig | None = None) -> None:
        self.cfg = cfg or CausalityRFConfig()
        self.feature_columns: list[str] = []
        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=self.cfg.n_estimators,
                        max_depth=self.cfg.max_depth,
                        min_samples_leaf=self.cfg.min_samples_leaf,
                        class_weight=self.cfg.class_weight,
                        random_state=self.cfg.random_state,
                    ),
                ),
            ]
        )

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        cols = multimodal_feature_columns(df)
        if not cols:
            return []
        values = df[cols].to_numpy(dtype=float)
        keep = np.any(np.isfinite(values), axis=0)
        return [col for col, use in zip(cols, keep) if use]

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        # Align to the saved per-scorer feature list; impute missing columns.
        values = df.reindex(columns=self.feature_columns).to_numpy(dtype=float)
        return np.abs(values)

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> dict:
        self.feature_columns = self._select_feature_columns(train_df)
        if not self.feature_columns:
            raise ValueError("Training frame has no mm_* feature columns.")

        x_train = self._feature_matrix(train_df)
        y_train = train_df["causality_label"].astype(int).to_numpy()
        self.model.fit(x_train, y_train)

        metrics = {
            "train_n": int(len(train_df)),
            "train_positives": int((y_train == 1).sum()),
            "train_negatives": int((y_train == 0).sum()),
            "n_features": int(len(self.feature_columns)),
        }
        if valid_df is not None and not valid_df.empty:
            metrics.update(self.evaluate(valid_df, prefix="valid_"))
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._feature_matrix(df)
        return self.model.predict_proba(x)[:, 1]

    def evaluate(self, df: pd.DataFrame, *, prefix: str = "") -> dict[str, float]:
        labels = df["causality_label"].astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            return {
                f"{prefix}causality_auroc": float("nan"),
                f"{prefix}causality_avg_precision": float("nan"),
                f"{prefix}n": float(len(df)),
            }
        scores = self.predict_proba(df)
        metrics = paper_causality_metrics(df, labels=labels, scores=scores)
        return {
            f"{prefix}causality_auroc": metrics["causality_auroc"],
            f"{prefix}causality_avg_precision": metrics["causality_avg_precision"],
            f"{prefix}n": float(len(df)),
        }

    def feature_importances(self) -> pd.DataFrame:
        classifier = self.model.named_steps["classifier"]
        importances = classifier.feature_importances_
        if len(importances) != len(self.feature_columns):
            raise ValueError(
                "Feature/importance length mismatch: "
                f"{len(self.feature_columns)} feature names vs {len(importances)} importances."
            )
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

def matched_track_zero_shot_scores(df: pd.DataFrame) -> np.ndarray:
    """Matched gene+tissue-track zero-shot score.

    Uses ``zero_shot_score`` — the AlphaGenome GeneMask effect on the variant's
    target gene in the QTL's own (matched) tissue. This is the single track that
    is both gene- and tissue-matched, not the max over all mm_* tracks. Variants
    with no matched-track score are treated as null effect (0), so the baseline is
    evaluated over the full cohort without falling back to any other track.
    """
    if "zero_shot_score" not in df.columns:
        raise ValueError("Matched-track zero-shot baseline requires a zero_shot_score column.")
    return df["zero_shot_score"].astype(float).abs().fillna(0.0).to_numpy()

def _evaluate_score_fn(df: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    labels = df["causality_label"].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        return {"causality_auroc": float("nan"), "causality_avg_precision": float("nan"), "n": float(len(df))}
    metrics = paper_causality_metrics(df, labels=labels, scores=scores)
    return {
        **metrics,
        "n": float(len(df)),
    }

def evaluate_matched_track_zero_shot_baseline(df: pd.DataFrame) -> dict[str, float]:
    return _evaluate_score_fn(df, matched_track_zero_shot_scores(df))

def bootstrap_causality_ci(
    df: pd.DataFrame,
    score_fn,
    *,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return {
            "causality_auroc": (float("nan"), float("nan")),
            "causality_avg_precision": (float("nan"), float("nan")),
        }

    aurocs: list[float] = []
    aps: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = df.iloc[idx]
        metrics = score_fn(sub)
        if np.isfinite(metrics.get("causality_auroc", float("nan"))):
            aurocs.append(float(metrics["causality_auroc"]))
        if np.isfinite(metrics.get("causality_avg_precision", float("nan"))):
            aps.append(float(metrics["causality_avg_precision"]))

    def _ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return float("nan"), float("nan")
        low, high = np.percentile(vals, [2.5, 97.5])
        return float(low), float(high)

    return {
        "causality_auroc": _ci(aurocs),
        "causality_avg_precision": _ci(aps),
    }

def fit_rf_and_evaluate_holdout(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    rf_cfg: CausalityRFConfig | None = None,
) -> tuple[CausalityRandomForest, dict[str, float]]:
    """Fit one RF on ``train_df`` and score ``test_df`` (shared by overall CV and subsample)."""
    model = CausalityRandomForest(rf_cfg)
    model.fit(train_df)
    return model, model.evaluate(test_df)

def save_feature_importances(importances: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(path, index=False)

def save_run_metadata(output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

import warnings
from collections import defaultdict

from util import load_examples_parquet, load_sibling_script

_cohorts = load_sibling_script("01_build_cohorts.py")
build_causality_cohort = _cohorts.build_causality_cohort
summarize_causality_pairs = _cohorts.summarize_causality_pairs
rebalance_cohort_exact = _cohorts.rebalance_cohort_exact
from gene_biotype_map import (
    CAUSALITY_EVAL_BIOTYPES,
    filter_cohort_to_causality_biotype,
    filter_to_causality_biotype,
)
from util import CAUSALITY_5_FOLD_CHROMOSOMES, filter_snv_examples

# Supp Fig 10c: score each version×biotype CV holdout with zero-shot + both RFs.
CATALOG_HOLDOUT_EVALS: tuple[tuple[str, str], ...] = (
    ("v10", "protein_coding"),
    ("v10", "lncRNA"),
    ("v11", "protein_coding"),
    ("v11", "lncRNA"),
)
CATALOG_HOLDOUT_MODEL_BIOTYPE = "lncRNA_protein_coding"
CATALOG_HOLDOUT_KEYS: tuple[str, ...] = tuple(
    f"{version}_{biotype}" for version, biotype in CATALOG_HOLDOUT_EVALS
)
CATALOG_SCORE_METHODS: tuple[str, ...] = ("zero_shot", "rf_v10", "rf_v11")
CATALOG_EVAL_BIOTYPES = frozenset({"lncRNA", "protein_coding"})
VERSION = "v11"

def is_catalog_comparison_row(df: pd.DataFrame) -> pd.Series:
    """Supp 10c rows: zero_shot / rf_v10 / rf_v11 on catalog holdout evals."""
    return (
        df["train_biotype"].isin(CATALOG_SCORE_METHODS)
        & df["eval_biotype"].isin(CATALOG_EVAL_BIOTYPES)
    )

# --- Biotype-stratified evaluation ---

def _mean_metric(values: list[float]) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")

def evaluate_cohort_metrics(
    cohort: pd.DataFrame,
    *,
    model: CausalityRandomForest | None = None,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict[str, float]:
    if cohort.empty or len(np.unique(cohort["causality_label"].astype(int))) < 2:
        return {
            "causality_auroc": float("nan"),
            "causality_avg_precision": float("nan"),
            "n": float(len(cohort)),
        }

    if model is None:
        metrics = evaluate_matched_track_zero_shot_baseline(cohort)
    else:
        scored = cohort.copy()
        scored["causality_score"] = model.predict_proba(scored)
        metrics = model.evaluate(scored)

    if n_bootstrap > 0:
        score_fn = (
            evaluate_matched_track_zero_shot_baseline
            if model is None
            else lambda sub: model.evaluate(sub.assign(causality_score=model.predict_proba(sub)))
        )
        cis = bootstrap_causality_ci(cohort, score_fn, n_boot=n_bootstrap, seed=seed)
        for metric, (low, high) in cis.items():
            metrics[f"{metric}_ci_low"] = low
            metrics[f"{metric}_ci_high"] = high
    return metrics

def evaluate_biotype_strata(
    cohort: pd.DataFrame,
    *,
    model: CausalityRandomForest | None = None,
    gene_classification: Path | None = None,
    dataset_version: str = "v11",
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate full cohort plus lncRNA / protein_coding / pseudogene subsets."""
    out: dict[str, dict[str, float]] = {
        "all": evaluate_cohort_metrics(
            cohort,
            model=model,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    }
    for biotype in CAUSALITY_EVAL_BIOTYPES:
        sub = filter_cohort_to_causality_biotype(
            cohort,
            biotype,
            gene_classification=gene_classification,
            version=dataset_version,
        )
        out[biotype] = evaluate_cohort_metrics(
            sub,
            model=model,
            n_bootstrap=n_bootstrap,
            seed=seed + hash(biotype) % 10_000,
        )
    return out

def aggregate_biotype_fold_metrics(
    fold_metrics: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Mean metrics across CV folds for each biotype stratum."""
    aggregated: dict[str, dict[str, float]] = {}
    for biotype, rows in fold_metrics.items():
        if not rows:
            continue
        aggregated[biotype] = {
            "causality_auroc": _mean_metric([row.get("causality_auroc", float("nan")) for row in rows]),
            "causality_avg_precision": _mean_metric(
                [row.get("causality_avg_precision", float("nan")) for row in rows]
            ),
            "n": float(sum(row.get("n", 0.0) for row in rows)),
        }
    return aggregated

def init_biotype_fold_metrics() -> dict[str, list[dict[str, float]]]:
    return defaultdict(list)

def load_cv_evaluation_report(path: Path, *, model_label: str) -> dict:
    """Load a CV evaluation JSON written by ``train_causality_rf`` / fold CV."""
    raw = json.loads(path.read_text())
    return {
        "model_label": model_label,
        "evaluation_mode": "causality_rf_cv",
        "eval_subset": "chromosome_5_fold_cv",
        "biotype_filter": raw.get("biotype_filter"),
        "overall": raw.get("overall", {}),
        "by_biotype": raw.get("by_biotype", {}),
        "cross_validate_folds": raw.get("cross_validate_folds"),
        "fold_results": raw.get("fold_results", []),
        "num_features": raw.get("num_features"),
    }

def catalog_holdout_balance_seed(fold: int, version: str, biotype: str) -> int:
    idx = CATALOG_HOLDOUT_KEYS.index(f"{version}_{biotype}")
    return 137 + 10_000 * fold + 1_000_000 * idx

def _catalog_holdout_model_dir(output_dir: Path, version: str) -> Path:
    return Path(output_dir) / f"{version}_causality_rf_{CATALOG_HOLDOUT_MODEL_BIOTYPE}"

def _catalog_holdout_train_cohort_path(output_dir: Path, version: str, fold: int) -> Path:
    return _catalog_holdout_model_dir(output_dir, version) / f"fold_{fold}" / "train_cohort.parquet"

def _catalog_holdout_test_cohort_path(
    output_dir: Path, version: str, biotype: str, fold: int
) -> Path:
    return (
        Path(output_dir)
        / f"{version}_causality_rf_{biotype}"
        / f"fold_{fold}"
        / "test_cohort.parquet"
    )

def _fit_catalog_holdout_model(train_cohort_path: Path) -> CausalityRandomForest:
    """Refit the fold RF from the step-03 train cohort (no pickled model)."""
    train = pd.read_parquet(train_cohort_path)
    model = CausalityRandomForest()
    model.fit(train)
    return model

def _tissue_weighted_auroc_from_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    tissues: pd.Series,
) -> float:
    """Tissue-weighted auROC for a single scored holdout slice."""
    work = pd.DataFrame({"label": labels, "score": scores, "tissue": tissues})
    return float(
        paper_causality_metrics(
            work,
            labels=work["label"].to_numpy(),
            scores=work["score"].to_numpy(dtype=float),
        ).get("causality_auroc", float("nan"))
    )

def catalog_fold_metrics_from_scored(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-variant catalog scores to fold-aggregated learning_curves rows.

    Expects columns ``dataset_version, biotype, fold, tissue, label`` plus score
    columns ``zero_shot_abs|zero_shot, rf_v10, rf_v11``.

    Returns one row per (dataset_version, eval_biotype, method) with mean ±
    fold-level 95% CI across chromosome folds (same schema as matched-N /
    zero-shot curve rows). Also attaches paired t-test p-values vs zero-shot
    and vs rf_v10 for Supp 10c significance brackets.
    """
    from scipy import stats as sp_stats

    score_col_map = {
        "zero_shot": "zero_shot" if "zero_shot" in scored.columns else "zero_shot_abs",
        "rf_v10": "rf_v10",
        "rf_v11": "rf_v11",
    }
    fold_rows: list[dict] = []
    group_cols = ["dataset_version", "biotype", "fold"]
    for keys, sub in scored.groupby(group_cols, sort=False):
        dataset_version, biotype, fold = keys
        labels = sub["label"].astype(int).to_numpy()
        tissues = sub["tissue"]
        for method, col in score_col_map.items():
            auroc = _tissue_weighted_auroc_from_scores(
                labels, sub[col].to_numpy(dtype=float), tissues
            )
            fold_rows.append(
                {
                    "dataset_version": dataset_version,
                    "eval_biotype": biotype,
                    "train_biotype": method,
                    "fold": int(fold),
                    "auroc": float(auroc),
                }
            )
    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_version",
                "eval_biotype",
                "train_biotype",
                "n_train",
                "fold",
                "mean",
                "ci_lower",
                "ci_upper",
                "std",
                "n_folds",
                "p_vs_zero_shot",
                "p_vs_rf_v10",
            ]
        )

    rows: list[dict] = []
    for (dataset_version, biotype), sub in fold_df.groupby(
        ["dataset_version", "eval_biotype"], sort=False
    ):
        method_vals = {
            method: (
                sub.loc[sub["train_biotype"] == method, ["fold", "auroc"]]
                .drop_duplicates("fold")
                .sort_values("fold")["auroc"]
                .to_numpy(dtype=float)
            )
            for method in CATALOG_SCORE_METHODS
        }
        p_vs_zero: dict[str, float] = {}
        p_vs_v10: dict[str, float] = {}
        zero_vals = method_vals["zero_shot"]
        v10_vals = method_vals["rf_v10"]
        for method, vals in method_vals.items():
            if method != "zero_shot" and len(vals) > 1 and len(vals) == len(zero_vals):
                p_vs_zero[method] = float(sp_stats.ttest_rel(zero_vals, vals).pvalue)
            else:
                p_vs_zero[method] = float("nan")
            if method == "rf_v11" and len(vals) > 1 and len(vals) == len(v10_vals):
                p_vs_v10[method] = float(sp_stats.ttest_rel(v10_vals, vals).pvalue)
            else:
                p_vs_v10[method] = float("nan")

        for method, vals in method_vals.items():
            stats = _auroc_summary_fold_level(vals)
            rows.append(
                {
                    "dataset_version": dataset_version,
                    "eval_biotype": biotype,
                    "train_biotype": method,
                    "n_train": np.nan,
                    "fold": np.nan,
                    "mean": stats["mean"],
                    "ci_lower": stats["ci_lower"],
                    "ci_upper": stats["ci_upper"],
                    "std": stats["std"],
                    "n_folds": stats["n_folds"],
                    "p_vs_zero_shot": p_vs_zero[method],
                    "p_vs_rf_v10": p_vs_v10[method],
                }
            )
    return pd.DataFrame(rows)

def build_catalog_comparison_metrics(
    output_dir: Path,
    *,
    n_folds: int = len(CAUSALITY_5_FOLD_CHROMOSOMES),
) -> pd.DataFrame:
    """Score catalog×biotype CV holdouts; return fold-aggregated auROC rows.

    Returns mean ± fold-level 95% CI rows for ``qtl_causality_auroc.tsv``
    (Supp 10c), matching the matched-N / zero-shot curve schema. Does not write
    a per-variant scores parquet. Requires step-03 biotype train dirs for
    ``protein_coding`` / ``lncRNA`` holdouts and ``lncRNA_protein_coding``
    train cohorts for both v10 and v11 (RFs are refit in memory).
    """
    output_dir = Path(output_dir)
    parts: list[pd.DataFrame] = []

    for version, biotype in CATALOG_HOLDOUT_EVALS:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            for fold in range(n_folds):
                cohort_path = _catalog_holdout_test_cohort_path(
                    output_dir, version, biotype, fold
                )
                if not cohort_path.exists():
                    raise FileNotFoundError(
                        f"Missing {cohort_path}. Train step-03 with "
                        f"--dataset-version {version} --biotype-filter {biotype}."
                    )
                train_v10_path = _catalog_holdout_train_cohort_path(output_dir, "v10", fold)
                train_v11_path = _catalog_holdout_train_cohort_path(output_dir, "v11", fold)
                if not train_v10_path.exists():
                    raise FileNotFoundError(
                        f"Missing {train_v10_path}. Train step-03 with "
                        f"--dataset-version v10 --biotype-filter "
                        f"{CATALOG_HOLDOUT_MODEL_BIOTYPE}."
                    )
                if not train_v11_path.exists():
                    raise FileNotFoundError(
                        f"Missing {train_v11_path}. Train step-03 with "
                        f"--dataset-version v11 --biotype-filter "
                        f"{CATALOG_HOLDOUT_MODEL_BIOTYPE}."
                    )
                cohort = rebalance_cohort_exact(
                    pd.read_parquet(cohort_path),
                    seed=catalog_holdout_balance_seed(fold, version, biotype),
                )
                model_v10 = _fit_catalog_holdout_model(train_v10_path)
                model_v11 = _fit_catalog_holdout_model(train_v11_path)
                scored = pd.DataFrame(
                    {
                        "dataset_version": version,
                        "biotype": biotype,
                        "fold": fold,
                        "tissue": cohort["tissue"].to_numpy(),
                        "label": cohort["causality_label"].astype(int).to_numpy(),
                        "zero_shot_abs": matched_track_zero_shot_scores(cohort),
                        "rf_v10": model_v10.predict_proba(cohort),
                        "rf_v11": model_v11.predict_proba(cohort),
                    }
                )
                parts.append(scored)

    scored_all = pd.concat(parts, ignore_index=True)
    return catalog_fold_metrics_from_scored(scored_all)

def append_catalog_metrics_to_learning_curves(
    output_root: Path,
    catalog_metrics: pd.DataFrame,
) -> Path:
    """Replace any prior catalog rows in ``qtl_causality_auroc.tsv``."""
    out = combined_curves_tsv_path(output_root)
    curve_cols = [
        "dataset_version",
        "eval_biotype",
        "train_biotype",
        "n_train",
        "fold",
        "mean",
        "ci_lower",
        "ci_upper",
        "std",
        "n_folds",
        "p_vs_zero_shot",
        "p_vs_rf_v10",
    ]
    if out.exists():
        curves = pd.read_csv(out, sep="\t")
    else:
        curves = pd.DataFrame(columns=curve_cols)
    if "dataset_version" not in curves.columns:
        curves["dataset_version"] = VERSION
    if "fold" not in curves.columns:
        curves["fold"] = np.nan
    for col in ("p_vs_zero_shot", "p_vs_rf_v10"):
        if col not in curves.columns:
            curves[col] = np.nan
        if col not in catalog_metrics.columns:
            catalog_metrics = catalog_metrics.copy()
            catalog_metrics[col] = np.nan
    # Keep matched-N rows; replace prior catalog block.
    matched = curves[~is_catalog_comparison_row(curves)].copy()
    matched["dataset_version"] = VERSION
    matched = matched.drop(columns=["p_vs_zero_shot", "p_vs_rf_v10"], errors="ignore")
    matched["p_vs_zero_shot"] = np.nan
    matched["p_vs_rf_v10"] = np.nan
    combined = pd.concat([matched, catalog_metrics], ignore_index=True)
    combined = combined.reindex(columns=curve_cols)
    combined = combined.sort_values(
        ["dataset_version", "eval_biotype", "train_biotype", "n_train", "fold"],
        na_position="first",
    ).reset_index(drop=True)
    write_label_subsample_curves_tsv(combined, out)
    return out

from gene_biotype_map import filter_to_causality_biotype
from util import CAUSALITY_5_FOLD_CHROMOSOMES, all_causality_cv_chromosomes

def run_causality_rf_cross_validation(
    examples: pd.DataFrame,
    *,
    gtf_path: Path,
    output_dir: Path,
    background_examples: pd.DataFrame,
    high_pip_threshold: float = 0.9,
    n_distance_bins: int = 10,
    seed: int = 42,
    rf_cfg: CausalityRFConfig | None = None,
    biotype_filter: str | None = None,
    dataset_version: str = "v11",
    gene_classification: Path | None = None,
) -> dict:
    """Train/evaluate one RF per chromosome fold; return mean auROC across folds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rf_cfg = rf_cfg or CausalityRFConfig()
    all_chromosomes = all_causality_cv_chromosomes()

    if biotype_filter is not None:
        examples = filter_to_causality_biotype(
            examples,
            biotype_filter,
            gene_classification=gene_classification,
            version=dataset_version,
        )
        background_examples = filter_to_causality_biotype(
            background_examples,
            biotype_filter,
            gene_classification=gene_classification,
            version=dataset_version,
        )

    fold_rows: list[dict] = []
    num_features: int | None = None
    biotype_fold_metrics = init_biotype_fold_metrics()

    for fold_idx, test_chromosomes in enumerate(CAUSALITY_5_FOLD_CHROMOSOMES):
        train_chromosomes = all_chromosomes - test_chromosomes
        train_cohort = build_causality_cohort(
            examples,
            gtf_path=gtf_path,
            background_examples=background_examples,
            chromosomes=set(train_chromosomes),
            high_pip_threshold=high_pip_threshold,
            n_distance_bins=n_distance_bins,
            seed=seed + fold_idx,
            require_multimodal_features=True,
        )
        test_cohort = build_causality_cohort(
            examples,
            gtf_path=gtf_path,
            background_examples=background_examples,
            chromosomes=set(test_chromosomes),
            high_pip_threshold=high_pip_threshold,
            n_distance_bins=n_distance_bins,
            seed=seed + 100 + fold_idx,
            require_multimodal_features=True,
        )
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        row = {
            "fold": fold_idx,
            "test_chromosomes": sorted(test_chromosomes),
            "train_pairs": summarize_causality_pairs(train_cohort),
            "test_pairs": summarize_causality_pairs(test_cohort),
        }

        if train_cohort.empty or test_cohort.empty:
            raise ValueError(
                f"Empty train or test cohort for fold {fold_idx} "
                f"(train_n={len(train_cohort)}, test_n={len(test_cohort)})."
            )

        model, rf_metrics = fit_rf_and_evaluate_holdout(train_cohort, test_cohort, rf_cfg=rf_cfg)
        num_features = len(model.feature_columns)
        save_feature_importances(model.feature_importances(), fold_dir / "feature_importances.csv")
        train_cohort.to_parquet(fold_dir / "train_cohort.parquet", index=False)
        test_cohort.to_parquet(fold_dir / "test_cohort.parquet", index=False)

        matched_track_metrics = evaluate_matched_track_zero_shot_baseline(test_cohort)

        row.update(rf_metrics)
        row["matched_track_zero_shot_auroc"] = matched_track_metrics.get("causality_auroc")

        if biotype_filter is None:
            fold_strata = evaluate_biotype_strata(
                test_cohort,
                model=model,
                gene_classification=gene_classification,
                dataset_version=dataset_version,
            )
            for stratum, metrics in fold_strata.items():
                biotype_fold_metrics[stratum].append(metrics)

        fold_rows.append(row)

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(output_dir / "cv_fold_results.csv", index=False)

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    test_ns = [
        float(fold.get("n", (fold.get("test_pairs") or {}).get("n_rows", float("nan"))))
        for fold in fold_rows
        if pd.notna(fold.get("n", (fold.get("test_pairs") or {}).get("n_rows")))
    ]
    train_ns = [
        float((fold.get("train_pairs") or {}).get("n_rows"))
        for fold in fold_rows
        if (fold.get("train_pairs") or {}).get("n_rows") is not None
    ]

    report = {
        "evaluation_mode": "causality_rf_cv",
        "cross_validate_folds": len(CAUSALITY_5_FOLD_CHROMOSOMES),
        "num_features": num_features,
        "biotype_filter": biotype_filter,
        "overall": {
            "n": float(sum(test_ns)) if test_ns else float("nan"),
            "train_pairs_mean_n_rows": _mean(train_ns),
            "causality_auroc": _mean(
                [
                    float(v)
                    for v in fold_table.get("causality_auroc", pd.Series(dtype=float))
                    if pd.notna(v)
                ]
            ),
            "causality_avg_precision": _mean(
                [
                    float(v)
                    for v in fold_table.get("causality_avg_precision", pd.Series(dtype=float))
                    if pd.notna(v)
                ]
            ),
            "matched_track_zero_shot_auroc": _mean(
                [
                    float(v)
                    for v in fold_table.get(
                        "matched_track_zero_shot_auroc", pd.Series(dtype=float)
                    )
                    if pd.notna(v)
                ]
            ),
        },
        "fold_results": fold_rows,
    }
    if biotype_filter is None:
        report["by_biotype"] = aggregate_biotype_fold_metrics(biotype_fold_metrics)
    (output_dir / "cv_evaluation_report.json").write_text(json.dumps(report, indent=2))
    return report

from dataclasses import dataclass

from util import filter_snv_examples

can_build_causality_train_cohort = _cohorts.can_build_causality_train_cohort

CAUSALITY_CV_FOLDS = 5

@dataclass
class TrainCausalityRFConfig:
    dataset_version: str
    multimodal_examples_parquet: str
    output_dir: str
    gtf_path: str
    allpairs_background_parquet: str
    seed: int = 42
    high_pip_threshold: float = 0.9
    n_distance_bins: int = 10
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 1
    biotype_filter: str | None = None
    gene_classification: str | None = None

def train_causality_rf_from_config(cfg: TrainCausalityRFConfig) -> Path:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_report_path = output_dir / "cv_evaluation_report.json"

    examples = filter_snv_examples(
        load_examples_parquet(Path(cfg.multimodal_examples_parquet)),
    )
    background = filter_snv_examples(
        load_examples_parquet(Path(cfg.allpairs_background_parquet)),
    )
    if cfg.biotype_filter is not None:
        if cfg.gene_classification is None:
            raise ValueError("gene_classification is required with biotype_filter")
        gene_classification = Path(cfg.gene_classification)
        examples = filter_to_causality_biotype(
            examples,
            cfg.biotype_filter,
            gene_classification=gene_classification,
            version=cfg.dataset_version,
        )
        background = filter_to_causality_biotype(
            background,
            cfg.biotype_filter,
            gene_classification=gene_classification,
            version=cfg.dataset_version,
        )

    if not can_build_causality_train_cohort(
        examples,
        background,
        high_pip_threshold=cfg.high_pip_threshold,
        require_multimodal_features=True,
    ):
        raise ValueError(
            "Cannot build causality train cohort "
            f"(dataset_version={cfg.dataset_version}, "
            f"biotype_filter={cfg.biotype_filter})."
        )

    rf_cfg = CausalityRFConfig(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.seed,
    )

    report = run_causality_rf_cross_validation(
        examples,
        gtf_path=Path(cfg.gtf_path),
        output_dir=output_dir,
        background_examples=background,
        high_pip_threshold=cfg.high_pip_threshold,
        n_distance_bins=cfg.n_distance_bins,
        seed=cfg.seed,
        rf_cfg=rf_cfg,
        biotype_filter=cfg.biotype_filter,
        dataset_version=cfg.dataset_version,
        gene_classification=(
            Path(cfg.gene_classification) if cfg.gene_classification else None
        ),
    )
    save_run_metadata(
        output_dir,
        {
            "dataset_version": cfg.dataset_version,
            "biotype_filter": cfg.biotype_filter,
            "cross_validate_folds": CAUSALITY_CV_FOLDS,
            "cv_report": report,
        },
    )
    return cv_report_path

from gene_biotype_map import CAUSALITY_SPLIT_BIOTYPES

def subsample_cohort(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw n rows, preserving causality_label proportions."""
    df = df.reset_index(drop=True)
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    labels = df["causality_label"].astype(int).to_numpy()
    chosen: list[np.ndarray] = []
    remaining = n
    unique_labs = np.unique(labels)
    for i, lab in enumerate(unique_labs):
        idx = np.flatnonzero(labels == lab)
        if i == len(unique_labs) - 1:
            take = min(remaining, len(idx))
        else:
            take = min(int(round(n * len(idx) / len(df))), len(idx), remaining)
        chosen.append(rng.choice(idx, size=take, replace=False))
        remaining -= take
    sel = np.concatenate(chosen)
    if len(sel) < n:
        leftover = np.setdiff1d(np.arange(len(df)), sel)
        extra = min(n - len(sel), len(leftover))
        if extra:
            sel = np.concatenate([sel, rng.choice(leftover, size=extra, replace=False)])
    elif len(sel) > n:
        sel = rng.choice(sel, size=n, replace=False)
    return df.iloc[sel].reset_index(drop=True)

def biotype_rf_dir(output_root: Path, version: str, biotype: str) -> Path:
    return output_root / f"{version}_causality_rf_{biotype}"

def load_split_cohort(
    output_root: Path,
    version: str,
    split_biotype: str,
    fold: int,
    split: str,
    *,
    gene_classification: Path | None,
) -> pd.DataFrame:
    from gene_biotype_map import filter_to_split_biotype, transfer_source_rf_biotype

    source = transfer_source_rf_biotype(split_biotype)
    path = biotype_rf_dir(output_root, version, source) / f"fold_{fold}" / f"{split}_cohort.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run biotype causality RF train first "
            f"({version}_causality_rf_{source})."
        )
    df = pd.read_parquet(path)
    if source != split_biotype:
        df = filter_to_split_biotype(
            df, split_biotype, gene_classification=gene_classification, version=version
        )
    return df

DEFAULT_EVAL_BIOTYPE = "protein_coding"
DEFAULT_EVAL_BIOTYPES = ("all_genes", "protein_coding", "existing_lncRNA", "novel_lncRNA")
DEFAULT_TRAIN_BIOTYPES = ("protein_coding", "novel_lncRNA", "existing_lncRNA")
DEFAULT_GRID = (50, 100, 500, 1000, 5000, 10000, 50000)
DEFAULT_N_REPLICATES = 20
ALL_GENES_BIOTYPES = ("lncRNA", "protein_coding")

def subsample_output_dir(
    output_root: Path,
    eval_biotype: str,
) -> Path:
    """Output root for label-subsample fold caches (exact 50/50 eval holdouts)."""
    if eval_biotype == DEFAULT_EVAL_BIOTYPE:
        return output_root / "v11_pc_label_subsample"
    return output_root / "v11_label_subsample" / f"eval_{eval_biotype}"

def eval_balance_seed(fold: int, eval_biotype: str) -> int:
    """Seed for exact 50/50 per-distance-bin balancing of a holdout/train cohort."""
    if eval_biotype in DEFAULT_EVAL_BIOTYPES:
        idx = list(DEFAULT_EVAL_BIOTYPES).index(eval_biotype)
    elif eval_biotype.startswith("train_"):
        train_bt = eval_biotype[len("train_") :]
        idx = 100 + (
            list(DEFAULT_TRAIN_BIOTYPES).index(train_bt)
            if train_bt in DEFAULT_TRAIN_BIOTYPES
            else 9
        )
    else:
        idx = 9
    return 137 + 10_000 * fold + 1_000_000 * idx

def load_eval_test_cohort(
    output_root: Path,
    eval_biotype: str,
    fold: int,
    *,
    gene_classification: Path | None,
) -> pd.DataFrame:
    """Load an eval holdout and enforce exact 50/50 per distance bin.

    Biotype-filtered subsets of an already-balanced pooled cohort can drift off
    50/50, so we always rebalance after load. Never downsamples by eval N —
    matched-N applies only to training-set size.
    """
    if eval_biotype == DEFAULT_EVAL_BIOTYPE:
        path = output_root / f"{VERSION}_causality_rf_{eval_biotype}" / f"fold_{fold}" / "test_cohort.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run protein_coding causality RF train first.")
        test = pd.read_parquet(path)
    elif eval_biotype == "all_genes":
        path = (
            output_root
            / f"{VERSION}_causality_rf_lncRNA_protein_coding"
            / f"fold_{fold}"
            / "test_cohort.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run lncRNA_protein_coding causality RF train first.")
        test = pd.read_parquet(path)
        test = test[test["gene_biotype"].isin(ALL_GENES_BIOTYPES)].copy().reset_index(drop=True)
    elif eval_biotype in CAUSALITY_SPLIT_BIOTYPES:
        test = load_split_cohort(
            output_root,
            VERSION,
            eval_biotype,
            fold,
            "test",
            gene_classification=gene_classification,
        )
    else:
        raise ValueError(
            f"Unsupported eval biotype {eval_biotype!r}; expected protein_coding, all_genes, "
            f"or one of {CAUSALITY_SPLIT_BIOTYPES}."
        )
    return rebalance_cohort_exact(test, seed=eval_balance_seed(fold, eval_biotype))

def subsample_seed(
    fold: int,
    n_train: int,
    train_biotype: str,
    train_biotypes: list[str],
    replicate: int = 0,
) -> int:
    return (
        42
        + 10_000 * fold
        + n_train
        + 1_000_000 * train_biotypes.index(train_biotype)
        + 100 * replicate
    )

def normalize_label_subsample_df(df: pd.DataFrame, eval_biotype: str) -> pd.DataFrame:
    work = df.copy()
    if "train_biotype" not in work.columns:
        work = work.assign(train_biotype=DEFAULT_EVAL_BIOTYPE)
    if "replicate" not in work.columns:
        work = work.assign(replicate=0)
    work["replicate"] = work["replicate"].fillna(0).astype(int)
    if "eval_biotype" in work.columns:
        work["eval_biotype"] = work["eval_biotype"].fillna(eval_biotype)
        work = work[work["eval_biotype"] == eval_biotype].copy()
    return work

def _fold_grid(grid: list[int], train_n: int) -> list[int]:
    """Grid points ≤ train_n, always including the full training-set size."""
    fold_grid = sorted({n for n in grid if n <= train_n})
    if train_n not in fold_grid:
        fold_grid.append(train_n)
    return fold_grid

def _replicate_range(n_train: int, train_n: int, n_replicates: int) -> range:
    if n_train >= train_n:
        return range(1)
    return range(n_replicates)

def parse_n_train_grid(grid_arg: str) -> list[int]:
    grid = sorted({int(part.strip()) for part in grid_arg.split(",") if part.strip()})
    if not grid:
        raise ValueError("Empty --grid")
    return grid

def parse_train_biotypes(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_TRAIN_BIOTYPES)
    biotypes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [b for b in biotypes if b not in CAUSALITY_SPLIT_BIOTYPES]
    if unknown:
        raise ValueError(
            f"Unsupported train biotypes {unknown}; expected subset of {CAUSALITY_SPLIT_BIOTYPES}."
        )
    return biotypes

def expected_label_subsample_ns(
    df: pd.DataFrame,
    train_bt: str,
    grid: list[int],
) -> list[int]:
    sub = df[df["train_biotype"] == train_bt].copy()
    if sub.empty:
        return list(grid)
    min_pool = int(sub.groupby("fold")["n_train_available"].first().min())
    return [n for n in grid if n <= min_pool]

def label_subsample_missing(
    df: pd.DataFrame | None,
    eval_biotype: str,
    *,
    grid: list[int] | None = None,
    train_biotypes: list[str] | None = None,
    n_folds: int = CAUSALITY_CV_FOLDS,
    n_replicates: int = DEFAULT_N_REPLICATES,
) -> dict[str, list[int | str]]:
    grid = list(grid or DEFAULT_GRID)
    train_biotypes = list(train_biotypes or DEFAULT_TRAIN_BIOTYPES)
    if df is None or len(df) == 0:
        return {bt: list(grid) + ["full"] for bt in train_biotypes}

    work = normalize_label_subsample_df(df, eval_biotype)
    missing_by_biotype: dict[str, list[int | str]] = {}
    for train_bt in train_biotypes:
        sub = work[work["train_biotype"] == train_bt].copy()
        if sub.empty:
            missing_by_biotype[train_bt] = list(grid) + ["full"]
            continue

        expected_ns = expected_label_subsample_ns(sub, train_bt, grid)
        missing: list[int | str] = []
        for n_train in expected_ns:
            rows = sub[(sub["n_train"] == n_train) & ~sub["is_full"]]
            have_reps = set(rows["replicate"].astype(int))
            expected_reps = set(range(n_replicates))
            if have_reps != expected_reps:
                missing.append(n_train)
        full_rows = sub[sub["is_full"]]
        if len(full_rows) < n_folds:
            missing.append("full")
        if missing:
            missing_by_biotype[train_bt] = missing
    return missing_by_biotype

def _auroc_summary_fold_level(
    values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    from scipy import stats as sp_stats

    vals = np.asarray(values, dtype=float)
    mean = float(vals.mean())
    if len(vals) > 1:
        half = float(sp_stats.sem(vals) * sp_stats.t.ppf(1 - alpha / 2, len(vals) - 1))
    else:
        half = 0.0
    return {
        "mean": mean,
        "ci_lower": mean - half,
        "ci_upper": mean + half,
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "n_folds": int(len(vals)),
    }

def _fold_mean_auroc_values(sub: pd.DataFrame) -> np.ndarray:
    return (
        sub.groupby("fold")["tissue_weighted_auroc"]
        .mean()
        .to_numpy(dtype=float)
    )

def summarize_label_subsample_curve_fold_level(
    sub: pd.DataFrame,
    expected_ns: list[int],
    n_folds: int,
    *,
    n_replicates: int = DEFAULT_N_REPLICATES,
    alpha: float = 0.05,
) -> pd.DataFrame:
    records: list[dict] = []
    grid_rows = sub[sub["n_train"].isin(expected_ns) & ~sub["is_full"]].copy()
    for n_train, grp in grid_rows.groupby("n_train"):
        if len(grp) < n_folds * n_replicates:
            continue
        fold_vals = _fold_mean_auroc_values(grp)
        if len(fold_vals) != n_folds:
            continue
        stats = _auroc_summary_fold_level(fold_vals, alpha=alpha)
        records.append({"n_train": int(n_train), "is_full": False, **stats})

    full_rows = sub[sub["is_full"]].copy()
    if len(full_rows) == n_folds:
        fold_vals = _fold_mean_auroc_values(full_rows)
        stats = _auroc_summary_fold_level(fold_vals, alpha=alpha)
        records.append(
            {
                "n_train": int(round(full_rows["n_train"].mean())),
                "is_full": True,
                **stats,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=["n_train", "mean", "ci_lower", "ci_upper", "std", "n_folds", "is_full"]
        )
    return pd.DataFrame(records).sort_values("n_train").reset_index(drop=True)

def combined_curves_tsv_path(output_root: Path) -> Path:
    """Single figure-ready fold-auROC table covering all eval biotypes."""
    return output_root / "qtl_causality_auroc.tsv"

ZERO_SHOT_TRAIN_BIOTYPE = "zero_shot"

def build_zero_shot_learning_curve_rows(
    output_root: Path,
    *,
    eval_biotypes: list[str] | None = None,
    n_folds: int = CAUSALITY_CV_FOLDS,
    gene_classification: Path | None = None,
) -> pd.DataFrame:
    """Tissue-weighted zero-shot auROC rows in ``learning_curves.tsv`` schema.

    Always evaluates on the full rebalanced chromosome-fold holdout (no eval-N
    downsampling). Matched-N applies only to RF training-set size, not to eval.
    Rows use ``train_biotype='zero_shot'`` and ``n_train`` / CI columns as NA.
    """
    eval_biotypes = list(eval_biotypes or DEFAULT_EVAL_BIOTYPES)

    rows: list[dict] = []
    for eval_bt in eval_biotypes:
        fold_aurocs: list[float] = []
        for fold in range(n_folds):
            test = load_eval_test_cohort(
                output_root,
                eval_bt,
                fold,
                gene_classification=gene_classification,
            )
            metrics = evaluate_matched_track_zero_shot_baseline(test)
            auroc = metrics.get("causality_auroc")
            if auroc is None or not np.isfinite(auroc):
                raise RuntimeError(
                    f"Non-finite zero-shot auROC for eval={eval_bt} fold={fold}"
                )
            fold_aurocs.append(float(auroc))
        rows.append(
            {
                "dataset_version": VERSION,
                "eval_biotype": eval_bt,
                "train_biotype": ZERO_SHOT_TRAIN_BIOTYPE,
                "n_train": np.nan,
                "fold": np.nan,
                "mean": float(np.mean(fold_aurocs)),
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "std": float(np.std(fold_aurocs)),
                "n_folds": int(len(fold_aurocs)),
            }
        )
    return pd.DataFrame(rows)

def build_label_subsample_curves_table(
    df: pd.DataFrame,
    eval_biotype: str,
    *,
    grid: list[int] | None = None,
    train_biotypes: list[str] | None = None,
    n_replicates: int = DEFAULT_N_REPLICATES,
) -> pd.DataFrame:
    """Summarize fold_auroc rows into a flat mean±CI table for plotting."""
    grid = list(grid or DEFAULT_GRID)
    train_biotypes = list(train_biotypes or DEFAULT_TRAIN_BIOTYPES)
    work = normalize_label_subsample_df(df, eval_biotype)
    still_missing = label_subsample_missing(
        work,
        eval_biotype,
        grid=grid,
        train_biotypes=train_biotypes,
        n_replicates=n_replicates,
    )
    if still_missing:
        raise RuntimeError(
            f"Label subsample incomplete for {eval_biotype}: {still_missing}. "
            "Re-run alphagenome_qtl/04_train_subsample_rf.py to finish missing grid points."
        )
    n_folds = int(work["fold"].nunique())
    parts: list[pd.DataFrame] = []
    for train_bt in train_biotypes:
        sub = work[work["train_biotype"] == train_bt].copy()
        expected_ns = expected_label_subsample_ns(sub, train_bt, grid)
        curve = summarize_label_subsample_curve_fold_level(
            sub,
            expected_ns,
            n_folds,
            n_replicates=n_replicates,
        )
        if curve.empty:
            continue
        curve = curve.copy()
        curve.insert(0, "train_biotype", train_bt)
        curve.insert(0, "eval_biotype", eval_biotype)
        parts.append(curve)
    empty_cols = [
        "dataset_version",
        "eval_biotype",
        "train_biotype",
        "n_train",
        "fold",
        "mean",
        "ci_lower",
        "ci_upper",
        "std",
        "n_folds",
    ]
    if not parts:
        return pd.DataFrame(columns=empty_cols)
    out = pd.concat(parts, ignore_index=True)
    # is_full is only used while aggregating raw fold runs; figure TSV is keyed by n_train.
    out = out.drop(columns=["is_full"], errors="ignore")
    out["dataset_version"] = VERSION
    out["fold"] = np.nan
    out = out[
        [
            "dataset_version",
            "eval_biotype",
            "train_biotype",
            "n_train",
            "fold",
            "mean",
            "ci_lower",
            "ci_upper",
            "std",
            "n_folds",
        ]
    ]
    return out.sort_values(
        ["eval_biotype", "train_biotype", "n_train"]
    ).reset_index(drop=True)

def write_label_subsample_curves_tsv(
    curves: pd.DataFrame,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(path, sep="\t", index=False)
    return path

def write_label_subsample_curves_for_eval(
    output_root: Path,
    eval_biotype: str,
    *,
    grid: list[int] | None = None,
    train_biotypes: list[str] | None = None,
    n_replicates: int = DEFAULT_N_REPLICATES,
    summary: pd.DataFrame | None = None,
) -> Path:
    """Write per-eval figure-ready curves TSV next to fold_auroc.parquet."""
    output_dir = subsample_output_dir(output_root, eval_biotype)
    if summary is None:
        cache = output_dir / "fold_auroc.parquet"
        if not cache.exists():
            raise FileNotFoundError(
                f"Missing {cache}; re-run alphagenome_qtl/04_train_subsample_rf.py first."
            )
        summary = pd.read_parquet(cache)
    curves = build_label_subsample_curves_table(
        summary,
        eval_biotype,
        grid=grid,
        train_biotypes=train_biotypes,
        n_replicates=n_replicates,
    )
    return write_label_subsample_curves_tsv(
        curves, output_dir / "learning_curves.tsv"
    )

def export_combined_label_subsample_curves(
    output_root: Path,
    *,
    eval_biotypes: list[str] | None = None,
    grid: list[int] | None = None,
    train_biotypes: list[str] | None = None,
    n_replicates: int = DEFAULT_N_REPLICATES,
    n_folds: int = CAUSALITY_CV_FOLDS,
    gene_classification: Path | None = None,
) -> Path:
    """Build one TSV over all requested eval biotypes under output_root.

    Appends ``train_biotype='zero_shot'`` baseline rows (Fig 3f dotted line)
    into the same curves TSV.
    """
    eval_biotypes = list(eval_biotypes or DEFAULT_EVAL_BIOTYPES)
    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for eval_bt in eval_biotypes:
        cache = subsample_output_dir(output_root, eval_bt) / "fold_auroc.parquet"
        if not cache.exists():
            missing.append(eval_bt)
            continue
        curves = build_label_subsample_curves_table(
            pd.read_parquet(cache),
            eval_bt,
            grid=grid,
            train_biotypes=train_biotypes,
            n_replicates=n_replicates,
        )
        # Keep per-eval copy in sync when gathering.
        write_label_subsample_curves_tsv(
            curves,
            cache.parent / "learning_curves.tsv",
        )
        parts.append(curves)
    if missing:
        raise FileNotFoundError(
            "Missing fold_auroc.parquet for eval biotype(s) "
            f"{missing} under {output_root}."
        )
    combined = pd.concat(parts, ignore_index=True).drop(
        columns=["is_full"], errors="ignore"
    )
    zero_shot = build_zero_shot_learning_curve_rows(
        output_root,
        eval_biotypes=eval_biotypes,
        n_folds=n_folds,
        gene_classification=gene_classification,
    )
    combined = pd.concat([combined, zero_shot], ignore_index=True)
    combined = combined.sort_values(
        ["eval_biotype", "train_biotype", "n_train"],
        na_position="first",
    ).reset_index(drop=True)
    out = combined_curves_tsv_path(output_root)
    write_label_subsample_curves_tsv(combined, out)
    print(
        json.dumps(
            {
                "curves_tsv": str(out),
                "n_rows": int(len(combined)),
                "eval_biotypes": sorted(combined["eval_biotype"].unique().tolist()),
                "includes_zero_shot": True,
            },
            indent=2,
        ),
        flush=True,
    )
    return out

def run_label_subsample(
    *,
    output_root: Path,
    output_dir: Path | None = None,
    eval_biotype: str = DEFAULT_EVAL_BIOTYPE,
    grid_arg: str | None = None,
    train_biotypes: list[str] | None = None,
    gene_classification: Path | None = None,
    n_folds: int = CAUSALITY_CV_FOLDS,
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = 42,
) -> Path:
    train_biotypes = list(train_biotypes or DEFAULT_TRAIN_BIOTYPES)
    output_dir = output_dir or subsample_output_dir(output_root, eval_biotype)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "fold_auroc.parquet"
    report_path = output_dir / "subsample_report.json"

    train_sizes: dict[str, list[int]] = {bt: [] for bt in train_biotypes}
    for fold in range(n_folds):
        _ = load_eval_test_cohort(
            output_root,
            eval_biotype,
            fold,
            gene_classification=gene_classification,
        )
        for train_bt in train_biotypes:
            train = load_split_cohort(
                output_root,
                VERSION,
                train_bt,
                fold,
                "train",
                gene_classification=gene_classification,
            )
            train = rebalance_cohort_exact(
                train, seed=eval_balance_seed(fold, f"train_{train_bt}")
            )
            train_sizes[train_bt].append(len(train))

    if not grid_arg:
        raise ValueError("grid_arg is required")
    grid = parse_n_train_grid(grid_arg)
    print(
        f"[label_subsample] folds={n_folds} replicates={n_replicates} "
        f"eval_biotype={eval_biotype} train_biotypes={train_biotypes} grid={grid} "
        f"train_sizes={train_sizes}",
        flush=True,
    )

    rf_cfg = CausalityRFConfig(random_state=seed)
    rows: list[dict] = []
    for train_bt in train_biotypes:
        for fold in range(n_folds):
            train = load_split_cohort(
                output_root,
                VERSION,
                train_bt,
                fold,
                "train",
                gene_classification=gene_classification,
            )
            train = rebalance_cohort_exact(
                train, seed=eval_balance_seed(fold, f"train_{train_bt}")
            )
            test = load_eval_test_cohort(
                output_root,
                eval_biotype,
                fold,
                gene_classification=gene_classification,
            )
            fold_grid = _fold_grid(grid, len(train))
            for n_train in fold_grid:
                for replicate in _replicate_range(n_train, len(train), n_replicates):
                    train_sub = subsample_cohort(
                        train,
                        n_train,
                        subsample_seed(fold, n_train, train_bt, train_biotypes, replicate),
                    )
                    if train_sub["causality_label"].nunique() < 2:
                        raise ValueError(
                            f"Subsample has one class: train_biotype={train_bt} "
                            f"fold={fold} n={n_train} replicate={replicate}"
                        )
                    print(
                        f"[label_subsample] train={train_bt} fold={fold} rep={replicate} "
                        f"n={len(train_sub)}/{len(train)} -> {eval_biotype} test",
                        flush=True,
                    )
                    _model, metrics = fit_rf_and_evaluate_holdout(
                        train_sub, test, rf_cfg=rf_cfg
                    )
                    rows.append(
                        {
                            "version": VERSION,
                            "train_biotype": train_bt,
                            "eval_biotype": eval_biotype,
                            "fold": fold,
                            "replicate": int(replicate),
                            "n_train": int(len(train_sub)),
                            "n_train_available": int(len(train)),
                            "is_full": bool(len(train_sub) == len(train)),
                            "n_test": int(len(test)),
                            "tissue_weighted_auroc": float(metrics["causality_auroc"]),
                            "tissue_weighted_ap": float(metrics["causality_avg_precision"]),
                        }
                    )

    summary = pd.DataFrame(rows)
    summary.to_parquet(summary_path, index=False)

    # Per-eval curves; 04_train_subsample_rf.py writes the combined TSV.
    per_eval_curves_path = write_label_subsample_curves_for_eval(
        output_root,
        eval_biotype,
        grid=grid,
        train_biotypes=train_biotypes,
        n_replicates=n_replicates,
        summary=summary,
    )

    report = {
        "evaluation_mode": "v11_label_subsample",
        "version": VERSION,
        "eval_biotype": eval_biotype,
        "train_biotypes": train_biotypes,
        "grid_arg": grid_arg,
        "n_folds": n_folds,
        "n_replicates": n_replicates,
        "train_sizes": train_sizes,
        "shared_grid": grid,
        "n_rows": int(len(summary)),
        "summary_path": str(summary_path),
        "curves_tsv": str(per_eval_curves_path),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "artifact_path": str(summary_path),
                "curves_tsv": str(per_eval_curves_path),
                "n_rows": int(len(summary)),
            },
            indent=2,
        )
    )
    return summary_path

