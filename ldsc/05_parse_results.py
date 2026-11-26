#!/usr/bin/env python3
"""Step 5 — build an enrichment + τ* summary TSV for figures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# MAF threshold defining the reference SNP set used to standardize τ* (LDSC M_5_50 convention).
MAF_MIN_DEFAULT = 0.05
CHROMOSOMES = [str(c) for c in range(1, 23)]


def parse_log_metrics(log_path: Path) -> dict[str, float]:
    metrics = {
        "h2": np.nan,
        "h2_se": np.nan,
        "h2_z": np.nan,
        "intercept": np.nan,
        "intercept_se": np.nan,
        "mean_chi2": np.nan,
        "lambda_gc": np.nan,
    }
    if not log_path.exists():
        return metrics
    text = log_path.read_text(errors="ignore")

    m = re.search(r"Total Observed scale h2:\s*([0-9eE.+-]+)\s*\(([0-9eE.+-]+)\)", text)
    if m:
        h2 = float(m.group(1))
        se = float(m.group(2))
        metrics["h2"] = h2
        metrics["h2_se"] = se
        if se > 0:
            metrics["h2_z"] = h2 / se
    m = re.search(r"Intercept:\s*([0-9eE.+-]+)\s*\(([0-9eE.+-]+)\)", text)
    if m:
        metrics["intercept"] = float(m.group(1))
        metrics["intercept_se"] = float(m.group(2))
    m = re.search(r"Mean Chi\^2:\s*([0-9eE.+-]+)", text)
    if m:
        metrics["mean_chi2"] = float(m.group(1))
    m = re.search(r"Lambda GC:\s*([0-9eE.+-]+)", text)
    if m:
        metrics["lambda_gc"] = float(m.group(1))
    return metrics


def pick_primary_row(df: pd.DataFrame, primary_annotation: str) -> pd.Series | None:
    if "Category" not in df.columns:
        return None
    cat = df["Category"].astype(str)
    exact = df[cat == primary_annotation]
    if len(exact) == 1:
        return exact.iloc[0]
    contains = df[cat.str.contains(primary_annotation, regex=False)]
    if len(contains) >= 1:
        return contains.iloc[0]
    return None


_L2_CAT = re.compile(r"^L2_(\d+)$")


def non_baseline_annot_names(model_row: pd.Series) -> list[str]:
    """Non-baseline LD-score names in the same order passed to ``ldsc.py --ref-ld-chr``.

    Order is: optional ``baseline_track``, then comma-separated ``annotations``.
    Paper models always include a baselineLD track, so LDSC labels custom strata L2_1..
    """
    names: list[str] = []
    track = str(model_row.get("baseline_track", "")).strip()
    if track:
        names.append(track)
    names.extend(
        a.strip() for a in str(model_row.get("annotations", "")).split(",") if a.strip()
    )
    return names


def _custom_l2_index_base(df: pd.DataFrame) -> int:
    """LDSC uses L2_1.. when baselineLD is included (paper models); else L2_0.."""
    l2_nums = []
    if "Category" in df.columns:
        for cat in df["Category"].astype(str):
            m = _L2_CAT.match(cat)
            if m:
                l2_nums.append(int(m.group(1)))
    # Paper manifests always include baselineLD, which shifts custom strata to L2_1..
    if l2_nums and min(l2_nums) == 0:
        return 0
    return 1


def primary_annotation_to_category(model_row: pd.Series, df: pd.DataFrame) -> str:
    """
    LDSC sometimes writes non-baseline annotation categories as generic 'L2_1', 'L2_2', ...
    rather than preserving the original annotation names. If the manifest's
    ``primary_annotation`` is not present as a ``Category`` in the .results file, map it
    to L2_k from the order of non-baseline names in the models manifest.
    """
    primary = str(model_row["primary_annotation"])
    if "Category" in df.columns and primary in set(df["Category"].astype(str).unique()):
        return primary

    names = non_baseline_annot_names(model_row)
    if primary not in names:
        return primary
    l2_base = _custom_l2_index_base(df)
    return f"L2_{l2_base + names.index(primary)}"


# ── Standardized effect size τ* (methods eq 3): τ*_c = τ_c · sd(c) / (h²_g / M) ──
# sd(c) is the standard deviation of annotation c and M the number of reference
# variants, both computed over reference SNPs with MAF ≥ maf_min — the same MAF≥5%
# set S-LDSC uses (via --frqfile-chr / M_5_50). We read the focal annotation's own
# .annot.gz tracks and the plink .frq files directly. There are no fallbacks:
# missing or mismatched inputs raise so the summary always aligns with the methods.

def _load_frq_mask(frqfile_prefix: str, chrom: str, maf_min: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (SNP array, MAF≥maf_min mask) for a chromosome's plink .frq file."""
    path = Path(f"{frqfile_prefix}{chrom}.frq")
    if not path.is_file():
        raise FileNotFoundError(f"τ* requires a plink .frq per chromosome; missing: {path}")
    frq = pd.read_csv(path, sep=r"\s+", usecols=["SNP", "MAF"])
    snp = frq["SNP"].to_numpy()
    mask = frq["MAF"].to_numpy(dtype=float) >= maf_min
    return snp, mask


def focal_annot_prefix(model_row: pd.Series, ldscores_dir: Path) -> str:
    """LD-score prefix of the focal annotation: ``<ldscores_dir>/<name>/<name>.``."""
    primary = str(model_row["primary_annotation"]).strip()
    if not primary:
        raise ValueError("Manifest row missing primary_annotation")
    return str(Path(ldscores_dir) / primary / f"{primary}.")


def compute_annot_sd_and_M(annot_prefix: str, frqfile_prefix: str, chroms: list[str], maf_min: float) -> tuple[float, int]:
    """Population sd(c) and M over reference SNPs with MAF ≥ maf_min for one annotation."""
    n = 0
    s = 0.0
    ss = 0.0
    for chrom in chroms:
        apath = Path(f"{annot_prefix}{chrom}.annot.gz")
        if not apath.is_file():
            raise FileNotFoundError(f"τ* requires the focal annotation .annot.gz; missing: {apath}")
        col = pd.read_csv(apath, sep="\t", nrows=0).columns[-1]
        adf = pd.read_csv(apath, sep="\t", usecols=["SNP", col])
        fsnp, fmask = _load_frq_mask(frqfile_prefix, chrom, maf_min)
        asnp = adf["SNP"].to_numpy()
        if asnp.shape[0] != fsnp.shape[0] or not np.array_equal(asnp, fsnp):
            raise ValueError(
                f"SNP set/order mismatch between {apath.name} and its .frq (chr {chrom}); "
                f"cannot compute a methods-aligned τ*."
            )
        vals = adf[col].to_numpy(dtype=float)[fmask]
        n += int(vals.size)
        s += float(vals.sum())
        ss += float(np.square(vals).sum())
    if n == 0:
        raise ValueError(f"No MAF≥{maf_min} reference SNPs found for annotation prefix {annot_prefix}.")
    mean = s / n
    var = ss / n - mean * mean
    if not np.isfinite(var) or var <= 0:
        raise ValueError(f"Non-positive/NaN annotation variance for {annot_prefix} (var={var}).")
    sd = float(np.sqrt(var))
    return sd, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="LDSC run root containing results/{TRAIT}/{TRAIT}.{model}.results from step 04.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Summary TSV to write (e.g. ldsc/ldsc_summary.tsv).",
    )
    parser.add_argument("--models_manifest", required=True)
    parser.add_argument(
        "--ldscores_dir",
        required=True,
        type=Path,
        help="Root of paper QTL annotation folders (<ldscores_dir>/<annot>/<annot>.{chr}.annot.gz).",
    )
    parser.add_argument(
        "--frqfile_prefix",
        required=True,
        help="Plink .frq prefix (per-chromosome '<prefix><chr>.frq') for the reference panel; "
        "used to standardize τ* over MAF≥maf_min SNPs. Required — no fallback.",
    )
    args = parser.parse_args()

    chroms = list(CHROMOSOMES)
    maf_min = MAF_MIN_DEFAULT
    output_dir = Path(args.output_dir)
    ldscores_dir = Path(args.ldscores_dir)

    models = pd.read_csv(args.models_manifest, sep="\t", dtype=str).fillna("")
    required = {"model_id", "annotations", "primary_annotation"}
    missing = required - set(models.columns)
    if missing:
        raise ValueError(
            f"Models manifest missing columns {sorted(missing)}; "
            "expected columns: model_id, baseline, annotations, "
            "primary_annotation, baseline_track"
        )
    if "baseline_track" not in models.columns:
        models["baseline_track"] = ""

    results_root = output_dir / "results"
    if not results_root.is_dir():
        raise FileNotFoundError(f"No LDSC results directory: {results_root}")

    trait_ids = sorted(p.name for p in results_root.iterdir() if p.is_dir())
    if not trait_ids:
        raise FileNotFoundError(f"No trait result folders under {results_root}")

    primary_rows = []
    qc_rows = []

    for _, model_row in models.iterrows():
        model_id = str(model_row["model_id"])

        for trait_id in trait_ids:
            result_path = results_root / trait_id / f"{trait_id}.{model_id}.results"
            if not result_path.is_file():
                continue

            df = pd.read_csv(result_path, sep=r"\s+")
            primary_cat = primary_annotation_to_category(model_row, df)
            primary = pick_primary_row(df, primary_cat)
            if primary is not None:
                row = primary.to_dict()
                row["trait_id"] = trait_id
                row["model_id"] = model_id
                row["primary_annotation"] = str(model_row["primary_annotation"])
                row["primary_category"] = primary_cat
                sd_c, m_ref = compute_annot_sd_and_M(
                    focal_annot_prefix(model_row, ldscores_dir),
                    args.frqfile_prefix,
                    chroms,
                    maf_min,
                )
                row["annot_sd"] = sd_c
                row["M_ref"] = m_ref
                primary_rows.append(row)

            # h2 (and related QC) come from the .log; figures filter on h2 > 0.
            qc = parse_log_metrics(result_path.with_suffix(".log"))
            qc["trait_id"] = trait_id
            qc["model_id"] = model_id
            qc_rows.append(qc)

    primary_df = pd.DataFrame(primary_rows)
    qc_df = pd.DataFrame(qc_rows)

    if primary_df.empty:
        primary_df = pd.DataFrame(
            columns=[
                "trait_id",
                "model_id",
                "primary_annotation",
                "primary_category",
            ]
        )
    primary_df = primary_df.merge(qc_df, on=["trait_id", "model_id"], how="left")

    # Standardized effect size τ* = τ_c · sd(c) / (h²_g / M) (methods eq 3).
    # The z-score is scale-invariant and equals the LDSC Coefficient z-score;
    # only the magnitude/SE carry the standardization. Figures meta-analyze τ*
    # themselves (they do not use a per-trait τ* p-value).
    if not primary_df.empty:
        for col in ("Coefficient", "Coefficient_std_error", "annot_sd", "M_ref", "h2"):
            if col not in primary_df.columns:
                raise ValueError(f"τ* computation requires column {col!r} in the primary summary.")
        coef = pd.to_numeric(primary_df["Coefficient"], errors="coerce")
        coef_se = pd.to_numeric(primary_df["Coefficient_std_error"], errors="coerce")
        sd_c = pd.to_numeric(primary_df["annot_sd"], errors="coerce")
        m_ref = pd.to_numeric(primary_df["M_ref"], errors="coerce")
        h2 = pd.to_numeric(primary_df["h2"], errors="coerce")
        scale = sd_c * m_ref / h2  # sd(c) / (h²_g / M)
        primary_df["tau_star"] = coef * scale
        primary_df["tau_star_se"] = coef_se * scale
        with np.errstate(divide="ignore", invalid="ignore"):
            primary_df["tau_star_z"] = primary_df["tau_star"] / primary_df["tau_star_se"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    primary_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path} ({len(primary_df)} rows)")


if __name__ == "__main__":
    main()
