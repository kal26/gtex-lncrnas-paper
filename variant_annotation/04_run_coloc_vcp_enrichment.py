#!/usr/bin/env python3
"""Step 4 — Firth VEP enrichment of high-VCP coloc lead variants (Supp Fig 15h).

For each CoS composition stratum at GWAS p < 1e-5:
  - cases = lead variants with lead VCP > cutoff in that stratum
  - universe = full cis-window pool for that stratum (no control subsample)
  - mixed cis pool = intersection of lncRNA and protein-coding cis windows
  - lncRNA only = ≥1 lncRNA, 0 protein-coding, 0 pseudogene (exact)
  - annotations = same full VEP + merged set as ``03_`` / ``qtl_odds.py``

Inputs (ColocBoost layout on Zenodo)::

  --coloc_dir  …/coloc/main
      {Tissue}.ColocBoost.parquet
      cross_tissue_loci.parquet

Locus composition and GWAS p < 1e-5 filter are derived by exploding
``member_cos_ids`` and joining tissue ColocBoost rows (lead = argmax VCP).

Writes Firth ORs for all annotations × strata × VCP cutoffs; notebooks filter.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from qtl_odds import add_merged_annotation_columns, fit_firth_annot_odds

try:
    from tqdm.auto import tqdm
except ImportError:  # odds pixi env is lean (firthmodels, no tqdm)

    def tqdm(iterable, **_kwargs):
        return iterable


COS_CAT_ORDER = ["lncRNA only", "protein-coding only", "mixed"]
_ENSG_STEM = re.compile(r"(ENSG\d+)")
_GWAS_SUB_NEGLOG10 = float(-np.log10(1e-5))
ANNOT_VERSION = "v11"
VCP_CUTOFFS = [0.9]


def _as_list(val) -> list:
    if val is None:
        return []
    try:
        if isinstance(val, float) and np.isnan(val):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(val, np.ndarray):
        return list(val.tolist())
    if isinstance(val, (list, tuple, set)):
        return list(val)
    s = str(val).strip()
    return [s] if s and s not in ("nan", "None", "<NA>") else []


def _as_stem_set(vals) -> set[str]:
    out: set[str] = set()
    for v in _as_list(vals):
        m = _ENSG_STEM.search(str(v))
        if m:
            out.add(m.group(1))
    return out


def _lead_from_lists(variants, vcp) -> tuple[str, float]:
    vs = [str(v) for v in _as_list(variants)]
    if not vs:
        return "", float("nan")
    ps = pd.to_numeric(pd.Series(_as_list(vcp)), errors="coerce")
    if len(ps) == len(vs) and ps.notna().any():
        i = int(ps.values.argmax())
        return vs[i], float(ps.iloc[i])
    return vs[0], float(ps.iloc[0]) if len(ps) else float("nan")


def _coloc_vid_to_snp(vid) -> str | None:
    if vid is None or vid in ("", "nan", "None"):
        return None
    parts = str(vid).split(":")
    if len(parts) < 4:
        return None
    chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    return f"{chrom}_{pos}_{ref}_{alt}_b38"


def _cos_composition(
    gene_stems: set[str],
    lncrna_ids: set[str],
    pc_ids: set[str],
    pseudogene_ids: set[str],
):
    """Exact composition: lncRNA-only excludes PC and pseudogene."""
    n_lnc = sum(1 for g in gene_stems if g in lncrna_ids)
    n_pc = sum(1 for g in gene_stems if g in pc_ids)
    n_pseudo = sum(1 for g in gene_stems if g in pseudogene_ids)
    if n_lnc >= 1 and n_pc >= 1:
        return "mixed"
    if n_lnc >= 1 and n_pc == 0 and n_pseudo == 0:
        return "lncRNA only"
    if n_pc >= 1 and n_lnc == 0 and n_pseudo == 0:
        return "protein-coding only"
    return None


def _parse_neglog10_list(val) -> list[float]:
    if val is None:
        return []
    try:
        if isinstance(val, float) and np.isnan(val):
            return []
    except (TypeError, ValueError):
        pass
    out: list[float] = []
    for tok in str(val).split(","):
        tok = tok.strip()
        if not tok or tok.lower() == "nan":
            out.append(float("nan"))
            continue
        try:
            out.append(float(tok))
        except ValueError:
            out.append(float("nan"))
    return out


def _gwas_traits_above_cut(phenotype_id, pval_nominal_phenotypes, cut: float) -> list[str]:
    parts = [p.strip() for p in str(phenotype_id or "").split(";") if p.strip()]
    vals = _parse_neglog10_list(pval_nominal_phenotypes)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i >= len(vals):
            break
        pl = part.lower()
        if "expression" in pl or "splicing" in pl:
            continue
        v = vals[i]
        if np.isfinite(v) and float(v) > cut:
            out.append(part)
    return out


def _nonempty_trait_list(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (list, tuple, np.ndarray)):
        return any(
            t is not None and str(t) not in ("", "nan", "None", "<NA>") for t in val
        )
    s = str(val).strip()
    return bool(s and s not in ("nan", "None", "<NA>"))


def _row_has_gwas_1e5(phenotype_id, pval_nominal_phenotypes, gwas_gws_traits=None) -> bool:
    if _nonempty_trait_list(gwas_gws_traits):
        return True
    return bool(
        _gwas_traits_above_cut(phenotype_id, pval_nominal_phenotypes, _GWAS_SUB_NEGLOG10)
    )


def load_cross_tissue_members(coloc_dir: Path) -> pd.DataFrame:
    """Explode loci ``member_cos_ids`` × tissue ColocBoost → one row per member CoS."""
    loci_path = coloc_dir / "cross_tissue_loci.parquet"
    if not loci_path.is_file():
        raise FileNotFoundError(f"Missing loci table: {loci_path}")
    loci = pd.read_parquet(loci_path)
    if "member_cos_ids" not in loci.columns or "locus_id" not in loci.columns:
        raise KeyError("cross_tissue_loci.parquet needs locus_id + member_cos_ids")

    frames = []
    for path in sorted(coloc_dir.glob("*.ColocBoost.parquet")):
        tissue = path.name.replace(".ColocBoost.parquet", "")
        df = pd.read_parquet(path)
        df = df.copy()
        df["tissue"] = tissue
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *.ColocBoost.parquet under {coloc_dir}")
    all_cos = pd.concat(frames, ignore_index=True)
    all_cos["cos_id"] = all_cos["cos_id"].astype(str)
    cos_index = set(all_cos["cos_id"])

    memb = loci[["locus_id", "member_cos_ids"]].explode("member_cos_ids").rename(
        columns={"member_cos_ids": "cos_id"}
    )
    memb["cos_id"] = memb["cos_id"].astype(str)
    missing = ~memb["cos_id"].isin(cos_index)
    if missing.any():
        print(f"WARN: dropping {int(missing.sum()):,} members with missing ColocBoost rows")
        memb = memb.loc[~missing]
    out = memb.merge(all_cos, on="cos_id", how="left")
    leads = [
        _lead_from_lists(v, p) for v, p in zip(out["variants"], out["vcp"])
    ]
    out["lead_variant_id"] = [t[0] for t in leads]
    out["lead_vcp"] = [t[1] for t in leads]
    print(
        f"Cross-tissue members: {len(out):,} CoS in "
        f"{out['locus_id'].nunique():,} loci "
        f"({out['tissue'].nunique()} tissues)"
    )
    return out


def build_locus_cos_cat_from_long(
    locus_long: pd.DataFrame,
    lncrna_ids: set[str],
    pc_ids: set[str],
    pseudogene_ids: set[str],
) -> pd.DataFrame:
    """Locus-level CoS composition from member rows (GWAS p < 1e-5)."""
    pheno_c = "phenotype_ids"
    pval_c = "cos_max_pval_nominal_phenotypes"
    for c in (pheno_c, pval_c, "gene_ids_robust", "locus_id"):
        if c not in locus_long.columns:
            raise KeyError(f"members frame missing {c}")
    has_gws = "gwas_gws_traits" in locus_long.columns
    rows = []
    for locus_id, grp in locus_long.groupby("locus_id", sort=False):
        has_1e5 = False
        for r in grp.itertuples(index=False):
            if _row_has_gwas_1e5(
                getattr(r, pheno_c, None),
                getattr(r, pval_c, None),
                getattr(r, "gwas_gws_traits", None) if has_gws else None,
            ):
                has_1e5 = True
                break
        if not has_1e5:
            continue
        gene_stems: set[str] = set()
        for vals in grp["gene_ids_robust"]:
            gene_stems |= _as_stem_set(vals)
        if not gene_stems:
            continue
        cos_cat = _cos_composition(gene_stems, lncrna_ids, pc_ids, pseudogene_ids)
        if cos_cat in COS_CAT_ORDER:
            rows.append({"locus_id": locus_id, "cos_cat": cos_cat})
    return pd.DataFrame(rows)


def lead_vcp_by_snp_and_stratum(
    locus_long: pd.DataFrame,
    locus_cos: pd.DataFrame,
) -> pd.DataFrame:
    """Max lead VCP per SNP × cos_cat."""
    long = locus_long.copy()
    pheno_c = "phenotype_ids"
    pval_c = "cos_max_pval_nominal_phenotypes"
    has_gws = "gwas_gws_traits" in long.columns
    keep = [
        _row_has_gwas_1e5(
            pid,
            pval,
            gws if has_gws else None,
        )
        for pid, pval, gws in zip(
            long[pheno_c],
            long[pval_c],
            long["gwas_gws_traits"] if has_gws else [None] * len(long),
        )
    ]
    long = long.loc[keep]
    long = long.merge(locus_cos, on="locus_id", how="inner")
    long["lead_vcp"] = pd.to_numeric(long["lead_vcp"], errors="coerce")
    long["SNP"] = long["lead_variant_id"].map(_coloc_vid_to_snp)
    long = long.dropna(subset=["SNP", "lead_vcp", "cos_cat"])
    if long.empty:
        raise RuntimeError("No lead VCP rows after composition / GWAS filters")

    out = (
        long.groupby(["SNP", "cos_cat"], sort=False)["lead_vcp"]
        .max()
        .rename("max_vcp")
        .reset_index()
    )
    print(
        f"Unique lead SNP×stratum with VCP: {len(out):,}; "
        f"per stratum {out.groupby('cos_cat').size().to_dict()}"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--combined_annotations",
        type=Path,
        required=True,
        help="odds_annotations.parquet from 02_combine_odds_annotations.py",
    )
    p.add_argument(
        "--coloc_dir",
        type=Path,
        required=True,
        help="Main coloc dir on Zenodo ({Tissue}.ColocBoost.parquet + cross_tissue_loci.parquet)",
    )
    p.add_argument(
        "--gencode_class",
        type=Path,
        required=True,
        help="gencode_v39_v47_gene_classification.txt",
    )
    p.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output TSV of Firth ORs (all annotations × strata × VCP cutoffs)",
    )
    args = p.parse_args()
    annot_version = ANNOT_VERSION
    vcp_cutoffs = list(VCP_CUTOFFS)

    gencode = pd.read_table(args.gencode_class, low_memory=False)
    lncrna_ids = set(
        gencode.loc[
            gencode["gene_type_grouped"].isin(["novel_lncRNA", "existing_lncRNA"]),
            "gene_id_stable",
        ].dropna()
    )
    pc_ids = set(
        gencode.loc[
            gencode["gene_type_grouped"] == "protein_coding", "gene_id_stable"
        ].dropna()
    )
    pseudogene_ids = set(
        gencode.loc[
            gencode["gene_type_grouped"] == "pseudogene", "gene_id_stable"
        ].dropna()
    )

    locus_long = load_cross_tissue_members(args.coloc_dir)
    locus_cos = build_locus_cos_cat_from_long(
        locus_long, lncrna_ids, pc_ids, pseudogene_ids
    )
    print("p<1e-5 loci by exact CoS composition:")
    print(locus_cos["cos_cat"].value_counts().reindex(COS_CAT_ORDER).to_string())

    snp_vcp = lead_vcp_by_snp_and_stratum(locus_long, locus_cos)

    print(f"Loading {args.combined_annotations} …")
    annot = pd.read_parquet(args.combined_annotations)
    if annot.index.name == "SNP" or "SNP" not in annot.columns:
        annot = annot.reset_index()
        if annot.columns[0] != "SNP":
            annot = annot.rename(columns={annot.columns[0]: "SNP"})
    annot["SNP"] = annot["SNP"].astype(str)
    annot = annot.dropna(subset=["af_c"]).copy()
    annot, all_annot_cols = add_merged_annotation_columns(annot)
    feature_cols = [f"{c}_{annot_version}" for c in all_annot_cols]
    missing = [c for c in feature_cols if c not in annot.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} annotation columns missing after merge "
            f"(e.g. {missing[:5]})"
        )

    loci_bool = {}
    for base in (
        "novel_lncRNA_eqtl_loci",
        "novel_lncRNA_sqtl_loci",
        "existing_lncRNA_eqtl_loci",
        "existing_lncRNA_sqtl_loci",
        "protein_coding_eqtl_loci",
        "protein_coding_sqtl_loci",
    ):
        if base not in annot.columns:
            raise KeyError(f"Missing loci column: {base}")
        loci_bool[base] = annot[base].fillna(False).astype(bool)

    in_lnc = (
        loci_bool["novel_lncRNA_eqtl_loci"]
        | loci_bool["novel_lncRNA_sqtl_loci"]
        | loci_bool["existing_lncRNA_eqtl_loci"]
        | loci_bool["existing_lncRNA_sqtl_loci"]
    )
    in_pc = loci_bool["protein_coding_eqtl_loci"] | loci_bool["protein_coding_sqtl_loci"]
    cis_mask = {
        "lncRNA only": in_lnc.to_numpy(),
        "protein-coding only": in_pc.to_numpy(),
        "mixed": (in_lnc & in_pc).to_numpy(),
    }

    out_rows = []
    for lab in tqdm(COS_CAT_ORDER, desc="Firth coloc-VCP ORs"):
        stratum = snp_vcp[snp_vcp["cos_cat"] == lab]
        max_vcp_map = stratum.set_index("SNP")["max_vcp"]
        case_any = set(max_vcp_map.index)
        mask = cis_mask[lab] | annot["SNP"].isin(case_any).to_numpy()
        work = annot.loc[mask].copy()
        work["max_vcp"] = work["SNP"].map(max_vcp_map).fillna(0.0)
        n_cis = int(cis_mask[lab].sum())
        print(
            f"{lab}: lead SNPs with VCP={len(case_any):,}; "
            f"cis pool={n_cis:,}; fit n={len(work):,}"
        )
        for cut in vcp_cutoffs:
            y = (work["max_vcp"].to_numpy() > float(cut)).astype(float)
            print(f"  lead VCP > {cut}: cases={int(y.sum()):,}")
            fit = fit_firth_annot_odds(work, y, feature_cols, covars=["af_c"])
            fit["cos_cat"] = lab
            fit["vcp_cutoff"] = float(cut)
            fit["n_cis_pool"] = n_cis
            fit["annot_version"] = annot_version
            out_rows.append(fit)

    out = pd.concat(out_rows, ignore_index=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, sep="\t", index=False)
    print(f"Wrote {args.output_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
