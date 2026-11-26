#!/usr/bin/env python3
"""CoS ablation: protein-coding, no-novel-lncRNA, and single-phenotype refits.

Selects CoS from main ``{Tissue}.ColocBoost.parquet`` (``--coloc_dir``),
refits each restricted design, and scores recovery. Modes under ``--output_dir``::

  {output_dir}/only_protein_coding/{Tissue}.only_protein_coding.ColocBoost.parquet
  {output_dir}/only_v39/{Tissue}.only_v39.ColocBoost.parquet
  {output_dir}/single_phenotype/{Tissue}.single_phenotype.ColocBoost.parquet
  {output_dir}/recovery_by_cos.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from cluster_loci import (
    OVERLAP_MEDIAN_ABS_R,
    OVERLAP_MIN_ABS_R,
    _abs_r_matrix,
    _load_ld_block,
    _variant_to_idx,
    parse_chrom_pos,
    read_ld_meta,
)
from coloc import (
    _load_gencode,
    format_job_tag,
    keep_gene_ids_for_mode,
    unstringify_object_columns,
    write_lines,
)

ABLATION_MODES = ("only_protein_coding", "only_v39", "single_phenotype")

# --- select CoS ---


ENSG_RE = re.compile(r"(ENSG\d+)")
COLOCBOOST_SUFFIX = ".ColocBoost.parquet"
PROTEIN_CODING = "protein_coding"
NEW_LNCRNA = "novel_lncRNA"
KEEP_GENE_MODES = ("only_protein_coding", "only_v39")
RECOVERY_SHORT = {
    "only_protein_coding": "pc_only",
    "only_v39": "v39",
    "single_phenotype": "sg",
}


def _as_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, np.ndarray):
        return [v for v in x.tolist()]
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if isinstance(x, pd.Series):
        return x.tolist()
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    if not s or s in ("nan", "None", "<NA>"):
        return []
    return [x]


def _core(gid) -> str:
    m = ENSG_RE.search(str(gid))
    return m.group(1) if m else str(gid).split(".")[0]


def _gene_type(gid: str, gene_id_to_type: dict):
    core = _core(gid)
    return gene_id_to_type.get(core, gene_id_to_type.get(gid))


def _truthy(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(x, (int, np.integer)):
        return int(x) != 0
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes"}


def _tissue_from_colocboost_name(name: str, mode_tag: Optional[str] = None) -> str:
    """Parse tissue from ``{Tissue}.ColocBoost.parquet`` or ``{Tissue}.{mode}.…``."""
    if not name.endswith(COLOCBOOST_SUFFIX):
        raise ValueError(name)
    stem = name[: -len(COLOCBOOST_SUFFIX)]
    if mode_tag and stem.endswith(f".{mode_tag}"):
        return stem[: -(len(mode_tag) + 1)]
    return stem


def _as_variants(val) -> List[str]:
    return [str(v) for v in _as_list(val) if str(v).strip()]


def _lead_from_row(row) -> str:
    vs = _as_variants(row.get("variants"))
    if not vs:
        return str(row.get("lead_variant_id", "") or "")
    vcps = _as_list(row.get("vcp"))
    if vcps and len(vcps) == len(vs):
        arr = pd.to_numeric(pd.Series(vcps), errors="coerce")
        if arr.notna().any():
            return vs[int(arr.values.argmax())]
    return vs[0]


def region_id_for_variant(vid: str, ld_meta: pd.DataFrame) -> str:
    """Map a variant to ``chrN_start_end`` via LD-block metadata."""
    cp = parse_chrom_pos(vid)
    if cp is None or ld_meta is None or ld_meta.empty:
        return ""
    chrom, pos = cp
    tiles = ld_meta[
        (ld_meta["chrom"].astype(str) == str(chrom))
        & (ld_meta["start"].astype(int) <= int(pos))
        & (ld_meta["end"].astype(int) >= int(pos))
    ]
    if tiles.empty:
        return ""
    row = tiles.iloc[0]
    return f"chr{chrom}_{int(row['start'])}_{int(row['end'])}"


def is_xqtl_phenotype(pid: str) -> bool:
    pl = str(pid).lower()
    return "expression" in pl or "splicing" in pl


def xqtl_phenotypes_from_field(phenotype_id) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for part in str(phenotype_id or "").split(";"):
        s = part.strip()
        if not s or s in seen or not is_xqtl_phenotype(s):
            continue
        seen.add(s)
        out.append(s)
    return out


def list_colocboost_files(
    coloc_dir: Path, *, mode_tag: Optional[str] = None
) -> List[Path]:
    if not coloc_dir.is_dir():
        return []
    if mode_tag:
        suf = f".{mode_tag}{COLOCBOOST_SUFFIX}"
        return sorted(p for p in coloc_dir.iterdir() if p.name.endswith(suf))
    # Main output: {Tissue}.ColocBoost.parquet (no extra mode tag)
    return sorted(
        p
        for p in coloc_dir.iterdir()
        if p.name.endswith(COLOCBOOST_SUFFIX) and p.name.count(".") == 1
    )


def unique_gene_ids(gene_ids: Sequence) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for gid in gene_ids:
        if gid is None:
            continue
        s = str(gid).strip()
        if not s or s in ("nan", "None", "<NA>"):
            continue
        core = _core(s)
        if not core or core in seen:
            continue
        seen.add(core)
        out.append(s)
    return out


def _is_keep_gene(gid: str, gene_id_to_type: dict, mode: str) -> bool:
    gt = _gene_type(gid, gene_id_to_type)
    if mode == "only_protein_coding":
        return gt == PROTEIN_CODING
    if mode == "only_v39":
        # Drop only genes labeled novel_lncRNA (v47-new); keep unknown / all others.
        return gt != NEW_LNCRNA
    raise ValueError(f"keep/drop split not defined for mode={mode!r}")


def split_keep_drop(
    gene_ids: Sequence, gene_id_to_type: dict, mode: str
) -> tuple[List[str], List[str]]:
    keep_ids: List[str] = []
    drop_ids: List[str] = []
    seen: Set[str] = set()
    for gid in gene_ids:
        if gid is None:
            continue
        s = str(gid).strip()
        if not s or s in ("nan", "None", "<NA>"):
            continue
        core = _core(s)
        if core in seen:
            continue
        seen.add(core)
        if _is_keep_gene(s, gene_id_to_type, mode):
            keep_ids.append(s)
        else:
            drop_ids.append(s)
    return keep_ids, drop_ids


def select_catalog(
    coloc_dir: Path,
    gene_id_to_type: dict,
    mode: str,
    ld_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    files = list_colocboost_files(coloc_dir)
    if not files:
        print(f"[select_ablation] No ColocBoost tables in {coloc_dir}", file=sys.stderr)
        return pd.DataFrame()
    for path in files:
        tissue = _tissue_from_colocboost_name(path.name)
        df = pd.read_parquet(path)
        df = unstringify_object_columns(df)
        if df.empty:
            continue
        if "gwas_gws_traits" not in df.columns:
            print(f"[select_ablation] skip {path.name}: missing gwas_gws_traits", file=sys.stderr)
            continue
        for _, row in df.iterrows():
            gws_traits = [str(t).strip() for t in _as_list(row.get("gwas_gws_traits")) if str(t).strip()]
            if not gws_traits:
                continue
            if "gene_ids_robust" not in df.columns:
                raise KeyError(f"{path.name}: missing gene_ids_robust")
            if "gene_types_robust" not in df.columns:
                raise KeyError(f"{path.name}: missing gene_types_robust")
            gene_ids = _as_list(row.get("gene_ids_robust"))
            if mode == "single_phenotype":
                partner_phenos = xqtl_phenotypes_from_field(
                    row.get("phenotype_ids", "")
                )
                if not partner_phenos or not gws_traits:
                    continue
                partner_ids = unique_gene_ids(gene_ids)
                if not partner_ids:
                    partner_ids = unique_gene_ids(
                        [m.group(0) for p in partner_phenos for m in [ENSG_RE.search(p)] if m]
                    )
                dropped_ids: List[str] = []
            else:
                partner_ids, dropped_ids = split_keep_drop(
                    gene_ids, gene_id_to_type, mode
                )
                partner_phenos = []
                if not dropped_ids or not partner_ids:
                    continue
            region_id = str(row.get("region_id", "") or "")
            lead = _lead_from_row(row)
            if not region_id and ld_meta is not None and lead:
                region_id = region_id_for_variant(lead, ld_meta)
            if not region_id:
                continue
            cid = str(row.get("cos_id", "") or "")
            cs_vars = _as_variants(row.get("variants"))
            rec = {
                "mode": mode,
                "tissue_id": tissue,
                "region_id": region_id,
                "region_item": f"{tissue}__{region_id}",
                "coloc_id": cid,  # public cos_id (Tissue.{mode}_N)
                "cos_id": cid,
                "lead_variant_id": lead,
                "cos_variants": cs_vars,
                "phenotype_ids": row.get("phenotype_ids", ""),
                "gene_ids": gene_ids,
                "gene_types": _as_list(row.get("gene_types_robust")),
                "partner_gene_ids": partner_ids,
                "partner_phenotype_ids": partner_phenos,
                "dropped_gene_ids": dropped_ids,
                "gwas_traits": _as_list(row.get("gwas_traits")),
                "gwas_gws_traits": gws_traits,
                "vcp": row.get("vcp", np.nan),
                "cos_npc": row.get("cos_npc", np.nan),
            }
            if mode == "only_protein_coding":
                rec["protein_coding_gene_ids"] = partner_ids
                rec["noncoding_gene_ids"] = dropped_ids
            elif mode == "only_v39":
                rec["v39_gene_ids"] = partner_ids
                rec["new_lncrna_gene_ids"] = dropped_ids
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_single_pheno_jobs(catalog: pd.DataFrame) -> pd.DataFrame:
    """One row per unique tissue × region × xQTL phenotype (joint GWAS).

    ``gwas_id`` is always ``joint`` — the coloc worker keeps all GWS traits in
    the block.
    """
    rows = []
    seen: Set[tuple] = set()
    job_i = 0
    for _, row in catalog.iterrows():
        tissue = str(row["tissue_id"])
        region = str(row["region_id"])
        phenos = _as_list(row.get("partner_phenotype_ids"))
        for pheno in phenos:
            pheno_s = str(pheno).strip()
            if not pheno_s:
                continue
            key = (tissue, region, pheno_s)
            if key in seen:
                continue
            seen.add(key)
            job_i += 1
            tag = format_job_tag(job_i)
            item = f"{tissue}__{region}__{tag}"
            rows.append(
                {
                    "job_item": item,
                    "job_tag": tag,
                    "tissue_id": tissue,
                    "region_id": region,
                    "region_item": f"{tissue}__{region}",
                    "phenotype_id": pheno_s,
                    "gwas_id": "joint",
                }
            )
    return pd.DataFrame(rows)




def write_select_outputs(
    *,
    coloc_dir: Path,
    mode: str,
    gencode_classification: str,
    items_list: Path,
    catalog_path: Path,
    keep_gene_ids_path: Path | None,
    jobs_tsv: Path | None,
    ld_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Select CoS and write items/catalog (and jobs / keep genes when needed)."""
    if not Path(gencode_classification).is_file():
        raise FileNotFoundError(f"Missing gencode classification: {gencode_classification}")
    gene_id_to_type = _load_gencode(gencode_classification)
    catalog = select_catalog(coloc_dir, gene_id_to_type, mode, ld_meta=ld_meta)

    keep_ids: list[str] = []
    if mode in KEEP_GENE_MODES:
        keep_ids = keep_gene_ids_for_mode(gene_id_to_type, mode)
        filter_desc = (
            "keep protein_coding genes"
            if mode == "only_protein_coding"
            else "keep non-novel_lncRNA (v39 / non-v47-new) genes"
        )
    else:
        filter_desc = "single phenotype + joint GWAS jobs"

    jobs = pd.DataFrame()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    if catalog.empty:
        write_lines(items_list, [])
        catalog.to_parquet(catalog_path, index=False)
        if mode in KEEP_GENE_MODES and keep_gene_ids_path is not None:
            write_lines(keep_gene_ids_path, keep_ids)
        if mode == "single_phenotype" and jobs_tsv is not None:
            jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                columns=[
                    "job_item",
                    "job_tag",
                    "tissue_id",
                    "region_id",
                    "region_item",
                    "phenotype_id",
                    "gwas_id",
                ]
            ).to_csv(jobs_tsv, sep="\t", index=False)
        if mode == "single_phenotype":
            print("No qualifying GWS CoS with xQTL phenotypes.")
        elif mode == "only_v39":
            print("No qualifying GWS shared non-new + novel_lncRNA CoS.")
        else:
            print("No qualifying GWS shared protein-coding + non-coding CoS.")
        return catalog

    if mode == "single_phenotype":
        jobs = build_single_pheno_jobs(catalog)
        items = jobs["region_item"].tolist()
        if jobs_tsv is None:
            raise ValueError("single_phenotype select requires jobs_tsv path")
        jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
        jobs.to_csv(jobs_tsv, sep="\t", index=False)
        print(f"Wrote {jobs_tsv} ({len(jobs)} jobs)")
        n_filter = len(jobs)
    else:
        items = catalog["region_item"].tolist()
        n_filter = len(keep_ids)
        if keep_gene_ids_path is None:
            raise ValueError(f"{mode} select requires keep_gene_ids_path")
        write_lines(keep_gene_ids_path, keep_ids)

    catalog.to_parquet(catalog_path, index=False)
    print(f"Wrote {catalog_path} ({len(catalog)} CoS)")
    write_lines(items_list, items)
    print(
        f"mode={mode}; Qualifying CoS={len(catalog):,}; "
        f"items={len(items):,}; "
        f"tissue×regions={catalog['region_item'].nunique():,}; "
        f"tissues={catalog['tissue_id'].nunique():,}; "
        f"{filter_desc}={n_filter:,}"
    )
    return catalog




# --- recovery scoring ---

# Single-pheno ablation tables sometimes append a duplicated ENSG / cluster tail.
_EXPR_DUP_RE = re.compile(r"(expression_ENSG\d+\.\d+)_(ENSG\d+\.\d+)$")
_SPLICE_DUP_RE = re.compile(r"(splicing_.+:ENSG\d+\.\d+)_(chr.+:ENSG\d+\.\d+)$")
MODE_SUMMARY = {
    "only_protein_coding": {
        "title": "Only-protein-coding coloc recovery",
        "original": "Original GWS shared CoS (protein-coding + non-coding partner)",
        "with_leads": "CoS with only-protein-coding leads",
        "dropped": "non-protein-coding",
    },
    "only_v39": {
        "title": "Only-v39 (non-novel_lncRNA) coloc recovery",
        "original": "Original GWS shared CoS (non-novel_lncRNA + novel_lncRNA partner)",
        "with_leads": "CoS with only-v39 leads",
        "dropped": "novel_lncRNA",
    },
    "single_phenotype": {
        "title": "Single-phenotype × GWAS coloc recovery",
        "original": "Original GWS CoS with ≥1 xQTL phenotype",
        "with_leads": "CoS with single-phenotype×GWAS leads",
        "dropped": "other co-coloc phenotypes",
    },
}


def _cores(vals) -> Set[str]:
    out: Set[str] = set()
    for v in _as_list(vals):
        c = _core(v)
        if c and c not in {"nan", "None", "<NA>"}:
            out.add(c)
    return out


def _strs(vals) -> Set[str]:
    out: Set[str] = set()
    for v in _as_list(vals):
        s = str(v).strip()
        if s and s not in {"nan", "None", "<NA>"}:
            out.add(s)
    return out


def normalize_phenotype_id(pid: str) -> str:
    """Strip duplicated ENSG / splicing-cluster suffixes from ablation labels."""
    s = str(pid or "").strip()
    if not s:
        return s
    s = _EXPR_DUP_RE.sub(r"\1", s)
    s = _SPLICE_DUP_RE.sub(r"\1", s)
    return s


def phenotype_ids_from_field(val) -> Set[str]:
    """Raw and normalized phenotype tokens from a ;-separated phenotype_ids field."""
    out: Set[str] = set()
    for part in str(val or "").split(";"):
        s = part.strip()
        if not s or s in {"nan", "None", "<NA>"}:
            continue
        out.add(s)
        n = normalize_phenotype_id(s)
        if n:
            out.add(n)
    return out



def load_drop_leads(
    drop_dir: Path,
    *,
    mode_tag: str,
    ld_meta: Optional[pd.DataFrame] = None,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Map (tissue, region_id) → drop-mode ColocBoost rows for recovery matching."""
    by_key: Dict[Tuple[str, str], pd.DataFrame] = {}
    if not drop_dir.is_dir():
        return by_key
    for path in list_colocboost_files(drop_dir, mode_tag=mode_tag):
        tissue = _tissue_from_colocboost_name(path.name, mode_tag=mode_tag)
        df = unstringify_object_columns(pd.read_parquet(path))
        if df.empty:
            continue
        df = df.copy()
        df["cos_variants"] = df["variants"].map(_as_variants) if "variants" in df.columns else [[] for _ in range(len(df))]
        df["lead_variant_id"] = [_lead_from_row(r) for _, r in df.iterrows()]
        df["coloc_lead_variants"] = df["lead_variant_id"].map(lambda x: [x] if x else [])
        df["coloc_id"] = df["cos_id"].astype(str) if "cos_id" in df.columns else df.index.astype(str)
        if "phenotype_ids" not in df.columns:
            raise KeyError(f"{path.name}: ColocBoost table missing phenotype_ids")
        if "region_id" not in df.columns:
            if ld_meta is None:
                continue
            df["region_id"] = [
                region_id_for_variant(str(v), ld_meta) for v in df["lead_variant_id"]
            ]
        df = df[df["region_id"].astype(str).str.len() > 0]
        if df.empty:
            continue
        for region, grp in df.groupby(df["region_id"].astype(str), sort=False):
            by_key[(tissue, str(region))] = grp.reset_index(drop=True)
    return by_key


class LdRCache:
    """GTEx EUR LD-tile loader for pairwise cross-set |r| (no cross-call block cache)."""

    def __init__(self, ld_meta: pd.DataFrame):
        self.meta = ld_meta

    def _load_block(self, bi, row):
        ld_path = str(row["ld_path"])
        bim_path = row.get("bim_path")
        bim_file = bim_path if isinstance(bim_path, str) and bim_path else ld_path + ".bim"
        try:
            return _load_ld_block(ld_path, bim_file)
        except Exception as e:  # noqa: BLE001
            print(f"[compare_ablation] WARN LD block {bi} ({ld_path}): {e}", file=sys.stderr)
            return None

    def cross_set_abs_r(self, vars_a: Set[str], vars_b: Set[str]) -> Optional[np.ndarray]:
        positions: List[Tuple[str, int]] = []
        chroms: Set[str] = set()
        for v in set(vars_a) | set(vars_b):
            cp = parse_chrom_pos(v)
            if cp is None:
                continue
            chroms.add(cp[0])
            positions.append(cp)
        if len(chroms) != 1 or not positions:
            return None
        chrom = next(iter(chroms))
        pos_min = min(p for _, p in positions)
        pos_max = max(p for _, p in positions)
        tiles = self.meta[
            (self.meta["chrom"].astype(str) == chrom)
            & (self.meta["end"].astype(int) >= pos_min)
            & (self.meta["start"].astype(int) <= pos_max)
        ]
        values: List[float] = []
        for bi, row in tiles.iterrows():
            blk = self._load_block(bi, row)
            if blk is None:
                continue
            id2idx, pos2idx, rows = blk
            idx_a = []
            idx_b = []
            for v in vars_a:
                i = _variant_to_idx(v, id2idx, pos2idx)
                if i is not None:
                    idx_a.append(i)
            for v in vars_b:
                i = _variant_to_idx(v, id2idx, pos2idx)
                if i is not None:
                    idx_b.append(i)
            if not idx_a or not idx_b:
                continue
            for ia in idx_a:
                for ib in idx_b:
                    values.append(_abs_r_matrix(rows, ia, ib))
        if not values:
            return None
        return np.asarray(values, dtype=np.float64)


def same_signal_between_purity(
    orig_lead: str,
    orig_variants: Set[str],
    drop_leads: Set[str],
    drop_variants: Set[str],
    ld_cache: Optional[LdRCache],
) -> bool:
    """Same signal under the cross-tissue between-purity rule."""
    lead = str(orig_lead or "").strip()
    drop_lead_set = {
        str(x).strip()
        for x in drop_leads
        if str(x).strip() not in {"", "nan", "None", "<NA>"}
    }
    if lead and lead not in {"nan", "None", "<NA>"} and lead in drop_lead_set:
        return True
    if not orig_variants or not drop_variants:
        return False
    if not (orig_variants & drop_variants):
        return False
    if ld_cache is None:
        return False
    arr = ld_cache.cross_set_abs_r(orig_variants, drop_variants)
    if arr is None or arr.size == 0:
        return False
    return float(arr.min()) > OVERLAP_MIN_ABS_R or float(np.median(arr)) > OVERLAP_MEDIAN_ABS_R


def _recovery_status(n_recovered: int, n_partners: int) -> str:
    """full / partial / lost. Any recovered partner is at least partial."""
    if n_partners <= 0 or n_recovered <= 0:
        return "lost"
    if n_recovered >= n_partners:
        return "full"
    return "partial"


def _best_match(
    drop_df: Optional[pd.DataFrame],
    partner_core: str,
    gws_traits: Set[str],
    orig_variants: Set[str],
    orig_lead: str = "",
    partner_phenotype: str = "",
    ld_cache: Optional[LdRCache] = None,
) -> Tuple[bool, bool, str, int, bool]:
    """Return (recovered_gwas, recovered_gwas_gws, drop_coloc_id, n_shared, same_signal).

    Same signal = cross-tissue between-purity rule (same lead, or shared CS
    variant with min |r| > 0.5 or median |r| > 0.8).
    If partner_phenotype is set, require that phenotype (raw or normalized) in
    the drop CoS phenotype_ids field.
    """
    if drop_df is None or drop_df.empty or not gws_traits:
        return False, False, "", 0, False
    if not partner_core and not partner_phenotype:
        return False, False, "", 0, False
    partner_pheno_keys: Set[str] = set()
    if partner_phenotype:
        partner_pheno_keys.add(partner_phenotype.strip())
        n = normalize_phenotype_id(partner_phenotype)
        if n:
            partner_pheno_keys.add(n)
    best_gws = False
    best_any = False
    best_cid = ""
    best_n_shared = 0
    best_same = False
    for cid, grp in drop_df.groupby(drop_df["coloc_id"].astype(str), sort=False):
        if partner_phenotype:
            drop_phenos: Set[str] = set()
            for val in grp["phenotype_ids"] if "phenotype_ids" in grp.columns else []:
                drop_phenos |= phenotype_ids_from_field(val)
            if not (partner_pheno_keys & drop_phenos):
                continue
        elif partner_core:
            genes: Set[str] = set()
            if "gene_ids_robust" not in grp.columns:
                continue
            for val in grp["gene_ids_robust"]:
                genes |= _cores(val)
            if partner_core not in genes:
                continue
        row = grp.iloc[0]
        drop_gwas = _strs(row.get("gwas_traits"))
        drop_gws = _strs(row.get("gwas_gws_traits"))
        # Traits can differ by lead row; union across the CoS.
        for _, r in grp.iterrows():
            drop_gwas |= _strs(r.get("gwas_traits"))
            drop_gws |= _strs(r.get("gwas_gws_traits"))
        hit_any = bool(gws_traits & drop_gwas)
        hit_gws = bool(gws_traits & drop_gws)
        if not hit_any:
            continue
        drop_vars = _strs(row.get("cos_variants"))
        drop_leads = _strs(row.get("coloc_lead_variants"))
        if not same_signal_between_purity(
            orig_lead, orig_variants, drop_leads, drop_vars, ld_cache
        ):
            continue
        n_shared = len(orig_variants & drop_vars) if orig_variants and drop_vars else 0
        if hit_gws:
            return True, True, str(cid), n_shared, True
        if not best_any or n_shared > best_n_shared:
            best_any = True
            best_cid = str(cid)
            best_gws = False
            best_n_shared = n_shared
            best_same = True
    return best_any, best_gws, best_cid, best_n_shared, best_same


def compare(
    catalog: pd.DataFrame,
    drop_by_key: dict,
    mode: str = "only_protein_coding",
    ld_cache: Optional[LdRCache] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    partner_rows = []
    cos_rows = []
    use_pheno = mode == "single_phenotype"
    for _, orig in catalog.iterrows():
        tissue = str(orig["tissue_id"])
        region = str(orig["region_id"])
        drop_df = drop_by_key.get((tissue, region))
        drop_status = "ok" if drop_df is not None else "missing_leads"
        gws_traits = _strs(orig.get("gwas_gws_traits"))
        orig_variants = _strs(orig.get("cos_variants"))
        orig_lead = str(orig.get("lead_variant_id", "") or "")
        if use_pheno:
            partners = _as_list(orig.get("partner_phenotype_ids"))
            if not partners:
                partners = [
                    p.strip()
                    for p in str(orig.get("phenotype_ids", "")).split(";")
                    if p.strip() and ("expression" in p.lower() or "splicing" in p.lower())
                ]
        else:
            partners = _as_list(orig.get("partner_gene_ids"))
        n_rec_any = 0
        n_rec_gws = 0
        rec_cids: List[str] = []
        n_shared_best = 0
        any_same = False
        for partner in partners:
            if use_pheno:
                pheno = str(partner).strip()
                core = _core(pheno)
                rec_any, rec_gws, cid, n_shared, same = _best_match(
                    drop_df,
                    core,
                    gws_traits,
                    orig_variants,
                    orig_lead=orig_lead,
                    partner_phenotype=pheno,
                    ld_cache=ld_cache,
                )
                partner_gene_id = core
                partner_phenotype_id = pheno
            else:
                core = _core(partner)
                rec_any, rec_gws, cid, n_shared, same = _best_match(
                    drop_df,
                    core,
                    gws_traits,
                    orig_variants,
                    orig_lead=orig_lead,
                    ld_cache=ld_cache,
                )
                partner_gene_id = partner
                partner_phenotype_id = ""
            if rec_any:
                n_rec_any += 1
            if rec_gws:
                n_rec_gws += 1
            if cid:
                rec_cids.append(cid)
            n_shared_best = max(n_shared_best, n_shared)
            any_same = any_same or same
            partner_rows.append(
                {
                    "tissue_id": tissue,
                    "region_id": region,
                    "region_item": orig.get("region_item", f"{tissue}__{region}"),
                    "original_coloc_id": orig.get("coloc_id", ""),
                    "partner_gene_id": partner_gene_id,
                    "partner_gene_core": core,
                    "partner_phenotype_id": partner_phenotype_id,
                    "gwas_gws_traits": sorted(gws_traits),
                    "n_original_cos_variants": len(orig_variants),
                    "n_shared_cos_variants": n_shared,
                    "same_signal": same,
                    "drop_status": drop_status,
                    "recovered_with_gwas": rec_any,
                    "recovered_with_gwas_gws": rec_gws,
                    "drop_coloc_id": cid,
                }
            )
        n_part = len(partners)
        status = _recovery_status(n_rec_any, n_part)
        status_gws = _recovery_status(n_rec_gws, n_part)
        cos_rows.append(
            {
                "tissue_id": tissue,
                "region_id": region,
                "region_item": orig.get("region_item", f"{tissue}__{region}"),
                "original_coloc_id": orig.get("coloc_id", ""),
                "cos_id": orig.get("cos_id", ""),
                "lead_variant_id": orig.get("lead_variant_id", ""),
                "n_other_recovered_with_gwas": n_rec_any,
                "n_other_recovered_with_gwas_gws": n_rec_gws,
                "recovery_status": status,
                "recovery_status_gws": status_gws,
                "drop_status": drop_status,
                "drop_coloc_ids": sorted(set(rec_cids)),
                "n_original_cos_variants": len(orig_variants),
                "n_shared_cos_variants": n_shared_best,
                "same_signal": any_same,
                "gwas_gws_traits": sorted(gws_traits),
                "partner_gene_ids": _as_list(orig.get("partner_gene_ids")),
                "partner_phenotype_ids": _as_list(orig.get("partner_phenotype_ids"))
                if use_pheno
                else [],
                "dropped_gene_ids": _as_list(orig.get("dropped_gene_ids")),
            }
        )
    return pd.DataFrame(cos_rows), pd.DataFrame(partner_rows)


def write_summary(
    cos: pd.DataFrame, partners: pd.DataFrame, path: Path, mode: str
) -> None:
    labels = MODE_SUMMARY[mode]
    n_cos = len(cos)
    n_run = int((cos["drop_status"] == "ok").sum()) if n_cos else 0
    n_missing = n_cos - n_run
    run = cos[cos["drop_status"] == "ok"] if n_cos else cos
    n_full = int((run["recovery_status"] == "full").sum()) if n_run else 0
    n_partial = int((run["recovery_status"] == "partial").sum()) if n_run else 0
    n_lost = int((run["recovery_status"] == "lost").sum()) if n_run else 0
    n_full_gws = int((run["recovery_status_gws"] == "full").sum()) if n_run else 0
    n_partial_gws = int((run["recovery_status_gws"] == "partial").sum()) if n_run else 0
    n_lost_gws = int((run["recovery_status_gws"] == "lost").sum()) if n_run else 0
    n_any = n_full + n_partial
    n_regions = int(cos["region_item"].nunique()) if n_cos else 0
    n_part = len(partners)
    n_part_any = int(partners["recovered_with_gwas"].sum()) if n_part else 0
    n_part_gws = int(partners["recovered_with_gwas_gws"].sum()) if n_part else 0
    dropped = labels["dropped"]

    def _frac(n: int) -> str:
        return f"{n} / {n_run}" + (f" ({100.0 * n / n_run:.1f}%)" if n_run else "")

    lines = [
        labels["title"],
        "",
        f"{labels['original']}: {n_cos}",
        f"Unique tissue × LD-blocks: {n_regions}",
        f"{labels['with_leads']}: {n_run}",
        f"CoS missing drop leads: {n_missing}",
        "",
        "Among CoS with drop leads (same signal = between-purity / cross-tissue rule):",
        f"  fully recovered (all partners still coloc with GWAS): {_frac(n_full)}",
        f"  partially recovered (some but not all partners still coloc with GWAS): {_frac(n_partial)}",
        f"  lost (no partner still coloc with GWAS): {_frac(n_lost)}",
        f"  any recovered (full + partial): {_frac(n_any)}",
        "",
        "  still GWS in the drop CoS:",
        f"    fully recovered: {_frac(n_full_gws)}",
        f"    partially recovered: {_frac(n_partial_gws)}",
        f"    lost: {_frac(n_lost_gws)}",
        "",
        f"Partner genes: {n_part}",
        f"  recovered with GWAS at the same signal: {n_part_any}"
        + (f" ({100.0 * n_part_any / n_part:.1f}%)" if n_part else ""),
        f"  recovered with GWAS still GWS at the same signal: {n_part_gws}"
        + (f" ({100.0 * n_part_gws / n_part:.1f}%)" if n_part else ""),
        "",
        "Same signal = same lead, or shared CS variant with min |r| > 0.5",
        "or median |r| > 0.8 (cross-tissue between-purity rule; CoS IDs not compared).",
        "A gene+GWAS coloc on a different independent signal in the block is not a recovery.",
        f"Lost CoS required {dropped} for that shared GWAS signal.",
        "Partial recovery means at least one remaining partner still coloc with the GWAS.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text, end="")
    print(f"Wrote {path}")




def write_compare_outputs(
    *,
    catalog_path: Path,
    drop_dir: Path,
    mode: str,
    ld_meta: str,
) -> pd.DataFrame:
    """Score recovery; return per-mode cos recovery table (keyed by cos_id)."""
    if not ld_meta:
        raise ValueError("--ld_meta is required for between-purity matching")
    if not Path(ld_meta).is_file():
        raise FileNotFoundError(f"--ld_meta not found: {ld_meta}")

    catalog = unstringify_object_columns(pd.read_parquet(catalog_path))
    meta = read_ld_meta(ld_meta)
    drop_by_key = load_drop_leads(drop_dir, mode_tag=mode, ld_meta=meta)
    ld_cache = LdRCache(meta)
    print(
        f"[compare_ablation] mode={mode}; catalog CoS={len(catalog):,}; "
        f"drop tissue×regions={len(drop_by_key):,}; same_signal=between_purity "
        f"(min|r|>{OVERLAP_MIN_ABS_R}, median|r|>{OVERLAP_MEDIAN_ABS_R})"
    )
    cos, partners = compare(catalog, drop_by_key, mode=mode, ld_cache=ld_cache)
    # Internal diagnostics under items/
    diag = catalog_path.parent / f"{mode}_recovery"
    cos.to_parquet(Path(str(diag) + ".cos.parquet"), index=False)
    partners.to_parquet(Path(str(diag) + ".partners.parquet"), index=False)
    write_summary(cos, partners, Path(str(diag) + ".summary.txt"), mode)
    return cos


def build_recovery_by_cos(
    mode_cos: Dict[str, pd.DataFrame],
    drop_dirs: Dict[str, Path],
) -> pd.DataFrame:
    """Wide recovery table keyed by main ``cos_id``."""
    frames = []
    for mode, cos in mode_cos.items():
        key = RECOVERY_SHORT.get(mode)
        if not key or cos is None or cos.empty:
            continue
        drop_lookup: Dict[str, pd.Series] = {}
        ddir = drop_dirs.get(mode)
        if ddir is not None and ddir.is_dir():
            for path in list_colocboost_files(ddir, mode_tag=mode):
                df = unstringify_object_columns(pd.read_parquet(path))
                if "cos_id" not in df.columns:
                    continue
                for _, r in df.iterrows():
                    drop_lookup[str(r["cos_id"])] = r
        rows = []
        for _, r in cos.iterrows():
            main_cid = str(r.get("cos_id", "") or "")
            if not main_cid:
                continue
            status = r.get("recovery_status", "lost")
            drop_cids = r.get("drop_coloc_ids", [])
            if isinstance(drop_cids, np.ndarray):
                drop_cids = drop_cids.tolist()
            if not isinstance(drop_cids, (list, tuple)):
                drop_cids = []
            pheno = pd.NA
            npc = pd.NA
            npc_out = pd.NA
            if status != "lost" and drop_cids:
                hit = drop_lookup.get(str(drop_cids[0]))
                if hit is not None:
                    pheno = hit.get("phenotype_ids", pd.NA)
                    npc = hit.get("cos_npc", pd.NA)
                    npc_out = hit.get("npc_outcomes", pd.NA)
            rows.append(
                {
                    "cos_id": main_cid,
                    f"{key}_phenotypes": pheno,
                    f"{key}_recovery": status,
                    f"{key}_cos_npc": npc,
                    f"{key}_npc_outcomes": npc_out,
                }
            )
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    rec = frames[0]
    for extra in frames[1:]:
        rec = rec.merge(extra, on="cos_id", how="outer")
    for key in RECOVERY_SHORT.values():
        for suffix in ("phenotypes", "recovery", "cos_npc", "npc_outcomes"):
            col = f"{key}_{suffix}"
            if col not in rec.columns:
                rec[col] = pd.NA
    return rec


def run_one_mode(mode: str, args: argparse.Namespace) -> pd.DataFrame:
    root = Path(args.output_dir)
    coloc_dir = Path(args.coloc_dir)

    drop_root = root / mode
    items_dir = drop_root / "items"
    items_list = items_dir / "items.txt"
    keep_gene_id_path = items_dir / "keep_gene_ids.txt"
    catalog = items_dir / "catalog.parquet"
    jobs_tsv = items_dir / "jobs.tsv"
    items_dir.mkdir(parents=True, exist_ok=True)

    meta = read_ld_meta(args.ld_meta)
    print(f"[ablation/{mode}] select CoS from {coloc_dir}")
    write_select_outputs(
        coloc_dir=coloc_dir,
        mode=mode,
        gencode_classification=args.gencode_classification,
        items_list=items_list,
        catalog_path=catalog,
        keep_gene_ids_path=keep_gene_id_path if mode in KEEP_GENE_MODES else None,
        jobs_tsv=jobs_tsv if mode == "single_phenotype" else None,
        ld_meta=meta,
    )

    if not items_list.is_file() or items_list.stat().st_size == 0:
        raise SystemExit(f"[ablation/{mode}] no qualifying CoS; nothing to refit")
    if mode in KEEP_GENE_MODES:
        if not keep_gene_id_path.is_file() or keep_gene_id_path.stat().st_size == 0:
            raise SystemExit(
                f"[ablation/{mode}] empty keep_gene_ids from gencode classification"
            )
    if mode == "single_phenotype":
        if not jobs_tsv.is_file() or jobs_tsv.stat().st_size == 0:
            raise SystemExit(f"[ablation/{mode}] empty jobs.tsv after select")
    if not catalog.is_file() or catalog.stat().st_size == 0:
        raise SystemExit(f"[ablation/{mode}] missing catalog.parquet after select")

    from coloc import coloc

    coloc(
        items_list=items_list,
        coloc_base_dir=drop_root / "colocboost_tmp",
        tissue_output_dir=drop_root,
        full_dist_output_dir=drop_root / "tissue_full_dist",
        paths=args,
        keep_gene_id_path=keep_gene_id_path if mode in KEEP_GENE_MODES else None,
        jobs_tsv=jobs_tsv if mode == "single_phenotype" else None,
        label=f"ablation/{mode}",
        mode_tag=mode,
        drop_region_id=True,
    )

    print(f"[ablation/{mode}] compare recovery")
    cos = write_compare_outputs(
        catalog_path=catalog,
        drop_dir=drop_root,
        mode=mode,
        ld_meta=args.ld_meta,
    )
    print(f"[ablation/{mode}] done → {drop_root}")
    return cos


def run_all_modes(args: argparse.Namespace) -> int:
    coloc_dir = Path(args.coloc_dir)
    if not coloc_dir.is_dir():
        raise FileNotFoundError(f"Missing --coloc_dir: {coloc_dir}")
    if not Path(args.gencode_classification).is_file():
        raise FileNotFoundError(
            f"Missing --gencode_classification: {args.gencode_classification}"
        )
    mode_cos: Dict[str, pd.DataFrame] = {}
    drop_dirs: Dict[str, Path] = {}
    for mode in ABLATION_MODES:
        mode_cos[mode] = run_one_mode(mode, args)
        drop_dirs[mode] = Path(args.output_dir) / mode
    rec = build_recovery_by_cos(mode_cos, drop_dirs)
    out = Path(args.output_dir) / "recovery_by_cos.parquet"
    rec.to_parquet(out, index=False)
    print(f"[ablation] wrote {out} ({len(rec):,})")
    print(f"[ablation] done all modes → {args.output_dir}")
    return 0


def main() -> int:
    from coloc import add_coloc_path_args

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True, help="Ablation output root")
    p.add_argument(
        "--coloc_dir",
        required=True,
        help="Main multi-tissue ColocBoost dir from 01 ({Tissue}.ColocBoost.parquet)",
    )
    add_coloc_path_args(p)
    return run_all_modes(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

