#!/usr/bin/env python3
"""Step 1 — build paper MaxPIP QTL .annot.gz sets and 0.01 cM coverage table.

Writes ``{output_dir}/{annot}/{annot}.{chr}.annot.gz`` for each of the 18 paper
annotations, then refreshes ``{output_dir}/annot_snp_cm_coverage.tsv`` from
whatever paper annots are present under ``--output_dir``.

Downsampled protein-coding tracks need ``lncRNA_all_*`` 0.01 cM coverage as a
target: one invocation builds those match annots first, then downsamples.
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DOWNSAMPLE_GENESET = "protein_coding_downsampled"
DOWNSAMPLE_MATCH_GENESET = "lncRNA_all"
DOWNSAMPLE_SOURCE_GENESET = "protein_coding"
DOWNSAMPLE_SEED = 10_001
# Match % of 0.01 cM blocks overlapped (Tashman et al.); sample genes, keep all CS per gene.
DOWNSAMPLE_BIN_CM = 0.01
CHROMOSOMES = [str(c) for c in range(1, 23)]
EQTL_SUSIE_PATTERN = "*.v11.eQTLs.SuSiE_summary.parquet"
SQTL_SUSIE_PATTERN = "*.v11.sQTLs.SuSiE_summary.parquet"

# SuSiE parquet filenames containing this token are excluded for *_noTestis annots.
NOTESTIS_EXCLUDE_TOKEN = "Testis"

# Supp Fig 11a coverage table (paper biotype × QTL × tissue-set grid).
COVERAGE_BIOTYPES = (
    "protein_coding",
    "lncRNA_all",
    "existing_lncRNA",
    "novel_lncRNA",
)
COVERAGE_QTL_TYPES = ("eqtl", "sqtl")
COVERAGE_TISSUE_SETS = ("all_tissues", "excl_testis")
COVERAGE_BIN_CM = 0.01

GENESETS = [
    "protein_coding",
    "existing_lncRNA",
    "novel_lncRNA",
    "lncRNA_all",
]

ANNOTATIONS: list[str] = []
for _b in COVERAGE_BIOTYPES:
    ANNOTATIONS.extend(
        (
            f"{_b}_eQTL",
            f"{_b}_sQTL",
            f"{_b}_eQTL_noTestis",
            f"{_b}_sQTL_noTestis",
        )
    )
ANNOTATIONS.extend(
    (
        f"{DOWNSAMPLE_GENESET}_eQTL_noTestis",
        f"{DOWNSAMPLE_GENESET}_sQTL_noTestis",
    )
)


def _is_downsample_pc_geneset(geneset: str) -> bool:
    return geneset == DOWNSAMPLE_GENESET


def _base_annot_name(annotation: str) -> str:
    if annotation.endswith("_noTestis"):
        return annotation[: -len("_noTestis")]
    return annotation


def _exclude_testis_for_annot(annotation: str) -> bool:
    return annotation.endswith("_noTestis")


def _filter_susie_paths(paths: list[Path], *, exclude_testis: bool) -> list[Path]:
    if not exclude_testis:
        return paths
    return [p for p in paths if NOTESTIS_EXCLUDE_TOKEN not in p.name]


def _tissue_from_parquet_path(path: Path) -> str:
    """Extract GTEx tissue token from v11 SuSiE summary parquet filename."""
    stem = path.name
    for token in (
        ".v11.eQTLs.SuSiE_summary.parquet",
        ".v11.sQTLs.SuSiE_summary.collapsed.parquet",
        ".v11.sQTLs.SuSiE_summary.parquet",
    ):
        if token in stem:
            return stem.split(token)[0]
    raise ValueError(f"Cannot parse tissue from SuSiE parquet path: {path}")


def _geneset_and_qtl_suffix(annotation: str) -> tuple[str, str]:
    stem = _base_annot_name(annotation)
    if stem.endswith("_eQTL"):
        geneset = stem[: -len("_eQTL")]
        qtl = "eQTL"
    elif stem.endswith("_sQTL"):
        geneset = stem[: -len("_sQTL")]
        qtl = "sQTL"
    else:
        raise ValueError(f"Unknown annotation {annotation!r}; expected eQTL/sQTL suffix")
    if geneset not in GENESETS and not _is_downsample_pc_geneset(geneset):
        raise ValueError(f"Invalid annotation name {annotation!r} (geneset {geneset!r})")
    return geneset, qtl


def _normalize_chr(x: str) -> str:
    x = str(x).strip()
    if x.lower().startswith("chr"):
        x = x[3:]
    return x


def _normalize_gene(x: str) -> str:
    return str(x).split(".")[0]


def _stable_gene_from_phenotype(series: pd.Series) -> pd.Series:
    """SuSiE phenotype_id: ENSG...transcript.ver, or chr:coords:...:ENSG.ver -> stable ENSG (no version)."""
    stem = series.astype(str).str.split(".", n=1).str[0]
    return stem.str.split(":").str[-1].map(_normalize_gene)


def _read_gene_types(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["gene_id_stable", "gene_id_v47", "gene_type_grouped"],
    )
    df["gene_id_stable"] = df["gene_id_stable"].astype(str).map(_normalize_gene)
    df["gene_id_v47"] = df["gene_id_v47"].astype(str)
    df["gene_type_grouped"] = df["gene_type_grouped"].astype(str).str.strip()
    return df


def _parse_variant_id(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.astype(str).str.split("_", n=2, expand=True)
    chrom = parts[0].map(_normalize_chr)
    pos = pd.to_numeric(parts[1], errors="coerce")
    return chrom, pos


def _pip_weights(df: pd.DataFrame) -> pd.Series:
    return df["pip"].fillna(0.0).astype(float)


def _gene_series(df: pd.DataFrame, gene_col: str, *, gene_from_phenotype_id: bool) -> pd.Series:
    if gene_from_phenotype_id:
        return _stable_gene_from_phenotype(df[gene_col])
    return df[gene_col].astype(str).map(_normalize_gene)


def _credible_set_key(
    tissue: str,
    gene: str,
    cs_id,
) -> tuple[str, str, str]:
    return (tissue, gene, str(cs_id))


def _001cm_coverage_from_annot_dir(
    annot_dir: Path,
    annot_name: str,
    chroms: list[str],
    bin_cm: float = DOWNSAMPLE_BIN_CM,
) -> tuple[dict[str, dict[int, float]], int, float]:
    """Fraction of 0.01 cM blocks overlapped (annot > 0), plus (chrom→bp→CM) map.

    Returns (bp_cm, n_blocks, frac_hit). A block is hit if any SNP in it has annot > 0.
    """
    bp_cm: dict[str, dict[int, float]] = {}
    n_blocks = 0
    n_hit = 0
    for chrom in chroms:
        path = annot_dir / f"{annot_name}.{chrom}.annot.gz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing match-annotation file for 0.01 cM target: {path}. "
                f"Build {annot_name} before downsampled protein_coding tracks."
            )
        cms: list[float] = []
        hit_bins: set[int] = set()
        chrom_bp: dict[int, float] = {}
        with gzip.open(path, "rt") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            try:
                cm_i = header.index("CM")
                bp_i = header.index("BP")
                ann_i = header.index(annot_name)
            except ValueError as exc:
                raise ValueError(f"Unexpected annot header in {path}: {header}") from exc
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                bp = int(parts[bp_i])
                cm = float(parts[cm_i])
                val = float(parts[ann_i])
                chrom_bp[bp] = cm
                if np.isfinite(cm) and cm >= 0:
                    cms.append(cm)
                    if val > 0:
                        hit_bins.add(int(np.floor(cm / bin_cm)))
        bp_cm[chrom] = chrom_bp
        if not cms:
            continue
        start = int(np.floor(min(cms) / bin_cm))
        end = int(np.floor(max(cms) / bin_cm))
        n_blocks += end - start + 1
        n_hit += sum(1 for b in hit_bins if start <= b <= end)
    if n_blocks == 0:
        raise ValueError(f"No 0.01 cM blocks found under {annot_dir} for {annot_name}")
    return bp_cm, n_blocks, n_hit / n_blocks


def _load_bim_bp_sets(bfile_prefix: str, chroms: list[str]) -> tuple[dict[str, set[int]], int]:
    bim_bps: dict[str, set[int]] = {}
    n_snps = 0
    for chrom in chroms:
        bim = _read_bim(Path(f"{bfile_prefix}{chrom}.bim"))
        bps = set(int(x) for x in bim["BP"].to_numpy())
        bim_bps[chrom] = bps
        n_snps += len(bps)
    return bim_bps, n_snps


def _index_credible_set_weights(
    parquet_paths: list[Path],
    gene_set: set[str],
    gene_col: str,
    bim_bps: dict[str, set[int]],
    *,
    gene_from_phenotype_id: bool = False,
) -> dict[tuple[str, str, str], list[tuple[str, int, float]]]:
    """Map each gene×tissue CS to its reference-bim (chrom, pos, value) contributions."""
    out: dict[tuple[str, str, str], dict[tuple[str, int], float]] = defaultdict(dict)
    cols = [gene_col, "variant_id", "pip", "cs_id"]
    allowed_chroms = set(bim_bps)
    for i_path, path in enumerate(parquet_paths, start=1):
        print(f"[annot] index CS weights {i_path}/{len(parquet_paths)}: {path.name}", flush=True)
        tissue = _tissue_from_parquet_path(path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=cols, batch_size=200_000):
            df = batch.to_pandas()
            if df.empty:
                continue
            genes = _gene_series(df, gene_col, gene_from_phenotype_id=gene_from_phenotype_id)
            keep = genes.isin(gene_set) & df["cs_id"].notna()
            if not keep.any():
                continue
            df = df.loc[keep].copy()
            df["gene"] = genes.loc[keep].to_numpy()
            df["cs_id"] = df["cs_id"].astype(str)
            chrom, pos = _parse_variant_id(df["variant_id"])
            df["chr"] = chrom
            df["pos"] = pos
            df = df[df["chr"].isin(allowed_chroms)].dropna(subset=["pos"])
            if df.empty:
                continue
            df["pos"] = df["pos"].astype(int)
            df["value"] = _pip_weights(df)
            for tissue_gene_cs, sub in df.groupby(["gene", "cs_id"], sort=False):
                gene, cs_id = tissue_gene_cs
                key = _credible_set_key(tissue, str(gene), cs_id)
                slot = out[key]
                for c, p, v in zip(sub["chr"].to_numpy(), sub["pos"].to_numpy(), sub["value"].to_numpy()):
                    c = str(c)
                    p = int(p)
                    if p not in bim_bps[c]:
                        continue
                    v = float(v)
                    prev = slot.get((c, p), 0.0)
                    if v > prev:
                        slot[(c, p)] = v
    flat: dict[tuple[str, str, str], list[tuple[str, int, float]]] = {}
    for key, pos_map in out.items():
        if not pos_map:
            continue
        flat[key] = [(c, p, v) for (c, p), v in pos_map.items()]
    print(f"[annot] indexed {len(flat):,} protein_coding CS with ≥1 reference SNP", flush=True)
    return flat


def _downsample_genes_match_001cm(
    cs_weights: dict[tuple[str, str, str], list[tuple[str, int, float]]],
    bp_cm: dict[str, dict[int, float]],
    n_blocks: int,
    target_frac: float,
    seed: int,
    bin_cm: float = DOWNSAMPLE_BIN_CM,
) -> tuple[set[tuple[str, str, str]], float, int, int]:
    """Random gene order; keep all CS per gene until % 0.01 cM blocks best matches target."""
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    if not cs_weights:
        raise ValueError("No protein_coding credible sets available to downsample")

    cs_by_gene: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for cs_key in cs_weights:
        _tissue, gene, _cs_id = cs_key
        cs_by_gene[gene].append(cs_key)

    rng = np.random.default_rng(seed)
    genes = sorted(cs_by_gene)
    genes = [genes[i] for i in rng.permutation(len(genes))]

    hit_bins: set[tuple[str, int]] = set()
    prefix_frac = [0.0]
    prefix_n_cs = [0]
    for gene in genes:
        for cs in cs_by_gene[gene]:
            for chrom, pos, val in cs_weights[cs]:
                if val <= 0:
                    continue
                cm = bp_cm.get(chrom, {}).get(pos)
                if cm is None or not np.isfinite(cm) or cm < 0:
                    continue
                hit_bins.add((chrom, int(np.floor(cm / bin_cm))))
        prefix_frac.append(len(hit_bins) / n_blocks)
        prefix_n_cs.append(prefix_n_cs[-1] + len(cs_by_gene[gene]))

    best_n = min(range(len(prefix_frac)), key=lambda n: (abs(prefix_frac[n] - target_frac), n))
    if best_n == 0 and target_frac > 0:
        best_n = 1
    picked_genes = genes[:best_n]
    picked = {cs for g in picked_genes for cs in cs_by_gene[g]}
    return picked, float(prefix_frac[best_n]), best_n, int(prefix_n_cs[best_n])


def _accumulate_variant_weights(
    parquet_paths: list[Path],
    gene_set: set[str],
    gene_col: str,
    allowed_chroms: set[str],
    *,
    gene_from_phenotype_id: bool = False,
    allowed_credible_sets: set[tuple[str, str, str]] | None = None,
) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    cols = [gene_col, "variant_id", "pip", "cs_id"]
    for i_path, path in enumerate(parquet_paths, start=1):
        print(f"[annot] scanning {i_path}/{len(parquet_paths)}: {path}", flush=True)
        tissue = _tissue_from_parquet_path(path)
        pf = pq.ParquetFile(path)
        n_batches = 0
        n_rows_kept = 0
        for batch in pf.iter_batches(columns=cols, batch_size=200_000):
            df = batch.to_pandas()
            if df.empty:
                continue
            n_batches += 1
            genes = _gene_series(df, gene_col, gene_from_phenotype_id=gene_from_phenotype_id)
            keep = genes.isin(gene_set)
            if allowed_credible_sets is not None:
                keep = keep & df["cs_id"].notna()
            if not keep.any():
                continue
            df = df.loc[keep].copy()
            if allowed_credible_sets is not None:
                df["gene"] = genes.loc[keep].to_numpy()
                df["cs_id"] = df["cs_id"].astype(str)
                cs_keys = [
                    _credible_set_key(tissue, g, cs)
                    for g, cs in zip(df["gene"], df["cs_id"])
                ]
                df = df.loc[[k in allowed_credible_sets for k in cs_keys]]
                if df.empty:
                    continue
            n_rows_kept += len(df)
            chrom, pos = _parse_variant_id(df["variant_id"])
            df["chr"] = chrom
            df["pos"] = pos
            df = df[df["chr"].isin(allowed_chroms)].dropna(subset=["pos"])
            if df.empty:
                continue
            df["pos"] = df["pos"].astype(int)
            df["value"] = _pip_weights(df)
            if df.empty:
                continue
            grouped = df.groupby(["chr", "pos"], as_index=False)["value"].max()
            for _, r in grouped.iterrows():
                c = r["chr"]
                p = int(r["pos"])
                v = float(r["value"])
                prev = out[c].get(p, 0.0)
                if v > prev:
                    out[c][p] = v
        print(f"[annot] done {path.name}: batches={n_batches} kept_rows={n_rows_kept}", flush=True)
    return out


def _read_bim(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"], usecols=[0, 1, 2, 3])
    df["CHR"] = df["CHR"].astype(str).map(_normalize_chr)
    df["BP"] = df["BP"].astype(int)
    # Must preserve Plink .bim row order. LDSC checks annot rows against the .bim file in file order;
    # sorting by BP breaks that and raises ValueError.
    return df.reset_index(drop=True)


def annot_001cm_coverage(
    annot_dir: Path,
    annot_name: str,
    *,
    bin_cm: float = COVERAGE_BIN_CM,
) -> dict[str, float | int]:
    """Fraction of 0.01 cM blocks with ≥1 SNP where annotation > 0."""
    n_blocks = 0
    n_hit = 0
    for chrom in range(1, 23):
        path = annot_dir / f"{annot_name}.{chrom}.annot.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing annotation file: {path}")
        df_ch = pd.read_csv(path, sep="\t", usecols=["CM", annot_name])
        vals = df_ch[annot_name].fillna(0.0).to_numpy()
        cm = df_ch["CM"].to_numpy()
        ok = np.isfinite(cm) & (cm >= 0)
        cm_ok, vals_ok = cm[ok], vals[ok]
        if len(cm_ok) == 0:
            continue
        start = np.floor(cm_ok.min() / bin_cm)
        end = np.floor(cm_ok.max() / bin_cm)
        n_blocks += int(end - start + 1)
        hit = np.unique(np.floor(cm_ok[vals_ok > 0] / bin_cm))
        n_hit += int(((hit >= start) & (hit <= end)).sum())
    pct_001cm = 100.0 * n_hit / n_blocks if n_blocks else np.nan
    return {
        "n_001cm_blocks": n_blocks,
        "n_001cm_hit": n_hit,
        "pct_001cm": pct_001cm,
    }


def build_coverage_table(ldscores_dir: Path) -> pd.DataFrame:
    """Paper biotype × QTL × tissue-set coverage; skips missing annotations."""
    rows: list[dict] = []
    skipped: list[str] = []
    for biotype in COVERAGE_BIOTYPES:
        for qtl in COVERAGE_QTL_TYPES:
            for tissues in COVERAGE_TISSUE_SETS:
                qtl_tag = "eQTL" if qtl == "eqtl" else "sQTL"
                suffix = "" if tissues == "all_tissues" else "_noTestis"
                annot = f"{biotype}_{qtl_tag}{suffix}"
                annot_dir = ldscores_dir / annot
                first = annot_dir / f"{annot}.1.annot.gz"
                if not first.exists():
                    skipped.append(annot)
                    continue
                stats = annot_001cm_coverage(annot_dir, annot)
                rows.append(
                    {
                        "biotype": biotype,
                        "qtl": qtl,
                        "tissues": tissues,
                        "annot": annot,
                        **stats,
                    }
                )
    if skipped:
        print(f"[annot] coverage skipped missing annots: {skipped}", flush=True)
    return pd.DataFrame(rows)


def _write_annot(df: pd.DataFrame, out_path: Path, annot_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as fh:
        fh.write(f"CHR\tBP\tSNP\tCM\t{annot_name}\n")
        for _, row in df.iterrows():
            fh.write(f"{row['CHR']}\t{row['BP']}\t{row['SNP']}\t{row['CM']}\t{row[annot_name]:.6f}\n")


def _apply_sparse_map_to_bim(bim: pd.DataFrame, sparse: dict[str, dict[int, float]], chrom: str) -> np.ndarray:
    vals = np.zeros(len(bim), dtype=float)
    cmap = sparse.get(chrom, {})
    if not cmap:
        return vals
    idx = {bp: i for i, bp in enumerate(bim["BP"].to_numpy())}
    for bp, w in cmap.items():
        i = idx.get(bp)
        if i is not None:
            vals[i] = max(vals[i], w)
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfile_prefix", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gencode_gene_classification_tsv", required=True)
    parser.add_argument("--eqtl_susie_dir", required=True)
    parser.add_argument("--sqtl_susie_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chroms = list(CHROMOSOMES)
    chrom_set = set(chroms)

    def _is_ds_annot(annot: str) -> bool:
        return _is_downsample_pc_geneset(_geneset_and_qtl_suffix(annot)[0])

    # Non-downsample first (includes lncRNA_all_* match annots), then downsampled PC.
    phase1 = [a for a in ANNOTATIONS if not _is_ds_annot(a)]
    phase2 = [a for a in ANNOTATIONS if _is_ds_annot(a)]

    print(f"[annot] start: phase1={phase1} phase2_downsample={phase2}", flush=True)
    print(f"[annot] chroms: {chroms}", flush=True)

    types_full = _read_gene_types(Path(args.gencode_gene_classification_tsv))
    v47_present = (
        types_full["gene_id_v47"].notna()
        & (types_full["gene_id_v47"] != "NA")
        & (types_full["gene_id_v47"] != "")
        & (types_full["gene_id_v47"] != "nan")
    )
    types = types_full[v47_present].copy()
    non_artifact = set(
        types.loc[types["gene_type_grouped"] != "artifact", "gene_id_stable"].astype(str).map(_normalize_gene)
    )

    sets: dict[str, set[str]] = {}
    sets["protein_coding"] = set(types.loc[types["gene_type_grouped"] == "protein_coding", "gene_id_stable"])
    sets["existing_lncRNA"] = set(types.loc[types["gene_type_grouped"] == "existing_lncRNA", "gene_id_stable"])
    sets["novel_lncRNA"] = set(types.loc[types["gene_type_grouped"] == "novel_lncRNA", "gene_id_stable"])
    sets["lncRNA_all"] = sets["existing_lncRNA"] | sets["novel_lncRNA"]

    for k, s in sets.items():
        sets[k] = set(s) & non_artifact

    eqtl_paths_all = sorted(Path(args.eqtl_susie_dir).glob(EQTL_SUSIE_PATTERN))
    if not eqtl_paths_all:
        raise FileNotFoundError(
            f"No eQTL SuSiE summary files found in {args.eqtl_susie_dir} with pattern {EQTL_SUSIE_PATTERN}"
        )
    sqtl_paths_all = sorted(Path(args.sqtl_susie_dir).glob(SQTL_SUSIE_PATTERN))
    if not sqtl_paths_all:
        raise FileNotFoundError(
            f"No sQTL SuSiE summary files found in {args.sqtl_susie_dir} with pattern {SQTL_SUSIE_PATTERN}"
        )

    def _build_sparse_maps(
        annots: list[str],
        *,
        allowed_cs_by_geneset: dict[str, set[tuple[str, str, str]]],
    ) -> dict[str, dict[str, dict[int, float]]]:
        sparse: dict[str, dict[str, dict[int, float]]] = {}
        for annot in annots:
            gs, kind = _geneset_and_qtl_suffix(annot)
            if _is_downsample_pc_geneset(gs):
                gset = sets[DOWNSAMPLE_SOURCE_GENESET]
                allowed_cs = allowed_cs_by_geneset[f"{gs}:{kind}"]
            else:
                gset = sets.get(gs, set())
                allowed_cs = None
            exclude_testis = _exclude_testis_for_annot(annot)
            print(
                f"[annot] build sparse annot={annot} geneset={gs} "
                f"exclude_testis={exclude_testis} n_genes={len(gset)}",
                flush=True,
            )
            if kind == "eQTL":
                paths = _filter_susie_paths(eqtl_paths_all, exclude_testis=exclude_testis)
                sparse[annot] = _accumulate_variant_weights(
                    paths,
                    gset,
                    "phenotype_id",
                    chrom_set,
                    gene_from_phenotype_id=True,
                    allowed_credible_sets=allowed_cs,
                )
            else:
                paths = _filter_susie_paths(sqtl_paths_all, exclude_testis=exclude_testis)
                sparse[annot] = _accumulate_variant_weights(
                    paths,
                    gset,
                    "gene_id",
                    chrom_set,
                    allowed_credible_sets=allowed_cs,
                )
        return sparse

    def _write_annots(
        annots: list[str],
        sparse_maps: dict[str, dict[str, dict[int, float]]],
    ) -> tuple[int, dict[str, int]]:
        def _write_one_chrom(chrom: str) -> tuple[int, dict[str, int]]:
            print(f"[annot] write chrom={chrom} annots={annots}", flush=True)
            bim = _read_bim(Path(f"{args.bfile_prefix}{chrom}.bim"))
            nchrom = len(bim)
            chrom_nz: dict[str, int] = {k: 0 for k in annots}
            for annot in annots:
                vals = _apply_sparse_map_to_bim(bim, sparse_maps[annot], chrom)
                chrom_nz[annot] += int((vals > 0).sum())
                df = bim.copy()
                df[annot] = vals
                _write_annot(df, out_dir / annot / f"{annot}.{chrom}.annot.gz", annot)
            return nchrom, chrom_nz

        total_snps = 0
        nonzero_counts = {k: 0 for k in annots}
        for chrom in chroms:
            nchrom, chrom_nz = _write_one_chrom(chrom)
            total_snps += nchrom
            for a in annots:
                nonzero_counts[a] += chrom_nz[a]

        for annot in annots:
            prop = nonzero_counts[annot] / total_snps if total_snps > 0 else 0.0
            print(
                f"[annot] summary annot={annot} nonzero={nonzero_counts[annot]} "
                f"total_snps={total_snps} prop={prop:.6g}",
                flush=True,
            )
        return total_snps, nonzero_counts

    # Phase 1: non-downsample tracks (includes lncRNA_all match annots).
    _write_annots(phase1, _build_sparse_maps(phase1, allowed_cs_by_geneset={}))

    # Phase 2: read match-annot coverage from disk, then downsampled PC tracks.
    allowed_cs_by_geneset: dict[str, set[tuple[str, str, str]]] = {}
    bim_bps, n_ref_snps = _load_bim_bp_sets(args.bfile_prefix, chroms)
    kinds_needed = sorted({_geneset_and_qtl_suffix(a)[1] for a in phase2})
    for kind in kinds_needed:
        no_testis = any(a.endswith("_noTestis") for a in phase2 if _geneset_and_qtl_suffix(a)[1] == kind)
        if kind == "eQTL":
            paths = _filter_susie_paths(eqtl_paths_all, exclude_testis=no_testis)
            gene_col, from_pheno = "phenotype_id", True
        else:
            paths = _filter_susie_paths(sqtl_paths_all, exclude_testis=no_testis)
            gene_col, from_pheno = "gene_id", False
        if not paths:
            raise FileNotFoundError(f"No SuSiE parquet paths available for downsample {kind} annotations")

        match_annot = f"{DOWNSAMPLE_MATCH_GENESET}_{kind}{'_noTestis' if no_testis else ''}"
        bp_cm, n_blocks, target_frac = _001cm_coverage_from_annot_dir(
            out_dir / match_annot, match_annot, chroms
        )
        print(
            f"[annot] downsample 0.01cM-block target ({kind}): {100.0 * target_frac:.4f}% "
            f"(match_annot={match_annot}, source={DOWNSAMPLE_SOURCE_GENESET}, "
            f"n_blocks={n_blocks:,}, n_ref_snps={n_ref_snps:,})",
            flush=True,
        )

        cs_weights = _index_credible_set_weights(
            paths,
            sets[DOWNSAMPLE_SOURCE_GENESET],
            gene_col,
            bim_bps,
            gene_from_phenotype_id=from_pheno,
        )
        picked, achieved_frac, n_genes, n_cs = _downsample_genes_match_001cm(
            cs_weights, bp_cm, n_blocks, target_frac, seed=DOWNSAMPLE_SEED
        )
        allowed_cs_by_geneset[f"{DOWNSAMPLE_GENESET}:{kind}"] = picked
        print(
            f"[annot] downsample genes {DOWNSAMPLE_GENESET} {kind}: "
            f"n_genes={n_genes:,} n_cs={n_cs:,} "
            f"pct_001cm={100.0 * achieved_frac:.4f}% (target={100.0 * target_frac:.4f}%)",
            flush=True,
        )

    _write_annots(phase2, _build_sparse_maps(phase2, allowed_cs_by_geneset=allowed_cs_by_geneset))

    coverage_out = out_dir / "annot_snp_cm_coverage.tsv"
    table = build_coverage_table(out_dir)
    if len(table):
        table.to_csv(coverage_out, sep="\t", index=False)
        print(f"[annot] wrote coverage {coverage_out} ({len(table)} rows)", flush=True)
    else:
        print("[annot] coverage table empty (no paper annots found yet)", flush=True)

    print("[annot] done", flush=True)


if __name__ == "__main__":
    main()