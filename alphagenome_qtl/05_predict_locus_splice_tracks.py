#!/usr/bin/env python3
"""Step 5 — AlphaGenome REF/ALT splice tracks for chr3:35639573 (Supp 19).

Calls predict_variant and writes SPLICE_JUNCTIONS + SPLICE_SITES (+ optional
SPLICE_SITE_USAGE) in the locus window. Uses the same pinned Kaggle model loader
as ``02_score_cohorts.py`` (``google/alphagenome/jax/all_folds/1`` by default).
"""
from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from util import load_sibling_script

_score = load_sibling_script("02_score_cohorts.py")
load_native_alphagenome_model = _score.load_native_alphagenome_model

FOCUS_ONTOLOGY = {
    "UBERON:0009834",  # Brain_Frontal_Cortex_BA9
    "UBERON:0001876",  # Brain_Amygdala
    "UBERON:0001870",  # Brain_Cortex
}

LOCUS_WIN = (35_638_000, 35_641_000)
# ENSG00000295459 CoS lead (filename / Zenodo stem without build suffix)
DEFAULT_VARIANT_ID = "chr3_35639573_G_A"
BASE_MODEL_VERSION = "all_folds"
KAGGLE_MODEL_VERSION = 1
SEQ_LEN = "1MB"
OUTPUT_TYPES = ["SPLICE_JUNCTIONS", "SPLICE_SITES", "SPLICE_SITE_USAGE"]
# Focus junctions / sites for Supp 19 locus dumps (start, end) or site positions.
FOCUS_JUNCTIONS = (
    ("annotated", 35638953, 35639754),
    ("annotated_ag", 35638953, 35639753),
    ("cryptic", 35639634, 35639754),
    ("cryptic_ag", 35639634, 35639753),
    ("enst227", 35639045, 35639754),
    ("enst227_ag", 35639045, 35639753),
)
FOCUS_SITES = (
    ("annotated_3SS", 35638953),
    ("cryptic_3SS", 35639634),
    ("enst227_3SS", 35639045),
    ("donor_5SS", 35639754),
    ("lead_SNP", 35639573),
)


def variant_table(variant_id: str = DEFAULT_VARIANT_ID) -> pd.DataFrame:
    parts = variant_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"Expected chr_pos_ref_alt[_build] variant id, got {variant_id!r}")
    return pd.DataFrame(
        [
            dict(
                variant_id=variant_id,
                CHROM=parts[0],
                POS=int(parts[1]),
                REF=parts[2],
                ALT=parts[3],
            )
        ]
    )


def _junction_table(jd, allele: str, variant_id: str) -> pd.DataFrame:
    """Flatten JunctionData → tidy rows (one per junction × track)."""
    if jd is None or len(jd) == 0:
        return pd.DataFrame()
    meta = jd.metadata.reset_index(drop=True)
    juncs = jd.junctions
    vals = np.asarray(jd.values, dtype=np.float32)  # (n_junc, n_tracks)
    n_j, n_t = vals.shape
    starts = np.array([int(j.start) for j in juncs], dtype=np.int64)
    ends = np.array([int(j.end) for j in juncs], dtype=np.int64)
    strands = np.array([str(j.strand) for j in juncs])
    chroms = np.array([str(j.chromosome) for j in juncs])

    ji = np.repeat(np.arange(n_j), n_t)
    ti = np.tile(np.arange(n_t), n_j)
    out = pd.DataFrame(
        {
            "variant_id": variant_id,
            "allele": allele,
            "chrom": chroms[ji],
            "junction_start": starts[ji],
            "junction_end": ends[ji],
            "strand": strands[ji],
            "value": vals.reshape(-1),
            "track_name": meta.loc[ti, "name"].to_numpy(),
        }
    )
    if "ontology_curie" in meta.columns:
        out["ontology_curie"] = meta.loc[ti, "ontology_curie"].astype(str).to_numpy()
    if "gtex_tissue" in meta.columns:
        out["gtex_tissue"] = meta.loc[ti, "gtex_tissue"].astype(str).to_numpy()
    return out


def _track_table(td, allele: str, variant_id: str, win: tuple[int, int]) -> pd.DataFrame:
    """Export TrackData values in a genomic window (SPLICE_SITES / USAGE)."""
    if td is None:
        return pd.DataFrame()
    interval = td.interval
    vals = np.asarray(td.values)
    if vals.ndim == 1:
        vals = vals[:, None]
    meta = td.metadata.reset_index(drop=True)
    start = int(interval.start)
    lo, hi = win
    i0 = max(0, lo - start)
    i1 = min(vals.shape[0], hi - start)
    if i1 <= i0:
        return pd.DataFrame()
    chunk = vals[i0:i1, :]  # (L, T)
    L, T = chunk.shape
    pos = np.arange(start + i0, start + i1, dtype=np.int64)
    pi = np.repeat(pos, T)
    ti = np.tile(np.arange(T), L)
    out = pd.DataFrame(
        {
            "variant_id": variant_id,
            "allele": allele,
            "pos": pi,
            "value": chunk.reshape(-1).astype(np.float32),
            "track_name": meta.loc[ti, "name"].to_numpy(),
        }
    )
    if "ontology_curie" in meta.columns:
        out["ontology_curie"] = meta.loc[ti, "ontology_curie"].astype(str).to_numpy()
    if "gtex_tissue" in meta.columns:
        out["gtex_tissue"] = meta.loc[ti, "gtex_tissue"].astype(str).to_numpy()
    if "strand" in meta.columns:
        out["track_strand"] = meta.loc[ti, "strand"].astype(str).to_numpy()
    return out


def _predict_variant(model, *, interval, variant, requested_outputs, ontology_terms, organism):
    """Call predict_variant; prefer transfer_to_host=False when the API supports it."""
    kwargs = dict(
        interval=interval,
        variant=variant,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
        organism=organism,
    )
    params = inspect.signature(model.predict_variant).parameters
    if "transfer_to_host" in params:
        kwargs["transfer_to_host"] = False
    return model.predict_variant(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory for locus splice-track parquets / DONE marker.",
    )
    p.add_argument("--fasta", required=True, help="GRCh38 FASTA used by AlphaGenome.")
    p.add_argument(
        "--gtf-feather",
        required=True,
        help="GENCODE v47 annotation GTF feather.",
    )
    args = p.parse_args()

    import jax
    from alphagenome.data import genome
    from alphagenome.models import dna_client

    ontology_terms = sorted(FOCUS_ONTOLOGY)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    vcf = variant_table(DEFAULT_VARIANT_ID)
    print(f"[INFO] {len(vcf)} variants", flush=True)

    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        print("[WARN] No GPU; falling back to CPU", file=sys.stderr, flush=True)
        device = jax.devices("cpu")[0]
    print(f"[INFO] Device: {device}", flush=True)
    _ = device  # device selection validates GPU availability; loader uses JAX default

    print(
        f"[INFO] Loading pinned model "
        f"google/alphagenome/jax/{BASE_MODEL_VERSION}/{KAGGLE_MODEL_VERSION} ...",
        flush=True,
    )
    model = load_native_alphagenome_model(
        base_model_version=BASE_MODEL_VERSION,
        kaggle_model_version=KAGGLE_MODEL_VERSION,
        reference_fasta=Path(args.fasta),
        gtf_feather_path=str(args.gtf_feather),
    )
    print("[INFO] Model loaded.", flush=True)

    seq_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[f"SEQUENCE_LENGTH_{SEQ_LEN}"]
    requested = [getattr(dna_client.OutputType, name) for name in OUTPUT_TYPES]
    print(
        f"[INFO] outputs={OUTPUT_TYPES} ontology={ontology_terms} seq_len={SEQ_LEN}",
        flush=True,
    )

    t0 = time.time()
    for _, row in vcf.iterrows():
        vid = str(row.variant_id)
        print(f"[PRED] {vid}", flush=True)
        variant = genome.Variant(
            chromosome=str(row.CHROM),
            position=int(row.POS),
            reference_bases=str(row.REF),
            alternate_bases=str(row.ALT),
            name=vid,
        )
        interval = variant.reference_interval.resize(seq_length)
        vo = _predict_variant(
            model,
            interval=interval,
            variant=variant,
            requested_outputs=requested,
            ontology_terms=ontology_terms,
            organism=dna_client.Organism.HOMO_SAPIENS,
        )

        j_frames = []
        for allele, out in ("ref", vo.reference), ("alt", vo.alternate):
            jd = out.splice_junctions
            tab = _junction_table(jd, allele, vid)
            if not tab.empty:
                j_frames.append(tab)
        if j_frames:
            junc = pd.concat(j_frames, ignore_index=True)
            jpath = outdir / f"{vid}.splice_junctions.parquet"
            junc.to_parquet(jpath, index=False)
            print(
                f"  wrote {jpath.name}: {len(junc):,} rows, "
                f"{junc[['junction_start','junction_end']].drop_duplicates().shape[0]:,} unique junctions",
                flush=True,
            )
            lo, hi = LOCUS_WIN
            focus = junc[
                (junc["junction_start"] <= hi) & (junc["junction_end"] >= lo)
            ]
            fpath = outdir / f"{vid}.splice_junctions.locus.parquet"
            focus.to_parquet(fpath, index=False)
            print(f"  wrote {fpath.name}: {len(focus):,} rows in locus window", flush=True)

            for label, a, b in FOCUS_JUNCTIONS:
                hit = focus[
                    (
                        ((focus["junction_start"] - a).abs() <= 2)
                        & ((focus["junction_end"] - b).abs() <= 2)
                    )
                    | (
                        ((focus["junction_start"] - b).abs() <= 2)
                        & ((focus["junction_end"] - a).abs() <= 2)
                    )
                ]
                if len(hit):
                    print(
                        f"  HIT {label}: n={len(hit)} "
                        f"max|alt-ref| preview after merge below",
                        flush=True,
                    )

            piv = (
                focus.groupby(
                    ["junction_start", "junction_end", "strand", "allele", "gtex_tissue"],
                    dropna=False,
                )["value"]
                .mean()
                .unstack("allele")
            )
            if {"ref", "alt"}.issubset(piv.columns):
                piv = piv.assign(
                    delta=piv["alt"] - piv["ref"],
                    abs_delta=(piv["alt"] - piv["ref"]).abs(),
                )
                top = piv.reset_index().sort_values("abs_delta", ascending=False).head(15)
                spath = outdir / f"{vid}.splice_junctions.locus_top_delta.tsv"
                top.to_csv(spath, sep="\t", index=False)
                print(f"  wrote {spath.name} (top |ALT-REF| at locus)", flush=True)
                print(top.head(8).to_string(index=False), flush=True)

        for field, tag in (
            ("splice_sites", "splice_sites"),
            ("splice_site_usage", "splice_site_usage"),
        ):
            frames = []
            for allele, out in ("ref", vo.reference), ("alt", vo.alternate):
                td = getattr(out, field, None)
                tab = _track_table(td, allele, vid, LOCUS_WIN)
                if not tab.empty:
                    frames.append(tab)
            if not frames:
                continue
            tr = pd.concat(frames, ignore_index=True)
            tpath = outdir / f"{vid}.{tag}.locus.txt"
            tr.to_csv(tpath, sep="\t", index=False)
            print(f"  wrote {tpath.name}: {len(tr):,} rows", flush=True)

            for label, pos in FOCUS_SITES:
                near = tr[(tr["pos"] - pos).abs() <= 2]
                if near.empty:
                    continue
                piv = (
                    near.groupby(["pos", "allele", "track_name"], dropna=False)["value"]
                    .mean()
                    .unstack("allele")
                )
                if {"ref", "alt"}.issubset(getattr(piv, "columns", [])):
                    d = (piv["alt"] - piv["ref"]).abs().max()
                    print(f"  {tag} near {label}@{pos}: max|Δ|={float(d):.4g}", flush=True)

    done = outdir / "DONE"
    done.write_text(
        f"finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"elapsed_min={(time.time()-t0)/60:.1f}\n"
    )
    print(f"[DONE] {(time.time()-t0)/60:.1f} min → {outdir}", flush=True)


if __name__ == "__main__":
    main()
