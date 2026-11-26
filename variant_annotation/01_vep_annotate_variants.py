#!/usr/bin/env python3
"""
Create a per-variant 0/1 annotation table using Ensembl VEP, with gene-type-split columns.

Transcript-level consequence columns are split by raw gencode ``gene_type_grouped`` values
(one binary column per category × per grouped type), e.g.:
  {category}_protein_coding_d, {category}_pseudogene_d, ``{category}_lncRNA_d``, ...

The transcript's Ensembl ``gene_id`` is mapped to ``gene_id_stable`` (no version suffix) and
looked up in ``--gencode_gene_types``. **Strict mode:** if any transcript consequence would
set a consequence column but ``gene_id`` is missing or the stable ID is absent from the
TSV, the script raises ``ValueError`` and stops (no VEP-biotype fallback).

Regulatory feature columns (promoter, enhancer, etc.) are NOT split by gene type.
Each regulatory feature gets a single binary column: {feature}_d.

Input variants are expected to be GTEx-style IDs:
  chr1_13550_G_A_b38
  chr10_10177_AC_A_b38

The script will:
  1) Convert variant IDs to a temporary VCF (VCF ID = original variant ID)
  2) Run VEP offline from ``--cache_dir`` (cache annotations + FASTA; no GTF/GFF)
  3) Collapse VEP JSON into per-variant indicator columns split by gene type
  4) Attach ``af_c`` from ``--af_table`` (WGS ``INFO/AF``)

Notes:
  - Always runs VEP ``--offline --cache --regulatory`` (cache must include
    regulation data for enhancer/promoter/… columns, plus reference FASTA).
  - ``--gencode_gene_types`` is **required**: TSV with columns ``gene_id_stable`` and
    ``gene_type_grouped`` (e.g. gencode_v39_v47_gene_classification.txt).
  - ``--af_table`` is required (variant ID + AF). Rebuild from GTEx WGS VCF
    ``INFO/AF``. The public lookup table has no AF; the AF-joined
    ``…MAF01.af.gene_split.txt.gz`` is on Zenodo.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Gencode gene_type_grouped (strict lookup; one consequence column per grouped value)
# ---------------------------------------------------------------------------


def load_gencode_gene_type_grouped(
    path: str,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Load gene_id_stable -> gene_type_grouped (exact string from TSV).

    Returns:
      mapping: stable ID -> gene_type_grouped
      groups: sorted unique gene_type_grouped values (defines consequence column suffixes)

    Raises:
      ValueError: duplicate stable ID with conflicting gene_type_grouped
    """
    mapping: Dict[str, str] = {}
    with open(path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = (row.get("gene_id_stable") or "").strip()
            grp = (row.get("gene_type_grouped") or "").strip()
            if not sid or not grp:
                continue
            if sid in mapping and mapping[sid] != grp:
                raise ValueError(
                    f"Inconsistent gene_type_grouped for {sid!r}: "
                    f"{mapping[sid]!r} vs {grp!r} in {path}"
                )
            mapping[sid] = grp
    groups = sorted(set(mapping.values()))
    return mapping, groups


def build_binary_header(gene_type_grouped_values: Sequence[str]) -> List[str]:
    """Regulatory single columns + consequence columns for each gencode grouped type."""
    return (
        [f"{base}_d" for base in BASE_REGULATORY_COLUMNS]
        + [
            f"{base}_{gt}_d"
            for base in BASE_CONSEQUENCE_COLUMNS
            for gt in gene_type_grouped_values
        ]
    )


def build_target_header(binary_header: Sequence[str]) -> List[str]:
    return ["SNP"] + list(binary_header) + CONTINUOUS_HEADER


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Regulatory feature: column base name -> set of VEP regulatory biotypes to match
BASE_REGULATORY_COLUMNS: Dict[str, Set[str]] = {
    "enhancer": {"enhancer"},
    "promoter": {"promoter"},
    "open_chromatin_region": {"open_chromatin_region"},
    "promoter_flanking_region": {"promoter_flanking_region"},
    "CTCF_binding_site": {"CTCF_binding_site", "ctcf_binding_site"},
    "TF_binding_site": {"TF_binding_site", "tf_binding_site"},
}

# Transcript consequence: column base name -> set of VEP consequence terms to match
BASE_CONSEQUENCE_COLUMNS: Dict[str, Set[str]] = {
    "3_prime_UTR_variant": {"3_prime_UTR_variant"},
    "5_prime_UTR_variant": {"5_prime_UTR_variant"},
    "frameshift_variant": {"frameshift_variant"},
    "intron_variant": {"intron_variant"},
    "missense_variant": {"missense_variant"},
    "non_coding_transcript_exon_variant": {"non_coding_transcript_exon_variant"},
    "splice_acceptor_variant": {"splice_acceptor_variant"},
    "splice_donor_variant": {"splice_donor_variant"},
    "splice_region_variant": {"splice_region_variant"},
    "stop_gained": {"stop_gained"},
    "synonymous_variant": {"synonymous_variant"},
}

CONTINUOUS_HEADER: List[str] = [
    "af_c",
]

# Precomputed lowercase lookup tables for fast matching
_REG_WANTED_LC: Dict[str, Set[str]] = {
    col_base: {w.lower() for w in wanted}
    for col_base, wanted in BASE_REGULATORY_COLUMNS.items()
}
_CONS_WANTED_LC: Dict[str, Set[str]] = {
    col_base: {w.lower() for w in wanted}
    for col_base, wanted in BASE_CONSEQUENCE_COLUMNS.items()
}

_AF_ID_ALIASES = {"snp", "variant_id", "id", "variant"}
_AF_VALUE_ALIASES = {"af_c", "af", "maf", "allele_freq", "info/af"}


# ---------------------------------------------------------------------------
# Variant ID parsing
# ---------------------------------------------------------------------------

VAR_ID_RE = re.compile(
    r"^(?P<chrom>chr[0-9XYM]+|[0-9XYM]+|chrMT|MT)_(?P<pos>[0-9]+)_(?P<ref>[ACGTNacgtn]+)_(?P<alt>[ACGTNacgtn]+)_b38$"
)


def open_maybe_gzip(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def read_variant_ids(path: str, input_format: str) -> List[str]:
    """
    Returns list of variant IDs (strings) in input order.

    input_format:
      - ids: first non-empty token on each line is a variant ID; header lines allowed
      - bim: PLINK .bim; variant ID is column 2
    """
    ids: List[str] = []
    with open_maybe_gzip(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if input_format == "ids":
                if line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                # Skip common headers (plain ID lists or GTEx lookup first column).
                if parts[0].lower() in {"snp", "variant_id", "id", "variant"}:
                    continue
                ids.append(parts[0])
            elif input_format == "bim":
                parts = line.split()
                if len(parts) < 2:
                    continue
                ids.append(parts[1])
            else:
                raise ValueError(f"Unsupported input_format: {input_format}")
    return ids


@dataclass(frozen=True)
class ParsedVariant:
    variant_id: str
    chrom: str
    pos: int
    ref: str
    alt: str


def parse_variant_id(variant_id: str) -> ParsedVariant:
    m = VAR_ID_RE.match(variant_id)
    if not m:
        raise ValueError(
            f"Variant ID does not match expected format chr_pos_ref_alt_b38: {variant_id}"
        )
    chrom = m.group("chrom")
    pos = int(m.group("pos"))
    ref = m.group("ref").upper()
    alt = m.group("alt").upper()
    return ParsedVariant(variant_id=variant_id, chrom=chrom, pos=pos, ref=ref, alt=alt)


# ---------------------------------------------------------------------------
# Gene type map loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# VCF writing
# ---------------------------------------------------------------------------

def write_vcf(variants: Sequence[ParsedVariant], path: str, strip_chr: bool) -> None:
    with open(path, "wt") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write("##source=01_vep_annotate_variants.py\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for v in variants:
            chrom = v.chrom
            if strip_chr and chrom.startswith("chr"):
                chrom = chrom[3:]
            out.write(
                f"{chrom}\t{v.pos}\t{v.variant_id}\t{v.ref}\t{v.alt}\t.\t.\t.\n"
            )


# ---------------------------------------------------------------------------
# VEP command building
# ---------------------------------------------------------------------------

def build_vep_command(
    *,
    input_vcf: str,
    output_json: str,
    cache_dir: str,
) -> List[str]:
    return [
        "vep",
        "--input_file",
        input_vcf,
        "--format",
        "vcf",
        "--output_file",
        output_json,
        "--json",
        "--no_stats",
        "--force_overwrite",
        "--assembly",
        "GRCh38",
        "--species",
        "homo_sapiens",
        "--offline",
        "--regulatory",
        "--cache",
        "--dir_cache",
        cache_dir,
    ]


# ---------------------------------------------------------------------------
# VEP JSON parsing
# ---------------------------------------------------------------------------

def variant_id_from_vep_record(record: dict) -> Optional[str]:
    vid = record.get("id")
    if isinstance(vid, str) and vid:
        return vid
    inp = record.get("input")
    if isinstance(inp, str) and inp:
        return inp
    return None


def init_row(binary_header: Sequence[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {col: 0 for col in binary_header}
    for col in CONTINUOUS_HEADER:
        row[col] = None
    return row


def _stable_id(gene_id: str) -> str:
    """Strip version suffix from Ensembl gene ID: ENSG00000000003.16 -> ENSG00000000003."""
    return gene_id.split(".")[0] if gene_id else ""


def update_row_from_record(
    row: Dict[str, Any],
    record: dict,
    gene_id_to_grouped: Dict[str, str],
) -> None:
    """
    Transcript consequences: stable gene_id -> gene_type_grouped from gencode TSV (strict).
    Raises ValueError if gene_id missing or not in TSV when a consequence column applies.
    """
    vid = variant_id_from_vep_record(record) or "<?unknown_variant?>"

    for tc in record.get("transcript_consequences", []):
        if not isinstance(tc, dict):
            continue

        ct_list = tc.get("consequence_terms")
        if not isinstance(ct_list, list):
            continue
        ct_lower = {t.lower() for t in ct_list if isinstance(t, str)}

        matched_bases = [
            col_base
            for col_base, wanted_lc in _CONS_WANTED_LC.items()
            if ct_lower & wanted_lc
        ]
        if not matched_bases:
            continue

        gene_id = tc.get("gene_id") or ""
        tid = tc.get("transcript_id") or ""
        sid = _stable_id(gene_id)
        if not sid:
            raise ValueError(
                f"{vid}: transcript {tid!r} has consequence matching {matched_bases} "
                f"but no gene_id in VEP output"
            )
        if sid not in gene_id_to_grouped:
            raise ValueError(
                f"{vid}: gene_id {gene_id!r} (stable {sid!r}) not in --gencode_gene_types"
            )
        gt_group = gene_id_to_grouped[sid]

        for col_base in matched_bases:
            col = f"{col_base}_{gt_group}_d"
            if col not in row:
                raise ValueError(
                    f"{vid}: column {col!r} not in row; gene_type_grouped {gt_group!r} "
                    f"not in header group list"
                )
            row[col] = 1

    # --- Pass 2: regulatory feature consequences ---
    # Regulatory features are not split by gene type; set a single binary column per feature.
    for rfc in record.get("regulatory_feature_consequences", []):
        if not isinstance(rfc, dict):
            continue
        biotype = rfc.get("biotype") or ""
        if not biotype:
            continue
        biotype_lc = biotype.lower()

        for col_base, wanted_lc in _REG_WANTED_LC.items():
            if biotype_lc in wanted_lc:
                row[f"{col_base}_d"] = 1


def load_vep_json(
    path: str,
    gene_id_to_grouped: Dict[str, str],
    binary_header: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: variant_id -> row dict of indicator columns.
    If VEP emits multiple records per variant (e.g. multi-allelic), indicators are OR-ed.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            vid = variant_id_from_vep_record(record)
            if not vid:
                continue
            row = rows.get(vid)
            if row is None:
                row = init_row(binary_header)
                rows[vid] = row
            update_row_from_record(row, record, gene_id_to_grouped)
    return rows


def _format_value(col: str, val: Any) -> str:
    if col in CONTINUOUS_HEADER:
        return "NA" if val is None else f"{float(val):.6g}"
    return str(val)


def load_af_table(path: str) -> Dict[str, float]:
    """Load variant_id -> AF from a 2-column TSV (header optional)."""
    af_map: Dict[str, float] = {}
    with open_maybe_gzip(path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty AF table: {path}")
    header = [c.strip().lower() for c in rows[0]]
    start = 0
    id_idx, af_idx = 0, 1
    if len(header) >= 2 and (
        header[0] in _AF_ID_ALIASES or header[1] in _AF_VALUE_ALIASES
    ):
        try:
            id_idx = next(i for i, c in enumerate(header) if c in _AF_ID_ALIASES)
            af_idx = next(i for i, c in enumerate(header) if c in _AF_VALUE_ALIASES)
        except StopIteration as e:
            raise ValueError(
                f"AF table header must include a variant ID column "
                f"({sorted(_AF_ID_ALIASES)}) and an AF column "
                f"({sorted(_AF_VALUE_ALIASES)}); got {rows[0]!r}"
            ) from e
        start = 1
    for row in rows[start:]:
        if len(row) <= max(id_idx, af_idx):
            continue
        vid = row[id_idx].strip()
        if not vid or vid.lower() in _AF_ID_ALIASES:
            continue
        try:
            af_map[vid] = float(row[af_idx])
        except ValueError:
            continue
    if not af_map:
        raise ValueError(f"No AF values parsed from {path}")
    return af_map


def attach_af(
    rows_by_variant: Dict[str, Dict[str, Any]],
    variant_ids: Sequence[str],
    binary_header: Sequence[str],
    af_map: Dict[str, float],
) -> None:
    """Set ``af_c`` on each row; create empty rows for variants missing from VEP."""
    n_hit = 0
    for vid in variant_ids:
        row = rows_by_variant.get(vid)
        if row is None:
            row = init_row(binary_header)
            rows_by_variant[vid] = row
        if vid in af_map:
            row["af_c"] = af_map[vid]
            n_hit += 1
    if n_hit == 0:
        raise ValueError(
            "No variant IDs from --variants matched --af_table; "
            "check that AF table IDs are GTEx-style chr_pos_ref_alt_b38"
        )
    print(
        f"[vep_annotate] Joined af_c for {n_hit:,}/{len(variant_ids):,} variants",
        file=sys.stderr,
    )


def write_output(
    variant_ids: Sequence[str],
    rows_by_variant: Dict[str, Dict[str, Any]],
    output_path: str,
    target_header: Sequence[str],
    binary_header: Sequence[str],
) -> None:
    with open_maybe_gzip(output_path, "wt") as out:
        out.write("\t".join(target_header) + "\n")
        for vid in variant_ids:
            row = rows_by_variant.get(vid) or init_row(binary_header)
            values = [vid] + [_format_value(col, row[col]) for col in target_header[1:]]
            out.write("\t".join(values) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variants",
        required=True,
        help="Input file of GTEx-style variant IDs (one per line, or gzipped).",
    )
    p.add_argument(
        "--output",
        required=True,
        help=(
            "Output TSV (optionally .gz), e.g. "
            "WGS_Feature_overlap_collapsed_VEP_short_4torus.v11.MAF01.af.gene_split.txt.gz"
        ),
    )
    p.add_argument(
        "--af_table",
        required=True,
        help=(
            "TSV of variant AF (variant ID + AF; header optional); joined as af_c. "
            "Rebuild from GTEx WGS VCF: bcftools query -f '%ID\\t%INFO/AF\\n' WGS.vcf.gz"
        ),
    )
    p.add_argument(
        "--gencode_gene_types",
        required=True,
        help=(
            "Required. TSV with columns gene_id_stable and gene_type_grouped "
            "(e.g. gencode_v39_v47_gene_classification.txt). "
            "Consequence columns are split by each distinct gene_type_grouped value; "
            "genes absent from this file cause a hard error."
        ),
    )
    p.add_argument(
        "--cache_dir",
        required=True,
        help="Local VEP cache directory (--cache + --dir_cache; annotations + FASTA from cache).",
    )
    args = p.parse_args(argv)

    try:
        gene_id_to_grouped, grouped_values = load_gencode_gene_type_grouped(
            args.gencode_gene_types
        )
    except ValueError as e:
        print(f"ERROR loading gencode TSV: {e}", file=sys.stderr)
        return 2

    binary_header = build_binary_header(grouped_values)
    target_header = build_target_header(binary_header)
    print(
        f"[vep_annotate] Loaded {len(gene_id_to_grouped):,} genes; "
        f"{len(grouped_values)} gene_type_grouped labels: {grouped_values} "
        f"({len(binary_header)} binary columns)",
        file=sys.stderr,
    )

    variant_ids = read_variant_ids(args.variants, "ids")
    if not variant_ids:
        print("ERROR: No variant IDs found in input.", file=sys.stderr)
        return 2

    parsed: List[ParsedVariant] = []
    for vid in variant_ids:
        try:
            parsed.append(parse_variant_id(vid))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory(prefix="vep_variant_annot_") as tmp_root:
        input_vcf = os.path.join(tmp_root, "variants.vcf")
        output_json = os.path.join(tmp_root, "vep.json")

        write_vcf(parsed, input_vcf, strip_chr=False)

        cmd = build_vep_command(
            input_vcf=input_vcf,
            output_json=output_json,
            cache_dir=args.cache_dir,
        )

        if shutil.which("vep") is None:
            print("ERROR: VEP executable not found on PATH: vep", file=sys.stderr)
            return 127

        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            print(f"ERROR: VEP failed with exit code {proc.returncode}", file=sys.stderr)
            print(f"VEP JSON (if any) at: {output_json}", file=sys.stderr)
            return proc.returncode

        try:
            rows_by_variant = load_vep_json(
                output_json, gene_id_to_grouped, binary_header
            )
        except ValueError as e:
            print(f"ERROR parsing VEP output: {e}", file=sys.stderr)
            return 1
        try:
            af_map = load_af_table(args.af_table)
            attach_af(rows_by_variant, variant_ids, binary_header, af_map)
        except ValueError as e:
            print(f"ERROR loading/joining AF table: {e}", file=sys.stderr)
            return 1
        write_output(
            variant_ids,
            rows_by_variant,
            args.output,
            target_header,
            binary_header,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
