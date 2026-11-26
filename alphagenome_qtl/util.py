"""Shared parquet I/O and chromosome-split helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Wide tables (many mm_* columns) can exceed pyarrow's default 100MB thrift metadata
# limit on read. These are READ-only options; pyarrow writers reject them.
_PARQUET_THRIFT_KWARGS = {
    "thrift_string_size_limit": 512 * 1024 * 1024,
    "thrift_container_size_limit": 512 * 1024 * 1024,
}

# Parquet footer alone is 8+ bytes; aborted writes often leave only the PAR1 magic (4 bytes).
_MIN_VALID_PARQUET_BYTES = 64

def is_valid_examples_parquet(path: Path, *, min_bytes: int = _MIN_VALID_PARQUET_BYTES) -> bool:
    """Return True if path is a readable parquet file (not a stub/corrupt write)."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        if path.stat().st_size < min_bytes:
            return False
        read_parquet_schema(path)
        return True
    except Exception:
        return False

def load_examples_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns, **_PARQUET_THRIFT_KWARGS).to_pandas()

def read_parquet_schema(path: Path):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path, **_PARQUET_THRIFT_KWARGS).schema_arrow

def unify_parquet_schemas(paths: list[Path]):
    import pyarrow as pa

    if not paths:
        raise ValueError("No parquet paths to unify")
    schemas = [read_parquet_schema(path) for path in paths]
    if len(schemas) == 1:
        return schemas[0]
    return pa.unify_schemas(schemas, promote_options="permissive")

def align_table_to_schema(table, schema):
    import pyarrow as pa
    import pyarrow.compute as pc

    arrays = []
    for field in schema:
        if field.name in table.column_names:
            col = table.column(field.name)
            if not col.type.equals(field.type):
                col = pc.cast(col, target_type=field.type, safe=False)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)

def open_examples_parquet_writer(path: Path, schema):
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(path, schema)

def write_merged_examples_parquet(part_paths: list[Path], output_path: Path) -> None:
    """Merge per-part derived parquets into one file (union schema over pruned parts only)."""
    import shutil

    import pyarrow.parquet as pq

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    if not part_paths:
        save_examples_parquet(pd.DataFrame(), output_path)
        return
    if len(part_paths) == 1:
        shutil.copy2(part_paths[0], output_path)
        return

    unified = unify_parquet_schemas(part_paths)
    writer = open_examples_parquet_writer(output_path, unified)
    n_parts = len(part_paths)
    for idx, part in enumerate(part_paths, start=1):
        print(f"[merge] part {idx}/{n_parts}: {part.name}", flush=True)
        table = pq.read_table(part, **_PARQUET_THRIFT_KWARGS)
        if not table.schema.equals(unified):
            table = align_table_to_schema(table, unified)
        writer.write_table(table)
    writer.close()
    if not is_valid_examples_parquet(output_path):
        raise RuntimeError(f"Merged parquet is invalid or incomplete: {output_path}")

def save_examples_parquet(df: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)

from dataclasses import dataclass

@dataclass(frozen=True)
class ChromosomeSplit:
    train: frozenset[str]
    valid: frozenset[str]
    test: frozenset[str]

    def assign(self, chromosome: str) -> str | None:
        chrom = normalize_chromosome(chromosome)
        if chrom in self.train:
            return "train"
        if chrom in self.valid:
            return "valid"
        if chrom in self.test:
            return "test"
        return None

def _chr_set(*chromosomes: int | str) -> frozenset[str]:
    out: set[str] = set()
    for chrom in chromosomes:
        if isinstance(chrom, int):
            out.add(f"chr{chrom}")
        else:
            token = str(chrom).strip()
            out.add(token if token.startswith("chr") else f"chr{token}")
    return frozenset(out)

# AlphaGenome paper: supervised train/valid/test chromosomes (used for metadata splits).
PAPER_TEST_CHROMOSOMES = _chr_set(3, 6, 9, 12, 16, 18, 19, 21)
PAPER_SUPERVISED_TRAIN_CHROMOSOMES = _chr_set(1, 4, 7, 8, 10, 13, 15)
PAPER_SUPERVISED_VALID_CHROMOSOMES = _chr_set(2, 5, 11, 14, 17, 20, 22, "X")

SUPERVISED_CHR_SPLIT = ChromosomeSplit(
    train=PAPER_SUPERVISED_TRAIN_CHROMOSOMES,
    valid=PAPER_SUPERVISED_VALID_CHROMOSOMES,
    test=PAPER_TEST_CHROMOSOMES,
)

DEFAULT_CHR_SPLIT = SUPERVISED_CHR_SPLIT

# Paper Method 6: 5-fold chromosome CV over ALL chromosomes (no held-out global
# test set). All autosomes + chrX are partitioned into 5 folds; each fold trains
# on the chromosomes not in the fold and is evaluated on the fold. Chromosomes are
# interleaved by index so large and small chromosomes are mixed across folds.
CAUSALITY_5_FOLD_CHROMOSOMES: tuple[frozenset[str], ...] = (
    _chr_set(1, 6, 11, 16, 21),
    _chr_set(2, 7, 12, 17, 22),
    _chr_set(3, 8, 13, 18, "X"),
    _chr_set(4, 9, 14, 19),
    _chr_set(5, 10, 15, 20),
)

def all_causality_cv_chromosomes() -> frozenset[str]:
    chromosomes: set[str] = set()
    for fold in CAUSALITY_5_FOLD_CHROMOSOMES:
        chromosomes.update(fold)
    return frozenset(chromosomes)

def normalize_chromosome(chromosome: str) -> str:
    chrom = str(chromosome).strip()
    if chrom.startswith("chr"):
        return chrom
    return f"chr{chrom}"

# --- variant ID helpers (SNV filter) ---

def _to_genome_variant(variant_id: str):
    # Lazy: RF / cohort CLIs should start without the AlphaGenome package installed.
    from alphagenome.data.genome import Variant, VariantFormat

    return Variant.from_str(str(variant_id), VariantFormat.GTEX)

def is_snv_variant_id(variant_id: str) -> bool:
    """Return True for single-nucleotide variants (paper cohort: SNVs only)."""
    try:
        return _to_genome_variant(variant_id).is_snv
    except ValueError:
        return False

def filter_snv_examples(examples: pd.DataFrame) -> pd.DataFrame:
    """Keep SNV rows only (paper cohort: single-nucleotide variants)."""
    if examples.empty:
        return examples.copy()
    if "variant_id" not in examples.columns:
        raise ValueError("SNV filter requires variant_id column.")
    return examples.loc[examples["variant_id"].map(is_snv_variant_id)].reset_index(drop=True)

def multimodal_feature_columns(df: pd.DataFrame) -> list[str]:
    """Names of AlphaGenome multimodal feature columns (``mm_*``)."""
    return sorted(col for col in df.columns if col.startswith("mm_"))

def strip_gene_version(gene_key: str) -> str:
    return str(gene_key).split(".")[0]

def add_score_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stable score_key = variant|gene|window."""
    from alphagenome.data import genome

    required = {"variant_id", "gene_key", "window_start", "window_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Examples missing columns for score_key: {sorted(missing)}")

    out = df.copy()
    out["gene_key_stable"] = out["gene_key"].astype(str).map(strip_gene_version)
    out["score_key"] = (
        out["variant_id"].astype(str)
        + "|"
        + out["gene_key_stable"].astype(str)
        + "|"
        + out["window_start"].astype(int).astype(str)
        + "-"
        + out["window_end"].astype(int).astype(str)
    )
    if "chromosome" not in out.columns:
        out["chromosome"] = out["variant_id"].map(
            lambda v: genome.Variant.from_str(v, genome.VariantFormat.GTEX).chromosome
        )
    out["chromosome"] = out["chromosome"].map(normalize_chromosome)
    return out

def load_sibling_script(filename: str):
    """Load another alphagenome_qtl/*.py by filename (supports numbered entrypoints)."""
    import importlib.util
    import re
    import sys

    path = Path(__file__).resolve().parent / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    mod_name = "alphagenome_step_" + re.sub(r"[^0-9A-Za-z_]", "_", path.stem)
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

