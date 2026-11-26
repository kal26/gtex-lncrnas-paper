# ldsc/ — Figure 4a–b; Supplementary Figure 11

Builds MaxPIP QTL annotations and runs stratified LD score regression for
heritability enrichment and τ*.

## Inputs

| Input                                         | Where to get it                                                                                                                                                                                                                                                                   |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1000G EUR PLINK bfiles + `.frq` (GRCh38)      | Zenodo (`GRCh38.tgz` → `1000G.EUR.hg38.{chr}.{bed,bim,fam,frq}`: https://zenodo.org/records/10515792)                                                                                                                                                                             |
| HapMap3 LDSC weights (GRCh38)                 | Zenodo (`GRCh38.tgz` → `weights.hm3_noMHC.{chr}.*`: https://zenodo.org/records/10515792)                                                                                                                                                                                          |
| HapMap3 SNP list                              | Zenodo (`hm3_no_MHC.list.txt`: https://zenodo.org/records/10515792)                                                                                                                                                                                                               |
| Baseline-LD panels                            | Public LDSC GRCh38 release (https://zenodo.org/records/10515792): full baseline-LD v2.2 and per-track files. Paper enrichment uses baseline-LD v2.2 **without** the GTEx eQTL MaxCPP track (`noGTExEQTL`); QTL-alone τ* uses the `base` track only (`baseOnly`); attenuation uses one baseline track at a time. Shared SNP list on Zenodo: `ldsc/baselines/baselineLD.snplist` (`--baseline_snplist`). Pass your local baselines folder as `--baseline_dir`. |
| Barbeira GWAS sumstats                        | Zenodo (`harmonized_imputed_gwas.tar` → `imputed_<Trait>.txt.gz`: https://doi.org/10.5281/zenodo.3629742)                                                                                                                                                                         |
| SuSiE summaries                               | GTEx portal (`GTEx_Analysis_v11_eQTL_SuSiE.tar`, `GTEx_Analysis_v11_sQTL_SuSiE.tar`: https://gtexportal.org/home/downloads/adult-gtex/qtl/)                                                                                                                                       |
| Custom GENCODE v39 vs v47 gene classification | Zenodo (https://zenodo.org/records/24550349): `gene_annotation/gencode_v39_v47_gene_classification.txt`; or rebuild with `annotation/01_compare_gencode.py`                                                                                                                                                             |
| GWAS trait manifest                           | `refs/gwas_ldsc_manifest.tsv` in this folder                                                                                                                                                                                                                                      |
| LDSC model manifest                           | `refs/ldsc_models.tsv` in this folder                                                                                                                                                                                                                                             |

## Analysis pipeline

To activate a Pixi environment with the required packages, use one of the two
envs below (bioconda LDSC is still Python 2 / `pandas<0.21`, so it stays
isolated). Activate **one** env per job — do not chain them in the same shell
unless you re-run `shell-hook` successfully (`python -V` should be 3.12 for
`01`/`05`, 2.7 for `02`–`04`):

```bash
# Steps 01 and 05
source <(pixi shell-hook --environment ldsc-prep --manifest-path pixi.toml)

# Steps 02–04 (ldsc.py / munge_sumstats.py)
source <(pixi shell-hook --environment ldsc --manifest-path pixi.toml)
```

### `00_prepare_baselines.py`

Subset public baseline-LD v2.2 ([zenodo 10515792](https://zenodo.org/records/10515792)) into the `baseline_dir` layout `04_run_ldsc.sh` expects:

- `baselineLD_v2.2_noGTExEQTL/` — full baseline minus `GTEx_eQTL_MaxCPP`
- `baselineLD_v2.2_baseOnly/` — `base` track only
- `single_tracks/<track>/` — optional one-at-a-time attenuation tracks

Env: `ldsc-prep`.

```bash
python 00_prepare_baselines.py \
  --public_baseline_prefix /path/to/baselineLD. \
  --baseline_dir /path/to/baselines \
  --tracks Coding_UCSC,H3K27ac_Hnisz   # optional
```

`--tracks` is a comma-separated list of baseline annotation names (public `.annot.gz` header / `baseline_track` in `refs/ldsc_models.tsv`).

### `01_build_annotations.py`

Build MaxPIP annotation (+ coverage). Env: `ldsc-prep`.

```bash
python 01_build_annotations.py \
  --output_dir /path/to/ldscores \
  --bfile_prefix /path/to/1000G.EUR.hg38. \
  --gencode_gene_classification_tsv /path/to/gencode_v39_v47_gene_classification.txt \
  --eqtl_susie_dir /path/to/v11_eqtl_susie \
  --sqtl_susie_dir /path/to/v11_sqtl_susie
```

**Output:** `{annot}/{annot}.{chr}.annot.gz`; `annot_snp_cm_coverage.tsv` under `--output_dir`

### `02_compute_ldscores.sh`

Compute LD scores for one annotation. Env: `ldsc`.

```bash
bash 02_compute_ldscores.sh \
  --annotation_name protein_coding_eQTL_noTestis \
  --ldscores_dir /path/to/ldscores \
  --bfile_prefix /path/to/1000G.EUR.hg38. \
  --baseline_snplist /path/to/ldsc/baselines/baselineLD.snplist
```

**Output:** `{annot}.{chr}.l2.ldscore.gz`, `.M`, `.M_5_50`

### `03_munge_gwas.sh`

Munge one GWAS trait to LDSC sumstats. Env: `ldsc`.

```bash
bash 03_munge_gwas.sh \
  --trait_id TRAIT \
  --gwas_manifest refs/gwas_ldsc_manifest.tsv \
  --gwas_dir /path/to/imputed_gwas \
  --output_dir /path/to/ldsc_run
```

**Output:** `munged/{TRAIT}.sumstats.gz`

### `04_run_ldsc.sh`

Run one stratified LDSC job (trait × model). Env: `ldsc`.

```bash
bash 04_run_ldsc.sh \
  --models_manifest refs/ldsc_models.tsv \
  --trait_id TRAIT \
  --model_id model_id \
  --output_dir /path/to/ldsc_run \
  --munged_dir /path/to/ldsc_run/munged \
  --baseline_dir /path/to/ldsc/baselines \
  --ldscores_dir /path/to/ldscores \
  --weights_ld_prefix /path/to/weights.hm3_noMHC. \
  --frqfile_prefix /path/to/1000G.EUR.hg38. \
  --hm3_snplist /path/to/hm3_no_MHC.list.txt
```

**Output:** `results/{TRAIT}/{TRAIT}.{model}.results`

### `05_parse_results.py`

Aggregate LDSC results into the figure summary table. Env: `ldsc-prep`.

```bash
python 05_parse_results.py \
  --output_dir /path/to/ldsc_run \
  --output /path/to/ldsc_summary.tsv \
  --models_manifest refs/ldsc_models.tsv \
  --ldscores_dir /path/to/ldscores \
  --frqfile_prefix /path/to/1000G.EUR.hg38.
```

**Output:** `ldsc_summary.tsv`

## Outputs provided

Zenodo (https://zenodo.org/records/24550349):

| Description | Path | Generated by |
|---|---|---|
| Annot coverage table | `ldsc/annot_snp_cm_coverage.tsv` | `01_build_annotations.py` |
| QTL `.annot.gz` + LD scores (noTestis + downsampled) | `ldsc/annotations/<annotation>/` | `01_build_annotations.py`; `02_compute_ldscores.sh` |
| Baseline SNP list | `ldsc/baselines/baselineLD.snplist` | extracted from baseline-LD |
| Enrichment / τ* summary | `ldsc/ldsc_summary.tsv` | `05_parse_results.py` |

Baseline-LD panels themselves are **not** on this Zenodo record; build them with `00_prepare_baselines.py` from https://zenodo.org/records/10515792.

## Notebooks

| Notebook | Notebook input file(s) | Generated by |
|---|---|---|
| `figure_4` (a–b) | `ldsc_summary.tsv` | `05_parse_results.py` |
| `supp_figure_11` | `ldsc_summary.tsv`; `annot_snp_cm_coverage.tsv`; `refs/ldsc_models.tsv` | `05_parse_results.py`; `01_build_annotations.py` |
