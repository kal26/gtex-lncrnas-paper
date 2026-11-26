# coloc/ — Figures 4c–e & 5; Supplementary Figures 13–18

Runs GWAS × QTL colocalisation with ColocBoost (per tissue × LD block), including
restricted phenotype discovery and ablation.

## Inputs

| Input                                                     | Where to get it                                                                                                                                                |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GENCODE v47 collapsed GTF                                 | GTEx portal (https://storage.googleapis.com/adult-gtex/references/v11/reference-tables/gencode.v47.genes.gtf)                                                  |
| Custom GENCODE v39 vs v47 gene classification             | Zenodo (https://zenodo.org/records/24550349): `gene_annotation/gencode_v39_v47_gene_classification.txt`; or rebuild with `annotation/01_compare_gencode.py`                                   |
| Coloc job lists (which tissue × LD blocks / pairs to run) | Under `refs/` in this folder (see table below). Pass each dir as `--coloc_items_dir*`.                                                                        |
| eQTL phenotype + covariate matrices                       | GTEx portal (`GTEx_Analysis_v11_eQTL_expression_matrices.tar`, `GTEx_Analysis_v11_eQTL_covariates.tar`: https://gtexportal.org/home/downloads/adult-gtex/qtl/) |
| sQTL phenotype + covariate matrices                       | GTEx portal (`GTEx_Analysis_v11_sQTL_phenotype_matrices.tar`, `GTEx_Analysis_v11_sQTL_covariates.tar`: https://gtexportal.org/home/downloads/adult-gtex/qtl/)  |
| GTEx WGS PLINK stem                                       | Protected access data (accession: phs000424.v11.p1) (https://gtexportal.org/home/protectedDataAccess)                                                          |
| GTEx subject phenotypes (EUR sample list)                 | GTEx portal (`GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt`; filter `RACE==3`)                                                                         |
| EUR LD-block boundaries                                   | LDblocks_GRCh38 (`pyrho_EUR_LD_blocks.bed`)                                                                                                                    |
| hg38 chrom.sizes                                          | UCSC (`hg38.chrom.sizes`)                                                                                                                                      |
| Barbeira imputed GWAS sumstats                            | Zenodo (`harmonized_imputed_gwas.tar` → `imputed_<Trait>.txt.gz`: https://doi.org/10.5281/zenodo.3629742)                                                      |
| Barbeira GWAS metadata                                    | Zenodo (`gwas_metadata.txt`: https://doi.org/10.5281/zenodo.3629742)                                                                                           |
| GWAS trait id list                                        | `refs/gwas_ids.txt` in this folder                                                                                                                             |
| pecotmr GWAS sumstats column map                          | `refs/gwas_column_matching.yml` in this folder                                                                                                                 |

## Analysis pipeline

Analyses used `r-pecotmr` 0.5.0 and `r-colocboost` 1.0.4 (pins in `pixi.toml`).
Those builds are no longer on the `dnachun` channel, so the `coloc` Pixi env is
not enabled. Paper outputs are on Zenodo; the Python drivers need pandas/pyarrow
(`pixi run -e notebooks …` works) and an R install with pecotmr + ColocBoost.

Shared path flags (abbreviated `…coloc_paths…` below):  
`--ld_meta /path/to/ld_blocks/ld_metadata.tsv --genotype_stem /path/to/GTEx_WGS_stem --expression_pheno_dir /path/to/eqtl_pheno --splicing_pheno_dir /path/to/sqtl_pheno --expression_covar_dir /path/to/eqtl_cov --splicing_covar_dir /path/to/sqtl_cov --gwas_id_list /path/to/coloc/refs/gwas_ids.txt --gwas_dir /path/to/imputed_gwas --gwas_meta /path/to/gwas_metadata.txt --gwas_column_matching /path/to/coloc/refs/gwas_column_matching.yml --gencode_classification /path/to/gencode_v39_v47_gene_classification.txt --collapsed_gtf /path/to/gencode.v47.genes.gtf`

| Folder | Used by | What’s inside |
|---|---|---|
| `refs/main/` | `01` | `items.txt` — all tissues × LD blocks |
| `refs/wb_joint/` | `04` joint | Whole Blood × LD blocks |
| `refs/wb_pc/` | `04` protein-coding | WB regions (PC filter at runtime) |

### `00_build_ld_meta.py`

EUR LD panels from WGS × LD-block BED.

### `01_run_main_coloc.py`

```bash
python 01_run_main_coloc.py \
  --output_dir /path/to/coloc_main \
  --coloc_items_dir refs/main \
  …coloc_paths…
```

**Output:** `{Tissue}.ColocBoost.parquet` (one row per CoS; `cos_id` = `Tissue_N`)

### `02_cluster_cross_tissue.py`

```bash
python 02_cluster_cross_tissue.py \
  --coloc_dir /path/to/coloc_main \
  --output /path/to/coloc_main/cross_tissue_loci.parquet \
  --ld_meta /path/to/ld_metadata.tsv
```

**Output:** `cross_tissue_loci.parquet` (`locus_id`, `member_cos_ids`, union genes/GWAS)

### `03_run_ablation.py`

```bash
python 03_run_ablation.py \
  --output_dir /path/to/recovery \
  --coloc_dir /path/to/main \
  …coloc_paths…
```

**Output:** `{only_protein_coding,only_v39}/{Tissue}.{mode}.ColocBoost.parquet` and `recovery_by_cos.parquet`

### `04_run_wb_discovery.py`

```bash
python 04_run_wb_discovery.py \
  --output_dir /path/to/discovery \
  --coloc_items_dir_joint refs/wb_joint \
  --coloc_items_dir_protein_coding refs/wb_pc \
  --coloc_items_dir_single_phenotype /path/to/wb_single_phenotype \
  …coloc_paths…
```

**Output:** `Whole_Blood.{joint,protein_coding,single_phenotype}.ColocBoost.parquet` (`cos_id` = `{mode}_N`)

### `05_cluster_cross_model.py`

```bash
python 05_cluster_cross_model.py \
  --discovery_dir /path/to/discovery \
  --output /path/to/discovery/cross_phenotype_set_loci.parquet \
  --ld_meta /path/to/ld_metadata.tsv
```

**Output:** `cross_phenotype_set_loci.parquet`

## Output schema

### Tissue / mode ColocBoost

| Column | Notes |
|---|---|
| `cos_id` | `Tissue_N` (main); `{mode}_N` (discovery); `Tissue.{mode}_N` (recovery) |
| `variants`, `vcp` | Aligned lists (VCP-descending); lead = `variants[argmax(vcp)]` |
| `cos_npc`, `npc_outcomes`, `phenotype_ids` | ColocBoost NPC fields |
| `cos_max_pval_nominal_phenotypes` | CoS-max −log10(p) per phenotype |
| `gene_ids_robust`, `gene_types_robust` | Robust assemble |
| `gwas_traits`, `gwas_gws_traits` | GWAS membership |
| `gene_types_relaxed`, `gene_ids_relaxed` | Matched non-robust assemble |

### Loci

`locus_id`, `member_cos_ids`, `gene_ids_robust`, `gene_types_robust`, `gwas_traits`, `gwas_gws_traits`

### Recovery

`recovery_by_cos.parquet`: `cos_id` plus `pc_only_*` / `v39_*` columns (`phenotypes`, `recovery`, `cos_npc`, `npc_outcomes`); missing = NA.

## Zenodo layout

| Path | Source |
|---|---|
| `coloc/main/{Tissue}.ColocBoost.parquet` | `01` |
| `coloc/main/cross_tissue_loci.parquet` | `02` |
| `coloc/discovery/Whole_Blood.{mode}.ColocBoost.parquet` | `04` |
| `coloc/discovery/cross_phenotype_set_loci.parquet` | `05` |
| `coloc/recovery/{mode}/{Tissue}.{mode}.ColocBoost.parquet` | `03` |
| `coloc/recovery/recovery_by_cos.parquet` | `03` |

## Notebooks

| Notebook | Inputs |
|---|---|
| `figure_4` | `main/cross_tissue_loci.parquet` (+ tissue ColocBoost joins) |
| `figure_5` | `main/{Tissue}.ColocBoost.parquet` + portal nominals |
| `supp_figure_14`–`16` | loci + tissue ColocBoost |
| `supp_figure_17` | `discovery/*` |
| `supp_figure_18` | `main` + `recovery` + `discovery` |
