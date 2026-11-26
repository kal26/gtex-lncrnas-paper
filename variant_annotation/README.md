# variant_annotation/ — Figure 3e; Supplementary Figures 9 & 15h

Annotates variants with VEP and estimates Firth enrichment of fine-mapped QTL
and coloc high-VCP variants in functional classes.

## Inputs

| Input                                         | Where to get it                                                                                                                                                                                                                                           |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GTEx WGS variants                             | GTEx portal lookup table (`GTEx_Analysis_2021-02-11_v11_WholeGenomeSeq_953Indiv.lookup_table.txt.gz`: https://storage.googleapis.com/adult-gtex/references/v11/reference-tables/GTEx_Analysis_2021-02-11_v11_WholeGenomeSeq_953Indiv.lookup_table.txt.gz) |
| Variant AF table                              | GTEx WGS VCF (protected access, accession: phs000424.v11.p1) (https://gtexportal.org/home/protectedDataAccess)                                                                                                                                            |
| VEP cache (incl. reference FASTA + regulation) | Ensembl VEP v113.3 cache via `vep_install` (https://www.ensembl.org/info/docs/tools/vep/index.html)                                                                                                                                                        |
| Gene cis-window positions                     | Zenodo (https://zenodo.org/records/24550349): `variant_annotation/gene_positions.txt` (4-col `#chr/start/end/gene_id`; no rebuild script in this repo)                                                                                                                             |
| SuSiE summaries                               | GTEx portal (`GTEx_Analysis_v11_eQTL_SuSiE.tar`, `GTEx_Analysis_v11_sQTL_SuSiE.tar`: https://gtexportal.org/home/downloads/adult-gtex/qtl/)                                                                                                               |
| Custom GENCODE v39 vs v47 gene classification | Zenodo (https://zenodo.org/records/24550349): `gene_annotation/gencode_v39_v47_gene_classification.txt`; or rebuild with `annotation/01_compare_gencode.py`                                                                                                                              |
| Custom lncRNA category table                  | Zenodo (https://zenodo.org/records/24550349): `gene_annotation/lncrna_category.txt`; or rebuild with `annotation/03_classify_lncrna.py`                                                                                                                                                  |
| Main ColocBoost + loci                        | Zenodo (https://zenodo.org/records/24550349): `coloc/main/{Tissue}.ColocBoost.parquet` + `coloc/main/cross_tissue_loci.parquet`; or rebuild with `coloc/01` + `coloc/02`                                                                                                                      |

## Analysis pipeline

To activate a Pixi environment with the required packages, run `pixi run -e variant-annotation …`.

### `01_vep_annotate_variants.py`

Annotate variants with VEP; join AF from WGS.

```bash
python 01_vep_annotate_variants.py \
  --variants /path/to/lookup_table.txt.gz \
  --af_table /path/to/variant_af.tsv \
  --output /path/to/WGS_Feature_overlap_collapsed_VEP_short_4torus.v11.MAF01.af.gene_split.txt.gz \
  --gencode_gene_types /path/to/gencode_v39_v47_gene_classification.txt \
  --cache_dir /path/to/vep_cache
```

**Output:** `WGS_Feature_overlap_collapsed_VEP_short_4torus.v11.MAF01.af.gene_split.txt.gz`

### `02_combine_odds_annotations.py`

Join VEP with max-PIP / cis-loci columns.

```bash
python 02_combine_odds_annotations.py \
  --vep_annotations /path/to/WGS_Feature_overlap_collapsed_VEP_short_4torus.v11.MAF01.af.gene_split.txt.gz \
  --gene_loci_pos /path/to/gene_positions.txt \
  --esusie_dir /path/to/v11_eqtl_susie \
  --ssusie_dir /path/to/v11_sqtl_susie \
  --gencode_class /path/to/gencode_v39_v47_gene_classification.txt \
  --lncrna_category_file /path/to/lncrna_category.txt \
  --output_path /path/to/odds_annotations.parquet
```

**Output:** `odds_annotations.parquet`

### `03_run_vep_qtl_enrichment.py`

Firth enrichment of SuSiE fine-mapped QTL variants.

```bash
python 03_run_vep_qtl_enrichment.py \
  --combined_annotations /path/to/odds_annotations.parquet \
  --output_dir /path/to/vep_odds \
  --esusie_dir /path/to/v11_eqtl_susie \
  --ssusie_dir /path/to/v11_sqtl_susie \
  --gencode_class /path/to/gencode_v39_v47_gene_classification.txt \
  --lncrna_category_file /path/to/lncrna_category.txt
```

**Output:** `qtl_vep_odds.txt`

### `04_run_coloc_vcp_enrichment.py`

Firth enrichment of ColocBoost CoS variants.

```bash
python 04_run_coloc_vcp_enrichment.py \
  --combined_annotations /path/to/odds_annotations.parquet \
  --coloc_dir /path/to/coloc/main \
  --gencode_class /path/to/gencode_v39_v47_gene_classification.txt \
  --output_path /path/to/coloc_vep_odds.txt
```

**Output:** `coloc_vep_odds.txt`

## Outputs provided

Zenodo (https://zenodo.org/records/24550349):

| Description | Path | Generated by |
|---|---|---|
| Collapsed VEP + AF annotations | `variant_annotation/WGS_Feature_overlap_collapsed_VEP_short_4torus.v11.MAF01.af.gene_split.txt.gz` | `01_vep_annotate_variants.py` |
| Odds for SuSiE fine-mapped QTL variants | `variant_annotation/qtl_vep_odds.txt` | `03_run_vep_qtl_enrichment.py` |
| Odds for ColocBoost colocalized CoS variants | `variant_annotation/coloc_vep_odds.txt` | `04_run_coloc_vcp_enrichment.py` |

## Notebooks

| Notebook | Notebook input file(s) | Generated by |
|---|---|---|
| `figure_3` (panel e) | `qtl_vep_odds.txt` | `03_run_vep_qtl_enrichment.py` |
| `supp_figure_9` | `qtl_vep_odds.txt` | `03_run_vep_qtl_enrichment.py` |
| `supp_figure_15` (panel h) | `coloc_vep_odds.txt` | `04_run_coloc_vcp_enrichment.py` |
