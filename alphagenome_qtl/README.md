# alphagenome_qtl/ — Figure 3f, Supplementary Figures 10 & 19

Scores distance-balanced eQTL cohorts with AlphaGenome, trains causality RFs,
and exports learning curves plus one-locus splice tracks.

## Inputs

| Input                                              | Where to get it                                                                                                                                                                                                       |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GENCODE v39 full GTF                               | GENCODE website (`gencode.v39.annotation.gtf.gz`: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/gencode.v39.annotation.gtf.gz)                                                                 |
| GENCODE v47 full GTF                               | GENCODE website (`gencode.v47.annotation.gtf.gz`: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz)                                                                 |
| AlphaGenome GTF + splice-site feathers (v39 / v47) | Zenodo (https://zenodo.org/records/24550349): `alphagenome_qtl/gencode.v{39,47}.annotation.gtf.feather` + `gencode.v{39,47}.splice_sites_{starts,ends}.feather`; or rebuild with `00_build_gtf_feather.py`                                           |
| Custom GENCODE v39 vs v47 gene classification      | Zenodo (https://zenodo.org/records/24550349): `gene_annotation/gencode_v39_v47_gene_classification.txt`; or rebuild with `annotation/01_compare_gencode.py`                                                                                          |
| 50 QTL tissue IDs                                  | `refs/qtl_tissue_ids.txt` in this folder                                                                                                                                                                               |
| SuSiE lead summaries                               | GTEx portal (`GTEx_Analysis_v11_eQTL_SuSiE.tar`; v10: `GTEx_v10_SuSiE_eQTL.tar`: https://gtexportal.org/home/downloads/adult-gtex/qtl/)                                                                               |
| cis-eQTL allpairs                                  | AnVIL requester-pays bucket (v11: `GTEx_Analysis_v11_eQTL_all_associations`; v10: `GTEx_Analysis_v10_eQTL_all_associations` under `gs://AnVIL_GTEx_public_data/adult-gtex/bulk-qtl/{v10,v11}/single-tissue-cis-qtl/`) |
| GRCh38 FASTA                                       | GENCODE website (`GRCh38.primary_assembly.genome.fa.gz`: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.primary_assembly.genome.fa.gz)                                                   |
| AlphaGenome model weights                          | AlphaGenome (https://github.com/google-deepmind/alphagenome)                                                                                                                                                          |

## Analysis pipeline

To activate a Pixi environment with the required packages, run `pixi run -e alphagenome-qtl …`.

### `00_build_gtf_feather.py`

Convert GENCODE GTF to AlphaGenome feather + splice-site feathers.

```bash
python 00_build_gtf_feather.py --gtf /path/to/gencode.v47.annotation.gtf
```

**Output:** `{gtf}.feather`; `gencode.v{39,47}.splice_sites_{starts,ends}.feather`

### `01_build_cohorts.py`

Build PIP≥0.9 eQTL leads and distance-matched backgrounds. Builds **one** cache type × version per invocation. Requires the `{gtf}.feather` from `00` next to `--full_gtf`. Example (v11 leads):

```bash
python 01_build_cohorts.py \
  --cache-type susie_lead --dataset-version v11 \
  --output_dir /path/to/alphagenome_qtl \
  --full_gtf /path/to/gencode.v47.annotation.gtf \
  --susie-dir /path/to/v11_eqtl_susie \
  --gene-classification /path/to/gencode_v39_v47_gene_classification.txt \
  --tissues $(cat alphagenome_qtl/refs/qtl_tissue_ids.txt)
```

Run all four combinations (swap paths/GTF as needed). `--allpairs-dir` is required only for `allpairs_background`:

| `--cache-type` | `--dataset-version` | GTF / QTL dirs |
|---|---|---|
| `susie_lead` | `v11` | GENCODE v47; v11 SuSiE |
| `susie_lead` | `v10` | GENCODE v39; v10 SuSiE |
| `allpairs_background` | `v11` | GENCODE v47; v11 SuSiE + allpairs (`--allpairs-dir`) |
| `allpairs_background` | `v10` | GENCODE v39; v10 SuSiE + allpairs (`--allpairs-dir`) |

**Output:** `{v10,v11}_susie_lead_metadata.parquet`, `{v10,v11}_allpairs_background_metadata.parquet`

### `02_score_cohorts.py`

Score cohorts with AlphaGenome multimodal predictors.

```bash
python 02_score_cohorts.py \
  --reference-fasta /path/to/GRCh38.fa \
  --gtf-feather-path-v10 /path/to/gencode.v39.annotation.gtf.feather \
  --gtf-feather-path-v11 /path/to/gencode.v47.annotation.gtf.feather \
  --output_dir /path/to/alphagenome_qtl \
  --v10-lead-metadata /path/to/v10_susie_lead_metadata.parquet \
  --v11-lead-metadata /path/to/v11_susie_lead_metadata.parquet \
  --v10-background-metadata /path/to/v10_allpairs_background_metadata.parquet \
  --v11-background-metadata /path/to/v11_allpairs_background_metadata.parquet
```

**Output:** `tidy/v{10,11}/chr1/leads_scores.parquet`, `…/chr1/background_scores.parquet` (per chrom)

### `03_train_overall_rf.py`

Rebuild RF example tables from tidy scores, then train one biotype (or unfiltered) CV run. Example (v11 lncRNA+PC model used for catalog / all_genes eval):

```bash
python 03_train_overall_rf.py \
  --dataset-version v11 \
  --biotype-filter lncRNA_protein_coding \
  --output_dir /path/to/alphagenome_qtl \
  --full_gtf /path/to/gencode.v47.annotation.gtf \
  --lead-metadata /path/to/v11_susie_lead_metadata.parquet \
  --background-metadata /path/to/v11_allpairs_background_metadata.parquet \
  --tidy-root /path/to/alphagenome_qtl/tidy/v11 \
  --gene-classification /path/to/gencode_v39_v47_gene_classification.txt
```

`04` needs these **six** biotype-filtered train dirs (unfiltered `{v}_causality_rf/` is not used by `04`):

| `--dataset-version` | `--biotype-filter` | Why |
|---|---|---|
| `v11` | `protein_coding` | Matched-N train/eval PC; catalog holdout |
| `v11` | `lncRNA` | Matched-N novel/existing lncRNA splits; catalog holdout |
| `v11` | `lncRNA_protein_coding` | Matched-N `all_genes` eval; catalog RF models |
| `v10` | `protein_coding` | Catalog holdout |
| `v10` | `lncRNA` | Catalog holdout |
| `v10` | `lncRNA_protein_coding` | Catalog RF models |

Use GENCODE v39 + v10 metadata/`tidy/v10` for the three v10 runs.

**Output:** `v{10,11}_qtl_variant_features.parquet`, `v{10,11}_allpairs_background.multimodal.scored.parquet`, `v{10,11}_causality_rf*/`

### `04_train_subsample_rf.py`

Train matched-N causality RFs.

```bash
python 04_train_subsample_rf.py \
  --output_dir /path/to/alphagenome_qtl \
  --gene-classification /path/to/gencode_v39_v47_gene_classification.txt
```

**Output:** `qtl_causality_auroc.tsv`

### `05_predict_locus_splice_tracks.py`

Predict REF/ALT splice tracks for one locus (Supp 19).

```bash
python 05_predict_locus_splice_tracks.py \
  --output_dir /path/to/alphagenome_qtl \
  --fasta /path/to/GRCh38.fa \
  --gtf-feather /path/to/gencode.v47.annotation.gtf.feather
```

Optional: set ``KAGGLEHUB_CACHE`` if model weights should download somewhere other than the default kagglehub cache.

**Output:** `chr3_35639573_G_A.splice_sites.locus.txt` (and related splice locus txt/parquets)

## Outputs provided

Zenodo (https://zenodo.org/records/24550349):

| Description                                                                                  | Path                                                                                                                     | Generated by                        |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| GENCODE v39/v47 AlphaGenome feathers                                                         | `alphagenome_qtl/gencode.v{39,47}.annotation.gtf.feather`; `alphagenome_qtl/gencode.v{39,47}.splice_sites_{starts,ends}.feather` | `00_build_gtf_feather.py`           |
| Per-variant AlphaGenome multimodal features for eQTL leads (before RF training)              | `alphagenome_qtl/v{10,11}_qtl_variant_features.parquet`                                                                      | `03_train_overall_rf.py`            |
| Fold-aggregated tissue-weighted auROC summaries (matched-N curves, zero-shot, catalog comparison) | `alphagenome_qtl/qtl_causality_auroc.tsv`                                                                                 | `04_train_subsample_rf.py`          |
| Locus splice tracks                                                                          | `alphagenome_qtl/chr3_35639573_G_A.splice_sites.locus.txt`                                                                   | `05_predict_locus_splice_tracks.py` |

## Notebooks

| Notebook | Notebook input file(s) | Generated by |
|---|---|---|
| `figure_3` (panel f) | `qtl_causality_auroc.tsv` | `04_train_subsample_rf.py` |
| `supp_figure_10` | `v{10,11}_qtl_variant_features.parquet`; `qtl_causality_auroc.tsv` | `03_train_overall_rf.py`; `04_train_subsample_rf.py` |
| `supp_figure_19` | `chr3_35639573_G_A.splice_sites.locus.txt` | `05_predict_locus_splice_tracks.py` |
