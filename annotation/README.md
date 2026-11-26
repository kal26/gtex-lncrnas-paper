# annotation/ — Figures 1–5; Supplementary Figures 7 & 12 (and shared inputs)

Builds GENCODE v39/v47 gene classification and `lncrna_category.txt` (constraint,
ORF, E2G, and cross-species expression flags).

## Inputs

| Input                                                           | Where to get it                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GENCODE v39 full GTF                                            | GENCODE website (`gencode.v39.annotation.gtf.gz`: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/gencode.v39.annotation.gtf.gz)                                                                                                                                                                                                                                   |
| GENCODE v39 collapsed GTF                                       | GTEx portal (https://storage.googleapis.com/adult-gtex/references/v10/reference-tables/gencode.v39.GRCh38.genes.gtf)                                                                                                                                                                                                                                                                    |
| GENCODE v47 full GTF                                            | GENCODE website (`gencode.v47.annotation.gtf.gz`: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz)                                                                                                                                                                                                                                   |
| GENCODE v47 collapsed GTF                                       | GTEx portal (https://storage.googleapis.com/adult-gtex/references/v11/reference-tables/gencode.v47.genes.gtf)                                                                                                                                                                                                                                                                           |
| Zoonomia 43-primate phastCons bigWig                            | Zoonomia track hub (`phyloPPrimates.bigWig`: https://cgl.gi.ucsc.edu/data/cactus/zoonomia-2021-track-hub/hg38/phyloPPrimates.bigWig; hub HTML describes this file as 43-primate phastCons). Local copy used in scoring: `zoonomia_primates.phastCons.bw`.                                                                                                                              |
| CLS riboSeqMin ORF table                                        | Perteghella et al. (`riboSeqMin_0.6_v47ndDecoy.tsv`: https://doi.org/10.1101/2024.10.29.620654)                                                                                                                                                                                                                                                                                         |
| ENCODE-rE2G prediction BEDs                                     | Gschwind et al. *Nature* 657:179–190 (2026); thresholded element–gene links from the ENCODE Portal (https://www.encodeproject.org/) for the 12 GTEx-matched tissues (accessions ENCFF194CBN, ENCFF162ADG, ENCFF844AWZ, ENCFF194QZX, ENCFF695YYV, ENCFF216HEG, ENCFF941LWH, ENCFF179LQB, ENCFF808GUC, ENCFF213ARV, ENCFF438NVX, ENCFF557BXA; also listed in `refs/e2g_tissue_bed.tsv`) |
| ENCODE-rE2G tissue×bed table                                    | `refs/e2g_tissue_bed.tsv` in this folder (12 GTEx-matched tissues × bed basename / accession)                                                                                                                                                                                                                                                                                         |
| Gene-vs-null expression tables (GTEx, dGTEx, macaque, marmoset) | Zenodo (https://zenodo.org/records/24550349): `expression/`; or rebuild with `intergenic_null/02_call_null_expression.py`                                                                                                                                                                                                                                                                                         |
| GTEx gene median TPM                                            | GTEx portal (`GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz`: https://storage.googleapis.com/adult-gtex/bulk-gex/v11/rna-seq/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz)                                                                                                                                                                     |
| GTEx sample attributes                                          | GTEx portal (`GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt`: https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt)                                                                                                                                                                                     |
| GTEx tissue name map                                            | `refs/tissue_name.txt` in this folder                                                                                                                                                                                                                                                                                                                                                 |
| Human–NHP lifted lncRNA orthologs                               | GTEx portal NHP-dGTEx (https://gtexportal.org/home/downloads/nhp-dgtex)                                                                                                                                                                                                                                                                                                                 |
| Uberon GTEx↔NHP tissue match table                              | `refs/uberon_id_matches.txt` in this folder                                                                                                                                                                                                                                                                                                                                           |
| NHP tissue-site median expression tables                        | GTEx portal NHP-dGTEx (https://gtexportal.org/home/downloads/nhp-dgtex)                                                                                                                                                                                                                                                                                                                 |

## Analysis pipeline

To activate a Pixi environment with the required packages, run `pixi run -e annotation …`.

### `01_compare_gencode.py`

Classify genes as new vs existing (v39 vs v47).

```bash
python 01_compare_gencode.py \
  --full-v39 /path/to/gencode.v39.annotation.gtf \
  --full-v47 /path/to/gencode.v47.annotation.gtf \
  --collapsed-v39 /path/to/gencode.v39.genes.gtf \
  --collapsed-v47 /path/to/gencode.v47.genes.gtf \
  --output-classification /path/to/gencode_v39_v47_gene_classification.txt
```

**Output:** `gencode_v39_v47_gene_classification.txt`

### `02_compute_gene_constraint_fractions.py`

Score genes for primate unique-exon / unique-SJ constraint.

```bash
python 02_compute_gene_constraint_fractions.py \
  --full_gtf /path/to/gencode.v47.annotation.gtf \
  --primate-bw /path/to/zoonomia_primates.phastCons.bw \
  --gencode-classification /path/to/gencode_v39_v47_gene_classification.txt \
  --output_dir /path/to/constraint
```

**Output:** `gene_constraint_fractions.tsv.gz`

### `03_classify_lncrna.py`

Classify lncRNAs from input annotations.

```bash
python 03_classify_lncrna.py \
  --gencode-v47 /path/to/gencode.v47.annotation.gtf \
  --gencode-classification /path/to/gencode_v39_v47_gene_classification.txt \
  --riboseq-orf /path/to/riboSeqMin_0.6_v47ndDecoy.tsv \
  --gtex-null /path/to/GTEx.null_quantifications.txt \
  --dgtex-null /path/to/dGTEx.null_quantifications.txt \
  --macaque-null /path/to/macaque.null_quantifications.txt \
  --marmoset-null /path/to/marmoset.null_quantifications.txt \
  --gtex-by-tissue-dir /path/to/by_tissue/GTEx \
  --dgtex-by-tissue-dir /path/to/by_tissue/dGTEx \
  --macaque-by-tissue-dir /path/to/by_tissue/macaque \
  --marmoset-by-tissue-dir /path/to/by_tissue/marmoset \
  --orthologs /path/to/dGTEx_nhpGTEx_liftedlncRNAs.tsv \
  --uberon-match refs/uberon_id_matches.txt \
  --gtex-sample-meta /path/to/SampleAttributesDS.txt \
  --tissue-names refs/tissue_name.txt \
  --gtex-median-tpm /path/to/gene_median_tpm.gct.gz \
  --macaque-site-medians /path/to/macaque_tissue_site_medians.txt \
  --marmoset-site-medians /path/to/marmoset_tissue_site_medians.txt \
  --gene-cons-frac /path/to/gene_constraint_fractions.tsv.gz \
  --e2g-dir /path/to/e2g_beds \
  --e2g-manifest refs/e2g_tissue_bed.tsv \
  --output /path/to/lncrna_category.txt \
  --e2g-by-tissue-out /path/to/gene_tss_e2g_by_tissue.parquet
```

**Output:** `lncrna_category.txt`; `gene_tss_e2g_by_tissue.parquet`

## Outputs provided

Zenodo (https://zenodo.org/records/24550349):

| Description | Path | Generated by |
|---|---|---|
| Custom GENCODE v39 vs v47 gene classification | `gene_annotation/gencode_v39_v47_gene_classification.txt` | `01_compare_gencode.py` |
| Custom lncRNA category table | `gene_annotation/lncrna_category.txt` | `03_classify_lncrna.py` |
| Constraint fractions | `gene_annotation/gene_constraint_fractions.tsv.gz` | `02_compute_gene_constraint_fractions.py` |
| E2G TSS overlap by tissue | `gene_annotation/gene_tss_e2g_by_tissue.parquet` | `03_classify_lncrna.py` |

## Notebooks

| Notebook | Notebook input file(s) | Generated by |
|---|---|---|
| Most main/supp figures | `gencode_v39_v47_gene_classification.txt`; `lncrna_category.txt` | `01_compare_gencode.py`; `03_classify_lncrna.py` |
| `supp_figure_12` | `gene_tss_e2g_by_tissue.parquet` | `03_classify_lncrna.py` |
