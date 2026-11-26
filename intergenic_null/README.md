# intergenic_null/ — Figure 2; Supplementary Figures 1–5 (expression)

Creates matched intergenic-null annotations and gene-vs-null expression calls
for adult GTEx, dGTEx, and NHP-dGTEx (macaque, marmoset).

## Inputs

| Input | Where to get it |
|---|---|
| GENCODE v47 full GTF | GENCODE website (`gencode.v47.annotation.gtf.gz`: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz) |
| GENCODE v47 collapsed GTF | GTEx portal (https://storage.googleapis.com/adult-gtex/references/v11/reference-tables/gencode.v47.genes.gtf) |
| GRCh38 chrom.sizes | UCSC (`hg38.chrom.sizes`: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes) |
| NHP full + collapsed GTFs (macaque, marmoset) | GTEx portal NHP-dGTEx (https://gtexportal.org/home/downloads/nhp-dgtex) (`Macaque_Mmul10_enhanced_2025-06-06.gtf`, `Macaque_Mmul10_enhanced_2025-06-06_genes_collapsed_stranded.gtf`; `Marmoset_mCalJa1.2.pat.X_enhanced_2025-06-06.gtf`, `Marmoset_mCalJa1.2.pat.X_enhanced_2025-06-06_genes_collapsed_stranded.gtf`) |
| NHP chrom.sizes (macaque, marmoset) | GTEx portal NHP-dGTEx (https://gtexportal.org/home/downloads/nhp-dgtex): derive from `GCA_003339765.3_Mmul_10_genomic_ucsc_chrs.fasta.fai` / `GCA_011100555.2_mCalJa1.2.pat.X_chrM_ucsc_chrs.fasta.fai` (columns 1–2) |
| GTEx RNA-SeQC BAMs | Protected access data (accession: phs000424.v11.p1) (https://gtexportal.org/home/protectedDataAccess) |
| dGTEx + NHP-dGTEx RNA-SeQC BAMs | dGTEx: protected access data (accession: phs003838.v1.p1) (https://gtexportal.org/home/protectedDataAccess); NHP-dGTEx: public AnVIL workspace (https://gtexportal.org/home/downloads/nhp-dgtex) |
| GTEx RNA-SeQC gene quantification | GTEx portal (`GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_reads.gct.gz`: https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression) |
| dGTEx + NHP-dGTEx RNA-SeQC gene quantification | GTEx portal dGTEx / NHP-dGTEx (https://gtexportal.org/home/downloads/developmental-gtex; https://gtexportal.org/home/downloads/nhp-dgtex) |
| GTEx sample attributes | GTEx portal (`GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt`: https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt); `--sample_meta_file` with `SAMPID`, `SMTSD` (keeps `SMAFRZE == RNASEQ`) |
| dGTEx + NHP-dGTEx sample attributes | GTEx portal (`dGTEx_Analysis_2026-01-30_v1_Sample_Attributes_Dataset_DS_open.xlsx`: https://storage.googleapis.com/developmental-gtex/annotations/v1/metadata-files/dGTEx_Analysis_2026-01-30_v1_Sample_Attributes_Dataset_DS_open.xlsx; `NHP_dGTEx_Analysis_2026-02-23_Macaque_v1_Sample_Attributes_Dataset_DS.xlsx`: https://storage.googleapis.com/nhp-dgtex/annotations/v1/metadata-files/macaques/NHP_dGTEx_Analysis_2026-02-23_Macaque_v1_Sample_Attributes_Dataset_DS.xlsx; `NHP_dGTEx_Analysis_2026-02-06_Marmoset_v1_Sample_Atttributes_Dataset_DS.xlsx`: https://storage.googleapis.com/nhp-dgtex/annotations/v1/metadata-files/marmosets/NHP_dGTEx_Analysis_2026-02-06_Marmoset_v1_Sample_Atttributes_Dataset_DS.xlsx); `--sample_meta_file` with `--sample_id_col SAMPLE_ID --tissue_col SMTSD` (keeps `SMAFRZE == RNASEQ`) |

## Analysis pipeline

To activate a Pixi environment with the required packages, run `pixi run -e intergenic-null …`.

`00`–`02` run **once per cohort** (swap GTFs, BAMs, gene matrices, and sample meta). Example commands below are adult GTEx; also run dGTEx, macaque, and marmoset:

| `--cohort_label` (`02`) | Null GTF (`00` / `01 --genes_gtf`) | `01` stranded | `02` sample-id col |
|---|---|---|---|
| `GTEx` | `gencode.v47.matched_null.unstranded.gtf` (human v47 GTFs + `hg38.chrom.sizes`) | omit | default `SAMPID` |
| `dGTEx` | `gencode.v47.matched_null.stranded.gtf` (same human inputs; write stranded output) | `--stranded rf` | `--sample_id_col SAMPLE_ID` |
| `macaque` | `Macaque_Mmul10.matched_null.stranded.gtf` (Mmul10 full + collapsed GTFs + chrom.sizes) | `--stranded rf` | `--sample_id_col SAMPLE_ID` |
| `marmoset` | `Marmoset_mCalJa1.2.matched_null.stranded.gtf` (mCalJa1.2 full + collapsed GTFs + chrom.sizes) | `--stranded rf` | `--sample_id_col SAMPLE_ID` |

### `00_create_matched_nongenic_null.py`

Build matched intergenic-null GTF.

```bash
python 00_create_matched_nongenic_null.py \
  --collapsed_gtf /path/to/gencode.v47.genes.gtf \
  --full_gtf /path/to/gencode.v47.annotation.gtf \
  --chrom_sizes /path/to/hg38.chrom.sizes \
  --output /path/to/gencode.v47.matched_null.unstranded.gtf
```

**Output:** `gencode.v47.matched_null.{stranded,unstranded}.gtf`; `{Macaque_Mmul10,Marmoset_mCalJa1.2}.matched_null.stranded.gtf`

### `01_quantify_rnaseqc_null.sh`

RNA-SeQC quantify one sample on the null GTF.

```bash
bash 01_quantify_rnaseqc_null.sh \
  --output_dir /path/to/null_quant \
  --sample_id SAMPLE \
  --bam_dir /path/to/bams \
  --genes_gtf /path/to/gencode.v47.matched_null.unstranded.gtf \
  --bam_file_end Aligned.sortedByCoord.out.patched.md.bam
```

Stranded cohorts: add `--stranded rf`.

**Output:** `rnaseqc_null/{sample_id}.gene_reads.gct.gz`

### `02_call_null_expression.py`

Gene-vs-null calls: pooled + every tissue with n ≥ 10.

```bash
python 02_call_null_expression.py \
  --cohort_label GTEx \
  --gene_matrix /path/to/gene_reads.gct.gz \
  --null_rnaseqc_dir /path/to/null_quant \
  --output_dir /path/to/null_quantifications \
  --sample_meta_file /path/to/SampleAttributesDS.txt
```

dGTEx/NHP: add `--sample_id_col SAMPLE_ID`.

**Output:** `{cohort}.null_quantifications.txt`; `by_tissue/{cohort}/{Tissue}.null_quantifications.txt`

## Outputs provided

Zenodo (https://zenodo.org/records/24550349):

| Description | Path | Generated by |
|---|---|---|
| Matched-null GTFs (v47, stranded, NHP) | `intergenic_null/gencode.v47.matched_null.{stranded,unstranded}.gtf`; `intergenic_null/{Macaque_Mmul10,Marmoset_mCalJa1.2}.matched_null.stranded.gtf` | `00_create_matched_nongenic_null.py` |
| Gene-vs-null expression calls (GTEx, dGTEx, macaque, marmoset) | `expression/{cohort}.null_quantifications.txt` (pooled) + `expression/by_tissue/{cohort}/{Tissue}.null_quantifications.txt` | `02_call_null_expression.py` |

## Notebooks

| Notebook | Notebook input file(s) | Generated by |
|---|---|---|
| `figure_2`, `supp_figure_4`–`5` | `GTEx.null_quantifications.txt`; `by_tissue/GTEx/{Tissue}.null_quantifications.txt` | `02_call_null_expression.py` |
