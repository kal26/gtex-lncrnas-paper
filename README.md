# lncRNA GTEx — paper analysis code

Scripts to reproduce the analyses in *Updated GTEx resource expands the catalog
of lncRNA-QTLs across human tissues*. Each Methods folder documents its inputs,
run order, and which figure notebooks consume its outputs.

Paper-specific tables and figure caches are on **Zenodo**
([zenodo.org/records/24550349](https://zenodo.org/records/24550349)).
Portal QTL / expression releases remain at [gtexportal.org](https://gtexportal.org/).

| Folder                                       | Figures                   |
| -------------------------------------------- | ------------------------- |
| [`annotation/`](annotation/)                 | Figs 1–5; Supp 7 & 12     |
| [`intergenic_null/`](intergenic_null/)       | Fig 2; Supp 1–5           |
| [`variant_annotation/`](variant_annotation/) | Fig 3e; Supp 9 & 15h      |
| [`alphagenome_qtl/`](alphagenome_qtl/)       | Fig 3f; Supp 10 & 19      |
| [`ldsc/`](ldsc/)                             | Fig 4a–b; Supp 11         |
| [`coloc/`](coloc/)                           | Figs 4c–e & 5; Supp 13–18 |

## Environments

Analysis dependencies are [pixi](https://pixi.sh) environments in
[`pixi.toml`](pixi.toml) (linux-64). Each Methods folder has a matching env,
plus `notebooks` for figure notebooks. Exception: [`ldsc/`](ldsc/) uses
`ldsc-prep` for annotation build / parse (`01`, `05`) and `ldsc` for the
bioconda LDSC binaries (`02`–`04`).

```bash
pixi install
pixi run -e <env> …   # e.g. annotation, coloc, notebooks
```

[`alphagenome_qtl/`](alphagenome_qtl/) needs a CUDA GPU and a Kaggle login for
AlphaGenome weights (`KAGGLE_USERNAME` / `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`).

## Data access

The README for each Methods subsection documents where to obtain the data needed
for that analysis.

## Figures

Figure notebooks live under [`notebooks/`](notebooks/) (`main_figures/` and
`supp_figures/`). Set each notebook’s `path/to/...` placeholders to your local
Zenodo / portal inputs before re-running. Rendered PDFs and PNGs are in
[`figures/`](figures/).

