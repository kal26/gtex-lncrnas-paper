#!/usr/bin/env Rscript
#' Colocalize one tissue × LD-region item (Python → this Rscript; no bash worker).
#' Item: {tissue}__chr{chrom}_{start}_{end} (single-phenotype pairs via --jobs_tsv).

library(argparser)

# --- helpers (load / fit / CS tables) ---

timestamp_cat <- function(...) {
  cat("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", ..., sep = "")
}

normalize_opt <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.na(x)) return(NULL)
  if (!nzchar(x)) return(NULL)
  if (tolower(x) == "none") return(NULL)
  x
}

# TSV reader that accepts plain / .gz / .xz job tables.
read_jobs_tsv <- function(path) {
  con <- if (grepl("\\.gz$", path, ignore.case = TRUE)) {
    gzfile(path, open = "rt")
  } else if (grepl("\\.xz$", path, ignore.case = TRUE)) {
    xzfile(path, open = "rt")
  } else {
    path
  }
  on.exit({
    if (!is.character(con)) try(close(con), silent = TRUE)
  }, add = TRUE)
  read.delim(con, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
}

# Vectorized: x can be a single string or character vector (e.g. semicolon-separated phenotype_id).
extract_core_gene_id <- function(x) {
  if (is.null(x)) return(NULL)
  if (length(x) == 0) return(character(0))
  out <- rep(NA_character_, length(x))
  ok <- !is.na(x) & nchar(x) > 0
  if (!any(ok)) return(out)
  m <- regexpr("ENSG[0-9]+", x[ok])
  matched <- m > 0
  out[ok][matched] <- regmatches(x[ok], m)
  out[ok][!matched] <- x[ok][!matched]
  if (length(out) == 1) return(out[[1]])
  out
}

# Write .focal.txt for credible set tables under output_prefix (rows where phenotype_id matches focal gene).
write_focal_filtered <- function(output_prefix, focal_gene_id) {
  if (is.null(focal_gene_id) || !nzchar(focal_gene_id)) return(invisible(NULL))
  focal_core <- extract_core_gene_id(focal_gene_id)
  if (!nzchar(focal_core)) return(invisible(NULL))
  base_dir <- dirname(output_prefix)
  base_stem <- basename(output_prefix)
  files <- list.files(base_dir, pattern = paste0("^", gsub("[.]", "[.]", base_stem), "[.].*[.]txt$"), full.names = TRUE)
  files <- files[!grepl("[.]focal[.]txt$", files)]
  if (length(files) == 0) {
    timestamp_cat("No credible-set .txt files found for prefix; skipping focal filter\n")
    return(invisible(NULL))
  }
  timestamp_cat("Focal filter: ", length(files), " file(s), focal_core=", focal_core, "\n")
  for (f in files) {
    d <- tryCatch(read.delim(f, header = TRUE, sep = "\t", stringsAsFactors = FALSE), error = function(e) NULL)
    if (is.null(d) || !"phenotype_id" %in% names(d) || nrow(d) == 0) next
    n_before <- nrow(d)
    ids <- strsplit(as.character(d$phenotype_id), ";")
    keep <- vapply(ids, function(x) any(extract_core_gene_id(x) == focal_core, na.rm = TRUE), logical(1))
    n_after <- sum(keep)
    out_f <- file.path(base_dir, sub("[.]txt$", ".focal.txt", basename(f)))
    if (any(keep)) {
      write.table(d[keep, ], file = out_f, sep = "\t", row.names = FALSE, quote = FALSE)
      timestamp_cat("Focal filter ", basename(f), ": filtered from ", n_before, " rows to ", n_after, " rows -> ", basename(out_f), "\n")
    } else {
      write.table(d[0, ], file = out_f, sep = "\t", row.names = FALSE, quote = FALSE)
      timestamp_cat("Focal filter ", basename(f), ": filtered from ", n_before, " rows to 0 rows -> ", basename(out_f), "\n")
    }
  }
  invisible(NULL)
}

# Load pecotmr + colocboost (single-threaded BLAS / data.table).
load_coloc_libs <- function() {
  library(RhpcBLASctl)
  library(magrittr)
  library(dplyr)
  library(readr)
  library(pecotmr)
  library(colocboost)
  omp_set_num_threads(1L)
  blas_set_num_threads(1L)
  data.table::setDTthreads(1L)
  invisible(NULL)
}


# Phenotype / covariate / condition path lists for one tissue.
write_pecotmr_lists <- function(
  tissue_id,
  expression_pheno_dir,
  splicing_pheno_dir,
  expression_covar_dir,
  splicing_covar_dir
) {
  gene_bed <- character(0)
  covar <- character(0)
  cond <- character(0)
  if (nzchar(expression_pheno_dir) && expression_pheno_dir != "none") {
    gene_bed <- c(gene_bed, file.path(expression_pheno_dir, paste0(tissue_id, ".v11.normalized_expression.bed.gz")))
    covar <- c(covar, file.path(expression_covar_dir, paste0(tissue_id, ".v11.covariates.txt")))
    cond <- c(cond, paste0(tissue_id, ".expression"))
  }
  if (nzchar(splicing_pheno_dir) && splicing_pheno_dir != "none") {
    gene_bed <- c(gene_bed, file.path(splicing_pheno_dir, paste0(tissue_id, ".v11.sQTL_phenotypes.bed.gz")))
    covar <- c(covar, file.path(splicing_covar_dir, paste0(tissue_id, ".v11.sQTL_covariates.txt")))
    cond <- c(cond, paste0(tissue_id, ".splicing"))
  }
  list(phenotype_list = gene_bed, covariate_list = covar, conditions_list = cond)
}

# Keep GWAS trait ids with >=1 variant at p < p_threshold in variant_region (tabix).
filter_gwas_for_region <- function(variant_region, gwas_ids, gwas_dir, p_threshold = 1e-5) {
  parts <- strsplit(variant_region, "[:\\-]")[[1]]
  stopifnot(length(parts) == 3L)
  chr_tabix <- sub("^chr", "", parts[[1]])
  region <- paste0(chr_tabix, ":", parts[[2]], "-", parts[[3]])
  p_threshold <- as.numeric(p_threshold)
  timestamp_cat("Filtering GWAS for p < ", p_threshold, " in ", variant_region, "\n")
  kept <- character(0)
  for (gwas_id in gwas_ids) {
    if (!nzchar(gwas_id)) next
    gwas_file <- file.path(gwas_dir, paste0("imputed_", gwas_id, ".txt.gz"))
    if (!file.exists(gwas_file)) {
      timestamp_cat("  ", gwas_id, ": SKIPPED (file not found)\n")
      next
    }
    tabix_out <- tryCatch(
      system2("tabix", c("-h", gwas_file, region), stdout = TRUE, stderr = TRUE),
      error = function(e) character(0)
    )
    strong <- 0L
    if (length(tabix_out)) {
      for (line in tabix_out) {
        if (!nzchar(line) || startsWith(line, "#")) next
        cols <- strsplit(line, "\t", fixed = TRUE)[[1]]
        if (length(cols) < 11L) next
        pval <- suppressWarnings(as.numeric(cols[[11]]))
        if (!is.na(pval) && pval < p_threshold) strong <- strong + 1L
      }
    }
    if (strong > 0L) {
      kept <- c(kept, gwas_id)
      timestamp_cat("  ", gwas_id, ": PASSED\n")
    } else {
      timestamp_cat("  ", gwas_id, ": FILTERED\n")
    }
  }
  timestamp_cat("Filtered to ", length(kept), " / ", length(gwas_ids), " GWAS traits\n")
  kept
}

apply_r_thread_env <- function() {
  Sys.setenv(
    VROOM_THREADS = "1",
    R_DATATABLE_NUM_THREADS = "1",
    OMP_NUM_THREADS = "1",
    OPENBLAS_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1",
    VECLIB_MAXIMUM_THREADS = "1",
    NUMEXPR_NUM_THREADS = "1",
    BLIS_NUM_THREADS = "1"
  )
  1L
}

# Load pecotmr region_data; returns NULL on soft-skip / empty.
# phenotype_list / covariate_list / conditions_list may be character vectors.
load_region_data_block <- function(
  gene_region,
  variant_region,
  genotype_stem,
  phenotype_list,
  covariate_list,
  conditions_list,
  include_gwas = FALSE,
  gwas_ids = NULL,
  gwas_paths = NULL,
  gwas_meta = NULL,
  ld_meta = NULL,
  gwas_column_matching = NULL,
  phenotype_id = NULL,
  keep_gene_ids = NULL,
  maf_cutoff = 0,
  mac_cutoff = 0,
  xvar_cutoff = 0,
  imiss_cutoff = 0,
  write_phenotype_ids_path = NULL
) {
  phenotype_id <- normalize_opt(phenotype_id)
  if (length(phenotype_list) == 0L || all(!nzchar(phenotype_list))) {
    timestamp_cat("No phenotypes - skipping\n")
    return(NULL)
  }
  if (length(covariate_list) != length(phenotype_list) ||
      length(conditions_list) != length(phenotype_list)) {
    stop("Length mismatch: phenotype_list, covariate_list, conditions_list")
  }
  n_filters <- sum(c(
    !is.null(phenotype_id),
    !is.null(keep_gene_ids) && length(keep_gene_ids) > 0L
  ))
  if (n_filters > 1L) {
    stop("Use at most one of phenotype_id, keep_gene_ids")
  }

  load_args <- list(
    region = gene_region,
    genotype_list = c(genotype_stem),
    phenotype_list = phenotype_list,
    covariate_list = covariate_list,
    conditions_list_individual = conditions_list,
    region_name_col = 4L,
    maf_cutoff = maf_cutoff,
    mac_cutoff = mac_cutoff,
    xvar_cutoff = xvar_cutoff,
    imiss_cutoff = imiss_cutoff,
    association_window = variant_region
  )

  if (isTRUE(include_gwas)) {
    if (is.null(gwas_ids) || is.null(gwas_paths) || is.null(gwas_meta) ||
        is.null(ld_meta) || is.null(gwas_column_matching)) {
      stop("GWAS args required when include_gwas=TRUE")
    }
    gwas_meta_df <- if (is.character(gwas_meta) && length(gwas_meta) == 1L && file.exists(gwas_meta)) {
      read.delim(gwas_meta, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
    } else {
      gwas_meta
    }
    gwas_meta_df$Tag <- as.character(gwas_meta_df$Tag)
    gwas_meta_matched <- gwas_meta_df[match(gwas_ids, gwas_meta_df$Tag), ]
    n_samples <- as.numeric(gwas_meta_matched$Sample_Size)
    n_cases <- as.numeric(gwas_meta_matched$Cases)
    n_controls <- as.numeric(gwas_meta_matched$Controls)
    for (i in seq_along(n_samples)) {
      if (!is.na(n_cases[i]) && n_cases[i] != 0) {
        n_samples[i] <- 0
        n_controls[i] <- as.numeric(gwas_meta_matched$Sample_Size[i]) - n_cases[i]
      } else {
        n_cases[i] <- 0
        n_controls[i] <- 0
      }
    }
    load_args$sumstat_path_list <- gwas_paths
    load_args$column_file_path_list <- rep(gwas_column_matching, length(gwas_paths))
    load_args$LD_meta_file_path_list <- ld_meta
    load_args$conditions_list_sumstat <- gwas_ids
    load_args$match_LD_sumstat <- list(gwas_ids)
    load_args$n_samples <- n_samples
    load_args$n_cases <- n_cases
    load_args$n_controls <- n_controls
    timestamp_cat("Loading region data with GWAS for ", gene_region, "\n")
  } else {
    timestamp_cat("Loading xQTL data for region ", gene_region, "\n")
  }

  region_data <- tryCatch({
    do.call(load_multitask_regional_data, load_args)
  }, error = function(e) {
    if (is_soft_skip_load_error(e)) {
      timestamp_cat(
        "WARNING: Skipping unusable region load for ", variant_region,
        " (", conditionMessage(e), ").\n"
      )
      return(NULL)
    }
    stop(e)
  })
  if (is.null(region_data)) {
    timestamp_cat("No data for region - skipping\n")
    return(NULL)
  }

  if (isTRUE(include_gwas) && !is.null(write_phenotype_ids_path)) {
    cond_names <- names(region_data$individual_data$residual_Y)
    pheno_ids <- character(0)
    for (i in seq_along(cond_names)) {
      y_data <- region_data$individual_data$residual_Y[[i]]
      ids <- if (is.matrix(y_data)) colnames(y_data) else names(y_data)
      if (is.null(ids) || length(ids) == 0L) next
      pheno_ids <- c(pheno_ids, paste0(cond_names[i], "_", ids))
    }
    dir.create(dirname(write_phenotype_ids_path), recursive = TRUE, showWarnings = FALSE)
    writeLines(pheno_ids, write_phenotype_ids_path)
    timestamp_cat("Saved phenotype IDs to ", write_phenotype_ids_path, " (", length(pheno_ids), " phenotypes)\n")
  }

  if (!is.null(phenotype_id)) {
    region_data <- filter_region_data_by_phenotype(region_data, phenotype_id, verbose = TRUE)
  } else if (!is.null(keep_gene_ids) && length(keep_gene_ids) > 0L) {
    region_data <- filter_region_data_by_genes(
      region_data, keep_gene_ids, verbose = TRUE, invert = FALSE, allow_empty = TRUE
    )
  }
  if (is.null(region_data)) {
    timestamp_cat("No xQTL phenotypes remaining after gene filter - skipping\n")
    return(NULL)
  }
  region_data
}

# Filter region_data to phenotypes whose gene ID is in gene_id_list (core ENSG match).
# invert=TRUE keeps genes *not* in the list. allow_empty=TRUE returns NULL if none remain.
# residual_X / maf / X_variance are per-variant (samples × SNPs), not per-phenotype — keep full.
filter_region_data_by_genes <- function(region_data, gene_id_list, verbose = TRUE,
                                        invert = FALSE, allow_empty = FALSE) {
  stopifnot(!is.null(region_data$individual_data$residual_Y))
  gene_list_core <- unique(sapply(gene_id_list, extract_core_gene_id))
  gene_list_core <- gene_list_core[!is.na(gene_list_core)]
  cond_names <- names(region_data$individual_data$residual_Y)
  residual_Y_f <- residual_X_f <- residual_Y_scalar_f <- residual_X_scalar_f <- maf_f <- X_variance_f <- list()
  dropped_X_f <- dropped_Y_f <- dropped_covar_f <- list()
  for (i in seq_along(cond_names)) {
    y_data <- region_data$individual_data$residual_Y[[i]]
    gene_ids <- if (is.matrix(y_data)) colnames(y_data) else names(y_data)
    if (is.null(gene_ids) || length(gene_ids) == 0) next
    keep <- sapply(gene_ids, extract_core_gene_id) %in% gene_list_core
    if (invert) keep <- !keep
    if (sum(keep) == 0) next
    if (verbose) {
      timestamp_cat(
        "  ", cond_names[i], ": ", sum(keep), " of ", length(gene_ids),
        if (invert) " kept (exclude)\n" else " matched\n"
      )
    }
    residual_Y_f[[cond_names[i]]] <- if (is.matrix(y_data)) y_data[, keep, drop = FALSE] else y_data[keep]
    x_data <- region_data$individual_data$residual_X[[i]]
    residual_X_f[[cond_names[i]]] <- if (is.matrix(x_data)) x_data else as.matrix(x_data)
    residual_Y_scalar_f[[cond_names[i]]] <- region_data$individual_data$residual_Y_scalar[[i]][keep]
    residual_X_scalar_f[[cond_names[i]]] <- region_data$individual_data$residual_X_scalar[[i]]
    maf_f[[cond_names[i]]] <- region_data$individual_data$maf[[i]]
    xv <- region_data$individual_data$X_variance[[i]]
    X_variance_f[[cond_names[i]]] <- if (is.matrix(xv)) xv else xv
    dropped_X_f[[cond_names[i]]] <- region_data$individual_data$dropped_sample$X[[i]]
    dropped_Y_f[[cond_names[i]]] <- region_data$individual_data$dropped_sample$Y[[i]]
    dropped_covar_f[[cond_names[i]]] <- region_data$individual_data$dropped_sample$covar[[i]]
  }
  if (length(residual_Y_f) == 0) {
    if (allow_empty) {
      if (verbose) timestamp_cat("No genes remaining after filter (invert=", invert, ")\n")
      return(NULL)
    }
    stop("No genes matched.")
  }
  out <- list(
    individual_data = list(
      residual_Y = residual_Y_f,
      residual_X = residual_X_f,
      residual_Y_scalar = residual_Y_scalar_f,
      residual_X_scalar = residual_X_scalar_f,
      dropped_sample = list(X = dropped_X_f, Y = dropped_Y_f, covar = dropped_covar_f),
      maf = maf_f,
      X = region_data$individual_data$X,
      chrom = region_data$individual_data$chrom,
      grange = region_data$individual_data$grange,
      X_variance = X_variance_f
    ),
    sumstat_data = region_data$sumstat_data
  )
  if ("Y_coordinates" %in% names(region_data$individual_data)) {
    out$individual_data$Y_coordinates <- region_data$individual_data$Y_coordinates
  }
  out
}

# Keep a single phenotype column (samples × SNPs X / maf stay full for that condition).
filter_region_data_by_phenotype <- function(region_data, phenotype_id, verbose = TRUE) {
  stopifnot(!is.null(region_data$individual_data$residual_Y))
  pheno_id <- as.character(phenotype_id)
  cond_names <- names(region_data$individual_data$residual_Y)
  for (i in seq_along(cond_names)) {
    y_data <- region_data$individual_data$residual_Y[[i]]
    ids <- if (is.matrix(y_data)) colnames(y_data) else names(y_data)
    if (is.null(ids)) next
    prefix <- paste0(cond_names[i], "_")
    col_id <- if (startsWith(pheno_id, prefix)) substring(pheno_id, nchar(prefix) + 1L) else pheno_id
    j <- match(col_id, ids)
    if (is.na(j)) next
    cond_keep <- cond_names[i]
    if (verbose) {
      timestamp_cat("Keeping phenotype ", pheno_id, " in condition ", cond_keep, "\n")
    }
    if (is.matrix(y_data)) {
      single_Y <- list(y_data[, j, drop = FALSE])
    } else {
      single_Y <- list(matrix(y_data, ncol = 1L, dimnames = list(names(y_data), col_id)))
    }
    names(single_Y) <- cond_keep
    single_X <- list(region_data$individual_data$residual_X[[i]])
    names(single_X) <- cond_keep
    xvar_i <- region_data$individual_data$X_variance[[i]]
    single_X_variance <- if (is.matrix(xvar_i)) list(xvar_i) else list(xvar_i)
    filtered <- list(
      individual_data = list(
        residual_Y = single_Y,
        residual_X = single_X,
        residual_Y_scalar = list(region_data$individual_data$residual_Y_scalar[[i]][j]),
        residual_X_scalar = list(region_data$individual_data$residual_X_scalar[[i]]),
        dropped_sample = list(
          X = list(region_data$individual_data$dropped_sample$X[[i]]),
          Y = list(region_data$individual_data$dropped_sample$Y[[i]]),
          covar = list(region_data$individual_data$dropped_sample$covar[[i]])
        ),
        maf = list(region_data$individual_data$maf[[i]]),
        X = region_data$individual_data$X,
        chrom = region_data$individual_data$chrom,
        grange = region_data$individual_data$grange,
        X_variance = single_X_variance
      ),
      sumstat_data = region_data$sumstat_data
    )
    names(filtered$individual_data$residual_Y_scalar) <- names(filtered$individual_data$residual_X_scalar) <-
      names(filtered$individual_data$maf) <- names(filtered$individual_data$X_variance) <- cond_keep
    names(filtered$individual_data$dropped_sample$X) <- names(filtered$individual_data$dropped_sample$Y) <-
      names(filtered$individual_data$dropped_sample$covar) <- cond_keep
    if ("Y_coordinates" %in% names(region_data$individual_data)) {
      filtered$individual_data$Y_coordinates <- region_data$individual_data$Y_coordinates
    }
    return(filtered)
  }
  stop("phenotype_id not found in region_data: ", pheno_id)
}


# Load failures that indicate empty / unusable regions rather than pipeline bugs.
# Includes NoSNPsError (empty genotype window, e.g. centromeric half-splits).
is_soft_skip_load_error <- function(e) {
  if (inherits(e, "NoSNPsError")) {
    return(TRUE)
  }
  msg <- conditionMessage(e)
  grepl(
    paste(
      "All phenotype files are empty",
      "No SNPs found",
      "subscript out of bounds",
      "No or only 1 overlapping variables",
      "colocalization cannot be performed",
      sep = "|"
    ),
    msg,
    ignore.case = TRUE
  )
}

# Package xQTL pipeline result for format_colocboost, keeping dual-assemble robust sibling.
pack_xqtl_coloc_result <- function(res) {
  out <- list(xqtl_coloc = res$xqtl_coloc)
  if (!is.null(res$xqtl_coloc_robust)) {
    out$xqtl_coloc_robust <- res$xqtl_coloc_robust
  }
  out
}

# Fit ColocBoost on an in-memory region_data and write RDS + CS tables.
# Returns invisibly; quits with status 42 if focal gene is PIP-skipped.
fit_colocboost_region <- function(
  region_data,
  output_dir,
  tissue_id,
  gene_item,
  mode = c("xqtl", "gwas"),
  output_stem = "none",
  focal_gene_id = NULL,
  run_joint_gwas = TRUE,
  run_separate_gwas = FALSE,
  maf_cutoff = 0.01,
  pip_threshold = -1,
  quit_on_focal_skip = TRUE
) {
  mode <- match.arg(mode)
  focal_gene_id <- normalize_opt(focal_gene_id)
  xqtl_coloc <- identical(mode, "xqtl")
  run_joint_gwas <- !xqtl_coloc && isTRUE(run_joint_gwas)
  run_separate_gwas <- !xqtl_coloc && isTRUE(run_separate_gwas)

  gene_names <- names(region_data$individual_data$residual_Y)
  n_pheno <- length(gene_names)
  n_gwas <- if (!is.null(region_data$sumstat_data)) length(region_data$sumstat_data) else 0L
  n_pheno_display <- sum(vapply(region_data$individual_data$residual_Y, function(y) {
    if (is.null(y)) 0L else if (is.matrix(y)) ncol(y) else length(y)
  }, integer(1L)))
  n_gwas_display <- if (!is.null(region_data$sumstat_data) && !is.null(region_data$sumstat_data$sumstats)) {
    length(unique(unlist(lapply(region_data$sumstat_data$sumstats, names))))
  } else 0L

  if (xqtl_coloc) {
    timestamp_cat("Running xQTL coloc (", n_pheno, " phenotype contexts)\n")
    pip_ind <- rep(pip_threshold, n_pheno)
    pip_sum <- pip_threshold
  } else {
    timestamp_cat(
      "Running GWAS coloc (", n_pheno_display, " phenotypes in ", n_pheno,
      " contexts, ", n_gwas_display, " GWAS), stem=", output_stem, "\n"
    )
    pip_ind <- rep(pip_threshold, n_pheno)
    pip_sum <- rep(pip_threshold, n_gwas)
  }

  res <- colocboost_analysis_pipeline(
    region_data,
    focal_gene = focal_gene_id,
    pip_cutoff_to_skip_ind = pip_ind,
    pip_cutoff_to_skip_sumstat = pip_sum,
    qc_method = "slalom",
    maf_cutoff = maf_cutoff,
    xqtl_coloc = xqtl_coloc,
    joint_gwas = run_joint_gwas,
    separate_gwas = run_separate_gwas,
    pvalue_cutoff = NULL,
    cos_npc_cutoff = 0.2,
    npc_outcome_cutoff = 0.2,
    robust_cos_npc_cutoff = 0.75,
    robust_npc_outcome_cutoff = 0.2
  )

  if (isTRUE(res$focal_gene_skipped)) {
    timestamp_cat("Focal gene ", focal_gene_id, " skipped by PIP screening. No output files written.\n")
    if (isTRUE(quit_on_focal_skip)) quit(save = "no", status = 42)
    return(invisible(structure(NULL, status = 42L)))
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  if (xqtl_coloc) {
    out_rds <- file.path(output_dir, paste0(tissue_id, ".", gene_item, ".xqtl_only.colocboost.rds"))
    saveRDS(pack_xqtl_coloc_result(res), file = out_rds)
  } else {
    out_rds <- file.path(output_dir, paste0(tissue_id, ".", gene_item, ".", output_stem, ".colocboost.rds"))
    saveRDS(res, file = out_rds)
  }
  timestamp_cat("Saved to ", out_rds, "\n")

  prefix <- sub("[.]rds$", "", out_rds)
  format_colocboost(
    out_rds, prefix, verbose = TRUE,
    xqtl_coloc = xqtl_coloc, joint_gwas = run_joint_gwas, separate_gwas = run_separate_gwas
  )
  if (!is.null(focal_gene_id) && nzchar(focal_gene_id)) {
    write_focal_filtered(prefix, focal_gene_id)
  }
  timestamp_cat(if (xqtl_coloc) "xQTL" else "GWAS", " coloc done\n")
  invisible(structure(out_rds, status = 0L))
}

# --- ColocBoost RDS → credible-set tables ---
# Dual-assemble: standard → {prefix}.txt; robust → {prefix}.robust.txt.
# Refuses header-only placeholders when RDS is NULL / empty.

colocboost_table_profiles <- list(
  standard = list(
    name = "standard",
    outfile_suffix = "",
    cos_npc_cutoff = 0.2,
    npc_outcome_cutoff = 0.2,
    pvalue_cutoff = 0.001,
    ucos_npc_outcome_cutoff = 0.2,
    ucos_pvalue_cutoff = 0.001
  ),
  robust = list(
    name = "robust",
    outfile_suffix = ".robust",
    cos_npc_cutoff = 0.75,
    npc_outcome_cutoff = 0.2,
    pvalue_cutoff = 0.001,
    ucos_npc_outcome_cutoff = 0.2,
    ucos_pvalue_cutoff = 1e-5
  )
)

format_colocboost <- function(pecotmr_colocboost, output_prefix, verbose = TRUE, 
                               xqtl_coloc = NULL, joint_gwas = NULL, separate_gwas = NULL) {
  if (is.null(pecotmr_colocboost) || identical(pecotmr_colocboost, "")) {
    stop("Missing required argument: pecotmr_colocboost", call. = FALSE)
  }
  if (is.null(output_prefix) || identical(output_prefix, "")) {
    stop("Missing required argument: output_prefix", call. = FALSE)
  }

  if (isTRUE(verbose)) {
    cat("Input RDS:", pecotmr_colocboost, "\n")
    cat("Output prefix:", output_prefix, "\n")
  }

  obj <- readRDS(pecotmr_colocboost)

  if (is.null(obj)) {
    stop("Input RDS file contains NULL object; refusing to write empty output files.", call. = FALSE)
  }

  processed_any <- FALSE

  is_colocboost_object <- function(obj) {
    inherits(obj, "colocboost") ||
      (is.list(obj) && !is.null(obj$data_info) && !is.null(obj$data_info$variables))
  }

  # Apply p-value filters with CoS NPC matching assemble; write one table.
  process_and_write <- function(res, prefix, profile, verbose = TRUE) {
    if (is.null(res)) {
      stop(
        "NULL colocboost object for prefix: ", prefix,
        "; refusing to write empty output files.",
        call. = FALSE
      )
    }
    if (!is_colocboost_object(res)) {
      if (isTRUE(verbose)) {
        warning("Element for prefix '", prefix, "' is not a colocboost object. Skipping.", call. = FALSE)
      }
      return(invisible(NULL))
    }

    if (isTRUE(verbose)) {
      cat(
        "Formatting ", profile$name, " table for ", prefix,
        " (assemble-matched CoS NPC=", profile$cos_npc_cutoff,
        "/", profile$npc_outcome_cutoff,
        "; p=", profile$pvalue_cutoff,
        "; uCoS npc=", profile$ucos_npc_outcome_cutoff,
        ", p=", profile$ucos_pvalue_cutoff, ")\n",
        sep = ""
      )
    }

    filtered <- tryCatch({
      suppressWarnings(colocboost::get_robust_colocalization(
        res,
        cos_npc_cutoff = profile$cos_npc_cutoff,
        npc_outcome_cutoff = profile$npc_outcome_cutoff,
        pvalue_cutoff = profile$pvalue_cutoff,
        weight_fudge_factor = 1.5,
        coverage = 0.95,
        ucos_npc_outcome_cutoff = profile$ucos_npc_outcome_cutoff,
        ucos_pvalue_cutoff = profile$ucos_pvalue_cutoff
      ))
    }, error = function(e) {
      stop(
        "Failed to filter colocalization for prefix '", prefix, "' (", profile$name, "): ",
        conditionMessage(e),
        call. = FALSE
      )
    })

    if (is.null(filtered)) {
      stop("get_robust_colocalization returned NULL for prefix '", prefix, "'", call. = FALSE)
    }

    cs_df <- tryCatch({
      extract_colocboost_sets(filtered)
    }, error = function(e) {
      stop(
        "Failed to extract credible sets for prefix '", prefix, "' (", profile$name, "): ",
        conditionMessage(e),
        call. = FALSE
      )
    })
    if (is.null(cs_df)) {
      stop("extract_colocboost_sets returned NULL for prefix '", prefix, "'", call. = FALSE)
    }

    outfile <- paste0(prefix, profile$outfile_suffix, ".txt")
    write_cs_table(cs_df, outfile, verbose = verbose)
    invisible(TRUE)
  }

  process_component_pair <- function(standard_comp, robust_comp, prefix, verbose = TRUE) {
    wrote <- FALSE
    if (!is.null(standard_comp) && is_colocboost_object(standard_comp)) {
      process_and_write(standard_comp, prefix, colocboost_table_profiles$standard, verbose = verbose)
      wrote <- TRUE
    }
    if (!is.null(robust_comp) && is_colocboost_object(robust_comp)) {
      process_and_write(robust_comp, prefix, colocboost_table_profiles$robust, verbose = verbose)
      wrote <- TRUE
    }
    wrote
  }

  if (is.list(obj)) {
    should_process_xqtl <- (is.null(xqtl_coloc) || isTRUE(xqtl_coloc)) && !isFALSE(xqtl_coloc)
    if (should_process_xqtl && ("xqtl_coloc" %in% names(obj) || "xqtl_coloc_robust" %in% names(obj))) {
      if (process_component_pair(
        obj$xqtl_coloc, obj$xqtl_coloc_robust,
        paste0(output_prefix, ".xqtl_coloc"), verbose = verbose
      )) {
        processed_any <- TRUE
      } else if (isTRUE(verbose)) {
        warning("xqtl_coloc / xqtl_coloc_robust are not colocboost objects. Skipping.", call. = FALSE)
      }
    }

    should_process_joint <- (is.null(joint_gwas) || isTRUE(joint_gwas)) && !isFALSE(joint_gwas)
    if (should_process_joint && ("joint_gwas" %in% names(obj) || "joint_gwas_robust" %in% names(obj))) {
      if (process_component_pair(
        obj$joint_gwas, obj$joint_gwas_robust,
        paste0(output_prefix, ".joint_gwas"), verbose = verbose
      )) {
        processed_any <- TRUE
      } else if (isTRUE(verbose)) {
        warning("joint_gwas / joint_gwas_robust are not colocboost objects. Skipping.", call. = FALSE)
      }
    }
  }

  should_process_separate <- (is.null(separate_gwas) || isTRUE(separate_gwas)) && !isFALSE(separate_gwas)
  if (should_process_separate && is.list(obj)) {
    sg <- obj$separate_gwas
    sg_rob <- obj$separate_gwas_robust
    keys <- unique(c(
      if (!is.null(sg)) names(sg) else character(0),
      if (!is.null(sg_rob)) names(sg_rob) else character(0)
    ))
    for (key in keys) {
      if (is.null(key) || !nzchar(key)) next
      std <- if (!is.null(sg) && key %in% names(sg)) sg[[key]] else NULL
      rob <- if (!is.null(sg_rob) && key %in% names(sg_rob)) sg_rob[[key]] else NULL
      if (process_component_pair(
        std, rob,
        paste0(output_prefix, ".separate_gwas.", key), verbose = verbose
      )) {
        processed_any <- TRUE
      }
    }
  }

  if (!processed_any) {
    stop(
      "No recognizable colocboost components to format ",
      "(xqtl_coloc[_robust], joint_gwas[_robust], or separate_gwas[_robust]). ",
      "Refusing to write empty placeholder tables.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

# ----------------------------
# Helpers
# ----------------------------
extract_colocboost_sets <- function(cb_output, outcome_names = NULL) {
  if (!is.list(cb_output) || is.null(cb_output$data_info) || is.null(cb_output$data_info$variables)) {
    if (!inherits(cb_output, "colocboost")) {
      stop("cb_output must be a 'colocboost' object or a list with data_info$variables")
    }
  }

  variables <- cb_output$data_info$variables
  outcomes  <- cb_output$data_info$outcome_info$outcome_names
  if (!is.null(outcome_names)) {
    if (length(outcome_names) != length(outcomes)) {
      stop("outcome_names length must match number of outcomes in cb_output$data_info$outcome_info$outcome_names")
    }
    outcomes <- outcome_names
  }

  rows <- list()

  cd <- cb_output$cos_details
  if (!is.null(cd) && !is.null(cd$cos$cos_index) && length(cd$cos$cos_index) > 0) {
    cos_indices   <- cd$cos$cos_index
    cos_outcomes  <- cd$cos_outcomes$outcome_index
    cos_npc       <- cd$cos_npc
    cos_min_npc   <- cd$cos_min_npc_outcome
    cos_vcp_list  <- cd$cos_vcp

    for (i in seq_along(cos_indices)) {
      cs_name <- names(cos_indices)[i]
      var_idx <- cos_indices[[i]]
      if (length(var_idx) == 0) next

      variant_id <- variables[var_idx]

      out_idx <- cos_outcomes[[i]]
      phenotype_id <- paste(outcomes[out_idx], collapse = ";")

      vcp_vec <- cos_vcp_list[[i]]
      if (length(vcp_vec) == length(variables)) {
        vcp <- vcp_vec[var_idx]
      } else if (length(vcp_vec) == length(var_idx)) {
        vcp <- vcp_vec
      } else {
        vcp <- rep(NA_real_, length(var_idx))
      }

      cos_npc_val      <- as.numeric(cos_npc[i])
      npc_outcome_val  <- as.numeric(cos_min_npc[i])

      outcomes_npc_str <- ""
      if (!is.null(cd$cos_outcomes_npc)) {
        oname <- names(cos_indices)[i]
        vnpc <- cd$cos_outcomes_npc[[oname]]
        if (is.null(vnpc) && i <= length(cd$cos_outcomes_npc)) {
          vnpc <- cd$cos_outcomes_npc[[i]]
        }
        if (is.data.frame(vnpc) || is.list(vnpc)) {
          vnpc <- vnpc$npc_outcome
        }
        if (!is.null(vnpc) && length(vnpc) == length(out_idx)) {
          outcomes_npc_str <- paste(as.numeric(vnpc), collapse = ";")
        }
      }

      variant_neg_log10_p_values_list <- vector("list", length(var_idx))
      for (v in seq_along(var_idx)) {
        values_for_variant <- numeric(length(out_idx))
        for (j in seq_along(out_idx)) {
          outcome_idx <- out_idx[j]
          z_scores <- cb_output$data_info$z[[outcome_idx]][var_idx]
          neg_log10_p_values <- -log10(2 * pnorm(-abs(z_scores)))
          values_for_variant[j] <- neg_log10_p_values[v]
        }
        variant_neg_log10_p_values_list[[v]] <- values_for_variant
      }

      rows[[length(rows) + 1L]] <- data.frame(
        phenotype_id = rep(phenotype_id, length(var_idx)),
        variant_id   = variant_id,
        cs_id        = rep(cs_name, length(var_idx)),
        vcp          = vcp,
        cos_npc      = rep(cos_npc_val,     length(var_idx)),
        npc_outcome  = rep(npc_outcome_val, length(var_idx)),
        outcomes_npc_str = rep(outcomes_npc_str, length(var_idx)),
        neg_log10_p_value = I(variant_neg_log10_p_values_list),
        stringsAsFactors = FALSE
      )
    }
  }

  ud <- cb_output$ucos_details
  if (!is.null(ud) &&
      !is.null(ud$ucos$ucos_index) &&
      length(ud$ucos$ucos_index) > 0) {

    ucos_indices   <- ud$ucos$ucos_index
    ucos_outcomes  <- ud$ucos_outcomes
    ucos_weights   <- ud$ucos_weight
    ucos_outcomes_npc <- ud$ucos_outcomes_npc

    for (j in seq_along(ucos_indices)) {
      u_name  <- names(ucos_indices)[j]
      var_idx <- ucos_indices[[j]]
      if (length(var_idx) == 0) next

      variant_id <- variables[var_idx]

      out_idx_list <- ucos_outcomes$outcome_index[[j]]
      out_name_list <- ucos_outcomes$outcome_name[[j]]
      
      if (!is.null(out_name_list) && length(out_name_list) > 0) {
        phenotype_id <- out_name_list[[1]]
        if (!is.null(out_idx_list) && length(out_idx_list) > 0) {
          out_idx <- out_idx_list[[1]]
        } else {
          out_idx <- match(phenotype_id, outcomes)
        }
      } else if (!is.null(out_idx_list) && length(out_idx_list) > 0) {
        out_idx <- out_idx_list[[1]]
        if (out_idx >= 1 && out_idx <= length(outcomes)) {
          phenotype_id <- outcomes[out_idx]
        } else {
          phenotype_id <- NA_character_
        }
      } else {
        phenotype_id <- NA_character_
        out_idx <- NA_integer_
      }

      w_vec <- rep(NA_real_, length(var_idx))
      if (!is.null(ucos_weights) && j <= length(ucos_weights)) {
        w <- ucos_weights[[j]]
        if (!is.null(w) && is.matrix(w) && ncol(w) >= 1) {
          w_all <- as.numeric(w[, 1])
          if (length(w_all) == length(variables) && all(var_idx <= length(w_all))) {
            w_vec <- w_all[var_idx]
          }
        } else if (!is.null(w) && (is.vector(w) || is.list(w))) {
          w_all <- as.numeric(w)
          if (length(w_all) == length(variables) && all(var_idx <= length(w_all))) {
            w_vec <- w_all[var_idx]
          } else if (length(w_all) == length(var_idx)) {
            w_vec <- w_all
          }
        }
      }

      npc_outcome_val <- NA_real_
      if (!is.null(ucos_outcomes_npc) && is.data.frame(ucos_outcomes_npc) && 
          "npc_outcome" %in% colnames(ucos_outcomes_npc) && 
          j <= nrow(ucos_outcomes_npc)) {
        npc_outcome_val <- as.numeric(ucos_outcomes_npc$npc_outcome[j])
        if (is.na(npc_outcome_val)) {
          npc_outcome_val <- NA_real_
        }
      }

      neg_log10_p_values <- rep(NA_real_, length(var_idx))
      if (!is.null(cb_output$data_info$z) && !is.na(out_idx) && length(cb_output$data_info$z) >= out_idx) {
        z_scores <- cb_output$data_info$z[[out_idx]][var_idx]
        neg_log10_p_values <- -log10(2 * pnorm(-abs(z_scores)))
      }

      outcomes_ucos_npc_str <- ""
      if (!is.na(npc_outcome_val)) {
        outcomes_ucos_npc_str <- as.character(npc_outcome_val)
      }

      rows[[length(rows) + 1L]] <- data.frame(
        phenotype_id = rep(phenotype_id, length(var_idx)),
        variant_id   = variant_id,
        cs_id        = rep(u_name, length(var_idx)),
        vcp          = w_vec,
        cos_npc      = rep(NA_real_, length(var_idx)),
        npc_outcome  = rep(npc_outcome_val, length(var_idx)),
        outcomes_npc_str = rep(outcomes_ucos_npc_str, length(var_idx)),
        neg_log10_p_value = neg_log10_p_values,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(rows) == 0) {
    return(data.frame(
      phenotype_id = character(0),
      variant_id   = character(0),
      cs_id        = character(0),
      vcp          = numeric(0),
      cos_npc      = numeric(0),
      npc_outcome  = numeric(0),
      outcomes_npc_str = character(0),
      neg_log10_p_value = numeric(0),
      stringsAsFactors = FALSE
    ))
  }

  do.call(rbind, rows)
}

empty_cs_df <- function() {
  data.frame(
    phenotype_id = character(0),
    variant_id   = character(0),
    cs_id        = character(0),
    vcp          = numeric(0),
    cos_npc      = numeric(0),
    npc_outcome  = numeric(0),
    outcomes_npc_str = character(0),
    neg_log10_p_value = numeric(0),
    stringsAsFactors = FALSE
  )
}

#' Write header-only .txt / .robust.txt for regions with no usable colocalization.
#' Uses the current schema (includes outcomes_npc_str).
write_empty_colocboost_placeholders <- function(output_prefix, components = "joint_gwas") {
  for (comp in components) {
    prefix <- paste0(output_prefix, ".", comp)
    write_cs_table(empty_cs_df(), paste0(prefix, ".txt"), verbose = FALSE)
    write_cs_table(empty_cs_df(), paste0(prefix, ".robust.txt"), verbose = FALSE)
  }
  invisible(TRUE)
}

write_cs_table <- function(cs_df, outfile, verbose = TRUE) {
  if (nrow(cs_df) > 0) {
    if (is.list(cs_df$neg_log10_p_value)) {
      cs_df$neg_log10_p_value <- vapply(
        cs_df$neg_log10_p_value,
        function(x) toString(signif(as.numeric(x), 6)),
        character(1)
      )
    } else {
      cs_df$neg_log10_p_value <- signif(as.numeric(cs_df$neg_log10_p_value), 6)
    }
    
    if (is.list(cs_df$npc_outcome)) {
      cs_df$npc_outcome <- vapply(
        cs_df$npc_outcome,
        function(x) toString(signif(as.numeric(x), 6)),
        character(1)
      )
    } else {
      cs_df$npc_outcome <- signif(as.numeric(cs_df$npc_outcome), 6)
    }
    
    if ("vcp" %in% names(cs_df)) {
      cs_df$vcp <- signif(as.numeric(cs_df$vcp), 8)
    }
  }

  outdir <- dirname(outfile)
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  write.table(cs_df, file = outfile, sep = "\t", row.names = FALSE, quote = FALSE)
  if (isTRUE(verbose)) {
    cat("Wrote credible sets table to:", outfile, "\n")
  }
}

# --- CLI: one tissue × LD-block item ---

# Paper Methods constants (always applied; not CLI).
GENE_PADDING <- 1000000L
P_THRESHOLD <- 1e-5
PIP_THRESHOLD <- 0.05
MIN_LD_SNPS <- 2L
STRONG_ONLY <- TRUE
RUN_XQTL_ONLY <- FALSE
RUN_XQTL_IF_NO_GWAS <- TRUE
RUN_SEPARATE_GWAS <- FALSE
RUN_JOINT_GWAS <- TRUE

p <- arg_parser("LD-block ColocBoost worker") |>
  add_argument("--region_item", type = "character", help = "tissue__chr_start_end") |>
  add_argument("--genotype_stem", type = "character", help = "GTEx WGS PLINK stem") |>
  add_argument("--expression_pheno_dir", type = "character", help = "eQTL phenotype matrix dir") |>
  add_argument("--splicing_pheno_dir", type = "character", help = "sQTL phenotype matrix dir") |>
  add_argument("--expression_covar_dir", type = "character", help = "eQTL covariate dir") |>
  add_argument("--splicing_covar_dir", type = "character", help = "sQTL covariate dir") |>
  add_argument("--output_dir", type = "character", help = "Output directory") |>
  add_argument("--gwas_id_list", type = "character", default = "none", help = "GWAS trait id list") |>
  add_argument("--gwas_dir", type = "character", default = "none", help = "Imputed GWAS dir") |>
  add_argument("--gwas_meta", type = "character", default = "none", help = "GWAS metadata TSV") |>
  add_argument("--ld_meta", type = "character", default = "none", help = "ld_metadata.tsv") |>
  add_argument("--gwas_column_matching", type = "character", default = "none", help = "Sumstats column map yml") |>
  add_argument("--keep_gene_id_path", type = "character", default = "none", help = "Gene allowlist file") |>
  add_argument("--jobs_tsv", type = "character", default = "none", help = "Single-phenotype jobs TSV")
args <- parse_args(p)

opt_path <- function(x) {
  x <- normalize_opt(x)
  if (is.null(x) || identical(x, "none")) return(NULL)
  x
}

apply_r_thread_env()
load_coloc_libs()

region_item <- args$region_item
idx <- regexpr("__", region_item, fixed = TRUE)
if (idx < 1L) {
  stop("region_item must be tissue__chr_start_end, got: ", region_item)
}
tissue_id <- substr(region_item, 1L, idx - 1L)
rest <- substr(region_item, idx + 2L, nchar(region_item))
m_block <- regexec("^(chr[^_]+)_([0-9]+)_([0-9]+)$", rest)
mm <- regmatches(rest, m_block)[[1]]
if (length(mm) != 4L) stop("cannot parse block from region_item: ", region_item)
chrom <- mm[[2]]; var_start <- as.integer(mm[[3]]); var_end <- as.integer(mm[[4]])
block_item <- paste0(chrom, "_", var_start, "_", var_end)
gene_item <- block_item

keep_gene_id_path <- opt_path(args$keep_gene_id_path)
jobs_tsv <- opt_path(args$jobs_tsv)
gwas_id_list_all <- opt_path(args$gwas_id_list)
gwas_dir <- opt_path(args$gwas_dir)
gwas_meta <- opt_path(args$gwas_meta)
ld_meta <- opt_path(args$ld_meta)
gwas_column_matching <- opt_path(args$gwas_column_matching)

ablation_batch_mode <- !is.null(jobs_tsv)
if (ablation_batch_mode && !is.null(keep_gene_id_path)) {
  stop("single-phenotype batch cannot combine with --keep_gene_id_path")
}

gene_start <- max(1L, var_start - GENE_PADDING)
gene_end <- var_end + GENE_PADDING
variant_region <- paste0(chrom, ":", var_start, "-", var_end)
gene_region <- paste0(chrom, ":", gene_start, "-", gene_end)

# Degenerate LD tiles: skip early when bim has too few SNPs.
if (!is.null(ld_meta) && file.exists(ld_meta)) {
  chrom_num <- sub("^chr", "", chrom)
  meta <- read.delim(ld_meta, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
  hit <- meta[
    (meta[[1]] == chrom_num | meta[[1]] == paste0("chr", chrom_num)) &
      meta[[2]] == var_start & meta[[3]] == var_end,
    ,
    drop = FALSE
  ]
  if (nrow(hit) == 1L) {
    parts <- strsplit(as.character(hit[[4]][[1]]), ",", fixed = TRUE)[[1]]
    bim <- if (length(parts) >= 2L) parts[[2]] else ""
    snps <- 0L
    if (nzchar(bim) && file.exists(bim)) {
      snps <- length(readLines(bim, warn = FALSE))
    }
    if (snps < MIN_LD_SNPS) {
      timestamp_cat(
        "WARNING: Degenerate LD block ", variant_region, " has ", snps,
        " SNP(s); skipping.\n"
      )
      quit(status = 0)
    }
  }
}

working_dir <- file.path(tempdir(), paste0("coloc_ld_block_", Sys.getpid()))
dir.create(working_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
timestamp_cat("LD-block coloc: tissue=", tissue_id, " variant_region=", variant_region, " gene_region=", gene_region, "\n")

lists <- write_pecotmr_lists(
  tissue_id,
  args$expression_pheno_dir, args$splicing_pheno_dir,
  args$expression_covar_dir, args$splicing_covar_dir
)

keep_gene_ids <- if (!is.null(keep_gene_id_path)) readLines(keep_gene_id_path) else NULL
gwas_ids_all <- if (!is.null(gwas_id_list_all) && file.exists(gwas_id_list_all)) {
  Filter(nzchar, readLines(gwas_id_list_all))
} else {
  character(0)
}

run_one <- function(
  gene_item,
  phenotype_id = NULL,
  gwas_id_override = NULL,
  strong_only = STRONG_ONLY,
  run_xqtl_only = RUN_XQTL_ONLY,
  run_xqtl_if_no_gwas = RUN_XQTL_IF_NO_GWAS,
  run_joint_gwas = RUN_JOINT_GWAS,
  run_separate_gwas = RUN_SEPARATE_GWAS
) {
  gwas_ids <- character(0)
  has_gwas <- FALSE
  if (run_separate_gwas || run_joint_gwas) {
    if (!is.null(gwas_id_override) && nzchar(gwas_id_override) &&
        !(toupper(gwas_id_override) %in% c("JOINT", "*"))) {
      gwas_ids <- gwas_id_override
      has_gwas <- TRUE
    } else if (isTRUE(strong_only)) {
      gwas_ids <- filter_gwas_for_region(
        variant_region, gwas_ids_all, gwas_dir, P_THRESHOLD
      )
      has_gwas <- length(gwas_ids) > 0L
    } else {
      gwas_ids <- gwas_ids_all
      has_gwas <- length(gwas_ids) > 0L
    }
  }

  pheno_keep <- phenotype_id
  gene_keep <- if (is.null(pheno_keep)) keep_gene_ids else NULL

  do_xqtl <- run_xqtl_only || (run_xqtl_if_no_gwas && !has_gwas)
  if (do_xqtl) {
    timestamp_cat("xQTL-only: load + fit\n")
    rd <- load_region_data_block(
      gene_region = gene_region,
      variant_region = variant_region,
      genotype_stem = args$genotype_stem,
      phenotype_list = lists$phenotype_list,
      covariate_list = lists$covariate_list,
      conditions_list = lists$conditions_list,
      include_gwas = FALSE,
      phenotype_id = pheno_keep,
      keep_gene_ids = gene_keep,
      maf_cutoff = 0, mac_cutoff = 0, xvar_cutoff = 0, imiss_cutoff = 0
    )
    if (!is.null(rd)) {
      fit_colocboost_region(
        region_data = rd,
        output_dir = args$output_dir,
        tissue_id = tissue_id,
        gene_item = gene_item,
        mode = "xqtl",
        pip_threshold = PIP_THRESHOLD,
        maf_cutoff = 0,
        quit_on_focal_skip = TRUE
      )
    }
  }

  if (has_gwas) {
    timestamp_cat("GWAS: load + fit\n")
    gwas_paths <- file.path(gwas_dir, paste0("imputed_", gwas_ids, ".txt.gz"))
    rd <- load_region_data_block(
      gene_region = gene_region,
      variant_region = variant_region,
      genotype_stem = args$genotype_stem,
      phenotype_list = lists$phenotype_list,
      covariate_list = lists$covariate_list,
      conditions_list = lists$conditions_list,
      include_gwas = TRUE,
      gwas_ids = gwas_ids,
      gwas_paths = gwas_paths,
      gwas_meta = gwas_meta,
      ld_meta = ld_meta,
      gwas_column_matching = gwas_column_matching,
      phenotype_id = pheno_keep,
      keep_gene_ids = gene_keep,
      maf_cutoff = 0
    )
    if (!is.null(rd)) {
      fit_colocboost_region(
        region_data = rd,
        output_dir = args$output_dir,
        tissue_id = tissue_id,
        gene_item = gene_item,
        mode = "gwas",
        output_stem = "none",
        run_joint_gwas = run_joint_gwas,
        run_separate_gwas = run_separate_gwas,
        pip_threshold = PIP_THRESHOLD,
        maf_cutoff = 0,
        quit_on_focal_skip = FALSE
      )
    }
  }

  cleanup_stem <- paste0(tissue_id, ".", gene_item)
  unlink(list.files(
    args$output_dir,
    pattern = paste0("^", gsub("([.|()\\^{}+$*?]|\\[|\\])", "\\\\\\1", cleanup_stem), "\\.region_data"),
    full.names = TRUE
  ))
  unlink(list.files(
    args$output_dir,
    pattern = paste0("^", gsub("([.|()\\^{}+$*?]|\\[|\\])", "\\\\\\1", cleanup_stem), ".*\\.focal\\.txt$"),
    full.names = TRUE
  ))
}

if (ablation_batch_mode) {
  jobs <- read_jobs_tsv(jobs_tsv)
  # columns: job_item job_tag tissue_id region_id region_item phenotype_id gwas_id
  # job_tag is j + zero-padded 1-based jobs.tsv row index (e.g. j0000042).
  rows <- jobs[jobs[[5]] == region_item, , drop = FALSE]
  if (nrow(rows) == 0L) {
    timestamp_cat("Single-pheno: no jobs for ", region_item, "; skip.\n")
    unlink(working_dir, recursive = TRUE)
    quit(status = 0)
  }
  pairs <- data.frame(
    job_item = rows[[1]],
    job_tag = rows[[2]],
    phenotype_id = rows[[6]],
    gwas_id = rows[[7]],
    stringsAsFactors = FALSE
  )
  timestamp_cat("Single-pheno jobs batch: ", nrow(pairs), " pair(s)\n")
  for (i in seq_len(nrow(pairs))) {
    pheno <- pairs$phenotype_id[[i]]
    gwas <- pairs$gwas_id[[i]]
    tag <- pairs$job_tag[[i]]
    gi <- paste0(block_item, ".", tag)
    if (toupper(gwas) %in% c("JOINT", "*") || !nzchar(gwas)) {
      timestamp_cat("Single-pheno: phenotype=", pheno, " gwas=joint\n")
      run_one(gi, phenotype_id = pheno, gwas_id_override = NULL, strong_only = TRUE,
              run_xqtl_only = FALSE, run_joint_gwas = TRUE, run_separate_gwas = FALSE)
    } else {
      timestamp_cat("Single-pheno: phenotype=", pheno, " gwas=", gwas, "\n")
      run_one(gi, phenotype_id = pheno, gwas_id_override = gwas, strong_only = FALSE,
              run_xqtl_only = FALSE, run_joint_gwas = TRUE, run_separate_gwas = FALSE)
    }
  }
  unlink(working_dir, recursive = TRUE)
  quit(status = 0)
}

run_one(gene_item)
unlink(working_dir, recursive = TRUE)
