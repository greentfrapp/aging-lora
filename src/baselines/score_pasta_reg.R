#!/usr/bin/env Rscript
# Score Pasta REG model on per-(cohort × cell-type) pseudobulk profiles.
# Reads gene × donor TSV files written by pseudobulk_for_pasta.py, runs the
# Pasta filter + rank-norm + REG predictor, and writes per-donor predictions
# back as CSV. Aggregates per (cohort × cell_type) into a summary file.

suppressMessages({
  library(pasta)
  library(glmnet)  # required for predict.cv.glmnet dispatch (Pasta uses requireNamespace which doesn't expose the S3 method)
  library(data.table)
  library(magrittr)
})

# Robust resolution of the project root: walk up from the script's directory.
# Uses the first argument-style trick so it works under both `source()` and `Rscript`.
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  # Fallback when sourced
  if (!is.null(sys.frames()) && length(sys.frames()) > 0) {
    f <- sys.frame(1)
    if (!is.null(f$ofile)) return(dirname(normalizePath(f$ofile)))
  }
  return(getwd())
}
PROJ_ROOT <- normalizePath(file.path(get_script_dir(), "..", ".."))
PSEUDOBULK_DIR <- file.path(PROJ_ROOT, "results", "baselines", "pasta_pseudobulk")
PER_DONOR_DIR <- file.path(PROJ_ROOT, "results", "baselines", "pasta_per_donor")
SUMMARY_PATH <- file.path(PROJ_ROOT, "results", "baselines", "pasta_loco_summary.csv")
dir.create(PER_DONOR_DIR, showWarnings = FALSE, recursive = TRUE)

cohorts <- c("onek1k", "stephenson", "terekhova", "aida")
ct_codes <- c("CD4T", "CD8T", "MONO", "NK", "B")

# Load the REG cv.glmnet model once into the script's global environment.
# Avoids repeated data() calls inside the loop.
data(cvfit_REG, package = "pasta")
stopifnot(inherits(cvfit_REG, "cv.glmnet"))

summary_rows <- list()
for (cohort in cohorts) {
  for (ct in ct_codes) {
    pb_path <- file.path(PSEUDOBULK_DIR, paste0(cohort, "_", ct, ".tsv"))
    meta_path <- file.path(PSEUDOBULK_DIR, paste0(cohort, "_", ct, "_meta.csv"))
    if (!file.exists(pb_path) || !file.exists(meta_path)) {
      cat(sprintf("[skip] %s x %s -- missing input\n", cohort, ct))
      next
    }
    cat(sprintf("=== %s x %s ===\n", cohort, ct))
    # Read gene x donor matrix
    pb <- fread(pb_path, sep = "\t", header = TRUE, data.table = FALSE)
    rownames(pb) <- pb[[1]]
    pb <- as.matrix(pb[, -1, drop = FALSE])
    cat(sprintf("  pseudobulk: %d genes x %d donors\n", nrow(pb), ncol(pb)))

    # Read donor metadata
    meta <- fread(meta_path, data.table = FALSE)
    stopifnot("donor_id" %in% colnames(meta))
    stopifnot(nrow(meta) == ncol(pb))
    # Make sure donor order matches matrix columns
    stopifnot(all(as.character(meta$donor_id) == colnames(pb)))

    # Filter to Pasta panel + rank-normalize per donor
    pb_filt <- filtering_age_model_genes_and_rank_norm(pb)
    cat(sprintf("  after filter+rank: %d genes x %d donors\n", nrow(pb_filt), ncol(pb_filt)))

    # Pasta expects sample x gene matrix for prediction (mat_t = transposed)
    mat_t <- t(pb_filt)
    # Direct predict.cv.glmnet dispatch — pasta's predicting_age_score uses
    # stats::predict + requireNamespace("glmnet"), which can fail to dispatch
    # to predict.cv.glmnet under Rscript. Loading the model + calling predict
    # directly with library(glmnet) attached is robust.
    pred_mat <- predict(cvfit_REG, newx = mat_t, s = "lambda.min", type = "link")
    pred <- as.numeric(pred_mat[, 1])
    cat(sprintf("  predict: returned %d age scores (range %.1f-%.1f)\n", length(pred), min(pred), max(pred)))
    stopifnot(length(pred) == nrow(mat_t))

    # Per-donor output
    per_donor <- data.frame(
      donor_id = meta$donor_id,
      true_age = meta$age,
      predicted_age = as.numeric(pred),
      n_cells_aggregated = meta$n_cells_aggregated
    )
    per_donor_path <- file.path(PER_DONOR_DIR, paste0(cohort, "_", ct, ".csv"))
    write.csv(per_donor, per_donor_path, row.names = FALSE)

    # Metrics
    valid <- !is.na(per_donor$true_age) & !is.na(per_donor$predicted_age)
    err <- per_donor$predicted_age[valid] - per_donor$true_age[valid]
    abs_err <- abs(err)
    if (sum(valid) >= 3 && sd(per_donor$predicted_age[valid]) > 0 && sd(per_donor$true_age[valid]) > 0) {
      ct_obj <- cor.test(per_donor$predicted_age[valid], per_donor$true_age[valid])
      r_val <- ct_obj$estimate
      p_val <- ct_obj$p.value
    } else {
      r_val <- NA_real_
      p_val <- NA_real_
    }
    summary_rows[[paste(cohort, ct)]] <- list(
      baseline = "Pasta-REG",
      training_cohorts = "Pasta-pretraining",
      eval_cohort = cohort,
      cell_type = ct,
      n_donors = sum(valid),
      median_abs_err_yr = median(abs_err),
      mean_abs_err_yr = mean(abs_err),
      pearson_r = unname(r_val),
      pearson_p = p_val,
      mean_bias_yr = mean(err)
    )
    cat(sprintf("  n=%d MAE=%.2fy R=%.3f p=%.2e bias=%+.2fy\n",
                sum(valid), median(abs_err), unname(r_val), p_val, mean(err)))
  }
}

if (length(summary_rows) > 0) {
  summary_df <- rbindlist(summary_rows)
  fwrite(summary_df, SUMMARY_PATH)
  cat(sprintf("wrote summary -> %s\n", SUMMARY_PATH))
} else {
  cat("no slices scored\n")
}
