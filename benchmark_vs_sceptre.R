# Benchmark the reference SCEPTRE R package on its own example dataset.
#
# Times a calibration check (1 CPU core, then 2 cores) on
# sceptredata::highmoi_example_data and writes the timings to
# sceptre_timing.txt for benchmark_vs_sceptre.py to compare against.
#
# Requires the sceptre + sceptredata packages (see the README section
# "Benchmark vs SCEPTRE" for a setup that installs them).
#
#   Rscript benchmark_vs_sceptre.R

suppressMessages({library(sceptre); library(sceptredata)})
data(highmoi_example_data, package = "sceptredata")
data(grna_target_data_frame_highmoi, package = "sceptredata")

N_PAIRS <- 1000   # calibration pairs to test (the matched workload)

build_object <- function() {
  so <- import_data(
    response_matrix        = highmoi_example_data$response_matrix,
    grna_matrix            = highmoi_example_data$grna_matrix,
    extra_covariates       = highmoi_example_data$extra_covariates,
    grna_target_data_frame = grna_target_data_frame_highmoi,
    moi                    = "high")
  so <- set_analysis_parameters(so)
  so <- assign_grnas(so, method = "thresholding")
  run_qc(so)
}

time_calibration <- function(parallel, n_processors) {
  so <- build_object()
  t0 <- Sys.time()
  so <- run_calibration_check(so,
                              n_calibration_pairs    = N_PAIRS,
                              calibration_group_size = 1,
                              parallel               = parallel,
                              n_processors           = n_processors)
  as.numeric(difftime(Sys.time(), t0, units = "secs"))
}

t_1core <- time_calibration(FALSE, 1)
# 2-core run; n_processors is set explicitly because detectCores() is unreliable
# inside some containers (e.g. Colab). NA if the parallel backend is unavailable.
t_2core <- tryCatch(time_calibration(TRUE, 2), error = function(e) NA_real_)

rm_ <- highmoi_example_data$response_matrix
cat(sprintf("SCEPTRE  1-core: %.1fs | 2-core: %s | %d pairs | %d genes x %d cells\n",
            t_1core, ifelse(is.na(t_2core), "NA", sprintf("%.1fs", t_2core)),
            N_PAIRS, nrow(rm_), ncol(rm_)))
writeLines(as.character(c(t_1core, t_2core, N_PAIRS, nrow(rm_), ncol(rm_))),
           "sceptre_timing.txt")
