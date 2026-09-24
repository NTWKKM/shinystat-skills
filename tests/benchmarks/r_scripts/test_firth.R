# File: tests/benchmarks/r_scripts/test_firth.R
# Purpose: Generate benchmark data for validating Python's jzluo/firthmodels
# Target Python Lib: https://github.com/jzluo/firthmodels

# -----------------------------------------------------------------------------
# 1. Setup & Dependencies
# -----------------------------------------------------------------------------
required_packages <- c("logistf", "survival", "coxphf", "dplyr")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(missing_packages)) {
    stop(paste(
        "Missing required R packages:", paste(missing_packages, collapse = ", "),
        ". Please install them manually or via CI configuration."
    ))
}

library(logistf)
library(survival)
library(coxphf)
library(dplyr)

# Output directory for CSV benchmarks
output_dir <- "tests/benchmarks/python_results"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# -----------------------------------------------------------------------------
# 2. Benchmark A: Firth Logistic Regression (Binary Outcome)
#    Target: Validate FirthLogisticRegression()
# -----------------------------------------------------------------------------
cat("\n--- Generating Firth Logistic Regression Benchmark ---\n")

# Use 'sex2' dataset from logistf (contains separation)
data(sex2)

# Fit Firth model using logistf
# Default: pl = TRUE (Profile Likelihood CIs), firth = TRUE
fit_firth <- logistf(case ~ age + oc + vic + vicl + vis + dia, data = sex2)

# Extract results strictly
results_firth <- data.frame(
    term = names(fit_firth$coefficients),
    estimate = as.numeric(fit_firth$coefficients),
    # logistf uses Profile Likelihood for CIs by default
    conf.low = fit_firth$ci.lower,
    conf.high = fit_firth$ci.upper,
    p.value = fit_firth$prob,
    stringsAsFactors = FALSE
)

# Export Benchmark
write.csv(results_firth, file.path(output_dir, "benchmark_firth_logistic.csv"), row.names = FALSE)
# Export Dataset for Python to use
write.csv(sex2, file.path(output_dir, "dataset_sex2.csv"), row.names = FALSE)

cat("✅ benchmark_firth_logistic.csv created.\n")

# -----------------------------------------------------------------------------
# 3. Benchmark B: Penalized Cox Regression (Survival)
#    Target: Validate FirthCoxPH()
# -----------------------------------------------------------------------------
cat("\n--- Generating Penalized Cox Regression Benchmark ---\n")

# Use 'breast' dataset from coxphf
data(breast)

# Fit Penalized Cox model
# Note: coxphf implements Firth's penalized partial likelihood
fit_cox_firth <- coxphf(Surv(TIME, CENS) ~ T + N + G + CD, data = breast)

# Extract results
results_cox_firth <- data.frame(
    term = names(fit_cox_firth$coefficients),
    estimate = as.numeric(fit_cox_firth$coefficients),
    conf.low = fit_cox_firth$ci.lower,
    conf.high = fit_cox_firth$ci.upper,
    p.value = fit_cox_firth$prob,
    stringsAsFactors = FALSE
)

# Export Benchmark
write.csv(results_cox_firth, file.path(output_dir, "benchmark_firth_cox.csv"), row.names = FALSE)
# Export Dataset for Python
write.csv(breast, file.path(output_dir, "dataset_breast.csv"), row.names = FALSE)

cat("✅ benchmark_firth_cox.csv created.\n")
cat("\n🚀 All Firth benchmarks generated successfully.\n")
