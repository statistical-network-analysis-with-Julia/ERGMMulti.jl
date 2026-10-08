# Golden fixture: statnet `ergm.multi` MCMLE of a DYAD-DEPENDENT multilayer ERGM.
#
# Regenerate from the package root (a few minutes: nine default fits and one
# long-chain fit, on 3 cores):
#
#   Rscript test/fixtures/r/multilayer_mcmle.R > test/fixtures/multilayer_mcmle.toml
#
# WHAT THIS FIXTURE PINS
#
# ERGMMulti.jl's `method=:mcmle` against ergm.multi's default estimator for a
# dyad-dependent model, which is MCMLE. Both sides carry Monte-Carlo error, so
# the comparison is made at the resolution R itself has: the model is refit
# under nine seeds and the seed-to-seed mean and standard deviation of every
# coefficient and standard error are frozen; the Julia testset compares its
# fits with R's mean inside a band derived from that spread (see [tolerance]).
#
# The data are two seeded 16-actor DIRECTED layers with built-in reciprocity
# in layer A and overlap between the layers (frozen as edge lists), so the
# dyad-dependent coefficients are far from zero and the MPLE and the MLE are
# different numbers. The model is
#
#   Layer(list(A = a, B = b)) ~ L(~edges, ~A) + L(~edges, ~B) + L(~mutual, ~A) +
#                               L(~edges, ~A & B) + L(~gwesp(0.25, fixed=TRUE), ~B)
#
# = [LayerEdges(1), LayerEdges(2), LayerMutual(1), InterlayerDependence(1, 2),
#    WithinLayer(GWESP(0.25), 2)] in ERGMMulti.jl.
#
# A tenth, long-chain fit (`hp_*`: MCMC.samplesize 16384, interval 4096) is
# frozen as the best available value of the MLE; the script checks that the
# default fits' mean agrees with it. The MPLE of the same model is frozen too
# (`mple_coefficients`), to show the two estimators apart. R's log-likelihood
# (bridge sampling) is frozen with its seed-to-seed spread.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(network)
  library(ergm.multi)
  library(parallel)
})

seed <- 20261002
set.seed(seed)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
ints <- function(x) paste(sprintf("%d", as.integer(x)), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")

n <- 16L
base <- matrix(rbinom(n * n, 1, 0.10), n, n)
symm <- matrix(rbinom(n * n, 1, 0.12), n, n)
symm[lower.tri(symm)] <- t(symm)[lower.tri(symm)]      # reciprocated ties
A <- pmin(base + symm, 1)
keep <- matrix(rbinom(n * n, 1, 0.55), n, n)
extra <- matrix(rbinom(n * n, 1, 0.08), n, n)
B <- pmin(A * keep + extra, 1)                         # overlaps layer A
diag(A) <- 0
diag(B) <- 0
na <- network(A, directed = TRUE)
nb <- network(B, directed = TRUE)
lnw <- Layer(list(A = na, B = nb))

f <- lnw ~ L(~edges, ~A) + L(~edges, ~B) + L(~mutual, ~A) + L(~edges, ~A & B) +
  L(~gwesp(0.25, fixed = TRUE), ~B)
julia_terms <- c("LayerEdges(1)", "LayerEdges(2)", "LayerMutual(1)",
                 "InterlayerDependence(1, 2)", "WithinLayer(GWESP(0.25), 2)")

fit_one <- function(s, ctrl = control.ergm(seed = s)) {
  set.seed(s)
  fit <- NULL
  invisible(capture.output(
    fit <- suppressWarnings(suppressMessages(ergm(f, control = ctrl))),
    type = "output"))
  ll <- NULL
  invisible(capture.output(ll <- suppressMessages(as.numeric(logLik(fit))),
                           type = "output"))
  list(coef = as.numeric(coef(fit)), se = as.numeric(sqrt(diag(vcov(fit)))),
       loglik = ll, names = names(coef(fit)), failed = isTRUE(fit$failure))
}

seeds <- 1:9
fits <- mclapply(seeds, fit_one, mc.cores = 3)
hp <- fit_one(4242, control.ergm(seed = 4242, MCMC.samplesize = 16384,
                                 MCMC.interval = 4096, MCMLE.maxit = 60))
stopifnot(!any(sapply(fits, `[[`, "failed")), !hp$failed)
coefs <- t(sapply(fits, `[[`, "coef"))
ses <- t(sapply(fits, `[[`, "se"))
lls <- sapply(fits, `[[`, "loglik")
coef_mean <- colMeans(coefs); coef_sd <- apply(coefs, 2, sd)
se_mean <- colMeans(ses); se_sd <- apply(ses, 2, sd)
stopifnot(all(abs(coef_mean - hp$coef) <= 4 * coef_sd * sqrt(1 / length(seeds) + 1)))

mple <- NULL
invisible(capture.output(mple <- suppressMessages(ergm(f, estimate = "MPLE")),
                         type = "output"))
mple_coef <- as.numeric(coef(mple))
# The same pseudo-likelihood taken to convergence (glm epsilon 1e-14) on R's
# OWN compressed design, `ergmMPLE(output = "matrix")`: ergm stops its MPLE
# glm at the default epsilon 1e-8, so the as-shipped MPLE is compared at a
# tolerance set from the measured slack, and the exact refit at 1e-6. (No
# random draw here: the values above are unchanged by this block.)
mm <- NULL
invisible(capture.output(mm <- suppressMessages(ergmMPLE(f, output = "matrix")),
                         type = "output"))
stopifnot(sum(mm$weights) == 2 * n * (n - 1))
g_exact <- glm.fit(mm$predictor, mm$response, weights = mm$weights,
                   family = binomial(), control = glm.control(epsilon = 1e-14, maxit = 200))
exact_mple_coef <- as.numeric(g_exact$coefficients)
mple_vs_exact_coef <- max(abs(mple_coef - exact_mple_coef))
# The as-shipped tolerance: 20x the measured glm slack d, rounded up to a
# power of ten. The 1e-12 floor on d leaves room for the last digits of
# either optimizer when R's glm happens to stop on the optimum itself; there
# is no floor on the tolerance, so it is always set by the measurement.
tol_from <- function(d) sprintf("%.1g", 10^ceiling(log10(20 * max(d, 1e-12))))
stats <- summary(f)
ai <- which(A > 0, arr.ind = TRUE)
bi <- which(B > 0, arr.ind = TRUE)

cat('name = "multilayer_mcmle"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_multi_version = "%s"\n', as.character(packageVersion("ergm.multi"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/multilayer_mcmle.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('model = "Layer(list(A = a, B = b)) ~ L(~edges,~A) + L(~edges,~B) + L(~mutual,~A) + L(~edges,~A&B) + L(~gwesp(0.25,fixed=TRUE),~B) on two seeded 16-actor directed layers (reciprocity in A, overlap between A and B), frozen as edge lists"\n')
cat('estimator = "ergm.multi MCMLE at control.ergm defaults, nine seeds (1..9); hp_* = one fit at MCMC.samplesize=16384, MCMC.interval=4096 (seed 4242); mple_* = estimate=\\"MPLE\\""\n')
cat("\n")

cat("[tolerance]\n")
cat("# Observed statistics are deterministic functions of the layers.\n")
cat("summary_statistics = 1e-10\n")
cat("#\n")
cat("# Both estimators are Monte-Carlo MLEs of the same likelihood, so they are\n")
cat("# compared at R's own resolution. For coefficient k a Julia fit must lie\n")
cat("# within  band_sd * sqrt(sd_k^2 + sd_k^2 / n_seeds)  of R's nine-seed mean,\n")
cat("# where sd_k is R's seed-to-seed standard deviation (`coefficients_sd`):\n")
cat("# the first term is one fit's own Monte-Carlo error (taken equal to R's),\n")
cat("# the second the error of R's mean. The band is floored at\n")
cat("# `floor_se_fraction` of R's mean standard error, so that a coefficient\n")
cat("# whose nine R fits happened to agree unusually closely does not demand more\n")
cat("# than either sampler's O(1/ESS) bias allows. Standard errors likewise,\n")
cat("# against `std_errors_sd`, with the same floor.\n")
cat("band_sd = 4.0\n")
cat("floor_se_fraction = 0.1\n")
cat("# The log-likelihood (bridge / path sampling on both sides): within\n")
cat("# band_sd * sqrt(2) * R's seed-to-seed sd, floored at 1.0 log unit.\n")
cat("loglik_floor = 1.0\n")
cat("# The MPLE is a deterministic function of the change-statistic design:\n")
cat("# the exact refit (glm epsilon 1e-14 on ergmMPLE's design) at 1e-6, the\n")
cat("# as-shipped MPLE at 20x the measured glm slack (mple_vs_exact_coefficients),\n")
cat("# rounded up to a power of ten (no floor). DO NOT LOOSEN.\n")
cat("exact_mple_coefficients = 1e-6\n")
cat(sprintf("mple_coefficients = %s\n", tol_from(mple_vs_exact_coef)))
cat("\n")

cat("[values]\n")
cat(sprintf("n_actors = %d\n", n))
cat("n_layers = 2\n")
cat("directed = true\n")
cat(sprintf("layer1_src = [%s]\n", ints(ai[, 1])))
cat(sprintf("layer1_dst = [%s]\n", ints(ai[, 2])))
cat(sprintf("layer2_src = [%s]\n", ints(bi[, 1])))
cat(sprintf("layer2_dst = [%s]\n", ints(bi[, 2])))
cat(sprintf("julia_terms = [%s]\n", strs(julia_terms)))
cat(sprintf("term_names = [%s]\n", strs(fits[[1]]$names)))
cat(sprintf("summary_statistics = [%s]\n", num(as.numeric(stats))))
cat(sprintf("seeds = [%s]\n", ints(seeds)))
for (k in seq_along(seeds)) {
  cat(sprintf("coefficients_seed%d = [%s]\n", seeds[k], num(coefs[k, ])))
  cat(sprintf("std_errors_seed%d = [%s]\n", seeds[k], num(ses[k, ])))
}
cat(sprintf("coefficients_mean = [%s]\n", num(coef_mean)))
cat(sprintf("coefficients_sd = [%s]\n", num(coef_sd)))
cat(sprintf("std_errors_mean = [%s]\n", num(se_mean)))
cat(sprintf("std_errors_sd = [%s]\n", num(se_sd)))
cat(sprintf("logliks = [%s]\n", num(lls)))
cat(sprintf("loglik_mean = %.17g\n", mean(lls)))
cat(sprintf("loglik_sd = %.17g\n", sd(lls)))
cat(sprintf("hp_coefficients = [%s]\n", num(hp$coef)))
cat(sprintf("hp_std_errors = [%s]\n", num(hp$se)))
cat(sprintf("hp_loglik = %.17g\n", hp$loglik))
cat(sprintf("mple_coefficients = [%s]\n", num(mple_coef)))
cat(sprintf("exact_mple_coefficients = [%s]\n", num(exact_mple_coef)))
cat(sprintf("mple_vs_exact_coefficients = %.17g\n", mple_vs_exact_coef))
