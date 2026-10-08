# Golden fixture: statistics at the BOUNDARY of their attainable range in
# statnet `ergm.multi`, and R's drop as ergm.multi users can write it.
#
# Regenerate from the package root (about a minute: thirty short MCMLE fits):
#
#   Rscript test/fixtures/r/boundary_multi.R > test/fixtures/boundary_multi.toml
#
# WHAT THIS FIXTURE PINS
#
# A statistic whose observed value is an end of its attainable range has no
# finite maximum (pseudo-)likelihood estimate. R ergm's default `drop=TRUE`
# fixes its coefficient at -Inf (+Inf) and estimates the rest with it held
# there. ergm.multi 0.3.0 does NOT do this for layer terms (its `L()`
# operator does not pass the term's attainable range on to
# `ergm.checkextreme.model`): at its defaults it reports NA (a statistic whose
# change statistics are all zero, "not varying") or a finite value where its
# estimation stopped. ERGMMulti.jl applies R ergm's drop. The drop is a model
# ergm.multi CAN fit when it is written out: `offset(L(...))` with
# `offset.coef = -Inf` holds the statistic at its bound in both the MPLE and
# the MCMC sampler. So each case freezes
#
#   * the observed statistics and their labels;
#   * what ergm.multi does at its defaults (the documented divergence): the
#     MPLE's finite coefficients and whether the boundary one is NA, and the
#     default MCMLE's coefficient for the boundary statistic;
#   * the MPLE's exact limit (the pseudo-likelihood on the dyads the dropped
#     statistic does not touch, `glm` epsilon 1e-14 on `ergmMPLE`'s own
#     design): ERGMMulti.jl's MPLE must equal it;
#   * the MLE with the statistic held at its bound — the offset fit at
#     `control.ergm` defaults under ten seeds: mean and seed-to-seed sd of
#     every finite coefficient and standard error (and of the log-likelihood
#     where R computes one) — which ERGMMulti.jl's default MCMLE (`method =
#     :auto`, which drops) must reproduce within a band set by R's own spread.
#
# THE CASES (data seeded below and frozen as edge lists)
#
# (a) undirected, 10 actors. Layer A is a perfect matching (5 ties, no
#     two-path), layer B a Bernoulli(0.3) graph. Model
#       L(~edges,~A) + L(~edges,~B) + L(~triangle,~A)
#     The triangle count is 0 and every one of its change statistics is 0 as
#     well, so the pseudo-likelihood design cannot see the bound (the
#     ERGMMulti.jl MPLE used to stop at 0 for every coefficient); only the
#     attainable range can.
# (b) directed, 6 actors. Layer A a 6-cycle plus the reverse of one tie,
#     layer B two disjoint 3-cycles, tied on no dyad of A. Model
#       L(~edges,~A) + L(~edges,~B) + L(~edges,~A&B) + L(~mutual,~A)
#     `L(~edges,~A&B)` is 0; the design shows the bound (one-signed change
#     statistics).
# (c) directed, 8 actors with a two-level attribute g; layer A has ties only
#     between the levels, some of them reciprocated, layer B is
#     Bernoulli(0.25). Model
#       L(~edges,~A) + L(~edges,~B) + L(~nodematch("g"),~A) + L(~mutual,~A)
#     The dropped `nodematch` is dyad-INDEPENDENT, so the log-likelihood of
#     the model with it held is computed on both sides (path sampling).
#
# TOLERANCES
#
# Statistics are integers: 1e-10. The MPLE's exact limit is a deterministic
# function of the design, fitted to convergence on both sides: 1e-8. For
# the MCMLE, both sides are Monte-Carlo MLEs: a Julia fit must lie within
# band_sd * sd_R * sqrt(1 + 1/n_seeds) of R's ten-seed mean, the first term
# one fit's own Monte-Carlo error (taken equal to R's) and the second the
# error of R's mean; band_sd is 4, as in multilayer_mcmle.toml. Taking the
# Julia fit's Monte-Carlo error equal to R's is checked on the Julia side:
# the testset asserts each fit's own Monte-Carlo standard error is at most
# 10 % of its standard error, and R's seed-to-seed sd is recorded here
# (`*_coef_sd`) next to R's mean standard error (`*_se_mean`).

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(network)
  library(ergm.multi)
})

seed <- 20261007
set.seed(seed)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
ints <- function(x) paste(sprintf("%d", as.integer(x)), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
bool <- function(x) if (isTRUE(x)) "true" else "false"

quiet <- function(expr) {
  out <- NULL
  invisible(capture.output(out <- suppressMessages(expr), type = "output"))
  out
}

# Warnings of an expression, as text, and its value
with_warnings <- function(expr) {
  w <- character(0)
  val <- withCallingHandlers(quiet(expr), warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = val, warnings = w)
}

# ---- data ----------------------------------------------------------------
na <- 10L
Aa <- network.initialize(na, directed = FALSE)
for (k in seq(1, na, 2)) Aa[k, k + 1] <- 1
Ba <- network.initialize(na, directed = FALSE)
for (i in 1:(na - 1)) for (j in (i + 1):na) if (runif(1) < 0.3) Ba[i, j] <- 1
lna <- Layer(list(A = Aa, B = Ba))

nb <- 6L
Ab <- network.initialize(nb, directed = TRUE)
for (e in list(c(1, 2), c(2, 1), c(2, 3), c(3, 4), c(4, 5), c(5, 6), c(6, 1))) Ab[e[1], e[2]] <- 1
Bb <- network.initialize(nb, directed = TRUE)
for (e in list(c(1, 3), c(3, 5), c(5, 1), c(2, 4), c(4, 6), c(6, 2))) Bb[e[1], e[2]] <- 1
lnb <- Layer(list(A = Ab, B = Bb))

nc <- 8L
gc <- rep(c("x", "y"), length.out = nc)
Ac <- network.initialize(nc, directed = TRUE)
Bc <- network.initialize(nc, directed = TRUE)
set.vertex.attribute(Ac, "g", gc)
set.vertex.attribute(Bc, "g", gc)
for (i in 1:(nc - 1)) for (j in (i + 1):nc) {
  if (gc[i] != gc[j] && runif(1) < 0.45) {       # a tie between the levels,
    if (runif(1) < 0.5) Ac[i, j] <- 1 else Ac[j, i] <- 1
    if (runif(1) < 0.4) Ac[i, j] <- Ac[j, i] <- 1  # reciprocated 40 % of the time
  }
}
for (i in 1:nc) for (j in 1:nc) if (i != j && runif(1) < 0.25) Bc[i, j] <- 1
lnc <- Layer(list(A = Ac, B = Bc))

cases <- list(
  a = list(lnw = lna, n = na, directed = FALSE, drop = 3L,
           f = lna ~ L(~edges, ~A) + L(~edges, ~B) + L(~triangle, ~A),
           fo = lna ~ L(~edges, ~A) + L(~edges, ~B) + offset(L(~triangle, ~A)),
           julia = c("LayerEdges(1)", "LayerEdges(2)", "LayerTriangle(1)")),
  b = list(lnw = lnb, n = nb, directed = TRUE, drop = 3L,
           f = lnb ~ L(~edges, ~A) + L(~edges, ~B) + L(~edges, ~A & B) + L(~mutual, ~A),
           fo = lnb ~ L(~edges, ~A) + L(~edges, ~B) + offset(L(~edges, ~A & B)) +
             L(~mutual, ~A),
           julia = c("LayerEdges(1)", "LayerEdges(2)", "InterlayerDependence(1, 2)",
                     "LayerMutual(1)")),
  c = list(lnw = lnc, n = nc, directed = TRUE, drop = 3L,
           f = lnc ~ L(~edges, ~A) + L(~edges, ~B) + L(~nodematch("g"), ~A) +
             L(~mutual, ~A),
           fo = lnc ~ L(~edges, ~A) + L(~edges, ~B) + offset(L(~nodematch("g"), ~A)) +
             L(~mutual, ~A),
           julia = c("LayerEdges(1)", "LayerEdges(2)", "WithinLayer(NodeMatch(:g), 1)",
                     "LayerMutual(1)"))
)

seeds <- 1:10
res <- list()
for (key in names(cases)) {
  cs <- cases[[key]]
  st <- summary(cs$f)
  k <- cs$drop
  stopifnot(st[k] == 0)

  # ergm.multi at its defaults: the MPLE, and the MCMLE (seed 1)
  mp <- with_warnings(ergm(cs$f, estimate = "MPLE"))
  mple_coef <- as.numeric(coef(mp$value))
  mc <- with_warnings(ergm(cs$f, control = control.ergm(seed = 1)))
  mcmle_coef <- as.numeric(coef(mc$value))

  # The MPLE's exact limit: the pseudo-likelihood on the dyads the boundary
  # statistic does not touch, without it, on ergmMPLE's own design
  mm <- quiet(ergmMPLE(cs$f, output = "matrix"))
  untouched <- mm$predictor[, k] == 0
  g_exact <- glm.fit(mm$predictor[untouched, -k, drop = FALSE], mm$response[untouched],
                     weights = mm$weights[untouched], family = binomial(),
                     control = glm.control(epsilon = 1e-14, maxit = 200))
  stopifnot(g_exact$converged)

  # R's drop written out: the offset fit, ten seeds
  fits <- lapply(seeds, function(s) {
    fit <- suppressWarnings(quiet(ergm(cs$fo, offset.coef = -Inf,
                                       control = control.ergm(seed = s))))
    ll <- suppressWarnings(quiet(as.numeric(logLik(fit))))
    list(coef = as.numeric(coef(fit))[-k], se = as.numeric(sqrt(diag(vcov(fit))))[-k],
         loglik = ll, failed = isTRUE(fit$failure), off = as.numeric(coef(fit))[k])
  })
  stopifnot(!any(sapply(fits, `[[`, "failed")), all(sapply(fits, `[[`, "off") == -Inf))
  coefs <- t(sapply(fits, `[[`, "coef"))
  ses <- t(sapply(fits, `[[`, "se"))
  lls <- sapply(fits, `[[`, "loglik")
  res[[key]] <- list(
    stats = st, names = names(st),
    mple_na = is.na(mple_coef[k]),
    mple_not_varying = any(grepl("not varying", mp$warnings)),
    mple_not_exist = any(grepl("MPLE does not exist", mp$warnings)),
    mple_finite = mple_coef[-k], mple_boundary = mple_coef[k],
    mcmle_boundary = mcmle_coef[k],
    exact_limit = as.numeric(g_exact$coefficients),
    exact_limit_dyads = sum(mm$weights[untouched]),
    coef_mean = colMeans(coefs), coef_sd = apply(coefs, 2, sd),
    se_mean = colMeans(ses), se_sd = apply(ses, 2, sd),
    loglik_mean = mean(lls), loglik_sd = sd(lls))
}

edgelist <- function(net) {
  el <- as.edgelist(net)
  list(src = el[, 1], dst = el[, 2])
}

cat('name = "boundary_multi"\n\n')
cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_multi_version = "%s"\n', as.character(packageVersion("ergm.multi"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/boundary_multi.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('model = "three designs with one statistic at the bottom of its attainable range: (a) L(~triangle,~A) on an undirected perfect matching (all change statistics zero), (b) L(~edges,~A&B) on directed layers tied on no common dyad, (c) a dyad-independent L(~nodematch(\\"g\\"),~A) with no within-level tie"\n')
cat('estimator = "ergm.multi defaults (MPLE; MCMLE seed 1); the exact limit of the MPLE (glm epsilon 1e-14 on ergmMPLE\'s design, rows untouched by the boundary statistic); the drop written as offset(...) with offset.coef = -Inf, MCMLE at control.ergm defaults under seeds 1..10"\n')

cat("\n[tolerance]\n")
cat("# Observed statistics are integers.\n")
for (key in names(cases)) cat(sprintf("%s_statistics = 1e-10\n", key))
cat("# The MPLE's exact limit: deterministic, fitted to convergence on both sides.\n")
for (key in names(cases)) cat(sprintf("%s_exact_limit = 1e-8\n", key))
cat("# MCMLE: within band_sd * sd_R * sqrt(1 + 1/n_seeds) of R's ten-seed mean\n")
cat("# (coefficients against *_coef_sd, standard errors against *_se_sd, the\n")
cat("# log-likelihood of case (c) against loglik_sd with sqrt(2) for both sides).\n")
cat("# R also reports a log-likelihood for (a) and (b), whose held statistic is\n")
cat("# dyad-dependent; ERGMMulti.jl reports NaN there, as ERGM.jl does (its path\n")
cat("# sampler starts from the dyad-independent part of the model, which cannot\n")
cat("# hold such a statistic at its bound), so those two are not compared.\n")
cat("band_sd = 4.0\n")

cat("\n[values]\n")
cat(sprintf("seeds = [%s]\n", ints(seeds)))
for (key in names(cases)) {
  cs <- cases[[key]]
  r <- res[[key]]
  cat(sprintf("\n# --- case (%s) ---\n", key))
  cat(sprintf("%s_n_actors = %d\n", key, cs$n))
  cat(sprintf("%s_directed = %s\n", key, bool(cs$directed)))
  for (l in c("A", "B")) {
    el <- edgelist(get(paste0(l, key)))
    cat(sprintf("%s_layer_%s_src = [%s]\n", key, l, ints(el$src)))
    cat(sprintf("%s_layer_%s_dst = [%s]\n", key, l, ints(el$dst)))
  }
  if (key == "c") cat(sprintf("c_g = [%s]\n", strs(gc)))
  cat(sprintf("%s_julia_terms = [%s]\n", key, strs(cs$julia)))
  cat(sprintf("%s_stat_names = [%s]\n", key, strs(r$names)))
  cat(sprintf("%s_statistics = [%s]\n", key, num(as.numeric(r$stats))))
  cat(sprintf("%s_boundary_index = %d\n", key, cs$drop))
  cat(sprintf("%s_r_mple_boundary_na = %s\n", key, bool(r$mple_na)))
  cat(sprintf("%s_r_mple_not_varying_warning = %s\n", key, bool(r$mple_not_varying)))
  cat(sprintf("%s_r_mple_not_exist_warning = %s\n", key, bool(r$mple_not_exist)))
  cat(sprintf("%s_r_mple_finite = [%s]\n", key, num(r$mple_finite)))
  if (!r$mple_na) cat(sprintf("%s_r_mple_boundary = %s\n", key, num(r$mple_boundary)))
  cat(sprintf("%s_r_mcmle_boundary = %s\n", key, num(r$mcmle_boundary)))
  cat(sprintf("%s_exact_limit = [%s]\n", key, num(r$exact_limit)))
  cat(sprintf("%s_exact_limit_dyads = %s\n", key, num(r$exact_limit_dyads)))
  cat(sprintf("%s_coef_mean = [%s]\n", key, num(r$coef_mean)))
  cat(sprintf("%s_coef_sd = [%s]\n", key, num(r$coef_sd)))
  cat(sprintf("%s_se_mean = [%s]\n", key, num(r$se_mean)))
  cat(sprintf("%s_se_sd = [%s]\n", key, num(r$se_sd)))
  cat(sprintf("%s_loglik_mean = %s\n", key, num(r$loglik_mean)))
  cat(sprintf("%s_loglik_sd = %s\n", key, num(r$loglik_sd)))
}
