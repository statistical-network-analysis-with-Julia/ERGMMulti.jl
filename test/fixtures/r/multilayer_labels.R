# Golden fixture: the COEFFICIENT LABELS statnet `ergm.multi` prints for every
# multilayer term ERGMMulti.jl offers, and the statistics they label.
#
# Regenerate from the package root (a few seconds):
#
#   Rscript test/fixtures/r/multilayer_labels.R > test/fixtures/multilayer_labels.toml
#
# WHAT IT PINS
#
# The labels a migrant reads in `coef()`/`summary()` and looks up by name:
# `L(friend)~edges` for one layer, `L((friend,advice))~edges` for a term pooled
# over a list of layers (double parentheses, even for a one-element list),
# `L(friend&advice)~edges` for a co-occurrence, `L(friend,advice)~mutual` for
# `mutualL`, and `offset(L(advice)~nodematch.g)` for an offset term. The other
# fixtures of this package store R's labels for the two-layer `A`/`B`
# designs; this one adds three layers with longer names (so a pool over a
# strict subset, over every layer and over one layer all occur), both orders
# of a layer pair, and the offset label, which only a fit prints.
#
# (i)   directed: `summary()` of nineteen statistics on three layers; names
#       exact, values at 1e-10 (deterministic counts);
# (ii)  undirected: the same for six statistics on two layers;
# (iii) the coefficient names of an MPLE fit with two offset terms.
#
# THE DATA are deterministic (no random draw): ties are arithmetic functions
# of the vertex indices, frozen below as edge lists, with the vertex
# attributes `g` (three levels) and `x` (numeric).

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(network)
  library(ergm.multi)
})

# No random draw is made; the seed is set (and recorded) only so that the
# fixture carries the provenance every fixture of the ecosystem carries.
seed <- 20261006L
set.seed(seed)

n <- 8
g <- rep(c("a", "b", "c"), length.out = n)
x <- (1:n) / 2
mk <- function(f, directed = TRUE) {
  A <- matrix(0, n, n)
  for (i in 1:n) for (j in 1:n) if (i != j && f(i, j)) A[i, j] <- 1
  if (!directed) A <- pmax(A, t(A))
  nw <- network(A, directed = directed)
  nw %v% "g" <- g
  nw %v% "x" <- x
  nw
}
friend <- mk(function(i, j) (i + 2 * j) %% 3 == 0)
advice <- mk(function(i, j) (2 * i + j) %% 3 == 0 || (i * j) %% 5 == 1)
cowork <- mk(function(i, j) (i + j) %% 2 == 1)
l3 <- Layer(list(friend = friend, advice = advice, cowork = cowork))

# --- (i) directed, three layers ----------------------------------------------
stat_formula <- l3 ~ L(~edges, ~friend) + L(~edges, c(~friend, ~advice, ~cowork)) +
  L(~edges, c(~friend, ~cowork)) + L(~edges, c(~advice)) +
  L(~mutual, ~friend) + L(~mutual, c(~friend, ~advice)) +
  L(~triangle, ~cowork) + L(~triangle, c(~friend, ~advice, ~cowork)) +
  L(~edges, ~friend & advice) + L(~edges, ~advice & cowork) + L(~edges, ~cowork & friend) +
  mutualL(Ls = list(~friend, ~advice)) + mutualL(Ls = list(~cowork, ~advice)) +
  L(~nodefactor("g"), ~friend) + L(~gwesp(0.5, fixed = TRUE), ~cowork) +
  L(~nodecov("x"), c(~friend, ~cowork)) + L(~idegree(1:2), ~advice)
stats <- summary(stat_formula)
julia_terms <- c("LayerEdges(1)", "LayerEdges(:)", "LayerEdges([1, 3])", "LayerEdges([2])",
                 "LayerMutual(1)", "LayerMutual([1, 2])",
                 "LayerTriangle(3)", "LayerTriangle(:)",
                 "InterlayerDependence(1, 2)", "InterlayerDependence(2, 3)",
                 "InterlayerDependence(3, 1)",
                 "MultiplexMutual(1, 2)", "MultiplexMutual(3, 2)",
                 "WithinLayer(NodeFactor(:g), 1)", "WithinLayer(GWESP(0.5), 3)",
                 "WithinLayer(NodeCov(:x), [1, 3])", "WithinLayer(IDegree(1:2), 2)")

# --- (ii) undirected, two layers ---------------------------------------------
ua <- mk(function(i, j) (i + j) %% 3 == 0, FALSE)
ub <- mk(function(i, j) (i * j + i + j) %% 3 == 2, FALSE)
l2 <- Layer(list(a = ua, b = ub))
und_formula <- l2 ~ L(~edges, ~a & b) + L(~edges, ~b & a) + L(~edges, c(~a, ~b)) +
  L(~triangle, ~a) + L(~kstar(2), ~b) + L(~gwesp(0.5, fixed = TRUE), c(~a, ~b))
und_stats <- summary(und_formula)
und_julia_terms <- c("InterlayerDependence(1, 2)", "InterlayerDependence(2, 1)",
                     "LayerEdges(:)", "LayerTriangle(1)", "WithinLayer(Kstar(2), 2)",
                     "WithinLayer(GWESP(0.5), :)")

# --- (iii) offset labels -------------------------------------------------------
fit <- suppressMessages(ergm(l3 ~ L(~edges, ~friend) + L(~edges, ~advice) +
                               offset(L(~nodematch("g"), ~advice)) +
                               offset(L(~edges, ~friend & advice)),
                             offset.coef = c(0.5, 0.2), estimate = "MPLE"))
offset_coef_names <- names(coef(fit))
offset_julia_terms <- c("LayerEdges(1)", "LayerEdges(2)",
                        "WithinLayer(NodeMatch(:g), 2)", "InterlayerDependence(1, 2)")

# --- output ---------------------------------------------------------------------
strs <- function(v) paste(sprintf('"%s"', v), collapse = ", ")
nums <- function(v) paste(sprintf("%.17g", v), collapse = ", ")
edges_of <- function(nw, directed = TRUE) {
  el <- as.matrix(nw, matrix.type = "edgelist")
  el <- el[order(el[, 1], el[, 2]), , drop = FALSE]
  paste(sprintf("[%d, %d]", el[, 1], el[, 2]), collapse = ", ")
}

cat('name = "multilayer_labels"\n\n')
cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_multi_version = "%s"\n', as.character(packageVersion("ergm.multi"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/multilayer_labels.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "deterministic: three directed 8-actor layers friend/advice/cowork and two undirected 8-actor layers a/b, ties arithmetic in the vertex indices; vertex attributes g = rep(a,b,c) and x = (1:8)/2; frozen below as edge lists"\n')
cat('statistics = "summary(Layer(list(friend, advice, cowork)) ~ ...) with layer terms on one layer, pooled over a subset, over every layer and over one layer, both co-occurrence orders, mutualL, and nodefactor/gwesp/nodecov/idegree inside L(); und_: the undirected pair"\n')
cat('model_offset = "ergm(l3 ~ L(~edges,~friend) + L(~edges,~advice) + offset(L(~nodematch(\\"g\\"),~advice)) + offset(L(~edges,~friend&advice)), offset.coef = c(0.5, 0.2), estimate = \\"MPLE\\") -- only its coefficient NAMES are frozen"\n')
cat("\n")
cat("[tolerance]\n")
cat("# Names are compared exactly. The statistics are exact counts (and a\n")
cat("# geometrically weighted decimal): both sides compute them in closed form.\n")
cat("# DO NOT LOOSEN.\n")
cat("statistics = 1e-10\n")
cat("und_statistics = 1e-10\n")
cat("\n")
cat("[values]\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("g = [%s]\n", strs(g)))
cat(sprintf("x = [%s]\n", nums(x)))
cat('layer_names = ["friend", "advice", "cowork"]\n')
cat(sprintf("edges_friend = [%s]\n", edges_of(friend)))
cat(sprintf("edges_advice = [%s]\n", edges_of(advice)))
cat(sprintf("edges_cowork = [%s]\n", edges_of(cowork)))
cat(sprintf("stat_julia_terms = [%s]\n", strs(julia_terms)))
cat(sprintf("stat_names = [%s]\n", strs(names(stats))))
cat(sprintf("statistics = [%s]\n", nums(as.numeric(stats))))
cat('und_layer_names = ["a", "b"]\n')
cat(sprintf("und_edges_a = [%s]\n", edges_of(ua)))
cat(sprintf("und_edges_b = [%s]\n", edges_of(ub)))
cat(sprintf("und_stat_julia_terms = [%s]\n", strs(und_julia_terms)))
cat(sprintf("und_stat_names = [%s]\n", strs(names(und_stats))))
cat(sprintf("und_statistics = [%s]\n", nums(as.numeric(und_stats))))
cat(sprintf("offset_julia_terms = [%s]\n", strs(offset_julia_terms)))
cat("offset_values = [0.5, 0.2]\n")
cat(sprintf("offset_coef_names = [%s]\n", strs(offset_coef_names)))
