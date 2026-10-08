# ERGMMulti.jl


[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.12+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ERGMMulti.jl icon" width="160">
</p>

ERGMs for multilayer networks in Julia — a port of the R `ergm.multi`
package (Krivitsky, Koehly & Marcum 2020).

## Installation

Requires Julia 1.12 or newer. The packages are not yet registered.

**Recommended: the ecosystem workspace.** It clones every package side by
side, develops them together in one environment, and adds the packages the
examples also use (CSV, DataFrames, Distributions, Graphs, StatsAPI,
StatsBase):

```bash
mkdir network-analysis && cd network-analysis
git clone https://github.com/statistical-network-analysis-with-Julia/statistical-network-analysis-with-Julia.github.io
julia statistical-network-analysis-with-Julia.github.io/tools/prepare_workspace.jl "$PWD" --clone
julia --project=.snippet-env
```

**Only this package, in your own environment.** Add its dependencies first,
in this order:

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/NetworkCore.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGM.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl")
```

The examples below load only `ERGMMulti`, `ERGM`, `NetworkCore` and the
standard library `Random`.

## The block-diagonal mechanism

`ergm.multi` models several relations on the same actors by combining the
layers into one **block-diagonal network** with layer-membership
attributes, restricting the dyad universe to within-layer dyads, and using
layer-aware terms. ERGMMulti.jl implements exactly this:

- `combine_networks(m)` builds the block-diagonal combined `Network`
  (`n × L` vertices with `:layer` and `:actor` attributes);
  `split_by_layer` inverts it. Both follow the ecosystem's conversion
  contract: a layer's **missing-dyad mask is preserved** at its block
  position (and restored on the way back — an unobserved tie never becomes
  an observed absent one), and what a block-diagonal network cannot hold
  (the layers' own attributes) is dropped and *named*: `combine_networks(m;
  report=true)` returns `(combined, ConversionReport)`. `as_multilayer`
  stores the layers as given (lossless).
- Estimation and simulation operate on the within-layer dyad universe.
  The sampler is the ERGM family's one Metropolis kernel (`ERGM.mh_toggle!`)
  with `(layer, i, j)` moves, allocation-free per step; `burnin`/`interval`
  default to the family's dyad-scaled rule (`20 × n_dyads`,
  `max(100, n_dyads ÷ 10)` over the within-layer dyads).
- The package ships a precompile workload: the first `fit_ergm_multi` of a
  session takes ~0.01 s instead of ~1.9 s (see the CHANGELOG).

## Terms

| Term | Meaning |
|------|---------|
| `LayerEdges(l)` / `LayerEdges()` | Per-layer or pooled edge count |
| `LayerMutual(l)` / `LayerMutual()` | Per-layer or pooled reciprocity (directed only) |
| `LayerTriangle(l)` / `LayerTriangle()` | Per-layer or pooled triangles |
| `WithinLayer(term, l)` | Any ERGM.jl term evaluated within layer `l` (`L(~term, ~A)`) |
| `WithinLayer(term, [l1, l2])` / `WithinLayer(term, :)` | Any ERGM.jl term pooled over layers: the sum of its per-layer statistics under one coefficient (`L(~term, c(~A, ~B))`) |
| `InterlayerDependence(l1, l2)` | Same-dyad co-occurrence across layers |
| `MultiplexMutual(l1, l2)` | Cross-layer reciprocity (`i→j` in `l1`, `j→i` in `l2`; directed only) |

All change statistics use ERGM.jl's add-direction convention and are
brute-force verified in the tests on a directed and an undirected fixture;
the term-by-term correspondence with `ergm.multi` (`L(~edges, ~A)`,
`L(~edges, ~A&B)`, `mutualL(Ls = ...)`, ...) and the fixture that pins each
row is tabulated in the documentation's Terms guide.
A `WithinLayer` term is validated against every layer it selects when the model is built
(ERGM.jl's own checks: attribute present on every vertex, covariate size,
direction requirement — `Kstar`/`GWDegree`/`Degree` are undirected-only,
`Mutual`/`OStar`/`IStar` directed-only). Coefficients carry the labels
`ergm.multi` prints, with the layers named by `layer_names(m)`:
`L(friendship)~edges`, `L(friendship)~gwesp.OTP.fixed.0.5` (ERGM.jl's
direction-aware label inside `L(...)~`), `L((friendship,advice))~mutual`
for a pooled term, `L(friendship&advice)~edges`,
`L(friendship,advice)~mutual` for `MultiplexMutual`, and
`offset(L(advice)~edges)` for an offset — pinned against `ergm.multi` by
`test/fixtures/multilayer_labels.toml`. An out-of-range layer index, a missing attribute, or a directed-only
term on undirected layers is an `ArgumentError` before any fitting — and so
is a slip at data entry: `add_layer_edge!` refuses an actor id outside
`1:n` (a 0-based id) or a self-loop on a `loops=false` layer instead of
dropping the tie silently, and `layer_network(m, 5)` on a two-layer network
names the layers it has rather than throwing a `BoundsError`.

## Quick Start

```julia
using ERGMMulti, NetworkCore, Random

m = MultilayerNetwork(30; directed = true)      # a MultilayerNetwork{true}
add_layer!(m, :friendship)
add_layer!(m, :advice)
rng = Xoshiro(1)
for i in 1:30, j in 1:30
    i == j && continue
    rand(rng) < 0.1 && add_layer_edge!(m, :friendship, i, j)
    rand(rng) < 0.1 && add_layer_edge!(m, :advice, i, j)
end

# Pooled edges + cross-layer dependence. `fit_ergm_multi` (the ecosystem's
# fit_<model> name) and `ergm_multi` (the R name) are the same function.
# The default, method = :auto, follows R: InterlayerDependence is
# dyad-dependent, so this is Monte-Carlo maximum likelihood (MCMLE), with
# standard errors (Fisher information + Monte-Carlo error), z, p, intervals
# and the log-likelihood.
result = fit_ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                        rng = Xoshiro(3))
coef(result)
coeftable(result)          # R's labels: L((friendship,advice))~edges, L(friendship&advice)~edges
confint(result)
loglikelihood(result)

# A dyad-independent formula is fitted by MPLE, which is then the exact MLE
exact = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
is_exact(exact)            # true

# Any ERGM.jl term within one layer, or pooled over layers (one coefficient);
# method = :mple asks for the pseudo-likelihood estimate (R's
# estimate = "MPLE"), which reports no z, p or interval for a dyad-dependent
# formula (see "What a fit tells you")
using ERGM
mple = ergm_multi(m, [LayerEdges(1), LayerEdges(2),
                      WithinLayer(GWESP(0.5), 1),         # L(~gwesp, ~friendship)
                      WithinLayer(Mutual(), [1, 2])];     # L(~mutual, c(~friendship, ~advice))
                  method = :mple)

# Fix a coefficient (ergm.multi-style offset)
result = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                    offsets = Dict(1 => -log(30)), rng = Xoshiro(4))

# Parametric-bootstrap standard errors for the MPLE of a dyad-dependent
# formula: calibrated z, p and confidence intervals
boot = ergm_multi(m, [LayerEdges(), LayerMutual()]; method = :mple,
                  se = :bootstrap, n_boot = 50, rng = Xoshiro(2))
confint(boot)              # Wald limits from the bootstrap standard errors

# Goodness of fit: model statistics, layer edges, multiplexity, and each
# layer's degree and edgewise-shared-partner distributions
gof(result; n_sim = 50, rng = Xoshiro(5))

# Simulate from the model (within-layer Metropolis sampler on the ERGM
# family's shared kernel; burnin/interval default to the dyad-scaled rule)
draws = simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                            [-1.5, 2.0]; n_sim = 100, rng = Xoshiro(3))
```

Estimation follows R's default (`method = :auto`, the rule of every
ERGM-family fitter): maximum pseudo-likelihood over the within-layer dyads
(the shared `NetworkCore.newton_fit` / `NetworkCore.logistic_derivatives`
kernel) when every term is dyad-independent, where it is the exact MLE, and
Monte-Carlo maximum likelihood otherwise — the estimator R's `ergm.multi`
uses for a dyad-dependent model. `method = :mple` and `method = :mcmle`
choose one explicitly; a keyword of the estimator not chosen (`se =
:bootstrap` on a dyad-dependent formula, say) is an `ArgumentError` that
says which `method` takes it. The MCMLE is checked against
exact enumeration (a 3-actor two-layer network, 4096 states) and against
`ergm.multi` fitted under nine R seeds
(`test/fixtures/multilayer_mcmle.toml`: coefficients, standard errors and
log-likelihood inside four of R's own seed-to-seed standard deviations). An
edges-only fit reproduces `logit(density)` per layer exactly, and four more
provenanced golden fixtures pin real `ergm.multi` 0.3.0 output — the
coefficient labels R prints for every multilayer term (compared exactly), a
dyad-independent fit with and without offsets (coefficients to ~1e-13),
`summary()` of every term with an `ergm.multi` counterpart plus a
dyad-dependent MPLE with `L(~edges, ~A&B)` and `L(~mutual, ~A)` (statistics
exact, coefficients to ~1e-7 against R's own design refit at `glm` epsilon
1e-14), and thirty-four ergm terms pooled over layers with
`L(~term, c(~A, ~B))` plus two MPLE fits with pooled terms — and
simulation→estimation round trips recover coefficients (tested).
Every fit answers the full StatsAPI surface (`coef`, `stderror`, `vcov`,
`confint`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable`).
Every exported name carries a docstring with a runnable example (`?LayerEdges`,
`?gof`, ...); the test suite executes all of them, and the ecosystem's
snippet checker executes this README and every documentation page.

### What a fit tells you when something is off

- `show` says which estimator ran and why, on its `Method:` line: `mcmle
  (Monte-Carlo maximum likelihood)` for the default fit of a dyad-dependent
  formula, `mple (maximum pseudo-likelihood, which is the likelihood: the
  formula is dyad-independent)`, or, for `method = :mple` on a dyad-dependent
  formula, `mple (maximum pseudo-likelihood: an approximation under dyadic
  dependence; …)`.
- The MPLE of a formula with dyad-dependent terms (`method = :mple`) is a
  pseudo-likelihood fit: `is_exact(fit)` is `false` and the point estimates
  are biased in finite samples. It reports the estimates and the naive
  pseudo-likelihood standard errors but **no z values, p-values or
  confidence intervals** (`NaN` in the table, `confint` throws, `show` and
  `approximations(fit)` say why): those standard errors under-cover.
  `se = :bootstrap` gives parametric-bootstrap standard errors for the MPLE
  with a full table; `se = :hessian` asks for the naive Wald table
  explicitly. A dyad-independent formula is exact and prints the full table.
- A statistic at the boundary of its attainable range (no co-occurring tie
  under `InterlayerDependence`, no mutual dyad, no triangle under
  `LayerTriangle` even when no single tie could close one, a `NodeMatch`
  with no within-group tie in its layer) has no finite MPLE and no finite
  MLE. Both estimators follow R ergm's default `drop=TRUE`: the warning
  "observed statistic(s) L(friendship&advice)~edges are at their smallest
  attainable values. Their coefficients will be fixed at -Inf" is printed,
  `coef(fit)` holds `-Inf` (or `+Inf`) for that term with `stderror` 0 and
  p-value 0, and the other coefficients are estimated with the statistic
  held at its bound: the MPLE on the dyads it does not touch (`bic` uses
  that dyad count), the MCMLE (the default for a dyad-dependent formula)
  with a sampler that never moves the statistic off its observed value.
  `dof` counts only finite coefficients, `is_exact(fit)` is `false`, and
  `show`/`approximations` carry a "fixed at -Inf" note. If the fixed
  statistic is dyad-dependent, the MCMLE's log-likelihood, AIC and BIC are
  `NaN` (the note says why). `drop = false` refuses such a model with an
  `ArgumentError` instead. `gof(fit)` simulates the fitted model, the
  statistic held at its bound; `se = :bootstrap` and
  `simulate_multi_ergm(fit.model, coef(fit))` are refused (nothing can be
  simulated at an infinite coefficient). A statistic with no identifiable
  coefficient (all its change statistics 0, or a combination of the others)
  is reported as `NaN`, R's `NA`, with a warning. `ergm.multi` 0.3.0 itself
  does *not* drop: it reports `NA`, or warns "The MPLE does not exist!" and
  returns a finite value (≈ −17.6 with a standard error in the thousands on
  the boundary design of `test/fixtures/twolayer_layer_terms.toml`). Written
  out as an offset, `offset(L(~edges, ~A&B))` with `offset.coef = -Inf`, R
  fits the same model as the drop here, and the default fit reproduces it
  (`test/fixtures/boundary_multi.toml`).
- Separation by a *combination* of statistics (no single one at its
  boundary, yet the ties are perfectly predicted) is R's "The MPLE does not
  exist!". The ecosystem's one separation verdict (NetworkCore's, a linear
  programme certified in exact arithmetic) is run on the design actually
  fitted, with or without offsets: the warning names the separating terms,
  `fit.converged` is `false`, `fit.separated` is `true` and
  `fit.separated_terms` lists them, z values, p-values and confidence
  intervals are `NaN`, `show`/`approximations` say the coefficients are
  where Newton stopped, and `se = :bootstrap` and an MCMLE started from it
  are refused.
- A Newton iteration that does not converge warns, reports
  `converged = false`, `is_exact(fit) == false`, and prints the caveat under
  the verdict. (Two copies of one statistic are not a convergence failure:
  the copy is reported as `NaN`, R's `NA`, with a warning.)
- A bootstrap refit with no finite MPLE (a simulated network with no
  triangle under `LayerTriangle`, no co-occurring tie, ...) is excluded from
  the covariance and kept as a `NaN` row of `fit.boot_replicates`; one warning
  reports "k of n_boot bootstrap refits did not converge", and the warning,
  `show` and `approximations(fit)` say: "The standard errors are conditional
  on a finite refit: the excluded replicates are the extreme ones, so the
  standard errors are biased downward." `threaded=false` gives the same
  numbers on one thread.
- `gof(fit)` reports the model statistics (which, for an MPLE fit of a
  dyad-dependent model, mostly measure the estimator's bias), the layer edge
  counts, the multiplexity distribution (dyads tied in exactly `k` layers)
  and, per layer, the degree and edgewise-shared-partner distributions —
  the auxiliary panels are where misfit shows.
- A masked (missing) dyad in any layer is refused, by `ergm_multi` and by
  `simulate_multi_ergm` alike, naming the layer: the MPLE enumerates every
  within-layer dyad as observed and the sampler toggles every one, and there
  is no `missing=` keyword on either.

## Not implemented

These are the parts of `ergm.multi` that have no counterpart here. Each
entry states exactly what a user gets instead; the same items appear on the
documentation index and in the CHANGELOG ("Known limitations").

- **The MCMLE is not R's implementation.** `ergm_multi` follows R's
  default (`method = :auto`): the MPLE, which is the exact MLE, when every
  term is dyad-independent, and Monte-Carlo maximum likelihood otherwise
  (`method = :mcmle`; checked against `ergm.multi` within R's seed-to-seed
  spread). The MCMLE uses the package's uniform (layer, i, j) proposal, not
  R's tie/no-tie proposals or `control.ergm` tuning, so R seeds do not carry
  over; it has no missing-data form (a masked layer is refused) and no MCMC
  diagnostics plot. A statistic at the boundary of its attainable range is
  fixed at `∓Inf` by both estimators (R ergm's `drop=TRUE`; `drop = false`
  refuses instead), where `ergm.multi` 0.3.0 reports `NA` or a finite value
  where its estimation stopped; when that statistic is dyad-dependent the
  MCMLE's log-likelihood, AIC and BIC are `NaN`. The MCMLE is refused when
  the MPLE start is separated (an `ArgumentError` naming the terms, where
  `ergm.multi` warns and returns a finite estimate with a huge standard
  error).
  The MPLE of a dyad-dependent formula (`method = :mple`) reports no z
  values, p-values or confidence intervals by default (`NaN` in `coeftable`,
  `confint` throws an `ArgumentError`), because its naive pseudo-likelihood
  standard errors under-cover (95% Wald intervals covered 0.76–0.94 in
  simulation); `se = :bootstrap` gives parametric-bootstrap standard errors
  (coverage 0.90–0.96), which are conditional on a finite refit and biased
  downward when replicates are dropped (counted in `show` and
  `approximations(fit)`), and `se = :hessian` is the written opt-in to the
  naive Wald table R prints for an MPLE fit.
- **No models over several networks.** `ergm.multi`'s `Networks()`
  construction — one model fitted to a sample of networks that may differ in
  size and composition, with `N(~edges, ~x)` network-level covariates and
  the `gofN` diagnostics — is not implemented. `MultiNetwork` is a
  descriptive container (`CrossNetEdges` only) and `ergm_multi` accepts a
  `MultilayerNetwork` only: layers on the *same* actors. What you get: the
  per-layer (`LayerEdges(l)`, `WithinLayer(term, l)`) and pooled
  (`LayerEdges()`, `WithinLayer(term, [1, 2])`) parameterisations of a
  multilayer network — every layer-specific effect is its own coefficient
  or an offset, never a function of a covariate.
- **Layer logic is limited to same-dyad conjunction of edges and cross-layer
  reciprocity.** `InterlayerDependence(l1, l2)` is `L(~edges, ~A&B)` and
  `MultiplexMutual(l1, l2)` is `mutualL(Ls = list(~A, ~B))`. Any ERGM.jl
  term can be evaluated within one layer (`WithinLayer(term, l)`,
  `L(~term, ~A)`) or pooled over layers (`WithinLayer(term, [1, 2])`,
  `L(~term, c(~A, ~B))`), but not on a *logical* layer: `L(~term, ~A&B)` for
  a term other than `edges`, `~A|B`, `~!A`, `~A&!B`, weighted layer sums,
  `twostarL`, the `espL`/`dspL`/`gwespL` cross-layer path families and
  `CMBL` are not implemented. Neither are `Layer(nw, c("attr1", "attr2"))`
  (layers read from edge attributes of one network — build one `Network`
  per layer), the `.symmetric` mixing of directed and undirected layers
  (`add_layer!` refuses a layer of the other directedness) or bipartite
  layers (`b1dspL`, ...). What you get: no such term exists to construct; a
  two-mode (bipartite) network is refused by `add_layer!`/`as_multilayer`
  with "layer :a is two-mode (bipartite); ERGMMulti models one-mode layers
  only — the within-layer dyad universe would enumerate the impossible
  within-mode pairs as observed non-ties, so the fit is refused instead
  (statnet's bipartite layer terms b1dspL, b2dspL, ... are not
  implemented)"; a pooled term that expands into different statistics on
  different layers (a `NodeFactor` whose levels differ between layers) and a
  `WithinLayer(Offset(...), l)` are `ArgumentError`s (use the `offsets=`
  keyword for the latter).
- **Multilevel statistics are descriptive.** `Nestedness`, `LevelHomophily`
  and `CrossLevelEdge` on a `MultilevelNetwork` are evaluated with
  `compute` only and never enter a fit; `ergm_multi(::MultilevelNetwork,
  ...)` is a `MethodError`. What you get: the numbers, and nothing that
  looks like an estimate.
- **Goodness of fit has no geodesic-distance panel.** `gof` compares the
  model statistics, the layer edge counts, the multiplexity distribution
  and each layer's degree and edgewise-shared-partner distributions;
  `gof.ergm`'s geodesic distances and the dyad-wise shared partners are not
  computed.

## Multilevel descriptives

`MultilevelNetwork` carries per-level networks, membership maps, and
explicit cross-level edges (`add_cross_level_edge!`); `Nestedness`,
`LevelHomophily`, and `CrossLevelEdge` are **descriptive statistics** for
such designs (they do not participate in estimation).

## References

1. Krivitsky, P.N., Koehly, L.M. & Marcum, C.S. (2020). Exponential-family
   random graph models for multi-layer networks. *Psychometrika*, 85(3),
   630-659.

2. Krivitsky, P.N. ergm.multi: Fit, Simulate and Diagnose Exponential-Family
   Models for Multiple or Multilayer Networks. R package.
   [https://cran.r-project.org/package=ergm.multi](https://cran.r-project.org/package=ergm.multi)

## Citation

If you use ERGMMulti.jl in your work, please cite it using the entry in
[`CITATION.bib`](CITATION.bib). Please also cite the R package it ports,
`ergm.multi` (reference 2), and the methods paper (reference 1); the
ecosystem's [How to cite](https://statistical-network-analysis-with-julia.github.io/citing/)
page lists the references for every package.

```biblatex
@misc{SNWJERGMMultiJL,
  author = {Santoni, Simone},
  title = {ERGMMulti.jl: Exponential Random Graph Models for Multilayer Networks in Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
