# ERGMMulti.jl


[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.12+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ERGMMulti.jl icon" width="160">
</p>

ERGMs for multilayer networks in Julia — a port of the R `ergm.multi`
package (Krivitsky, Koehly & Marcum 2020).

## Installation

Requires Julia 1.12+. ERGMMulti.jl depends on the unregistered
[Networks.jl](https://github.com/statistical-network-analysis-with-Julia/Networks.jl) and [ERGM.jl](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl) packages, which must be added first (in this order):

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/Networks.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGM.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl")
```

For development, you can instead clone all ecosystem repositories side by
side (the monorepo layout) and start Julia with the root workspace project
(`julia --project=.` in the clone root): the `[sources]` path dependencies
then wire the packages together with no ordered installs needed.

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
| `LayerTriangle(l)` | Per-layer triangles |
| `WithinLayer(term, l)` | Any ERGM.jl term lifted into layer `l` (the analogue of `Layer(~term)`) |
| `InterlayerDependence(l1, l2)` | Same-dyad co-occurrence across layers |
| `MultiplexMutual(l1, l2)` | Cross-layer reciprocity (`i→j` in `l1`, `j→i` in `l2`; directed only) |

All change statistics use ERGM.jl's add-direction convention and are
brute-force verified in the tests on a directed and an undirected fixture;
the term-by-term correspondence with `ergm.multi` (`L(~edges, ~A)`,
`L(~edges, ~A&B)`, `mutualL(Ls = ...)`, ...) and the fixture that pins each
row is tabulated in the documentation's Terms guide.
A `WithinLayer` term is validated against its layer when the model is built
(ERGM.jl's own checks: attribute present on every vertex, covariate size,
direction requirement — `Kstar`/`GWDegree`/`Degree` are undirected-only,
`Mutual`/`OStar`/`IStar` directed-only), and its coefficient carries
ERGM.jl's direction-aware label (`L1.gwesp.OTP.fixed.0.5` on a directed
layer). An out-of-range layer index, a missing attribute, or a directed-only
term on undirected layers is an `ArgumentError` before any fitting — and so
is a slip at data entry: `add_layer_edge!` refuses an actor id outside
`1:n` (a 0-based id) or a self-loop on a `loops=false` layer instead of
dropping the tie silently, and `layer_network(m, 5)` on a two-layer network
names the layers it has rather than throwing a `BoundsError`.

## Quick Start

```julia
using ERGMMulti, Networks, Random

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
result = fit_ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)])
coef(result)
coeftable(result)          # the R-style table `show` prints, inspectable
confint(result)            # Wald limits

# Fix a coefficient (ergm.multi-style offset)
result = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                    offsets = Dict(1 => -log(30)))

# Parametric-bootstrap standard errors for a dyad-dependent formula
result = ergm_multi(m, [LayerEdges(), LayerMutual()]; se = :bootstrap,
                    n_boot = 50, rng = Xoshiro(2))

# Simulate from the model (within-layer Metropolis sampler on the ERGM
# family's shared kernel; burnin/interval default to the dyad-scaled rule)
draws = simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                            [-1.5, 2.0]; n_sim = 100, rng = Xoshiro(3))
```

Estimation is maximum pseudo-likelihood over the within-layer dyads (the
shared `Networks.newton_fit` / `Networks.logistic_derivatives` kernel). An
edges-only fit reproduces `logit(density)` per layer exactly, two
provenanced golden fixtures pin real `ergm.multi` 0.3.0 output — a
dyad-independent fit with and without offsets (coefficients to ~1e-13), and
`summary()` of every term with an `ergm.multi` counterpart plus a
dyad-dependent MPLE with `L(~edges, ~A&B)` and `L(~mutual, ~A)` (statistics
exact, coefficients to ~1e-7 against R's own design refit at `glm` epsilon
1e-14) — and simulation→estimation round trips recover coefficients (tested).
Every fit answers the full StatsAPI surface (`coef`, `stderror`, `vcov`,
`confint`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable`).
Every exported name carries a docstring with a runnable example (`?LayerEdges`,
`?gof`, ...); the test suite executes all of them, and the ecosystem's
snippet checker executes this README and every documentation page.

### What a fit tells you when something is off

- A formula with dyad-dependent terms prints a pseudo-likelihood caveat;
  `is_exact(fit)` is `false`; `se = :bootstrap` is the remedy for the
  standard errors.
- A statistic at the boundary of its attainable range (no co-occurring tie
  under `InterlayerDependence`, no mutual dyad, a `NodeMatch` with no
  within-group tie in its layer) has no finite MPLE. As R ergm's default
  `drop=TRUE` for one-mode terms: the warning "observed statistic(s) duplex.1.2 are at
  their smallest attainable values. Their coefficients will be fixed at
  -Inf" is printed, `coef(fit)` holds `-Inf` (or `+Inf`) for that term with
  `stderror` 0 and p-value 0, the other coefficients are the MPLE on the
  dyads the dropped statistic does not touch (`bic` uses that dyad count,
  `dof` counts only finite coefficients), `is_exact(fit)` is `false`, and
  `show`/`approximations` carry a "fixed at -Inf" note. `se = :bootstrap`,
  `gof(fit)` and `simulate_multi_ergm(fit.model, coef(fit))` are all refused
  on such a fit with an `ArgumentError` naming the term (nothing can be
  simulated at an infinite coefficient; the chain would reject every
  proposal and `gof` would report p = 1 everywhere) — drop the term and
  refit. (`ergm.multi` 0.3.0 itself does *not* drop: it warns "The MPLE does
  not exist!" and returns a finite ≈ −17.6 with a standard error in the
  thousands for the same design; the finite coefficients agree exactly. The
  R output is frozen in `test/fixtures/twolayer_layer_terms.toml`.)
- Perfect separation by a *combination* of statistics (no single one at its
  boundary, yet the ties are perfectly predicted) is R's "The MPLE does not
  exist!": the same sentence is printed, `fit.converged` is `false` and
  `fit.separated` is `true`, and `show`/`approximations` say the coefficients
  are the point where Newton stopped on its flat asymptote — with or without
  offsets.
- A Newton iteration that does not converge warns, reports
  `converged = false`, `is_exact(fit) == false`, and prints the caveat under
  the verdict; a rank-deficient design (two copies of a statistic) also
  reports `NaN` standard errors with Networks.jl's "Hessian is not negative
  definite" warning.
- A bootstrap refit with no finite MPLE (a simulated network with no
  triangle under `LayerTriangle`, no co-occurring tie, ...) is excluded from
  the covariance and kept as a `NaN` row of `fit.boot_replicates`; one warning
  reports "k of n_boot bootstrap refits did not converge". `threaded=false`
  gives the same numbers on one thread.
- A masked (missing) dyad in any layer is refused, by `ergm_multi` and by
  `simulate_multi_ergm` alike, naming the layer: the MPLE enumerates every
  within-layer dyad as observed and the sampler toggles every one, and there
  is no `missing=` keyword on either.

## Not implemented

These are the parts of `ergm.multi` that have no counterpart here. Each
entry states exactly what a user gets instead; the same four items appear
on the documentation index and in the CHANGELOG.

- **No MCMLE.** Estimation is maximum pseudo-likelihood over the
  within-layer dyads, for every model; there is no `method=:mcmle`
  keyword, so for a dyad-dependent multilayer model R's
  `ergm(Layer(...) ~ ...)` default estimator (MCMLE) has no counterpart.
  What you get: an MPLE point estimate, `is_exact(fit) == false`, a
  pseudo-likelihood caveat printed by `show`, `approximations(fit)` naming
  the approximation, and `se = :bootstrap` as the offered remedy for the
  covariance (the point estimates are unchanged by it).
- **No covariate-driven multi-network models.** `ergm.multi`'s
  `Networks()`-style models with per-network coefficients driven by a
  network-level covariate (`N(~edges, ~x)`) are not available;
  `MultiNetwork` is a descriptive container (`CrossNetEdges` only) and
  `ergm_multi` accepts a `MultilayerNetwork` only. What you get: the
  per-layer (`LayerEdges(l)`) and pooled (`LayerEdges()`,
  `LayerEdges([1, 2])`) parameterisations of a multilayer network — every
  layer-specific effect is its own coefficient or an offset, never a
  function of a covariate.
- **Layer logic is limited to same-dyad conjunction and cross-layer
  reciprocity.** `InterlayerDependence(l1, l2)` is `L(~edges, ~A&B)` and
  `MultiplexMutual(l1, l2)` is `mutualL(Ls = list(~A, ~B))`. `twostarL`,
  the `espL`/`dspL`/`gwespL` cross-layer path families, `CMBL`, general
  layer-logic formulas (`~A|B`, `~!A`, `~A&!B`, ...) and bipartite layers
  (`b1dspL`, ...) are not implemented. What you get: no such term exists to
  construct, and a two-mode (bipartite) network is refused by
  `add_layer!`/`as_multilayer` with "layer :a is two-mode (bipartite);
  ERGMMulti models one-mode layers only — the within-layer dyad universe
  would enumerate the impossible within-mode pairs as observed non-ties, so
  the fit is refused instead (statnet's bipartite layer terms b1dspL,
  b2dspL, ... are not implemented)"; a single-layer statistic is expressed
  with `WithinLayer(term, l)` wherever ERGM.jl has the term.
- **Multilevel statistics are descriptive.** `Nestedness`, `LevelHomophily`
  and `CrossLevelEdge` on a `MultilevelNetwork` are evaluated with
  `compute` only and never enter a fit; `ergm_multi(::MultilevelNetwork,
  ...)` is a `MethodError`. What you get: the numbers, and nothing that
  looks like an estimate.

One smaller limitation of the estimator is documented where it arises (the
"What a fit tells you" list above and the estimation guide): the
inverse-Hessian standard errors of a dyad-dependent model are
anticonservative (the bootstrap is the remedy).

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
[`CITATION.bib`](CITATION.bib):

```biblatex
@misc{SNWJERGMMultiJL,
  author = {{Statistical Network Analysis with Julia}},
  title = {ERGMMulti.jl: Exponential Random Graph Models for Multilayer Networks in Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
