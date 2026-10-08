# ERGMMulti.jl

Model several binary relations observed on the same actors, such as marriage and business ties among the same families. ERGMMulti.jl provides layer-specific and pooled effects, selected cross-layer statistics, and estimation by R's rule: maximum pseudo-likelihood (the exact MLE) for a dyad-independent formula, Monte-Carlo maximum likelihood otherwise.

| First analysis | Learn the model or data | Reference and detail |
|:--|:--|:--|
| [Assemble and fit layers](getting_started.md) | [Understand multilayer data](guide/structures.md) | [Estimation and uncertainty](guide/estimation.md) |

!!! note "Supported scope"

    Fitting accepts one-mode layers with the same actors and directedness. Estimation follows R's default (`method = :auto`): MPLE when every term is dyad-independent, MCMLE otherwise; `method = :mple` and `method = :mcmle` choose one explicitly. Multilevel and multiple-network containers have descriptive uses but are not alternate fitted model families. See [Not implemented](@ref not-implemented) for the precise boundary.

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

## Quick Start

Compare the baseline tie propensities of two relations on the same 16 Florentine families:

```julia
using NetworkCore, ERGMMulti

marriage = load_dataset(:florentine_marriage)
business = load_dataset(:florentine_business)
layers = as_multilayer([marriage, business], [:marriage, :business])
fit = fit_ergm_multi(layers, [LayerEdges(1), LayerEdges(2)])
display(fit)
```

Each coefficient is a separate layer’s baseline log-odds. This model does not test whether the two relations co-occur: that requires a cross-layer statistic such as `InterlayerDependence(1, 2)`, which makes the formula dyad-dependent, so the default fit is then the Monte-Carlo MLE, as in R (see [Model Estimation](@ref)).

## The block-diagonal mechanism

The layers of a [`MultilayerNetwork`](@ref) combine into one
block-diagonal network ([`combine_networks`](@ref)) whose vertices carry
`:layer` and `:actor` attributes (a layer's missing-dyad mask is carried
along and restored by [`split_by_layer`](@ref); pass `report=true` for a
`ConversionReport` of what the block-diagonal network cannot hold). The
model's dyad universe is the within-layer dyads; layer-aware terms then
express per-layer effects, pooled effects, and cross-layer dependence.

## Contents

```@contents
Pages = [
    "getting_started.md",
    "guide/structures.md",
    "guide/terms.md",
    "guide/estimation.md",
    "guide/multilevel.md",
    "api/types.md",
    "api/terms.md",
    "api/estimation.md",
]
Depth = 2
```

## [Not implemented](@id not-implemented)

These are the parts of `ergm.multi` that have no counterpart here. Each
entry states exactly what a user gets instead; the same items appear in the
README and in the CHANGELOG ("Known limitations").

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

## References

1. Krivitsky, P.N., Koehly, L.M. & Marcum, C.S. (2020). Exponential-family
   random graph models for multi-layer networks. *Psychometrika*, 85(3), 630-659.

2. Krivitsky, P.N. ergm.multi: Fit, Simulate and Diagnose Exponential-Family
   Models for Multiple or Multilayer Networks. R package.
   [https://cran.r-project.org/package=ergm.multi](https://cran.r-project.org/package=ergm.multi)


## Citation

If you use ERGMMulti.jl in your work, please cite it using the entry in
[`CITATION.bib`](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl/blob/main/CITATION.bib).
Please also cite the R package it ports, `ergm.multi` (reference 2), and the
methods paper (reference 1); the ecosystem's
[How to cite](https://statistical-network-analysis-with-julia.github.io/citing/)
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

## Module

```@docs
ERGMMulti
```
