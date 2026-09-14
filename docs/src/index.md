# ERGMMulti.jl

Model several binary relations observed on the same actors, such as marriage and business ties among the same families. ERGMMulti.jl provides layer-specific and pooled effects, selected cross-layer statistics, and maximum pseudo-likelihood estimation.

| First analysis | Learn the model or data | Reference and detail |
|:--|:--|:--|
| [Assemble and fit layers](getting_started.md) | [Understand multilayer data](guide/structures.md) | [Estimation and uncertainty](guide/estimation.md) |

!!! note "Supported scope"

    Fitting accepts one-mode layers with the same actors and directedness. Estimation is MPLE; MCMC-MLE is not implemented. Multilevel and multiple-network containers have descriptive uses but are not alternate fitted model families. See [Not implemented](@ref not-implemented) for the precise boundary.

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

## Quick Start

Compare the baseline tie propensities of two relations on the same 16 Florentine families:

```julia
using Networks, ERGMMulti

marriage = load_dataset(:florentine_marriage)
business = load_dataset(:florentine_business)
layers = as_multilayer([marriage, business], [:marriage, :business])
fit = fit_ergm_multi(layers, [LayerEdges(1), LayerEdges(2)])
display(fit)
```

Each coefficient is a separate layer’s baseline log-odds. This model does not test whether the two relations co-occur: that requires a cross-layer statistic such as `InterlayerDependence(1, 2)` and attention to the resulting pseudo-likelihood approximation.

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
entry states exactly what a user gets instead; the same four items appear
in the README and in the CHANGELOG.

- **No MCMLE.** Estimation is maximum pseudo-likelihood over the
  within-layer dyads, for every model; there is no `method=:mcmle`
  keyword, so for a dyad-dependent multilayer model R's
  `ergm(Layer(...) ~ ...)` default estimator (MCMLE) has no counterpart.
  What you get: an MPLE point estimate, `is_exact(fit) == false`, a
  pseudo-likelihood caveat printed by `show`, `approximations(fit)` naming
  the approximation, and `se = :bootstrap` for model-based simulation and refitting
  of the covariance (the point estimates are unchanged by it).
- **No covariate-driven multi-network models.** `ergm.multi`'s
  `Networks()`-style models with per-network coefficients driven by a
  network-level covariate (`N(~edges, ~x)`) are not available;
  [`MultiNetwork`](@ref) is a descriptive container ([`CrossNetEdges`](@ref)
  only) and [`ergm_multi`](@ref) accepts a [`MultilayerNetwork`](@ref) only.
  What you get: the per-layer (`LayerEdges(l)`) and pooled (`LayerEdges()`,
  `LayerEdges([1, 2])`) parameterisations of a multilayer network — every
  layer-specific effect is its own coefficient or an offset, never a
  function of a covariate.
- **Layer logic is limited to same-dyad conjunction and cross-layer
  reciprocity.** [`InterlayerDependence`](@ref)`(l1, l2)` is
  `L(~edges, ~A&B)` and [`MultiplexMutual`](@ref)`(l1, l2)` is
  `mutualL(Ls = list(~A, ~B))`. `twostarL`, the `espL`/`dspL`/`gwespL`
  cross-layer path families, `CMBL`, general layer-logic formulas
  (`~A|B`, `~!A`, `~A&!B`, ...) and bipartite layers (`b1dspL`, ...) are
  not implemented. What you get: no such term exists to construct, and a
  two-mode (bipartite) network is refused by
  [`add_layer!`](@ref)/[`as_multilayer`](@ref) with "layer :a is two-mode
  (bipartite); ERGMMulti models one-mode layers only — the within-layer dyad
  universe would enumerate the impossible within-mode pairs as observed
  non-ties, so the fit is refused instead (statnet's bipartite layer terms
  b1dspL, b2dspL, ... are not implemented)"; a single-layer statistic is
  expressed with [`WithinLayer`](@ref)`(term, l)` wherever ERGM.jl has the
  term.
- **Multilevel statistics are descriptive.** [`Nestedness`](@ref),
  [`LevelHomophily`](@ref) and [`CrossLevelEdge`](@ref) on a
  [`MultilevelNetwork`](@ref) are evaluated with `compute` only and never
  enter a fit; `ergm_multi(::MultilevelNetwork, ...)` is a `MethodError`.
  What you get: the numbers, and nothing that looks like an estimate.

One smaller limitation of the estimator is documented where it arises in
[Model Estimation](@ref): the inverse-Hessian standard errors of a
dyad-dependent model can be anticonservative. A model-based bootstrap can
account for dependence under the fitted model, but requires adequate
simulation and does not remove the approximation in the point estimator.

## References

1. Krivitsky, P.N., Koehly, L.M. & Marcum, C.S. (2020). Exponential-family
   random graph models for multi-layer networks. *Psychometrika*, 85(3), 630-659.


## Citation

If you use ERGMMulti.jl in your work, please cite it using the entry in
[`CITATION.bib`](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl/blob/main/CITATION.bib):

```biblatex
@misc{SNWJERGMMultiJL,
  author = {{Statistical Network Analysis with Julia}},
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
