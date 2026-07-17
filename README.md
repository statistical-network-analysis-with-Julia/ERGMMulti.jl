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
  `split_by_layer` inverts it.
- Estimation and simulation operate on the within-layer dyad universe.

## Terms

| Term | Meaning |
|------|---------|
| `LayerEdges(l)` / `LayerEdges()` | Per-layer or pooled edge count |
| `LayerMutual(l)` / `LayerMutual()` | Per-layer or pooled reciprocity |
| `LayerTriangle(l)` | Per-layer triangles |
| `WithinLayer(term, l)` | Any ERGM.jl term lifted into layer `l` (the analogue of `Layer(~term)`) |
| `InterlayerDependence(l1, l2)` | Same-dyad co-occurrence across layers |
| `MultiplexMutual(l1, l2)` | Cross-layer reciprocity (`i→j` in `l1`, `j→i` in `l2`) |

All change statistics use ERGM.jl's add-direction convention and are
brute-force verified in the tests.

## Quick Start

```julia
using ERGMMulti, Network

m = MultilayerNetwork(30; directed = true)
add_layer!(m, :friendship)
add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2)
# ...

# Pooled edges + cross-layer dependence, per-layer offset supported
# (fit_multi_ergm is the standardized alias of the same function)
result = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)])

# Fix a coefficient (ergm.multi-style offset)
result = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                    offsets = Dict(1 => -log(30)))

# Simulate from the model (within-layer Metropolis sampler)
draws = simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                            [-1.5, 2.0]; n_sim = 100)
```

Estimation is maximum pseudo-likelihood over the within-layer dyads
(via the shared `ERGM.newton_fit` Newton-Raphson-with-step-halving
optimizer); fits of dyad-dependent formulas print a standard-error
caveat, and an edges-only fit reproduces
`logit(density)` per layer exactly, and simulation→estimation round trips
recover coefficients (tested).

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
