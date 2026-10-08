# Getting Started

Construct aligned layers, inspect layer statistics, and fit pooled or layer-specific parameters. The simulation example shows what a fitted model generates; the estimation guide explains how the estimator is chosen and the limits of MPLE uncertainty for dependent layers.

!!! note "Before you begin"

    Fitting accepts one-mode layers with the same actors and directedness. Estimation follows R's default (`method = :auto`): MPLE when every term is dyad-independent, MCMLE otherwise. Multilevel and multiple-network containers have descriptive uses but are not alternate fitted model families. See [Not implemented](@ref not-implemented) for the precise boundary.

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

Run the blocks below in order in that environment. They build on variables from earlier steps; stochastic examples use seeded random number generators where shown.

## Building a multilayer network

```julia
using ERGMMulti, NetworkCore
using ERGM: compute, Triangle   # generic evaluator + terms shared with ERGM.jl

m = MultilayerNetwork(30; directed = true)   # a MultilayerNetwork{true}
add_layer!(m, :friendship)
add_layer!(m, :advice)

add_layer_edge!(m, :friendship, 1, 2)
add_layer_edge!(m, :advice, 1, 2)

# Or from existing same-sized networks (the first one's directedness sets D;
# a layer of the other directedness is refused)
friend_net = network(30; directed = true)
advice_net = network(30; directed = true)
m = as_multilayer([friend_net, advice_net], [:friendship, :advice])
is_directed(m)                               # true
```

## Statistics

```julia
compute(LayerEdges(1), m)                    # per-layer edges
compute(LayerEdges(), m)                     # pooled over layers
compute(InterlayerDependence(1, 2), m)       # co-occurring dyads
compute(WithinLayer(Triangle(), 1), m)       # any ERGM term, one layer
compute(WithinLayer(Triangle(), [1, 2]), m)  # ... or pooled over layers
```

## Fitting

`fit_ergm_multi` and `ergm_multi` are the same function (the ecosystem's
`fit_<model>` name and the R name). The terms are validated against the
layers before anything is fitted: an out-of-range layer index, a vertex
attribute a layer lacks, or an undirected-only term on a directed layer throw
an `ArgumentError` that names the fix.

```julia
using Random
rng = Xoshiro(1)
for i in 1:30, j in 1:30
    i == j && continue
    rand(rng) < 0.1 && add_layer_edge!(m, :friendship, i, j)
    rand(rng) < 0.1 && add_layer_edge!(m, :advice, i, j)
end

# `InterlayerDependence` is dyad-dependent, so the default (`method = :auto`,
# R's rule) is Monte-Carlo maximum likelihood (MCMLE), with a full table
result = fit_ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                        rng = Xoshiro(6))
println(result)            # `Method: mcmle (Monte-Carlo maximum likelihood)`
coeftable(result)          # the printed table, inspectable, with R's labels
confint(result)

# The MPLE on request: for a dyad-dependent formula it reports estimates and
# naive standard errors but no z, p or confidence interval; the parametric
# bootstrap provides them
mple = fit_ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                      method = :mple)
boot = fit_ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                      method = :mple, se = :bootstrap, n_boot = 50, rng = Xoshiro(5))
confint(boot)              # Wald limits, one row per coefficient

# Per-layer parameterization
result = ergm_multi(m, [LayerEdges(1), LayerEdges(2),
                        InterlayerDependence(1, 2)]; rng = Xoshiro(7))

# Offsets fix coefficients (ergm.multi's offset mechanism)
result = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                    offsets = Dict(1 => -log(30)), rng = Xoshiro(8))
```

## Simulation

```julia
draws = simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                            [-1.5, 2.0]; n_sim = 100, rng = Xoshiro(2))
```

The sampler only toggles within-layer dyads, matching the block-diagonal
dyad universe; it runs on the ERGM family's one Metropolis kernel
(`ERGM.mh_toggle!`), and `burnin`/`interval` default to the family's
dyad-scaled rule.
