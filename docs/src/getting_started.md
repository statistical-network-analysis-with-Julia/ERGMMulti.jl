# Getting Started

## Building a multilayer network

```julia
using ERGMMulti, Network

m = MultilayerNetwork(30; directed = true)
add_layer!(m, :friendship)
add_layer!(m, :advice)

add_layer_edge!(m, :friendship, 1, 2)
add_layer_edge!(m, :advice, 1, 2)

# Or from existing same-sized networks
m = as_multilayer([friend_net, advice_net], [:friendship, :advice])
```

## Statistics

```julia
compute(LayerEdges(1), m)                    # per-layer edges
compute(LayerEdges(), m)                     # pooled over layers
compute(InterlayerDependence(1, 2), m)       # co-occurring dyads
compute(WithinLayer(Triangle(), 1), m)       # any ERGM term, one layer
```

## Fitting

```julia
result = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)])
println(result)

# Per-layer parameterization
result = ergm_multi(m, [LayerEdges(1), LayerEdges(2),
                        InterlayerDependence(1, 2)])

# Offsets fix coefficients (ergm.multi's offset mechanism)
result = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                    offsets = Dict(1 => -log(30)))
```

## Simulation

```julia
draws = simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                            [-1.5, 2.0]; n_sim = 100)
```

The sampler only toggles within-layer dyads, matching the block-diagonal
dyad universe.
