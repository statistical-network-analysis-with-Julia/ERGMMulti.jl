# Multi-Network Structures

## MultilayerNetwork

[`MultilayerNetwork`](@ref)`{D}` holds L same-sized networks over one actor
set, all directed (`D = true`) or all undirected (`D = false`) — the
directedness is a type parameter, read with `is_directed(m)`, and
[`add_layer!`](@ref) refuses a layer of the other kind. Access layers by
index or name with [`layer_network`](@ref); add ties with
[`add_layer_edge!`](@ref). Data entry is loud: an actor id outside `1:n`
or a self-loop on a layer built without `loops=true` is an `ArgumentError`
naming the ids and the range (never a silently dropped tie), and an
unknown layer name or an out-of-range layer index is an `ArgumentError`
naming the layers the network has.

## The combined network

[`combine_networks`](@ref) produces the `ergm.multi`-style block-diagonal
`Network`: `n × L` vertices where vertex `(l−1)·n + a` is actor `a`'s
copy in layer `l`, carrying `:layer` and `:actor` vertex attributes, with
every layer's edges inside its own block and no cross-block edges.
[`split_by_layer`](@ref) inverts the construction (and rejects
cross-block edges).

Both are conversions in the sense of Networks.jl's conversion contract:
preserve what the target can represent, report what it cannot. A
`Network` can carry a missing-dyad mask, so **the mask is preserved** — a
masked (unobserved) dyad of layer `l` is masked at its block position of
the combined network and comes back onto its layer through
`split_by_layer`; a masked cross-block dyad is refused like a cross-block
edge. What the block-diagonal network cannot hold — the layers' own
vertex, edge and network attributes (only `:layer`/`:actor` are carried) —
is dropped and named: pass `report=true` to get the
`Networks.ConversionReport` alongside the result.

```julia
using ERGMMulti, Networks
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2)
set_missing_dyad!(layer_network(m, :advice), 2, 3)       # unobserved in layer 2
set_vertex_attribute!(layer_network(m, :friendship), :grp,
                      Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B"))
c, rep = combine_networks(m; report=true)
is_missing_dyad(c, 6, 7)          # true — actor 2→3 in block 2
n_missing_dyads(c)                # 1
dropped_fields(rep)               # [:grp]
m2 = split_by_layer(c, 4, 2; names=[:friendship, :advice])
is_missing_dyad(layer_network(m2, :advice), 2, 3)        # true — round trip
```

[`as_multilayer`](@ref) stores the layer networks as given, so their
attributes and masks survive inside the `MultilayerNetwork`; its report
(`as_multilayer(nets, names; report=true)`) is lossless. None of the three
adapters reads a masked dyad's face value (`supports_missing` is `true`
for each); the guard against analysing a masked layer sits on the
estimator, which refuses it.

The estimation and simulation routines work directly on the layered
representation, whose within-layer dyads are exactly the free dyads of
the block-diagonal network.

## MultiNetwork

[`MultiNetwork`](@ref) is a container for independent networks of
possibly different sizes, with the descriptive [`CrossNetEdges`](@ref)
statistic.
