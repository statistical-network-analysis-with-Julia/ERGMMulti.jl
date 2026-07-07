# Multi-Network Structures

## MultilayerNetwork

[`MultilayerNetwork`](@ref) holds L same-sized, same-directedness
networks over one actor set. Access layers by index or name with
[`layer_network`](@ref); add ties with [`add_layer_edge!`](@ref).

## The combined network

[`combine_networks`](@ref) produces the `ergm.multi`-style block-diagonal
`Network`: `n × L` vertices where vertex `(l−1)·n + a` is actor `a`'s
copy in layer `l`, carrying `:layer` and `:actor` vertex attributes, with
every layer's edges inside its own block and no cross-block edges.
[`split_by_layer`](@ref) inverts the construction (and rejects
cross-block edges).

The estimation and simulation routines work directly on the layered
representation, whose within-layer dyads are exactly the free dyads of
the block-diagonal network.

## MultiNetwork

[`MultiNetwork`](@ref) is a container for independent networks of
possibly different sizes, with the descriptive [`CrossNetEdges`](@ref)
statistic.
