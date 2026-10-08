"""
    ERGMMulti.jl - ERGMs for Multiple and Multilayer Networks

Fits ERGMs to multilayer network data (the same actors observed on several
relations), following R `ergm.multi` (Krivitsky, Koehly & Marcum 2020):
the layers form a block-diagonal combined network with layer membership
attributes ([`combine_networks`](@ref)), the model's dyad universe is the
set of *within-layer* dyads, and layer-aware terms carry per-layer,
pooled, and cross-layer effects. Estimation ([`fit_ergm_multi`](@ref), the
R name [`ergm_multi`](@ref) being the same function) follows R's default
(`method=:auto`): maximum pseudo-likelihood over the within-layer dyads,
which is the exact MLE, when every term is dyad-independent, and
Monte-Carlo maximum likelihood otherwise, with support for per-layer
`offset` coefficients;
simulation
([`simulate_multi_ergm`](@ref)) uses the ERGM family's Metropolis kernel
restricted to within-layer dyads.

Also provides containers for multiple independent networks
(`MultiNetwork`) and descriptive statistics for multilevel designs
(`MultilevelNetwork`); neither enters a fit.
"""
module ERGMMulti

using Distributions
using ERGM
using Graphs
using LinearAlgebra
using NetworkCore
using PrecompileTools: @setup_workload, @compile_workload
using Random
using Statistics

# The ERGM.jl term protocol (shared statistic generics `name`/`compute` come
# from NetworkCore.jl through ERGM), the dependence and direction traits, the
# tuple-backed `TermSet`, and the ONE Metropolis toggle kernel every
# ERGM-family sampler runs on (`mh_toggle!`).
import ERGM: name, compute, change_stat, is_dyad_dependent, has_dyad_dependent,
             requires_directed, requires_undirected, TermSet, mh_toggle!
# The shared numerics and presentation helpers live in NetworkCore.jl: the
# Newton optimizer and the allocation-free logistic derivative builder, the
# one z → p helper, the one `se=` validator, the generic coefficient table
# `coeftable` returns, and the separation policy's caveat (the verdict itself
# and its warning come with the fit, from `ERGM.Extension.mple_fit_design`).
import NetworkCore: newton_fit, logistic_derivatives, z_pvalues, check_se,
                 CoefficientTable, missing_policies, separation_caveat
import StatsAPI
import StatsAPI: coef, coefnames, stderror, vcov, confint, loglikelihood, aic, bic,
                 nobs, dof, coeftable

# `gof` extends the ONE shared NetworkCore.jl generic (every model package adds
# methods for its own result types), so `gof(fit)` works uniformly across the
# ecosystem and loading several model packages never collides on the name.
import NetworkCore: gof

# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# ERGMMulti adds methods for `MultiERGMResult`; `fit_metadata(fit)` collects them.
import NetworkCore: estimand, objective, is_exact, se_method, missing_method,
                 approximations

# Data structures
export MultiNetwork, MultilayerNetwork, MultilevelNetwork
export add_layer!, add_layer_edge!, layer_network, layer_names
export add_cross_level_edge!

# Multilayer terms
export LayerEdges, LayerMutual, LayerTriangle, WithinLayer
export InterlayerDependence, MultiplexMutual
export CrossNetEdges
export change_stat_layer

# Multilevel descriptive statistics
export Nestedness, CrossLevelEdge, LevelHomophily

# Estimation: the harmonised `fit_<model>` name and the R name (one function)
export fit_ergm_multi, ergm_multi, MultiERGMModel, MultiERGMResult

# Simulation
export simulate_multi_ergm

# Diagnostics (`gof` is NetworkCore.jl's shared generic, extended with a method
# for MultiERGMResult)
export gof

# Utilities
export as_multilayer, combine_networks, split_by_layer

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using ERGMMulti`)
export coef, coefnames, stderror, vcov, confint, loglikelihood, aic, bic, nobs, dof,
       coeftable

# =============================================================================
# Data Structures
# =============================================================================

"""
    MultilayerNetwork{D}
    MultilayerNetwork(n::Int; directed::Bool=true) -> MultilayerNetwork{directed}

Several relations ("layers") on the same actor set. All layers share the
number of actors and the directedness `D`, which is a **type parameter**
(as it is for `Network{T,D}`): `layers` is a concretely typed
`Vector{Network{Int,D}}`, `is_directed(m)` reads `D`, and a layer of the
other directedness is refused by [`add_layer!`](@ref).

# Fields
- `n::Int`: Number of actors
- `layers::Vector{Network{Int,D}}`: One network per layer
- `layer_names::Vector{Symbol}`

# Example
```julia
using ERGMMulti, NetworkCore
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :friendship)
add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2)
is_directed(m)                       # true
m isa MultilayerNetwork{true}        # true
eltype(m.layers) === Network{Int,true}   # true
layer_names(m)                       # [:friendship, :advice]
```
"""
struct MultilayerNetwork{D}
    n::Int
    layers::Vector{Network{Int, D}}
    layer_names::Vector{Symbol}

    function MultilayerNetwork{D}(n::Int) where {D}
        D isa Bool || throw(ArgumentError(
            "the directedness parameter D of MultilayerNetwork{D} must be a Bool (got $D)"))
        n >= 0 || throw(ArgumentError("MultilayerNetwork: n must be non-negative (got $n)"))
        new{D}(n, Network{Int, D}[], Symbol[])
    end
end

MultilayerNetwork(n::Int; directed::Bool=true) = MultilayerNetwork{directed}(n)

"""
    is_directed(m::MultilayerNetwork{D}) -> Bool

Whether the layers are directed — the type parameter `D`, so the branch
costs nothing in the sampler and the design builder.
"""
Graphs.is_directed(::MultilayerNetwork{D}) where {D} = D
Graphs.is_directed(::Type{MultilayerNetwork{D}}) where {D} = D

n_layers(m::MultilayerNetwork) = length(m.layers)

"""
    layer_names(m::MultilayerNetwork) -> Vector{Symbol}

The layer names in index order — the mapping behind every integer layer
selector (`LayerEdges(2)` is the layer `layer_names(m)[2]`, and the
validation errors quote this list). Returns a copy; the network's own vector
is not exposed.

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(3; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
layer_names(m)                                # [:friendship, :advice]
findfirst(==(:advice), layer_names(m))        # 2 — the index `LayerEdges(2)` selects
```
"""
layer_names(m::MultilayerNetwork) = copy(m.layer_names)

_dirword(directed::Bool) = directed ? "directed" : "undirected"

function Base.show(io::IO, m::MultilayerNetwork)
    print(io, "MultilayerNetwork: $(m.n) actors, $(n_layers(m)) $(_dirword(is_directed(m))) ",
          "layer(s) $(Tuple(m.layer_names))")
end

"""
    add_layer!(m::MultilayerNetwork, name::Symbol;
               net::Union{Network,Nothing}=nothing) -> MultilayerNetwork

Add a layer, either empty or from an existing `Network` with matching size
and directedness. A network of the other directedness is refused with an
`ArgumentError` naming both (a `MultilayerNetwork{true}` holds directed
layers only, and vice versa), and so is a **two-mode (bipartite) network**:
ERGMMulti models one-mode layers only, and a bipartite layer would have its
impossible within-mode pairs enumerated as observed non-ties by the
within-layer MPLE (statnet's bipartite layer terms `b1dspL`, ... are not
implemented). A layer that *allows* self-loops (`loops=true`) is accepted
here; a layer that *contains* one is refused by [`MultiERGMModel`](@ref).

# Example
```julia
using ERGMMulti, NetworkCore
m = MultilayerNetwork(3; directed=false)
add_layer!(m, :a)                                    # empty undirected layer
net = network(3; directed=false); add_edge!(net, 1, 2)
add_layer!(m, :b; net=net)                           # from an existing network
try
    add_layer!(m, :c; net=network(3; directed=true))
catch e
    occursin("undirected", e.msg)                    # true
end
```
"""
function add_layer!(m::MultilayerNetwork{D}, lname::Symbol;
                    net::Union{Network, Nothing}=nothing) where {D}
    lname in m.layer_names &&
        throw(ArgumentError("layer :$lname already exists"))
    if isnothing(net)
        net = network(m.n; directed=D)
    else
        Int(nv(net)) == m.n ||
            throw(ArgumentError("layer :$lname must have $(m.n) vertices " *
                                "(got $(nv(net)))"))
        is_directed(net) == D ||
            throw(ArgumentError(
                "layer :$lname is $(_dirword(is_directed(net))) but the multilayer " *
                "network is $(_dirword(D)) (MultilayerNetwork{$D}); every layer " *
                "must be $(_dirword(D))"))
        net isa Network{Int} ||
            throw(ArgumentError("layer :$lname must be a Network{Int,$D} " *
                                "(got $(typeof(net)))"))
        is_two_mode(net) && throw(ArgumentError(
            "layer :$lname is two-mode (bipartite); ERGMMulti models one-mode " *
            "layers only — the within-layer dyad universe would enumerate the " *
            "impossible within-mode pairs as observed non-ties, so the fit is " *
            "refused instead (statnet's bipartite layer terms b1dspL, b2dspL, ... " *
            "are not implemented; see README 'Not implemented')."))
    end
    push!(m.layers, net)
    push!(m.layer_names, lname)
    return m
end

"""
    add_layer_edge!(m::MultilayerNetwork, layer, i, j) -> MultilayerNetwork

Add the edge `(i, j)` in the given layer (an index or a name) — `add_edge!`
on that layer's `Network`, so on an undirected `MultilayerNetwork{false}`
the pair is unordered and a repeated edge is a no-op.

An actor id outside `1:n` (a mistyped id, a 0-based id from Python data)
and a self-loop `(i, i)` on a layer built without `loops=true` are
**refused with an `ArgumentError`** naming the ids and the range — never
silently dropped, which is what `add_edge!`'s `false` return used to
become. An unknown layer name or an out-of-range layer index is an
`ArgumentError` too ([`layer_network`](@ref)).

# Example
```julia
using ERGMMulti, NetworkCore
m = MultilayerNetwork(3; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2)          # by name
add_layer_edge!(m, 2, 2, 3)                    # by index
ne(layer_network(m, :friendship)), ne(layer_network(m, :advice))   # (1, 1)
has_edge(layer_network(m, :advice), 3, 2)      # false — directed
try
    add_layer_edge!(m, :advice, 0, 2)          # 0-based id
catch e
    occursin("actor ids must lie in 1:3", e.msg)   # true
end
```
"""
function add_layer_edge!(m::MultilayerNetwork, layer, i::Int, j::Int)
    net = layer_network(m, layer)          # validates the layer index / name
    n = m.n
    (1 <= i <= n && 1 <= j <= n) || throw(ArgumentError(
        "add_layer_edge!: actor ids must lie in 1:$n (got ($i, $j)); the " *
        "multilayer network has $n actors. Ids are 1-based — a 0 usually " *
        "means 0-based data — and an edge is never dropped silently."))
    if i == j && !net.loops
        lname = layer isa Symbol ? layer : m.layer_names[layer]
        throw(ArgumentError(
            "add_layer_edge!: ($i, $i) is a self-loop, and layer :$lname was " *
            "built without `loops=true`; ERGMMulti models the off-diagonal " *
            "dyads only (a layer that allows loops is accepted here, but a " *
            "model on one that contains a loop is refused by MultiERGMModel)."))
    end
    add_edge!(net, i, j)
    return m
end

"""
    layer_network(m::MultilayerNetwork, layer) -> Network

The `Network{Int,D}` of a layer, by index or by name — the object itself,
not a copy, so attributes and missing-dyad masks set on it are seen by the
model. An unknown name, or an index outside `1:n_layers`, is an
`ArgumentError` naming the layers the network has (never a raw
`BoundsError`).

# Example
```julia
using ERGMMulti, NetworkCore
m = MultilayerNetwork(3; directed=false)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :advice, 1, 3)
layer_network(m, 2) === layer_network(m, :advice)     # true — the same Network
has_edge(layer_network(m, :advice), 3, 1)              # true — undirected
try
    layer_network(m, :trust)
catch e
    occursin("no layer named :trust", e.msg)           # true
end
try
    layer_network(m, 5)
catch e
    occursin("layer index 5 out of range", e.msg)      # true
end
```
"""
function layer_network(m::MultilayerNetwork, l::Int)
    1 <= l <= n_layers(m) || throw(ArgumentError(
        "layer index $l out of range: the network has $(n_layers(m)) layer" *
        "$(n_layers(m) == 1 ? "" : "s") ($(join((":" * String(s) for s in m.layer_names), ", "))); " *
        "layer indices must lie in 1:$(n_layers(m)) (see `layer_names(m)`)."))
    return m.layers[l]
end
function layer_network(m::MultilayerNetwork, lname::Symbol)
    idx = findfirst(==(lname), m.layer_names)
    isnothing(idx) && throw(ArgumentError("no layer named :$lname"))
    return m.layers[idx]
end

# =============================================================================
# Adapters: NetworkCore ↔ MultilayerNetwork ↔ the block-diagonal Network
# =============================================================================
#
# These three functions are conversions in the sense of the ecosystem's
# conversion contract (NetworkCore.jl `src/conversion.jl`; per-path invariants in
# NetworkCore.jl's `docs/src/guide/conversion_invariants.md`): preserve what the
# target can represent, reject or policy-gate what it cannot, report what was
# dropped (`report=true` returns `(result, ConversionReport)`). The missing-dyad
# mask is what matters most: a `Network` CAN carry it, so `combine_networks`
# carries every layer's mask into its block and `split_by_layer` carries it
# back — an unobserved within-layer dyad never becomes an observed absent one
# on the way through (it used to: the combined network was built from the edge
# sets alone and reported zero masked dyads). Nothing here reads a face value,
# which is what `supports_missing` says.

"""
    as_multilayer(nets::Vector{<:Network}, names::Vector{Symbol};
                  report=false) -> MultilayerNetwork

Build a multilayer network from same-sized networks. The directedness of the
first network sets the type parameter; every other layer must match
([`add_layer!`](@ref) refuses a mismatch). The layer networks are stored **as
given** — not copied — so their vertex/edge/network attributes and their
missing-dyad masks survive inside the `MultilayerNetwork` unchanged: the
conversion is lossless, and with `report=true` it returns
`(m, ::ConversionReport)` whose `is_lossless` is `true`.

# Example
```julia
using ERGMMulti, NetworkCore
a = network(4; directed=true); add_edge!(a, 1, 2); set_missing_dyad!(a, 3, 4)
b = network(4; directed=true); add_edge!(b, 2, 3)
m, rep = as_multilayer([a, b], [:friendship, :advice]; report=true)
is_lossless(rep)                                   # true
n_missing_dyads(layer_network(m, :friendship))     # 1 — the mask is kept
```
"""
function as_multilayer(nets::Vector{<:Network}, lnames::Vector{Symbol};
                       report::Bool=false)
    isempty(nets) && throw(ArgumentError("need at least one layer"))
    length(nets) == length(lnames) ||
        throw(ArgumentError("need one name per layer"))
    m = MultilayerNetwork(Int(nv(nets[1])); directed=is_directed(nets[1]))
    for (net, lname) in zip(nets, lnames)
        add_layer!(m, lname; net=net)
    end
    report || return m
    return m, ConversionReport(:Network, :MultilayerNetwork)
end

"""
    combine_networks(m::MultilayerNetwork; report=false) -> Network
    combine_networks(m; report=true) -> (Network, ConversionReport)

The **block-diagonal combined network** of `ergm.multi`'s `Layer()`
construct: one network with `n × L` vertices, where vertex
`(l-1)·n + a` is actor `a`'s copy in layer `l`. Each vertex carries the
`:layer` (layer index) and `:actor` (original actor ID) attributes, and
every layer's edges are placed within its own block. Cross-block dyads
are structurally empty — the model's dyad universe is the within-block
dyads only.

**The missing-dyad mask is preserved**: a masked (unobserved) dyad `(i, j)`
of layer `l` is masked at `(off + i, off + j)` of the combined network
(`off = (l-1)·n`), so `n_missing_dyads(combined)` is the sum over the
layers and [`split_by_layer`](@ref) restores every mask. An unobserved
within-layer tie never becomes an observed absent one on the way through.
The `loops` flag is carried when any layer allows self-loops.

What a block-diagonal network cannot hold is **dropped and reported**: the
layers' own vertex attributes (only `:layer`/`:actor` are carried), edge
attributes and network attributes. (A two-mode layer never gets this far:
[`add_layer!`](@ref) refuses it.) With
`report=true` the function returns `(combined, ::ConversionReport)` whose
`dropped_fields` name each dropped attribute (and the layer it came from,
in the reason), so a covariate that vanished is never silent.

# Example
```julia
using ERGMMulti, NetworkCore
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2)
set_missing_dyad!(layer_network(m, :advice), 2, 3)          # unobserved in layer 2
set_vertex_attribute!(layer_network(m, :friendship), :grp,
                      Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B"))
c, rep = combine_networks(m; report=true)
nv(c), ne(c)                              # (8, 1)
is_missing_dyad(c, 6, 7)                  # true — actor 2→3 in block 2
n_missing_dyads(c)                        # 1
dropped_fields(rep)                       # [:grp]
```
"""
function combine_networks(m::MultilayerNetwork{D}; report::Bool=false) where {D}
    L = n_layers(m)
    L >= 1 || throw(ArgumentError("no layers to combine"))
    n = m.n
    loops = any(net.loops for net in m.layers)
    combined = network(n * L; directed=D, loops=loops)

    layer_attr = Dict{Int, Any}()
    actor_attr = Dict{Int, Any}()
    for l in 1:L, a in 1:n
        v = (l - 1) * n + a
        layer_attr[v] = l
        actor_attr[v] = a
    end
    set_vertex_attribute!(combined, :layer, layer_attr)
    set_vertex_attribute!(combined, :actor, actor_attr)

    for l in 1:L
        off = (l - 1) * n
        net = m.layers[l]
        for e in edges(net)
            add_edge!(combined, off + src(e), off + dst(e))
        end
        # The mask travels with its block: unobserved stays unobserved
        for (i, j) in missing_dyads(net)
            set_missing_dyad!(combined, off + i, off + j)
        end
    end

    report || return combined
    return combined, _combine_report(m)
end

# What the block-diagonal network drops, attribute by attribute, with the
# layer it came from: the combined network carries only its own `:layer` and
# `:actor` vertex attributes. (No entry for a two-mode partition: `add_layer!`
# is the guard, so no layer of a MultilayerNetwork is bipartite.)
function _combine_report(m::MultilayerNetwork)
    rep = ConversionReport(:MultilayerNetwork, :Network)
    for (l, net) in enumerate(m.layers)
        where_ = "layer $l (:$(m.layer_names[l]))"
        for a in sort(list_vertex_attributes(net))
            record_drop!(rep, a, "vertex attribute of $where_: the block-diagonal " *
                                 "network carries only :layer and :actor")
        end
        for a in sort(list_edge_attributes(net))
            record_drop!(rep, a, "edge attribute of $where_: the block-diagonal " *
                                 "network carries edge sets only")
        end
        for a in sort(list_network_attributes(net))
            record_drop!(rep, a, "network attribute of $where_: the block-diagonal " *
                                 "network has no per-layer network attributes")
        end
    end
    return rep
end

"""
    split_by_layer(combined::Network, n::Int, L::Int;
                   names=Symbol[], report=false) -> MultilayerNetwork

Inverse of [`combine_networks`](@ref): recover the `L` layers of `n` actors
from a block-diagonal combined network. Every edge must lie within a block
(a cross-block edge is an `ArgumentError`), and so must every **masked
dyad**: the mask of block `l` is remapped onto layer `l`
(`is_missing_dyad(layer_l, i, j)` for a masked `(off + i, off + j)`), and
a masked cross-block dyad is refused with the same `ArgumentError` as a
cross-block edge — a dyad the block-diagonal model declares structurally
empty cannot also be unobserved. The `loops` flag is carried to every
layer.

With `report=true` returns `(m, ::ConversionReport)`: the combined
network's vertex attributes other than `:layer`/`:actor`, its edge
attributes and its network attributes cannot be attributed to a layer and
are dropped and named.

# Example
```julia
using ERGMMulti, NetworkCore
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :advice, 4, 3)
set_missing_dyad!(layer_network(m, :advice), 2, 3)
c = combine_networks(m)
m2 = split_by_layer(c, 4, 2; names=[:friendship, :advice])
has_edge(layer_network(m2, :advice), 4, 3)              # true
is_missing_dyad(layer_network(m2, :advice), 2, 3)       # true — round trip
```
"""
function split_by_layer(combined::Network, n::Int, L::Int;
                        names::Vector{Symbol}=Symbol[], report::Bool=false)
    Int(nv(combined)) == n * L ||
        throw(ArgumentError("combined network must have n × L vertices"))
    lnames = isempty(names) ? [Symbol("layer$(l)") for l in 1:L] : names
    length(lnames) == L ||
        throw(ArgumentError("need one name per layer: got $(length(lnames)) " *
                            "names for $L layers"))
    m = MultilayerNetwork(n; directed=is_directed(combined))
    for l in 1:L
        add_layer!(m, lnames[l];
                   net=network(n; directed=is_directed(combined), loops=combined.loops))
    end
    for e in edges(combined)
        i, j = Int(src(e)), Int(dst(e))
        li, lj = div(i - 1, n) + 1, div(j - 1, n) + 1
        li == lj ||
            throw(ArgumentError("combined network has a cross-block edge ($i, $j)"))
        add_layer_edge!(m, li, i - (li - 1) * n, j - (lj - 1) * n)
    end
    # The mask comes back with its block; a masked cross-block dyad is as
    # ill-formed as a cross-block edge
    for (i, j) in missing_dyads(combined)
        i, j = Int(i), Int(j)
        li, lj = div(i - 1, n) + 1, div(j - 1, n) + 1
        li == lj ||
            throw(ArgumentError("combined network has a masked cross-block dyad " *
                                "($i, $j): cross-block dyads are structurally " *
                                "empty in the block-diagonal construction and " *
                                "cannot be unobserved"))
        set_missing_dyad!(m.layers[li], i - (li - 1) * n, j - (li - 1) * n)
    end
    report || return m
    rep = ConversionReport(:Network, :MultilayerNetwork)
    for a in sort(list_vertex_attributes(combined))
        a in (:layer, :actor) && continue
        record_drop!(rep, a, "vertex attribute of the combined network: only " *
                             ":layer and :actor describe the block structure")
    end
    for a in sort(list_edge_attributes(combined))
        record_drop!(rep, a, "edge attribute of the combined network: the layers " *
                             "are rebuilt from the edge sets")
    end
    for a in sort(list_network_attributes(combined))
        record_drop!(rep, a, "network attribute of the combined network: it belongs " *
                             "to no single layer")
    end
    return m, rep
end

# The adapters never read a masked dyad's face value: the mask is carried
# across (`combine_networks`, `split_by_layer`) or kept in place
# (`as_multilayer`). That is the principled treatment the trait asks about.
NetworkCore.supports_missing(::typeof(as_multilayer)) = true
NetworkCore.supports_missing(::typeof(combine_networks)) = true
NetworkCore.supports_missing(::typeof(split_by_layer)) = true

"""
    MultiNetwork(networks::Vector{<:Network}, names::Vector{Symbol})

A collection of independent networks (possibly of different sizes and
directedness), for **descriptive pooling** with [`CrossNetEdges`](@ref).
It never enters a fit: `ergm.multi`'s covariate-driven `Networks()` models
(`N(~edges, ~x)`) are not implemented, and [`ergm_multi`](@ref) accepts a
[`MultilayerNetwork`](@ref) only.

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
a = network(4; directed=false); add_edge!(a, 1, 2)
b = network(6; directed=false); add_edge!(b, 1, 2); add_edge!(b, 3, 4)
mn = MultiNetwork([a, b], [:school1, :school2])
length(mn)                                    # 2
compute(CrossNetEdges(), mn)                  # 3.0 — edges pooled over the networks
```
"""
struct MultiNetwork
    networks::Vector{Network{Int}}
    names::Vector{Symbol}

    function MultiNetwork(nets::Vector{<:Network}, names::Vector{Symbol})
        length(nets) == length(names) ||
            throw(ArgumentError("need one name per network"))
        new(collect(Network{Int}, nets), names)
    end
end

Base.length(mn::MultiNetwork) = length(mn.networks)

function Base.show(io::IO, mn::MultiNetwork)
    parts = ("$(nm) ($(nv(net)) vertices, $(ne(net)) edges, $(_dirword(is_directed(net))))"
             for (nm, net) in zip(mn.names, mn.networks))
    print(io, "MultiNetwork: $(length(mn)) network$(length(mn) == 1 ? "" : "s") — ",
          join(parts, "; "))
end

"""
    MultilevelNetwork

A hierarchical design: one network per level plus membership maps and
explicit cross-level edges. The associated statistics (`Nestedness`,
`CrossLevelEdge`, `LevelHomophily`) are **descriptive**.

# Fields
- `level_nets::Vector{Network{Int}}`: One network per level
- `membership::Vector{Dict{Int,Int}}`: `membership[l][node] = group`, the
  level-(l+1) unit that level-l node `node` belongs to
- `cross_level_edges::Vector{NTuple{4,Int}}`: `(level_from, node_from,
  level_to, node_to)` ties added via [`add_cross_level_edge!`](@ref)

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
people = network(5; directed=false)
add_edge!(people, 1, 2); add_edge!(people, 2, 3); add_edge!(people, 3, 4)
orgs = network(2; directed=false)
membership = [Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2, 5 => 2)]   # person => organisation
ml = MultilevelNetwork([people, orgs], membership)
compute(LevelHomophily(1), ml)      # 2.0 — (1,2) and (3,4) stay inside one organisation
compute(Nestedness(1), ml)          # 0.666… — 2 of the 3 person ties
```
"""
struct MultilevelNetwork
    level_nets::Vector{Network{Int}}
    membership::Vector{Dict{Int, Int}}
    cross_level_edges::Vector{NTuple{4, Int}}

    function MultilevelNetwork(level_nets::Vector{<:Network},
                               membership::Vector{Dict{Int, Int}})
        length(membership) == length(level_nets) - 1 ||
            throw(ArgumentError("need one membership map per adjacent level pair"))
        new(collect(Network{Int}, level_nets), membership, NTuple{4, Int}[])
    end
end

n_levels(m::MultilevelNetwork) = length(m.level_nets)

function Base.show(io::IO, ml::MultilevelNetwork)
    L = n_levels(ml)
    sizes = join(("level $l: $(nv(net)) nodes, $(ne(net)) edges"
                  for (l, net) in enumerate(ml.level_nets)), "; ")
    print(io, "MultilevelNetwork: $L level$(L == 1 ? "" : "s") ($sizes), ",
          "$(length(ml.cross_level_edges)) cross-level edge",
          length(ml.cross_level_edges) == 1 ? "" : "s")
end

"""
    add_cross_level_edge!(m::MultilevelNetwork, level_from, node_from,
                          level_to, node_to)

Record a cross-level tie (e.g. a person–organization affiliation beyond
the nesting structure); counted by [`CrossLevelEdge`](@ref). A level the
network does not have is an `ArgumentError`.

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
people = network(3; directed=false)
orgs = network(2; directed=false)
ml = MultilevelNetwork([people, orgs], [Dict(1 => 1, 2 => 1, 3 => 2)])
add_cross_level_edge!(ml, 1, 3, 2, 1)     # person 3 is also affiliated with organisation 1
compute(CrossLevelEdge(), ml)             # 1.0
try
    add_cross_level_edge!(ml, 1, 1, 3, 1) # there is no level 3
catch e
    occursin("level out of range", e.msg) # true
end
```
"""
function add_cross_level_edge!(m::MultilevelNetwork, level_from::Int,
                               node_from::Int, level_to::Int, node_to::Int)
    (1 <= level_from <= n_levels(m) && 1 <= level_to <= n_levels(m)) ||
        throw(ArgumentError("level out of range"))
    push!(m.cross_level_edges, (level_from, node_from, level_to, node_to))
    return m
end

# =============================================================================
# Multilayer Terms
# =============================================================================
#
# Terms operate on a MultilayerNetwork. Each implements:
#   compute(term, m)::Float64
#   change_stat_layer(term, m, l, i, j)::Float64 — the ADD-DIRECTION change
#     in the statistic when edge (i,j) is added in layer l, independent of
#     that dyad's current state (the multilayer analogue of ERGM.jl's
#     change_stat convention).

"""
    change_stat_layer(term, m::MultilayerNetwork, l, i, j) -> Float64

Add-direction change statistic for adding edge (i,j) in layer `l`,
holding every other (layer, dyad) fixed. Must not depend on the dyad's
own current state (the multilayer analogue of ERGM.jl's `change_stat`
convention; the tests verify every term against brute-force recomputation).
The MPLE design has one row of these per within-layer dyad, and the
Metropolis sampler evaluates them once per proposal.

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(3; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :b, 1, 2)
change_stat_layer(LayerEdges(1), m, 1, 1, 2)                # 1.0 — an edge in layer 1
change_stat_layer(LayerEdges(1), m, 2, 1, 2)                # 0.0 — layer 2 is not selected
change_stat_layer(InterlayerDependence(1, 2), m, 1, 1, 2)   # 1.0 — 1→2 is already in layer 2
change_stat_layer(InterlayerDependence(1, 2), m, 1, 2, 1)   # 0.0
```
"""
function change_stat_layer end

# Layer selectors: an Int, a Vector{Int}, or Colon (all layers, pooled)
const LayerSel = Union{Int, Vector{Int}, Colon}

_in_layers(sel::Colon, l::Int) = true
_in_layers(sel::Int, l::Int) = l == sel
_in_layers(sel::Vector{Int}, l::Int) = l in sel

# The layer part of a coefficient label, as `ergm.multi` prints it: `L(A)~x`
# for one layer, `L((A,B))~x` for a term pooled over a list of layers (R's
# `L(~x, c(~A, ~B))`, double parentheses even for a one-element list),
# `L(A&B)~edges` for a co-occurrence and `L(A,B)~mutual` for `mutualL`. On a
# network the layers are named by `layer_names(m)`, as R names them by the
# `Layer()` list; without a network (a term not yet attached to data) they
# are named by index, which is also what R prints for an unnamed layer list
# (`L(1)~edges`), and a pool over every layer is `(:)`.
_layer_spec(sel::Int) = string(sel)
_layer_spec(sel::Vector{Int}) = "(" * join(sel, ",") * ")"
_layer_spec(::Colon) = "(:)"
_layer_spec(sel::Int, m) = String(m.layer_names[sel])
_layer_spec(sel::Vector{Int}, m) = "(" * join((String(m.layer_names[l]) for l in sel), ",") * ")"
_layer_spec(::Colon, m) = "(" * join(String.(m.layer_names), ",") * ")"

# The ERGM.jl sentence for a directed-only term on undirected data, reused
# verbatim so a migrant reads the same error from every package of the
# family. The branch is on the type parameter, so it is free in the hot loops
# of a directed model.
_directed_only_message(t) =
    "term '$(name(t))' is only defined for directed networks, but the " *
    "network is undirected. Remove the term or use a directed network " *
    "(R ergm raises the same error)."

@inline function _refuse_undirected(t, ::MultilayerNetwork{D}) where {D}
    D || _throw_directed_only(t)
    return nothing
end
@noinline _throw_directed_only(t) = throw(ArgumentError(_directed_only_message(t)))

"""
    LayerEdges(layers=:) <: AbstractERGMTerm

Edge count within the selected layer(s). With a layer index this is the
per-layer edges term (`ergm.multi`'s `L(~edges, ~A)`); with `:` (or a
vector) it pools the coefficient across layers — the `edges` statistic of
the block-diagonal combined network, one coefficient for all selected
layers. Dyad-independent; labelled as `ergm.multi` labels it, `L(A)~edges` for
a layer named `:A` and `L((A,B))~edges` for a pool (see [`WithinLayer`](@ref)).

# Example
```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2); add_layer_edge!(m, :friendship, 2, 3)
add_layer_edge!(m, :advice, 1, 2)
compute(LayerEdges(1), m)          # 2.0 — layer 1 only
compute(LayerEdges(), m)           # 3.0 — pooled over all layers
compute(LayerEdges([1, 2]), m)     # 3.0 — the same pool, spelled out
name(LayerEdges(2), m)             # "L(advice)~edges"
name(LayerEdges(), m)              # "L((friendship,advice))~edges"
name(LayerEdges(2))                # "L(2)~edges" — no network: layers by index
```
"""
struct LayerEdges <: AbstractERGMTerm
    layers::LayerSel
    LayerEdges(layers::LayerSel=Colon()) = new(layers)
end

# Layers are selected by INDEX in terms (the 0.2 decision: a term is built
# before it meets a network, so a name cannot be resolved there). A Symbol,
# the selector every other verb of the API accepts, is refused with the
# migration note instead of a MethodError.
function _symbol_selector_error(T, sel; usage="$T(k)")
    s = sel isa Symbol ? ":" * String(sel) : "[" * join((":" * String(x) for x in sel), ", ") * "]"
    ex = sel isa Symbol ? ":" * String(sel) : ":" * String(first(sel))
    throw(ArgumentError(
        "$T selects layers by index, not by name (got $s): use $usage with " *
        "k = findfirst(==($ex), layer_names(m)) — the index of that layer in " *
        "`layer_names(m)`" * (T in ("LayerEdges", "LayerMutual", "LayerTriangle") ?
        " — or a vector of such indices" : "") * ". (Layer NAMES are accepted " *
        "by `add_layer_edge!` and `layer_network`; terms are built before they " *
        "meet a network, so they carry indices.)"))
end
LayerEdges(sel::Union{Symbol, AbstractVector{Symbol}}) = _symbol_selector_error("LayerEdges", sel)

name(t::LayerEdges) = "L($(_layer_spec(t.layers)))~edges"
function name(t::LayerEdges, m::MultilayerNetwork)
    _check_layer_selector(t, t.layers, m)
    return "L($(_layer_spec(t.layers, m)))~edges"
end

function compute(t::LayerEdges, m::MultilayerNetwork)
    return sum(Float64(ne(m.layers[l])) for l in 1:n_layers(m)
               if _in_layers(t.layers, l); init=0.0)
end

change_stat_layer(t::LayerEdges, m::MultilayerNetwork, l::Int, i::Int, j::Int) =
    _in_layers(t.layers, l) ? 1.0 : 0.0

"""
    LayerMutual(layers=:) <: AbstractERGMTerm

Mutual (reciprocated) dyads within the selected layer(s). **Directed
multilayer networks only**: `requires_directed(LayerMutual()) == true`, so
on a `MultilayerNetwork{false}` model construction, `compute` and
`change_stat_layer` all throw the same `ArgumentError` ERGM.jl raises for
`Mutual()` on an undirected network (it used to return `0.0` silently).
`ergm.multi`'s `L(~mutual, ~A)`; labelled `L(A)~mutual`.

# Example
```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(3; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 1); add_layer_edge!(m, :a, 2, 3)
compute(LayerMutual(1), m)                 # 1.0 — the (1,2) dyad is reciprocated
compute(LayerMutual(), m)                  # 1.0 — pooled; layer :b is empty
change_stat_layer(LayerMutual(1), m, 1, 3, 2)   # 1.0 — 3→2 would reciprocate 2→3
u = MultilayerNetwork(3; directed=false); add_layer!(u, :a)
try
    compute(LayerMutual(1), u)             # refused, never a silent 0.0
catch e
    occursin("only defined for directed networks", e.msg)   # true
end
```
"""
struct LayerMutual <: AbstractERGMTerm
    layers::LayerSel
    LayerMutual(layers::LayerSel=Colon()) = new(layers)
end
LayerMutual(sel::Union{Symbol, AbstractVector{Symbol}}) = _symbol_selector_error("LayerMutual", sel)

name(t::LayerMutual) = "L($(_layer_spec(t.layers)))~mutual"
function name(t::LayerMutual, m::MultilayerNetwork)
    _check_layer_selector(t, t.layers, m)
    return "L($(_layer_spec(t.layers, m)))~mutual"
end
requires_directed(::LayerMutual) = true

function compute(t::LayerMutual, m::MultilayerNetwork)
    _refuse_undirected(t, m)
    total = 0.0
    for l in 1:n_layers(m)
        _in_layers(t.layers, l) || continue
        total += compute(Mutual(), m.layers[l])
    end
    return total
end

function change_stat_layer(t::LayerMutual, m::MultilayerNetwork, l::Int, i::Int, j::Int)
    _refuse_undirected(t, m)
    _in_layers(t.layers, l) || return 0.0
    return change_stat(Mutual(), m.layers[l], i, j)
end

"""
    LayerTriangle(layers=:) <: AbstractERGMTerm

Triangle count within the selected layer(s), using ERGM.jl's `Triangle`
statistic per layer (`ergm.multi`'s `L(~triangle, ~A)`; labelled
`L(A)~triangle`). Dyad-dependent, so a fit containing it carries the
pseudo-likelihood caveat.

# Example
```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(4; directed=false)
add_layer!(m, :a); add_layer!(m, :b)
for (i, j) in ((1, 2), (2, 3), (1, 3), (3, 4))
    add_layer_edge!(m, :a, i, j)
end
add_layer_edge!(m, :b, 1, 2)
compute(LayerTriangle(1), m)                     # 1.0
change_stat_layer(LayerTriangle(1), m, 1, 2, 4)  # 1.0 — adding 2-4 closes {2,3,4}
change_stat_layer(LayerTriangle(1), m, 2, 2, 4)  # 0.0 — layer 2 is not selected
```
"""
struct LayerTriangle <: AbstractERGMTerm
    layers::LayerSel
    LayerTriangle(layers::LayerSel=Colon()) = new(layers)
end
LayerTriangle(sel::Union{Symbol, AbstractVector{Symbol}}) = _symbol_selector_error("LayerTriangle", sel)

name(t::LayerTriangle) = "L($(_layer_spec(t.layers)))~triangle"
function name(t::LayerTriangle, m::MultilayerNetwork)
    _check_layer_selector(t, t.layers, m)
    return "L($(_layer_spec(t.layers, m)))~triangle"
end

function compute(t::LayerTriangle, m::MultilayerNetwork)
    total = 0.0
    for l in 1:n_layers(m)
        _in_layers(t.layers, l) || continue
        total += compute(Triangle(), m.layers[l])
    end
    return total
end

function change_stat_layer(t::LayerTriangle, m::MultilayerNetwork, l::Int, i::Int, j::Int)
    _in_layers(t.layers, l) || return 0.0
    return change_stat(Triangle(), m.layers[l], i, j)
end

"""
    WithinLayer(term, layer) <: AbstractERGMTerm
    WithinLayer(term, layers)        # a vector of layer indices, or `:`

Lift **any ERGM.jl term** into the layers of a multilayer network.

- `WithinLayer(term, l)` with one layer index is the statistic (and its
  add-direction change statistic) of `term` evaluated on layer `l`'s network
  — `ergm.multi`'s `L(~term, ~A)`.
- `WithinLayer(term, [l1, l2, ...])` (or `:` for every layer) **pools** the
  term across the selected layers: the statistic is the *sum* of the term's
  per-layer statistics, with one coefficient — `ergm.multi`'s
  `L(~term, c(~A, ~B))` (pinned against R in `pooled_layer_terms.toml`).
  `WithinLayer(Triangle(), [1, 2])` is `LayerTriangle([1, 2])`.

The wrapped term is **validated against every selected layer** when a
[`MultiERGMModel`](@ref) is built (through the public
`ERGM.Extension.validate_formula`): a vertex attribute the layer lacks, or has on
only some vertices, an `EdgeCov` of the wrong size, and a direction
requirement the layer does not meet (`Kstar`, `GWDegree`, `Degree` are
undirected-only; `Mutual`, `OStar`, `IStar`, `GWODegree`, ... directed-only)
raise ERGM.jl's own actionable errors instead of a silent `0.0`. Attribute
terms are then materialized per layer (`ERGM.Extension.materialize`), so their change
statistics read a dense snapshot rather than a `Dict` per toggle, and a term
that expands into several statistics (`NodeFactor(:g)` with several levels,
`Degree(0:2)`) contributes one coefficient per statistic, as in ERGM.jl. A
pooled expanding term must expand identically on every selected layer (the
same attribute levels); otherwise the constructor refuses, naming the layers.

Refused, with an `ArgumentError`: an `ERGM.Offset` term (fix a multilayer
coefficient with the `offsets=` keyword instead), an empty layer vector and a
repeated layer index.

The coefficient label is `ergm.multi`'s: `"L(<layer>)~" * name(term,
layer_network)`, with the layer named by `layer_names(m)` and the wrapped
term's direction-aware ERGM.jl/R label. On layers named `:a` and `:b`, a
directed layer's `WithinLayer(GWESP(0.5), 1)` is `L(a)~gwesp.OTP.fixed.0.5`,
an undirected one `L(a)~gwesp.fixed.0.5`, and the pooled forms
(`WithinLayer(term, [1, 2])`, `WithinLayer(term, :)`, R's
`L(~term, c(~a, ~b))`) are `L((a,b))~gwesp.fixed.0.5`. Without a network
the layers are named by index (`L(1)~…`, `L((1,2))~…`, `L((:))~…`).

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 1)
add_layer_edge!(m, :b, 3, 4); add_layer_edge!(m, :b, 4, 3)
compute(WithinLayer(Mutual(), 1), m)           # 1.0
compute(WithinLayer(Mutual(), [1, 2]), m)      # 2.0 — pooled: one coefficient, both layers
compute(WithinLayer(Mutual(), :), m)           # 2.0
name(WithinLayer(GWESP(0.5), 1), m)            # "L(a)~gwesp.OTP.fixed.0.5"
name(WithinLayer(GWESP(0.5), [1, 2]), m)       # "L((a,b))~gwesp.OTP.fixed.0.5"
try
    MultiERGMModel([WithinLayer(Kstar(2), 1)], m)   # undirected-only term
catch e
    occursin("OStar(2)", e.msg)                # true — ERGM.jl's own hint
end
```
"""
struct WithinLayer{T <: AbstractERGMTerm, S <: LayerSel} <: AbstractERGMTerm
    term::T
    layer::S

    function WithinLayer(term::T, layer::S) where {T <: AbstractERGMTerm, S <: LayerSel}
        term isa ERGM.Offset && throw(ArgumentError(
            "WithinLayer does not take an `Offset(...)` term: fix a multilayer " *
            "coefficient with the `offsets=` keyword of `ergm_multi` / " *
            "`MultiERGMModel` (e.g. `offsets=Dict(k => value)` for the k-th " *
            "term), which is ergm.multi's offset mechanism here."))
        if layer isa Vector{Int}
            allunique(layer) || throw(ArgumentError(
                "WithinLayer: the layer vector $(layer) repeats a layer; each " *
                "selected layer is counted once (pass distinct indices)."))
        end
        new{T, S}(term, layer)
    end
end
WithinLayer(term::AbstractERGMTerm, layers::AbstractVector{<:Integer}) =
    WithinLayer(term, collect(Int, layers))
WithinLayer(term::AbstractERGMTerm, layer::Union{Symbol, AbstractVector{Symbol}}) =
    _symbol_selector_error("WithinLayer", layer; usage="WithinLayer(term, k)")

# The materialized form of a POOLED within-layer term: ERGM.jl's
# `ERGM.Extension.materialize` snapshots the attributes of ONE network, so a term pooled
# over several layers carries one materialized twin per selected layer
# (`slot[l]` is its position, 0 for an unselected layer). Internal — built
# only by `_materialize_multilayer`; it reports the wrapped term's label and
# traits.
struct _PerLayer{T <: AbstractERGMTerm} <: AbstractERGMTerm
    terms::Vector{T}
    slot::Vector{Int}
end
name(t::_PerLayer) = name(first(t.terms))
name(t::_PerLayer, net::Network) = name(first(t.terms), net)
is_dyad_dependent(t::_PerLayer) = any(is_dyad_dependent, t.terms)
requires_directed(t::_PerLayer) = requires_directed(first(t.terms))
requires_undirected(t::_PerLayer) = requires_undirected(first(t.terms))

# The term evaluated on layer `l`, and whether layer `l` is selected
@inline _layer_term(t::AbstractERGMTerm, l::Int) = t
@inline _layer_term(t::_PerLayer, l::Int) = @inbounds t.terms[t.slot[l]]
@inline _selects(t::WithinLayer, l::Int) = _in_layers(t.layer, l)
@inline _selects(t::WithinLayer{<:_PerLayer}, l::Int) = @inbounds(t.term.slot[l]) != 0

# The layers a selector picks on `m`, in order
_selected_layers(sel::Int, m::MultilayerNetwork) = (sel,)
_selected_layers(sel::Vector{Int}, m::MultilayerNetwork) = sel
_selected_layers(::Colon, m::MultilayerNetwork) = 1:n_layers(m)

name(t::WithinLayer) = "L($(_layer_spec(t.layer)))~$(name(t.term))"

"""
    name(t::WithinLayer, m::MultilayerNetwork) -> String

The coefficient label of a within-layer term on `m`: `"L<layers>."` followed
by the wrapped term's label *on that layer's network* (`name(term, net)`),
so it agrees with ERGM.jl and R on direction-dependent labels. A pooled term
(`WithinLayer(term, [1, 2])`, `WithinLayer(term, :)`) is labelled
`L((A,B))~`, as `ergm.multi` labels `L(~term, c(~A, ~B))`.
"""
function name(t::WithinLayer, m::MultilayerNetwork)
    _check_layer_selector(t, t.layer, m)
    l1 = first(_selected_layers(t.layer, m))
    return "L($(_layer_spec(t.layer, m)))~" * name(_layer_term(t.term, l1), m.layers[l1])
end

requires_directed(t::WithinLayer) = requires_directed(t.term)
requires_undirected(t::WithinLayer) = requires_undirected(t.term)

compute(t::WithinLayer{<:AbstractERGMTerm, Int}, m::MultilayerNetwork) =
    compute(t.term, m.layers[t.layer])

function compute(t::WithinLayer, m::MultilayerNetwork)
    total = 0.0
    for l in _selected_layers(t.layer, m)
        total += compute(_layer_term(t.term, l), m.layers[l])
    end
    return total
end

change_stat_layer(t::WithinLayer{<:AbstractERGMTerm, Int}, m::MultilayerNetwork,
                  l::Int, i::Int, j::Int) =
    l == t.layer ? change_stat(t.term, m.layers[l], i, j) : 0.0

change_stat_layer(t::WithinLayer, m::MultilayerNetwork, l::Int, i::Int, j::Int) =
    _selects(t, l) ? change_stat(_layer_term(t.term, l), m.layers[l], i, j) : 0.0

"""
    InterlayerDependence(l1, l2) <: AbstractERGMTerm

Cross-layer co-occurrence: the number of dyads (i, j) with an edge in
both layers `l1` and `l2` (ordered dyads for directed networks). A
positive coefficient means a tie in one layer predicts the same tie in
the other. This is `ergm.multi`'s `L(~edges, ~A&B)` — the only layer-logic
conjunction implemented (see "Not implemented" in the README); labelled
`L(A&B)~edges`, as in R. Dyad-dependent over the `(layer, i, j)` universe.

# Example
```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(3; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2); add_layer_edge!(m, :friendship, 2, 3)
add_layer_edge!(m, :advice, 1, 2);     add_layer_edge!(m, :advice, 3, 2)
compute(InterlayerDependence(1, 2), m)                     # 1.0 — only 1→2 is in both
change_stat_layer(InterlayerDependence(1, 2), m, 1, 3, 2)  # 1.0 — 3→2 exists in :advice
name(InterlayerDependence(1, 2), m)                        # "L(friendship&advice)~edges"
```
"""
struct InterlayerDependence <: AbstractERGMTerm
    l1::Int
    l2::Int

    function InterlayerDependence(l1::Int, l2::Int)
        l1 != l2 || throw(ArgumentError("layers must differ"))
        new(l1, l2)
    end
end
InterlayerDependence(l1::Union{Int, Symbol}, l2::Union{Int, Symbol}) =
    _symbol_selector_error("InterlayerDependence", l1 isa Symbol ? l1 : l2;
                           usage="InterlayerDependence(k1, k2)")

name(t::InterlayerDependence) = "L($(t.l1)&$(t.l2))~edges"
function name(t::InterlayerDependence, m::MultilayerNetwork)
    _check_layer_index(t, t.l1, m); _check_layer_index(t, t.l2, m)
    return "L($(_layer_spec(t.l1, m))&$(_layer_spec(t.l2, m)))~edges"
end

function compute(t::InterlayerDependence, m::MultilayerNetwork)
    a, b = m.layers[t.l1], m.layers[t.l2]
    total = 0.0
    for e in edges(a)
        has_edge(b, src(e), dst(e)) && (total += 1.0)
    end
    return total
end

function change_stat_layer(t::InterlayerDependence, m::MultilayerNetwork,
                           l::Int, i::Int, j::Int)
    if l == t.l1
        return has_edge(m.layers[t.l2], i, j) ? 1.0 : 0.0
    elseif l == t.l2
        return has_edge(m.layers[t.l1], i, j) ? 1.0 : 0.0
    end
    return 0.0
end

"""
    MultiplexMutual(l1, l2) <: AbstractERGMTerm

Cross-layer reciprocity: the number of ordered dyads (i, j) with `i→j` in
layer `l1` and `j→i` in layer `l2` (e.g. friendship reciprocated by
advice). **Directed networks only**: `requires_directed(MultiplexMutual(1, 2))
== true`, and on a `MultilayerNetwork{false}` model construction, `compute`
and `change_stat_layer` all throw ERGM.jl's directed-only `ArgumentError`
(it used to return `0.0` silently). This is `ergm.multi`'s
`mutualL(Ls = list(~A, ~B))`; labelled `L(A,B)~mutual`, as in R.

# Example
```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(3; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2)      # 1 names 2 a friend ...
add_layer_edge!(m, :advice, 2, 1)          # ... and 2 seeks 1's advice
compute(MultiplexMutual(1, 2), m)          # 1.0
compute(InterlayerDependence(1, 2), m)     # 0.0 — different dyads, no co-occurrence
u = MultilayerNetwork(3; directed=false); add_layer!(u, :a); add_layer!(u, :b)
try
    MultiERGMModel([MultiplexMutual(1, 2)], u)
catch e
    occursin("only defined for directed networks", e.msg)   # true
end
```
"""
struct MultiplexMutual <: AbstractERGMTerm
    l1::Int
    l2::Int

    function MultiplexMutual(l1::Int, l2::Int)
        l1 != l2 || throw(ArgumentError("layers must differ; use LayerMutual within a layer"))
        new(l1, l2)
    end
end
MultiplexMutual(l1::Union{Int, Symbol}, l2::Union{Int, Symbol}) =
    _symbol_selector_error("MultiplexMutual", l1 isa Symbol ? l1 : l2;
                           usage="MultiplexMutual(k1, k2)")

name(t::MultiplexMutual) = "L($(t.l1),$(t.l2))~mutual"
function name(t::MultiplexMutual, m::MultilayerNetwork)
    _check_layer_index(t, t.l1, m); _check_layer_index(t, t.l2, m)
    return "L($(_layer_spec(t.l1, m)),$(_layer_spec(t.l2, m)))~mutual"
end
requires_directed(::MultiplexMutual) = true

function compute(t::MultiplexMutual, m::MultilayerNetwork)
    _refuse_undirected(t, m)
    a, b = m.layers[t.l1], m.layers[t.l2]
    total = 0.0
    for e in edges(a)
        has_edge(b, dst(e), src(e)) && (total += 1.0)
    end
    return total
end

function change_stat_layer(t::MultiplexMutual, m::MultilayerNetwork,
                           l::Int, i::Int, j::Int)
    _refuse_undirected(t, m)
    if l == t.l1
        return has_edge(m.layers[t.l2], j, i) ? 1.0 : 0.0
    elseif l == t.l2
        return has_edge(m.layers[t.l1], j, i) ? 1.0 : 0.0
    end
    return 0.0
end

# Dependence classification (extends ERGM.is_dyad_dependent, whose fallback
# is the conservative `true`). In the multilayer model the "dyads" are
# (layer, i, j) triples: a term is dyad-dependent when its change statistic
# depends on the state of *any other* (layer, dyad) — so the cross-layer
# terms (`InterlayerDependence`, `MultiplexMutual`) and the within-layer
# structural terms (`LayerMutual`, `LayerTriangle`) are dyad-dependent
# (covered by the fallback), while pure edge-count terms are not.
is_dyad_dependent(::LayerEdges) = false
is_dyad_dependent(t::WithinLayer) = is_dyad_dependent(t.term)

# The add-direction change statistics of every term of a TermSet for the
# multilayer move (l, i, j), filled in place. Generated per term count, like
# ERGM.jl's `change_stat_all!`: one straight-line sequence of statically
# dispatched calls, no `map` over the tuple (which falls back to a boxed
# Vector{Any} from 32 terms on) and no dynamic dispatch through an
# abstractly typed vector. The one row of the MPLE design, and the one
# `change!` of the Metropolis kernel.
@generated function _change_stat_layer_all!(dest::AbstractVector{Float64}, ts::TermSet{T},
                                            m::MultilayerNetwork, l::Int, i::Int,
                                            j::Int) where {T <: Tuple}
    p = length(T.parameters)
    body = [:(dest[$k] = change_stat_layer(ts.terms[$k], m, l, i, j)) for k in 1:p]
    return quote
        $(body...)
        return dest
    end
end

"""
    CrossNetEdges <: AbstractERGMTerm

Total edge count across the networks of a [`MultiNetwork`](@ref)
(descriptive pooling over independent networks; it does not enter a fit).

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
a = network(4; directed=false); add_edge!(a, 1, 2)
b = network(6; directed=true);  add_edge!(b, 1, 2); add_edge!(b, 3, 4)
compute(CrossNetEdges(), MultiNetwork([a, b], [:school1, :school2]))   # 3.0
```
"""
struct CrossNetEdges <: AbstractERGMTerm end

name(::CrossNetEdges) = "crossnet.edges"

compute(::CrossNetEdges, mn::MultiNetwork) =
    sum(Float64(ne(net)) for net in mn.networks; init=0.0)

# =============================================================================
# Multilevel descriptive statistics
# =============================================================================

"""
    Nestedness(level) <: AbstractERGMTerm

Descriptive: the proportion of level-`level` edges whose endpoints share
the same higher-level unit (`0.0` when the level has no edge between nodes
with a recorded membership). Never enters a fit; `level` must have a parent
level.

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
people = network(4; directed=false)
add_edge!(people, 1, 2); add_edge!(people, 2, 3); add_edge!(people, 3, 4)
orgs = network(2; directed=false)
ml = MultilevelNetwork([people, orgs], [Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2)])
compute(Nestedness(1), ml)          # 0.666… — (1,2) and (3,4) are within-organisation
```
"""
struct Nestedness <: AbstractERGMTerm
    level::Int
end

name(t::Nestedness) = "nestedness.$(t.level)"

function compute(t::Nestedness, m::MultilevelNetwork)
    t.level < n_levels(m) ||
        throw(ArgumentError("nestedness needs a level with a parent level"))
    net = m.level_nets[t.level]
    mem = m.membership[t.level]
    total = 0
    nested = 0
    for e in edges(net)
        i, j = Int(src(e)), Int(dst(e))
        (haskey(mem, i) && haskey(mem, j)) || continue
        total += 1
        mem[i] == mem[j] && (nested += 1)
    end
    return total == 0 ? 0.0 : nested / total
end

"""
    CrossLevelEdge <: AbstractERGMTerm

Descriptive: the number of explicit cross-level edges (added with
[`add_cross_level_edge!`](@ref)). Never enters a fit.

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
ml = MultilevelNetwork([network(3; directed=false), network(2; directed=false)],
                       [Dict(1 => 1, 2 => 1, 3 => 2)])
compute(CrossLevelEdge(), ml)             # 0.0
add_cross_level_edge!(ml, 1, 3, 2, 1)
compute(CrossLevelEdge(), ml)             # 1.0
```
"""
struct CrossLevelEdge <: AbstractERGMTerm end

name(::CrossLevelEdge) = "crosslevel.edges"

compute(::CrossLevelEdge, m::MultilevelNetwork) =
    Float64(length(m.cross_level_edges))

"""
    LevelHomophily(level) <: AbstractERGMTerm

Descriptive: the number of level-`level` edges between nodes belonging to
the same higher-level unit. Membership is looked up by the level's own
node IDs (a missing membership excludes the edge). Never enters a fit;
[`Nestedness`](@ref) is the same count as a proportion.

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
people = network(4; directed=false)
add_edge!(people, 1, 2); add_edge!(people, 2, 3); add_edge!(people, 3, 4)
orgs = network(2; directed=false)
ml = MultilevelNetwork([people, orgs], [Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2)])
compute(LevelHomophily(1), ml)      # 2.0 — (1,2) and (3,4); (2,3) crosses organisations
```
"""
struct LevelHomophily <: AbstractERGMTerm
    level::Int
end

name(t::LevelHomophily) = "levelhomophily.$(t.level)"

function compute(t::LevelHomophily, m::MultilevelNetwork)
    t.level < n_levels(m) ||
        throw(ArgumentError("level homophily needs a level with a parent level"))
    net = m.level_nets[t.level]
    mem = m.membership[t.level]
    total = 0.0
    for e in edges(net)
        i, j = Int(src(e)), Int(dst(e))
        (haskey(mem, i) && haskey(mem, j)) || continue
        mem[i] == mem[j] && (total += 1.0)
    end
    return total
end

# =============================================================================
# Model construction: validation and materialization
# =============================================================================
#
# Mirrors `ERGMModel`'s inner constructor: every user error that would
# otherwise produce a silently wrong fit — a layer index the network does not
# have (which used to yield all-zero coefficients, NaN standard errors and
# `converged = false`), a within-layer term whose attribute the layer lacks
# (which used to compute 0.0), a direction requirement the layers do not meet
# — is an ArgumentError at construction, before any fitting.

function _layer_range_error(t, l::Int, m::MultilayerNetwork)
    L = n_layers(m)
    names_str = join((":" * String(s) for s in m.layer_names), ", ")
    throw(ArgumentError(
        "term '$(name(t))' refers to layer $l, but the network has $L layer" *
        "$(L == 1 ? "" : "s") ($names_str); layer indices must lie in 1:$L " *
        "(see `layer_names(m)` for the index of each layer)."))
end

function _check_layer_index(t, l::Int, m::MultilayerNetwork)
    1 <= l <= n_layers(m) || _layer_range_error(t, l, m)
    return nothing
end

_check_layer_selector(t, sel::Colon, m::MultilayerNetwork) = nothing
_check_layer_selector(t, sel::Int, m::MultilayerNetwork) = _check_layer_index(t, sel, m)
function _check_layer_selector(t, sel::Vector{Int}, m::MultilayerNetwork)
    isempty(sel) && throw(ArgumentError(
        "term '$(name(t))' selects no layer (empty layer vector); pass a layer " *
        "index, a non-empty vector of indices, or `:` for all layers."))
    for l in sel
        _check_layer_index(t, l, m)
    end
    return nothing
end

# Direction requirements of the multilayer terms themselves, with ERGM.jl's
# sentences. (Within-layer terms are validated against their layer below.)
function _check_direction(t, m::MultilayerNetwork)
    if requires_directed(t) && !is_directed(m)
        throw(ArgumentError(_directed_only_message(t)))
    end
    if requires_undirected(t) && is_directed(m)
        throw(ArgumentError(
            "term '$(name(t))' is only defined for undirected networks, but the " *
            "network is directed (R ergm raises the same error)."))
    end
    return nothing
end

# Per-term validation. The generic fallback covers user-defined multilayer
# terms through the direction traits — after checking that the term IS a
# multilayer term, i.e. has a `change_stat_layer` method. A bare ERGM.jl term
# (`Mutual()` where `WithinLayer(Mutual(), l)` was meant — the ergm.multi
# migrant's most likely slip, `L(~mutual, ~A)` → `Mutual()`) and a multilevel
# descriptive (`Nestedness`) used to pass this check and die inside the
# design builder with a MethodError.
function _validate_multilayer_term(t::AbstractERGMTerm, m::MultilayerNetwork)
    _has_multilayer_change_stat(t) || throw(ArgumentError(
        "term '$(name(t))' ($(nameof(typeof(t)))) has no multilayer change " *
        "statistic (`change_stat_layer`): it is a single-network ERGM.jl term " *
        "or a multilevel descriptive, not a multilayer term. In a multilayer " *
        "model lift an ERGM.jl term into a layer with " *
        "`WithinLayer($(nameof(typeof(t)))(...), l)` (the analogue of " *
        "ergm.multi's `L(~$(name(t)), ~A)`), or use the pooled/per-layer forms " *
        "`LayerEdges`, `LayerMutual`, `LayerTriangle` and the cross-layer " *
        "`InterlayerDependence`/`MultiplexMutual`; multilevel statistics never " *
        "enter a fit."))
    _check_direction(t, m)
end

_has_multilayer_change_stat(t::AbstractERGMTerm) =
    hasmethod(change_stat_layer, Tuple{typeof(t), MultilayerNetwork, Int, Int, Int})

for T in (:LayerEdges, :LayerMutual, :LayerTriangle)
    @eval function _validate_multilayer_term(t::$T, m::MultilayerNetwork)
        _check_layer_selector(t, t.layers, m)
        _check_direction(t, m)
    end
end

for T in (:InterlayerDependence, :MultiplexMutual)
    @eval function _validate_multilayer_term(t::$T, m::MultilayerNetwork)
        _check_layer_index(t, t.l1, m)
        _check_layer_index(t, t.l2, m)
        _check_direction(t, m)
    end
end

# A within-layer term is validated against ITS layer by ERGM.jl's own
# formula validator (attributes present on every vertex, covariate sizes,
# direction requirements), so the error a migrant reads is the one ERGM.jl
# and TERGM.jl raise for the same slip.
function _validate_multilayer_term(t::WithinLayer, m::MultilayerNetwork)
    _check_layer_selector(t, t.layer, m)
    for l in _selected_layers(t.layer, m)
        try
            ERGM.Extension.validate_formula(TermSet((t.term,)), m.layers[l])
        catch e
            (e isa ArgumentError && !(t.layer isa Int)) || rethrow()
            throw(ArgumentError("layer $l (:$(m.layer_names[l])): " * e.msg))
        end
    end
    return nothing
end

# Materialization: a within-layer attribute term takes a dense snapshot of
# its layer's attribute (`ERGM.Extension.materialize`); a term that expands into
# several statistics (multi-level `NodeFactor`, `Degree(0:2)`, multi-cell
# `NodeMix`) becomes one `WithinLayer` per statistic. A POOLED term is
# materialized once per selected layer (each twin snapshots its own layer's
# attributes) and carried as a `_PerLayer`; an expanding pooled term must
# expand identically on every selected layer. Every other term passes through.
_materialize_multilayer(t::AbstractERGMTerm, m::MultilayerNetwork) = t
function _materialize_multilayer(t::WithinLayer{<:AbstractERGMTerm, Int}, m::MultilayerNetwork)
    inner = ERGM.Extension.materialize(t.term, m.layers[t.layer])
    inner isa AbstractVector ? [WithinLayer(x, t.layer) for x in inner] :
                               WithinLayer(inner, t.layer)
end
function _materialize_multilayer(t::WithinLayer, m::MultilayerNetwork)
    sel = collect(Int, _selected_layers(t.layer, m))
    slot = zeros(Int, n_layers(m))
    for (k, l) in enumerate(sel)
        slot[l] = k
    end
    inners = [ERGM.Extension.materialize(t.term, m.layers[l]) for l in sel]
    if !(first(inners) isa AbstractVector)
        return WithinLayer(_PerLayer([x for x in inners], slot), t.layer)
    end
    # An expanding term: the k-th statistic of every layer must be the same
    # statistic (same label), or the pooled sum would add different things
    labels = [[name(x, m.layers[l]) for x in inner] for (inner, l) in zip(inners, sel)]
    all(==(first(labels)), labels) || throw(ArgumentError(
        "term '$(name(t.term))' pooled over layers $(join(sel, ", ")) expands " *
        "into different statistics on different layers (" *
        join(("layer $l: " * join(lb, ", ") for (l, lb) in zip(sel, labels)), "; ") *
        "): a pooled term needs the same attribute levels on every selected " *
        "layer. Give the layers the same levels, or use one " *
        "`WithinLayer(term, l)` per layer."))
    return [WithinLayer(_PerLayer([inner[k] for inner in inners], slot), t.layer)
            for k in eachindex(first(inners))]
end

function _collect_multilayer_terms(terms)
    terms isa AbstractERGMTerm && return AbstractERGMTerm[terms]
    terms isa AbstractVector || throw(ArgumentError(
        "terms must be a vector of ERGM terms (e.g. `[LayerEdges(), " *
        "InterlayerDependence(1, 2)]`); got $(typeof(terms))"))
    out = AbstractERGMTerm[]
    for (k, t) in enumerate(terms)
        t isa AbstractERGMTerm || throw(ArgumentError(
            "element $k of the term list is not an ERGM term: got $(repr(t)) of " *
            "type $(typeof(t))" *
            (t isa Type && t <: AbstractERGMTerm ?
             " — `$(nameof(t))` is a term type, not a term; did you mean " *
             "`$(nameof(t))()`?" : "") *
            ". Every element must be an `AbstractERGMTerm` (e.g. `LayerEdges()`, " *
            "`WithinLayer(NodeMatch(:grp), 1)`)."))
        push!(out, t)
    end
    isempty(out) && throw(ArgumentError(
        "the term list is empty; a multilayer ERGM needs at least one term " *
        "(e.g. `[LayerEdges()]`)"))
    return out
end

# =============================================================================
# Estimation
# =============================================================================

"""
    MultiERGMModel{D}
    MultiERGMModel(terms, m::MultilayerNetwork{D}; offsets=Dict{Int,Float64}())

Multilayer ERGM specification: terms, data, and any per-term offsets. `D` is
the directedness of the network (`is_directed(model)`), as for
`ERGM.ERGMModel{T,D}`.

The constructor **validates the formula against the data** before anything
is fitted, throwing an `ArgumentError` that names the fix:

- every layer a term refers to (`LayerEdges(3)`, `InterlayerDependence(1, 3)`,
  `WithinLayer(t, 3)`) must exist — the message lists `layer_names(m)`;
- every `WithinLayer(t, l)` is checked against layer `l` with ERGM.jl's own
  formula validator (`ERGM.Extension.validate_formula`): a missing or partial vertex
  attribute, an `EdgeCov` of the wrong size, and an undirected-only term
  (`Kstar`, `GWDegree`, `Degree`) on a directed layer or a directed-only one
  (`Mutual`, `OStar`, ...) on an undirected layer are refused;
- `LayerMutual` and `MultiplexMutual` are directed-only
  (`requires_directed`) and are refused on a `MultilayerNetwork{false}`;
- a bare ERGM.jl term (`Mutual()` where `WithinLayer(Mutual(), l)` was
  meant — ergm.multi's `L(~mutual, ~A)`) or a multilevel descriptive
  (`Nestedness`) has no multilayer change statistic and is refused with the
  `WithinLayer` fix spelled out (it used to fail at fit time with a
  `MethodError`);
- no layer may contain a self-loop (a `loops=true` layer with none is fine):
  the statistics would count it but the within-layer design, `nobs`, the
  sampler and `gof` never touch the diagonal — the same refusal as
  `ERGM.ERGMModel`'s ("R ergm warns \"This network contains loops\" here");
- the network must have at least one layer and at least two actors — a
  `MultilayerNetwork(0)` or `MultilayerNetwork(1)` has no within-layer dyad,
  so there is nothing to estimate or simulate (it used to run Newton on an
  empty design and return `coef = [0.0]`, `NaN` standard errors and two
  misleading non-convergence warnings);
- `offsets` must index terms (`1:length(terms)`), an offset term must not
  expand into several statistics (pin each level/degree with its own
  single-statistic term instead), and at least one coefficient must be free.

(A layer selected by *name* inside a term — `LayerEdges(:advice)` — is an
`ArgumentError` at the term's construction, before the model: terms carry
indices, and the message gives `findfirst(==(:advice), layer_names(m))`.)

Attribute terms inside `WithinLayer` are then materialized
(`ERGM.Extension.materialize`), so the stored `terms` — an `ERGM.TermSet` whose
`names` are the direction-aware coefficient labels
([`name`](@ref)`(term, m)`) — may hold more statistics than terms were
given (a multi-level `NodeFactor`, a `Degree(0:2)`); `offsets` keys are
remapped to the expanded positions.

# Fields
- `terms::ERGM.TermSet`: the (materialized, expanded) statistics and their labels
- `network::MultilayerNetwork{D}`
- `offsets::Dict{Int,Float64}`: fixed coefficients, keyed by statistic index

# Example
```julia
using ERGMMulti, ERGM, NetworkCore
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
model = MultiERGMModel([LayerEdges(1), WithinLayer(GWESP(0.5), 2)], m)
model.terms.names          # ["L(friendship)~edges", "L(advice)~gwesp.OTP.fixed.0.5"]
is_directed(model)         # true
try
    MultiERGMModel([LayerEdges(3)], m)
catch e
    occursin("layer 3", e.msg) && occursin(":advice", e.msg)   # true
end
```
"""
struct MultiERGMModel{D}
    terms::TermSet
    network::MultilayerNetwork{D}
    offsets::Dict{Int, Float64}

    function MultiERGMModel(terms, m::MultilayerNetwork{D};
                            offsets::AbstractDict=Dict{Int, Float64}()) where {D}
        n_layers(m) >= 1 || throw(ArgumentError("network has no layers"))
        _n_within_dyads(m) > 0 || throw(ArgumentError(
            "the network has no within-layer dyads (n = $(m.n) actor" *
            "$(m.n == 1 ? "" : "s"), $(n_layers(m)) layer" *
            "$(n_layers(m) == 1 ? "" : "s")); a multilayer ERGM needs at least " *
            "two actors — nothing to estimate or simulate"))
        _refuse_layer_self_loops(m)
        raw = _collect_multilayer_terms(terms)
        for t in raw
            _validate_multilayer_term(t, m)
        end

        # Offsets index the terms AS GIVEN; validate before expansion so the
        # message can name the term the user wrote.
        p_raw = length(raw)
        for (k, c) in offsets
            k isa Integer || throw(ArgumentError(
                "offset keys must be term indices (Int); got $(repr(k))"))
            1 <= k <= p_raw || throw(ArgumentError(
                "offset indices must reference terms (1:$p_raw); got $k"))
            (c isa Real && isfinite(c)) || throw(ArgumentError(
                "offset $k must be a finite coefficient; got $(repr(c))"))
        end

        # Materialize and expand, remembering which given term each statistic
        # came from so the offsets can follow it.
        expanded = AbstractERGMTerm[]
        origin = Int[]
        for (k, t) in enumerate(raw)
            mt = _materialize_multilayer(t, m)
            if mt isa AbstractVector
                append!(expanded, mt)
                append!(origin, fill(k, length(mt)))
            else
                push!(expanded, mt)
                push!(origin, k)
            end
        end
        ts = TermSet(Tuple(expanded), [name(t, m) for t in expanded])

        mapped = Dict{Int, Float64}()
        for (k, c) in offsets
            idx = findall(==(k), origin)
            length(idx) == 1 || throw(ArgumentError(
                "offset $k targets '$(name(raw[k]))', which expands into " *
                "$(length(idx)) statistics ($(join(ts.names[idx], ", "))); pin " *
                "each with its own single-statistic term (e.g. " *
                "`NodeFactor(:attr; level=x)`, `Degree(d)`) and offset those."))
            mapped[idx[1]] = Float64(c)
        end
        length(mapped) < length(expanded) || throw(ArgumentError(
            "all coefficients are offsets; nothing to estimate"))

        new{D}(ts, m, mapped)
    end
end

Graphs.is_directed(::MultiERGMModel{D}) where {D} = D

function Base.show(io::IO, model::MultiERGMModel{D}) where {D}
    m = model.network
    L = n_layers(m)
    print(io, "MultiERGMModel{$D}: $(m.n) actors, $L $(_dirword(D)) layer",
          L == 1 ? "" : "s", " $(Tuple(m.layer_names)); terms: ",
          join(model.terms.names, " + "))
    if !isempty(model.offsets)
        ks = sort!(collect(keys(model.offsets)))
        print(io, "; offsets: ", join(("$k => $(model.offsets[k])" for k in ks), ", "))
    end
    return nothing
end

# A self-loop is refused, as `ERGM.ERGMModel` does (`ERGM.Extension.require_supported_network`):
# the statistics would count it, but the within-layer MPLE design, `nobs`,
# the sampler's proposals and `gof`'s simulations range over the
# off-diagonal dyads only — so the observed statistic would be compared
# against a model that cannot reproduce it. A layer built with `loops=true`
# that holds no loop is accepted, as in ERGM.jl.
function _refuse_layer_self_loops(m::MultilayerNetwork)
    for (l, net) in enumerate(m.layers)
        loops = [Int(v) for v in vertices(net) if has_edge(net, v, v)]
        isempty(loops) && continue
        shown = length(loops) > 10 ? join(loops[1:10], ", ") * ", …" : join(loops, ", ")
        throw(ArgumentError(
            "layer :$(m.layer_names[l]) (layer $l) contains $(length(loops)) " *
            "self-loop$(length(loops) == 1 ? "" : "s") (at vertex $shown). " *
            "ERGMMulti models the off-diagonal dyads of every layer only: the " *
            "term statistics would count a loop, but the pseudo-likelihood, " *
            "nobs, the MH proposal and every simulation never touch the diagonal, " *
            "so the observed statistics would be compared against a model that " *
            "cannot reproduce them (R ergm warns \"This network contains loops\" " *
            "here). Remove the loops (`rem_edge!(layer_network(m, $l), v, v)`) " *
            "or build the layer with `loops=false`."))
    end
    return m
end

"""
    has_dyad_dependent(model::MultiERGMModel) -> Bool

Whether any term of the multilayer formula is dyad-dependent, where a "dyad" is
a `(layer, i, j)` triple (see the dependence classification above). This is THE
predicate that decides whether the within-layer MPLE is an approximation: with
only dyad-independent terms the conditionals it multiplies are the model's own,
so the pseudo-likelihood is the likelihood. A method of ERGM.jl's shared
`has_dyad_dependent` generic (`ERGMMulti.has_dyad_dependent ===
ERGM.has_dyad_dependent`), used by both `show(::MultiERGMResult)` (the prose
caveat) and `is_exact(::MultiERGMResult)` (the machine-readable answer), so the
two cannot drift apart.

# Example
```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
has_dyad_dependent(MultiERGMModel([LayerEdges(1), LayerEdges(2)], m))      # false
has_dyad_dependent(MultiERGMModel([LayerEdges(), InterlayerDependence(1, 2)], m))  # true
```
"""
has_dyad_dependent(model::MultiERGMModel) =
    any(is_dyad_dependent(t) for t in model.terms)

"""
    MultiERGMResult{D}

Results from [`ergm_multi`](@ref) on a `MultilayerNetwork{D}`. Offset terms
report their fixed coefficient with `NaN` standard error (and `NaN`
rows/columns of `vcov`); `loglik` is the maximized pseudo-log-likelihood over
the within-layer dyads.

`se_type` records how `std_errors`/`vcov` were ACTUALLY obtained — `:hessian`
(the inverse negative pseudo-Hessian, which under-covers under dependence) or
`:bootstrap` (the parametric bootstrap of `ergm_multi(...; se=:bootstrap)`) or
`:mcmc` (a `method=:mcmle` fit; reported as `:fisher`). It
is what `NetworkCore.se_method(fit)` reports, and what `show` reads before deciding
whether the anticonservatism caveat still applies. Offset rows stay `NaN` under
either option: a fixed coefficient carries no uncertainty.

`boot_replicates` is the `n_boot × p_free` matrix of refitted free
coefficients under `se=:bootstrap` (a replicate whose refit did not converge
is a `NaN` row, excluded from the covariance), `nothing` otherwise.

`inference_withheld` is `true` for an MPLE fit (`method=:mple`, `se` not
given) of a dyad-dependent formula: the naive pseudo-likelihood standard errors
under-cover, so `coeftable`/`show` report `NaN` z and p-values and `confint`
refuses (see [`ergm_multi`](@ref), "Standard errors"). An explicit
`se=:hessian` or `se=:bootstrap` sets it to `false`.

`method` is `:mple` or `:mcmle`. For a `method=:mcmle` fit of a
dyad-dependent formula `loglik` is the log-likelihood itself (path sampling),
`se_type` is `:mcmc`, and `mcmc` holds the Monte-Carlo record: `convergence`
(an `ERGM.MCMLEConvergence`), `mc_std_errors` and the final `samples` of the
estimated statistics (`estimated`: their indices — the free ones, less any
fixed at `∓Inf` by a boundary statistic or held because not identifiable),
`n_samples`, `burnin`, `interval`, `loglik_mc_se`, the `start` (MPLE)
coefficients, the stopping rule (`termination`, `termination_p`,
`conv_confidence`, `conv_precision`) and `loglik_note` (why the
log-likelihood is `NaN`, or `nothing`); `mcmc` is `nothing` otherwise. The final sample is the one drawn at the last iterate, *before*
the final step to the returned coefficients (as in R), so the t-ratios and
Hotelling p-value in `convergence` describe that sample, not the returned
coefficients: the convergence verdict is the stopping rule, which `show`,
`approximations` and the non-convergence warning quote (under the default
`termination=:confidence`, the equivalence test's p-value and the step
length γ).

`converged` is `false` when the Newton iteration did not meet its tolerance
**or** when the pseudo-likelihood has no finite maximum by separation
(`separated == true`, R's "The MPLE does not exist!"): the coefficients are
then the point where Newton stopped on its flat asymptote and are unreliable.
`separated_terms` names the coefficients that carry the direction along which
the pseudo-likelihood keeps increasing (NetworkCore's separation verdict); it
is empty exactly when `separated` is `false`. A separated fit reports `NaN` z
values, p-values and confidence intervals.
A statistic at the boundary of its attainable range is handled differently —
its coefficient is fixed at `∓Inf` (see [`ergm_multi`](@ref)).

The full StatsAPI surface is implemented: `coef`, `stderror`, `vcov`,
`confint`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic` and `coeftable`
(a `NetworkCore.CoefficientTable` — the table `show` prints). The shared
result-metadata accessors (`is_exact`, `se_method`, `approximations`, ...)
read the same fields.

# Example
```julia
using ERGMMulti, ERGM, NetworkCore, Random
rng = Xoshiro(2)
m = MultilayerNetwork(12; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
for l in 1:2, i in 1:12, j in 1:12
    i != j && rand(rng) < 0.25 && add_layer_edge!(m, l, i, j)
end
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
fit isa MultiERGMResult{true}      # true
fit.converged, fit.separated       # (true, false)
fit.separated_terms                # String[]
se_method(fit)                     # :hessian
is_exact(fit)                      # true — every term is dyad-independent
fit.boot_replicates === nothing    # true — no bootstrap was run
fit.inference_withheld             # false — dyad-independent: the Wald table is exact
approximations(fit)                # String[] — nothing to disclose
```
"""
struct MultiERGMResult{D}
    model::MultiERGMModel{D}
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    vcov::Matrix{Float64}
    loglik::Float64
    aic::Float64
    bic::Float64
    converged::Bool
    se_type::Symbol
    boot_replicates::Union{Nothing, Matrix{Float64}}
    separated::Bool
    inference_withheld::Bool
    method::Symbol
    mcmc::Union{Nothing, NamedTuple}
    separated_terms::Vector{String}

    # The one constructor, positional over all fifteen fields: `ergm_multi` is
    # the only producer. (No "compatibility" constructors that default
    # `se_type`/`boot_replicates`/`separated`: a result whose flags do not
    # describe the numbers it carries must not be constructible by accident.)
    # The separation flag and the flagged terms are checked against each other.
    function MultiERGMResult(model::MultiERGMModel{D}, coefficients::Vector{Float64},
                             std_errors::Vector{Float64}, vcov::Matrix{Float64},
                             loglik::Float64, aic::Float64, bic::Float64,
                             converged::Bool, se_type::Symbol,
                             boot_replicates::Union{Nothing, Matrix{Float64}},
                             separated::Bool, inference_withheld::Bool,
                             method::Symbol, mcmc::Union{Nothing, NamedTuple},
                             separated_terms::Vector{String}) where {D}
        separated == !isempty(separated_terms) || throw(ArgumentError(
            "MultiERGMResult: separated = $separated disagrees with " *
            "separated_terms = $(repr(separated_terms))"))
        (separated && converged) && throw(ArgumentError(
            "MultiERGMResult: a separated fit cannot be converged"))
        new{D}(model, coefficients, std_errors, vcov, loglik, aic, bic, converged,
               se_type, boot_replicates, separated, inference_withheld, method,
               mcmc, separated_terms)
    end
end

Graphs.is_directed(::MultiERGMResult{D}) where {D} = D

# Coefficient labels as printed: the model's direction-aware statistic names,
# an offset row wrapped as R wraps it, `offset(L(A)~nodematch.g)` (its
# coefficient is fixed, not estimated)
_coef_names(r::MultiERGMResult) =
    [haskey(r.model.offsets, k) ? "offset($n)" : n
     for (k, n) in enumerate(r.model.terms.names)]

# The one non-convergence sentence, printed by `show` right under the verdict
# and listed by `approximations`: an unconverged fit is never a fit with a
# footnote.
_nonconvergence_caveat(r::MultiERGMResult) =
    r.separated ? _separation_caveat(r) :
    r.mcmc !== nothing ?
    "MCMLE did not converge in $(r.mcmc.convergence.iterations) iterations " *
    "($(_termination_verdict(r.mcmc))): the estimates are the last iterate, " *
    "NOT a maximum of the likelihood — raise `mcmle_maxiter` or `n_samples`" :
    "the Newton iteration on the within-layer pseudo-likelihood did not " *
    "converge: the maximum pseudo-likelihood estimate does not exist or was " *
    "not reached (a statistic at the boundary of its attainable range, perfect " *
    "separation, or too few iterations); point estimates and standard errors " *
    "are unreliable — simplify the formula, or raise `maxiter`"

# The separation verdict (R's `mple.existence`, "The MPLE does not exist!"):
# a combination of the model's statistics perfectly predicts the ties, so the
# pseudo-likelihood has no finite maximum and Newton stopped on its flat
# asymptote. The shared NetworkCore sentence, naming the flagged terms.
_separation_caveat(r::MultiERGMResult) =
    "the MPLE does not exist; " * separation_caveat(r.separated_terms) *
    " ($_R_MULTI_MPLE_NOT_EXIST) Remove, merge or coarsen the separating " *
    "term(s), or collect more ties"

# What the MCMLE's stopping rule said, in the words of `ERGM.mcmle`'s own
# verdict: under the default `termination=:confidence` the equivalence test's
# p-value and the step length γ — the rule itself, never the classical
# t-ratio / Hotelling diagnostics, which are computed on the sample drawn
# BEFORE the last step and are not the rule; under `:hotelling` its p-value,
# the largest t-ratio (part of that rule) and γ
function _termination_verdict(mc::NamedTuple)
    f3(x) = string(round(x, sigdigits=3))
    detail = mc.termination === :confidence ?
        "$(round(Int, 100 * mc.conv_confidence))% equivalence test p " *
        "$(f3(mc.termination_p)) (needs < $(f3(1 - mc.conv_confidence)); tolerance " *
        "precision $(f3(mc.conv_precision)), $(mc.n_samples) draws)" :
        "Hotelling p $(f3(mc.termination_p)) ($(mc.n_samples) draws), max t-ratio " *
        "$(f3(maximum(mc.convergence.t_ratios; init=NaN)))"
    return detail * ", step length γ $(f3(mc.convergence.step_length))"
end

# A coefficient fixed at ∓Inf by a boundary statistic (R's `drop`), or not
# identifiable (NaN, R's NA): said in `show` and in `approximations`, read
# off the coefficients themselves
function _fixed_coefficient_notes(r::MultiERGMResult)
    names = r.model.terms.names
    lo = [names[k] for k in eachindex(names) if r.coefficients[k] == -Inf]
    hi = [names[k] for k in eachindex(names) if r.coefficients[k] == Inf]
    nan = [names[k] for k in eachindex(names) if isnan(r.coefficients[k])]
    notes = String[]
    if !(isempty(lo) && isempty(hi))
        parts = String[]
        isempty(lo) || push!(parts, "$(join(lo, ", ")) fixed at -Inf (observed " *
                                    "statistic at its smallest attainable value)")
        isempty(hi) || push!(parts, "$(join(hi, ", ")) fixed at +Inf (observed " *
                                    "statistic at its largest attainable value)")
        push!(notes, join(parts, "; ") * (r.mcmc === nothing ?
            ": no finite maximum pseudo-likelihood estimate exists (R ergm's " *
            "drop=TRUE semantics; ergm.multi 0.3.0's layer operator does not " *
            "propagate the range and returns a finite ~-17 with a huge standard " *
            "error after warning \"The MPLE does not exist!\"); its standard error " *
            "and p-value are 0 by convention, and the remaining coefficients were " *
            "estimated on the dyads it does not touch" :
            ": no finite maximum-likelihood estimate exists (R ergm's drop=TRUE " *
            "semantics; ergm.multi 0.3.0 does not drop and reports a finite value " *
            "where its estimation stopped); its standard error and p-value are 0 " *
            "by convention, the sampler held the statistic at its observed bound, " *
            "and the remaining coefficients are the maximum-likelihood estimates " *
            "with it fixed there"))
    end
    isempty(nan) || push!(notes, "$(join(nan, ", ")) not identifiable (the " *
        "statistic does not vary on the dyads fitted, or is a linear combination " *
        "of the others): reported as NaN, as R reports NA, and the other " *
        "coefficients were estimated without it")
    if r.mcmc !== nothing && r.mcmc.loglik_note !== nothing
        push!(notes, r.mcmc.loglik_note)
    end
    return notes
end

# The sentence every bootstrap caller of the ERGM family uses, verbatim, to
# say what excluding the failed refits does to the standard errors (in the
# warning, `show` and `approximations`)
const _BOOT_EXCLUSION_BIAS =
    "The standard errors are conditional on a finite refit: the excluded " *
    "replicates are the extreme ones, so the standard errors are biased downward."

# Bootstrap replicates excluded from the covariance (NaN rows), if any
function _boot_exclusion_note(r::MultiERGMResult)
    r.boot_replicates === nothing && return nothing
    n_boot = size(r.boot_replicates, 1)
    n_bad = count(b -> !all(isfinite, view(r.boot_replicates, b, :)), 1:n_boot)
    n_bad == 0 && return nothing
    return "$n_bad of the $n_boot bootstrap refits had no finite MPLE (a " *
           "boundary statistic, separation or non-convergence on the simulated " *
           "network) and were excluded; the standard errors are the empirical " *
           "covariance of the remaining $(n_boot - n_bad) refits. " *
           "$_BOOT_EXCLUSION_BIAS (`fit.boot_replicates` holds every refit, the " *
           "excluded ones as NaN rows)"
end

# What the `Method:` line of `show` says the estimator is (the family's
# wording, as `ERGM.fit_ergm`'s results print it)
function _method_gloss(r::MultiERGMResult)
    if r.method === :mcmle
        return r.mcmc === nothing ?
            " (maximum likelihood, exact: the formula is dyad-independent, so the " *
            "MPLE is the MLE)" : " (Monte-Carlo maximum likelihood)"
    end
    return has_dyad_dependent(r.model) ?
        " (maximum pseudo-likelihood: an approximation under dyadic dependence; " *
        "the default method=:auto fits the MCMLE here, as R does)" :
        " (maximum pseudo-likelihood, which is the likelihood: the formula is " *
        "dyad-independent)"
end

function Base.show(io::IO, r::MultiERGMResult)
    println(io, "Multilayer ERGM Results")
    println(io, "=======================")
    mc = r.mcmc
    println(io, "Method: $(r.method)", _method_gloss(r))
    if mc !== nothing
        println(io, "Layers: $(n_layers(r.model.network)); " *
                    "log-likelihood: $(round(r.loglik, digits=4))" *
                    (isfinite(mc.loglik_mc_se) ?
                     " (path sampling; MC s.e. $(round(mc.loglik_mc_se, sigdigits=3)))" : ""))
    elseif r.method === :mcmle
        println(io, "Layers: $(n_layers(r.model.network)); " *
                    "log-likelihood: $(round(r.loglik, digits=4))")
    else
        println(io, "Layers: $(n_layers(r.model.network)); " *
                    "pseudo-log-likelihood: $(round(r.loglik, digits=4))")
    end
    println(io, "AIC: $(round(r.aic, digits=2)), BIC: $(round(r.bic, digits=2)); " *
                "converged: $(r.converged)")
    mc === nothing || println(io, "Termination: $(mc.termination) rule, p = " *
                                  "$(round(mc.termination_p, sigdigits=3)) " *
                                  "($(mc.convergence.iterations) iterations, " *
                                  "$(mc.n_samples) draws in the final sample)")
    if !r.converged
        println(io, "  Warning: ", _nonconvergence_caveat(r))
    end
    println(io, "Std. errors: ",
            r.se_type === :bootstrap ? "parametric bootstrap" :
            r.se_type === :mcmc ? "inverse Fisher information + Monte-Carlo error" :
            "inverse pseudo-Hessian")
    println(io)
    println(io, "Coefficients:")

    # Shared ecosystem presentation layer: the printed table IS
    # `coeftable(r)` (a NetworkCore.CoefficientTable rendered through
    # `print_coeftable`: Estimate / Std.Error / z value / Pr(>|z|) with
    # significance codes), so what is shown and what is inspected agree.
    # Offset terms are tagged and print NaN standard errors / p-values.
    show(io, coeftable(r))

    # Honest-uncertainty caveat (mirroring ERGM.jl's show): pseudo-likelihood
    # fits of dyad-dependent formulas have suspect inverse-Hessian standard
    # errors. Dyad-independent formulas need no caveat — there the
    # pseudo-likelihood is the likelihood. Neither does a bootstrap fit: a
    # parametric-bootstrap covariance does not treat the dyads as independent,
    # so calling it anticonservative would be a lie. (The POINT ESTIMATE is
    # still an MPLE either way, which is what the remaining note says.)
    if has_dyad_dependent(r.model) && r.mcmc === nothing
        println(io)
        if r.se_type === :bootstrap
            println(io, "Note: this model contains dyad-dependent terms and was fit by")
            println(io, "maximum pseudolikelihood (MPLE), so the point estimates are biased in")
            println(io, "finite samples. The standard errors are a parametric bootstrap and do")
            println(io, "not assume the dyad conditionals are independent.")
        elseif r.inference_withheld
            println(io, "Note: z values and p-values are not reported (NaN). This model contains")
            println(io, "dyad-dependent terms and was fit by maximum pseudolikelihood (MPLE); the")
            println(io, "standard errors shown are the naive pseudolikelihood ones, which treat")
            println(io, "dependent (layer, i, j) dyads as independent and under-cover (95% Wald")
            println(io, "intervals covered 0.76-0.94 in simulation), so no test or interval is")
            println(io, "built on them. The point estimates are biased in finite samples. For")
            println(io, "inference refit with method=:mcmle (maximum likelihood, R's default)")
            println(io, "or se=:bootstrap; se=:hessian requests the naive Wald table explicitly.")
        else
            println(io, "Warning: this model contains dyad-dependent terms and was fit by")
            println(io, "maximum pseudolikelihood (MPLE). The standard errors are based on")
            println(io, "the naive pseudolikelihood and are suspect (typically")
            println(io, "anticonservative); treat the inference with caution, or refit with")
            println(io, "`se=:bootstrap` for a parametric-bootstrap covariance.")
        end
    end
    for note in _fixed_coefficient_notes(r)
        println(io)
        println(io, "Note: ", note)
    end
    excluded = _boot_exclusion_note(r)
    if excluded !== nothing
        println(io)
        println(io, "Note: ", excluded)
    end
end

# ============================================================================
# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`)
# ============================================================================
#
# `fit_metadata(fit)` collects these accessors. They read the SAME
# `has_dyad_dependent` predicate as the prose caveat in `show`, so the printed
# warning and the machine-readable answer cannot disagree.

estimand(::MultiERGMResult) = :multilayer_ergm

objective(r::MultiERGMResult) =
    r.method === :mcmle ? :likelihood : :pseudolikelihood

"""
    is_exact(r::MultiERGMResult) -> Bool

`true` iff every term is dyad-independent over the `(layer, i, j)` dyad universe
— there the within-layer pseudo-likelihood *is* the likelihood and the MPLE is
the exact MLE — and the Newton iteration converged. Any cross-layer or
within-layer structural term (`LayerMutual`, `LayerTriangle`,
`InterlayerDependence`, `MultiplexMutual`, ...) makes the same estimator an
approximation; an unconverged fit, or one with a coefficient fixed at `∓Inf`
by a boundary statistic (R's `drop`), is not the exact MLE of anything; all
report `false`.
"""
is_exact(r::MultiERGMResult) =
    r.converged && all(isfinite, r.coefficients) && !has_dyad_dependent(r.model)

"""
    se_method(r::MultiERGMResult) -> Symbol

What the reported standard errors ACTUALLY are: `:hessian` (the inverse negative
pseudo-Hessian), `:bootstrap` (the parametric bootstrap of
`ergm_multi(...; se=:bootstrap)`) or `:fisher` (a `method=:mcmle` fit: the
inverse Fisher information of the final sample plus the Monte-Carlo error).
Read straight off the fit, so it can never claim an estimator that was not used.
"""
se_method(r::MultiERGMResult) = r.se_type === :mcmc ? :fisher : r.se_type

# `ergm_multi` calls `require_observed` on every layer with the default `:error`
# policy: the MPLE enumerates every within-layer dyad as observed, so a masked
# dyad would enter the pseudo-likelihood at its face value and is refused.
missing_method(::MultiERGMResult) = :rejected

function approximations(r::MultiERGMResult)
    out = String[]
    r.converged || push!(out, _nonconvergence_caveat(r))
    if r.mcmc !== nothing
        mc = r.mcmc
        push!(out, "MCMLE: the likelihood is approximated by an MCMC sample of " *
                   "$(mc.n_samples) multilayer networks, so the estimates carry " *
                   "Monte-Carlo error (largest MC standard error " *
                   "$(round(maximum(mc.mc_std_errors; init=0.0), sigdigits=3)), " *
                   "included in the reported standard errors); stopping rule " *
                   ":$(mc.termination), p = $(round(mc.termination_p, sigdigits=3))")
        isfinite(r.loglik) && push!(out,
            "log-likelihood estimated by path sampling from the dyad-independent " *
            "part of the model (MC standard error " *
            "$(round(mc.loglik_mc_se, sigdigits=3)))")
    elseif has_dyad_dependent(r.model)
        # The POINT ESTIMATE is a pseudo-likelihood estimate however the standard
        # errors were computed: the bootstrap replaces the covariance, not θ̂.
        push!(out, "maximum pseudo-likelihood of a dyad-dependent multilayer " *
                   "model: the (layer, i, j) conditionals are multiplied as if " *
                   "independent, so the point estimates are biased in finite samples")
        if r.se_type === :hessian
            push!(out, "inverse-Hessian standard errors of the naive pseudo-likelihood: " *
                       "expected anticonservative under dependence (refit with " *
                       "`se=:bootstrap` for a parametric-bootstrap covariance)")
        end
        r.inference_withheld &&
            push!(out, "z values, p-values and confidence intervals withheld: " *
                       "the naive pseudo-likelihood standard errors are not " *
                       "calibrated under dyadic dependence (refit with " *
                       "method=:mcmle or se=:bootstrap; se=:hessian opts in)")
    end
    if r.se_type === :bootstrap
        push!(out, "standard errors are a parametric bootstrap of the multilayer MPLE " *
                   "(simulate within-layer networks at θ̂, refit, empirical " *
                   "covariance): they do not assume the (layer, i, j) conditionals " *
                   "are independent, but they are Monte-Carlo estimates and assume " *
                   "the fitted model generated the data")
    end
    append!(out, _fixed_coefficient_notes(r))
    excluded = _boot_exclusion_note(r)
    excluded === nothing || push!(out, excluded)
    isempty(r.model.offsets) ||
        push!(out, "$(length(r.model.offsets)) offset term(s) held fixed, not " *
                   "estimated: their coefficients carry no uncertainty (NaN " *
                   "standard errors) and the reported dof excludes them")
    return out
end

# All within-layer dyads of the multilayer network as (l, i, j)
function _layer_dyads(m::MultilayerNetwork{D}) where {D}
    dyads = NTuple{3, Int}[]
    for l in 1:n_layers(m), i in 1:m.n
        for j in (D ? (1:m.n) : (i+1:m.n))
            i == j && continue
            push!(dyads, (l, i, j))
        end
    end
    return dyads
end

# Number of within-layer dyads — the model's dyad universe
function _n_within_dyads(m::MultilayerNetwork{D}) where {D}
    per = D ? m.n * (m.n - 1) : m.n * (m.n - 1) ÷ 2
    return n_layers(m) * per
end

# Multilayer MPLE enumerates every within-layer dyad as observed, so a masked
# (unobserved) dyad in any layer would enter the pseudo-likelihood at its
# face value. Reject rather than invent data.
function _require_observed_layers(m::MultilayerNetwork, context::AbstractString)
    for (l, layer) in enumerate(m.layers)
        require_observed(layer; context="$context (layer $l)", face_ok=false)
    end
    return nothing
end

"""
    ergm_multi(m::MultilayerNetwork, terms; offsets=Dict{Int,Float64}(),
               method=:auto, maxiter=100, tol=1e-8, se=nothing, n_boot=100,
               boot_burnin=nothing, boot_interval=nothing,
               rng=Random.default_rng(), threaded=true,
               n_samples=1024, burnin=nothing, interval=nothing,
               mcmle_maxiter=60, termination=:confidence, conv_precision=0.1,
               conv_confidence=0.99, max_n_samples=nothing, bridge_rungs=16,
               bridge_samples=nothing, init=nothing, drop=true) -> MultiERGMResult
    ergm_multi(model::MultiERGMModel; kwargs...) -> MultiERGMResult
    fit_ergm_multi(...)                            # the same function

Fit a multilayer ERGM — by default (`method=:auto`) as R's `ergm.multi`
does: by maximum pseudo-likelihood when every term is dyad-independent
(where it is the exact MLE) and by Monte-Carlo maximum likelihood otherwise
(see "Estimators" below) — over the
**within-layer dyads** (the dyad universe of `ergm.multi`'s
block-diagonal construction): each dyad's conditional edge probability is
logistic in `θ'Δg`, with `Δg` the add-direction change statistics of the
layer-aware terms. `ergm_multi` is the R (`ergm.multi`) name and
[`fit_ergm_multi`](@ref) the ecosystem's harmonised `fit_<model>` name; they
are one `const` function (`fit_ergm_multi === ergm_multi`).

The terms are validated against the data when the [`MultiERGMModel`](@ref)
is built: an out-of-range layer index, a within-layer term whose attribute
the layer lacks, an undirected-only term on a directed layer (or the
reverse), and a masked (missing) dyad in any layer all throw an
`ArgumentError` before any fitting. An unconverged Newton iteration is
loud: a warning is emitted, `fit.converged` is `false`, `show` prints the
caveat and `approximations(fit)` records it. A statistic at the boundary of
its attainable range (no co-occurring tie under `InterlayerDependence`, no
mutual dyad under `LayerMutual`, no triangle under `LayerTriangle` — even
when no single tie could close one —, a perfectly separated `NodeMatch`,
...) has no finite estimate: as R ergm's default `drop=TRUE`, its
coefficient is fixed at `∓Inf` with standard error 0 (R's warning is
printed), the other coefficients are estimated with it held there (the
MPLE on the dyads it does not touch; the MCMLE with the sampler never
moving it off its observed bound), `is_exact(fit)` is `false`, and
`show`/`approximations` say so. ergm.multi 0.3.0 does not apply the drop to
layer terms (it reports `NA`, or a finite value where its estimation
stopped); its offset spelling, `offset(L(~term, ~A))` with
`offset.coef = -Inf`, fits the same model as the drop here. `drop=false`
(R's `control.ergm(drop=FALSE)`) refuses such a statistic with an
`ArgumentError` before any fit. A statistic with no identifiable
coefficient — its change statistics are all 0 on the dyads fitted, or a
linear combination of the others' — is reported as `NaN` (R's `NA`) with a
warning, and the others are fitted without it. Separation by a
*combination* of statistics (no single column at its boundary, yet the ties
are perfectly predicted — R's "The MPLE does not exist!") is detected by
NetworkCore's separation verdict on the design actually fitted, with or
without offsets: a warning names the separating terms, and the fit is
returned with `converged == false`, `separated == true`, the terms in
`fit.separated_terms`, and `NaN` z values, p-values and confidence
intervals; `show`/`approximations` carry the verdict and `se=:bootstrap` is
refused. (A `method=:mcmle` fit refuses a separated MPLE start.)

`offsets` maps term indices to **fixed** coefficients (per-layer offsets,
e.g. `Dict(1 => -log(n))` for a size adjustment); the remaining
coefficients are estimated with the offset contribution absorbed into the
linear predictor.

# Estimators

`method=:auto` (the default) follows R: `:mple` when no term is
dyad-dependent and `:mcmle` otherwise (`ERGM.resolve_method`, the rule of
every ERGM-family fitter).

`method=:mple` maximizes the product of the (layer, i, j) conditionals. For a
dyad-independent formula that product is the likelihood and the fit is the
exact MLE. For a dyad-dependent formula it is **not** what R reports, and
must be asked for explicitly.

`method=:mcmle` is R's estimator. Starting from the MPLE, each iteration
draws `n_samples` multilayer networks at the current coefficients with the
sampler of [`simulate_multi_ergm`](@ref) (one retained state every `interval`
proposals after `burnin`; the chain continues across iterations, offsets
included) and takes the Monte-Carlo Newton step
`θ ← θ + γ·Σ̂⁻¹(g(y_obs) − ḡ)` on the free coefficients, with Hummel's step
length `γ`. The iteration is ERGM.jl's own (`ERGM.Extension.mcmle_solve`, the loop
`ERGM.mcmle` runs), so the stopping rule is R ergm 4's: at `γ = 1`, with
confidence `conv_confidence` (0.99), the estimating equations at the updated
coefficients must lie within the tolerance region set by `conv_precision`
(0.1); when the test fails near the solution the sample is boosted, up to
`max_n_samples` (default `16·n_samples`). `termination=:hotelling` selects
the older t-ratio (`conv_threshold`) and Hotelling T² (`hotelling_alpha`)
rule at a fixed sample size. The reported coefficients are the update from
the sample that passed; the standard errors are that sample's inverse
Fisher information plus the Monte-Carlo error (`ERGM.Extension.mcmle_covariance`, as
`ERGM.mcmle`), with full z, p and `confint`. `loglikelihood(fit)` is the
log-likelihood itself, by path sampling from the dyad-independent part of the
model (whose normalizer is exact) over `bridge_rungs` Simpson segments of
`bridge_samples` draws (default `n_samples`; `bridge_rungs=0`
skips it and reports `NaN`); `aic`/`bic` are built on it. `fit.method` is
`:mcmle`, `fit.mcmc` holds the record (`convergence`, `mc_std_errors`,
`samples`, `n_samples`, `burnin`, `interval`, `loglik_mc_se`, `start`,
`termination`, `termination_p`, `conv_confidence`, `conv_precision`,
`estimated`, `loglik_note`; the samples and `convergence` diagnostics are
those of the pre-step sample),
`se_method(fit) == :fisher` and `objective(fit) == :likelihood`. The result is
reproducible from `rng` and independent of the thread count.

`method=:mcmle` on a dyad-independent formula returns the exact MPLE with no
Monte Carlo (`fit.mcmc === nothing`). A statistic at the boundary of its
attainable range is fixed at `∓Inf` (above): the step, the stopping rule and
the covariance run over the other coefficients (`fit.mcmc.estimated` lists
them), and if the fixed statistic is dyad-dependent the log-likelihood (and
AIC, BIC) is `NaN`, because the path sampler's dyad-independent reference
model cannot hold it at its bound (`show`/`approximations` say so). A
dyad-independent statistic with no identifiable coefficient is held at 0
and reported `NaN`; a dyad-dependent one needs `init=`. The MCMLE is refused
with an `ArgumentError` when the MPLE start is separated or did not
converge (unless `init=` is given), and when statistics would be fixed at
both `-Inf` and `+Inf` (the sampler cannot hold both).

Each estimator takes its own keywords: `se`, `n_boot`, `boot_burnin`,
`boot_interval` and `threaded` are the MPLE's; `n_samples`, `burnin`,
`interval`, `mcmle_maxiter`, `termination`, `conv_precision`,
`conv_confidence`, `conv_threshold`, `hotelling_alpha`, `max_n_samples`,
`bridge_rungs`, `bridge_samples` and `init` the MCMLE's; `maxiter`, `tol`,
`rng` and `drop` both. A keyword the chosen estimator does not take is an
`ArgumentError` that names the estimator that does — for example
`se=:bootstrap` on a dyad-dependent formula, which `method=:auto` sends to the
MCMLE: pass `method=:mple` explicitly.

# Standard errors of the MPLE

- default (`se=nothing`) — the inverse negative pseudo-Hessian. With only
  dyad-independent terms the pseudo-likelihood is the likelihood, these
  standard errors are correct and the full Wald table is reported. With any
  dyad-dependent term (`LayerMutual`, `LayerTriangle`,
  `InterlayerDependence`, `MultiplexMutual`, a `WithinLayer` of a
  dyad-dependent term, ...) the pseudo-likelihood multiplies the
  (layer, i, j) conditionals as if independent and the naive standard errors
  under-cover (95% Wald intervals covered 0.76–0.94 in simulation), so the
  fit reports the estimates and the naive standard errors but **withholds the
  inference built on them**: z and p are `NaN`, `show` says why,
  [`confint`](@ref) throws an `ArgumentError`, `approximations(fit)` records
  it (`fit.inference_withheld == true`). This is ERGM.jl's `mple` default;
  the MCMLE (the `method=:auto` default for such a formula) or
  `se=:bootstrap` give calibrated inference.
- `se=:hessian` — the same standard errors, with the naive Wald table (z, p,
  `confint`) reported: the written opt-in to what R prints for an MPLE fit.
  `show` keeps the anticonservatism warning.
- `se=:bootstrap` — parametric bootstrap: simulate `n_boot` multilayer networks
  from the fitted model at θ̂ (offsets included) with
  [`simulate_multi_ergm`](@ref), refit `ergm_multi` on each with the same
  offsets, and report the empirical covariance of the refits. The point
  estimates are unchanged; only the covariance is replaced. Offset rows stay
  `NaN` — a fixed coefficient carries no uncertainty. A replicate whose refit
  does not converge is excluded (warned once; `fit.boot_replicates` keeps it
  as a `NaN` row). The standard errors are conditional on a finite refit:
  the excluded replicates are the extreme ones, so the standard errors are
  biased downward (the warning, `show` and `approximations` say so). This is
  the same option, with the same keywords and the same semantics, as
  `ERGM.mple`'s, and it runs on the ONE shared `NetworkCore.bootstrap_cov`
  loop. It is refused (`ArgumentError`) when a coefficient is fixed at
  `∓Inf` by a boundary statistic (a network cannot be simulated at an
  infinite coefficient) and on a separated fit.

# Keyword Arguments
- `offsets::Dict{Int,Float64}`: fixed coefficients by term index
- `method::Symbol=:auto`: `:auto`, `:mple` or `:mcmle` (above; anything else
  is an `ArgumentError`)
- `n_samples`, `burnin`, `interval`, `mcmle_maxiter`, `termination`,
  `conv_precision`, `conv_confidence`, `conv_threshold`,
  `hotelling_alpha`, `max_n_samples`, `bridge_rungs`,
  `bridge_samples`, `init`: the MCMLE's controls (above); `burnin`/`interval`
  default to the dyad-scaled rule of every ERGM-family sampler
- `maxiter::Int=100`, `tol::Float64=1e-8`: Newton iteration controls (of the
  MPLE, and of the MCMLE's MPLE start)
- `drop::Bool=true`: fix a statistic at the boundary of its attainable range
  at `∓Inf` (R ergm's default); `false` refuses such a model instead
- `se=nothing`: `nothing`, `:hessian` or `:bootstrap` (above; anything else
  is refused by the shared `NetworkCore.check_se`)
- `n_boot::Int=100`: number of bootstrap replicates (`se=:bootstrap` only)
- `boot_burnin`, `boot_interval`: MCMC controls for the bootstrap simulations;
  `nothing` (default) resolves to the dyad-scaled rule shared with every
  ERGM-family sampler (`ERGM.Extension.mcmc_defaults`: `20 × n_dyads` and
  `max(100, n_dyads ÷ 10)` over the within-layer dyads)
- `rng::AbstractRNG=Random.default_rng()`: source of the bootstrap randomness —
  a fixed `rng` reproduces the standard errors exactly
- `threaded::Bool=true`: run the bootstrap refits on all threads. The
  replicates are simulated from `rng` before any refit and each refit is
  deterministic, so the standard errors are identical whatever the thread
  count (pinned by a test)

# Example
```julia
using ERGMMulti, ERGM, NetworkCore, Random
rng = Xoshiro(1)
m = MultilayerNetwork(12; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
for i in 1:12, j in 1:12
    i == j && continue
    rand(rng) < 0.2 && add_layer_edge!(m, :friendship, i, j)
    rand(rng) < 0.2 && add_layer_edge!(m, :advice, i, j)
end
terms = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)]
mle = fit_ergm_multi(m, terms; n_samples=400, rng=Xoshiro(3))
mle.method, se_method(mle)       # (:mcmle, :fisher) — dyad-dependent: R's estimator
coeftable(mle).names             # ["L(friendship)~edges", "L(advice)~edges", "L(friendship&advice)~edges"]
size(confint(mle))               # (3, 2)
fit = fit_ergm_multi(m, terms; method=:mple)
fit.converged                    # true
fit.inference_withheld           # true — the MPLE of a dependent formula: no z, p or interval
boot = fit_ergm_multi(m, terms; method=:mple, se=:bootstrap, n_boot=50, rng=Xoshiro(2))
size(confint(boot))              # (3, 2) — parametric-bootstrap intervals
try
    fit_ergm_multi(m, terms; se=:bootstrap)     # :auto chose the MCMLE
catch e
    occursin("method=:mple", e.msg)             # true
end
```
"""
function ergm_multi(m::MultilayerNetwork, terms;
                    offsets::AbstractDict=Dict{Int, Float64}(),
                    kwargs...)
    n_layers(m) >= 1 || throw(ArgumentError("network has no layers"))
    _require_observed_layers(m, "ergm_multi")
    model = MultiERGMModel(terms, m; offsets=offsets)
    return ergm_multi(model; kwargs...)
end

function ergm_multi(model::MultiERGMModel; method::Symbol=:auto, kwargs...)
    est = ERGM.resolve_method(method, model; context="ergm_multi")
    _check_multi_keywords(est, method, keys(kwargs))
    if est === :mcmle && has_dyad_dependent(model)
        _require_observed_layers(model.network, "ergm_multi")
        return _multi_mcmle(model; kwargs...)
    end
    # `method=:mcmle` on a dyad-independent model: its MPLE is the exact MLE,
    # with exact standard errors and log-likelihood (the MCMLE's own controls
    # have nothing to do)
    est === :mcmle && return _multi_mple_estimate(model; method=:mcmle,
        (k => v for (k, v) in kwargs if k in _SHARED_KEYWORDS)...)
    return _multi_mple_estimate(model; method=:mple, kwargs...)
end

# The keywords each estimator takes, read from its own signature so the
# lists cannot drift from the code; `warn` is internal to both.
_multi_keywords(::Val{:mple}) =
    filter(!=(:method), Base.kwarg_decl(which(_multi_mple_estimate, Tuple{MultiERGMModel})))
_multi_keywords(::Val{:mcmle}) =
    filter(!=(:warn), Base.kwarg_decl(which(_multi_mcmle, Tuple{MultiERGMModel})))
const _SHARED_KEYWORDS = (:maxiter, :tol, :rng, :drop)

# A keyword the chosen estimator does not take is refused in words — most
# often an MPLE keyword (`se=:bootstrap`) on a dyad-dependent formula, which
# `method=:auto` sends to the MCMLE (the family's rule, as `ERGM.fit_ergm`).
function _check_multi_keywords(est::Symbol, method::Symbol, keys)
    accepted = _multi_keywords(Val(est))
    bad = [k for k in keys if !(k in accepted)]
    isempty(bad) && return nothing
    other = est === :mple ? :mcmle : :mple
    other_accepted = _multi_keywords(Val(other))
    listed = join(("`$k`" for k in bad), ", ")
    why = method === :auto ?
        " method=:auto chose $(repr(est)) because the formula is " *
        (est === :mcmle ? "dyad-dependent (R's ergm.multi fits the Monte-Carlo " *
                          "MLE there)." :
                          "dyad-independent (the MPLE is the exact MLE there).") : ""
    hint = all(in(other_accepted), bad) ?
        " $(length(bad) == 1 ? "It is a keyword" : "They are keywords") of " *
        "method=$(repr(other)); pass method=$(repr(other)) explicitly to use " *
        "$(length(bad) == 1 ? "it" : "them")." :
        " See `?ergm_multi` for the keywords of each estimator."
    throw(ArgumentError("ergm_multi: keyword $listed is not accepted by " *
                        "method=$(repr(est)).$why$hint"))
end

# The MPLE (and the exact MLE of a dyad-independent formula, which is the
# same computation), with its three standard-error options
function _multi_mple_estimate(model::MultiERGMModel; method::Symbol=:mple,
                              maxiter::Int=100, tol::Float64=1e-8,
                              se::Union{Nothing, Symbol}=nothing,
                              n_boot::Int=100,
                              boot_burnin::Union{Nothing, Int}=nothing,
                              boot_interval::Union{Nothing, Int}=nothing,
                              rng::Random.AbstractRNG=Random.default_rng(),
                              threaded::Bool=true, drop::Bool=true)
    # `se=nothing` (the default) is `:hessian` whose inference is withheld
    # under dyadic dependence; an explicit `se=:hessian` opts in to R's naive
    # Wald table (ERGM.jl's `mple` pattern)
    naive_opt_in = se === :hessian
    se = something(se, :hessian)
    check_se(se, (:hessian, :bootstrap); context="ergm_multi")
    m = model.network
    _require_observed_layers(m, "ergm_multi")
    terms = model.terms
    offsets = model.offsets
    p = length(terms)
    free = [k for k in 1:p if !haskey(offsets, k)]

    fit = _multi_mple_fit(m, terms, offsets, free; maxiter=maxiter, tol=tol, drop=drop)
    β = fit.θ
    ll = fit.loglik
    converged = fit.converged
    separated = fit.separated
    # A separated design has already printed the separation warning
    # (`_multi_mple_fit`)
    (converged || separated) ||
        @warn "ergm_multi: " * _nonconvergence_caveat_short(maxiter)
    vcov_free = fit.vcov
    se_free = fit.se
    pf = length(free)

    coefficients = zeros(p)
    for (kf, k) in enumerate(free)
        coefficients[k] = β[kf]
    end
    for (k, c) in offsets
        coefficients[k] = c
    end

    # `se=:bootstrap` replaces the covariance of the FREE coefficients only:
    # the offsets are fixed, so they carry no uncertainty under either option.
    replicates = nothing
    if se === :bootstrap
        # A coefficient fixed at ∓Inf cannot be simulated from (the sampler's
        # θ'δ would be NaN wherever the dropped statistic changes), so there
        # is nothing to bootstrap: say so instead of returning NaN errors —
        # the same refusal as `ERGM.mple`'s. Nor can a NaN one (a statistic
        # with no identifiable coefficient): no model has been fitted for it.
        if any(isinf, coefficients)
            fixed = [nm for (nm, c) in zip(terms.names, coefficients) if isinf(c)]
            throw(ArgumentError(
                "ergm_multi: se=:bootstrap is not available when a coefficient is " *
                "fixed at ±Inf by a statistic at the boundary of its attainable " *
                "range ($(join(fixed, ", "))): a multilayer network cannot be " *
                "simulated at an infinite coefficient. Remove the term (as R's " *
                "drop=TRUE does) or use se=:hessian, which reports " *
                "standard error 0 for the fixed coefficient and the inverse-Hessian " *
                "errors of the rest."))
        end
        if any(isnan, coefficients)
            bad = [nm for (nm, c) in zip(terms.names, coefficients) if isnan(c)]
            throw(ArgumentError(
                "ergm_multi: se=:bootstrap is not available when a coefficient is " *
                "not identifiable ($(join(bad, ", ")) = NaN: the statistic does not " *
                "vary on the dyads fitted, or is a linear combination of the " *
                "others): there is no fitted value to simulate it at. Remove the " *
                "term and refit."))
        end
        # A separated MPLE has no finite maximum: the bootstrap would simulate
        # at the point where Newton stopped on the asymptote
        separated && throw(ArgumentError(
            "ergm_multi: se=:bootstrap is not available on a separated fit: the " *
            "MPLE does not exist (the pseudo-likelihood keeps increasing as the " *
            "coefficient(s) on $(join(fit.separated_terms, ", ")) run to ±Inf), " *
            "so there is no fitted model to simulate from. Remove, merge or " *
            "coarsen the separating term(s)."))
        vcov_free, se_free, replicates =
            _multi_bootstrap_cov(model, coefficients, β, free;
                                 n_boot=n_boot, boot_burnin=boot_burnin,
                                 boot_interval=boot_interval,
                                 maxiter=maxiter, tol=tol, rng=rng,
                                 threaded=threaded)
    end

    std_errors = fill(NaN, p)
    vcov_full = fill(NaN, p, p)
    vcov_full[free, free] = vcov_free
    for (kf, k) in enumerate(free)
        std_errors[k] = se_free[kf]
    end

    # Model size for AIC/BIC: the finite free coefficients, on the dyads they
    # were estimated on (`n_kept` — every within-layer dyad unless a boundary
    # statistic was dropped; R's `logLik` nobs attribute)
    k_est = count(isfinite, β)
    aic = -2 * ll + 2 * k_est
    bic = -2 * ll + k_est * log(fit.n_kept)

    withheld = se === :hessian && !naive_opt_in && has_dyad_dependent(model)
    return MultiERGMResult(model, coefficients, std_errors, vcov_full, ll,
                           aic, bic, converged, se, replicates, separated, withheld,
                           method, nothing, fit.separated_terms)
end

_nonconvergence_caveat_short(maxiter::Int) =
    "the Newton iteration did not converge within maxiter = $maxiter " *
    "iterations (the maximum pseudo-likelihood estimate may not exist: a " *
    "statistic at the boundary of its attainable range, or " *
    "separation); the result reports converged = false and its point " *
    "estimates and standard errors are unreliable"

"""
    fit_ergm_multi

The harmonised `fit_<model>` name of the multilayer ERGM fitter — a `const`
alias of [`ergm_multi`](@ref) (`fit_ergm_multi === ergm_multi`), which is the
R `ergm.multi` name. Both are exported; use whichever reads better.

# Example
```julia
using ERGMMulti
fit_ergm_multi === ergm_multi   # true
```
"""
const fit_ergm_multi = ergm_multi

# The estimator offers no `missing=` keyword at all: the MPLE enumerates every
# within-layer dyad as an observed row, so the only policy is to refuse a
# masked network (the capability generator prints this tuple).
missing_policies(::typeof(ergm_multi)) = (:error,)

# Dyad-scaled sampler defaults over the within-layer dyad universe — the ONE
# rule of the ERGM family (`ERGM.Extension.mcmc_defaults`), so the
# budgets cannot drift apart between packages.
function _resolve_multi_mcmc(m::MultilayerNetwork, burnin, interval)
    if burnin === nothing || interval === nothing
        d = ERGM.Extension.mcmc_defaults(_n_within_dyads(m))
        burnin = something(burnin, d.burnin)
        interval = something(interval, d.interval)
    end
    return Int(burnin), Int(interval)
end

# Parametric-bootstrap covariance of the multilayer MPLE: simulate `n_boot`
# multilayer networks at the fitted coefficients (offsets included — they are
# part of the data-generating model), refit `ergm_multi` on each with the SAME
# offsets, and take the empirical covariance of the free coefficients. The loop
# is the shared `NetworkCore.bootstrap_cov`; this supplies only the two callbacks
# that are ERGMMulti's. A replicate whose refit does not converge (a simulated
# network with a statistic at the boundary of its attainable range) has no
# finite MPLE: its row is NaN, excluded from the covariance and counted — the
# same exclusion pattern as `ERGM.mple`'s bootstrap.
function _multi_bootstrap_cov(model::MultiERGMModel, coefficients::Vector{Float64},
                              β_free::Vector{Float64}, free::Vector{Int};
                              n_boot::Int, boot_burnin, boot_interval,
                              maxiter::Int, tol::Float64,
                              rng::Random.AbstractRNG, threaded::Bool=true)
    terms = model.terms
    offsets = model.offsets
    burnin, interval = _resolve_multi_mcmc(model.network, boot_burnin, boot_interval)

    simulate(rng, B) = simulate_multi_ergm(model, coefficients;
                                           n_sim=B, burnin=burnin,
                                           interval=interval, rng=rng)

    # A replicate whose refit is unconverged, separated, or dropped a
    # boundary column (a ∓Inf coefficient) has no finite MPLE: a NaN row.
    function refit(sim::MultilayerNetwork)
        r = _multi_mple_fit(sim, terms, offsets, free; maxiter=maxiter, tol=tol,
                            warn=false)
        return (r.converged && all(isfinite, r.θ)) ? r.θ : fill(NaN, length(free))
    end

    # The replicates are all simulated from `rng` first and every refit is
    # deterministic, so `threaded` changes the wall time, never the numbers.
    boot = bootstrap_cov(refit, simulate, β_free; n_boot=n_boot, rng=rng,
                         threaded=threaded)
    replicates = boot.replicates
    ok = [all(isfinite, view(replicates, b, :)) for b in 1:n_boot]
    n_ok = count(ok)
    n_ok == n_boot && return boot.vcov, boot.se, replicates

    n_ok >= 2 || throw(ArgumentError(
        "ergm_multi: se=:bootstrap — only $n_ok of the $n_boot bootstrap refits " *
        "converged (the others simulated a multilayer network on which the " *
        "pseudo-likelihood has no finite maximum); a covariance needs at least " *
        "2. The model is near-degenerate at its MPLE: increase n_boot or " *
        "simplify the formula."))
    @warn "ergm_multi: se=:bootstrap — $(n_boot - n_ok) of the $n_boot bootstrap " *
          "refits did not converge (the simulated multilayer network put a " *
          "statistic at the boundary of its attainable range, or perfectly " *
          "separated the ties) and were excluded; the standard errors are the " *
          "empirical covariance of the $n_ok converged refits. " *
          "$_BOOT_EXCLUSION_BIAS This is about the " *
          "simulated replicates, not about the observed network. " *
          "`fit.boot_replicates` holds every refit (NaN rows excluded); " *
          "`approximations(fit)` records the exclusion."
    V = Matrix{Float64}(cov(replicates[ok, :]))
    return V, sqrt.(max.(diag(V), 0.0)), replicates
end

# The MPLE design over the within-layer dyads: the FREE columns of the change
# statistics, the observed tie indicators, and the offset contribution to the
# linear predictor (`ergm.multi`'s offset mechanism — the fixed coefficients are
# dropped from the parameter vector but NOT from the linear predictor).
function _multi_mple_design(m::MultilayerNetwork, terms::TermSet,
                            offsets::Dict{Int, Float64}, free::Vector{Int})
    p = length(terms)
    dyads = _layer_dyads(m)
    n_dyads = length(dyads)

    X = Matrix{Float64}(undef, n_dyads, p)
    y = Vector{Bool}(undef, n_dyads)
    row = Vector{Float64}(undef, p)
    for (r, (l, i, j)) in enumerate(dyads)
        _change_stat_layer_all!(row, terms, m, l, i, j)
        @inbounds for k in 1:p
            X[r, k] = row[k]
        end
        y[r] = has_edge(m.layers[l], i, j)
    end
    η0 = [sum(X[r, k] * offsets[k] for k in keys(offsets); init=0.0)
          for r in 1:n_dyads]
    return X[:, free], y, η0
end

# The attainable range of a multilayer statistic — R ergm's `minval`/`maxval`
# — on networks the size of `m`: what `ergm.checkextreme.model` compares the
# observed statistic with (an observed value at an end of its range fixes the
# coefficient at ∓Inf). Declared as methods of ERGM.jl's extension-API
# generic `ERGM.Extension.attainable_range` for this package's term types on
# a `MultilayerNetwork`, so `ERGM.Extension.extreme_statistics(terms, m)`
# reads them exactly as it reads ERGM.jl's own; anything without a method
# declares no range (ERGM.jl's default). Edge-type counts end at the number
# of dyads they range over: within each selected layer for `LayerEdges` and
# `LayerMutual` (unordered pairs), ordered dyads for the cross-layer
# `InterlayerDependence` on directed layers (unordered on undirected ones)
# and for `MultiplexMutual`; `LayerTriangle` counts from 0 with no finite
# top.
#
# The check matters where the design cannot see the bound: a statistic whose
# change statistics are all ZERO on the observed network — `LayerTriangle` or
# `WithinLayer(Triangle(), l)` on a layer with no two-path — gives the
# pseudo-likelihood a flat direction, not a one-signed gradient, and Newton
# stopped on the singular information at 0 for every coefficient.
_multi_n_pairs(m::MultilayerNetwork{D}) where {D} =
    D ? m.n * (m.n - 1) : m.n * (m.n - 1) ÷ 2
ERGM.Extension.attainable_range(t::LayerEdges, m::MultilayerNetwork) =
    (0.0, Float64(length(_selected_layers(t.layers, m)) * _multi_n_pairs(m)))
ERGM.Extension.attainable_range(t::LayerMutual, m::MultilayerNetwork) =
    (0.0, Float64(length(_selected_layers(t.layers, m)) * (m.n * (m.n - 1) ÷ 2)))
ERGM.Extension.attainable_range(::LayerTriangle, m::MultilayerNetwork) = (0.0, Inf)
ERGM.Extension.attainable_range(::InterlayerDependence, m::MultilayerNetwork) =
    (0.0, Float64(_multi_n_pairs(m)))
ERGM.Extension.attainable_range(::MultiplexMutual, m::MultilayerNetwork) =
    (0.0, Float64(m.n * (m.n - 1)))

# A within-layer term pools the wrapped term's statistic over the selected
# layers (`compute` sums the per-layer values), so its range is the sum of
# the ranges ERGM.jl declares for the term as materialized on each selected
# layer's own network; the pooled value sits at the bottom (top) of it
# exactly when every selected layer's does. No selected layer: no range.
function ERGM.Extension.attainable_range(t::WithinLayer, m::MultilayerNetwork)
    lo = hi = 0.0
    any_layer = false
    for l in _selected_layers(t.layer, m)
        _selects(t, l) || continue
        a, b = ERGM.Extension.attainable_range(_layer_term(t.term, l), m.layers[l])
        lo += a
        hi += b
        any_layer = true
    end
    return any_layer ? (lo, hi) : (-Inf, Inf)
end

# Core multilayer MPLE over the within-layer dyads: build the design, then
# fit the FREE coefficients with ERGM.jl's pseudo-likelihood fitter
# `ERGM.Extension.mple_fit_design` — the one code path every ERGM-family MPLE
# runs on (TERGM's CMPLE, ERGMRank's swap MPLE, ERGM's own). Shared by
# `ergm_multi`, the MCMLE's start and the parametric bootstrap's refits.
#
# That function applies, in order:
#
# 1. R ergm's boundary drop. A statistic at an end of its attainable range —
#    `extreme` (`ERGM.Extension.extreme_statistics`, through this package's
#    `attainable_range` methods: its observed value equals the bound, which
#    the design cannot show when its change statistics are all zero)
#    iterated with the design test (`ERGM.Extension.boundary_columns`, R's
#    `ergm.checkextreme.model`) — is fixed at ∓Inf with standard error 0,
#    and the other coefficients are the MPLE on the rows the dropped columns
#    do not touch, with the offset `η0` kept there (the exact limit of the
#    pseudo-likelihood). ergm.multi 0.3.0 does NOT drop: its layer operator
#    does not propagate the range, so R warns "The MPLE does not exist!" and
#    returns a finite ~−17 with a standard error in the thousands for the
#    same design, while its finite coefficients agree with ours (section
#    (iii) of test/fixtures/twolayer_layer_terms.toml, `boundary_multi.toml`);
# 2. aliasing: a statistic that does not vary on the rows fitted, or is a
#    linear combination of the ones before it, is reported as NaN (R: NA,
#    with "Model statistics ... are not varying") and the rest fitted
#    without it;
# 3. Newton (`NetworkCore.newton_fit` on `NetworkCore.logistic_derivatives`)
#    and NetworkCore's separation verdict on the design actually fitted,
#    returned with the fit (so it is never computed twice).
#
# Its warnings close with R ergm's sentences, and two of them would misstate
# what ergm.multi does ("R ergm reports the same" of the drop), so they are
# worded in ergm.multi's terms through its `note` keyword
# (`_MULTI_MPLE_NOTES`). `warn=false` (the bootstrap's refits, the MCMLE's
# start) silences them: a boundary in a SIMULATED replicate is not a fact
# about the data. `drop=false` refuses a boundary statistic instead, before
# anything is fitted (R's `control.ergm(drop=FALSE)`, whose "MLE is poorly
# defined", is not implemented).
#
# Returns `(θ, se, vcov, loglik, converged, separated, separated_terms,
# n_kept, boundary, aliased)` over the FREE columns: `n_kept` is the number
# of dyads the finite coefficients were estimated on (the BIC sample size),
# `boundary` the `(free column, :min/:max)` pairs, `aliased` the NaN columns.
function _multi_mple_fit(m::MultilayerNetwork, terms::TermSet,
                         offsets::Dict{Int, Float64}, free::Vector{Int};
                         maxiter::Int=100, tol::Float64=1e-8, warn::Bool=true,
                         drop::Bool=true, context::AbstractString="ergm_multi")
    Xf, y, η0 = _multi_mple_design(m, terms, offsets, free)
    n_tot = ones(length(y))
    n_one = Float64.(y)
    free_names = terms.names[free]
    pos = Dict(k => j for (j, k) in enumerate(free))
    extreme = Tuple{Int, Symbol}[(pos[k], s) for (k, s) in
                                 ERGM.Extension.extreme_statistics(terms, m) if haskey(pos, k)]
    boundary = ERGM.Extension.boundary_columns(Xf, n_tot, n_one; fixed=extreme)
    drop || isempty(boundary) || _refuse_multi_no_drop(free_names, boundary; context=context)

    has_offset = any(!iszero, η0)
    r = ERGM.Extension.mple_fit_design(Xf, n_tot, n_one, free_names; maxiter=maxiter,
                                       tol=tol, warn=warn, context="ergm_multi",
                                       note=_MULTI_MPLE_NOTES, extreme=boundary,
                                       offset=has_offset ? η0 : nothing)
    # Nothing left to fit (every free column dropped or aliased): the
    # pseudo-log-likelihood is the offsets' alone on the untouched rows
    loglik = isempty(r.fitted) ?
        sum(y[i] ? -log1p(exp(-η0[i])) : -log1p(exp(η0[i])) for i in r.kept_rows; init=0.0) :
        r.loglik
    return (θ=r.coefficients, se=r.std_errors, vcov=r.var_cov, loglik=loglik,
            converged=r.converged, separated=r.separated,
            separated_terms=r.separated_terms, n_kept=Int(r.n_kept),
            boundary=r.boundary, aliased=r.aliased)
end

# The closing sentences of the MPLE's warnings, in ergm.multi's terms
# (`ERGM.Extension.mple_fit_design`'s `note` keyword). The boundary sentence
# is R ergm's, but ERGM.jl's parenthesis "R ergm reports the same" is true of
# `ergm` and false of `ergm.multi` 0.3.0: its layer operator does not
# propagate the range, so for the same design R warns "The MPLE does not
# exist!" and returns a finite coefficient with a huge standard error
# (frozen as section (iii) of `twolayer_layer_terms.toml`). A migrant who has
# just seen R return a finite number must not be told R reports the same;
# the estimation guide quotes this sentence. The separation sentence is the
# one ergm.multi itself prints for the same design (checked on the separated
# test design: R warns and returns finite coefficients).
const _R_MULTI_MPLE_NOT_EXIST =
    "R ergm.multi warns \"The MPLE does not exist!\" for the same design."
const _MULTI_MPLE_NOTES = (
    boundary="R ergm's drop=TRUE; ergm.multi 0.3.0 warns \"The MPLE does not exist!\" " *
             "and returns a finite value with a standard error in the thousands " *
             "instead — see the estimation guide",
    not_varying="ergm.multi warns \"Model statistics ... are not varying\" and reports NA",
    linear_dependence="R's glm reports NA",
    separation=_R_MULTI_MPLE_NOT_EXIST)

# `drop=false`: a statistic at the boundary of its attainable range is refused
# before any start instead of fixed at ∓Inf (R's `drop=FALSE` keeps the term
# and fits a model whose "MLE is poorly defined", which is not implemented)
function _refuse_multi_no_drop(names::Vector{String},
                               boundary::Vector{Tuple{Int, Symbol}};
                               context::AbstractString)
    lo = [names[j] for (j, s) in boundary if s === :min]
    hi = [names[j] for (j, s) in boundary if s === :max]
    parts = String[]
    isempty(lo) || push!(parts, "observed statistic(s) $(join(lo, ", ")) are at their " *
                                "smallest attainable values (coefficient -Inf)")
    isempty(hi) || push!(parts, "observed statistic(s) $(join(hi, ", ")) are at their " *
                                "largest attainable values (coefficient +Inf)")
    throw(ArgumentError(
        "$context: " * join(parts, "; ") * ". No finite estimate exists, and " *
        "drop=false asks to keep such a statistic in the model (R's drop=FALSE, " *
        "whose \"MLE is poorly defined\"), which is not implemented. Use the " *
        "default drop=true — the coefficient fixed at ±Inf and the rest " *
        "estimated, as R ergm does — or remove the term(s)."))
end

# StatsAPI interface: methods on the shared statistics generics (mirroring
# ERGM.jl), so results interoperate with StatsBase/GLM-style tooling

# The StatsAPI accessors. The examples below share one 4-actor network on
# which every quantity has a closed form: 12 ordered dyads per directed
# layer, 2 ties in layer 1 and 1 in layer 2, so the per-layer edges MPLE is
# logit(density) exactly (the pseudo-likelihood IS the likelihood here).

"""
    coef(r::MultiERGMResult) -> Vector{Float64}

The fitted coefficients, one per statistic of `r.model.terms` (labelled by
`r.model.terms.names`). An offset term reports its fixed value; a statistic
at the boundary of its attainable range reports `∓Inf` (R's `drop`).

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 3); add_layer_edge!(m, :b, 1, 3)
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
coef(fit) ≈ [log(2 / 10), log(1 / 11)]    # true — logit(density) per layer
```
"""
StatsAPI.coef(r::MultiERGMResult) = r.coefficients

"""
    coefnames(r::MultiERGMResult) -> Vector{String}

The coefficient labels, in `coef(r)` order — `ergm.multi`'s labels
(`L(a)~edges`, `L(a&b)~edges`, `L((a,b))~mutual`, …), an offset row wrapped
as `offset(<label>)`, as R's `names(coef(fit))` prints them. They are the row
names of `coeftable(r)`; the vector is a copy, so changing it does not change
the fit. A method of `StatsAPI.coefnames` (StatsBase's `coefnames`).

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 3); add_layer_edge!(m, :b, 1, 3)
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)]; offsets=Dict(2 => -2.0))
coefnames(fit)                          # ["L(a)~edges", "offset(L(b)~edges)"]
coefnames(fit) == coeftable(fit).names  # true
```
"""
StatsAPI.coefnames(r::MultiERGMResult) = _coef_names(r)

"""
    stderror(r::MultiERGMResult) -> Vector{Float64}

The standard errors the fit reports — inverse pseudo-Hessian or parametric
bootstrap, whichever `se_method(r)` says; `NaN` for an offset term, `0.0`
for a coefficient fixed at `∓Inf`.

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 3); add_layer_edge!(m, :b, 1, 3)
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
stderror(fit)[1] ≈ sqrt(1 / (12 * (2 / 12) * (10 / 12)))   # true — binomial information
fit_off = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)]; offsets=Dict(2 => -2.0))
isnan(stderror(fit_off)[2])                                 # true — fixed, no uncertainty
```
"""
StatsAPI.stderror(r::MultiERGMResult) = r.std_errors

"""
    vcov(r::MultiERGMResult) -> Matrix{Float64}

The covariance matrix of the coefficients (`p × p`), of which
`stderror(r)` is the square root of the diagonal; offset rows and columns
are `NaN`.

# Example
```julia
using ERGMMulti, LinearAlgebra
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 3); add_layer_edge!(m, :b, 1, 3)
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
size(vcov(fit))                              # (2, 2)
sqrt.(diag(vcov(fit))) ≈ stderror(fit)       # true
vcov(fit)[1, 2] ≈ 0.0                        # true — the layers' MPLEs are independent
```
"""
StatsAPI.vcov(r::MultiERGMResult) = r.vcov

"""
    loglikelihood(r::MultiERGMResult) -> Float64

The maximized **pseudo**-log-likelihood over the within-layer dyads — the
log-likelihood itself only when every term is dyad-independent
(`is_exact(r)`) — for an MPLE fit; for a `method=:mcmle` fit, the
log-likelihood at the estimates, by path sampling (`r.mcmc.loglik_mc_se` is
its Monte-Carlo standard error).

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 3); add_layer_edge!(m, :b, 1, 3)
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
ll = 2 * log(2 / 12) + 10 * log(10 / 12) + log(1 / 12) + 11 * log(11 / 12)
loglikelihood(fit) ≈ ll                      # true — two independent binomials
```
"""
StatsAPI.loglikelihood(r::MultiERGMResult) = r.loglik

"""
    aic(r::MultiERGMResult) -> Float64

`-2 · loglikelihood(r) + 2 · dof(r)` (pseudo-likelihood based whenever the
model has a dyad-dependent term).

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 3); add_layer_edge!(m, :b, 1, 3)
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
aic(fit) ≈ -2 * loglikelihood(fit) + 2 * dof(fit)    # true
```
"""
StatsAPI.aic(r::MultiERGMResult) = r.aic

"""
    bic(r::MultiERGMResult) -> Float64

`-2 · loglikelihood(r) + dof(r) · log(n)`, where `n` is the number of
within-layer dyads the coefficients were estimated on (`nobs(r)`, unless a
boundary statistic was dropped — then the dyads it does not touch).

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 3); add_layer_edge!(m, :b, 1, 3)
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
bic(fit) ≈ -2 * loglikelihood(fit) + dof(fit) * log(nobs(fit))    # true
```
"""
StatsAPI.bic(r::MultiERGMResult) = r.bic

"""
    nobs(r::MultiERGMResult) -> Int

The number of within-layer dyads — the rows of the pseudo-likelihood:
`L · n · (n − 1)` for directed layers, `L · n · (n − 1) / 2` for undirected.

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2)
nobs(fit_ergm_multi(m, [LayerEdges()]))      # 24 — 2 layers × 12 ordered dyads
```
"""
StatsAPI.nobs(r::MultiERGMResult) = _n_within_dyads(r.model.network)
"""
    dof(r::MultiERGMResult) -> Int

The number of estimated parameters: the free (non-offset) coefficients that
are finite — a coefficient fixed at `∓Inf` by a boundary statistic is not an
estimated parameter, and neither is an offset.

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :b, 1, 3)
dof(fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)]))                          # 2
dof(fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)]; offsets=Dict(1 => -2.0)))  # 1
```
"""
StatsAPI.dof(r::MultiERGMResult) =
    count(k -> isfinite(r.coefficients[k]) && !haskey(r.model.offsets, k),
          eachindex(r.coefficients))

"""
    confint(r::MultiERGMResult; level=0.95) -> Matrix{Float64}

Normal-theory (Wald) confidence limits `θ̂ ± z_{(1+level)/2} · se`, one row
per coefficient with the lower limit in column 1 and the upper in column 2
(a method of `StatsAPI.confint`). The standard errors are the ones the fit
reports (`se_method(r)`: inverse pseudo-Hessian, parametric bootstrap or
Fisher information plus Monte-Carlo error). For an MPLE fit
(`method=:mple`) of a dyad-dependent model with `se` not given, whose naive
pseudo-likelihood SEs under-cover, `confint` refuses with an `ArgumentError`
(`r.inference_withheld`); refit with the MCMLE (the default) or
`se=:bootstrap` for calibrated intervals, or with an explicit
`se=:hessian` to accept the naive ones. A separated fit
(`r.separated`) has no interval: every row is `NaN`. Offset rows are `NaN`:
a fixed coefficient has no interval.

# Example
```julia
using ERGMMulti, ERGM, NetworkCore, Random
rng = Xoshiro(3)
m = MultilayerNetwork(10; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
for l in 1:2, i in 1:10, j in 1:10
    i != j && rand(rng) < 0.3 && add_layer_edge!(m, l, i, j)
end
fit = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
ci = confint(fit)                          # 2×2
all(ci[:, 1] .< coef(fit) .< ci[:, 2])     # true
```
"""
function StatsAPI.confint(r::MultiERGMResult; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("confint: level must be in (0, 1) (got $level)"))
    # A separated fit has no finite maximum: no interval on any coefficient
    # (the ecosystem's separation policy)
    r.separated && return fill(NaN, length(r.coefficients), 2)
    r.inference_withheld && throw(ArgumentError(
        "confint: no interval is reported for the MPLE of a " *
        "dyad-dependent multilayer formula — its naive pseudo-likelihood " *
        "standard errors under-cover (95% Wald intervals covered 0.76-0.94 in " *
        "simulation). Refit with method=:mcmle (maximum likelihood) or " *
        "se=:bootstrap (parametric bootstrap), or pass " *
        "se=:hessian explicitly to accept the naive Wald intervals."))
    q = quantile(Normal(), 1 - (1 - level) / 2)
    θ, se = r.coefficients, r.std_errors
    return hcat(θ .- q .* se, θ .+ q .* se)
end

"""
    coeftable(r::MultiERGMResult) -> NetworkCore.CoefficientTable

The R-style coefficient table (`Estimate`, `Std.Error`, `z value`,
`Pr(>|z|)`) as an inspectable `NetworkCore.CoefficientTable` — exactly the table
`show(r)` prints, built from the same vectors (a method of
`StatsAPI.coeftable`). Rows are labelled with the model's direction-aware
statistic names (`r.model.terms.names`, `ergm.multi`'s labels), offset rows
labelled `offset(<name>)` as R labels them, with `NaN` standard error, z
and p; the z → p map is the shared `NetworkCore.z_pvalues`. For an MPLE fit
of a dyad-dependent formula with `se` not given (`r.inference_withheld`)
every finite coefficient's z and p is `NaN`, and so is every z and p of a
separated fit (`r.separated`).

# Example
```julia
using ERGMMulti, ERGM, NetworkCore, Random
rng = Xoshiro(3)
m = MultilayerNetwork(10; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
for l in 1:2, i in 1:10, j in 1:10
    i != j && rand(rng) < 0.3 && add_layer_edge!(m, l, i, j)
end
fit = ergm_multi(m, [LayerEdges(1), LayerEdges(2)]; offsets=Dict(2 => -1.0))
tbl = coeftable(fit)
tbl.names                                   # ["L(a)~edges", "offset(L(b)~edges)"]
tbl["L(a)~edges"].estimate == coef(fit)[1]  # true
isnan(tbl[2].p_value)                       # true
```
"""
function StatsAPI.coeftable(r::MultiERGMResult)
    z = r.coefficients ./ r.std_errors
    p = z_pvalues(z)
    if r.separated
        # No finite maximum: no Wald inference on any coefficient (the others
        # are estimated conditionally on the separated ones being infinite)
        fill!(z, NaN)
        fill!(p, NaN)
    elseif r.inference_withheld
        # No test is built on the naive standard errors of a dyad-dependent
        # MPLE unless asked for in writing (`se=:hessian`). A coefficient
        # fixed at ∓Inf keeps its conventional z = ∓Inf, p = 0.
        for k in eachindex(z)
            isfinite(r.coefficients[k]) || continue
            z[k] = NaN
            p[k] = NaN
        end
    end
    return CoefficientTable(_coef_names(r), r.coefficients, r.std_errors;
                            z_values=z, p_values=p)
end

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate_multi_ergm(m::MultilayerNetwork, terms, θ; n_sim=1, burnin=nothing,
                        interval=nothing, rng=Random.default_rng())
        -> Vector{MultilayerNetwork}
    simulate_multi_ergm(model::MultiERGMModel, θ; kwargs...)

Simulate multilayer networks from `P(y) ∝ exp(θ'g(y))` with a Metropolis
sampler restricted to **within-layer dyads** (matching the dyad universe
of the block-diagonal model): propose toggling a random (layer, i, j),
accept an addition with `min(1, exp(θ'Δg))` and a removal with
`min(1, exp(−θ'Δg))`. The chain is the ONE shared Metropolis kernel of the
ERGM family, `ERGM.mh_toggle!`, with `(layer, i, j)` moves; the rng is drawn
proposal-first, then acceptance, so seeded runs are reproducible.

The `(m, terms, θ)` form builds a [`MultiERGMModel`](@ref) first, so the
terms are validated against the layers (an out-of-range layer, a missing
attribute, an undirected-only term on a directed layer, ... throw before the
chain starts). `θ` has one entry per **statistic** of the model
(`model.terms.names`), which equals one per term unless a `WithinLayer` term
expands (a multi-level `NodeFactor`, a `Degree(0:2)`).

`burnin`/`interval` default (`nothing`) to the dyad-scaled rule shared with
every ERGM-family sampler (`ERGM.Extension.mcmc_defaults`): `20 × n_dyads` and
`max(100, n_dyads ÷ 10)` over the within-layer dyads.

A layer with a masked (missing) dyad is refused with an `ArgumentError`
naming the layer (`NetworkCore.missing_policies(simulate_multi_ergm) ==
(:error,)`): the chain would otherwise toggle the unobserved dyad at its
face value. **Every coefficient must be finite**: a `∓Inf` (a boundary
statistic's R-drop value, `coef(fit)` of such a fit) or `NaN` entry is
refused with an `ArgumentError` naming it ("simulate_multi_ergm: every
coefficient must be finite (got L(a&b)~edges = -Inf) …") — the chain would
otherwise reject every proposal (θ'Δ is NaN) and return copies of the
input. Remove the dropped term, as R's `drop=TRUE` does, and simulate from
that model.

# Example
```julia
using ERGMMulti, ERGM, Random
m = MultilayerNetwork(8; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
draws = simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                            [-1.5, 2.0]; n_sim=10, rng=Xoshiro(1))
length(draws)                         # 10
draws[1] isa MultilayerNetwork{true}  # true
```
"""
function simulate_multi_ergm(m::MultilayerNetwork, terms, θ::AbstractVector{<:Real};
                             kwargs...)
    n_layers(m) >= 1 || throw(ArgumentError("network has no layers"))
    return simulate_multi_ergm(MultiERGMModel(terms, m), θ; kwargs...)
end

function simulate_multi_ergm(model::MultiERGMModel, θ::AbstractVector{<:Real};
                             n_sim::Int=1,
                             burnin::Union{Nothing, Int}=nothing,
                             interval::Union{Nothing, Int}=nothing,
                             rng::Random.AbstractRNG=Random.default_rng())
    terms = model.terms
    length(θ) == length(terms) || throw(ArgumentError(
        "θ must have one coefficient per model statistic: got $(length(θ)) for " *
        "$(length(terms)) statistics ($(join(terms.names, ", ")))"))
    _require_finite_coefficients(θ, terms.names, "simulate_multi_ergm")
    n_sim >= 0 || throw(ArgumentError("n_sim must be ≥ 0 (got $n_sim)"))
    m = model.network
    # A masked dyad would be toggled at its face value by the chain: refuse
    # (the missing-data contract; `missing_policies(simulate_multi_ergm) ==
    # (:error,)`)
    _require_observed_layers(m, "simulate_multi_ergm")
    burnin, interval = _resolve_multi_mcmc(m, burnin, interval)
    # Function barrier: `model.terms` is an abstractly typed field; the kernel
    # below runs on the concrete TermSet{T}.
    return _multi_mh_run(rng, m, terms, Vector{Float64}(θ), n_sim, burnin, interval)
end

# A ∓Inf or NaN coefficient (a boundary statistic's R-drop value, or a slip)
# would freeze the chain, not error it: `dot(θ, Δ)` is NaN wherever the term
# changes (or everywhere, for NaN), `log(u) < NaN` is always false and every
# proposal is rejected — the "simulated" networks are copies of the input and
# a `gof` built on them reports p = 1 for every statistic. Refuse it instead,
# naming the coefficients, as the bootstrap already does.
function _require_finite_coefficients(θ, names, context::AbstractString)
    all(isfinite, θ) && return nothing
    bad = [string(nm, " = ", c) for (nm, c) in zip(names, θ) if !isfinite(c)]
    throw(ArgumentError(
        "$context: every coefficient must be finite (got $(join(bad, ", "))): a " *
        "multilayer network cannot be simulated at an infinite or NaN " *
        "coefficient — the chain would reject every proposal and return copies " *
        "of the observed network. A coefficient fixed at ±Inf comes from a " *
        "statistic at the boundary of its attainable range (R's drop): remove " *
        "the dropped term as R's drop=TRUE does and refit, then simulate from " *
        "that model."))
end

function _multi_mh_run(rng::Random.AbstractRNG, m::MultilayerNetwork{D},
                       terms::TermSet, θ::Vector{Float64},
                       n_sim::Int, burnin::Int, interval::Int) where {D}
    current = as_multilayer([_copy_net(net) for net in m.layers], m.layer_names)
    layers = current.layers
    draws = MultilayerNetwork{D}[]
    on_sample = k -> push!(draws, as_multilayer([_copy_net(net) for net in layers],
                                                m.layer_names))
    _multi_mh_chain!(rng, current, terms, θ, (delta, removal) -> nothing, on_sample,
                     n_sim, burnin, interval)
    return draws
end

# THE multilayer chain (`simulate_multi_ergm`, `gof`, the bootstrap and the
# MCMLE all run it): `ERGM.mh_toggle!` over `current`, which is mutated.
# `on_apply(delta, removal)` is called after every accepted move with that
# move's add-direction change statistics; `on_sample(k)` at every retained
# state.
function _multi_mh_chain!(rng::Random.AbstractRNG, current::MultilayerNetwork{D},
                          terms::TermSet, θ::Vector{Float64},
                          on_apply::FA, on_sample::FS,
                          n_sim::Int, burnin::Int, interval::Int) where {D, FA, FS}
    L = n_layers(current)
    n = current.n
    layers = current.layers
    delta = Vector{Float64}(undef, length(terms))

    # A move is the within-layer dyad (l, i, j): layer uniform, then an ordered
    # pair of distinct actors — swapped to i < j on undirected layers so every
    # unordered dyad is proposed with equal probability.
    propose = rng -> begin
        l = rand(rng, 1:L)
        i = rand(rng, 1:n)
        j = rand(rng, 1:(n - 1))
        j >= i && (j += 1)
        (!D && j < i) && ((i, j) = (j, i))
        (l, i, j)
    end
    change! = (delta, move) -> begin
        l, i, j = move
        _change_stat_layer_all!(delta, terms, current, l, i, j)
        has_edge(layers[l], i, j)
    end
    apply! = (move, removal) -> begin
        l, i, j = move
        removal ? rem_edge!(layers[l], i, j) : add_edge!(layers[l], i, j)
        on_apply(delta, removal)
        nothing
    end

    mh_toggle!(rng, θ, delta, propose, change!, apply!, on_sample;
               burnin=burnin, interval=interval, n_samples=n_sim)
    return current
end

# Attribute-preserving copy: delegates to `Base.copy(::Network)`, which
# duplicates the graph and all vertex/edge/network attributes, so attribute
# terms (e.g. `WithinLayer(NodeMatch(...), l)`) keep seeing covariates on
# the sampler's working copies.
_copy_net(net::Network) = copy(net)

# The sampler offers no `missing=` keyword: the chain toggles every within-layer
# dyad, so a masked dyad in any layer is refused (`require_observed` on each
# layer, naming it), never conditioned on at face value.
missing_policies(::typeof(simulate_multi_ergm)) = (:error,)

# =============================================================================
# Monte-Carlo maximum likelihood (ergm.multi's default for dependent models)
# =============================================================================

# The statistics of the model over a chain: `G[k, :]` after the k-th retained
# state. The chain is `_multi_mh_chain!` — the sampler of `simulate_multi_ergm`
# — continued from `current` (mutated); the statistics are carried along by
# adding each accepted move's change statistics, so no network is copied.
function _multi_mh_stats!(rng::Random.AbstractRNG, current::MultilayerNetwork,
                          terms::TermSet, θ::Vector{Float64}, G::Matrix{Float64},
                          burnin::Int, interval::Int)
    stats = compute_all(terms, current)
    on_apply = (delta, removal) -> begin
        if removal
            @inbounds for k in eachindex(stats)
                stats[k] -= delta[k]
            end
        else
            @inbounds for k in eachindex(stats)
                stats[k] += delta[k]
            end
        end
        nothing
    end
    on_sample = k -> begin
        @inbounds for c in eachindex(stats)
            G[k, c] = stats[c]
        end
        nothing
    end
    _multi_mh_chain!(rng, current, terms, θ, on_apply, on_sample,
                     size(G, 1), burnin, interval)
    return G
end

# Variance of the mean of a (possibly autocorrelated) scalar series by batch
# means, √n batches
function _batch_var_of_mean(x::AbstractVector{Float64})
    n = length(x)
    b = max(2, floor(Int, sqrt(n)))
    len = n ÷ b
    len >= 1 || return NaN
    means = [sum(@view x[((k - 1) * len + 1):(k * len)]) / len for k in 1:b]
    mu = sum(means) / b
    return sum(abs2, means .- mu) / (b - 1) / b
end

# Path-sampling estimate of the log-likelihood θ'g(y_obs) − log Z(θ) and its
# Monte-Carlo standard error. θ₀ is θ with the dyad-dependent coordinates
# zeroed: a dyad-independent model over the within-layer dyads, whose
# log-likelihood at y_obs is the logistic pseudo-log-likelihood exactly. Along
# θ(u) = θ₀ + u(θ − θ₀), d/du log Z = (θ − θ₀)'E_{θ(u)}[g], integrated by
# ERGM.jl's `ERGM.Extension.bridge_integrate` (composite Simpson's rule over `nrungs`
# segments) from a chain at each grid point. Every grid point has its own generator seeded from `rng` up front and
# its own chain from the observed network, so the estimate is reproducible and
# independent of the thread count.
function _multi_bridge_loglik(model::MultiERGMModel, θ::Vector{Float64},
                              g_obs::Vector{Float64}; nrungs::Int, n_samples::Int,
                              burnin::Int, interval::Int, rng::Random.AbstractRNG)
    isodd(nrungs) && (nrungs += 1)
    m = model.network
    terms = model.terms
    p = length(θ)
    θ0 = copy(θ)
    for (k, t) in enumerate(terms.terms)
        is_dyad_dependent(t) && (θ0[k] = 0.0)
    end
    Δ = θ .- θ0

    # The path integral is ERGM.jl's `ERGM.Extension.bridge_integrate` (composite
    # Simpson over the grid, rungs on separate tasks); this supplies each
    # rung's chain and keeps its batch-means variance for the MC error
    seeds = rand(rng, UInt64, nrungs + 1)
    vars = zeros(nrungs + 1)
    function rung_mean(θu, k)
        G = Matrix{Float64}(undef, n_samples, p)
        current = as_multilayer([_copy_net(net) for net in m.layers], m.layer_names)
        _multi_mh_stats!(Random.Xoshiro(seeds[k]), current, terms, θu, G,
                         burnin, interval)
        vars[k] = _batch_var_of_mean(G * Δ)
        return vec(sum(G, dims=1)) ./ n_samples
    end
    integral = ERGM.Extension.bridge_integrate(rung_mean, θ0, θ; rungs=nrungs, threaded=true)
    # Simpson weights (1, 4, 2, ..., 4, 1)/(3·nrungs) on the rung variances
    variance = 0.0
    for r in 1:(nrungs + 1)
        w = (r == 1 || r == nrungs + 1) ? 1.0 : (iseven(r) ? 4.0 : 2.0)
        variance += (w / (3 * nrungs))^2 * vars[r]
    end

    # Exact log-likelihood of the dyad-independent reference model at y_obs,
    # in the form that stays exact when a statistic held at its bound puts
    # η at ±1e300 (log P(y) is then 0 on the dyads it touches: the observed
    # network is at the bound, so each of them has the one value allowed)
    X, y, _ = _multi_mple_design(m, terms, Dict{Int, Float64}(), collect(1:p))
    η = X * θ0
    ll0 = sum(y[r] ? -log1p(exp(-η[r])) : -log1p(exp(η[r])) for r in eachindex(y);
              init=0.0)
    return ll0 + dot(Δ, g_obs) - integral, sqrt(max(variance, 0.0))
end

# The finite stand-in for a coefficient fixed at ∓Inf in the sampler (ERGM.jl
# holds a dropped statistic the same way): any proposal that moves the
# statistic off its observed bound has θ'Δ ≤ −1e300 and is rejected, while
# θ'Δ stays a number (an infinite coefficient times a zero change statistic
# would be NaN and freeze the chain)
const _HOLD_AT_BOUND = 1e300

# R's sentence for the statistics `method=:mcmle` fixes at ∓Inf (R ergm's
# default `drop=TRUE`), one warning per side, with what ergm.multi does
function _warn_multi_drop(names::Vector{String}, boundary::Vector{Tuple{Int, Symbol}})
    for (side, word, at) in ((:min, "smallest", "-Inf"), (:max, "largest", "+Inf"))
        cols = [names[j] for (j, s) in boundary if s === side]
        isempty(cols) && continue
        @warn "ergm_multi: observed statistic(s) $(join(cols, ", ")) are at their " *
              "$word attainable values. Their coefficients will be fixed at $at " *
              "(no finite maximum-likelihood estimate exists; R ergm's drop=TRUE — " *
              "ergm.multi 0.3.0 does not drop and reports a finite value where its " *
              "estimation stopped). The remaining coefficients are estimated with " *
              "these held fixed: the sampler never moves the statistic off its " *
              "observed bound. Pass drop=false to refuse such a model instead."
    end
    return nothing
end

# Monte-Carlo maximum likelihood over the FREE coefficients (offsets stay
# fixed and enter every chain). See `ergm_multi`, "Estimators".
function _multi_mcmle(model::MultiERGMModel; maxiter::Int=100, tol::Float64=1e-8,
                      n_samples::Int=1024,
                      burnin::Union{Nothing, Int}=nothing,
                      interval::Union{Nothing, Int}=nothing,
                      mcmle_maxiter::Int=60,
                      conv_threshold::Float64=0.1, hotelling_alpha::Float64=0.05,
                      termination::Symbol=:confidence,
                      conv_precision::Float64=0.1, conv_confidence::Float64=0.99,
                      max_n_samples::Union{Nothing, Int}=nothing,
                      bridge_rungs::Int=16,
                      bridge_samples::Union{Nothing, Int}=nothing,
                      init::Union{Nothing, AbstractVector{<:Real}}=nothing,
                      rng::Random.AbstractRNG=Random.default_rng(),
                      drop::Bool=true, warn::Bool=true)
    m = model.network
    terms = model.terms
    offsets = model.offsets
    names = terms.names
    p = length(terms)
    free = [k for k in 1:p if !haskey(offsets, k)]
    pf = length(free)
    n_samples >= 16 || throw(ArgumentError(
        "ergm_multi: n_samples must be at least 16 for method=:mcmle (got $n_samples)"))
    mcmle_maxiter >= 1 || throw(ArgumentError(
        "ergm_multi: mcmle_maxiter must be at least 1 (got $mcmle_maxiter)"))
    bridge_rungs >= 0 || throw(ArgumentError(
        "ergm_multi: bridge_rungs must be non-negative (got $bridge_rungs)"))
    termination in (:confidence, :hotelling) || throw(ArgumentError(
        "ergm_multi: termination must be :confidence or :hotelling (got :$termination)"))
    n_max = something(max_n_samples, 16 * n_samples)
    n_max >= n_samples || throw(ArgumentError(
        "ergm_multi: max_n_samples ($n_max) is below n_samples ($n_samples)"))
    all(isfinite, values(offsets)) || throw(ArgumentError(
        "ergm_multi: method=:mcmle needs finite offsets (a chain cannot be run " *
        "at an infinite coefficient)"))

    # The MPLE start, with R's boundary drop: a statistic at an end of its
    # attainable range has no finite MLE either. As R ergm's default
    # `drop=TRUE`, its coefficient is fixed at ∓Inf and the sampler holds the
    # statistic at its observed bound (±1e300 in θ, so every proposal that
    # moves it off the bound is rejected); the other coefficients are
    # estimated with it fixed. `drop=false` refuses instead, before any start.
    start = _multi_mple_fit(m, terms, offsets, free; maxiter=maxiter, tol=tol,
                            warn=false, drop=drop)
    dropped = Dict(free[j] => side for (j, side) in start.boundary)
    warn && !isempty(dropped) &&
        _warn_multi_drop(names, [(k, dropped[k]) for k in sort!(collect(keys(dropped)))])
    # A statistic with no identifiable coefficient at the start (it does not
    # vary on the dyads fitted, or is a combination of the others: NaN, R's
    # NA). A dyad-independent one stays so under every θ — its change
    # statistics do not depend on the network — so it is held at 0 and
    # reported NaN; a dyad-dependent one may vary once the chain moves, so it
    # is estimated from `init=`, or refused without one.
    aliased = [free[j] for j in start.aliased]
    held = [k for k in aliased if !is_dyad_dependent(terms.terms[k])]
    dd_aliased = [k for k in aliased if is_dyad_dependent(terms.terms[k])]
    (isempty(dd_aliased) || init !== nothing) || throw(ArgumentError(
        "ergm_multi: method=:mcmle — statistic(s) " *
        join(names[dd_aliased], ", ") * " do not vary on the observed network's " *
        "within-layer dyads (or are linear combinations of the others), so the " *
        "MPLE start has no value for their coefficients. Remove the term(s), or " *
        "pass `init=` to start the Monte-Carlo iteration from chosen values."))
    _refuse_mixed_holds([haskey(dropped, k) ? (dropped[k] === :min ? -Inf : Inf) : 0.0
                         for k in 1:p], names, "ergm_multi: method=:mcmle")
    est = [k for k in free if !haskey(dropped, k) && !(k in held)]
    isempty(est) && throw(ArgumentError(
        "ergm_multi: method=:mcmle — every free coefficient is fixed at ±Inf by a " *
        "statistic at the boundary of its attainable range or is not " *
        "identifiable, so there is nothing to estimate; method=:mple reports " *
        "the limits."))
    start.separated && throw(ArgumentError(
        "ergm_multi: method=:mcmle — the MPLE start does not exist (separation: " *
        "the pseudo-likelihood keeps increasing as the coefficient(s) on " *
        join(start.separated_terms, ", ") * " run to ±Inf), so there is nothing " *
        "to start the Monte-Carlo iteration from; a design separated this way " *
        "has no finite maximum-likelihood estimate either. Remove, merge or " *
        "coarsen the separating term(s)."))
    (start.converged || init !== nothing) || throw(ArgumentError(
        "ergm_multi: method=:mcmle — the MPLE start did not converge; fix the " *
        "model (see method=:mple), raise `maxiter`, or pass `init=`."))

    # θ as the sampler sees it: offsets, the dropped coordinates at ±1e300,
    # the held ones at 0, the estimated ones at their start
    θ = zeros(p)
    for (k, c) in offsets
        θ[k] = c
    end
    for (k, side) in dropped
        θ[k] = side === :min ? -_HOLD_AT_BOUND : _HOLD_AT_BOUND
    end
    start_full = copy(θ)
    start_full[free] = start.θ
    if init === nothing
        θ[est] = start_full[est]
    else
        length(init) == p || throw(ArgumentError(
            "ergm_multi: init has $(length(init)) coefficients for $p statistics"))
        θ[est] = Float64.(init[est])
    end
    all(isfinite, θ[est]) || throw(ArgumentError(
        "ergm_multi: method=:mcmle needs finite starting coefficients"))

    g_obs = compute_all(terms, m)
    burn, intv = _resolve_multi_mcmc(m, burnin, interval)
    current = as_multilayer([_copy_net(net) for net in m.layers], m.layer_names)

    # The iteration is ERGM.jl's `ERGM.Extension.mcmle_solve` — the loop `ERGM.mcmle`
    # runs (Hummel-stepped Newton update, R's confidence stopping rule with
    # its sample boost, Fisher + Monte-Carlo covariance). This supplies only
    # the sampler: THE chain of `simulate_multi_ergm`, continued across
    # draws, with the offsets fixed in every draw.
    θfull = copy(θ)
    function draw(θf, n)
        θfull[est] = θf
        G = Matrix{Float64}(undef, n, p)
        _multi_mh_stats!(rng, current, terms, θfull, G, burn, intv)
        return G[:, est]
    end
    solve() = ERGM.Extension.mcmle_solve(draw, θ[est]; labels=names[est],
                                n_samples=n_samples, maxiter=mcmle_maxiter,
                                termination=termination,
                                conv_precision=conv_precision,
                                conv_confidence=conv_confidence,
                                conv_threshold=conv_threshold,
                                hotelling_alpha=hotelling_alpha,
                                max_n_samples=n_max, target=g_obs[est],
                                context="ergm_multi MCMLE")
    sol = warn ? solve() :
          Base.CoreLogging.with_logger(solve, Base.CoreLogging.NullLogger())
    θ[est] = sol.coef
    converged = sol.converged
    iterations = sol.iterations
    Vtot, se, mcse = sol.vcov, sol.se, sol.mcmc_se
    singular = !converged && all(isnan, se)
    Gfinal = Matrix{Float64}(sol.final.samples)
    n_final = size(Gfinal, 1)
    convergence = ERGM.MCMLEConvergence((iterations, sol.step_length,
                                         sol.tests.t_ratios, sol.tests.hotelling_p,
                                         sol.tests.n_eff))

    # The log-likelihood by path sampling from the dyad-independent part of
    # the model. A dropped dyad-independent statistic is held in the
    # reference model too, so the bridge does not move it; a dropped
    # dyad-DEPENDENT one is not representable there (the reference model's
    # normalizer would range over networks the fitted model excludes), so the
    # log-likelihood is not computed and the note says why.
    dropped_dd = [names[k] for k in keys(dropped) if is_dyad_dependent(terms.terms[k])]
    loglik_note = isempty(dropped_dd) ? nothing :
        "log-likelihood, AIC and BIC not computed (NaN): the statistic(s) " *
        "$(join(sort!(dropped_dd), ", ")) held at their bound are dyad-dependent, " *
        "so the dyad-independent reference model of the path sampler cannot hold " *
        "them there"
    loglik, ll_se = NaN, NaN
    if bridge_rungs > 0 && !singular && loglik_note === nothing
        loglik, ll_se = _multi_bridge_loglik(
            model, θ, g_obs; nrungs=bridge_rungs,
            n_samples=something(bridge_samples, n_samples),
            burnin=burn, interval=intv, rng=rng)
    end

    # Report: a dropped coefficient is ∓Inf with standard error 0 (R's drop),
    # a held one NaN (R's NA); neither enters the step, the stopping rule or
    # the covariance
    θ_report = copy(θ)
    for (k, side) in dropped
        θ_report[k] = side === :min ? -Inf : Inf
    end
    θ_report[held] .= NaN
    start_full[collect(keys(dropped))] .= θ_report[collect(keys(dropped))]
    std_errors = fill(NaN, p)
    vcov_full = fill(NaN, p, p)
    vcov_full[est, est] = Vtot
    std_errors[est] = se
    for k in keys(dropped)
        std_errors[k] = 0.0
        vcov_full[k, free] .= 0.0
        vcov_full[free, k] .= 0.0
    end
    pe = length(est)
    aic = -2 * loglik + 2 * pe
    bic = -2 * loglik + pe * log(start.n_kept)
    mcmc = (convergence=convergence, mc_std_errors=mcse, samples=Gfinal,
            n_samples=n_final, burnin=burn, interval=intv, loglik_mc_se=ll_se,
            start=start_full, termination=termination,
            termination_p=sol.termination_p, conv_confidence=conv_confidence,
            conv_precision=conv_precision, estimated=est, loglik_note=loglik_note)
    if warn && !converged && !singular
        @warn "ergm_multi: MCMLE did not converge in $iterations iterations " *
              "($(_termination_verdict(mcmc))): the estimates are the last " *
              "iterate, not a maximum of the likelihood. Raise `mcmle_maxiter` or " *
              "`n_samples`; `fit.converged == false` records this."
    end
    return MultiERGMResult(model, θ_report, std_errors, vcov_full, loglik, aic, bic,
                           converged, :mcmc, nothing, false, false, :mcmle, mcmc,
                           String[])
end

# =============================================================================
# Goodness of fit
# =============================================================================

# The coefficients `gof` simulates at. A coefficient fixed at ∓Inf by R's drop
# is the statistic held at its bound: the chain starts at the observed
# network, which is there, and `_HOLD_AT_BOUND` rejects every move off it —
# the fitted model, not an approximation of it. A NaN (not identifiable)
# coefficient of a dyad-independent statistic is simulated at 0: such a
# statistic is a constant or a combination of the others under every θ, so
# any value gives the same model. A dyad-dependent NaN has no such
# reading and is refused, naming it.
function _gof_theta(result::MultiERGMResult)
    terms = result.model.terms
    θ = copy(result.coefficients)
    for k in eachindex(θ)
        if isinf(θ[k])
            θ[k] = sign(θ[k]) * _HOLD_AT_BOUND
        elseif isnan(θ[k]) && !is_dyad_dependent(terms.terms[k])
            θ[k] = 0.0
        end
    end
    _require_finite_coefficients(θ, terms.names, "gof")
    _refuse_mixed_holds(result.coefficients, terms.names, "gof")
    return θ
end

# Statistics fixed at -Inf AND at +Inf in one model cannot all be held by
# the ±1e300 stand-ins: on a dyad whose change statistics touch both, the two
# cancel in θ'Δ and the move is no longer forbidden
function _refuse_mixed_holds(θ, names, context::AbstractString)
    lo = [names[k] for k in eachindex(θ) if θ[k] == -Inf]
    hi = [names[k] for k in eachindex(θ) if θ[k] == Inf]
    (isempty(lo) || isempty(hi)) && return nothing
    throw(ArgumentError(
        "$context: $(join(lo, ", ")) fixed at -Inf and $(join(hi, ", ")) fixed at " *
        "+Inf cannot be held at their bounds together by the sampler. Remove the " *
        "term(s) and refit."))
end

"""
    gof(result::MultiERGMResult; n_sim=100, burnin=nothing, interval=nothing,
        rng=Random.default_rng()) -> GOFResult

Goodness-of-fit assessment for a fitted multilayer ERGM: simulate `n_sim`
multilayer networks at the fitted (and offset) coefficients with
[`simulate_multi_ergm`](@ref) and compare the observed data against the
simulated distributions. The panels, in order:

- `"model statistics"` — the fitted terms' statistics (one level per
  statistic, labelled as in `coeftable(result)`). For an MPLE fit of a
  dyad-dependent model this panel mostly measures the MPLE's bias, not
  misfit — read the auxiliary panels for that;
- `"layer edges"` — the edge count of each layer (one level per layer);
- `"multiplexity"` — the number of dyads tied in exactly `k` of the `L`
  layers, `k = 0, …, L` (ordered dyads on directed layers): the cross-layer
  structure no single-layer panel sees;
- per layer `<name>`, the auxiliary distributions of `gof.ergm`, none of
  which the model fits by construction:
  `"degree: <name>"` on undirected layers, `"idegree: <name>"` and
  `"odegree: <name>"` on directed ones (the number of actors with degree
  `0, 1, …`), and `"esp: <name>"` (the number of ties whose endpoints share
  `0, 1, …` partners; outgoing two-paths `i→k→j` on directed layers,
  statnet's default `OTP`). Levels run up to the largest value seen in the
  observed or any simulated network.

`burnin`/`interval` default to the dyad-scaled rule of every ERGM-family
sampler (`ERGM.Extension.mcmc_defaults`). Extends NetworkCore.jl's shared `gof` generic
and returns the shared `NetworkCore.GOFResult` container; per-level p-values
are two-sided Monte-Carlo p-values computed with the `(1 + k)/(N + 1)`
estimator (never exactly zero). The fitted network is never masked (the
estimator refused it), so the simulation needs no `missing=` policy.
Geodesic-distance panels are not implemented.

A fit with a coefficient fixed at `∓Inf` (a statistic at the boundary of
its attainable range, R's `drop`) is simulated as fitted: the chain starts
at the observed network, which is at that bound, and never moves the
statistic off it. A `NaN` coefficient of a dyad-independent statistic is
simulated at 0 (any value gives the same model); a dyad-dependent `NaN`, or
statistics fixed at both `-Inf` and `+Inf`, are refused with an
`ArgumentError` naming them.

# Example
```julia
using ERGMMulti, ERGM, NetworkCore, Random
rng = Xoshiro(4)
m = MultilayerNetwork(10; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
for l in 1:2, i in 1:10, j in 1:10
    i != j && rand(rng) < 0.3 && add_layer_edge!(m, l, i, j)
end
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
g = gof(fit; n_sim=20, burnin=2000, interval=50, rng=Xoshiro(5))
g isa GOFResult                            # true
[panel.name for panel in g.statistics]
# ["model statistics", "layer edges", "multiplexity", "idegree: a", "odegree: a",
#  "esp: a", "idegree: b", "odegree: b", "esp: b"]
g.statistics[2].labels                     # ["a", "b"] — one level per layer
g.statistics[3].labels                     # ["0", "1", "2"] — dyads tied in k layers
all(0 .< g.statistics[1].p_values .<= 1)   # true — (1+k)/(N+1), never 0
```
"""
function gof(result::MultiERGMResult; n_sim::Int=100,
             burnin::Union{Nothing, Int}=nothing,
             interval::Union{Nothing, Int}=nothing,
             rng::Random.AbstractRNG=Random.default_rng())
    model = result.model
    m = model.network
    terms = model.terms
    sims = simulate_multi_ergm(model, _gof_theta(result); n_sim=n_sim,
                               burnin=burnin, interval=interval, rng=rng)

    # Panel 1: the model's own statistics, observed vs simulated
    obs_stats = compute_all(terms, m)
    sim_stats = Matrix{Float64}(undef, length(sims), length(terms))
    for (s, sim) in enumerate(sims)
        sim_stats[s, :] .= compute_all(terms, sim)
    end
    panels = GOFStatistic[GOFStatistic("model statistics", copy(terms.names),
                                       obs_stats, sim_stats)]

    # Panel 2: per-layer edge counts
    L = n_layers(m)
    obs_edges = [Float64(ne(m.layers[l])) for l in 1:L]
    sim_edges = Float64[ne(s.layers[l]) for s in sims, l in 1:L]
    push!(panels, GOFStatistic("layer edges", String.(m.layer_names),
                               obs_edges, sim_edges))

    # Panel 3: multiplexity — dyads tied in exactly k layers, k = 0..L
    push!(panels, GOFStatistic("multiplexity", string.(0:L),
                               _multiplexity_counts(m),
                               _stack_counts([_multiplexity_counts(s) for s in sims], L + 1)))

    # Auxiliary per-layer distributions (gof.ergm's degree and ESP panels)
    for l in 1:L
        lname = String(m.layer_names[l])
        if is_directed(m)
            _push_distribution!(panels, "idegree: $lname",
                                _degree_counts(m.layers[l], :in),
                                [_degree_counts(s.layers[l], :in) for s in sims])
            _push_distribution!(panels, "odegree: $lname",
                                _degree_counts(m.layers[l], :out),
                                [_degree_counts(s.layers[l], :out) for s in sims])
        else
            _push_distribution!(panels, "degree: $lname",
                                _degree_counts(m.layers[l], :total),
                                [_degree_counts(s.layers[l], :total) for s in sims])
        end
        _push_distribution!(panels, "esp: $lname", _esp_counts(m.layers[l]),
                            [_esp_counts(s.layers[l]) for s in sims])
    end

    return GOFResult(panels; model="Multilayer ERGM")
end

# counts[k + 1] = number of actors of (in/out/total) degree k, k = 0..n−1
function _degree_counts(net::Network, mode::Symbol)
    n = nv(net)
    counts = zeros(Float64, n)
    for v in 1:n
        d = mode === :in ? indegree(net, v) : mode === :out ? outdegree(net, v) :
            degree(net, v)
        counts[min(d, n - 1) + 1] += 1.0
    end
    return counts
end

# counts[k + 1] = number of ties whose endpoints share exactly k partners:
# common neighbours on an undirected layer, outgoing two-paths i→k→j on a
# directed one (statnet's OTP, the `esp` of `gof.ergm`)
function _esp_counts(net::Network{T, D}) where {T, D}
    n = nv(net)
    counts = zeros(Float64, max(n - 1, 1))
    for e in edges(net)
        i, j = src(e), dst(e)
        sp = 0
        for k in outneighbors(net, i)
            (k == i || k == j) && continue
            has_edge(net, k, j) && (sp += 1)
        end
        counts[sp + 1] += 1.0
    end
    return counts
end

# counts[k + 1] = number of dyads (ordered on directed layers) tied in
# exactly k layers, k = 0..L
function _multiplexity_counts(m::MultilayerNetwork{D}) where {D}
    L = n_layers(m)
    counts = zeros(Float64, L + 1)
    for i in 1:m.n, j in (D ? (1:m.n) : (i+1:m.n))
        i == j && continue
        k = 0
        for l in 1:L
            has_edge(m.layers[l], i, j) && (k += 1)
        end
        counts[k + 1] += 1.0
    end
    return counts
end

_stack_counts(rows::Vector{Vector{Float64}}, width::Int) =
    Float64[rows[s][k] for s in eachindex(rows), k in 1:width]

# A distribution panel, trimmed to the largest level seen in the observed or
# any simulated network (at least level 0)
function _push_distribution!(panels, pname::String, obs::Vector{Float64},
                             sims::Vector{Vector{Float64}})
    top = something(findlast(!iszero, obs), 1)
    for s in sims
        top = max(top, something(findlast(!iszero, s), 1))
    end
    push!(panels, GOFStatistic(pname, string.(0:(top - 1)), obs[1:top],
                               _stack_counts(sims, top)))
    return panels
end

# =============================================================================
# Precompile workload
# =============================================================================
#
# The documented first session — build a two-layer network, fit, print, read
# the table and the intervals, simulate, assess fit, bootstrap — on a directed
# AND an undirected `MultilayerNetwork{D}` (two type parameters, two
# specializations of every kernel), so a user's first `ergm_multi` runs native
# code instead of compiling it. Measured in a fresh process (Julia 1.12.6,
# Linux x86-64; full table in the CHANGELOG): first `ergm_multi` 1.90 s →
# 0.010 s, first `coeftable`/`confint`/`show` 0.35 s → 0.003 s, first
# `simulate_multi_ergm` 0.22 s → 0.004 s, first `gof` 0.45 s → 0.027 s;
# `using ERGMMulti` itself stays at ~0.8 s (dependency loading). The tiny
# bootstrap and the 10-step chains are there for the code paths, not the
# numbers; they are silenced through a devnull console logger of the type a
# REPL has (caching the logging path too).
@setup_workload begin
    function _pc_multilayer(directed::Bool)
        m = MultilayerNetwork(6; directed=directed)
        add_layer!(m, :friendship)
        add_layer!(m, :advice)
        a_edges = directed ? ((1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (1, 5), (2, 1)) :
                             ((1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (2, 5))
        b_edges = directed ? ((1, 2), (2, 4), (3, 4), (5, 6), (6, 1), (3, 1)) :
                             ((1, 2), (2, 4), (3, 4), (5, 6), (1, 6))
        for (i, j) in a_edges
            add_layer_edge!(m, :friendship, i, j)
        end
        for (i, j) in b_edges
            add_layer_edge!(m, :advice, i, j)
        end
        grp = Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B", 5 => "A", 6 => "B")
        set_vertex_attribute!(m.layers[1], :grp, grp)
        set_vertex_attribute!(m.layers[2], :grp, grp)
        return m
    end
    _pc_terms = [LayerEdges(1), LayerEdges(2), WithinLayer(NodeMatch(:grp), 1),
                 InterlayerDependence(1, 2)]
    _pc_null = Base.CoreLogging.ConsoleLogger(devnull, Base.CoreLogging.Warn)
    @compile_workload begin
        Base.CoreLogging.with_logger(_pc_null) do
            for _pc_directed in (true, false)
                _pc_m = _pc_multilayer(_pc_directed)
                _pc_rng = Random.Xoshiro(20260912)
                # The MPLE of this dyad-dependent model (the default
                # `method=:auto` fits the MCMLE, below on a tiny budget)
                # withholds its naive Wald inference; `se=:hessian` is the opt-in
                _pc_fit = fit_ergm_multi(_pc_m, _pc_terms; method=:mple)
                coeftable(_pc_fit)
                confint(fit_ergm_multi(_pc_m, _pc_terms; method=:mple, se=:hessian))
                show(devnull, _pc_fit); sprint(show, _pc_fit)
                simulate_multi_ergm(_pc_m, _pc_terms, coef(_pc_fit);
                                    n_sim=1, burnin=10, interval=1, rng=_pc_rng)
                gof(_pc_fit; n_sim=2, burnin=10, interval=1, rng=_pc_rng)
                # The bootstrap path: a 3-replicate, 10-step chain. A replicate
                # simulated onto a boundary statistic is a NaN row (warned
                # into devnull); fewer than two finite refits is the one
                # ArgumentError the path can raise, and a workload must not
                # fail precompilation over the draws of a toy chain.
                try
                    fit_ergm_multi(_pc_m, _pc_terms; method=:mple, se=:bootstrap, n_boot=3,
                                   boot_burnin=10, boot_interval=1, rng=_pc_rng)
                catch e
                    e isa ArgumentError || rethrow()
                end
                # The MCMLE path on a tiny budget (chain with carried
                # statistics, Newton step, covariance, path sampling); a toy
                # network may sit on a boundary statistic, which is refused
                try
                    sprint(show, fit_ergm_multi(_pc_m, _pc_terms;
                                                n_samples=32, burnin=50, interval=5,
                                                mcmle_maxiter=2, bridge_rungs=2,
                                                bridge_samples=16, rng=_pc_rng))
                catch e
                    e isa ArgumentError || rethrow()
                end
            end
        end
    end
end

end # module
