"""
    ERGMMulti.jl - ERGMs for Multiple and Multilayer Networks

Fits ERGMs to multilayer network data (the same actors observed on several
relations), following R `ergm.multi` (Krivitsky, Koehly & Marcum 2020):
the layers form a block-diagonal combined network with layer membership
attributes ([`combine_networks`](@ref)), the model's dyad universe is the
set of *within-layer* dyads, and layer-aware terms carry per-layer,
pooled, and cross-layer effects. Estimation ([`fit_ergm_multi`](@ref), the
R name [`ergm_multi`](@ref) being the same function) is by maximum
pseudo-likelihood over the within-layer dyads with support for per-layer
`offset` coefficients — there is no MCMLE (see "Not implemented" in the
README and the documentation index); simulation
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
using Networks
using PrecompileTools: @setup_workload, @compile_workload
using Random
using Statistics

# The ERGM.jl term protocol (shared statistic generics `name`/`compute` come
# from Networks.jl through ERGM), the dependence and direction traits, the
# tuple-backed `TermSet`, and the ONE Metropolis toggle kernel every
# ERGM-family sampler runs on (`mh_toggle!`, panel 2026-09 item 28).
import ERGM: name, compute, change_stat, is_dyad_dependent, has_dyad_dependent,
             requires_directed, requires_undirected, TermSet, mh_toggle!
# The shared numerics and presentation helpers live in Networks.jl (items
# 13/14/15/28): the Newton optimizer and the allocation-free logistic
# derivative builder, the one z → p helper, the one `se=` validator, and the
# generic coefficient table `coeftable` returns.
import Networks: newton_fit, logistic_derivatives, z_pvalues, check_se,
                 CoefficientTable, missing_policies
import StatsAPI
import StatsAPI: coef, stderror, vcov, confint, loglikelihood, aic, bic, nobs,
                 dof, coeftable

# `gof` extends the ONE shared Networks.jl generic (every model package adds
# methods for its own result types), so `gof(fit)` works uniformly across the
# ecosystem and loading several model packages never collides on the name.
import Networks: gof

# The shared result-metadata protocol (Networks.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# ERGMMulti adds methods for `MultiERGMResult`; `fit_metadata(fit)` collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
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

# Estimation: the harmonised `fit_<model>` name, the R name, and the
# deprecated pre-0.2 alias
export fit_ergm_multi, ergm_multi, fit_multi_ergm, MultiERGMModel, MultiERGMResult

# Simulation
export simulate_multi_ergm

# Diagnostics (`gof` is Networks.jl's shared generic, extended with a method
# for MultiERGMResult)
export gof

# Utilities
export as_multilayer, combine_networks, split_by_layer

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using ERGMMulti`)
export coef, stderror, vcov, confint, loglikelihood, aic, bic, nobs, dof, coeftable

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
using ERGMMulti, Networks
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
using ERGMMulti, Networks
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
using ERGMMulti, Networks
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
using ERGMMulti, Networks
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
# Adapters: Networks ↔ MultilayerNetwork ↔ the block-diagonal Network
# =============================================================================
#
# These three functions are conversions in the sense of the ecosystem's
# conversion contract (Networks.jl `src/conversion.jl`; per-path invariants in
# Networks.jl's `docs/src/guide/conversion_invariants.md`): preserve what the
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
using ERGMMulti, Networks
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
using ERGMMulti, Networks
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
using ERGMMulti, Networks
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
Networks.supports_missing(::typeof(as_multilayer)) = true
Networks.supports_missing(::typeof(combine_networks)) = true
Networks.supports_missing(::typeof(split_by_layer)) = true

"""
    MultiNetwork(networks::Vector{<:Network}, names::Vector{Symbol})

A collection of independent networks (possibly of different sizes and
directedness), for **descriptive pooling** with [`CrossNetEdges`](@ref).
It never enters a fit: `ergm.multi`'s covariate-driven `Networks()` models
(`N(~edges, ~x)`) are not implemented, and [`ergm_multi`](@ref) accepts a
[`MultilayerNetwork`](@ref) only.

# Example
```julia
using ERGMMulti, ERGM, Networks
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
using ERGMMulti, ERGM, Networks
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
using ERGMMulti, ERGM, Networks
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

_sel_string(sel::Colon) = "all"
_sel_string(sel::Int) = string(sel)
_sel_string(sel::Vector{Int}) = join(sel, "+")

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
layers. Dyad-independent; labelled `L.edges.<sel>`.

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
name(LayerEdges(2))                # "L.edges.2"
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

name(t::LayerEdges) = "L.edges.$(_sel_string(t.layers))"

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
`ergm.multi`'s `L(~mutual, ~A)`; labelled `L.mutual.<sel>`.

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

name(t::LayerMutual) = "L.mutual.$(_sel_string(t.layers))"
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
`L.triangle.<sel>`). Dyad-dependent, so a fit containing it carries the
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

name(t::LayerTriangle) = "L.triangle.$(_sel_string(t.layers))"

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

Lift any ERGM.jl term into a single layer: the statistic (and its
add-direction change statistic) of `term` evaluated on that layer's
network. This is the analogue of `ergm.multi`'s `Layer(~term)`.

The wrapped term is **validated against the layer** when a
[`MultiERGMModel`](@ref) is built (through the public
`ERGM._validate_formula`): a vertex attribute the layer lacks, or has on
only some vertices, an `EdgeCov` of the wrong size, and a direction
requirement the layer does not meet (`Kstar`, `GWDegree`, `Degree` are
undirected-only; `Mutual`, `OStar`, `IStar`, `GWODegree`, ... directed-only)
raise ERGM.jl's own actionable errors instead of a silent `0.0`. Attribute
terms are then materialized (`ERGM._materialize`), so their change
statistics read a dense snapshot rather than a `Dict` per toggle, and a term
that expands into several statistics (`NodeFactor(:g)` with several levels,
`Degree(0:2)`) contributes one coefficient per statistic, as in ERGM.jl.

The coefficient label is `"L<layer>." * name(term, layer_network)` — the
direction-aware ERGM.jl/R label, so a directed layer's `WithinLayer(GWESP(0.5), 1)`
is `L1.gwesp.OTP.fixed.0.5` and an undirected one `L1.gwesp.fixed.0.5`.

# Example
```julia
using ERGMMulti, ERGM, Networks
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :a, 2, 1)
compute(WithinLayer(Mutual(), 1), m)           # 1.0
name(WithinLayer(GWESP(0.5), 1), m)            # "L1.gwesp.OTP.fixed.0.5"
try
    MultiERGMModel([WithinLayer(Kstar(2), 1)], m)   # undirected-only term
catch e
    occursin("OStar(2)", e.msg)                # true — ERGM.jl's own hint
end
```
"""
struct WithinLayer{T <: AbstractERGMTerm} <: AbstractERGMTerm
    term::T
    layer::Int
end
WithinLayer(term::AbstractERGMTerm, layer::Symbol) =
    _symbol_selector_error("WithinLayer", layer; usage="WithinLayer(term, k)")

name(t::WithinLayer) = "L$(t.layer).$(name(t.term))"

"""
    name(t::WithinLayer, m::MultilayerNetwork) -> String

The coefficient label of a within-layer term on `m`: `"L<layer>."` followed
by the wrapped term's label *on that layer's network* (`name(term, net)`),
so it agrees with ERGM.jl and R on direction-dependent labels.
"""
function name(t::WithinLayer, m::MultilayerNetwork)
    _check_layer_index(t, t.layer, m)
    return "L$(t.layer)." * name(t.term, m.layers[t.layer])
end

requires_directed(t::WithinLayer) = requires_directed(t.term)
requires_undirected(t::WithinLayer) = requires_undirected(t.term)

compute(t::WithinLayer, m::MultilayerNetwork) =
    compute(t.term, m.layers[t.layer])

change_stat_layer(t::WithinLayer, m::MultilayerNetwork, l::Int, i::Int, j::Int) =
    l == t.layer ? change_stat(t.term, m.layers[l], i, j) : 0.0

"""
    InterlayerDependence(l1, l2) <: AbstractERGMTerm

Cross-layer co-occurrence: the number of dyads (i, j) with an edge in
both layers `l1` and `l2` (ordered dyads for directed networks). A
positive coefficient means a tie in one layer predicts the same tie in
the other. This is `ergm.multi`'s `L(~edges, ~A&B)` — the only layer-logic
conjunction implemented (see "Not implemented" in the README); labelled
`duplex.<l1>.<l2>`. Dyad-dependent over the `(layer, i, j)` universe.

# Example
```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(3; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2); add_layer_edge!(m, :friendship, 2, 3)
add_layer_edge!(m, :advice, 1, 2);     add_layer_edge!(m, :advice, 3, 2)
compute(InterlayerDependence(1, 2), m)                     # 1.0 — only 1→2 is in both
change_stat_layer(InterlayerDependence(1, 2), m, 1, 3, 2)  # 1.0 — 3→2 exists in :advice
name(InterlayerDependence(1, 2))                           # "duplex.1.2"
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

name(t::InterlayerDependence) = "duplex.$(t.l1).$(t.l2)"

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
`mutualL(Ls = list(~A, ~B))`; labelled `duplex.mutual.<l1>.<l2>`.

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

name(t::MultiplexMutual) = "duplex.mutual.$(t.l1).$(t.l2)"
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
using ERGMMulti, ERGM, Networks
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
using ERGMMulti, ERGM, Networks
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
using ERGMMulti, ERGM, Networks
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
using ERGMMulti, ERGM, Networks
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
    _check_layer_index(t, t.layer, m)
    ERGM._validate_formula(TermSet((t.term,)), m.layers[t.layer])
    return nothing
end

# Materialization: a within-layer attribute term takes a dense snapshot of
# its layer's attribute (`ERGM._materialize`); a term that expands into
# several statistics (multi-level `NodeFactor`, `Degree(0:2)`, multi-cell
# `NodeMix`) becomes one `WithinLayer` per statistic. Every other term
# passes through.
_materialize_multilayer(t::AbstractERGMTerm, m::MultilayerNetwork) = t
function _materialize_multilayer(t::WithinLayer, m::MultilayerNetwork)
    inner = ERGM._materialize(t.term, m.layers[t.layer])
    inner isa AbstractVector ? [WithinLayer(x, t.layer) for x in inner] :
                               WithinLayer(inner, t.layer)
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
  formula validator (`ERGM._validate_formula`): a missing or partial vertex
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
(`ERGM._materialize`), so the stored `terms` — an `ERGM.TermSet` whose
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
using ERGMMulti, ERGM, Networks
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
model = MultiERGMModel([LayerEdges(1), WithinLayer(GWESP(0.5), 2)], m)
model.terms.names          # ["L.edges.1", "L2.gwesp.OTP.fixed.0.5"]
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

# A self-loop is refused, as `ERGM.ERGMModel` does (`_refuse_self_loops`):
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
(the inverse negative pseudo-Hessian, anticonservative under dependence) or
`:bootstrap` (the parametric bootstrap of `ergm_multi(...; se=:bootstrap)`). It
is what `Networks.se_method(fit)` reports, and what `show` reads before deciding
whether the anticonservatism caveat still applies. Offset rows stay `NaN` under
either option: a fixed coefficient carries no uncertainty.

`boot_replicates` is the `n_boot × p_free` matrix of refitted free
coefficients under `se=:bootstrap` (a replicate whose refit did not converge
is a `NaN` row, excluded from the covariance), `nothing` otherwise.

`converged` is `false` when the Newton iteration did not meet its tolerance
**or** when the pseudo-likelihood has no finite maximum by perfect separation
(`separated == true`, R's "The MPLE does not exist!"): the coefficients are
then the point where Newton stopped on its flat asymptote and are unreliable.
A statistic at the boundary of its attainable range is handled differently —
its coefficient is fixed at `∓Inf` (see [`ergm_multi`](@ref)).

The full StatsAPI surface is implemented: `coef`, `stderror`, `vcov`,
`confint`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic` and `coeftable`
(a `Networks.CoefficientTable` — the table `show` prints). The shared
result-metadata accessors (`is_exact`, `se_method`, `approximations`, ...)
read the same fields.

# Example
```julia
using ERGMMulti, ERGM, Networks, Random
rng = Xoshiro(2)
m = MultilayerNetwork(12; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
for l in 1:2, i in 1:12, j in 1:12
    i != j && rand(rng) < 0.25 && add_layer_edge!(m, l, i, j)
end
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
fit isa MultiERGMResult{true}      # true
fit.converged, fit.separated       # (true, false)
se_method(fit)                     # :hessian
is_exact(fit)                      # true — every term is dyad-independent
fit.boot_replicates === nothing    # true — no bootstrap was run
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
end
# The one constructor is the default positional one over all eleven fields:
# `ergm_multi` is the only producer. (No "compatibility" constructors that
# default `se_type`/`boot_replicates`/`separated`: a result whose flags do
# not describe the numbers it carries must not be constructible by accident,
# and no released layout needs them — 0.1.0's `MultiERGMResult` was a
# different struct altogether.)

Graphs.is_directed(::MultiERGMResult{D}) where {D} = D

# Coefficient labels as printed: the model's direction-aware statistic names,
# offset rows tagged (their coefficients are fixed, not estimated)
_coef_names(r::MultiERGMResult) =
    [n * (haskey(r.model.offsets, k) ? " (offset)" : "")
     for (k, n) in enumerate(r.model.terms.names)]

# The one non-convergence sentence, printed by `show` right under the verdict
# and listed by `approximations`: an unconverged fit is never a fit with a
# footnote.
_nonconvergence_caveat(r::MultiERGMResult) =
    r.separated ? _separation_caveat(r) :
    "the Newton iteration on the within-layer pseudo-likelihood did not " *
    "converge: the maximum pseudo-likelihood estimate does not exist or was " *
    "not reached (a statistic at the boundary of its attainable range, perfect " *
    "separation, or too few iterations); point estimates and standard errors " *
    "are unreliable — simplify the formula, or raise `maxiter`"

# R's `mple.existence` verdict ("The MPLE does not exist!"): a combination of
# the model's statistics perfectly predicts the ties, so the pseudo-likelihood
# has no finite maximum and Newton stopped on its flat asymptote.
_separation_caveat(::MultiERGMResult) =
    "the MPLE does not exist (perfect separation): a combination of the " *
    "model's statistics perfectly predicts the within-layer ties, so the " *
    "pseudo-likelihood has no finite maximum and the coefficients are the " *
    "point where Newton stopped on its flat asymptote — arbitrarily large, " *
    "with meaningless standard errors (R ergm warns \"The MPLE does not " *
    "exist!\" for the same design); remove or coarsen a term, or collect " *
    "more ties"

# A coefficient fixed at ∓Inf by a boundary statistic (R's `drop`): said in
# `show` and in `approximations`, read off the coefficients themselves
function _fixed_coefficient_note(r::MultiERGMResult)
    names = r.model.terms.names
    lo = [names[k] for k in eachindex(names) if r.coefficients[k] == -Inf]
    hi = [names[k] for k in eachindex(names) if r.coefficients[k] == Inf]
    (isempty(lo) && isempty(hi)) && return nothing
    parts = String[]
    isempty(lo) || push!(parts, "$(join(lo, ", ")) fixed at -Inf (observed " *
                                "statistic at its smallest attainable value)")
    isempty(hi) || push!(parts, "$(join(hi, ", ")) fixed at +Inf (observed " *
                                "statistic at its largest attainable value)")
    return join(parts, "; ") * ": no finite maximum pseudo-likelihood estimate " *
           "exists (R ergm's drop=TRUE semantics; ergm.multi 0.3.0's layer " *
           "operator does not propagate the range and returns a finite ~-17 with " *
           "a huge standard error after warning \"The MPLE does not exist!\"); " *
           "its standard error and p-value are 0 by convention, and the remaining " *
           "coefficients were estimated on the dyads it does not touch"
end

# Bootstrap replicates excluded from the covariance (NaN rows), if any
function _boot_exclusion_note(r::MultiERGMResult)
    r.boot_replicates === nothing && return nothing
    n_boot = size(r.boot_replicates, 1)
    n_bad = count(b -> !all(isfinite, view(r.boot_replicates, b, :)), 1:n_boot)
    n_bad == 0 && return nothing
    return "$n_bad of the $n_boot bootstrap refits did not converge and were " *
           "excluded; the standard errors are the empirical covariance of the " *
           "remaining $(n_boot - n_bad) refits (`fit.boot_replicates` holds every " *
           "refit, the excluded ones as NaN rows)"
end

function Base.show(io::IO, r::MultiERGMResult)
    println(io, "Multilayer ERGM Results")
    println(io, "=======================")
    println(io, "Layers: $(n_layers(r.model.network)); " *
                "pseudo-log-likelihood: $(round(r.loglik, digits=4))")
    println(io, "AIC: $(round(r.aic, digits=2)), BIC: $(round(r.bic, digits=2)); " *
                "converged: $(r.converged)")
    if !r.converged
        println(io, "  Warning: ", _nonconvergence_caveat(r))
    end
    println(io, "Std. errors: ", r.se_type === :bootstrap ?
                "parametric bootstrap" : "inverse pseudo-Hessian")
    println(io)
    println(io, "Coefficients:")

    # Shared ecosystem presentation layer: the printed table IS
    # `coeftable(r)` (a Networks.CoefficientTable rendered through
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
    if has_dyad_dependent(r.model)
        println(io)
        if r.se_type === :bootstrap
            println(io, "Note: this model contains dyad-dependent terms and was fit by")
            println(io, "maximum pseudolikelihood (MPLE), so the point estimates are biased in")
            println(io, "finite samples. The standard errors are a parametric bootstrap and do")
            println(io, "not assume the dyad conditionals are independent.")
        else
            println(io, "Warning: this model contains dyad-dependent terms and was fit by")
            println(io, "maximum pseudolikelihood (MPLE). The standard errors are based on")
            println(io, "the naive pseudolikelihood and are suspect (typically")
            println(io, "anticonservative); treat the inference with caution, or refit with")
            println(io, "`se=:bootstrap` for a parametric-bootstrap covariance.")
        end
    end
    fixed = _fixed_coefficient_note(r)
    if fixed !== nothing
        println(io)
        println(io, "Note: ", fixed)
    end
    excluded = _boot_exclusion_note(r)
    if excluded !== nothing
        println(io)
        println(io, "Note: ", excluded)
    end
end

# ============================================================================
# The shared result-metadata protocol (Networks.jl `src/results.jl`)
# ============================================================================
#
# `fit_metadata(fit)` collects these accessors. They read the SAME
# `has_dyad_dependent` predicate as the prose caveat in `show`, so the printed
# warning and the machine-readable answer cannot disagree.

estimand(::MultiERGMResult) = :multilayer_ergm

objective(::MultiERGMResult) = :pseudolikelihood

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
pseudo-Hessian) or `:bootstrap` (the parametric bootstrap of
`ergm_multi(...; se=:bootstrap)`). Read straight off the fit, so it can never
claim an estimator that was not used.
"""
se_method(r::MultiERGMResult) = r.se_type

# `ergm_multi` calls `require_observed` on every layer with the default `:error`
# policy: the MPLE enumerates every within-layer dyad as observed, so a masked
# dyad would enter the pseudo-likelihood at its face value and is refused.
missing_method(::MultiERGMResult) = :rejected

function approximations(r::MultiERGMResult)
    out = String[]
    r.converged || push!(out, _nonconvergence_caveat(r))
    if has_dyad_dependent(r.model)
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
    end
    if r.se_type === :bootstrap
        push!(out, "standard errors are a parametric bootstrap of the multilayer MPLE " *
                   "(simulate within-layer networks at θ̂, refit, empirical " *
                   "covariance): they do not assume the (layer, i, j) conditionals " *
                   "are independent, but they are Monte-Carlo estimates and assume " *
                   "the fitted model generated the data")
    end
    fixed = _fixed_coefficient_note(r)
    fixed === nothing || push!(out, fixed)
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
               maxiter=100, tol=1e-8, se=:hessian, n_boot=100,
               boot_burnin=nothing, boot_interval=nothing,
               rng=Random.default_rng(), threaded=true) -> MultiERGMResult
    ergm_multi(model::MultiERGMModel; kwargs...) -> MultiERGMResult
    fit_ergm_multi(...)                            # the same function

Fit a multilayer ERGM by maximum pseudo-likelihood over the
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
mutual dyad under `LayerMutual`, a perfectly separated `NodeMatch`, ...) has
no finite MPLE: as R ergm's default `drop=TRUE`, its coefficient is fixed at
`∓Inf` with standard error 0 (R's warning is printed), the other
coefficients are estimated on the dyads it does not touch, `is_exact(fit)`
is `false`, and `show`/`approximations` say so. Perfect separation by a
*combination* of statistics (no single column at its boundary, yet the ties
are perfectly predicted — R's "The MPLE does not exist!") is detected by
ERGM.jl's asymptote test, with or without offsets: R's sentence is printed,
the fit is returned with `converged == false` and `separated == true`, and
`show`/`approximations` carry the verdict.

`offsets` maps term indices to **fixed** coefficients (per-layer offsets,
e.g. `Dict(1 => -log(n))` for a size adjustment); the remaining
coefficients are estimated with the offset contribution absorbed into the
linear predictor.

# Standard errors

- `se=:hessian` (default) — the inverse negative pseudo-Hessian. **Caution:**
  with any dyad-dependent term (`LayerMutual`, `LayerTriangle`,
  `InterlayerDependence`, `MultiplexMutual`, ...) the pseudo-likelihood
  multiplies the (layer, i, j) conditionals as if independent, so these
  standard errors are expected to be *anticonservative*. With only
  dyad-independent terms the pseudo-likelihood is the likelihood and they are
  correct.
- `se=:bootstrap` — parametric bootstrap: simulate `n_boot` multilayer networks
  from the fitted model at θ̂ (offsets included) with
  [`simulate_multi_ergm`](@ref), refit `ergm_multi` on each with the same
  offsets, and report the empirical covariance of the refits. The point
  estimates are unchanged; only the covariance is replaced. Offset rows stay
  `NaN` — a fixed coefficient carries no uncertainty. A replicate whose refit
  does not converge is excluded (warned once; `fit.boot_replicates` keeps it
  as a `NaN` row). This is the same option, with the same keywords and the
  same semantics, as `ERGM.mple`'s, and it runs on the ONE shared
  `Networks.bootstrap_cov` loop. It is refused (`ArgumentError`) when a
  coefficient is fixed at `∓Inf` by a boundary statistic: a network cannot
  be simulated at an infinite coefficient.

# Keyword Arguments
- `offsets::Dict{Int,Float64}`: fixed coefficients by term index
- `maxiter::Int=100`, `tol::Float64=1e-8`: Newton iteration controls
- `se::Symbol=:hessian`: `:hessian` or `:bootstrap` (above; anything else is
  refused by the shared `Networks.check_se`)
- `n_boot::Int=100`: number of bootstrap replicates (`se=:bootstrap` only)
- `boot_burnin`, `boot_interval`: MCMC controls for the bootstrap simulations;
  `nothing` (default) resolves to the dyad-scaled rule shared with every
  ERGM-family sampler (`ERGM._mcmc_defaults`: `20 × n_dyads` and
  `max(100, n_dyads ÷ 10)` over the within-layer dyads)
- `rng::AbstractRNG=Random.default_rng()`: source of the bootstrap randomness —
  a fixed `rng` reproduces the standard errors exactly
- `threaded::Bool=true`: run the bootstrap refits on all threads. The
  replicates are simulated from `rng` before any refit and each refit is
  deterministic, so the standard errors are identical whatever the thread
  count (pinned by a test)

# Example
```julia
using ERGMMulti, ERGM, Networks, Random
rng = Xoshiro(1)
m = MultilayerNetwork(12; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
for i in 1:12, j in 1:12
    i == j && continue
    rand(rng) < 0.2 && add_layer_edge!(m, :friendship, i, j)
    rand(rng) < 0.2 && add_layer_edge!(m, :advice, i, j)
end
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)])
fit.converged                    # true
coeftable(fit).names             # ["L.edges.1", "L.edges.2", "duplex.1.2"]
size(confint(fit))               # (3, 2)
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

function ergm_multi(model::MultiERGMModel;
                    maxiter::Int=100, tol::Float64=1e-8,
                    se::Symbol=:hessian,
                    n_boot::Int=100,
                    boot_burnin::Union{Nothing, Int}=nothing,
                    boot_interval::Union{Nothing, Int}=nothing,
                    rng::Random.AbstractRNG=Random.default_rng(),
                    threaded::Bool=true)
    check_se(se, (:hessian, :bootstrap); context="ergm_multi")
    m = model.network
    _require_observed_layers(m, "ergm_multi")
    terms = model.terms
    offsets = model.offsets
    p = length(terms)
    free = [k for k in 1:p if !haskey(offsets, k)]

    fit = _multi_mple_fit(m, terms, offsets, free; maxiter=maxiter, tol=tol)
    β = fit.θ
    ll = fit.loglik
    converged = fit.converged
    separated = fit.separated
    # A separated design has already printed R's own sentence (`_multi_mple_fit`)
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
        # the same refusal as `ERGM.mple`'s.
        if any(isinf, coefficients)
            fixed = [nm for (nm, c) in zip(terms.names, coefficients) if isinf(c)]
            throw(ArgumentError(
                "ergm_multi: se=:bootstrap is not available when a coefficient is " *
                "fixed at ±Inf by a statistic at the boundary of its attainable " *
                "range ($(join(fixed, ", "))): a multilayer network cannot be " *
                "simulated at an infinite coefficient. Remove the term (as R's " *
                "drop=TRUE does) or keep the default se=:hessian, which reports " *
                "standard error 0 for the fixed coefficient and the inverse-Hessian " *
                "errors of the rest."))
        end
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

    return MultiERGMResult(model, coefficients, std_errors, vcov_full, ll,
                           aic, bic, converged, se, replicates, separated)
end

_nonconvergence_caveat_short(maxiter::Int) =
    "the Newton iteration did not converge within maxiter = $maxiter " *
    "iterations (the maximum pseudo-likelihood estimate may not exist: a " *
    "statistic at the boundary of its attainable range, or perfect " *
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

"""
    fit_multi_ergm(args...; kwargs...)

**Deprecated** pre-0.2 alias of [`ergm_multi`](@ref): emits a deprecation
warning and forwards every argument to `ergm_multi`. Use
[`fit_ergm_multi`](@ref) (the ecosystem's `fit_<model>` name) or `ergm_multi`
(the R name).

# Example
```julia
using ERGMMulti
m = MultilayerNetwork(4; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
add_layer_edge!(m, :a, 1, 2); add_layer_edge!(m, :b, 2, 3)
fit = fit_multi_ergm(m, [LayerEdges()])      # warns: deprecated, use fit_ergm_multi
coef(fit) == coef(fit_ergm_multi(m, [LayerEdges()]))   # true
```
"""
function fit_multi_ergm(args...; kwargs...)
    Base.depwarn("fit_multi_ergm is deprecated, use fit_ergm_multi", :fit_multi_ergm)
    return ergm_multi(args...; kwargs...)
end

# The estimator offers no `missing=` keyword at all: the MPLE enumerates every
# within-layer dyad as an observed row, so the only policy is to refuse a
# masked network (the capability generator prints this tuple).
missing_policies(::typeof(ergm_multi)) = (:error,)

# Dyad-scaled sampler defaults over the within-layer dyad universe — the ONE
# rule of the ERGM family (`ERGM._mcmc_defaults`, panel item 24e), so the
# budgets cannot drift apart between packages.
function _resolve_multi_mcmc(m::MultilayerNetwork, burnin, interval)
    if burnin === nothing || interval === nothing
        d = ERGM._mcmc_defaults(_n_within_dyads(m))
        burnin = something(burnin, d.burnin)
        interval = something(interval, d.interval)
    end
    return Int(burnin), Int(interval)
end

# Parametric-bootstrap covariance of the multilayer MPLE: simulate `n_boot`
# multilayer networks at the fitted coefficients (offsets included — they are
# part of the data-generating model), refit `ergm_multi` on each with the SAME
# offsets, and take the empirical covariance of the free coefficients. The loop
# is the shared `Networks.bootstrap_cov`; this supplies only the two callbacks
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
          "empirical covariance of the $n_ok converged refits. This is about the " *
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

# Core multilayer MPLE over the within-layer dyads: build the design, then
# maximize the pseudo-log-likelihood of the FREE coefficients with the shared
# `Networks.newton_fit`. Shared by `ergm_multi` and by the parametric
# bootstrap's refits (which need only `.θ` and `.converged`).
#
# The derivatives come from the shared `Networks.logistic_derivatives` (review
# finding 15): the pseudo-likelihood over the within-layer dyads IS a logistic
# likelihood with an offset, and its derivatives are gemv/gemm over the whole
# design — not a per-dyad `x * x'` outer product allocating a pf×pf matrix on
# every one of the n_dyads rows of every Newton evaluation. Never paste the loop
# back in; ERGM, TERGM and ERGMRank run on the same one.
#
# A statistic at the boundary of its attainable range (ERGM.jl's
# `_boundary_columns_iterated`, R ergm's `ergm.checkextreme.model`) has no
# finite MPLE — Newton would otherwise "converge" on the flat asymptote at
# −20-something with a standard error in the tens of thousands. As R ergm does
# for one-mode terms under its default `drop=TRUE`, the coefficient is fixed
# at ∓Inf with standard error 0 and the other coefficients are the MPLE on
# the rows the dropped columns do not touch — the exact limit of the
# pseudo-likelihood. (ergm.multi 0.3.0 does NOT inherit the drop: its layer
# operator does not propagate the statistic's attainable range to
# `ergm.checkextreme.model`, so R warns "The MPLE does not exist!" and returns
# a finite ~−17 with a standard error in the thousands for the same design,
# while its finite coefficients agree with ours exactly — frozen as section
# (iii) of test/fixtures/twolayer_layer_terms.toml.) `_warn_multi_boundary`
# prints R ergm's sentence — ERGM.jl's `public` `_warn_boundary` — with THAT
# divergence in its closing parenthesis through the helper's `note=`
# (its default, "R ergm reports the same", is not true of ergm.multi).
# `warn=false` (the bootstrap's refits) silences the
# sentence: a boundary in a SIMULATED replicate is not a fact about the data,
# and `_multi_bootstrap_cov` reports those replicates once, in aggregate.
#
# A design on which the pseudo-likelihood has no finite maximum for another
# reason — complete or quasi-complete separation by a COMBINATION of columns,
# which no single-column boundary test can see — is caught by ERGM.jl's
# asymptote test (`_separated`, R's "The MPLE does not exist!") and returned
# with `converged = false, separated = true` instead of the point where Newton
# met its tolerance on the flat asymptote. `ERGM._separated` recomputes the
# linear predictor as `X * θ`, so `_multi_separated` hands it the offset
# contribution `η0` as one more column with a FIXED coefficient of 1 (and
# standard error 0): the linear predictor it inspects is then `Xf θ + η0`,
# the one the fit used, and the derivative closure it re-evaluates ignores
# that fixed entry. Offset and no-offset fits are tested alike.
#
# Returns `(θ, se, vcov, loglik, converged, separated, n_kept, boundary)`:
# `n_kept` is the number of dyads the finite coefficients were estimated on
# (the BIC sample size, as in ERGM.jl), `boundary` the `(free column,
# :min/:max)` pairs.
function _multi_separated(f, Xf::AbstractMatrix, y::AbstractVector,
                          η0::AbstractVector, θ::AbstractVector, se::AbstractVector)
    pf = length(θ)
    pf == 0 && return false
    n_tot = ones(length(y))
    n_one = Float64.(y)
    all(iszero, η0) && return ERGM._separated(f, Xf, n_tot, n_one, θ, se)
    # The offset as a fixed column: coefficient 1, standard error 0, and a
    # derivative closure over the free coefficients only.
    fa = θa -> f(view(θa, 1:pf))
    return ERGM._separated(fa, hcat(Xf, η0), n_tot, n_one, vcat(θ, 1.0), vcat(se, 0.0))
end

# R ergm's boundary sentence (the wording of ERGM.jl's `_warn_boundary`, which
# every one-mode ERGM package prints) — except for its closing parenthesis.
# ERGM.jl's says "R ergm reports the same", which is true of `ergm` and false
# of `ergm.multi` 0.3.0: its layer operator does not propagate the range, so
# for the same design R warns "The MPLE does not exist!" and returns a finite
# coefficient with a huge standard error (frozen as section (iii) of
# `twolayer_layer_terms.toml`). A migrant who has just seen R return a finite
# number must not be told R reports the same; the estimation guide quotes
# this sentence.
function _warn_multi_boundary(names::Vector{String},
                              boundary::Vector{Tuple{Int, Symbol}})
    ERGM._warn_boundary(names, boundary; context="ergm_multi",
                        note="R ergm's drop=TRUE; ergm.multi 0.3.0 warns \"The MPLE " *
                             "does not exist!\" and returns a finite value with a " *
                             "standard error in the thousands instead — see the " *
                             "estimation guide")
    return nothing
end

function _multi_mple_fit(m::MultilayerNetwork, terms::TermSet,
                         offsets::Dict{Int, Float64}, free::Vector{Int};
                         maxiter::Int=100, tol::Float64=1e-8, warn::Bool=true)
    Xf, y, η0 = _multi_mple_design(m, terms, offsets, free)
    pf = length(free)
    n_rows = length(y)
    boundary = ERGM._boundary_columns_iterated(Xf, ones(n_rows), Float64.(y))

    if isempty(boundary)
        derivatives = logistic_derivatives(Xf, y; offset=η0)
        fit = newton_fit(derivatives, zeros(pf); maxiter=maxiter, tol=tol)
        separated = _multi_separated(derivatives, Xf, y, η0, fit.θ, fit.se)
        separated && warn && ERGM._warn_separated("ergm_multi")
        return (θ=fit.θ, se=fit.se, vcov=fit.vcov, loglik=fit.loglik,
                converged=fit.converged && !separated, separated=separated,
                n_kept=n_rows, boundary=boundary)
    end

    warn && _warn_multi_boundary(terms.names[free], boundary)
    dropped = Dict(boundary)
    keep = [j for j in 1:pf if !haskey(dropped, j)]
    rows = [r for r in 1:n_rows if all(Xf[r, j] == 0 for j in keys(dropped))]

    θ = zeros(pf)
    se = zeros(pf)
    V = zeros(pf, pf)
    for (j, side) in boundary
        θ[j] = side === :min ? -Inf : Inf
    end
    ll = 0.0
    converged = true
    separated = false
    if !isempty(keep) && !isempty(rows)
        Xr, yr, ηr = Xf[rows, keep], y[rows], η0[rows]
        derivatives = logistic_derivatives(Xr, yr; offset=ηr)
        fit = newton_fit(derivatives, zeros(length(keep)); maxiter=maxiter, tol=tol)
        θ[keep] = fit.θ
        se[keep] = fit.se
        V[keep, keep] = fit.vcov
        ll = fit.loglik
        separated = _multi_separated(derivatives, Xr, yr, ηr, fit.θ, fit.se)
        separated && warn && ERGM._warn_separated("ergm_multi")
        converged = fit.converged && !separated
    elseif !isempty(keep)
        # Every row touches a dropped column: the kept coefficients are not
        # identified by any dyad
        se[keep] .= NaN
        V[keep, :] .= NaN
        V[:, keep] .= NaN
        converged = false
    else
        # Only the offsets remain on the untouched rows
        ll = sum(η0[r] * y[r] - log1p(exp(η0[r])) for r in rows; init=0.0)
    end
    return (θ=θ, se=se, vcov=V, loglik=ll, converged=converged,
            separated=separated, n_kept=length(rows), boundary=boundary)
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
(`is_exact(r)`).

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
reports (`se_method(r)`: inverse pseudo-Hessian or parametric bootstrap), so
for a dyad-dependent model under `se=:hessian` the intervals inherit the
anticonservative pseudo-likelihood SEs. Offset rows are `NaN`: a fixed
coefficient has no interval.

# Example
```julia
using ERGMMulti, ERGM, Networks, Random
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
    q = quantile(Normal(), 1 - (1 - level) / 2)
    θ, se = r.coefficients, r.std_errors
    return hcat(θ .- q .* se, θ .+ q .* se)
end

"""
    coeftable(r::MultiERGMResult) -> Networks.CoefficientTable

The R-style coefficient table (`Estimate`, `Std.Error`, `z value`,
`Pr(>|z|)`) as an inspectable `Networks.CoefficientTable` — exactly the table
`show(r)` prints, built from the same vectors (a method of
`StatsAPI.coeftable`). Rows are labelled with the model's direction-aware
statistic names (`r.model.terms.names`), offset rows tagged `" (offset)"`
with `NaN` standard error, z and p; the z → p map is the shared
`Networks.z_pvalues`.

# Example
```julia
using ERGMMulti, ERGM, Networks, Random
rng = Xoshiro(3)
m = MultilayerNetwork(10; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
for l in 1:2, i in 1:10, j in 1:10
    i != j && rand(rng) < 0.3 && add_layer_edge!(m, l, i, j)
end
fit = ergm_multi(m, [LayerEdges(1), LayerEdges(2)]; offsets=Dict(2 => -1.0))
tbl = coeftable(fit)
tbl.names                                   # ["L.edges.1", "L.edges.2 (offset)"]
tbl["L.edges.1"].estimate == coef(fit)[1]   # true
isnan(tbl[2].p_value)                       # true
```
"""
function StatsAPI.coeftable(r::MultiERGMResult)
    z = r.coefficients ./ r.std_errors
    return CoefficientTable(_coef_names(r), r.coefficients, r.std_errors;
                            z_values=z, p_values=z_pvalues(z))
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
every ERGM-family sampler (`ERGM._mcmc_defaults`): `20 × n_dyads` and
`max(100, n_dyads ÷ 10)` over the within-layer dyads.

A layer with a masked (missing) dyad is refused with an `ArgumentError`
naming the layer (`Networks.missing_policies(simulate_multi_ergm) ==
(:error,)`): the chain would otherwise toggle the unobserved dyad at its
face value. **Every coefficient must be finite**: a `∓Inf` (a boundary
statistic's R-drop value, `coef(fit)` of such a fit) or `NaN` entry is
refused with an `ArgumentError` naming it ("simulate_multi_ergm: every
coefficient must be finite (got duplex.1.2 = -Inf) …") — the chain would
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
    L = n_layers(m)
    n = m.n
    current = as_multilayer([_copy_net(net) for net in m.layers], m.layer_names)
    layers = current.layers
    draws = MultilayerNetwork{D}[]
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
        nothing
    end
    on_sample = k -> push!(draws, as_multilayer([_copy_net(net) for net in layers],
                                                m.layer_names))

    mh_toggle!(rng, θ, delta, propose, change!, apply!, on_sample;
               burnin=burnin, interval=interval, n_samples=n_sim)
    return draws
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
# Goodness of fit
# =============================================================================

"""
    gof(result::MultiERGMResult; n_sim=100, burnin=nothing, interval=nothing,
        rng=Random.default_rng()) -> GOFResult

Goodness-of-fit assessment for a fitted multilayer ERGM: simulate `n_sim`
multilayer networks at the fitted (and offset) coefficients with
[`simulate_multi_ergm`](@ref) and compare the observed data against the
simulated distributions on two panels:

- `"model statistics"` — the fitted terms' statistics (one level per
  statistic, labelled as in `coeftable(result)`);
- `"layer edges"` — the edge count of each layer (one level per layer).

`burnin`/`interval` default to the dyad-scaled rule of every ERGM-family
sampler (`ERGM._mcmc_defaults`). Extends Networks.jl's shared `gof` generic
and returns the shared `Networks.GOFResult` container; per-level p-values
are two-sided Monte-Carlo p-values computed with the `(1 + k)/(N + 1)`
estimator (never exactly zero). The fitted network is never masked (the
estimator refused it), so the simulation needs no `missing=` policy.

A fit with a coefficient fixed at `∓Inf` (a statistic at the boundary of
its attainable range, R's `drop`) is refused with an `ArgumentError` naming
the term ("gof: every coefficient must be finite (got duplex.1.2 = -Inf)
…"): nothing can be simulated at an infinite coefficient, and the frozen
chain would otherwise report `p = 1` for every statistic. Drop the term and
refit, as R's `drop=TRUE` does, then assess that model.

# Example
```julia
using ERGMMulti, ERGM, Networks, Random
rng = Xoshiro(4)
m = MultilayerNetwork(10; directed=true)
add_layer!(m, :a); add_layer!(m, :b)
for l in 1:2, i in 1:10, j in 1:10
    i != j && rand(rng) < 0.3 && add_layer_edge!(m, l, i, j)
end
fit = fit_ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
g = gof(fit; n_sim=20, burnin=2000, interval=50, rng=Xoshiro(5))
g isa GOFResult                            # true
[panel.name for panel in g.statistics]     # ["model statistics", "layer edges"]
g.statistics[2].labels                     # ["a", "b"] — one level per layer
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
    # A fit with a coefficient fixed at ∓Inf (R's drop) cannot be assessed by
    # simulation: say so here, naming the terms, rather than reporting the
    # frozen chain's p = 1 for every statistic
    _require_finite_coefficients(result.coefficients, terms.names, "gof")

    sims = simulate_multi_ergm(model, result.coefficients; n_sim=n_sim,
                               burnin=burnin, interval=interval, rng=rng)

    # Panel 1: the model's own statistics, observed vs simulated
    obs_stats = compute_all(terms, m)
    sim_stats = Matrix{Float64}(undef, length(sims), length(terms))
    for (s, sim) in enumerate(sims)
        sim_stats[s, :] .= compute_all(terms, sim)
    end
    stats_panel = GOFStatistic("model statistics", copy(terms.names),
                               obs_stats, sim_stats)

    # Panel 2: per-layer edge counts
    L = n_layers(m)
    obs_edges = [Float64(ne(m.layers[l])) for l in 1:L]
    sim_edges = [Float64(ne(s.layers[l])) for s in sims, l in 1:L]
    edges_panel = GOFStatistic("layer edges", m.layer_names,
                               obs_edges, sim_edges)

    return GOFResult([stats_panel, edges_panel]; model="Multilayer ERGM")
end

# =============================================================================
# Precompile workload (panel 2026-09, item 18)
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
                _pc_fit = fit_ergm_multi(_pc_m, _pc_terms)
                coeftable(_pc_fit); confint(_pc_fit)
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
                    fit_ergm_multi(_pc_m, _pc_terms; se=:bootstrap, n_boot=3,
                                   boot_burnin=10, boot_interval=1, rng=_pc_rng)
                catch e
                    e isa ArgumentError || rethrow()
                end
            end
        end
    end
end

end # module
