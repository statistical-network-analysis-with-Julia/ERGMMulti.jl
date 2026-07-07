"""
    ERGMMulti.jl - ERGMs for Multiple and Multilayer Networks

Fits ERGMs to multilayer network data (the same actors observed on several
relations), following R `ergm.multi` (Krivitsky, Koehly & Marcum 2020):
the layers form a block-diagonal combined network with layer membership
attributes ([`combine_networks`](@ref)), the model's dyad universe is the
set of *within-layer* dyads, and layer-aware terms carry per-layer,
pooled, and cross-layer effects. Estimation is by maximum
pseudo-likelihood over the within-layer dyads with support for per-layer
`offset` coefficients; simulation uses a Metropolis sampler restricted to
within-layer dyads.

Also provides containers for multiple independent networks
(`MultiNetwork`) and descriptive statistics for multilevel designs
(`MultilevelNetwork`).
"""
module ERGMMulti

using Distributions
using ERGM
using Graphs
using LinearAlgebra
using Network
using Random
using Statistics

import ERGM: name, compute, change_stat

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

# Estimation
export ergm_multi, fit_multi_ergm, MultiERGMModel, MultiERGMResult

# Simulation
export simulate_multi_ergm

# Utilities
export as_multilayer, combine_networks, split_by_layer

# =============================================================================
# Data Structures
# =============================================================================

"""
    MultilayerNetwork

Several relations ("layers") on the same actor set. All layers share the
number of actors and directedness.

# Fields
- `n::Int`: Number of actors
- `directed::Bool`
- `layers::Vector{Network{Int}}`: One network per layer
- `layer_names::Vector{Symbol}`
"""
struct MultilayerNetwork
    n::Int
    directed::Bool
    layers::Vector{Network{Int}}
    layer_names::Vector{Symbol}

    function MultilayerNetwork(n::Int; directed::Bool=true)
        new(n, directed, Network{Int}[], Symbol[])
    end
end

n_layers(m::MultilayerNetwork) = length(m.layers)
layer_names(m::MultilayerNetwork) = copy(m.layer_names)

function Base.show(io::IO, m::MultilayerNetwork)
    dir = m.directed ? "directed" : "undirected"
    print(io, "MultilayerNetwork: $(m.n) actors, $(n_layers(m)) $dir layer(s) ",
          "$(Tuple(m.layer_names))")
end

"""
    add_layer!(m::MultilayerNetwork, name::Symbol;
               net::Union{Network,Nothing}=nothing) -> MultilayerNetwork

Add a layer, either empty or from an existing `Network` with matching size
and directedness.
"""
function add_layer!(m::MultilayerNetwork, lname::Symbol;
                    net::Union{Network, Nothing}=nothing)
    lname in m.layer_names &&
        throw(ArgumentError("layer :$lname already exists"))
    if isnothing(net)
        net = network(m.n; directed=m.directed)
    else
        Int(nv(net)) == m.n ||
            throw(ArgumentError("layer network must have $(m.n) vertices"))
        is_directed(net) == m.directed ||
            throw(ArgumentError("layer directedness must match the multilayer network"))
    end
    push!(m.layers, net)
    push!(m.layer_names, lname)
    return m
end

"""
    add_layer_edge!(m::MultilayerNetwork, layer, i, j)

Add edge (i, j) in the given layer (index or name).
"""
function add_layer_edge!(m::MultilayerNetwork, layer, i::Int, j::Int)
    add_edge!(layer_network(m, layer), i, j)
    return m
end

"""
    layer_network(m::MultilayerNetwork, layer) -> Network

The network of a layer, by index or name.
"""
layer_network(m::MultilayerNetwork, l::Int) = m.layers[l]
function layer_network(m::MultilayerNetwork, lname::Symbol)
    idx = findfirst(==(lname), m.layer_names)
    isnothing(idx) && throw(ArgumentError("no layer named :$lname"))
    return m.layers[idx]
end

_layer_index(m::MultilayerNetwork, l::Int) = l
function _layer_index(m::MultilayerNetwork, lname::Symbol)
    idx = findfirst(==(lname), m.layer_names)
    isnothing(idx) && throw(ArgumentError("no layer named :$lname"))
    return idx
end

"""
    as_multilayer(nets::Vector{<:Network}, names::Vector{Symbol}) -> MultilayerNetwork

Build a multilayer network from same-sized networks.
"""
function as_multilayer(nets::Vector{<:Network}, lnames::Vector{Symbol})
    isempty(nets) && throw(ArgumentError("need at least one layer"))
    length(nets) == length(lnames) ||
        throw(ArgumentError("need one name per layer"))
    m = MultilayerNetwork(Int(nv(nets[1])); directed=is_directed(nets[1]))
    for (net, lname) in zip(nets, lnames)
        add_layer!(m, lname; net=net)
    end
    return m
end

"""
    combine_networks(m::MultilayerNetwork) -> Network

The **block-diagonal combined network** of `ergm.multi`'s `Layer()`
construct: one network with `n × L` vertices, where vertex
`(l-1)·n + a` is actor `a`'s copy in layer `l`. Each vertex carries the
`:layer` (layer index) and `:actor` (original actor ID) attributes, and
every layer's edges are placed within its own block. Cross-block dyads
are structurally empty — the model's dyad universe is the within-block
dyads only.
"""
function combine_networks(m::MultilayerNetwork)
    L = n_layers(m)
    L >= 1 || throw(ArgumentError("no layers to combine"))
    n = m.n
    combined = network(n * L; directed=m.directed)

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
        for e in edges(m.layers[l])
            add_edge!(combined, off + src(e), off + dst(e))
        end
    end

    return combined
end

"""
    split_by_layer(combined::Network, n::Int, L::Int;
                   names=Symbol[]) -> MultilayerNetwork

Inverse of [`combine_networks`](@ref): recover the layers from a
block-diagonal combined network.
"""
function split_by_layer(combined::Network, n::Int, L::Int;
                        names::Vector{Symbol}=Symbol[])
    Int(nv(combined)) == n * L ||
        throw(ArgumentError("combined network must have n × L vertices"))
    lnames = isempty(names) ? [Symbol("layer$(l)") for l in 1:L] : names
    m = MultilayerNetwork(n; directed=is_directed(combined))
    for l in 1:L
        add_layer!(m, lnames[l])
    end
    for e in edges(combined)
        i, j = Int(src(e)), Int(dst(e))
        li, lj = div(i - 1, n) + 1, div(j - 1, n) + 1
        li == lj ||
            throw(ArgumentError("combined network has a cross-block edge ($i, $j)"))
        add_layer_edge!(m, li, i - (li - 1) * n, j - (lj - 1) * n)
    end
    return m
end

"""
    MultiNetwork

A collection of independent networks (possibly different sizes), for
descriptive pooling.
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

"""
    add_cross_level_edge!(m::MultilevelNetwork, level_from, node_from,
                          level_to, node_to)

Record a cross-level tie (e.g. a person–organization affiliation beyond
the nesting structure).
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
own current state.
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

"""
    LayerEdges(layers=:) <: AbstractERGMTerm

Edge count within the selected layer(s). With a layer index this is the
per-layer edges term; with `:` (or a vector) it pools the coefficient
across layers, `ergm.multi`-style.
"""
struct LayerEdges <: AbstractERGMTerm
    layers::LayerSel
    LayerEdges(layers::LayerSel=Colon()) = new(layers)
end

name(t::LayerEdges) = "L.edges.$(_sel_string(t.layers))"

function compute(t::LayerEdges, m::MultilayerNetwork)
    return sum(Float64(ne(m.layers[l])) for l in 1:n_layers(m)
               if _in_layers(t.layers, l); init=0.0)
end

change_stat_layer(t::LayerEdges, m::MultilayerNetwork, l::Int, i::Int, j::Int) =
    _in_layers(t.layers, l) ? 1.0 : 0.0

"""
    LayerMutual(layers=:) <: AbstractERGMTerm

Mutual (reciprocated) dyads within the selected layer(s). Directed
multilayer networks only (0 otherwise).
"""
struct LayerMutual <: AbstractERGMTerm
    layers::LayerSel
    LayerMutual(layers::LayerSel=Colon()) = new(layers)
end

name(t::LayerMutual) = "L.mutual.$(_sel_string(t.layers))"

function compute(t::LayerMutual, m::MultilayerNetwork)
    m.directed || return 0.0
    total = 0.0
    for l in 1:n_layers(m)
        _in_layers(t.layers, l) || continue
        total += compute(Mutual(), m.layers[l])
    end
    return total
end

function change_stat_layer(t::LayerMutual, m::MultilayerNetwork, l::Int, i::Int, j::Int)
    (m.directed && _in_layers(t.layers, l)) || return 0.0
    return change_stat(Mutual(), m.layers[l], i, j)
end

"""
    LayerTriangle(layers=:) <: AbstractERGMTerm

Triangle count within the selected layer(s), using ERGM.jl's `Triangle`
statistic per layer.
"""
struct LayerTriangle <: AbstractERGMTerm
    layers::LayerSel
    LayerTriangle(layers::LayerSel=Colon()) = new(layers)
end

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
"""
struct WithinLayer <: AbstractERGMTerm
    term::AbstractERGMTerm
    layer::Int
end

name(t::WithinLayer) = "L$(t.layer).$(name(t.term))"

compute(t::WithinLayer, m::MultilayerNetwork) =
    compute(t.term, m.layers[t.layer])

change_stat_layer(t::WithinLayer, m::MultilayerNetwork, l::Int, i::Int, j::Int) =
    l == t.layer ? change_stat(t.term, m.layers[l], i, j) : 0.0

"""
    InterlayerDependence(l1, l2) <: AbstractERGMTerm

Cross-layer co-occurrence: the number of dyads (i, j) with an edge in
both layers `l1` and `l2` (ordered dyads for directed networks). A
positive coefficient means a tie in one layer predicts the same tie in
the other.
"""
struct InterlayerDependence <: AbstractERGMTerm
    l1::Int
    l2::Int

    function InterlayerDependence(l1::Int, l2::Int)
        l1 != l2 || throw(ArgumentError("layers must differ"))
        new(l1, l2)
    end
end

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
advice). Directed networks only.
"""
struct MultiplexMutual <: AbstractERGMTerm
    l1::Int
    l2::Int

    function MultiplexMutual(l1::Int, l2::Int)
        l1 != l2 || throw(ArgumentError("layers must differ; use LayerMutual within a layer"))
        new(l1, l2)
    end
end

name(t::MultiplexMutual) = "duplex.mutual.$(t.l1).$(t.l2)"

function compute(t::MultiplexMutual, m::MultilayerNetwork)
    m.directed || return 0.0
    a, b = m.layers[t.l1], m.layers[t.l2]
    total = 0.0
    for e in edges(a)
        has_edge(b, dst(e), src(e)) && (total += 1.0)
    end
    return total
end

function change_stat_layer(t::MultiplexMutual, m::MultilayerNetwork,
                           l::Int, i::Int, j::Int)
    m.directed || return 0.0
    if l == t.l1
        return has_edge(m.layers[t.l2], j, i) ? 1.0 : 0.0
    elseif l == t.l2
        return has_edge(m.layers[t.l1], j, i) ? 1.0 : 0.0
    end
    return 0.0
end

"""
    CrossNetEdges <: AbstractERGMTerm

Total edge count across the networks of a `MultiNetwork` (descriptive
pooling over independent networks).
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
the same higher-level unit.
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
[`add_cross_level_edge!`](@ref)).
"""
struct CrossLevelEdge <: AbstractERGMTerm end

name(::CrossLevelEdge) = "crosslevel.edges"

compute(::CrossLevelEdge, m::MultilevelNetwork) =
    Float64(length(m.cross_level_edges))

"""
    LevelHomophily(level) <: AbstractERGMTerm

Descriptive: the number of level-`level` edges between nodes belonging to
the same higher-level unit. Membership is looked up by the level's own
node IDs (a missing membership excludes the edge).
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
# Estimation
# =============================================================================

"""
    MultiERGMModel

Multilayer ERGM specification: terms, data, and any per-term offsets.
"""
struct MultiERGMModel
    terms::Vector{AbstractERGMTerm}
    network::MultilayerNetwork
    offsets::Dict{Int, Float64}
end

"""
    MultiERGMResult

Results from `ergm_multi`. Offset terms report their fixed coefficient
with `NaN` standard error; `loglik` is the maximized pseudo-log-likelihood
over the within-layer dyads.
"""
struct MultiERGMResult
    model::MultiERGMModel
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    loglik::Float64
    aic::Float64
    bic::Float64
    converged::Bool
end

function Base.show(io::IO, r::MultiERGMResult)
    println(io, "Multilayer ERGM Results")
    println(io, "=======================")
    println(io, "Layers: $(n_layers(r.model.network)); " *
                "pseudo-log-likelihood: $(round(r.loglik, digits=4))")
    println(io, "AIC: $(round(r.aic, digits=2)), BIC: $(round(r.bic, digits=2)); " *
                "converged: $(r.converged)")
    println(io)
    println(io, "Coefficients:")
    for (k, term) in enumerate(r.model.terms)
        tag = haskey(r.model.offsets, k) ? " (offset)" : ""
        se = isnan(r.std_errors[k]) ? "--" : string(round(r.std_errors[k], digits=4))
        println(io, "  $(rpad(name(term) * tag, 28)) " *
                    "$(lpad(round(r.coefficients[k], digits=4), 10)) (SE: $se)")
    end
end

# All within-layer dyads of the multilayer network as (l, i, j)
function _layer_dyads(m::MultilayerNetwork)
    dyads = NTuple{3, Int}[]
    for l in 1:n_layers(m), i in 1:m.n
        for j in (m.directed ? (1:m.n) : (i+1:m.n))
            i == j && continue
            push!(dyads, (l, i, j))
        end
    end
    return dyads
end

"""
    ergm_multi(m::MultilayerNetwork, terms; offsets=Dict{Int,Float64}(),
               maxiter=100, tol=1e-8) -> MultiERGMResult

Fit a multilayer ERGM by maximum pseudo-likelihood over the
**within-layer dyads** (the dyad universe of `ergm.multi`'s
block-diagonal construction): each dyad's conditional edge probability is
logistic in `θ'Δg`, with `Δg` the add-direction change statistics of the
layer-aware terms.

`offsets` maps term indices to **fixed** coefficients (per-layer offsets,
e.g. `Dict(1 => -log(n))` for a size adjustment); the remaining
coefficients are estimated with the offset contribution absorbed into the
linear predictor.
"""
function ergm_multi(m::MultilayerNetwork, terms::Vector{<:AbstractERGMTerm};
                    offsets::Dict{Int, Float64}=Dict{Int, Float64}(),
                    maxiter::Int=100, tol::Float64=1e-8)
    n_layers(m) >= 1 || throw(ArgumentError("network has no layers"))
    p = length(terms)
    all(1 .<= collect(keys(offsets)) .<= p) ||
        throw(ArgumentError("offset indices must reference terms"))
    free = [k for k in 1:p if !haskey(offsets, k)]
    isempty(free) && throw(ArgumentError("all coefficients are offsets; nothing to estimate"))

    dyads = _layer_dyads(m)
    n_dyads = length(dyads)

    # Design matrix over within-layer dyads and offset contribution
    X = Matrix{Float64}(undef, n_dyads, p)
    y = Vector{Bool}(undef, n_dyads)
    for (r, (l, i, j)) in enumerate(dyads)
        for (k, t) in enumerate(terms)
            X[r, k] = change_stat_layer(t, m, l, i, j)
        end
        y[r] = has_edge(m.layers[l], i, j)
    end
    η0 = [sum(X[r, k] * offsets[k] for k in keys(offsets); init=0.0)
          for r in 1:n_dyads]
    Xf = X[:, free]
    pf = length(free)

    function derivatives(β)
        ll = 0.0
        grad = zeros(pf)
        hess = zeros(pf, pf)
        for r in 1:n_dyads
            η = η0[r] + dot(β, @view Xf[r, :])
            pr = 1.0 / (1.0 + exp(-η))
            ll += y[r] ? (η < 0 ? η - log1p(exp(η)) : -log1p(exp(-η))) :
                         (η < 0 ? -log1p(exp(η)) : -η - log1p(exp(-η)))
            resid = (y[r] ? 1.0 : 0.0) - pr
            x = @view Xf[r, :]
            grad .+= resid .* x
            hess .-= (pr * (1 - pr)) .* (x * x')
        end
        return ll, grad, hess
    end

    β = zeros(pf)
    ll, grad, hess = derivatives(β)
    converged = false

    for _ in 1:maxiter
        step = try
            -hess \ grad
        catch
            break
        end

        stepsize = 1.0
        ll_new, grad_new, hess_new = ll, grad, hess
        for _ in 1:10
            ll_new, grad_new, hess_new = derivatives(β .+ stepsize .* step)
            ll_new >= ll && break
            stepsize /= 2
        end

        β .+= stepsize .* step
        ll_change = abs(ll_new - ll)
        ll, grad, hess = ll_new, grad_new, hess_new

        if ll_change < tol && norm(grad) < sqrt(tol)
            converged = true
            break
        end
    end

    se_free = try
        sqrt.(abs.(diag(pinv(-hess))))
    catch
        fill(NaN, pf)
    end

    coefficients = zeros(p)
    std_errors = fill(NaN, p)
    for (kf, k) in enumerate(free)
        coefficients[k] = β[kf]
        std_errors[k] = se_free[kf]
    end
    for (k, c) in offsets
        coefficients[k] = c
    end

    aic = -2 * ll + 2 * pf
    bic = -2 * ll + pf * log(n_dyads)

    model = MultiERGMModel(collect(AbstractERGMTerm, terms), m, offsets)
    return MultiERGMResult(model, coefficients, std_errors, ll, aic, bic, converged)
end

const fit_multi_ergm = ergm_multi

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate_multi_ergm(m, terms, θ; n_sim=1, burnin=..., interval=...,
                        rng=Random.default_rng()) -> Vector{MultilayerNetwork}

Simulate multilayer networks from `P(y) ∝ exp(θ'g(y))` with a Metropolis
sampler restricted to **within-layer dyads** (matching the dyad universe
of the block-diagonal model): propose toggling a random (layer, i, j),
accept an addition with `min(1, exp(θ'Δg))` and a removal with
`min(1, exp(−θ'Δg))`.
"""
function simulate_multi_ergm(m::MultilayerNetwork,
                             terms::Vector{<:AbstractERGMTerm},
                             θ::Vector{Float64};
                             n_sim::Int=1,
                             burnin::Int=1000,
                             interval::Int=100,
                             rng::Random.AbstractRNG=Random.default_rng())
    length(θ) == length(terms) ||
        throw(ArgumentError("θ must have one coefficient per term"))
    L = n_layers(m)
    L >= 1 || throw(ArgumentError("network has no layers"))

    current = as_multilayer([_copy_net(net) for net in m.layers], m.layer_names)
    draws = MultilayerNetwork[]
    p = length(terms)
    delta = Vector{Float64}(undef, p)

    for step in 1:(burnin + n_sim * interval)
        l = rand(rng, 1:L)
        i = rand(rng, 1:m.n)
        j = rand(rng, 1:(m.n - 1))
        j >= i && (j += 1)
        (!m.directed && j < i) && ((i, j) = (j, i))

        for (k, t) in enumerate(terms)
            delta[k] = change_stat_layer(t, current, l, i, j)
        end
        log_accept = dot(θ, delta)
        net_l = current.layers[l]
        if has_edge(net_l, i, j)
            log_accept = -log_accept
        end

        if log(rand(rng)) < log_accept
            if has_edge(net_l, i, j)
                rem_edge!(net_l, i, j)
            else
                add_edge!(net_l, i, j)
            end
        end

        if step > burnin && (step - burnin) % interval == 0
            push!(draws, as_multilayer([_copy_net(net) for net in current.layers],
                                       m.layer_names))
        end
    end

    return draws
end

function _copy_net(net::Network{T}) where T
    c = Network{T}(; n=Int(nv(net)), directed=is_directed(net))
    for e in edges(net)
        add_edge!(c, src(e), dst(e))
    end
    return c
end

end # module
