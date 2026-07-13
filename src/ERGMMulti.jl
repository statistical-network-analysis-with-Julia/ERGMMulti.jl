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
using Networks
using Random
using Statistics

import ERGM: name, compute, change_stat, is_dyad_dependent, newton_fit,
              logistic_derivatives
import StatsAPI
import StatsAPI: coef, stderror, vcov, loglikelihood, aic, bic, nobs, dof

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

# Estimation
export ergm_multi, fit_multi_ergm, MultiERGMModel, MultiERGMResult

# Simulation
export simulate_multi_ergm

# Diagnostics (`gof` is Networks.jl's shared generic, extended with a method
# for MultiERGMResult)
export gof

# Utilities
export as_multilayer, combine_networks, split_by_layer

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using ERGMMulti`)
export coef, stderror, vcov, loglikelihood, aic, bic, nobs, dof

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

# Dependence classification (extends ERGM.is_dyad_dependent, whose fallback
# is the conservative `true`). In the multilayer model the "dyads" are
# (layer, i, j) triples: a term is dyad-dependent when its change statistic
# depends on the state of *any other* (layer, dyad) — so the cross-layer
# terms (`InterlayerDependence`, `MultiplexMutual`) and the within-layer
# structural terms (`LayerMutual`, `LayerTriangle`) are dyad-dependent
# (covered by the fallback), while pure edge-count terms are not.
is_dyad_dependent(::LayerEdges) = false
is_dyad_dependent(t::WithinLayer) = is_dyad_dependent(t.term)

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
with `NaN` standard error (and `NaN` rows/columns of `vcov`); `loglik` is
the maximized pseudo-log-likelihood over the within-layer dyads.

`se_type` records how `std_errors`/`vcov` were ACTUALLY obtained — `:hessian`
(the inverse negative pseudo-Hessian, anticonservative under dependence) or
`:bootstrap` (the parametric bootstrap of `ergm_multi(...; se=:bootstrap)`). It
is what `Networks.se_method(fit)` reports, and what `show` reads before deciding
whether the anticonservatism caveat still applies. Offset rows stay `NaN` under
either option: a fixed coefficient carries no uncertainty.
"""
struct MultiERGMResult
    model::MultiERGMModel
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    vcov::Matrix{Float64}
    loglik::Float64
    aic::Float64
    bic::Float64
    converged::Bool
    se_type::Symbol
end

# Backwards-compatible constructor: a result built without an `se_type` reports
# the inverse-Hessian standard errors it in fact had.
MultiERGMResult(model, coefficients, std_errors, vcov, loglik, aic, bic,
                converged) =
    MultiERGMResult(model, coefficients, std_errors, vcov, loglik, aic, bic,
                    converged, :hessian)

"""
    _has_dyad_dependent(model::MultiERGMModel) -> Bool

Whether any term of the multilayer formula is dyad-dependent, where a "dyad" is
a `(layer, i, j)` triple (see the dependence classification above). This is THE
predicate that decides whether the within-layer MPLE is an approximation: with
only dyad-independent terms the conditionals it multiplies are the model's own,
so the pseudo-likelihood is the likelihood. Defined once and used by both
`show(::MultiERGMResult)` (the prose caveat) and `is_exact(::MultiERGMResult)`
(the machine-readable answer), so the two cannot drift apart.
"""
_has_dyad_dependent(model::MultiERGMModel) =
    any(is_dyad_dependent(t) for t in model.terms)

function Base.show(io::IO, r::MultiERGMResult)
    println(io, "Multilayer ERGM Results")
    println(io, "=======================")
    println(io, "Layers: $(n_layers(r.model.network)); " *
                "pseudo-log-likelihood: $(round(r.loglik, digits=4))")
    println(io, "AIC: $(round(r.aic, digits=2)), BIC: $(round(r.bic, digits=2)); " *
                "converged: $(r.converged)")
    println(io, "Std. errors: ", r.se_type === :bootstrap ?
                "parametric bootstrap" : "inverse pseudo-Hessian")
    println(io)
    println(io, "Coefficients:")

    # Shared ecosystem presentation layer (Networks.print_coeftable):
    # Estimate / Std.Error / z value / Pr(>|z|) with significance codes.
    # Offset terms are tagged and print NaN standard errors / p-values
    # (their coefficients are fixed, not estimated).
    term_names = [name(term) * (haskey(r.model.offsets, k) ? " (offset)" : "")
                  for (k, term) in enumerate(r.model.terms)]
    z = r.coefficients ./ r.std_errors
    print_coeftable(io, term_names, r.coefficients, r.std_errors,
                    ERGM._z_pvalues(z); z_values=z)

    # Honest-uncertainty caveat (mirroring ERGM.jl's show): pseudo-likelihood
    # fits of dyad-dependent formulas have suspect inverse-Hessian standard
    # errors. Dyad-independent formulas need no caveat — there the
    # pseudo-likelihood is the likelihood. Neither does a bootstrap fit: a
    # parametric-bootstrap covariance does not treat the dyads as independent,
    # so calling it anticonservative would be a lie. (The POINT ESTIMATE is
    # still an MPLE either way, which is what the remaining note says.)
    if _has_dyad_dependent(r.model)
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
end

# ============================================================================
# The shared result-metadata protocol (Networks.jl `src/results.jl`)
# ============================================================================
#
# `fit_metadata(fit)` collects these accessors. They read the SAME
# `_has_dyad_dependent` predicate as the prose caveat in `show`, so the printed
# warning and the machine-readable answer cannot disagree.

estimand(::MultiERGMResult) = :multilayer_ergm

objective(::MultiERGMResult) = :pseudolikelihood

"""
    is_exact(r::MultiERGMResult) -> Bool

`true` iff every term is dyad-independent over the `(layer, i, j)` dyad universe
— there the within-layer pseudo-likelihood *is* the likelihood and the MPLE is
the exact MLE. Any cross-layer or within-layer structural term (`LayerMutual`,
`LayerTriangle`, `InterlayerDependence`, `MultiplexMutual`, ...) makes the same
estimator an approximation, and this reports `false`.
"""
is_exact(r::MultiERGMResult) = !_has_dyad_dependent(r.model)

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
    if _has_dyad_dependent(r.model)
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
    isempty(r.model.offsets) ||
        push!(out, "$(length(r.model.offsets)) offset term(s) held fixed, not " *
                   "estimated: their coefficients carry no uncertainty (NaN " *
                   "standard errors) and the reported dof excludes them")
    return out
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
               maxiter=100, tol=1e-8, se=:hessian, n_boot=100,
               boot_burnin=1000, boot_interval=100, rng=Random.default_rng())
        -> MultiERGMResult

Fit a multilayer ERGM by maximum pseudo-likelihood over the
**within-layer dyads** (the dyad universe of `ergm.multi`'s
block-diagonal construction): each dyad's conditional edge probability is
logistic in `θ'Δg`, with `Δg` the add-direction change statistics of the
layer-aware terms.

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
  `NaN` — a fixed coefficient carries no uncertainty. This is the same option,
  with the same keywords and the same semantics, as `ERGM.mple`'s, and it runs
  on the ONE shared `Networks.bootstrap_cov` loop.

# Keyword Arguments
- `se::Symbol=:hessian`: `:hessian` or `:bootstrap` (above)
- `n_boot::Int=100`: number of bootstrap replicates (`se=:bootstrap` only)
- `boot_burnin::Int=1000`, `boot_interval::Int=100`: MCMC controls for the
  bootstrap simulations (the [`simulate_multi_ergm`](@ref) defaults)
- `rng::AbstractRNG=Random.default_rng()`: source of the bootstrap randomness —
  a fixed `rng` reproduces the standard errors exactly
"""
function ergm_multi(m::MultilayerNetwork, terms::Vector{<:AbstractERGMTerm};
                    offsets::Dict{Int, Float64}=Dict{Int, Float64}(),
                    maxiter::Int=100, tol::Float64=1e-8,
                    se::Symbol=:hessian,
                    n_boot::Int=100,
                    boot_burnin::Int=1000,
                    boot_interval::Int=100,
                    rng::Random.AbstractRNG=Random.default_rng())
    se in (:hessian, :bootstrap) ||
        throw(ArgumentError("se must be :hessian or :bootstrap, got :$se"))
    n_layers(m) >= 1 || throw(ArgumentError("network has no layers"))
    # Multilayer MPLE enumerates every within-layer dyad as observed, so a
    # masked (unobserved) dyad in any layer would enter the pseudo-likelihood
    # at its face value. Reject rather than invent data.
    for (l, layer) in enumerate(m.layers)
        require_observed(layer; context="ergm_multi (layer $l)", face_ok=false)
    end
    p = length(terms)
    all(1 .<= collect(keys(offsets)) .<= p) ||
        throw(ArgumentError("offset indices must reference terms"))
    free = [k for k in 1:p if !haskey(offsets, k)]
    isempty(free) && throw(ArgumentError("all coefficients are offsets; nothing to estimate"))

    model = MultiERGMModel(collect(AbstractERGMTerm, terms), m, offsets)
    fit = _multi_mple_fit(m, model.terms, offsets, free; maxiter=maxiter, tol=tol)
    β = fit.θ
    ll = fit.loglik
    converged = fit.converged
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
    if se === :bootstrap
        vcov_free, se_free = _multi_bootstrap_cov(model, coefficients, β, free;
                                                  n_boot=n_boot,
                                                  boot_burnin=boot_burnin,
                                                  boot_interval=boot_interval,
                                                  maxiter=maxiter, tol=tol,
                                                  rng=rng)
    end

    std_errors = fill(NaN, p)
    vcov_full = fill(NaN, p, p)
    vcov_full[free, free] = vcov_free
    for (kf, k) in enumerate(free)
        std_errors[k] = se_free[kf]
    end

    n_dyads = _n_within_dyads(m)
    aic = -2 * ll + 2 * pf
    bic = -2 * ll + pf * log(n_dyads)

    return MultiERGMResult(model, coefficients, std_errors, vcov_full, ll,
                           aic, bic, converged, se)
end

# Parametric-bootstrap covariance of the multilayer MPLE: simulate `n_boot`
# multilayer networks at the fitted coefficients (offsets included — they are
# part of the data-generating model), refit `ergm_multi` on each with the SAME
# offsets, and take the empirical covariance of the free coefficients. The loop
# is the shared `Networks.bootstrap_cov`; this supplies only the two callbacks
# that are ERGMMulti's.
function _multi_bootstrap_cov(model::MultiERGMModel, coefficients::Vector{Float64},
                              β_free::Vector{Float64}, free::Vector{Int};
                              n_boot::Int, boot_burnin::Int, boot_interval::Int,
                              maxiter::Int, tol::Float64,
                              rng::Random.AbstractRNG)
    terms = model.terms
    offsets = model.offsets

    simulate(rng, B) = simulate_multi_ergm(model.network, terms, coefficients;
                                           n_sim=B, burnin=boot_burnin,
                                           interval=boot_interval, rng=rng)

    refit(sim::MultilayerNetwork) =
        _multi_mple_fit(sim, terms, offsets, free; maxiter=maxiter, tol=tol).θ

    boot = bootstrap_cov(refit, simulate, β_free; n_boot=n_boot, rng=rng)
    return boot.vcov, boot.se
end

# The MPLE design over the within-layer dyads: the FREE columns of the change
# statistics, the observed tie indicators, and the offset contribution to the
# linear predictor (`ergm.multi`'s offset mechanism — the fixed coefficients are
# dropped from the parameter vector but NOT from the linear predictor).
function _multi_mple_design(m::MultilayerNetwork, terms::Vector{AbstractERGMTerm},
                            offsets::Dict{Int, Float64}, free::Vector{Int})
    p = length(terms)
    dyads = _layer_dyads(m)
    n_dyads = length(dyads)

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
    return X[:, free], y, η0
end

# Core multilayer MPLE over the within-layer dyads: build the design, then
# maximize the pseudo-log-likelihood of the FREE coefficients with the shared
# `ERGM.newton_fit`. Shared by `ergm_multi` and by the parametric bootstrap's
# refits (which need only `.θ`).
#
# The derivatives come from the shared `ERGM.logistic_derivatives` (review
# finding 15): the pseudo-likelihood over the within-layer dyads IS a logistic
# likelihood with an offset, and its derivatives are gemv/gemm over the whole
# design — not a per-dyad `x * x'` outer product allocating a pf×pf matrix on
# every one of the n_dyads rows of every Newton evaluation. Never paste the loop
# back in; TERGM and ERGMRank run on the same one.
function _multi_mple_fit(m::MultilayerNetwork, terms::Vector{AbstractERGMTerm},
                         offsets::Dict{Int, Float64}, free::Vector{Int};
                         maxiter::Int=100, tol::Float64=1e-8)
    Xf, y, η0 = _multi_mple_design(m, terms, offsets, free)
    derivatives = logistic_derivatives(Xf, y; offset=η0)
    return newton_fit(derivatives, zeros(length(free)); maxiter=maxiter, tol=tol)
end

const fit_multi_ergm = ergm_multi

# StatsAPI interface: methods on the shared statistics generics (mirroring
# ERGM.jl), so results interoperate with StatsBase/GLM-style tooling

# Number of within-layer dyads — the model's dyad universe
function _n_within_dyads(m::MultilayerNetwork)
    per = m.directed ? m.n * (m.n - 1) : m.n * (m.n - 1) ÷ 2
    return n_layers(m) * per
end

StatsAPI.coef(r::MultiERGMResult) = r.coefficients
StatsAPI.stderror(r::MultiERGMResult) = r.std_errors
StatsAPI.vcov(r::MultiERGMResult) = r.vcov
StatsAPI.loglikelihood(r::MultiERGMResult) = r.loglik
StatsAPI.aic(r::MultiERGMResult) = r.aic
StatsAPI.bic(r::MultiERGMResult) = r.bic
StatsAPI.nobs(r::MultiERGMResult) = _n_within_dyads(r.model.network)
StatsAPI.dof(r::MultiERGMResult) =
    length(r.coefficients) - length(r.model.offsets)

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

# Attribute-preserving copy: delegates to `Base.copy(::Network)`, which
# duplicates the graph and all vertex/edge/network attributes, so attribute
# terms (e.g. `WithinLayer(NodeMatch(...), l)`) keep seeing covariates on
# the sampler's working copies.
_copy_net(net::Network) = copy(net)

# =============================================================================
# Goodness of fit
# =============================================================================

"""
    gof(result::MultiERGMResult; n_sim=100, burnin=1000, interval=100,
        rng=Random.default_rng()) -> GOFResult

Goodness-of-fit assessment for a fitted multilayer ERGM: simulate `n_sim`
multilayer networks at the fitted (and offset) coefficients with
[`simulate_multi_ergm`](@ref) and compare the observed data against the
simulated distributions on two panels:

- `"model statistics"` — the fitted terms' statistics (one level per term);
- `"layer edges"` — the edge count of each layer (one level per layer).

Extends Networks.jl's shared `gof` generic and returns the shared
`Networks.GOFResult` container; per-level p-values are two-sided Monte-Carlo
p-values computed with the `(1 + k)/(N + 1)` estimator (never exactly zero).
"""
function gof(result::MultiERGMResult; n_sim::Int=100, burnin::Int=1000,
             interval::Int=100,
             rng::Random.AbstractRNG=Random.default_rng())
    m = result.model.network
    terms = result.model.terms

    sims = simulate_multi_ergm(m, terms, result.coefficients; n_sim=n_sim,
                               burnin=burnin, interval=interval, rng=rng)

    # Panel 1: the model's own statistics, observed vs simulated
    obs_stats = [compute(t, m) for t in terms]
    sim_stats = [compute(t, s) for s in sims, t in terms]
    stats_panel = GOFStatistic("model statistics", name.(terms),
                               obs_stats, sim_stats)

    # Panel 2: per-layer edge counts
    L = n_layers(m)
    obs_edges = [Float64(ne(m.layers[l])) for l in 1:L]
    sim_edges = [Float64(ne(s.layers[l])) for s in sims, l in 1:L]
    edges_panel = GOFStatistic("layer edges", m.layer_names,
                               obs_edges, sim_edges)

    return GOFResult([stats_panel, edges_panel]; model="Multilayer ERGM")
end

end # module
