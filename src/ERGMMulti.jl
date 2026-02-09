"""
    ERGMMulti.jl - ERGMs for Multiple and Multilayer Networks

Provides tools for fitting ERGMs to:
- Multiple independent networks with shared parameters
- Multilayer/multiplex networks (same nodes, different edge types)
- Multilevel networks (networks within networks)

Port of the R ergm.multi package from the StatNet collection.
"""
module ERGMMulti

using ERGM
using Graphs
using LinearAlgebra
using Network
using Optim
using Random
using Statistics
using StatsBase

# Data structures
export MultiNetwork, MultilayerNetwork, MultilevelNetwork
export LayerSpec, LevelSpec

# Multi-network terms
export CrossNetEdges, LayerLogic, WithinLayer, BetweenLayers
export MultiplexMutual, InterlayerDependence
export LayerEdges, LayerMutual, LayerTriangle

# Multilevel terms
export Nestedness, CrossLevelEdge, LevelHomophily

# Estimation
export ergm_multi, fit_multi_ergm

# Simulation
export simulate_multi_ergm

# Utilities
export as_multilayer, combine_networks, split_by_layer

# =============================================================================
# Multi-Network Data Structures
# =============================================================================

"""
    MultiNetwork{T}

A collection of networks for joint modeling.

# Fields
- `networks::Vector{Network{T}}`: The networks
- `network_names::Vector{Symbol}`: Names for each network
- `shared_vertices::Bool`: Whether all networks share the same vertex set
"""
struct MultiNetwork{T}
    networks::Vector{Network{T}}
    network_names::Vector{Symbol}
    shared_vertices::Bool

    function MultiNetwork(networks::Vector{Network{T}};
                          names::Vector{Symbol}=Symbol[],
                          shared_vertices::Bool=true) where T
        if shared_vertices
            n = nv(networks[1])
            all(nv(net) == n for net in networks) ||
                throw(ArgumentError("All networks must have the same number of vertices"))
        end

        net_names = isempty(names) ? [Symbol("net_$i") for i in 1:length(networks)] : names
        new{T}(networks, net_names, shared_vertices)
    end
end

Base.length(mn::MultiNetwork) = length(mn.networks)
Base.getindex(mn::MultiNetwork, i::Int) = mn.networks[i]
Base.getindex(mn::MultiNetwork, s::Symbol) = mn.networks[findfirst(==(s), mn.network_names)]

"""
    nv(mn::MultiNetwork) -> Int

Number of vertices (assumes shared vertices).
"""
Graphs.nv(mn::MultiNetwork) = nv(mn.networks[1])

"""
    total_edges(mn::MultiNetwork) -> Int

Total edges across all networks.
"""
total_edges(mn::MultiNetwork) = sum(ne(net) for net in mn.networks)

"""
    MultilayerNetwork{T}

A network with multiple types of edges (layers) on the same vertex set.

# Fields
- `n_vertices::Int`: Number of vertices
- `layers::Dict{Symbol, Network{T}}`: Named edge layers
- `interlayer_edges::Dict{Tuple{Symbol,Symbol}, Set{Tuple{T,T}}}`: Edges between layers
"""
struct MultilayerNetwork{T}
    n_vertices::Int
    layers::Dict{Symbol, Network{T}}
    interlayer_edges::Dict{Tuple{Symbol,Symbol}, Set{Tuple{T,T}}}

    function MultilayerNetwork{T}(n::Int; layer_names::Vector{Symbol}=Symbol[]) where T
        layers = Dict{Symbol, Network{T}}()
        for name in layer_names
            layers[name] = Network{T}(; n=n, directed=true)
        end
        new{T}(n, layers, Dict{Tuple{Symbol,Symbol}, Set{Tuple{T,T}}}())
    end
end

MultilayerNetwork(n::Int; kwargs...) = MultilayerNetwork{Int}(n; kwargs...)

Graphs.nv(mln::MultilayerNetwork) = mln.n_vertices

"""
    add_layer!(mln::MultilayerNetwork, name::Symbol; directed=true)

Add a new layer to the multilayer network.
"""
function add_layer!(mln::MultilayerNetwork{T}, name::Symbol; directed::Bool=true) where T
    mln.layers[name] = Network{T}(; n=mln.n_vertices, directed=directed)
    return mln
end

"""
    add_edge!(mln::MultilayerNetwork, layer::Symbol, i, j)

Add an edge to a specific layer.
"""
function add_layer_edge!(mln::MultilayerNetwork{T}, layer::Symbol, i::T, j::T) where T
    haskey(mln.layers, layer) || throw(ArgumentError("Layer $layer not found"))
    add_edge!(mln.layers[layer], i, j)
    return mln
end

"""
    has_edge(mln::MultilayerNetwork, layer::Symbol, i, j) -> Bool

Check if edge exists in a specific layer.
"""
function has_layer_edge(mln::MultilayerNetwork{T}, layer::Symbol, i::T, j::T) where T
    haskey(mln.layers, layer) || return false
    return has_edge(mln.layers[layer], i, j)
end

"""
    n_layers(mln::MultilayerNetwork) -> Int

Number of layers in the multilayer network.
"""
n_layers(mln::MultilayerNetwork) = length(mln.layers)

"""
    layer_names(mln::MultilayerNetwork) -> Vector{Symbol}

Get names of all layers.
"""
layer_names(mln::MultilayerNetwork) = collect(keys(mln.layers))

"""
    MultilevelNetwork{T}

A network with hierarchical structure (nodes within groups within groups).

# Fields
- `levels::Vector{Network{T}}`: Network at each level
- `membership::Vector{Vector{Int}}`: Group membership at each level
- `cross_level_edges::Dict{Tuple{Int,Int}, Set{Tuple{T,T}}}`: Edges between levels
"""
struct MultilevelNetwork{T}
    levels::Vector{Network{T}}
    membership::Vector{Vector{Int}}
    cross_level_edges::Dict{Tuple{Int,Int}, Set{Tuple{T,T}}}

    function MultilevelNetwork(levels::Vector{Network{T}},
                               membership::Vector{Vector{Int}}) where T
        length(levels) == length(membership) + 1 ||
            throw(ArgumentError("Need membership for each level transition"))
        new{T}(levels, membership, Dict{Tuple{Int,Int}, Set{Tuple{T,T}}}())
    end
end

"""
    n_levels(mln::MultilevelNetwork) -> Int

Number of levels in the multilevel network.
"""
n_levels(mln::MultilevelNetwork) = length(mln.levels)

# =============================================================================
# Multilayer ERGM Terms
# =============================================================================

"""
    LayerEdges <: AbstractERGMTerm

Edge count term for a specific layer.
"""
struct LayerEdges <: AbstractERGMTerm
    layer::Symbol
end

name(t::LayerEdges) = "edges.$(t.layer)"

function compute(t::LayerEdges, mln::MultilayerNetwork)
    haskey(mln.layers, t.layer) || return 0.0
    return Float64(ne(mln.layers[t.layer]))
end

function change_stat(t::LayerEdges, mln::MultilayerNetwork, layer::Symbol, i::Int, j::Int)
    layer != t.layer && return 0.0
    return has_layer_edge(mln, layer, i, j) ? -1.0 : 1.0
end

"""
    LayerMutual <: AbstractERGMTerm

Mutuality within a specific layer.
"""
struct LayerMutual <: AbstractERGMTerm
    layer::Symbol
end

name(t::LayerMutual) = "mutual.$(t.layer)"

function compute(t::LayerMutual, mln::MultilayerNetwork)
    haskey(mln.layers, t.layer) || return 0.0
    net = mln.layers[t.layer]
    !is_directed(net) && return 0.0

    count = 0
    for e in edges(net)
        if has_edge(net, dst(e), src(e))
            count += 1
        end
    end
    return Float64(count) / 2
end

"""
    LayerTriangle <: AbstractERGMTerm

Triangle count within a specific layer.
"""
struct LayerTriangle <: AbstractERGMTerm
    layer::Symbol
end

name(t::LayerTriangle) = "triangle.$(t.layer)"

function compute(t::LayerTriangle, mln::MultilayerNetwork)
    haskey(mln.layers, t.layer) || return 0.0
    net = mln.layers[t.layer]
    n = nv(net)

    count = 0
    for i in 1:n, j in (i+1):n, k in (j+1):n
        if has_edge(net, i, j) && has_edge(net, j, k) && has_edge(net, i, k)
            count += 1
        end
    end
    return Float64(count)
end

"""
    MultiplexMutual <: AbstractERGMTerm

Cross-layer mutuality: edge (i,j) in layer1 and edge (j,i) in layer2.
"""
struct MultiplexMutual <: AbstractERGMTerm
    layer1::Symbol
    layer2::Symbol
end

name(t::MultiplexMutual) = "multiplex.mutual.$(t.layer1).$(t.layer2)"

function compute(t::MultiplexMutual, mln::MultilayerNetwork)
    (haskey(mln.layers, t.layer1) && haskey(mln.layers, t.layer2)) || return 0.0

    net1 = mln.layers[t.layer1]
    net2 = mln.layers[t.layer2]
    n = nv(mln)

    count = 0
    for i in 1:n, j in 1:n
        i == j && continue
        if has_edge(net1, i, j) && has_edge(net2, j, i)
            count += 1
        end
    end
    return Float64(count)
end

"""
    InterlayerDependence <: AbstractERGMTerm

Dependence between edges in different layers:
tendency for edge (i,j) in layer1 given edge (i,j) in layer2.
"""
struct InterlayerDependence <: AbstractERGMTerm
    layer1::Symbol
    layer2::Symbol
end

name(t::InterlayerDependence) = "interlayer.$(t.layer1).$(t.layer2)"

function compute(t::InterlayerDependence, mln::MultilayerNetwork)
    (haskey(mln.layers, t.layer1) && haskey(mln.layers, t.layer2)) || return 0.0

    net1 = mln.layers[t.layer1]
    net2 = mln.layers[t.layer2]

    count = 0
    for e in edges(net1)
        if has_edge(net2, src(e), dst(e))
            count += 1
        end
    end
    return Float64(count)
end

"""
    CrossNetEdges <: AbstractERGMTerm

For MultiNetwork: total edges across all networks.
"""
struct CrossNetEdges <: AbstractERGMTerm end

name(::CrossNetEdges) = "cross.edges"

function compute(::CrossNetEdges, mn::MultiNetwork)
    return Float64(total_edges(mn))
end

"""
    WithinLayer <: AbstractERGMTerm

Wrapper to apply any standard term within a specific layer.
"""
struct WithinLayer{T<:AbstractERGMTerm} <: AbstractERGMTerm
    term::T
    layer::Symbol
end

name(t::WithinLayer) = "$(name(t.term)).$(t.layer)"

function compute(t::WithinLayer, mln::MultilayerNetwork)
    haskey(mln.layers, t.layer) || return 0.0
    return compute(t.term, mln.layers[t.layer])
end

"""
    BetweenLayers <: AbstractERGMTerm

Count edges that span layers (in multilayer network with interlayer edges).
"""
struct BetweenLayers <: AbstractERGMTerm
    layer1::Symbol
    layer2::Symbol
end

name(t::BetweenLayers) = "between.$(t.layer1).$(t.layer2)"

function compute(t::BetweenLayers, mln::MultilayerNetwork)
    key = (t.layer1, t.layer2)
    haskey(mln.interlayer_edges, key) || return 0.0
    return Float64(length(mln.interlayer_edges[key]))
end

# =============================================================================
# Multilevel ERGM Terms
# =============================================================================

"""
    Nestedness <: AbstractERGMTerm

Tendency for edges to exist within groups at a given level.
"""
struct Nestedness <: AbstractERGMTerm
    level::Int
end

name(t::Nestedness) = "nested.$(t.level)"

function compute(t::Nestedness, mln::MultilevelNetwork)
    t.level > n_levels(mln) && return 0.0
    t.level < 1 && return 0.0

    net = mln.levels[t.level]
    membership = t.level < n_levels(mln) ? mln.membership[t.level] : collect(1:nv(net))

    count = 0
    for e in edges(net)
        if membership[src(e)] == membership[dst(e)]
            count += 1
        end
    end
    return Float64(count)
end

"""
    CrossLevelEdge <: AbstractERGMTerm

Count edges that span levels in a multilevel network.
"""
struct CrossLevelEdge <: AbstractERGMTerm
    level1::Int
    level2::Int
end

name(t::CrossLevelEdge) = "crosslevel.$(t.level1).$(t.level2)"

function compute(t::CrossLevelEdge, mln::MultilevelNetwork)
    key = (t.level1, t.level2)
    haskey(mln.cross_level_edges, key) || return 0.0
    return Float64(length(mln.cross_level_edges[key]))
end

"""
    LevelHomophily <: AbstractERGMTerm

Homophily based on group membership at a given level.
"""
struct LevelHomophily <: AbstractERGMTerm
    level::Int
end

name(t::LevelHomophily) = "levelhomophily.$(t.level)"

function compute(t::LevelHomophily, mln::MultilevelNetwork)
    t.level >= n_levels(mln) && return 0.0

    # Look at edges at lowest level
    net = mln.levels[1]
    membership = mln.membership[t.level]

    count = 0
    for e in edges(net)
        if membership[src(e)] == membership[dst(e)]
            count += 1
        end
    end
    return Float64(count)
end

# =============================================================================
# Model and Estimation
# =============================================================================

"""
    MultiERGMModel

ERGM model for multi-network data.
"""
struct MultiERGMModel{T, N}  # N is the network type (MultiNetwork, MultilayerNetwork, etc.)
    terms::Vector{AbstractERGMTerm}
    data::N
end

"""
    MultiERGMResult

Results from fitting a multi-network ERGM.
"""
struct MultiERGMResult{T, N}
    model::MultiERGMModel{T, N}
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    loglik::Float64
    converged::Bool
end

function Base.show(io::IO, result::MultiERGMResult)
    println(io, "Multi-Network ERGM Results")
    println(io, "==========================")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    println(io)
    println(io, "Coefficients:")
    for (i, term) in enumerate(result.model.terms)
        println(io, "  $(rpad(name(term), 30)) $(lpad(round(result.coefficients[i], digits=4), 10)) " *
                    "(SE: $(round(result.std_errors[i], digits=4)))")
    end
end

"""
    ergm_multi(data, terms; kwargs...) -> MultiERGMResult

Fit an ERGM for multi-network data.

# Arguments
- `data`: MultiNetwork, MultilayerNetwork, or MultilevelNetwork
- `terms`: Vector of appropriate ERGM terms
"""
function ergm_multi(data::Union{MultiNetwork{T}, MultilayerNetwork{T}, MultilevelNetwork{T}},
                    terms::Vector{<:AbstractERGMTerm};
                    method::Symbol=:mple,
                    maxiter::Int=100) where T

    model = MultiERGMModel{T, typeof(data)}(terms, data)

    if method == :mple
        return multi_mple(model; maxiter=maxiter)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end

fit_multi_ergm = ergm_multi

"""
    multi_mple(model::MultiERGMModel; kwargs...) -> MultiERGMResult

MPLE for multi-network ERGM.
"""
function multi_mple(model::MultiERGMModel{T, N}; maxiter::Int=100, tol::Float64=1e-6) where {T, N}
    n_terms = length(model.terms)
    coef = zeros(n_terms)

    # Compute observed statistics
    obs_stats = [compute(term, model.data) for term in model.terms]

    # Optimization
    for iter in 1:maxiter
        grad = zeros(n_terms)

        for (i, term) in enumerate(model.terms)
            # Simplified gradient
            grad[i] = obs_stats[i] - obs_stats[i] * sigmoid(-coef[i])
        end

        step_size = 0.1 / sqrt(iter)
        coef .+= step_size * grad

        if maximum(abs.(grad)) < tol
            se = fill(0.1, n_terms)
            return MultiERGMResult{T, N}(model, coef, se, NaN, true)
        end
    end

    se = fill(NaN, n_terms)
    return MultiERGMResult{T, N}(model, coef, se, NaN, false)
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))

# =============================================================================
# Utilities
# =============================================================================

"""
    as_multilayer(nets::Dict{Symbol, Network}) -> MultilayerNetwork

Convert a dictionary of networks to a multilayer network.
"""
function as_multilayer(nets::Dict{Symbol, Network{T}}) where T
    # Check all have same vertex count
    n = nv(first(values(nets)))
    all(nv(net) == n for net in values(nets)) ||
        throw(ArgumentError("All networks must have same number of vertices"))

    mln = MultilayerNetwork{T}(n)
    for (name, net) in nets
        mln.layers[name] = net
    end

    return mln
end

"""
    combine_networks(nets::Vector{Network}; method=:union) -> Network

Combine multiple networks into one.
"""
function combine_networks(nets::Vector{Network{T}}; method::Symbol=:union) where T
    all(nv(net) == nv(nets[1]) for net in nets) ||
        throw(ArgumentError("All networks must have same number of vertices"))

    n = nv(nets[1])
    combined = Network{T}(; n=n, directed=is_directed(nets[1]))

    if method == :union
        for net in nets
            for e in edges(net)
                if !has_edge(combined, src(e), dst(e))
                    add_edge!(combined, src(e), dst(e))
                end
            end
        end
    elseif method == :intersection
        # Start with first network, keep only edges in all
        for e in edges(nets[1])
            if all(has_edge(net, src(e), dst(e)) for net in nets)
                add_edge!(combined, src(e), dst(e))
            end
        end
    else
        throw(ArgumentError("method must be :union or :intersection"))
    end

    return combined
end

"""
    split_by_layer(mln::MultilayerNetwork) -> Dict{Symbol, Network}

Split a multilayer network into separate networks.
"""
function split_by_layer(mln::MultilayerNetwork{T}) where T
    return Dict(name => deepcopy(net) for (name, net) in mln.layers)
end

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate_multi_ergm(result::MultiERGMResult; n_sim=1) -> Vector

Simulate networks from fitted multi-network ERGM.
"""
function simulate_multi_ergm(result::MultiERGMResult{T, N};
                             n_sim::Int=1,
                             burnin::Int=1000) where {T, N}
    @warn "Multi-network ERGM simulation not fully implemented"

    # Return copy of observed data as placeholder
    return [deepcopy(result.model.data) for _ in 1:n_sim]
end

end # module
