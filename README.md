# ERGMMulti.jl


[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ERGMMulti.jl icon" width="160">
</p>

ERGMs for Multiple and Multilayer Networks in Julia.

## Overview

ERGMMulti.jl provides tools for fitting ERGMs to:
- Multiple independent networks with shared parameters
- Multilayer/multiplex networks (same nodes, different edge types)
- Multilevel networks (networks within networks)

This package is a Julia port of the R `ergm.multi` package from the StatNet collection.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGMMulti.jl")
```

## Features

- **Data structures**: MultiNetwork, MultilayerNetwork, MultilevelNetwork
- **Layer terms**: Within-layer and between-layer effects
- **Multilevel terms**: Cross-level dependencies
- **Estimation**: MPLE for multi-network ERGMs

## Quick Start

```julia
using Network
using ERGMMulti

# Create multilayer network
mln = MultilayerNetwork{Int}(50; layer_names=[:friendship, :advice])

# Add edges to layers
add_layer_edge!(mln, :friendship, 1, 2)
add_layer_edge!(mln, :advice, 1, 3)

# Define terms
terms = [
    LayerEdges(:friendship),
    LayerEdges(:advice),
    InterlayerDependence(:friendship, :advice)
]

# Fit model
result = ergm_multi(mln, terms)
```

## MultiNetwork (Multiple Independent Networks)

For analyzing multiple networks with shared parameters:

```julia
# Create from vector of networks
networks = [net1, net2, net3, net4]
mn = MultiNetwork(networks; names=[:school1, :school2, :school3, :school4])

# Access
mn[1]          # First network
mn[:school1]   # By name
length(mn)     # Number of networks
total_edges(mn)

# Terms
CrossNetEdges()  # Total edges across all networks
```

## MultilayerNetwork (Multiplex)

For networks with multiple edge types on the same nodes:

```julia
# Create multilayer network
mln = MultilayerNetwork{Int}(n; layer_names=[:friendship, :advice, :trust])

# Add/check edges
add_layer!(mln, :new_layer)
add_layer_edge!(mln, :friendship, i, j)
has_layer_edge(mln, :friendship, i, j)

# Query
n_layers(mln)
layer_names(mln)
nv(mln)
```

## MultilevelNetwork

For hierarchical network structures:

```julia
# Networks at different levels
# Level 1: Individual ties
# Level 2: Group membership
# Level 3: Organization membership

levels = [individual_net, group_net, org_net]
membership = [
    individual_to_group,  # Which group each individual belongs to
    group_to_org          # Which org each group belongs to
]

mln = MultilevelNetwork(levels, membership)
n_levels(mln)
```

## Layer Terms

### Within-Layer
```julia
LayerEdges(:layer)      # Edge count in layer
LayerMutual(:layer)     # Mutuality in layer
LayerTriangle(:layer)   # Triangles in layer

# Apply any standard term to a layer
WithinLayer(Edges(), :friendship)
WithinLayer(GWESP(0.5), :advice)
```

### Between-Layer (Interlayer)
```julia
# Dependence: edge in layer1 given edge in layer2
InterlayerDependence(:friendship, :advice)

# Cross-layer mutuality: (i,j) in L1 and (j,i) in L2
MultiplexMutual(:friendship, :advice)

# Edges spanning layers (if applicable)
BetweenLayers(:layer1, :layer2)
```

## Multilevel Terms

```julia
# Nestedness: edges within groups
Nestedness(level)

# Cross-level edges
CrossLevelEdge(level1, level2)

# Homophily by group membership
LevelHomophily(level)
```

## Model Fitting

```julia
# Fit multi-network ERGM
result = ergm_multi(data, terms; method=:mple)

# View results
println(result)
```

## Utilities

```julia
# Convert dict of networks to multilayer
mln = as_multilayer(Dict(:L1 => net1, :L2 => net2))

# Combine networks
combined = combine_networks([net1, net2]; method=:union)
combined = combine_networks([net1, net2]; method=:intersection)

# Split multilayer into separate networks
layers = split_by_layer(mln)
```

## Example: Multiplex Social Network

```julia
# Friendship and advice ties among employees
mln = MultilayerNetwork{Int}(100; layer_names=[:friendship, :advice])
# ... populate layers ...

terms = [
    LayerEdges(:friendship),              # Friendship density
    LayerEdges(:advice),                  # Advice density
    LayerMutual(:friendship),             # Friendship reciprocity
    LayerMutual(:advice),                 # Advice reciprocity
    InterlayerDependence(:friendship, :advice),  # Friends give advice
    MultiplexMutual(:friendship, :advice) # Cross-layer reciprocity
]

result = ergm_multi(mln, terms)

# Positive InterlayerDependence → friendship predicts advice ties
```

## Example: Multilevel Organization

```julia
# Employees within teams within departments
terms = [
    Nestedness(1),        # Within-team ties
    Nestedness(2),        # Within-department ties
    CrossLevelEdge(1, 2), # Team-department ties
    LevelHomophily(1)     # Same-team homophily
]
```

## Documentation

For more detailed documentation, see:

- [Stable Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl/stable/)
- [Development Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl/dev/)

## References

1. Krivitsky, P.N., Koehly, L.M., Marcum, C.S. (2020). Exponential-family random graph models for multi-layer networks. *Psychometrika*, 85(3), 630-659.

2. Wang, P., Robins, G., Pattison, P., Lazega, E. (2013). Exponential random graph models for multilevel networks. *Social Networks*, 35(1), 96-115.

3. Hunter, D.R., Handcock, M.S., Butts, C.T., Goodreau, S.M., Morris, M. (2008). ergm: A package to fit, simulate and diagnose exponential-family models for networks. *Journal of Statistical Software*, 24(3), 1-29.

## License

MIT License - see [LICENSE](LICENSE) for details.
