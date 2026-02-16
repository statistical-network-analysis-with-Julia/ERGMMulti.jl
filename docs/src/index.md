# ERGMMulti.jl

*ERGMs for Multiple and Multilayer Networks in Julia*

A Julia package for fitting Exponential Random Graph Models to multiple, multilayer, and multilevel network data.

## Overview

ERGMMulti.jl extends the ERGM framework to handle complex network data involving multiple networks, multiple edge types (layers), and hierarchical structures (levels). It provides data structures, specialized ERGM terms, and estimation routines for these multi-network settings.

ERGMMulti.jl is a port of the R [ergm.multi](https://github.com/statnet/ergm.multi) package from the StatNet collection.

### What Are Multi-Network ERGMs?

Standard ERGMs model a single network. In many applications, however, data involves multiple networks that are structurally related:

```text
Multi-Network:   Net1  Net2  Net3       (shared parameters)
Multilayer:      L1 ──── L2 ──── L3    (same nodes, different ties)
Multilevel:      Org → Team → Person   (hierarchical nesting)
```

ERGMMulti.jl provides tools for all three settings.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **MultiNetwork** | A collection of independent networks analyzed with shared parameters |
| **MultilayerNetwork** | A single node set with multiple types of edges (layers) |
| **MultilevelNetwork** | Networks at different hierarchical levels with membership links |
| **Layer Term** | An ERGM term applied within or between specific layers |
| **Interlayer Dependence** | Statistical dependence between edges in different layers |

### Applications

Multi-network ERGMs are widely used in:

- **Organizational research**: Modeling friendship, advice, and trust ties among the same employees
- **International relations**: Analyzing trade, alliance, and conflict networks simultaneously
- **Multilevel analysis**: Studying individual ties nested within teams, departments, or organizations
- **Comparative studies**: Pooling multiple classroom, school, or community networks
- **Multiplex social networks**: Understanding how different relationship types interact

## Features

- **Rich data structures**: `MultiNetwork`, `MultilayerNetwork`, and `MultilevelNetwork` types
- **Within-layer terms**: Apply standard ERGM terms (edges, mutuality, triangles) to individual layers
- **Between-layer terms**: Model interlayer dependence and cross-layer reciprocity
- **Multilevel terms**: Nestedness, cross-level edges, and level homophily
- **Utility functions**: Convert, combine, and split network structures
- **MPLE estimation**: Maximum pseudo-likelihood for multi-network models

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/ERGMMulti.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/ERGMMulti.jl")
```

## Quick Start

```julia
using Network
using ERGMMulti

# Create a multilayer network with 50 nodes and two layers
mln = MultilayerNetwork{Int}(50; layer_names=[:friendship, :advice])

# Add edges to each layer
add_layer_edge!(mln, :friendship, 1, 2)
add_layer_edge!(mln, :friendship, 2, 3)
add_layer_edge!(mln, :advice, 1, 3)
add_layer_edge!(mln, :advice, 3, 2)

# Define multi-network ERGM terms
terms = [
    LayerEdges(:friendship),                        # Friendship density
    LayerEdges(:advice),                            # Advice density
    LayerMutual(:friendship),                       # Friendship reciprocity
    InterlayerDependence(:friendship, :advice),     # Friends give advice
]

# Fit the model
result = ergm_multi(mln, terms)
println(result)
```

## Choosing Terms

| Use Case | Recommended Terms |
|----------|-------------------|
| Layer-specific density | [`LayerEdges`](@ref) |
| Within-layer reciprocity | [`LayerMutual`](@ref) |
| Within-layer clustering | [`LayerTriangle`](@ref), [`WithinLayer`](@ref) |
| Cross-layer dependence | [`InterlayerDependence`](@ref) |
| Cross-layer reciprocity | [`MultiplexMutual`](@ref) |
| Multilevel nesting | [`Nestedness`](@ref), [`LevelHomophily`](@ref) |
| Cross-level ties | [`CrossLevelEdge`](@ref) |
| Shared parameters | [`CrossNetEdges`](@ref) |

## Documentation

```@contents
Pages = [
    "getting_started.md",
    "guide/structures.md",
    "guide/terms.md",
    "guide/estimation.md",
    "guide/multilevel.md",
    "api/types.md",
    "api/terms.md",
    "api/estimation.md",
]
Depth = 2
```

## Theoretical Background

### Multi-Network ERGMs

The probability of observing a set of networks $\mathbf{Y} = (Y_1, \ldots, Y_K)$ is modeled as:

$$P(\mathbf{Y} = \mathbf{y}) = \frac{1}{\kappa(\boldsymbol{\theta})} \exp\left(\sum_k \boldsymbol{\theta}^\top \mathbf{g}_k(\mathbf{y})\right)$$

Where:

- $\mathbf{g}_k(\mathbf{y})$ are sufficient statistics computed from the $k$-th network (or across networks)
- $\boldsymbol{\theta}$ are shared parameters
- $\kappa(\boldsymbol{\theta})$ is the normalizing constant

### Multilayer Dependence

For multilayer networks, cross-layer terms capture how edges in one layer influence edges in another. For example, the interlayer dependence statistic counts dyads with edges in both layers:

$$g_{\text{interlayer}}(\mathbf{y}) = \sum_{i \neq j} y_{ij}^{(1)} \cdot y_{ij}^{(2)}$$

A positive coefficient indicates that the presence of an edge in one layer increases the probability of an edge in the other.

## References

1. Krivitsky, P.N., Koehly, L.M., Marcum, C.S. (2020). Exponential-family random graph models for multi-layer networks. *Psychometrika*, 85(3), 630-659.

2. Wang, P., Robins, G., Pattison, P., Lazega, E. (2013). Exponential random graph models for multilevel networks. *Social Networks*, 35(1), 96-115.

3. Lazega, E., Snijders, T.A.B. (Eds.) (2015). *Multilevel Network Analysis for the Social Sciences*. Springer.

4. Koehly, L.M., Marcum, C.S. (2016). Multi-relational measurement for latent construct networks. *Psychological Methods*, 21(4), 452-469.
