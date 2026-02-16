# Getting Started

This tutorial walks through common use cases for ERGMMulti.jl, from basic multilayer modeling to advanced multilevel analysis.

## Installation

Install ERGMMulti.jl from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/ERGMMulti.jl")
```

## Basic Workflow

The typical ERGMMulti.jl workflow consists of four steps:

1. **Construct multi-network data** - Create the appropriate data structure
2. **Define terms** - Choose within-layer, between-layer, or multilevel terms
3. **Fit the model** - Estimate parameters via MPLE
4. **Interpret results** - Analyze the fitted model

## Step 1: Create a Multilayer Network

The most common use case is a multilayer (multiplex) network where the same set of actors have different types of ties:

```julia
using Network
using ERGMMulti

# Create a multilayer network with 30 actors and two layers
mln = MultilayerNetwork{Int}(30; layer_names=[:friendship, :advice])

# Add friendship edges
add_layer_edge!(mln, :friendship, 1, 2)
add_layer_edge!(mln, :friendship, 2, 3)
add_layer_edge!(mln, :friendship, 3, 1)
add_layer_edge!(mln, :friendship, 4, 5)
add_layer_edge!(mln, :friendship, 5, 6)

# Add advice edges (may overlap with friendship, or not)
add_layer_edge!(mln, :advice, 1, 3)
add_layer_edge!(mln, :advice, 2, 1)
add_layer_edge!(mln, :advice, 3, 4)
add_layer_edge!(mln, :advice, 5, 4)

# Query the structure
println("Layers: ", layer_names(mln))         # [:friendship, :advice]
println("Vertices: ", nv(mln))                 # 30
println("Layers: ", n_layers(mln))             # 2
```

### Checking Edges

```julia
# Check for edges in specific layers
has_layer_edge(mln, :friendship, 1, 2)  # true
has_layer_edge(mln, :advice, 1, 2)      # false
has_layer_edge(mln, :advice, 1, 3)      # true
```

### Adding Layers Dynamically

```julia
# Add a new layer after construction
add_layer!(mln, :trust)
add_layer_edge!(mln, :trust, 1, 2)
add_layer_edge!(mln, :trust, 2, 3)

println("Layers: ", n_layers(mln))  # 3
```

## Step 2: Create a MultiNetwork

When you have multiple independent networks to analyze together:

```julia
# Create individual networks
net1 = Network{Int}(; n=20, directed=true)
add_edge!(net1, 1, 2)
add_edge!(net1, 2, 3)
add_edge!(net1, 3, 1)

net2 = Network{Int}(; n=20, directed=true)
add_edge!(net2, 1, 3)
add_edge!(net2, 2, 4)

net3 = Network{Int}(; n=20, directed=true)
add_edge!(net3, 1, 2)
add_edge!(net3, 3, 4)
add_edge!(net3, 4, 5)

# Combine into a MultiNetwork
mn = MultiNetwork([net1, net2, net3]; names=[:school1, :school2, :school3])

# Access individual networks
mn[1]           # First network
mn[:school2]    # By name
length(mn)      # 3
total_edges(mn) # Total across all networks
```

## Step 3: Define Terms

Terms capture different aspects of multi-network structure:

```julia
# Within-layer terms for a multilayer network
terms = [
    LayerEdges(:friendship),              # Friendship density
    LayerEdges(:advice),                  # Advice density
    LayerMutual(:friendship),             # Friendship reciprocity
    LayerTriangle(:friendship),           # Friendship transitivity
    InterlayerDependence(:friendship, :advice),  # Cross-layer dependence
    MultiplexMutual(:friendship, :advice),       # Cross-layer reciprocity
]
```

### Exploring Available Terms

ERGMMulti.jl provides terms organized by scope:

| Category | Terms | Description |
|----------|-------|-------------|
| **Within-Layer** | `LayerEdges`, `LayerMutual`, `LayerTriangle` | Effects within a single layer |
| **Wrapper** | `WithinLayer` | Apply any standard ERGM term to a layer |
| **Between-Layer** | `InterlayerDependence`, `MultiplexMutual`, `BetweenLayers` | Cross-layer effects |
| **Multi-Network** | `CrossNetEdges` | Effects across independent networks |
| **Multilevel** | `Nestedness`, `CrossLevelEdge`, `LevelHomophily` | Hierarchical effects |

### Using the WithinLayer Wrapper

Apply any standard ERGM term to a specific layer:

```julia
using ERGM

# Apply standard ERGM terms within layers
terms = [
    WithinLayer(Edges(), :friendship),       # Same as LayerEdges(:friendship)
    WithinLayer(GWESP(0.5), :friendship),    # GWESP within friendship layer
    WithinLayer(Edges(), :advice),
]
```

## Step 4: Fit the Model

```julia
# Fit the multi-network ERGM
result = ergm_multi(mln, terms)
println(result)
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Estimation method | `:mple` |
| `maxiter` | Maximum iterations | `100` |

### Viewing Results

```julia
# Print formatted summary
println(result)

# Output:
# Multi-Network ERGM Results
# ==========================
# Log-likelihood: -12.3456
# Converged: true
#
# Coefficients:
#   edges.friendship               0.1234 (SE: 0.0567)
#   edges.advice                   0.0987 (SE: 0.0432)
#   mutual.friendship              0.5678 (SE: 0.1234)
#   ...
```

### Accessing Results Programmatically

```julia
# Coefficient vector
result.coefficients

# Standard errors
result.std_errors

# Log-likelihood
result.loglik

# Convergence status
result.converged
```

### Interpreting Coefficients

Coefficients have the same interpretation as standard ERGM coefficients - they are log-odds ratios:

| Coefficient | Interpretation |
|-------------|----------------|
| `edges.friendship = -2.0` | Low baseline friendship density |
| `mutual.friendship = 1.5` | Strong friendship reciprocity |
| `interlayer.friendship.advice = 0.8` | Friends are likely to also have advice ties |
| `multiplex.mutual.friendship.advice = 0.5` | Cross-layer reciprocity present |

## Complete Example: Multiplex Social Network

```julia
using Network
using ERGMMulti

# Create a multiplex network of 100 employees
n = 100
mln = MultilayerNetwork{Int}(n; layer_names=[:friendship, :advice])

# Populate layers with random ties
using Random
Random.seed!(42)

for _ in 1:200
    i, j = rand(1:n), rand(1:n)
    i != j && add_layer_edge!(mln, :friendship, i, j)
end

for _ in 1:150
    i, j = rand(1:n), rand(1:n)
    i != j && add_layer_edge!(mln, :advice, i, j)
end

# Define model
terms = [
    LayerEdges(:friendship),
    LayerEdges(:advice),
    LayerMutual(:friendship),
    LayerMutual(:advice),
    InterlayerDependence(:friendship, :advice),
    MultiplexMutual(:friendship, :advice),
]

# Fit
result = ergm_multi(mln, terms)
println(result)

# Positive InterlayerDependence -> friendship predicts advice ties
# Positive MultiplexMutual -> cross-layer reciprocity exists
```

## Complete Example: Multilevel Network

```julia
using Network
using ERGMMulti

# Level 1: Individual ties (20 people)
individual = Network{Int}(; n=20, directed=true)
add_edge!(individual, 1, 2)
add_edge!(individual, 2, 3)
add_edge!(individual, 4, 5)
add_edge!(individual, 5, 6)

# Level 2: Team ties (4 teams)
team = Network{Int}(; n=4, directed=false)
add_edge!(team, 1, 2)
add_edge!(team, 2, 3)

# Membership: which team each individual belongs to
individual_to_team = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

# Create multilevel network
mln = MultilevelNetwork([individual, team], [individual_to_team])

# Define multilevel terms
terms = [
    Nestedness(1),        # Within-team ties
    LevelHomophily(1),    # Same-team homophily
    CrossLevelEdge(1, 2), # Cross-level edges
]

# Fit
result = ergm_multi(mln, terms)
println(result)
```

## Utility Functions

### Converting Between Formats

```julia
# Convert dictionary of networks to multilayer
net1 = Network{Int}(; n=10, directed=true)
net2 = Network{Int}(; n=10, directed=true)
mln = as_multilayer(Dict(:L1 => net1, :L2 => net2))

# Split multilayer into separate networks
layers = split_by_layer(mln)  # Dict{Symbol, Network}

# Combine networks
combined = combine_networks([net1, net2]; method=:union)
combined = combine_networks([net1, net2]; method=:intersection)
```

## Best Practices

1. **Start with within-layer terms**: Model each layer independently before adding cross-layer terms
2. **Check convergence**: Verify `result.converged == true`
3. **Compare models**: Add cross-layer terms incrementally and compare log-likelihoods
4. **Watch for sparsity**: Sparse layers may cause estimation issues
5. **Use meaningful layers**: Each layer should represent a distinct relationship type
6. **Shared vertex set**: Ensure all layers in a multilayer network have the same actors

## Next Steps

- Learn about [Multi-Network Structures](guide/structures.md) in detail
- Explore all [Terms](guide/terms.md) available
- Understand [Model Estimation](guide/estimation.md) for multi-network models
- Study [Multilevel Networks](guide/multilevel.md) for hierarchical data
