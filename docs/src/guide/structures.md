# Multi-Network Structures

ERGMMulti.jl provides three data structures for organizing multi-network data, each suited to different research designs. This guide covers their construction, properties, and usage patterns.

## Overview

| Structure | Use Case | Example |
|-----------|----------|---------|
| `MultiNetwork` | Multiple independent networks with shared parameters | Classrooms, organizations |
| `MultilayerNetwork` | Same nodes, multiple edge types | Friendship + advice + trust |
| `MultilevelNetwork` | Hierarchical nesting | People in teams in departments |

## MultiNetwork

A `MultiNetwork` holds multiple independent networks that share a common parameter vector. This is useful for comparative studies where the same model should apply across different groups.

### Creating a MultiNetwork

```julia
using Network
using ERGMMulti

# Create individual networks
net1 = Network{Int}(; n=30, directed=true)
add_edge!(net1, 1, 2)
add_edge!(net1, 2, 3)

net2 = Network{Int}(; n=30, directed=true)
add_edge!(net2, 1, 3)
add_edge!(net2, 3, 4)

net3 = Network{Int}(; n=30, directed=true)
add_edge!(net3, 2, 4)

# Create MultiNetwork with named networks
mn = MultiNetwork([net1, net2, net3]; names=[:class_A, :class_B, :class_C])
```

### Shared vs. Non-Shared Vertices

By default, all networks must have the same number of vertices:

```julia
# Shared vertices (default) - all networks must have same nv()
mn = MultiNetwork([net1, net2]; shared_vertices=true)

# Non-shared vertices - networks can differ in size
net_small = Network{Int}(; n=10, directed=true)
net_large = Network{Int}(; n=50, directed=true)
mn = MultiNetwork([net_small, net_large]; shared_vertices=false)
```

### Accessing Networks

```julia
# By index
mn[1]           # First network
mn[2]           # Second network

# By name
mn[:class_A]    # Named access
mn[:class_B]

# Properties
length(mn)      # Number of networks
nv(mn)          # Vertices (assumes shared)
total_edges(mn) # Total edges across all networks
```

### When to Use MultiNetwork

Use `MultiNetwork` when you have:

- Multiple separate networks from the same population (e.g., different schools)
- Repeated measurements of the same network (panel data)
- Experimental conditions with independent network observations
- Any setting where you want to pool networks for shared parameter estimation

## MultilayerNetwork

A `MultilayerNetwork` represents a single set of actors connected by multiple types of relationships (layers). Each layer is a separate network on the same vertex set.

### Creating a MultilayerNetwork

```julia
# Create with predefined layers
mln = MultilayerNetwork{Int}(50; layer_names=[:friendship, :advice, :trust])

# Or create empty and add layers
mln = MultilayerNetwork{Int}(50)
add_layer!(mln, :friendship)
add_layer!(mln, :advice; directed=true)
add_layer!(mln, :trust)
```

### Working with Layers

```julia
# Add edges to specific layers
add_layer_edge!(mln, :friendship, 1, 2)
add_layer_edge!(mln, :friendship, 2, 3)
add_layer_edge!(mln, :advice, 1, 3)
add_layer_edge!(mln, :trust, 1, 2)

# Check for edges
has_layer_edge(mln, :friendship, 1, 2)  # true
has_layer_edge(mln, :advice, 1, 2)      # false

# Query structure
n_layers(mln)        # 3
layer_names(mln)     # [:friendship, :advice, :trust]
nv(mln)              # 50
```

### Accessing Individual Layers

Each layer is a full `Network` object:

```julia
# Access the friendship network
friendship_net = mln.layers[:friendship]
ne(friendship_net)     # Number of friendship edges
is_directed(friendship_net)

# Access the advice network
advice_net = mln.layers[:advice]
ne(advice_net)
```

### Interlayer Edges

Some multilayer networks have edges that span layers (nodes connected across layers):

```julia
# Access interlayer edges (if set)
mln.interlayer_edges   # Dict{Tuple{Symbol,Symbol}, Set{Tuple{Int,Int}}}
```

### Example: Employee Relations

```julia
# Model three types of workplace ties
n_employees = 100
mln = MultilayerNetwork{Int}(n_employees;
    layer_names=[:friendship, :advice, :reporting])

# Friendship is undirected
add_layer!(mln, :friendship; directed=false)

# Advice and reporting are directed
# (already created as directed by default)

# Add ties
add_layer_edge!(mln, :friendship, 1, 2)
add_layer_edge!(mln, :advice, 1, 3)      # Person 1 seeks advice from 3
add_layer_edge!(mln, :reporting, 1, 5)    # Person 1 reports to 5
```

### When to Use MultilayerNetwork

Use `MultilayerNetwork` when you have:

- Multiple relationship types among the same actors (friendship, trust, collaboration)
- Multiplex network data (same nodes, different edge types)
- Communication networks with multiple channels (email, chat, phone)
- Any setting where cross-layer dependence is of theoretical interest

## MultilevelNetwork

A `MultilevelNetwork` represents hierarchical network structures where actors at one level are nested within groups at a higher level. Edges can exist both within and between levels.

### Creating a MultilevelNetwork

```julia
# Level 1: Individual ties (20 people)
individual_net = Network{Int}(; n=20, directed=true)
add_edge!(individual_net, 1, 2)
add_edge!(individual_net, 2, 3)
add_edge!(individual_net, 6, 7)
add_edge!(individual_net, 11, 12)

# Level 2: Team ties (4 teams)
team_net = Network{Int}(; n=4, directed=false)
add_edge!(team_net, 1, 2)
add_edge!(team_net, 2, 3)

# Membership: which team each individual belongs to
# People 1-5 in Team 1, 6-10 in Team 2, 11-15 in Team 3, 16-20 in Team 4
membership = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

# Create multilevel network
mln = MultilevelNetwork([individual_net, team_net], [membership])
```

### Querying Multilevel Structure

```julia
# Number of levels
n_levels(mln)  # 2

# Access networks at each level
mln.levels[1]   # Individual network
mln.levels[2]   # Team network

# Access membership
mln.membership[1]  # Individual -> team mapping
```

### Three-Level Example

```julia
# People in teams in departments
individual = Network{Int}(; n=60, directed=true)
team = Network{Int}(; n=12, directed=false)
department = Network{Int}(; n=3, directed=false)

# Individual -> team (5 people per team)
ind_to_team = repeat(1:12, inner=5)

# Team -> department (4 teams per department)
team_to_dept = repeat(1:3, inner=4)

mln = MultilevelNetwork(
    [individual, team, department],
    [ind_to_team, team_to_dept]
)

n_levels(mln)  # 3
```

### When to Use MultilevelNetwork

Use `MultilevelNetwork` when you have:

- Individuals nested in groups (students in classrooms)
- Multiple organizational levels (employees, teams, divisions)
- Hierarchical community structures
- Any setting where cross-level effects are of interest

## Converting Between Formats

### Dictionary to Multilayer

```julia
# From a dictionary of named networks
nets = Dict(
    :friendship => net1,
    :advice => net2,
)
mln = as_multilayer(nets)
```

### Combining Networks

```julia
# Union: include all edges from any network
combined = combine_networks([net1, net2, net3]; method=:union)

# Intersection: keep only edges present in all networks
combined = combine_networks([net1, net2, net3]; method=:intersection)
```

### Splitting Multilayer Networks

```julia
# Extract each layer as a separate Network
layers = split_by_layer(mln)
# Returns Dict{Symbol, Network}

layers[:friendship]  # The friendship layer as a standalone Network
layers[:advice]      # The advice layer as a standalone Network
```

## Design Considerations

### Choosing the Right Structure

| Question | Answer | Structure |
|----------|--------|-----------|
| Do networks share the same actors? | No | `MultiNetwork` |
| Do networks share the same actors? | Yes, different edge types | `MultilayerNetwork` |
| Is there a hierarchy? | Yes | `MultilevelNetwork` |
| Are networks independent? | Yes | `MultiNetwork` |
| Is cross-layer dependence relevant? | Yes | `MultilayerNetwork` |

### Memory Considerations

- `MultiNetwork`: Stores separate `Network` objects, memory scales with total edges
- `MultilayerNetwork`: Stores one `Network` per layer, efficient for many layers on the same nodes
- `MultilevelNetwork`: Stores networks at each level plus membership vectors

### Type Parameters

All structures are parameterized by vertex type `T`:

```julia
# Integer vertices (default)
mln = MultilayerNetwork{Int}(50)

# Use type inference
mln = MultilayerNetwork(50)  # Defaults to Int
```
